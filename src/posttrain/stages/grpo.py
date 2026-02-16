from __future__ import annotations

from dataclasses import dataclass

from posttrain.data import load_grpo_data
from posttrain.modeling import (
    load_causal_lm,
    load_reference_causal_lm,
    resolve_adapter_mode,
    resolve_device,
)
from posttrain.stages.base import StageContext
from posttrain.stages.common import (
    generate_completion,
    response_logprob,
    write_dry_run_marker,
)
from posttrain.types import StageName


@dataclass(slots=True)
class GRPOStage:
    stage_name: StageName = StageName.GRPO
    group_size: int = 2
    kl_coeff: float = 0.05

    def run(self, context: StageContext):
        artifact = context.artifacts.create_stage_artifact(self.stage_name)

        if context.config.runtime.dry_run:
            write_dry_run_marker(artifact.output_dir, self.stage_name.value)
            context.artifacts.write_metrics(
                artifact,
                {
                    "stage": self.stage_name.value,
                    "dry_run": True,
                    "adapter_mode": context.adapter_mode_reason,
                },
            )
            return artifact

        import torch

        rows = load_grpo_data(context.config.data.grpo_path)
        resolved_device = resolve_device(context.config.runtime.device)
        adapter_mode = resolve_adapter_mode(context.config.model, resolved_device)

        policy = load_causal_lm(
            model_cfg=context.config.model,
            adapter_mode=adapter_mode,
            stage_name=self.stage_name.value,
            prior_adapter_path=context.prior_adapter_path,
        )
        reference = load_reference_causal_lm(
            model_cfg=context.config.model,
            adapter_mode=adapter_mode,
            adapter_path=context.prior_adapter_path,
        )

        device = torch.device(adapter_mode.device)
        if adapter_mode.device == "cpu":
            policy.model.to(device)
            reference.model.to(device)

        reference.model.eval()
        optimizer = torch.optim.AdamW(policy.model.parameters(), lr=context.config.grpo.learning_rate)

        losses: list[float] = []
        rewards_seen: list[float] = []
        policy.model.train()
        for step in range(context.config.grpo.max_steps):
            row = rows[step % len(rows)]
            prompt = row["prompt"]
            target = row["target"].strip().lower()

            completions: list[str] = []
            rewards: list[float] = []
            for _ in range(self.group_size):
                completion = generate_completion(policy.model, policy.tokenizer, prompt, device)
                completions.append(completion)
                reward = 1.0 if target in completion.strip().lower() else 0.0
                rewards.append(reward)
                rewards_seen.append(reward)

            reward_mean = sum(rewards) / max(len(rewards), 1)

            total_loss = torch.tensor(0.0, device=device)
            for completion, reward in zip(completions, rewards, strict=True):
                advantage = reward - reward_mean
                logp = response_logprob(policy.model, policy.tokenizer, prompt, completion, device)
                with torch.no_grad():
                    ref_logp = response_logprob(reference.model, reference.tokenizer, prompt, completion, device)
                kl = logp - ref_logp
                total_loss = total_loss + (-(advantage * logp) + self.kl_coeff * kl)

            loss = total_loss / max(len(completions), 1)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            losses.append(float(loss.detach().cpu()))

        policy.model.save_pretrained(artifact.adapter_dir)
        context.artifacts.write_metrics(
            artifact,
            {
                "stage": self.stage_name.value,
                "dry_run": False,
                "adapter_mode": adapter_mode.reason,
                "steps": context.config.grpo.max_steps,
                "group_size": self.group_size,
                "kl_coeff": self.kl_coeff,
                "loss_mean": sum(losses) / max(len(losses), 1),
                "loss_last": losses[-1],
                "reward_mean": sum(rewards_seen) / max(len(rewards_seen), 1),
            },
        )
        return artifact
