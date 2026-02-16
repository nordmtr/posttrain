from __future__ import annotations

from dataclasses import dataclass

from posttrain.data import load_rlhf_data
from posttrain.modeling import (
    load_causal_lm,
    load_reference_causal_lm,
    load_reward_model,
    resolve_adapter_mode,
    resolve_device,
)
from posttrain.stages.base import StageContext
from posttrain.stages.common import (
    generate_completion,
    reward_score,
    response_logprob,
    write_dry_run_marker,
)
from posttrain.types import StageName


@dataclass(slots=True)
class RLHFStage:
    stage_name: StageName = StageName.RLHF

    def run(self, context: StageContext):
        artifact = context.artifacts.create_stage_artifact(self.stage_name)

        if context.config.runtime.dry_run:
            write_dry_run_marker(artifact.output_dir, self.stage_name.value)
            (artifact.output_dir / "reward_model").mkdir(parents=True, exist_ok=True)
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
        import torch.nn.functional as F

        rows = load_rlhf_data(context.config.data.rlhf_path)
        resolved_device = resolve_device(context.config.runtime.device)
        adapter_mode = resolve_adapter_mode(context.config.model, resolved_device)
        device = torch.device(adapter_mode.device)

        reward_loaded = load_reward_model(
            model_cfg=context.config.model,
            adapter_mode=adapter_mode,
            prior_adapter_path=None,
        )
        if adapter_mode.device == "cpu":
            reward_loaded.model.to(device)

        reward_optimizer = torch.optim.AdamW(
            reward_loaded.model.parameters(),
            lr=context.config.rlhf.reward_train.learning_rate,
        )

        reward_losses: list[float] = []
        reward_loaded.model.train()
        for step in range(context.config.rlhf.reward_train.max_steps):
            row = rows[step % len(rows)]
            chosen_text = f"{row['prompt']} {row['chosen']}"
            rejected_text = f"{row['prompt']} {row['rejected']}"

            chosen_score = reward_score(
                reward_loaded.model, reward_loaded.tokenizer, chosen_text, device
            )
            rejected_score = reward_score(
                reward_loaded.model, reward_loaded.tokenizer, rejected_text, device
            )

            loss = -F.logsigmoid(chosen_score - rejected_score)
            loss.backward()

            reward_optimizer.step()
            reward_optimizer.zero_grad(set_to_none=True)

            reward_losses.append(float(loss.detach().cpu()))

        reward_dir = artifact.output_dir / "reward_model"
        reward_dir.mkdir(parents=True, exist_ok=True)
        reward_loaded.model.save_pretrained(reward_dir)

        policy_loaded = load_causal_lm(
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

        if adapter_mode.device == "cpu":
            policy_loaded.model.to(device)
            reference.model.to(device)

        reward_loaded.model.eval()
        reference.model.eval()

        policy_optimizer = torch.optim.AdamW(
            policy_loaded.model.parameters(),
            lr=context.config.rlhf.reward_train.learning_rate,
        )

        policy_losses: list[float] = []
        rewards_seen: list[float] = []
        policy_loaded.model.train()
        for step in range(context.config.rlhf.ppo_steps):
            row = rows[step % len(rows)]
            prompt = row["prompt"]
            completion = generate_completion(
                policy_loaded.model, policy_loaded.tokenizer, prompt, device
            )

            with torch.no_grad():
                reward = reward_score(
                    reward_loaded.model,
                    reward_loaded.tokenizer,
                    f"{prompt} {completion}",
                    device,
                )
            rewards_seen.append(float(reward.detach().cpu()))

            logp = response_logprob(
                policy_loaded.model, policy_loaded.tokenizer, prompt, completion, device
            )
            with torch.no_grad():
                ref_logp = response_logprob(
                    reference.model, reference.tokenizer, prompt, completion, device
                )

            kl_term = logp - ref_logp
            advantage = reward - context.config.rlhf.kl_coeff * kl_term.detach()
            loss = -(advantage * logp)
            loss.backward()

            policy_optimizer.step()
            policy_optimizer.zero_grad(set_to_none=True)

            policy_losses.append(float(loss.detach().cpu()))

        policy_loaded.model.save_pretrained(artifact.adapter_dir)

        context.artifacts.write_metrics(
            artifact,
            {
                "stage": self.stage_name.value,
                "dry_run": False,
                "adapter_mode": adapter_mode.reason,
                "reward_steps": context.config.rlhf.reward_train.max_steps,
                "ppo_steps": context.config.rlhf.ppo_steps,
                "reward_loss_mean": sum(reward_losses) / max(len(reward_losses), 1),
                "policy_loss_mean": sum(policy_losses) / max(len(policy_losses), 1),
                "reward_signal_mean": sum(rewards_seen) / max(len(rewards_seen), 1),
            },
        )
        return artifact
