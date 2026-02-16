from __future__ import annotations

from dataclasses import dataclass

from posttrain.data import load_dpo_data
from posttrain.modeling import (
    load_causal_lm,
    load_reference_causal_lm,
    resolve_adapter_mode,
    resolve_device,
)
from posttrain.stages.base import StageContext
from posttrain.stages.common import response_logprob, write_dry_run_marker
from posttrain.types import StageName


@dataclass(slots=True)
class DPOStage:
    stage_name: StageName = StageName.DPO
    beta: float = 0.1

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
        import torch.nn.functional as F

        rows = load_dpo_data(context.config.data.dpo_path)
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
        optimizer = torch.optim.AdamW(
            policy.model.parameters(), lr=context.config.dpo.learning_rate
        )

        losses: list[float] = []
        policy.model.train()
        for step in range(context.config.dpo.max_steps):
            row = rows[step % len(rows)]
            prompt = row["prompt"]

            pi_chosen = response_logprob(
                policy.model, policy.tokenizer, prompt, row["chosen"], device
            )
            pi_rejected = response_logprob(
                policy.model, policy.tokenizer, prompt, row["rejected"], device
            )

            with torch.no_grad():
                ref_chosen = response_logprob(
                    reference.model, reference.tokenizer, prompt, row["chosen"], device
                )
                ref_rejected = response_logprob(
                    reference.model,
                    reference.tokenizer,
                    prompt,
                    row["rejected"],
                    device,
                )

            dpo_logit = self.beta * (
                (pi_chosen - pi_rejected) - (ref_chosen - ref_rejected)
            )
            loss = -F.logsigmoid(dpo_logit)
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
                "steps": context.config.dpo.max_steps,
                "beta": self.beta,
                "loss_mean": sum(losses) / max(len(losses), 1),
                "loss_last": losses[-1],
            },
        )
        return artifact
