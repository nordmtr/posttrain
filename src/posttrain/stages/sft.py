from __future__ import annotations

from dataclasses import dataclass

from posttrain.data import load_sft_data
from posttrain.modeling import load_causal_lm, resolve_adapter_mode, resolve_device
from posttrain.stages.base import StageContext
from posttrain.stages.common import move_batch_to_device, write_dry_run_marker
from posttrain.types import StageName


@dataclass(slots=True)
class SFTStage:
    stage_name: StageName = StageName.SFT

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

        rows = load_sft_data(context.config.data.sft_path)
        resolved_device = resolve_device(context.config.runtime.device)
        adapter_mode = resolve_adapter_mode(context.config.model, resolved_device)

        loaded = load_causal_lm(
            model_cfg=context.config.model,
            adapter_mode=adapter_mode,
            stage_name=self.stage_name.value,
            prior_adapter_path=context.prior_adapter_path,
        )

        device = torch.device(adapter_mode.device)
        if adapter_mode.device == "cpu":
            loaded.model.to(device)

        optimizer = torch.optim.AdamW(
            loaded.model.parameters(), lr=context.config.sft.learning_rate
        )

        losses: list[float] = []
        loaded.model.train()
        for step in range(context.config.sft.max_steps):
            row = rows[step % len(rows)]
            text = f"Instruction: {row['prompt']}\nResponse: {row['response']}"
            encoded = loaded.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            )
            encoded = move_batch_to_device(encoded, device)
            outputs = loaded.model(**encoded, labels=encoded["input_ids"])
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            losses.append(float(loss.detach().cpu()))

        loaded.model.save_pretrained(artifact.adapter_dir)
        context.artifacts.write_metrics(
            artifact,
            {
                "stage": self.stage_name.value,
                "dry_run": False,
                "adapter_mode": adapter_mode.reason,
                "steps": context.config.sft.max_steps,
                "loss_mean": sum(losses) / max(len(losses), 1),
                "loss_last": losses[-1],
            },
        )
        return artifact
