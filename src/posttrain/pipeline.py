from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from posttrain.artifacts import ArtifactRegistry
from posttrain.config import RunSummary, SandboxConfig
from posttrain.modeling import resolve_adapter_mode, resolve_device
from posttrain.stages import (
    DPOStage,
    GRPOStage,
    RLHFStage,
    SFTStage,
    StageContext,
    StageRunner,
)
from posttrain.types import StageName


def default_runners() -> dict[StageName, StageRunner]:
    return {
        StageName.SFT: SFTStage(),
        StageName.DPO: DPOStage(),
        StageName.GRPO: GRPOStage(),
        StageName.RLHF: RLHFStage(),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value


class PipelineOrchestrator:
    def __init__(
        self, cfg: SandboxConfig, runners: dict[StageName, StageRunner] | None = None
    ) -> None:
        self.cfg = cfg
        self.runners = runners or default_runners()

    def run(self, stages: list[StageName], run_name: str | None = None) -> RunSummary:
        artifacts = ArtifactRegistry(self.cfg.runtime.output_dir, run_name=run_name)
        resolved_device = resolve_device(self.cfg.runtime.device)
        adapter_mode = resolve_adapter_mode(self.cfg.model, resolved_device)

        artifacts.write_resolved_config(
            {
                **_jsonable(asdict(self.cfg)),
                "resolved_device": resolved_device,
                "adapter_mode": adapter_mode.reason,
                "stages": [stage.value for stage in stages],
            }
        )

        prior_adapter: Path | None = None
        metrics: dict[StageName, Path] = {}
        for stage in stages:
            runner = self.runners.get(stage)
            if runner is None:
                raise ValueError(f"missing stage runner for {stage.value}")

            context = StageContext(
                config=self.cfg,
                artifacts=artifacts,
                stage=stage,
                prior_adapter_path=prior_adapter,
                adapter_mode_reason=adapter_mode.reason,
            )
            artifact = runner.run(context)
            prior_adapter = artifact.adapter_dir
            metrics[stage] = artifact.metrics_path

        return RunSummary(
            run_dir=artifacts.run_dir, stages=stages, metrics_files=metrics
        )
