from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from posttrain.config import apply_config_overrides
from posttrain.pipeline import PipelineOrchestrator
from posttrain.presets import load_preset
from posttrain.stages.base import StageContext
from posttrain.types import StageName


@dataclass
class DummyRunner:
    stage_name: StageName

    def run(self, context: StageContext):
        artifact = context.artifacts.create_stage_artifact(self.stage_name)
        (artifact.adapter_dir / "adapter.bin").write_text("ok", encoding="utf-8")
        context.artifacts.write_metrics(
            artifact, {"stage": self.stage_name.value, "dry_run": True}
        )
        return artifact


def test_pipeline_orchestrates_stage_sequence(tmp_path: Path) -> None:
    cfg = load_preset("smoke", repo_root=tmp_path)
    cfg = apply_config_overrides(
        cfg,
        {
            "runtime": {"dry_run": True, "output_dir": str(tmp_path / "runs")},
        },
    )

    orchestrator = PipelineOrchestrator(
        cfg,
        runners={
            StageName.SFT: DummyRunner(StageName.SFT),
            StageName.DPO: DummyRunner(StageName.DPO),
            StageName.GRPO: DummyRunner(StageName.GRPO),
            StageName.RLHF: DummyRunner(StageName.RLHF),
        },
    )

    summary = orchestrator.run([StageName.SFT, StageName.DPO], run_name="test-run")

    assert summary.run_dir == tmp_path / "runs" / "test-run"
    assert (summary.run_dir / "config.resolved.json").exists()
    assert summary.metrics_files[StageName.SFT].exists()
    assert summary.metrics_files[StageName.DPO].exists()
