from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from posttrain.types import StageName


@dataclass(slots=True)
class StageArtifact:
    stage: StageName
    output_dir: Path
    adapter_dir: Path
    metrics_path: Path


class ArtifactRegistry:
    def __init__(self, output_root: Path, run_name: str | None = None) -> None:
        self.output_root = output_root
        self.run_name = run_name or datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        self.run_dir = self.output_root / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def stage_dir(self, stage: StageName) -> Path:
        path = self.run_dir / stage.value
        path.mkdir(parents=True, exist_ok=True)
        return path

    def create_stage_artifact(self, stage: StageName) -> StageArtifact:
        stage_dir = self.stage_dir(stage)
        adapter_dir = stage_dir / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = stage_dir / "metrics.json"
        return StageArtifact(
            stage=stage,
            output_dir=stage_dir,
            adapter_dir=adapter_dir,
            metrics_path=metrics_path,
        )

    def write_metrics(self, artifact: StageArtifact, metrics: dict[str, Any]) -> None:
        artifact.metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    def write_resolved_config(self, serialized: dict[str, Any]) -> Path:
        path = self.run_dir / "config.resolved.json"
        path.write_text(json.dumps(serialized, indent=2, sort_keys=True), encoding="utf-8")
        return path
