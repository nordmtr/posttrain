from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from posttrain.artifacts import ArtifactRegistry, StageArtifact
from posttrain.config import SandboxConfig
from posttrain.types import StageName


@dataclass(slots=True)
class StageContext:
    config: SandboxConfig
    artifacts: ArtifactRegistry
    stage: StageName
    prior_adapter_path: Path | None
    adapter_mode_reason: str


class StageRunner(Protocol):
    stage_name: StageName

    def run(self, context: StageContext) -> StageArtifact: ...
