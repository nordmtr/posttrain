from __future__ import annotations

from pathlib import Path

from posttrain.config import apply_config_overrides
from posttrain.presets import load_preset


def test_apply_config_overrides_updates_runtime_and_model(tmp_path: Path) -> None:
    cfg = load_preset("smoke", repo_root=tmp_path)

    updated = apply_config_overrides(
        cfg,
        {
            "runtime": {"device": "auto", "dry_run": True},
            "model": {"model_id": "new/model"},
            "sft": {"max_steps": 11},
        },
    )

    assert updated.runtime.device == "auto"
    assert updated.runtime.dry_run is True
    assert updated.model.model_id == "new/model"
    assert updated.sft.max_steps == 11


def test_load_gpu_preset_defaults(tmp_path: Path) -> None:
    cfg = load_preset("gpu", repo_root=tmp_path)
    assert cfg.runtime.device == "cuda"
    assert cfg.model.model_id == "Qwen/Qwen2.5-1.5B-Instruct"
