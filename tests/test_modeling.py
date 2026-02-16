from __future__ import annotations

import importlib.util

import pytest

from posttrain.config import ModelConfig
from posttrain.modeling import resolve_adapter_mode


def _model_cfg(*, use_4bit: bool) -> ModelConfig:
    return ModelConfig(model_id="x/y", tokenizer_id="x/y", use_4bit=use_4bit)


def test_resolve_adapter_mode_falls_back_on_cpu() -> None:
    mode = resolve_adapter_mode(_model_cfg(use_4bit=True), resolved_device="cpu")
    assert mode.active_4bit is False
    assert "falling back" in mode.reason.lower()


def test_resolve_adapter_mode_honors_disabled_4bit() -> None:
    mode = resolve_adapter_mode(_model_cfg(use_4bit=False), resolved_device="cuda")
    assert mode.active_4bit is False
    assert mode.reason == "4-bit disabled"


def test_resolve_adapter_mode_requires_bitsandbytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda _: None)

    with pytest.raises(RuntimeError, match="bitsandbytes"):
        resolve_adapter_mode(_model_cfg(use_4bit=True), resolved_device="cuda")
