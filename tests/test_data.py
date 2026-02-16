from __future__ import annotations

from pathlib import Path

import pytest

from posttrain.data import DataValidationError, load_dpo_data, load_sft_data


def test_load_sft_data_success(tmp_path: Path) -> None:
    path = tmp_path / "sft.jsonl"
    path.write_text('{"prompt":"p","response":"r"}\n', encoding="utf-8")

    rows = load_sft_data(path)
    assert rows[0]["prompt"] == "p"


def test_load_dpo_data_requires_keys(tmp_path: Path) -> None:
    path = tmp_path / "dpo.jsonl"
    path.write_text('{"prompt":"p","chosen":"a"}\n', encoding="utf-8")

    with pytest.raises(DataValidationError, match="schema mismatch"):
        load_dpo_data(path)
