from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class DataValidationError(ValueError):
    pass


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise DataValidationError(f"missing dataset file: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                decoded = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise DataValidationError(
                    f"invalid JSON on line {idx} in {path}"
                ) from exc
            if not isinstance(decoded, dict):
                raise DataValidationError(f"expected object on line {idx} in {path}")
            rows.append(decoded)

    if not rows:
        raise DataValidationError(f"dataset has no rows: {path}")

    return rows


def _require_keys(
    rows: list[dict[str, Any]], required: tuple[str, ...], dataset_name: str
) -> None:
    missing: list[str] = []
    for idx, row in enumerate(rows, start=1):
        for key in required:
            if key not in row or row[key] in (None, ""):
                missing.append(f"line {idx}: {key}")
    if missing:
        joined = "; ".join(missing[:5])
        raise DataValidationError(f"{dataset_name} schema mismatch: {joined}")


def load_sft_data(path: Path) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    _require_keys(rows, ("prompt", "response"), "sft")
    return rows


def load_dpo_data(path: Path) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    _require_keys(rows, ("prompt", "chosen", "rejected"), "dpo")
    return rows


def load_grpo_data(path: Path) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    _require_keys(rows, ("prompt", "target"), "grpo")
    return rows


def load_rlhf_data(path: Path) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    _require_keys(rows, ("prompt", "chosen", "rejected"), "rlhf")
    return rows
