from __future__ import annotations

from enum import StrEnum


class StageName(StrEnum):
    SFT = "sft"
    DPO = "dpo"
    GRPO = "grpo"
    RLHF = "rlhf"


ORDERED_STAGES: tuple[StageName, ...] = (
    StageName.SFT,
    StageName.DPO,
    StageName.GRPO,
    StageName.RLHF,
)


def parse_stages(value: str) -> list[StageName]:
    if not value.strip():
        raise ValueError("stages value cannot be empty")

    parsed: list[StageName] = []
    seen: set[StageName] = set()
    for raw in value.split(","):
        normalized = raw.strip().lower()
        if not normalized:
            continue
        try:
            stage = StageName(normalized)
        except ValueError as exc:
            allowed = ", ".join(item.value for item in ORDERED_STAGES)
            raise ValueError(
                f"unknown stage '{normalized}', expected one of: {allowed}"
            ) from exc

        if stage not in seen:
            parsed.append(stage)
            seen.add(stage)

    if not parsed:
        raise ValueError("no valid stages were provided")

    return parsed
