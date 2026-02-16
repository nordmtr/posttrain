from __future__ import annotations

import pytest

from posttrain.types import StageName, parse_stages


def test_parse_stages_deduplicates_and_preserves_order() -> None:
    stages = parse_stages("sft,dpo,sft,grpo")
    assert stages == [StageName.SFT, StageName.DPO, StageName.GRPO]


def test_parse_stages_rejects_unknown_stage() -> None:
    with pytest.raises(ValueError, match="unknown stage"):
        parse_stages("sft,nope")
