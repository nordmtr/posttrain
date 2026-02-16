from __future__ import annotations

from pathlib import Path

from posttrain.config import (
    DataConfig,
    ModelConfig,
    RLHFConfig,
    RuntimeConfig,
    SandboxConfig,
    StageTrainConfig,
)

PRESET_SMOKE = "smoke"
PRESET_GPU = "gpu"


def available_presets() -> tuple[str, ...]:
    return (PRESET_SMOKE, PRESET_GPU)


def load_preset(name: str, repo_root: Path) -> SandboxConfig:
    normalized = name.lower().strip()
    if normalized == PRESET_SMOKE:
        return _smoke_preset(repo_root)
    if normalized == PRESET_GPU:
        return _gpu_preset(repo_root)

    allowed = ", ".join(available_presets())
    raise ValueError(f"unknown preset '{name}', expected one of: {allowed}")


def _smoke_preset(repo_root: Path) -> SandboxConfig:
    data_root = repo_root / "data" / "smoke"
    return SandboxConfig(
        runtime=RuntimeConfig(
            output_dir=repo_root / "runs",
            device="cpu",
            seed=7,
            dry_run=False,
            log_steps=1,
        ),
        model=ModelConfig(
            model_id="sshleifer/tiny-gpt2",
            tokenizer_id="sshleifer/tiny-gpt2",
            use_4bit=True,
            lora_r=4,
            lora_alpha=8,
            lora_dropout=0.05,
            target_modules=("c_attn",),
        ),
        data=DataConfig(
            sft_path=data_root / "sft.jsonl",
            dpo_path=data_root / "dpo.jsonl",
            grpo_path=data_root / "grpo.jsonl",
            rlhf_path=data_root / "rlhf.jsonl",
        ),
        sft=StageTrainConfig(max_steps=2, num_train_epochs=1.0),
        dpo=StageTrainConfig(max_steps=2, num_train_epochs=1.0),
        grpo=StageTrainConfig(max_steps=2, num_train_epochs=1.0),
        rlhf=RLHFConfig(
            reward_train=StageTrainConfig(max_steps=2, num_train_epochs=1.0),
            ppo_steps=2,
        ),
    )


def _gpu_preset(repo_root: Path) -> SandboxConfig:
    data_root = repo_root / "data" / "smoke"
    return SandboxConfig(
        runtime=RuntimeConfig(
            output_dir=repo_root / "runs",
            device="cuda",
            seed=7,
            dry_run=False,
            log_steps=10,
        ),
        model=ModelConfig(
            model_id="Qwen/Qwen2.5-1.5B-Instruct",
            tokenizer_id="Qwen/Qwen2.5-1.5B-Instruct",
            use_4bit=True,
            lora_r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=(
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "up_proj",
                "down_proj",
                "gate_proj",
            ),
        ),
        data=DataConfig(
            sft_path=data_root / "sft.jsonl",
            dpo_path=data_root / "dpo.jsonl",
            grpo_path=data_root / "grpo.jsonl",
            rlhf_path=data_root / "rlhf.jsonl",
        ),
        sft=StageTrainConfig(max_steps=100, num_train_epochs=1.0),
        dpo=StageTrainConfig(max_steps=100, num_train_epochs=1.0),
        grpo=StageTrainConfig(max_steps=100, num_train_epochs=1.0),
        rlhf=RLHFConfig(
            reward_train=StageTrainConfig(max_steps=100, num_train_epochs=1.0),
            ppo_steps=100,
        ),
    )
