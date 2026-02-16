from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from posttrain.types import StageName


@dataclass(slots=True)
class RuntimeConfig:
    output_dir: Path = Path("runs")
    device: str = "auto"
    seed: int = 7
    dry_run: bool = False
    log_steps: int = 1


@dataclass(slots=True)
class ModelConfig:
    model_id: str
    tokenizer_id: str | None = None
    use_4bit: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = ("c_attn", "c_proj")


@dataclass(slots=True)
class DataConfig:
    sft_path: Path
    dpo_path: Path
    grpo_path: Path
    rlhf_path: Path


@dataclass(slots=True)
class StageTrainConfig:
    learning_rate: float = 1e-4
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    max_steps: int = 4
    num_train_epochs: float = 1.0


@dataclass(slots=True)
class RLHFConfig:
    reward_train: StageTrainConfig = field(default_factory=StageTrainConfig)
    ppo_steps: int = 4
    ppo_batch_size: int = 1
    kl_coeff: float = 0.1


@dataclass(slots=True)
class SandboxConfig:
    runtime: RuntimeConfig
    model: ModelConfig
    data: DataConfig
    sft: StageTrainConfig = field(default_factory=StageTrainConfig)
    dpo: StageTrainConfig = field(default_factory=StageTrainConfig)
    grpo: StageTrainConfig = field(default_factory=StageTrainConfig)
    rlhf: RLHFConfig = field(default_factory=RLHFConfig)

    def with_output_dir(self, output_dir: Path) -> SandboxConfig:
        cfg = SandboxConfig(
            runtime=RuntimeConfig(
                output_dir=output_dir,
                device=self.runtime.device,
                seed=self.runtime.seed,
                dry_run=self.runtime.dry_run,
                log_steps=self.runtime.log_steps,
            ),
            model=self.model,
            data=self.data,
            sft=self.sft,
            dpo=self.dpo,
            grpo=self.grpo,
            rlhf=self.rlhf,
        )
        return cfg


@dataclass(slots=True)
class RunSummary:
    run_dir: Path
    stages: list[StageName]
    metrics_files: dict[StageName, Path]


def load_config_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("config file must decode to a mapping")
    return raw


def _resolve_stage_train(
    raw: dict[str, Any] | None, defaults: StageTrainConfig
) -> StageTrainConfig:
    if raw is None:
        return defaults
    return StageTrainConfig(
        learning_rate=float(raw.get("learning_rate", defaults.learning_rate)),
        per_device_train_batch_size=int(
            raw.get("per_device_train_batch_size", defaults.per_device_train_batch_size)
        ),
        gradient_accumulation_steps=int(
            raw.get("gradient_accumulation_steps", defaults.gradient_accumulation_steps)
        ),
        max_steps=int(raw.get("max_steps", defaults.max_steps)),
        num_train_epochs=float(raw.get("num_train_epochs", defaults.num_train_epochs)),
    )


def apply_config_overrides(
    base: SandboxConfig, overrides: dict[str, Any]
) -> SandboxConfig:
    runtime_raw = overrides.get("runtime", {})
    model_raw = overrides.get("model", {})
    data_raw = overrides.get("data", {})

    runtime = RuntimeConfig(
        output_dir=Path(runtime_raw.get("output_dir", base.runtime.output_dir)),
        device=str(runtime_raw.get("device", base.runtime.device)),
        seed=int(runtime_raw.get("seed", base.runtime.seed)),
        dry_run=bool(runtime_raw.get("dry_run", base.runtime.dry_run)),
        log_steps=int(runtime_raw.get("log_steps", base.runtime.log_steps)),
    )

    model = ModelConfig(
        model_id=str(model_raw.get("model_id", base.model.model_id)),
        tokenizer_id=model_raw.get("tokenizer_id", base.model.tokenizer_id),
        use_4bit=bool(model_raw.get("use_4bit", base.model.use_4bit)),
        lora_r=int(model_raw.get("lora_r", base.model.lora_r)),
        lora_alpha=int(model_raw.get("lora_alpha", base.model.lora_alpha)),
        lora_dropout=float(model_raw.get("lora_dropout", base.model.lora_dropout)),
        target_modules=tuple(
            model_raw.get("target_modules", base.model.target_modules)
        ),
    )

    data = DataConfig(
        sft_path=Path(data_raw.get("sft_path", base.data.sft_path)),
        dpo_path=Path(data_raw.get("dpo_path", base.data.dpo_path)),
        grpo_path=Path(data_raw.get("grpo_path", base.data.grpo_path)),
        rlhf_path=Path(data_raw.get("rlhf_path", base.data.rlhf_path)),
    )

    sft = _resolve_stage_train(overrides.get("sft"), base.sft)
    dpo = _resolve_stage_train(overrides.get("dpo"), base.dpo)
    grpo = _resolve_stage_train(overrides.get("grpo"), base.grpo)

    rlhf_raw = overrides.get("rlhf", {})
    rlhf = RLHFConfig(
        reward_train=_resolve_stage_train(
            rlhf_raw.get("reward_train"), base.rlhf.reward_train
        ),
        ppo_steps=int(rlhf_raw.get("ppo_steps", base.rlhf.ppo_steps)),
        ppo_batch_size=int(rlhf_raw.get("ppo_batch_size", base.rlhf.ppo_batch_size)),
        kl_coeff=float(rlhf_raw.get("kl_coeff", base.rlhf.kl_coeff)),
    )

    return SandboxConfig(
        runtime=runtime, model=model, data=data, sft=sft, dpo=dpo, grpo=grpo, rlhf=rlhf
    )
