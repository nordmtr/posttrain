from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from posttrain.config import ModelConfig


@dataclass(slots=True)
class AdapterMode:
    requested_4bit: bool
    active_4bit: bool
    reason: str
    device: str


@dataclass(slots=True)
class LoadedModel:
    model: Any
    tokenizer: Any
    adapter_mode: AdapterMode


def _missing_training_extra(exc: ImportError) -> RuntimeError:
    return RuntimeError(
        "Missing training dependencies. Install them with `uv sync --extra training` "
        "before running non-dry-run training stages."
    )


def resolve_device(device_preference: str) -> str:
    normalized = device_preference.lower().strip()
    if normalized == "auto":
        try:
            import torch
        except ImportError:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    if normalized not in {"cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    return normalized


def resolve_adapter_mode(model_cfg: ModelConfig, resolved_device: str) -> AdapterMode:
    if not model_cfg.use_4bit:
        return AdapterMode(
            requested_4bit=False,
            active_4bit=False,
            reason="4-bit disabled",
            device=resolved_device,
        )

    if resolved_device != "cuda":
        return AdapterMode(
            requested_4bit=True,
            active_4bit=False,
            reason="CPU path does not support QLoRA quantization; falling back to LoRA.",
            device=resolved_device,
        )

    if importlib.util.find_spec("bitsandbytes") is None:
        raise RuntimeError(
            "GPU run requested with 4-bit quantization, but bitsandbytes is not installed. "
            "Install it in the environment before running the GPU preset."
        )

    return AdapterMode(
        requested_4bit=True,
        active_4bit=True,
        reason="QLoRA active",
        device=resolved_device,
    )


def _target_task_type(stage_name: str) -> str:
    if stage_name == "rlhf_reward":
        return "SEQ_CLS"
    return "CAUSAL_LM"


def _build_lora_config(model_cfg: ModelConfig, stage_name: str) -> Any:
    try:
        from peft import LoraConfig
    except ImportError as exc:
        raise _missing_training_extra(exc) from exc

    return LoraConfig(
        r=model_cfg.lora_r,
        lora_alpha=model_cfg.lora_alpha,
        lora_dropout=model_cfg.lora_dropout,
        target_modules=list(model_cfg.target_modules),
        task_type=_target_task_type(stage_name),
    )


def load_tokenizer(model_cfg: ModelConfig) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise _missing_training_extra(exc) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.tokenizer_id or model_cfg.model_id
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_causal_lm(
    model_cfg: ModelConfig,
    adapter_mode: AdapterMode,
    stage_name: str,
    prior_adapter_path: Path | None,
) -> LoadedModel:
    try:
        from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    except ImportError as exc:
        raise _missing_training_extra(exc) from exc

    tokenizer = load_tokenizer(model_cfg)

    quantization_config = None
    model_kwargs: dict[str, Any] = {}
    if adapter_mode.active_4bit:
        import torch

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model_kwargs["device_map"] = "auto"
    elif adapter_mode.device == "cuda":
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.model_id,
        quantization_config=quantization_config,
        **model_kwargs,
    )

    if adapter_mode.active_4bit:
        model = prepare_model_for_kbit_training(model)

    if prior_adapter_path is not None and prior_adapter_path.exists():
        model = PeftModel.from_pretrained(
            model, str(prior_adapter_path), is_trainable=True
        )
    else:
        lora_config = _build_lora_config(model_cfg, stage_name)
        model = get_peft_model(model, lora_config)

    return LoadedModel(model=model, tokenizer=tokenizer, adapter_mode=adapter_mode)


def load_reference_causal_lm(
    model_cfg: ModelConfig,
    adapter_mode: AdapterMode,
    adapter_path: Path | None,
) -> LoadedModel:
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    except ImportError as exc:
        raise _missing_training_extra(exc) from exc

    tokenizer = load_tokenizer(model_cfg)
    quantization_config = None
    model_kwargs: dict[str, Any] = {}

    if adapter_mode.active_4bit:
        import torch

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model_kwargs["device_map"] = "auto"
    elif adapter_mode.device == "cuda":
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.model_id,
        quantization_config=quantization_config,
        **model_kwargs,
    )

    if adapter_path is not None and adapter_path.exists():
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)

    for parameter in model.parameters():
        parameter.requires_grad = False

    return LoadedModel(model=model, tokenizer=tokenizer, adapter_mode=adapter_mode)


def load_reward_model(
    model_cfg: ModelConfig,
    adapter_mode: AdapterMode,
    prior_adapter_path: Path | None,
) -> LoadedModel:
    try:
        from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig
    except ImportError as exc:
        raise _missing_training_extra(exc) from exc

    tokenizer = load_tokenizer(model_cfg)
    quantization_config = None
    model_kwargs: dict[str, Any] = {}

    if adapter_mode.active_4bit:
        import torch

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model_kwargs["device_map"] = "auto"
    elif adapter_mode.device == "cuda":
        model_kwargs["device_map"] = "auto"

    model = AutoModelForSequenceClassification.from_pretrained(
        model_cfg.model_id,
        num_labels=1,
        quantization_config=quantization_config,
        **model_kwargs,
    )

    if adapter_mode.active_4bit:
        model = prepare_model_for_kbit_training(model)

    if prior_adapter_path is not None and prior_adapter_path.exists():
        model = PeftModel.from_pretrained(
            model, str(prior_adapter_path), is_trainable=True
        )
    else:
        lora_config = _build_lora_config(model_cfg, "rlhf_reward")
        model = get_peft_model(model, lora_config)

    return LoadedModel(model=model, tokenizer=tokenizer, adapter_mode=adapter_mode)
