from __future__ import annotations

from pathlib import Path
from typing import Any


def write_dry_run_marker(path: Path, stage_name: str) -> None:
    marker = path / "DRY_RUN.txt"
    marker.write_text(
        f"{stage_name} executed in dry-run mode. No model weights were updated.\n",
        encoding="utf-8",
    )


def move_batch_to_device(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if hasattr(value, "to"):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def response_logprob(
    model: Any,
    tokenizer: Any,
    prompt: str,
    response: str,
    device: Any,
    max_length: int = 512,
) -> Any:
    import torch

    prompt_ids = tokenizer(
        prompt, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)
    full_ids = tokenizer(
        prompt + response,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    ).input_ids.to(device)

    if full_ids.size(1) <= prompt_ids.size(1):
        return torch.tensor(0.0, device=device)

    outputs = model(input_ids=full_ids)
    logits = outputs.logits[:, :-1, :]
    targets = full_ids[:, 1:]
    token_logprobs = (
        torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    )

    response_start = max(prompt_ids.size(1) - 1, 0)
    return token_logprobs[:, response_start:].sum(dim=-1).mean()


def generate_completion(
    model: Any,
    tokenizer: Any,
    prompt: str,
    device: Any,
    max_new_tokens: int = 32,
    temperature: float = 1.0,
) -> str:
    import torch

    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = move_batch_to_device(encoded, device)

    with torch.no_grad():
        generated = model.generate(
            **encoded,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_len = encoded["input_ids"].shape[1]
    completion_ids = generated[0][prompt_len:]
    completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
    return completion.strip()


def reward_score(
    model: Any, tokenizer: Any, text: str, device: Any, max_length: int = 512
) -> Any:
    encoded = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=max_length
    )
    encoded = move_batch_to_device(encoded, device)
    outputs = model(**encoded)
    return outputs.logits.squeeze(-1).mean()
