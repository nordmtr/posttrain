# Posttraining Sandbox

A small posttraining sandbox that runs separate stages for:

- `sft`
- `dpo`
- `grpo` (reasoning-oriented reward optimization)
- `rlhf` (reward model training + KL-regularized policy optimization)

The project uses adapter training (`LoRA` / `QLoRA`) and avoids full-model fine-tuning.

## Requirements

- Python 3.13+
- `uv`

## Setup

```bash
uv sync
```

For GPU QLoRA runs, install the GPU dependency group:

```bash
uv sync --group gpu
```

## Run

### CPU smoke run (tiny model)

```bash
uv run python -m posttrain run --preset smoke --stages sft,dpo,grpo,rlhf
```

### CPU dry run (pipeline wiring only)

```bash
uv run python -m posttrain run --preset smoke --stages sft,dpo,grpo,rlhf --dry-run
```

### GPU run (1.5B model)

```bash
uv run python -m posttrain run --preset gpu --stages sft,dpo,grpo,rlhf --device cuda
```

## Key behaviors

- `smoke` preset uses `sshleifer/tiny-gpt2` and local fixture datasets in `data/smoke/`.
- `gpu` preset uses `Qwen/Qwen2.5-1.5B-Instruct`.
- If `use_4bit=true` and device is CPU, the pipeline logs a fallback to LoRA (non-quantized).
- If `use_4bit=true` and GPU is requested without the `gpu` group (`bitsandbytes`), execution fails with a clear error.
- On Linux x86_64, upstream `torch` wheels include CUDA runtime transitive dependencies even for CPU execution.

## Stage data schemas

- SFT (`data/smoke/sft.jsonl`): `prompt`, `response`
- DPO (`data/smoke/dpo.jsonl`): `prompt`, `chosen`, `rejected`
- GRPO (`data/smoke/grpo.jsonl`): `prompt`, `target`
- RLHF (`data/smoke/rlhf.jsonl`): `prompt`, `chosen`, `rejected`

## Outputs

Each run creates a directory under `runs/<timestamp>/`:

- per-stage adapter outputs
- per-stage `metrics.json`
- resolved run config (`config.resolved.json`)
