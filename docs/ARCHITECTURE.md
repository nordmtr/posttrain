# Architecture Map

## Entrypoint

- `src/posttrain/cli.py`
: `main`, `run_command`, `_build_config` parse CLI flags, apply preset/config overrides, and start orchestration.
- `main.py`
: delegates to `posttrain.cli.main`.

## Orchestration

- `src/posttrain/pipeline.py`
: `PipelineOrchestrator.run` executes selected stages in order and threads prior stage adapters into the next stage.
- `src/posttrain/artifacts.py`
: `ArtifactRegistry` creates run/stage folders and persists stage metrics and resolved config.

## Configuration

- `src/posttrain/config.py`
: dataclasses for runtime/model/data/stage hyperparameters and `apply_config_overrides`.
- `src/posttrain/presets.py`
: built-in `smoke` and `gpu` presets with model IDs and default paths.
- `pyproject.toml`
: smoke-stage runtime stack (`torch`, `transformers`, `peft`) is in base dependencies; GPU-specific QLoRA dependency (`bitsandbytes`) is in the `gpu` dependency group.

## Model Loading and Adapter Strategy

- `src/posttrain/modeling.py`
: `resolve_device`, `resolve_adapter_mode`, `load_causal_lm`, `load_reference_causal_lm`, `load_reward_model`.
- Behavior:
: requests QLoRA for GPU (`use_4bit=true`) and falls back to LoRA on CPU.

## Stage Implementations

- `src/posttrain/stages/sft.py`
: `SFTStage.run` performs supervised adapter tuning on `prompt/response` examples.
- `src/posttrain/stages/dpo.py`
: `DPOStage.run` applies pairwise DPO-style optimization using policy vs reference log-prob differences.
- `src/posttrain/stages/grpo.py`
: `GRPOStage.run` performs grouped sampling with reward-baseline advantages and KL regularization.
- `src/posttrain/stages/rlhf.py`
: `RLHFStage.run` trains reward model from preferences, then performs reward- and KL-guided policy updates.
- `src/posttrain/stages/common.py`
: shared token/logprob/reward/generation helpers.

## Data Layer

- `src/posttrain/data.py`
: JSONL loading and strict schema validation for SFT/DPO/GRPO/RLHF datasets.
- `data/smoke/*.jsonl`
: deterministic tiny fixtures for smoke testing.

## Tests

- `tests/test_types.py`
: stage parser behavior.
- `tests/test_config.py`
: preset + override behavior.
- `tests/test_data.py`
: dataset validation behavior.
- `tests/test_pipeline.py`
: orchestration behavior with injected stage runners.
