# Knowledge

- The CPU smoke path intentionally allows `use_4bit=true` but falls back to non-quantized LoRA; this keeps the same stage flow without requiring CUDA.
- Stage datasets are strict JSONL schemas validated in `src/posttrain/data.py`; invalid/missing keys fail fast.
- `PipelineOrchestrator` supports dependency-injected stage runners, which keeps integration tests fast and avoids downloading models.
- Heavy ML dependencies (`transformers`, `trl`, `peft`, etc.) are in the optional `training` extra; use `uv sync --extra training` before non-dry-run training.
