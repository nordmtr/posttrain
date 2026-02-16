# Knowledge

- The CPU smoke path intentionally allows `use_4bit=true` but falls back to non-quantized LoRA; this keeps the same stage flow without requiring CUDA.
- Stage datasets are strict JSONL schemas validated in `src/posttrain/data.py`; invalid/missing keys fail fast.
- `PipelineOrchestrator` supports dependency-injected stage runners, which keeps integration tests fast and avoids downloading models.
- Smoke-stage runtime dependencies (`torch`, `transformers`, `peft`) are base dependencies.
- GPU-only QLoRA dependency (`bitsandbytes`) lives in the `gpu` dependency group; install it with `uv sync --group gpu`.
- On Linux x86_64, `torch` currently pulls CUDA runtime packages transitively; this is upstream packaging behavior and independent of the `gpu` group.
