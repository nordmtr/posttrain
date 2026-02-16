from __future__ import annotations

import argparse
from pathlib import Path

from posttrain.config import SandboxConfig, apply_config_overrides, load_config_file
from posttrain.pipeline import PipelineOrchestrator
from posttrain.presets import available_presets, load_preset
from posttrain.types import parse_stages


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Posttraining sandbox pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run one or more posttraining stages")
    run.add_argument("--preset", choices=available_presets(), default="smoke")
    run.add_argument("--stages", default="sft,dpo,grpo,rlhf")
    run.add_argument("--config", type=Path, default=None)
    run.add_argument("--model-id", default=None)
    run.add_argument("--output-dir", type=Path, default=None)
    run.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    run.add_argument("--seed", type=int, default=None)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--run-name", default=None)

    return parser


def _apply_cli_overrides(
    base: SandboxConfig, args: argparse.Namespace
) -> SandboxConfig:
    model_overrides: dict[str, object] = {}
    runtime_overrides: dict[str, object] = {}

    if args.model_id:
        model_overrides["model_id"] = args.model_id
        if base.model.tokenizer_id is not None:
            model_overrides["tokenizer_id"] = args.model_id

    if args.output_dir is not None:
        runtime_overrides["output_dir"] = args.output_dir
    if args.device is not None:
        runtime_overrides["device"] = args.device
    if args.seed is not None:
        runtime_overrides["seed"] = args.seed
    if args.dry_run:
        runtime_overrides["dry_run"] = True

    merged: dict[str, object] = {}
    if runtime_overrides:
        merged["runtime"] = runtime_overrides
    if model_overrides:
        merged["model"] = model_overrides

    if not merged:
        return base
    return apply_config_overrides(base, merged)


def _build_config(args: argparse.Namespace) -> SandboxConfig:
    repo_root = Path.cwd()
    config = load_preset(args.preset, repo_root=repo_root)

    if args.config is not None:
        config = apply_config_overrides(config, load_config_file(args.config))

    config = _apply_cli_overrides(config, args)
    return config


def run_command(args: argparse.Namespace) -> int:
    stages = parse_stages(args.stages)
    config = _build_config(args)

    orchestrator = PipelineOrchestrator(config)
    summary = orchestrator.run(stages=stages, run_name=args.run_name)

    print(f"Run complete: {summary.run_dir}")
    for stage in summary.stages:
        print(f"- {stage.value}: {summary.metrics_files[stage]}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        return run_command(args)

    parser.error(f"unknown command: {args.command}")
    return 2
