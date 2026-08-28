#!/usr/bin/env python3
"""Run a provider/model/prompt suite from an experiment config."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsr.utils.io import read_yaml


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _safe_name(text: str) -> str:
    return text.replace("/", "_").replace(":", "_").replace(".", "_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_multimodel_80.yaml")
    parser.add_argument("--limit-models", nargs="*", default=None)
    parser.add_argument("--limit-prompts", nargs="*", default=None)
    parser.add_argument("--skip-repair", action="store_true")
    args = parser.parse_args()

    config_path = _resolve(args.config)
    config = read_yaml(config_path)
    output_dir = _resolve(config.get("output_dir", "outputs/predictions/model_suite"))
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_path = config.get("graph_path", "data/processed/test_graphs.jsonl")
    models = config.get("models", [])
    prompts = config.get("prompts", [])
    if args.limit_models:
        models = [model for model in models if model["name"] in set(args.limit_models)]
    if args.limit_prompts:
        prompts = [prompt for prompt in prompts if prompt in set(args.limit_prompts)]

    for model in models:
        provider = model["provider"]
        model_name = model["name"]
        for prompt in prompts:
            run_name = f"{_safe_name(provider)}__{_safe_name(model_name)}__{_safe_name(prompt)}"
            generation = output_dir / f"{run_name}_generation.jsonl"
            verification = output_dir / f"{run_name}_verification.jsonl"
            repaired = output_dir / f"{run_name}_repaired.jsonl"
            metrics = output_dir / f"{run_name}_metrics.json"
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_llm_generation.py",
                    "--config",
                    str(config_path.relative_to(ROOT)),
                    "--provider",
                    provider,
                    "--model",
                    model_name,
                    "--prompt",
                    prompt,
                    "--output",
                    str(generation.relative_to(ROOT)),
                ],
                cwd=ROOT,
                check=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_verification.py",
                    "--input",
                    str(generation.relative_to(ROOT)),
                    "--graphs",
                    graph_path,
                    "--output",
                    str(verification.relative_to(ROOT)),
                ],
                cwd=ROOT,
                check=True,
            )
            if args.skip_repair:
                subprocess.run(
                    [
                        sys.executable,
                        "scripts/evaluate.py",
                        "--input",
                        str(verification.relative_to(ROOT)),
                        "--output",
                        str(metrics.relative_to(ROOT)),
                    ],
                    cwd=ROOT,
                    check=True,
                )
                continue
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_repair.py",
                    "--input",
                    str(verification.relative_to(ROOT)),
                    "--graphs",
                    graph_path,
                    "--output",
                    str(repaired.relative_to(ROOT)),
                ],
                cwd=ROOT,
                check=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    "scripts/evaluate.py",
                    "--input",
                    str(repaired.relative_to(ROOT)),
                    "--output",
                    str(metrics.relative_to(ROOT)),
                ],
                cwd=ROOT,
                check=True,
            )


if __name__ == "__main__":
    main()
