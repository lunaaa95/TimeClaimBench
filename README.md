# TimeClaimBench Review Artifact

This is an anonymous review artifact for the paper "TimeClaimBench: A Claim-Level Benchmark for Grounded LLM Explanations of Time Series".

The artifact focuses on reproducing the reported evaluation from saved test splits, extracted event graphs, model generations, verifier outputs, repair outputs, metric files, and paper tables. It does not include author identity files, paper source files, API caches, raw third-party downloads, or private human-audit annotation sheets.

## Setup

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

## Quick Checks

```bash
pytest -q
```

If `pytest` is unavailable:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

## Main Reproducibility Commands

Recompute verification, verifier-guided repair, metrics, and the summary CSV from saved full synthetic generations:

```bash
PYTHONPATH=src python3 scripts/recompute_metrics_from_generations.py \
  --output-dir outputs/predictions/full_synthetic_400 \
  --graphs data/processed/test_graphs.jsonl \
  --config configs/verifier.yaml
```

Regenerate the main result tables from metric files:

```bash
PYTHONPATH=src python3 scripts/make_tables.py
PYTHONPATH=src python3 scripts/analyze_pattern_and_errors.py
PYTHONPATH=src python3 scripts/analyze_coverage_informativeness.py
PYTHONPATH=src python3 scripts/make_event_source_ablation_table.py
PYTHONPATH=src python3 scripts/make_mismatched_event_table.py
PYTHONPATH=src python3 scripts/analyze_real_nab_results.py
```

Bootstrap and verifier-human calibration outputs reported in the paper are included under `outputs/analysis/` and `outputs/tables/`. Re-running the combined calibration script requires private audit CSV files, which are intentionally excluded from this anonymous review artifact.

Run no-key local generation smoke tests with the dummy provider:

```bash
PYTHONPATH=src python3 scripts/run_llm_generation.py --config configs/experiment.yaml --provider dummy --prompt event_grounded
```

Real provider calls are optional and read credentials only from environment variables such as `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `DEEPSEEK_API_KEY`, `GEMINI_API_KEY`, or `GOOGLE_API_KEY`. No API keys are stored in this repository.

## Included

- `src/tsr/`: claim parsing, event extraction/graphs, verification, repair, metrics, prompt templates, and provider wrappers.
- `scripts/`: evaluation, verification, repair, table generation, and diagnostic scripts used for reported results.
- `data/synthetic/*.jsonl`: released review synthetic splits, including train/dev/test and the 80-example pilot subset.
- `data/processed/*.jsonl`: extracted and diagnostic event graphs.
- `data/real/nab_windows.jsonl`: processed Real-NAB localization diagnostic windows.
- `outputs/predictions/`: saved generations, verifier outputs, repair outputs, and metric summaries for the reported experiments.
- `outputs/analysis/`, `outputs/tables/`, and `outputs/verification_baselines/`: aggregate analysis files and generated tables.

## Not Included

- Author names, affiliations, emails, paper LaTeX source, submitted PDF, Overleaf metadata, and local build logs.
- Raw API cache files under `outputs/cache`.
- Raw NAB downloads under `data/raw`.
- Private human-audit annotation files and annotation apps.
- Synthetic data construction utilities, which are not required to recompute the reported test-set metrics in this review artifact.
