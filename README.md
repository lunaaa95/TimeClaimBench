# TimeClaimBench: A Claim-Level Benchmark for Grounded LLM Explanations of Time Series

Sijia Peng<sup>&#42;</sup>, Fan Zhang<sup>&#42;</sup>, Yixi Zhou, Changlun Li, Yun Xiong, Yangyong Zhu, Xi Chen, Yanwei Yu, and Nan Tang

<sup>&#42;</sup> Equal contribution.

**Accepted to Findings of the Association for Computational Linguistics: EMNLP 2026.**

TimeClaimBench evaluates whether natural-language explanations of time series are grounded in temporal evidence. It converts series into event graphs, parses generated explanations into atomic claims, verifies each claim against the graph, and supports verifier-guided repair.

## Overview

LLMs can produce fluent time-series explanations that still contain unsupported temporal claims, incorrect intervals, or hallucinated patterns. TimeClaimBench makes this failure mode measurable at the claim level: every generated claim receives an operational `SUPPORTED`, `PARTIAL`, or `UNSUPPORTED` verdict under the released parser, event graph, and verifier rules.

![Motivation figure](figs/motivation.png)

The artifact contains the benchmark data, saved model generations, extracted event graphs, verification and repair outputs, evaluation scripts, aggregate human-study results, and the additional diagnostics reported in the camera-ready paper.

## Repository Contents

```text
.
|-- configs/                         # Experiment, model, and verifier configurations
|-- data/
|   |-- synthetic/                   # Train, development, test, and pilot splits
|   |-- processed/                   # Extracted and diagnostic event graphs
|   `-- real/nab_windows.jsonl       # Processed Real-NAB diagnostic windows
|-- figs/                            # Motivation and pipeline figures
|-- outputs/
|   |-- predictions/                 # Saved generations, verdicts, repairs, and metrics
|   |-- verification_baselines/      # Claim-verification baseline outputs
|   |-- analysis/                    # Aggregate analyses and camera-ready diagnostics
|   `-- tables/                      # Generated paper tables
|-- scripts/
|   `-- camera_ready/                # Final-paper diagnostic and validation scripts
|-- src/tsr/                         # Core TimeClaimBench implementation
|-- tests/                           # Unit tests
|-- ARTIFACT_MANIFEST.md             # Reproducibility and release-scope manifest
|-- Makefile
|-- pyproject.toml
`-- requirements.txt
```

## Pipeline

![Pipeline figure](figs/pipeline.png)

| Stage | Component | Purpose | Main output |
|---|---|---|---|
| 0 | Time-series input | Load synthetic series or processed real diagnostic windows | Time-series samples |
| 1 | Event extraction | Detect typed temporal events and construct interval-aware evidence | Event graphs |
| 2 | Explanation generation | Prompt LLMs under Direct, CoT, Structured, and Event-grounded conditions | Natural-language explanations |
| 3 | Claim verification | Parse explanations into atomic claims and match them to graph evidence | Claim-level support verdicts |
| 4 | Repair and evaluation | Repair unsupported claims and aggregate faithfulness, coverage, and diagnostic metrics | Repaired explanations and result tables |

## Key Results

- Event-grounded prompting gives the strongest **operational extracted-event support** on the synthetic benchmark: strict `SUPPORTED` is `0.676` vs. `0.493` for Direct, and operational `UNSUPPORTED` falls from `0.392` to `0.113`.
- Verifier-guided repair improves operational strict support from `0.704` to `0.991`, while reducing operational `UNSUPPORTED` from `0.138` to `0.006`.
- Event-graph verification is substantially stronger than type-only checking: Macro-F1 `0.733` vs. `0.456`.
- Real-NAB diagnostics show a high anomaly-window localization hit rate (`0.960`), while claim-level support remains harder on real-world data.

These automatic quantities measure consistency with the operational extracted event graph; they are not estimates of human truth. The camera-ready diagnostics further show that:

- The extractor recovers `394/400` designed primary events by exact type and interval IoU >= `0.5` (`0.985` recall). Precision is not identifiable from sparse, non-exhaustive event annotations.
- In a blinded two-annotator study on 40 series for GPT-4.1-mini, Event-grounded answers have the highest same-sample operational strict support (`0.711`, vs. `0.577` Direct and `0.355` CoT), but lower whole-explanation human ratings and preferences. This preference rank reversal shows that representation consistency and holistic human judgment are distinct evaluation axes.
- Re-verification against sparse designed-event graphs also reverses strict-support rankings. Because those graphs omit valid local patterns, this is a reference-sensitivity analysis rather than an alternative ground truth.

## Quick Start

Python `3.9+` is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
pytest -q
```

If `pytest` is unavailable, run the standard-library test suite:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

Run a no-key smoke test with the dummy provider:

```bash
PYTHONPATH=src python3 scripts/run_llm_generation.py \
  --config configs/experiment.yaml \
  --provider dummy \
  --prompt event_grounded
```

## Reproducing the Paper Results

Recompute verification, verifier-guided repair, metrics, and the summary CSV from the saved full-synthetic generations:

```bash
PYTHONPATH=src python3 scripts/recompute_metrics_from_generations.py \
  --output-dir outputs/predictions/full_synthetic_400 \
  --graphs data/processed/test_graphs.jsonl \
  --config configs/verifier.yaml
```

Regenerate the main tables and diagnostics:

```bash
PYTHONPATH=src python3 scripts/make_tables.py
PYTHONPATH=src python3 scripts/analyze_pattern_and_errors.py
PYTHONPATH=src python3 scripts/analyze_coverage_informativeness.py
PYTHONPATH=src python3 scripts/make_event_source_ablation_table.py
PYTHONPATH=src python3 scripts/make_mismatched_event_table.py
PYTHONPATH=src python3 scripts/analyze_real_nab_results.py
```

Regenerate the released camera-ready automatic diagnostics and validate the aggregate arithmetic:

```bash
make camera-ready-diagnostics
```

The sparse-gold reference analysis uses saved generations and makes no API calls. Human-study bootstrap intervals cannot be regenerated without the withheld raw ratings; the released aggregate counts can be checked with:

```bash
make validate-diagnostics
```

See [`outputs/analysis/camera_ready/README.md`](outputs/analysis/camera_ready/README.md) for the exact aggregate release boundary.

## Running New LLM Calls

All saved outputs and reproducibility checks can be used without API credentials. Live provider calls require users to supply their own keys through environment variables:

```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export DEEPSEEK_API_KEY="your-key"
export GEMINI_API_KEY="your-key"
```

The implementation reads credentials only from the environment. No API key is stored in this repository.

## Data Release and Privacy

The public artifact includes processed benchmark data, saved model outputs, automatic evaluation artifacts, and anonymous aggregate human-study statistics. It intentionally excludes:

- raw annotation sheets and row-level ratings;
- annotator names, email addresses, identifiers, and free-text notes;
- candidate packets, annotation applications, and randomization keys;
- raw API caches, credentials, paper source files, and raw third-party downloads.

Consequently, the human-study aggregate arithmetic is publicly checkable, but private-input bootstrap resampling cannot be reproduced from this repository. See [`ARTIFACT_MANIFEST.md`](ARTIFACT_MANIFEST.md) for the full release manifest.

## Citation

If you use TimeClaimBench, please cite the EMNLP 2026 Findings paper:

```bibtex
@inproceedings{peng-etal-2026-timeclaimbench,
  title = {{TimeClaimBench}: A Claim-Level Benchmark for Grounded {LLM} Explanations of Time Series},
  author = {Peng, Sijia and Zhang, Fan and Zhou, Yixi and Li, Changlun and
            Xiong, Yun and Zhu, Yangyong and Chen, Xi and Yu, Yanwei and Tang, Nan},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2026},
  year = {2026}
}
```
