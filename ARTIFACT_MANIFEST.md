# Artifact Manifest

## Purpose

This repository accompanies the EMNLP 2026 Findings version of TimeClaimBench. It supports inspection and reproduction of the benchmark's automatic evaluation and releases aggregate results for the human studies added during revision.

## Reproducibility coverage

- Main synthetic 400-example evaluation: data, saved generations, verification outputs, repair outputs, metrics, and scripts included.
- Multi-provider 80-example pilot: data and saved outputs included.
- Real-NAB localization diagnostic: processed windows, graphs, saved outputs, and analysis script included.
- Verifier baselines and event-source diagnostics: included.
- Designed-event recovery (`394/400`): script and CSV/JSON outputs included.
- Synthetic-real scale-invariant shape coverage: script and CSV/JSON outputs included.
- Sparse-gold reference sensitivity over the saved generations: script and aggregate outputs included; no model calls are needed.
- Independent 90-claim, two-annotator audit: aggregate JSON and confusion matrices included.
- Blinded 40-series, two-annotator preference study: aggregate ratings, confidence intervals, preferences, agreement, by-pattern results, and same-sample automatic verifier results included.
- Period-4 and failure diagnostics: aggregate text counts, stored verifier mismatch-code counts, and two deidentified traces included.

## Metric scope

The automatic `SUPPORTED`, `PARTIAL`, and `UNSUPPORTED` labels measure consistency with the operational extracted event graph under the released parser and verifier rules. They should not be read as human truth labels. The 40-series preference study exhibits a rank reversal: event-grounded outputs rank highest on operational strict support but below Direct and CoT under blinded whole-explanation ratings. The artifact preserves both results because they measure different properties.

## Data minimization

- Raw provider caches are excluded because they may contain request metadata.
- Raw third-party NAB downloads are excluded; processed diagnostic windows are included.
- Paper source files are maintained separately from the code and data artifact.
- Raw annotation sheets, candidate packets, free-text notes, randomization keys, and annotator identities are excluded.
- Human-study outputs are aggregate-only. Public scripts validate their recoverable arithmetic but cannot recreate private-input bootstrap samples.

## Artifact verification commands

```bash
pytest -q
make camera-ready-diagnostics
```

The second command regenerates every released automatic camera-ready diagnostic and validates all aggregate count arithmetic.

## Safety scan before a release

Run from the repository root:

```bash
rg -n -i "sk-[A-Za-z0-9]|api[_-]?key|secret|password|bearer[[:space:]]+[A-Za-z0-9._-]+|/Users/|randomization_key|annotator_[12]_preference|claim_audit_annotator" .
find . -name ".DS_Store" -print
```

Expected API-related matches are environment-variable names or placeholder documentation. No private annotation file, randomization key, annotator identity, personal path, or real credential should be present.
