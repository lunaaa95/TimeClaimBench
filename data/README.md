# Data

This artifact includes the processed data needed to inspect and reproduce the reported test-set evaluations.

- `synthetic/train.jsonl`: 1,200-example synthetic training split.
- `synthetic/dev.jsonl`: 200-example synthetic development split.
- `synthetic/test.jsonl`: 400-example synthetic test split used for the main results.
- `synthetic/test_stratified_80.jsonl`: 80-example pilot subset.
- `processed/test_graphs.jsonl`: extracted event graphs for the synthetic test split.
- `processed/test_gold_graphs.jsonl`: sparse graphs containing each synthetic series' designed primary event. They support recovery and reference-sensitivity diagnostics but are not exhaustive annotations of every valid local pattern.
- `processed/test_stratified_80_mismatched_graphs.jsonl`: mismatched-event diagnostic graphs.
- `processed/nab_windows_graphs.jsonl`: extracted event graphs for Real-NAB diagnostic windows.
- `real/nab_windows.jsonl`: processed Real-NAB localization diagnostic windows.

Raw third-party downloads, annotation sheets, annotator identities, free-text notes, and randomization keys are intentionally not included.
