# Camera-ready diagnostics

This directory contains the additional diagnostics incorporated after the original evaluation. The automatic analyses can be regenerated from the released data and saved model outputs. Human-study files contain aggregate statistics only.

## Operational versus human evaluation

`Faith.` in the original automatic tables is the strict `SUPPORTED` rate under the released parser, extracted event graph, and matching rules. It measures **operational extracted-event support**, not human truth. The 40-series blinded preference study produces the opposite ordering for one model: event-grounded answers have the highest same-sample operational strict support (`0.711`, versus `0.577` for Direct and `0.355` for CoT), while Direct and CoT receive higher whole-explanation human ratings and preferences. This rank reversal is a substantive diagnostic: grounding may improve consistency with its supplied representation while introducing representation-linked claims that holistic raters do not prefer.

## Files

- `event_extractor_recovery.{csv,json}`: exact-type recovery at interval IoU >= `0.5`; `394/400` designed primary events are recovered. This is recall only because the sparse gold annotation is not exhaustive.
- `synthetic_real_shape_coverage.{csv,json}`: eight scale-invariant features after per-window z-normalization. Mean Real-NAB coverage by the synthetic 5th-95th percentile ranges is `0.845625`; this is not a distribution-equivalence claim.
- `gold_reference_prompt_summary.csv` and `gold_reference_sensitivity.json`: the same saved generations re-verified against sparse designed-event graphs. Strict support reverses under this reference; sparse gold graphs are not exhaustive human truth.
- `claim_audit_90.json` and `claim_audit_90_confusion_matrices.csv`: aggregate results from the independently sampled, two-annotator, 90-claim audit.
- `human_preference_40.json`, `human_preference_40.csv`, `human_preference_40_agreement.csv`, and `human_preference_40_by_pattern.csv`: aggregate blinded preference results and agreement.
- `human_preference_same_sample_automatic.csv`: operational verifier labels on the same 40 series.
- `period4_failure_diagnostics.json`: aggregate period-4 text counts, verifier mismatch-code counts, and two deidentified pipeline traces.

Raw annotator sheets, free-text notes, candidate packets, randomization keys, and annotator identities are intentionally excluded. Consequently, the human bootstrap intervals cannot be regenerated from this public directory; the released count tables can still be checked for internal arithmetic consistency.

## Commands

From the repository root:

```bash
PYTHONPATH=src python3 scripts/camera_ready/analyze_event_extractor_recovery.py
PYTHONPATH=src python3 scripts/camera_ready/analyze_synthetic_real_shape_coverage.py
PYTHONPATH=src python3 scripts/camera_ready/analyze_gold_reference_sensitivity.py
python3 scripts/camera_ready/validate_aggregate_diagnostics.py
```

The gold-reference analysis performs no API calls but processes all saved verification records and runs `5,000` paired bootstrap replicates, so it is slower than the other checks.
