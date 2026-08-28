.PHONY: test recompute-full tables diagnostics camera-ready-diagnostics validate-diagnostics smoke

export PYTHONPATH := src

test:
	pytest -q

recompute-full:
	python3 scripts/recompute_metrics_from_generations.py --output-dir outputs/predictions/full_synthetic_400 --graphs data/processed/test_graphs.jsonl --config configs/verifier.yaml

tables:
	python3 scripts/make_tables.py

diagnostics:
	python3 scripts/analyze_pattern_and_errors.py
	python3 scripts/analyze_coverage_informativeness.py
	python3 scripts/make_event_source_ablation_table.py
	python3 scripts/make_mismatched_event_table.py
	python3 scripts/analyze_real_nab_results.py

camera-ready-diagnostics:
	python3 scripts/camera_ready/analyze_event_extractor_recovery.py
	python3 scripts/camera_ready/analyze_synthetic_real_shape_coverage.py
	python3 scripts/camera_ready/analyze_gold_reference_sensitivity.py
	python3 scripts/camera_ready/validate_aggregate_diagnostics.py

validate-diagnostics:
	python3 scripts/camera_ready/validate_aggregate_diagnostics.py

smoke:
	python3 scripts/run_llm_generation.py --config configs/experiment.yaml --provider dummy --prompt event_grounded
