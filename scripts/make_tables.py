#!/usr/bin/env python3
"""Create LaTeX result table placeholders and script-generated summaries."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsr.utils.io import read_jsonl, write_text


PILOT_RESULT_DIRS = [
    ROOT / "outputs" / "predictions" / "multimodel_80",
    ROOT / "outputs" / "predictions" / "requested_80",
]

FULL_RESULT_DIRS = [
    ROOT / "outputs" / "predictions" / "full_synthetic_400",
]


def _metric_paths(result_dirs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for base in result_dirs:
        paths.extend(sorted(base.glob("*_metrics.json")))
    return paths


def _pilot_metric_paths() -> list[Path]:
    return _metric_paths(PILOT_RESULT_DIRS)


def _full_metric_paths() -> list[Path]:
    return _metric_paths(FULL_RESULT_DIRS)


def _display_model(provider: str, model: str) -> str:
    display = {
        ("openai", "gpt-4.1-mini"): "OpenAI GPT-4.1-mini",
        ("openai", "gpt-5.4-mini"): "OpenAI GPT-5.4-mini",
        ("deepseek", "deepseek-v4-flash"): "DeepSeek-V4-Flash",
        ("gemini", "gemini-3-flash-preview"): "Gemini-3-Flash",
        ("anthropic", "claude-sonnet-4-5-20250929"): "Claude Sonnet 4.5",
    }
    return display.get((provider, model), f"{provider} {model}")


def _count_jsonl(path: Path) -> int:
    return len(read_jsonl(path)) if path.exists() else 0


def _copy_to_outputs(name: str, content: str) -> None:
    write_text(ROOT / "outputs" / "tables" / name, content)


def dataset_stats_table() -> str:
    rows = []
    for split in ["train", "dev", "test"]:
        path = ROOT / "data" / "synthetic" / f"{split}.jsonl"
        n = _count_jsonl(path)
        claims = n
        events = n
        patterns = 8 if n else 0
        rows.append(f"Synthetic {split.title()} & {n if n else '--'} & {claims if claims else '--'} & {events if events else '--'} & {patterns if patterns else '--'} \\\\")
    real_nab_path = ROOT / "data" / "real" / "nab_windows.jsonl"
    if real_nab_path.exists():
        real_rows = read_jsonl(real_nab_path)
        n = len(real_rows)
        claims = sum(len(row.get("reference_claims", [])) for row in real_rows)
        events = sum(len(row.get("events", [])) for row in real_rows)
        patterns = len({tuple(row.get("metadata", {}).get("pattern_types", [])) for row in real_rows})
        rows.append(f"Real-NAB Extension & {n} & {claims} & {events} & {patterns} \\\\")
    return r"""\begin{table}[t]
\centering
\small
\resizebox{\columnwidth}{!}{%%
\begin{tabular}{lrrrr}
\toprule
Split & Series & Claims & Events & Patterns \\
\midrule
%s
\bottomrule
\end{tabular}
}
\caption{Dataset statistics. Values are generated from local JSONL files when available.}
\label{tab:dataset}
\end{table}
""" % "\n".join(rows)


def generation_results_table() -> str:
    return r"""\begin{table*}[t]
\centering
\small
\begin{tabular}{lrrrrr}
\toprule
Method & ClaimFaith. $\uparrow$ & Unsup. Rate $\downarrow$ & Interval IoU $\uparrow$ & Chron. Viol. $\downarrow$ & Claims/Expl. \\
\midrule
Direct Prompt & -- & -- & -- & -- & -- \\
CoT Prompt & -- & -- & -- & -- & -- \\
Structured Prompt & -- & -- & -- & -- & -- \\
Tool Prompt & -- & -- & -- & -- & -- \\
Event-Grounded Prompt & -- & -- & -- & -- & -- \\
Event-Grounded + Repair & -- & -- & -- & -- & -- \\
\bottomrule
\end{tabular}
\caption{Explanation generation results. Placeholder rows remain unset until non-dummy experiments are executed.}
\label{tab:generation}
\end{table*}
"""


def verification_results_table() -> str:
    path = ROOT / "outputs" / "verification_baselines" / "full_synthetic_400" / "verification_baselines.json"
    if path.exists():
        import json

        reports = json.loads(path.read_text(encoding="utf-8"))["reports"]
        rows = "\n".join(
            (
                f"{report['name']} & {report['macro_f1']:.3f} & {report['supported_f1']:.3f} "
                f"& {report['partial_f1']:.3f} & {report['unsupported_f1']:.3f} "
                f"& {report['support_iou']:.3f} \\\\"
            )
            for report in reports
        )
        return r"""\begin{table}[t]
\centering
\small
\resizebox{\columnwidth}{!}{%%
\begin{tabular}{lrrrrr}
\toprule
Verifier & Macro-F1 & Sup. F1 & Part. F1 & Unsup. F1 & IoU \\
\midrule
%s
\bottomrule
\end{tabular}
}
\caption{Verifier baseline and ablation results on 1,600 synthetic claim-level cases derived from the 400-example test split. The benchmark contains one supported claim, one interval-mislocalized partial claim, and two unsupported claims per series. IoU is computed for supported and partial cases with predicted support intervals.}
\label{tab:verification}
\end{table}
""" % rows
    return r"""\begin{table}[t]
\centering
\small
\resizebox{\columnwidth}{!}{%%
\begin{tabular}{lrrrr}
\toprule
Verifier & Macro-F1 & Sup. F1 & Unsup. F1 & IoU \\
\midrule
LLM-as-Judge & -- & -- & -- & -- \\
Rule-only & -- & -- & -- & -- \\
Event Graph & -- & -- & -- & -- \\
Event Graph + LLM Parser & -- & -- & -- & -- \\
\bottomrule
\end{tabular}
}
\caption{Claim-level verification results. Fill these rows after labeled verification experiments.}
\label{tab:verification}
\end{table}
"""


def pilot_results_table() -> str:
    metric_paths = _pilot_metric_paths()
    if not metric_paths:
        return r"""\begin{table*}[t]
\centering
\small
\begin{tabular}{llrrrrrrr}
\toprule
Model & Prompt & Faith. & Unsup. & Claims/Expl. & Faith.+Repair & Unsup.+Repair & Retention & Cost/Expl. \\
\midrule
-- & -- & -- & -- & -- & -- & -- & -- & -- \\
\bottomrule
\end{tabular}
\caption{Pilot results on the stratified synthetic subset.}
\label{tab:pilot-results}
\end{table*}
"""

    import json

    order = {"direct": 0, "cot": 1, "structured": 2, "event_grounded": 3}
    rows = []
    for path in metric_paths:
        name = path.stem[: -len("_metrics")]
        provider, model, prompt = name.split("__")
        model = model.replace("_", ".")
        metrics = json.loads(path.read_text(encoding="utf-8"))
        before = metrics["before_repair"]
        after = metrics["after_repair"]
        cost = metrics["cost"]
        rows.append(
            {
                "model": _display_model(provider, model),
                "prompt": prompt.replace("_", "-"),
                "prompt_key": prompt,
                "faith": before["claim_faithfulness"],
                "unsup": before["unsupported_claim_rate"],
                "claims": before["claims_per_explanation"],
                "faith_repair": after["claim_faithfulness"],
                "unsup_repair": after["unsupported_claim_rate"],
                "retention": metrics["claim_retention_rate"],
                "cost": cost["cost_per_explanation_usd"],
            }
        )
    rows.sort(key=lambda row: (row["model"], order.get(row["prompt_key"], 99)))
    body = "\n".join(
        (
            f"{row['model']} & {row['prompt']} & {row['faith']:.3f} & {row['unsup']:.3f} "
            f"& {row['claims']:.2f} & {row['faith_repair']:.3f} & {row['unsup_repair']:.3f} "
            f"& {row['retention']:.3f} & \\${row['cost']:.5f} \\\\"
        )
        for row in rows
    )
    return r"""\begin{table*}[t]
\centering
\small
\resizebox{\textwidth}{!}{%%
\begin{tabular}{llrrrrrrr}
\toprule
Model & Prompt & Faith. $\uparrow$ & Unsup. $\downarrow$ & Claims/Expl. & Faith.+Repair $\uparrow$ & Unsup.+Repair $\downarrow$ & Retention $\uparrow$ & Cost/Expl. $\downarrow$ \\
\midrule
%s
\bottomrule
\end{tabular}
}
\caption{Multi-provider pilot results on the 80-example stratified synthetic subset. Faith. denotes claim faithfulness before repair. Retention measures the fraction of extracted claims retained after verifier-guided repair. Costs are estimated from provider token usage and the pricing configuration.}
\label{tab:pilot-results}
\end{table*}
""" % body


def pilot_appendix_table() -> str:
    metric_paths = _pilot_metric_paths()
    if not metric_paths:
        return "% Pilot appendix table unavailable until experiments are run.\n"
    import json

    order = {"direct": 0, "cot": 1, "structured": 2, "event_grounded": 3}
    rows = []
    for path in metric_paths:
        name = path.stem[: -len("_metrics")]
        provider, model, prompt = name.split("__")
        model = model.replace("_", ".")
        metrics = json.loads(path.read_text(encoding="utf-8"))
        before = metrics["before_repair"]
        after = metrics["after_repair"]
        cost = metrics["cost"]
        rows.append(
            {
                "provider": provider,
                "model": _display_model(provider, model),
                "prompt": prompt.replace("_", "-"),
                "prompt_key": prompt,
                "partial": before["partial_support_rate"],
                "chronology": before["chronology_violation_rate"],
                "claims_after": after["claims_per_explanation"],
                "preservation": metrics["supported_claim_preservation"],
                "total_cost": cost["generation_cost_usd"],
                "tokens": cost["tokens_per_explanation"],
            }
        )
    rows.sort(key=lambda row: (row["model"], order.get(row["prompt_key"], 99)))
    body = "\n".join(
        (
            f"{row['model']} & {row['prompt']} & {row['partial']:.3f} & {row['chronology']:.3f} "
            f"& {row['claims_after']:.2f} "
            f"& {row['preservation']:.3f} & \\${row['total_cost']:.4f} & {row['tokens']:.1f} \\\\"
        )
        for row in rows
    )
    return r"""\begin{table*}[t]
\centering
\small
\resizebox{\textwidth}{!}{%%
\begin{tabular}{llrrrrrr}
\toprule
Model & Prompt & Partial Before & Chron. Viol. & Claims/Expl. After & Semantic Pres. & Total Cost & Tokens/Expl. \\
\midrule
%s
\bottomrule
\end{tabular}
}
\caption{Additional multi-provider pilot diagnostics. Semantic preservation is implemented as supported-claim preservation: an originally supported claim is counted as preserved when the repaired explanation contains a supported claim grounded by an overlapping support event. Total cost is measured over the 80 examples in each model--prompt run.}
\label{tab:pilot-appendix}
\end{table*}
""" % body


def full_results_table() -> str:
    metric_paths = _full_metric_paths()
    if not metric_paths:
        return "% Full synthetic results unavailable until experiments are run.\n"

    import json

    order = {"direct": 0, "cot": 1, "structured": 2, "event_grounded": 3}
    rows = []
    for path in metric_paths:
        name = path.stem[: -len("_metrics")]
        provider, model, prompt = name.split("__")
        model = model.replace("_", ".")
        metrics = json.loads(path.read_text(encoding="utf-8"))
        before = metrics["before_repair"]
        after = metrics["after_repair"]
        cost = metrics["cost"]
        rows.append(
            {
                "model": _display_model(provider, model),
                "prompt": prompt.replace("_", "-"),
                "prompt_key": prompt,
                "faith": before["claim_faithfulness"],
                "unsup": before["unsupported_claim_rate"],
                "claims": before["claims_per_explanation"],
                "faith_repair": after["claim_faithfulness"],
                "unsup_repair": after["unsupported_claim_rate"],
                "retention": metrics["claim_retention_rate"],
                "cost": cost["cost_per_explanation_usd"],
            }
        )
    rows.sort(key=lambda row: (row["model"], order.get(row["prompt_key"], 99)))
    body = "\n".join(
        (
            f"{row['model']} & {row['prompt']} & {row['faith']:.3f} & {row['unsup']:.3f} "
            f"& {row['claims']:.2f} & {row['faith_repair']:.3f} & {row['unsup_repair']:.3f} "
            f"& {row['retention']:.3f} & \\${row['cost']:.5f} \\\\"
        )
        for row in rows
    )
    return r"""\begin{table*}[t]
\centering
\small
\resizebox{\textwidth}{!}{%%
\begin{tabular}{llrrrrrrr}
\toprule
Model & Prompt & Faith. $\uparrow$ & Unsup. $\downarrow$ & Claims/Expl. & Faith.+Repair $\uparrow$ & Unsup.+Repair $\downarrow$ & Retention $\uparrow$ & Cost/Expl. $\downarrow$ \\
\midrule
%s
\bottomrule
\end{tabular}
}
\caption{Full synthetic test results on 400 examples across five LLMs. Faith. denotes claim faithfulness before repair. Retention measures the fraction of extracted claims retained after verifier-guided repair. Costs are estimated from provider token usage and the pricing configuration.}
\label{tab:full-results}
\end{table*}
""" % body


def full_appendix_table() -> str:
    metric_paths = _full_metric_paths()
    if not metric_paths:
        return "% Full synthetic appendix table unavailable until experiments are run.\n"
    import json

    order = {"direct": 0, "cot": 1, "structured": 2, "event_grounded": 3}
    rows = []
    for path in metric_paths:
        name = path.stem[: -len("_metrics")]
        provider, model, prompt = name.split("__")
        model = model.replace("_", ".")
        metrics = json.loads(path.read_text(encoding="utf-8"))
        before = metrics["before_repair"]
        after = metrics["after_repair"]
        cost = metrics["cost"]
        rows.append(
            {
                "model": _display_model(provider, model),
                "prompt": prompt.replace("_", "-"),
                "prompt_key": prompt,
                "partial": before["partial_support_rate"],
                "chronology": before["chronology_violation_rate"],
                "claims_after": after["claims_per_explanation"],
                "preservation": metrics["supported_claim_preservation"],
                "total_cost": cost["generation_cost_usd"],
                "tokens": cost["tokens_per_explanation"],
            }
        )
    rows.sort(key=lambda row: (row["model"], order.get(row["prompt_key"], 99)))
    body = "\n".join(
        (
            f"{row['model']} & {row['prompt']} & {row['partial']:.3f} & {row['chronology']:.3f} "
            f"& {row['claims_after']:.2f} "
            f"& {row['preservation']:.3f} & \\${row['total_cost']:.4f} & {row['tokens']:.1f} \\\\"
        )
        for row in rows
    )
    return r"""\begin{table*}[t]
\centering
\small
\resizebox{\textwidth}{!}{%%
\begin{tabular}{llrrrrrr}
\toprule
Model & Prompt & Partial Before & Chron. Viol. & Claims/Expl. After & Semantic Pres. & Total Cost & Tokens/Expl. \\
\midrule
%s
\bottomrule
\end{tabular}
}
\caption{Additional full synthetic diagnostics on 400 examples. Semantic preservation is implemented as supported-claim preservation: an originally supported claim is counted as preserved when the repaired explanation contains a supported claim grounded by an overlapping support event.}
\label{tab:full-appendix}
\end{table*}
""" % body


def main() -> None:
    tables = {
        "dataset_stats.tex": dataset_stats_table(),
        "generation_results.tex": generation_results_table(),
        "verification_results.tex": verification_results_table(),
        "full_results.tex": full_results_table(),
        "full_appendix.tex": full_appendix_table(),
        "pilot_results.tex": pilot_results_table(),
        "pilot_appendix.tex": pilot_appendix_table(),
    }
    for name, content in tables.items():
        _copy_to_outputs(name, content)
        print(f"Wrote {name}")


if __name__ == "__main__":
    main()
