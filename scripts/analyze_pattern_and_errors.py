#!/usr/bin/env python3
"""Generate offline pattern-wise diagnostics and representative error cases."""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsr.utils.io import read_jsonl, write_json, write_text


RESULT_DIR = ROOT / "outputs" / "predictions" / "full_synthetic_400"
ANALYSIS_DIR = ROOT / "outputs" / "analysis"
TABLE_DIR = ROOT / "outputs" / "tables"

PATTERN_ORDER = [
    "upward_trend",
    "downward_trend",
    "sharp_rise",
    "sharp_drop",
    "volatility_shift",
    "change_point",
    "seasonality",
    "anomaly",
]
PROMPT_ORDER = ["direct", "cot", "structured", "event_grounded"]

PATTERN_LABELS = {
    "upward_trend": "Upward trend",
    "downward_trend": "Downward trend",
    "sharp_rise": "Sharp rise",
    "sharp_drop": "Sharp drop",
    "volatility_shift": "Volatility shift",
    "change_point": "Change point",
    "seasonality": "Seasonality",
    "anomaly": "Anomaly",
}

PROMPT_LABELS = {
    "direct": "Direct",
    "cot": "CoT",
    "structured": "Structured",
    "event_grounded": "Event-grounded",
}

MODEL_LABELS = {
    ("openai", "gpt-4.1-mini"): "GPT-4.1-mini",
    ("openai", "gpt-5.4-mini"): "GPT-5.4-mini",
    ("deepseek", "deepseek-v4-flash"): "DeepSeek-V4-Flash",
    ("gemini", "gemini-3-flash-preview"): "Gemini-3-Flash",
    ("anthropic", "claude-sonnet-4-5-20250929"): "Claude Sonnet 4.5",
}

CAUSAL_RE = re.compile(
    r"\b(because|caus|due to|driven by|external|intervention|shock|fundamental)\b",
    re.IGNORECASE,
)


def _latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def _truncate_words(text: str, max_words: int = 30) -> str:
    words = text.replace("\n", " ").split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]) + "..."


def _parse_run(path: Path) -> tuple[str, str, str]:
    name = path.name[: -len("_verification.jsonl")]
    provider, model, prompt = name.split("__")
    return provider, model.replace("_", "."), prompt


def _display_model(provider: str, model: str) -> str:
    return MODEL_LABELS.get((provider, model), f"{provider} {model}")


def _metric(row: dict, label: str) -> float:
    n_claims = row["n_claims"]
    return row[label] / n_claims if n_claims else 0.0


def _claims_per_explanation(row: dict) -> float:
    n_records = row["n_records"]
    return row["n_claims"] / n_records if n_records else 0.0


def load_patterns() -> dict[str, str]:
    samples = read_jsonl(ROOT / "data" / "synthetic" / "test.jsonl")
    return {
        sample["id"]: sample.get("metadata", {}).get("pattern_types", ["unknown"])[0]
        for sample in samples
    }


def aggregate_pattern_metrics(patterns: dict[str, str]) -> list[dict]:
    aggregate = defaultdict(
        lambda: {
            "n_records": 0,
            "n_claims": 0,
            "supported": 0,
            "partial": 0,
            "unsupported": 0,
        }
    )

    for path in sorted(RESULT_DIR.glob("*_verification.jsonl")):
        provider, model, prompt = _parse_run(path)
        for record in read_jsonl(path):
            pattern = patterns.get(record["id"], "unknown")
            key = (prompt, pattern)
            aggregate[key]["n_records"] += 1
            for result in record.get("verification_results", []):
                label = result.get("label")
                if label not in {"supported", "partial", "unsupported"}:
                    continue
                aggregate[key]["n_claims"] += 1
                aggregate[key][label] += 1

    rows = []
    for prompt in PROMPT_ORDER:
        for pattern in PATTERN_ORDER:
            values = aggregate[(prompt, pattern)]
            rows.append(
                {
                    "prompt": prompt,
                    "pattern": pattern,
                    "n_explanations": values["n_records"],
                    "n_claims": values["n_claims"],
                    "faithfulness": _metric(values, "supported"),
                    "unsupported": _metric(values, "unsupported"),
                    "partial": _metric(values, "partial"),
                    "claims_per_explanation": _claims_per_explanation(values),
                }
            )
    return rows


def write_pattern_csv(rows: list[dict]) -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    path = ANALYSIS_DIR / "pattern_breakdown.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "prompt",
                "pattern",
                "n_explanations",
                "n_claims",
                "faithfulness",
                "unsupported",
                "partial",
                "claims_per_explanation",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def pattern_main_table(rows: list[dict]) -> str:
    by_key = {(row["prompt"], row["pattern"]): row for row in rows}
    body_rows = []
    for pattern in PATTERN_ORDER:
        direct = by_key[("direct", pattern)]
        grounded = by_key[("event_grounded", pattern)]
        delta = grounded["faithfulness"] - direct["faithfulness"]
        body_rows.append(
            (
                f"{PATTERN_LABELS[pattern]} & {direct['faithfulness']:.3f} & "
                f"{grounded['faithfulness']:.3f} & {delta:+.3f} & "
                f"{grounded['unsupported']:.3f} & {grounded['partial']:.3f} & "
                f"{grounded['claims_per_explanation']:.2f} \\\\"
            )
        )
    return r"""\begin{table*}[t]
\centering
\small
\resizebox{\textwidth}{!}{%%
\begin{tabular}{lrrrrrr}
\toprule
Pattern & Direct Faith. & Event Faith. & $\Delta$ Faith. & Event Unsup. & Event Partial & Event Claims/Expl. \\
\midrule
%s
\bottomrule
\end{tabular}
}
\caption{Pattern-wise diagnostic breakdown on the 400-example synthetic test split, pooled over the five full-test models. Direct and event-grounded columns compare claim faithfulness before repair. The event-grounded columns also show remaining unsupported and partial-support rates, exposing which temporal phenomena remain difficult after grounding.}
\label{tab:pattern-breakdown}
\end{table*}
""" % "\n".join(body_rows)


def pattern_appendix_table(rows: list[dict]) -> str:
    body_rows = []
    for pattern in PATTERN_ORDER:
        for prompt in PROMPT_ORDER:
            row = next(r for r in rows if r["pattern"] == pattern and r["prompt"] == prompt)
            body_rows.append(
                (
                    f"{PATTERN_LABELS[pattern]} & {PROMPT_LABELS[prompt]} & "
                    f"{row['faithfulness']:.3f} & {row['unsupported']:.3f} & "
                    f"{row['partial']:.3f} & {row['claims_per_explanation']:.2f} \\\\"
                )
            )
    return r"""\begin{table*}[t]
\centering
\small
\resizebox{\textwidth}{!}{%%
\begin{tabular}{llrrrr}
\toprule
Pattern & Prompt & Faith. & Unsup. & Partial & Claims/Expl. \\
\midrule
%s
\bottomrule
\end{tabular}
}
\caption{Full pattern-by-prompt diagnostics on the synthetic test split, pooled over the five full-test models. Metrics are computed before repair from claim-level verifier labels.}
\label{tab:pattern-prompt-appendix}
\end{table*}
""" % "\n".join(body_rows)


def _iter_verification_records(provider: str, model: str, prompt: str) -> Iterable[dict]:
    file_model = model.replace(".", "_")
    path = RESULT_DIR / f"{provider}__{file_model}__{prompt}_verification.jsonl"
    if not path.exists():
        return []
    return read_jsonl(path)


def _claim_by_id(record: dict) -> dict[str, dict]:
    return {claim["claim_id"]: claim for claim in record.get("claims", [])}


def _good_claim_text(text: str) -> bool:
    clean = text.strip()
    return len(clean) > 50 and not clean.startswith("#") and not clean.lower().startswith("explanation")


def _find_pattern_hallucination(patterns: dict[str, str]) -> dict:
    records = _iter_verification_records("openai", "gpt-4.1-mini", "direct")
    for record in records:
        gold_pattern = patterns[record["id"]]
        claims = _claim_by_id(record)
        for result in record.get("verification_results", []):
            claim = claims.get(result["claim_id"], {})
            text = claim.get("text", "")
            claim_type = claim.get("claim_type")
            if (
                result.get("label") == "unsupported"
                and result.get("error_type") in {"pattern_confusion", "pattern_hallucination"}
                and claim_type != gold_pattern
                and _good_claim_text(text)
                and not CAUSAL_RE.search(text)
            ):
                return _case_row(
                    "Unsupported pattern",
                    "openai",
                    "gpt-4.1-mini",
                    "direct",
                    record,
                    gold_pattern,
                    claim,
                    result,
                    "The model introduces an anomaly/change-point narrative not supported by the event graph.",
                )
    raise RuntimeError("Could not find unsupported pattern hallucination case.")


def _find_causal_hallucination(patterns: dict[str, str]) -> dict:
    records = _iter_verification_records("openai", "gpt-4.1-mini", "cot")
    candidates = []
    for record in records:
        gold_pattern = patterns[record["id"]]
        claims = _claim_by_id(record)
        for result in record.get("verification_results", []):
            claim = claims.get(result["claim_id"], {})
            text = claim.get("text", "")
            if result.get("label") == "unsupported" and _good_claim_text(text) and CAUSAL_RE.search(text):
                priority = (
                    int(bool(re.search(r"\b(external|caus|because|due to|driven by|intervention)\b", text, re.IGNORECASE))),
                    len(text),
                )
                candidates.append((priority, record, gold_pattern, claim, result))
    if candidates:
        _, record, gold_pattern, claim, result = max(candidates, key=lambda item: item[0])
        return _case_row(
            "Causal hallucination",
            "openai",
            "gpt-4.1-mini",
            "cot",
            record,
            gold_pattern,
            claim,
            result,
            "CoT turns a descriptive temporal change into an external-cause explanation.",
        )
    raise RuntimeError("Could not find causal hallucination case.")


def _find_interval_case(patterns: dict[str, str]) -> dict:
    records = _iter_verification_records("deepseek", "deepseek-v4-flash", "event_grounded")
    for record in records:
        gold_pattern = patterns[record["id"]]
        claims = _claim_by_id(record)
        for result in record.get("verification_results", []):
            claim = claims.get(result["claim_id"], {})
            text = claim.get("text", "")
            if (
                result.get("label") == "partial"
                and result.get("error_type") == "interval_mislocalization"
                and _good_claim_text(text)
                and any(marker in text.lower() for marker in ["from index", "between indices", "to index", "from the start"])
            ):
                return _case_row(
                    "Interval mislocalization",
                    "deepseek",
                    "deepseek-v4-flash",
                    "event_grounded",
                    record,
                    gold_pattern,
                    claim,
                    result,
                    "The claim identifies the right phenomenon but grounds it to a narrower or shifted span.",
                )
    raise RuntimeError("Could not find interval mislocalization case.")


def _case_row(
    failure_mode: str,
    provider: str,
    model: str,
    prompt: str,
    record: dict,
    gold_pattern: str,
    claim: dict,
    result: dict,
    interpretation: str,
) -> dict:
    return {
        "failure_mode": failure_mode,
        "provider": provider,
        "model": model,
        "model_display": _display_model(provider, model),
        "prompt": prompt,
        "prompt_display": PROMPT_LABELS[prompt],
        "series_id": record["id"],
        "gold_pattern": gold_pattern,
        "claim_text": claim.get("text", ""),
        "claim_type": claim.get("claim_type"),
        "verifier_label": result.get("label"),
        "error_type": result.get("error_type"),
        "support_interval": result.get("support_interval"),
        "interpretation": interpretation,
    }


def select_error_cases(patterns: dict[str, str]) -> list[dict]:
    cases = [
        _find_pattern_hallucination(patterns),
        _find_interval_case(patterns),
        _find_causal_hallucination(patterns),
    ]
    write_json(ANALYSIS_DIR / "error_cases.json", {"cases": cases})
    return cases


def error_case_table(cases: list[dict]) -> str:
    body_rows = []
    for case in cases:
        run = f"{case['model_display']} / {case['prompt_display']}"
        pattern = PATTERN_LABELS[case["gold_pattern"]]
        claim = _truncate_words(case["claim_text"], 28)
        diagnosis = (
            f"{case['verifier_label']} ({case['error_type']}). "
            f"{case['interpretation']}"
        )
        body_rows.append(
            (
                f"{_latex_escape(case['failure_mode'])} & {_latex_escape(run)} & "
                f"{_latex_escape(case['series_id'])}; {_latex_escape(pattern)} & "
                f"{_latex_escape(claim)} & {_latex_escape(diagnosis)} \\\\"
            )
        )
    return r"""\begin{table*}[t]
\centering
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}p{0.14\textwidth}p{0.15\textwidth}p{0.13\textwidth}p{0.32\textwidth}p{0.20\textwidth}@{}}
\toprule
Failure mode & Run & Series & Problematic claim & Verifier diagnosis \\
\midrule
%s
\bottomrule
\end{tabular}
\caption{Representative claim-level failure cases selected from the full synthetic verification outputs. The cases illustrate unsupported pattern introduction, interval mislocalization, and causal over-interpretation.}
\label{tab:error-cases}
\end{table*}
""" % "\n".join(body_rows)


def write_tables(pattern_rows: list[dict], cases: list[dict]) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    tables = {
        "pattern_breakdown.tex": pattern_main_table(pattern_rows),
        "pattern_prompt_appendix.tex": pattern_appendix_table(pattern_rows),
        "error_cases.tex": error_case_table(cases),
    }
    for name, content in tables.items():
        write_text(TABLE_DIR / name, content)
        print(f"Wrote {name}")


def main() -> None:
    patterns = load_patterns()
    pattern_rows = aggregate_pattern_metrics(patterns)
    write_pattern_csv(pattern_rows)
    cases = select_error_cases(patterns)
    write_tables(pattern_rows, cases)
    print(f"Wrote pattern diagnostics to {ANALYSIS_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
