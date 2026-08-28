#!/usr/bin/env python3
"""Create the deterministic event verbalizer baseline table."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsr.utils.io import write_text


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _read_metrics(path: Path) -> dict:
    metrics = json.loads(path.read_text(encoding="utf-8"))
    before = metrics["before_repair"]
    cost = metrics["cost"]
    return {
        "faith": before["claim_faithfulness"],
        "unsupported": before["unsupported_claim_rate"],
        "partial": before["partial_support_rate"],
        "claims": before["claims_per_explanation"],
        "cost": cost["cost_per_explanation_usd"],
    }


def _latex_table(synthetic: dict, real_nab: dict) -> str:
    body = "\n".join(
        [
            (
                f"Synthetic test & 400 & {synthetic['faith']:.3f} & {synthetic['unsupported']:.3f} "
                f"& {synthetic['partial']:.3f} & {synthetic['claims']:.2f} & \\${synthetic['cost']:.5f} \\\\"
            ),
            (
                f"Real-NAB & 200 & {real_nab['faith']:.3f} & {real_nab['unsupported']:.3f} "
                f"& {real_nab['partial']:.3f} & {real_nab['claims']:.2f} & \\${real_nab['cost']:.5f} \\\\"
            ),
        ]
    )
    return r"""\begin{table}[t]
\centering
\small
\resizebox{\columnwidth}{!}{%%
\begin{tabular}{lrrrrrr}
\toprule
Dataset & N & Faith. $\uparrow$ & Unsup. $\downarrow$ & Partial & Claims/Expl. & Cost/Expl. \\
\midrule
%s
\bottomrule
\end{tabular}
}
\caption{Deterministic event verbalizer baseline. The baseline converts the extracted event graph directly into template sentences and is evaluated by the same claim parser and verifier as LLM outputs. It is not an LLM generation method; it tests how much of the automatic score can be achieved by directly verbalizing the event extractor.}
\label{tab:event-verbalizer}
\end{table}
""" % body


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--synthetic-metrics",
        default="outputs/predictions/event_verbalizer/deterministic__event-verbalizer__event_grounded_metrics.json",
    )
    parser.add_argument(
        "--real-nab-metrics",
        default="outputs/predictions/real_nab/deterministic__event-verbalizer__event_grounded_metrics.json",
    )
    parser.add_argument("--table", default="outputs/tables/event_verbalizer_baseline.tex")
    args = parser.parse_args()

    table = _latex_table(
        synthetic=_read_metrics(_resolve(args.synthetic_metrics)),
        real_nab=_read_metrics(_resolve(args.real_nab_metrics)),
    )
    write_text(_resolve(args.table), table)
    write_text(ROOT / "outputs" / "tables" / "event_verbalizer_baseline.tex", table)
    print(f"Table: {args.table}")


if __name__ == "__main__":
    main()
