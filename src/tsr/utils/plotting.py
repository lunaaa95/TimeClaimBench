"""Small plotting helpers."""

from __future__ import annotations

from pathlib import Path


def save_figure(fig, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
