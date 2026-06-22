"""Shared helpers for plotting benchmark outputs."""

from __future__ import annotations

from typing import Iterable, TypeVar

from matplotlib.ticker import ScalarFormatter


T = TypeVar("T")


def to_float(value: str | None) -> float:
    return float(value) if value not in {"", None} else float("nan")


def ordered_unique(values: Iterable[T]) -> list[T]:
    out = []
    for value in values:
        if value not in out:
            out.append(value)
    return out


def style_scientific_y_axis(ax) -> None:
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 2))
    ax.yaxis.set_major_formatter(formatter)
