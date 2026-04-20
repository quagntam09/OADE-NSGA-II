"""
Draw HV and IGD bar charts from an ablation table CSV.

Expected CSV columns:
- problem
- algorithm
- hv_mean
- igd_mean

Example:
    .\\.venv\\Scripts\\python.exe comparison\\plot_hv_igd_from_table.py \
        outputs/table_ablation_hv_igd.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_INPUT_CSV = "outputs/table_ablation_hv_igd.csv"
DEFAULT_HV_OUTPUT = "outputs/ablation_hv_from_table.png"
DEFAULT_IGD_OUTPUT = "outputs/ablation_igd_from_table.png"


def _load_rows(path: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with open(path, "r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            rows.append(
                {
                    "problem": row["problem"],
                    "algorithm": row["algorithm"],
                    "hv_mean": float(row["hv_mean"]),
                    "igd_mean": float(row["igd_mean"]),
                }
            )
    return rows


def _short_algorithm_label(name: str) -> str:
    mapping = {
        "improved_nsga2_ablation_no_adaptive_de": "no adaptive DE",
        "improved_nsga2_ablation_no_obl_init": "no OBL init",
        "improved_nsga2_ablation_no_periodic_obl": "no periodic OBL",
        "improved_nsga2_ablation_no_restart": "no restart",
        "improved_nsga2_ablation_sbx_only": "SBX only",
    }
    if name in mapping:
        return mapping[name]

    for prefix in ("improved_nsga2_ablation_", "improved_nsga2_", "pymoo_"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    return name.replace("_", " ")


def _build_metric_map(
    rows: List[Dict[str, object]],
    metric_key: str,
) -> Tuple[List[str], List[str], Dict[Tuple[str, str], float]]:
    problems = sorted({str(r["problem"]) for r in rows})
    algorithms = sorted({str(r["algorithm"]) for r in rows})

    metric_map: Dict[Tuple[str, str], float] = {}
    for row in rows:
        metric_map[(str(row["problem"]), str(row["algorithm"]))] = float(row[metric_key])

    return problems, algorithms, metric_map


def _plot_grouped_bar(
    problems: List[str],
    algorithms: List[str],
    value_map: Dict[Tuple[str, str], float],
    title: str,
    y_label: str,
    output_path: str,
    ui_scale: float,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11.5 * ui_scale, 5.2 * ui_scale), constrained_layout=True)
    cmap = plt.get_cmap("tab20")

    x = np.arange(len(problems), dtype=float)
    n_algorithms = max(1, len(algorithms))
    total_width = 0.82
    bar_width = total_width / n_algorithms
    start = -total_width / 2.0 + bar_width / 2.0

    for i, algorithm in enumerate(algorithms):
        values = [value_map.get((problem, algorithm), 0.0) for problem in problems]
        positions = x + start + i * bar_width
        bars = ax.bar(
            positions,
            values,
            width=bar_width,
            label=_short_algorithm_label(algorithm),
            color=cmap(i % cmap.N),
            alpha=0.9,
        )
        labels = [f"{v:.3e}" if abs(v) < 1e-3 else f"{v:.4f}" for v in values]
        ax.bar_label(bars, labels=labels, padding=3, fontsize=max(7, int(8 * ui_scale)))

    ax.set_title(title, fontsize=13 * ui_scale)
    ax.set_xlabel("Problem", fontsize=11 * ui_scale)
    ax.set_ylabel(y_label, fontsize=11 * ui_scale)
    ax.set_xticks(x)
    ax.set_xticklabels(problems, fontsize=10 * ui_scale)
    ax.tick_params(axis="y", labelsize=10 * ui_scale)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=9 * ui_scale)

    fig.savefig(path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw HV and IGD bar charts from ablation table CSV")
    parser.add_argument("input_csv", nargs="?", default=DEFAULT_INPUT_CSV, help="Input ablation table CSV")
    parser.add_argument("--hv-output", default=DEFAULT_HV_OUTPUT, help="Output PNG for HV chart")
    parser.add_argument("--igd-output", default=DEFAULT_IGD_OUTPUT, help="Output PNG for IGD chart")
    parser.add_argument("--ui-scale", type=float, default=1.2, help="Global chart scaling")
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    rows = _load_rows(str(input_path))
    if not rows:
        raise ValueError("Input CSV is empty")

    problems, algorithms, hv_map = _build_metric_map(rows, "hv_mean")
    _, _, igd_map = _build_metric_map(rows, "igd_mean")

    _plot_grouped_bar(
        problems=problems,
        algorithms=algorithms,
        value_map=hv_map,
        title="Ablation HV by Problem",
        y_label="HV mean",
        output_path=args.hv_output,
        ui_scale=args.ui_scale,
    )
    _plot_grouped_bar(
        problems=problems,
        algorithms=algorithms,
        value_map=igd_map,
        title="Ablation IGD by Problem",
        y_label="IGD mean",
        output_path=args.igd_output,
        ui_scale=args.ui_scale,
    )

    print(f"Saved HV chart: {args.hv_output}")
    print(f"Saved IGD chart: {args.igd_output}")


if __name__ == "__main__":
    main()
