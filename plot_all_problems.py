"""Plot charts from all_problems benchmark CSV outputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


def load_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def to_float(v: str) -> float:
    return float(v) if v is not None and v != "" else float("nan")


def plot_metric(rows, metric: str, out_path: Path, with_error: bool) -> None:
    problems = []
    algos = []
    for r in rows:
        if r["problem"] not in problems:
            problems.append(r["problem"])
        if r["algorithm"] not in algos:
            algos.append(r["algorithm"])

    data = {(r["problem"], r["algorithm"]): r for r in rows}

    width = 0.18
    x = list(range(len(problems)))

    fig, ax = plt.subplots(figsize=(14, 7))

    for i, algo in enumerate(algos):
        xpos = [v + (i - (len(algos) - 1) / 2) * width for v in x]
        y = []
        yerr = []

        for p in problems:
            row = data[(p, algo)]
            if with_error:
                y.append(to_float(row[f"{metric}_mean"]))
                yerr.append(to_float(row[f"{metric}_std"]))
            else:
                y.append(to_float(row[f"{metric}_best"]))

        if with_error:
            ax.bar(xpos, y, width=width, yerr=yerr, capsize=3, label=algo)
        else:
            ax.bar(xpos, y, width=width, label=algo)

    ax.set_xticks(x)
    ax.set_xticklabels([p.upper() for p in problems])
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric.upper()} {'Mean±Std' if with_error else 'Best'} Across ZDT Problems")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 2))
    ax.yaxis.set_major_formatter(formatter)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))

    y_max = max(p.get_height() for p in ax.patches) if ax.patches else 1.0
    ax.set_ylim(0, y_max * 1.10)

    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results_zdt")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    mean_rows = load_csv(results_dir / "all_problems_igd_hv_mean_std.csv")
    best_rows = load_csv(results_dir / "all_problems_igd_hv_best.csv")

    plot_metric(mean_rows, "IGD", results_dir / "plot_igd_mean_std.png", with_error=True)
    plot_metric(mean_rows, "HV", results_dir / "plot_hv_mean_std.png", with_error=True)
    plot_metric(best_rows, "IGD", results_dir / "plot_igd_best.png", with_error=False)
    plot_metric(best_rows, "HV", results_dir / "plot_hv_best.png", with_error=False)

    print((results_dir / "plot_igd_mean_std.png").as_posix())
    print((results_dir / "plot_hv_mean_std.png").as_posix())
    print((results_dir / "plot_igd_best.png").as_posix())
    print((results_dir / "plot_hv_best.png").as_posix())


if __name__ == "__main__":
    main()
