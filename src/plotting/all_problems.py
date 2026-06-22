"""Plot charts from all_problems benchmark CSV outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from src.utils.config import load_yaml_config
from src.utils.csv_io import read_csv

from .common import style_scientific_y_axis, to_float


DEFAULT_CONFIG_PATH = Path("config") / "plotting.yaml"


def plot_metric(rows: list[dict], metric: str, out_path: Path, with_error: bool) -> None:
    problems = []
    algos = []
    for row in rows:
        if row["problem"] not in problems:
            problems.append(row["problem"])
        if row["algorithm"] not in algos:
            algos.append(row["algorithm"])

    data = {(row["problem"], row["algorithm"]): row for row in rows}

    width = 0.18
    x = list(range(len(problems)))

    fig, ax = plt.subplots(figsize=(14, 7))

    for i, algo in enumerate(algos):
        xpos = [v + (i - (len(algos) - 1) / 2) * width for v in x]
        y = []
        yerr = []

        for problem in problems:
            row = data[(problem, algo)]
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
    ax.set_xticklabels([problem.upper() for problem in problems])
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric.upper()} {'Mean+/-Std' if with_error else 'Best'} Across ZDT Problems")
    ax.legend()
    style_scientific_y_axis(ax)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))

    y_max = max(patch.get_height() for patch in ax.patches) if ax.patches else 1.0
    ax.set_ylim(0, y_max * 1.10)

    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args(argv)

    cfg = load_yaml_config(Path(args.config))["all_problems"]
    results_dir = Path(cfg["results_dir"])
    outputs = cfg["outputs"]
    mean_rows = read_csv(results_dir / str(cfg["mean_std_csv"]))
    best_rows = read_csv(results_dir / str(cfg["best_csv"]))

    plot_metric(mean_rows, "IGD", results_dir / str(outputs["igd_mean_std"]), with_error=True)
    plot_metric(mean_rows, "HV", results_dir / str(outputs["hv_mean_std"]), with_error=True)
    plot_metric(best_rows, "IGD", results_dir / str(outputs["igd_best"]), with_error=False)
    plot_metric(best_rows, "HV", results_dir / str(outputs["hv_best"]), with_error=False)

    print((results_dir / str(outputs["igd_mean_std"])).as_posix())
    print((results_dir / str(outputs["hv_mean_std"])).as_posix())
    print((results_dir / str(outputs["igd_best"])).as_posix())
    print((results_dir / str(outputs["hv_best"])).as_posix())


if __name__ == "__main__":
    main()
