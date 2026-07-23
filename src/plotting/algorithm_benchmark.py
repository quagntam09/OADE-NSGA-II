"""Plot unified algorithm benchmark outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.benchmarking.algorithm_benchmark import SUMMARY_CSV
from src.utils.config import load_yaml_config
from src.utils.csv_io import read_csv

from .common import ordered_unique, style_scientific_y_axis, to_float


DEFAULT_CONFIG_PATH = Path("config") / "algorithm_benchmark.yaml"


def plot_metric(rows: list[dict], metric: str, out_path: Path, with_error: bool, y_scale: str = "linear") -> None:
    problems = ordered_unique(row["problem"] for row in rows)
    algorithms = ordered_unique(row["algorithm"] for row in rows)
    data = {(row["problem"], row["algorithm"]): row for row in rows}

    width = min(0.22, 0.8 / max(1, len(algorithms)))
    x = list(range(len(problems)))
    fig, ax = plt.subplots(figsize=(14, 7))

    for i, algorithm in enumerate(algorithms):
        xpos = [value + (i - (len(algorithms) - 1) / 2) * width for value in x]
        if with_error:
            y = [to_float(data[(problem, algorithm)][f"{metric}_mean"]) for problem in problems]
            yerr = [to_float(data[(problem, algorithm)][f"{metric}_std"]) for problem in problems]
            ax.bar(xpos, y, width=width, yerr=yerr, capsize=3, label=algorithm)
        else:
            y = [to_float(data[(problem, algorithm)][f"{metric}_best"]) for problem in problems]
            ax.bar(xpos, y, width=width, label=algorithm)

    ax.set_xticks(x)
    ax.set_xticklabels([problem.upper() for problem in problems])
    ax.set_ylabel(metric.upper())
    scale_label = " Log Scale" if y_scale == "log" else ""
    ax.set_title(f"{metric.upper()} {'Mean +/- Std' if with_error else 'Best'}{scale_label}: Algorithm Benchmark")
    ax.legend()
    style_scientific_y_axis(ax)
    if y_scale == "log":
        ax.set_yscale("log")
    else:
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))

    heights = [patch.get_height() for patch in ax.patches]
    max_height = max(heights) if heights else 0.0
    if max_height > 0.0 and y_scale != "log":
        ax.set_ylim(0, max_height * 1.12)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args(argv)

    cfg = load_yaml_config(Path(args.config))
    out_dir = Path(cfg["global"]["out_dir"])
    rows = read_csv(out_dir / SUMMARY_CSV)
    outputs = cfg["plots"]

    plot_metric(rows, "IGD", out_dir / str(outputs["igd_mean_std"]), with_error=True)
    plot_metric(rows, "HV", out_dir / str(outputs["hv_mean_std"]), with_error=True)
    plot_metric(rows, "IGD", out_dir / str(outputs["igd_best"]), with_error=False)
    plot_metric(rows, "HV", out_dir / str(outputs["hv_best"]), with_error=False)
    if "igd_mean_std_log" in outputs:
        plot_metric(rows, "IGD", out_dir / str(outputs["igd_mean_std_log"]), with_error=True, y_scale="log")
    if "igd_best_log" in outputs:
        plot_metric(rows, "IGD", out_dir / str(outputs["igd_best_log"]), with_error=False, y_scale="log")

    print((out_dir / str(outputs["igd_mean_std"])).as_posix())
    print((out_dir / str(outputs["hv_mean_std"])).as_posix())
    print((out_dir / str(outputs["igd_best"])).as_posix())
    print((out_dir / str(outputs["hv_best"])).as_posix())
    if "igd_mean_std_log" in outputs:
        print((out_dir / str(outputs["igd_mean_std_log"])).as_posix())
    if "igd_best_log" in outputs:
        print((out_dir / str(outputs["igd_best_log"])).as_posix())


if __name__ == "__main__":
    main()
