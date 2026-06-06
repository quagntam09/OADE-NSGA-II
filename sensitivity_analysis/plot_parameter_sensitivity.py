"""
Plot parameter sensitivity outputs.

Input files are produced by run_parameter_sensitivity.py. This script does not
touch the main plot_all_problems.py benchmark plotting script.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def to_float(value: str) -> float:
    return float(value) if value not in {"", None} else float("nan")


def ordered_unique(values) -> list[str]:
    out = []
    for value in values:
        if value not in out:
            out.append(value)
    return out


def summary_lookup(summary_rows: list[dict], study: str, problem: str, label: str) -> dict | None:
    for row in summary_rows:
        if row["study"] == study and row["problem"] == problem and row["parameter_label"] == label:
            return row
    return None


def style_axis(ax) -> None:
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 2))
    ax.yaxis.set_major_formatter(formatter)


def plot_metric(summary_rows: list[dict], study: str, metric: str, out_dir: Path) -> None:
    study_rows = [row for row in summary_rows if row["study"] == study]
    labels = ordered_unique(row["parameter_label"] for row in study_rows)
    problems = ordered_unique(row["problem"] for row in study_rows)
    if not labels or not problems:
        return

    ncols = 2
    nrows = math.ceil(len(problems) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4.2 * nrows), squeeze=False)

    for idx, problem in enumerate(problems):
        ax = axes[idx // ncols][idx % ncols]
        means = []
        errors = []
        for label in labels:
            row = summary_lookup(summary_rows, study, problem, label)
            means.append(to_float(row[f"{metric}_mean"]) if row else float("nan"))
            errors.append(to_float(row[f"{metric}_std"]) if row else 0.0)

        x = list(range(len(labels)))
        ax.errorbar(x, means, yerr=errors, marker="o", capsize=4, linewidth=1.6)
        ax.set_title(problem.upper())
        ax.set_ylabel(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        style_axis(ax)

    for idx in range(len(problems), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle(f"{study}: {metric} mean +/- std", y=0.995)
    fig.tight_layout()
    fig.savefig(out_dir / f"{study}_{metric.lower()}_sensitivity.png", dpi=220)
    plt.close(fig)


def plot_runtime_restart(summary_rows: list[dict], study: str, out_dir: Path) -> None:
    study_rows = [row for row in summary_rows if row["study"] == study]
    labels = ordered_unique(row["parameter_label"] for row in study_rows)
    problems = ordered_unique(row["problem"] for row in study_rows)
    if not labels or not problems:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), squeeze=False)
    for ax, field, title in [
        (axes[0][0], "runtime_mean", "Runtime"),
        (axes[0][1], "restart_count_mean", "Restart count"),
    ]:
        for problem in problems:
            values = []
            for label in labels:
                row = summary_lookup(summary_rows, study, problem, label)
                values.append(to_float(row[field]) if row else float("nan"))
            ax.plot(labels, values, marker="o", linewidth=1.4, label=problem.upper())
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=35)
        style_axis(ax)

    axes[0][0].set_ylabel("seconds")
    axes[0][1].set_ylabel("mean restarts")
    axes[0][1].legend(loc="best", fontsize=8)
    fig.suptitle(f"{study}: runtime and restart behavior", y=0.995)
    fig.tight_layout()
    fig.savefig(out_dir / f"{study}_runtime_restart.png", dpi=220)
    plt.close(fig)


def plot_fcr_trace(trace_rows: list[dict], out_dir: Path) -> None:
    rows = [row for row in trace_rows if row["study"] == "f_cr_init"]
    if not rows:
        return

    labels = ordered_unique(row["parameter_label"] for row in rows)
    grouped = {}
    for row in rows:
        key = (row["parameter_label"], int(row["generation"]))
        item = grouped.setdefault(key, {"mean_F": 0.0, "mean_CR": 0.0, "count": 0})
        item["mean_F"] += to_float(row["mean_F"])
        item["mean_CR"] += to_float(row["mean_CR"])
        item["count"] += 1

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), squeeze=False)
    for label in labels:
        generations = sorted(gen for current_label, gen in grouped if current_label == label)
        mean_f = [grouped[(label, gen)]["mean_F"] / grouped[(label, gen)]["count"] for gen in generations]
        mean_cr = [grouped[(label, gen)]["mean_CR"] / grouped[(label, gen)]["count"] for gen in generations]
        axes[0][0].plot(generations, mean_f, linewidth=1.4, label=label)
        axes[0][1].plot(generations, mean_cr, linewidth=1.4, label=label)

    axes[0][0].set_title("Mean F trace")
    axes[0][1].set_title("Mean CR trace")
    for ax in axes[0]:
        ax.set_xlabel("generation")
        style_axis(ax)
    axes[0][0].set_ylabel("mean F")
    axes[0][1].set_ylabel("mean CR")
    axes[0][1].legend(loc="best", fontsize=8)
    fig.suptitle("F/CR initialization sensitivity", y=0.995)
    fig.tight_layout()
    fig.savefig(out_dir / "f_cr_init_trace.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="sensitivity_analysis/results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = read_csv(results_dir / "sensitivity_summary.csv")
    trace_rows = read_csv(results_dir / "sensitivity_fcr_trace.csv")
    studies = ordered_unique(row["study"] for row in summary_rows)

    for study in studies:
        plot_metric(summary_rows, study, "IGD", plots_dir)
        plot_metric(summary_rows, study, "HV", plots_dir)
        plot_runtime_restart(summary_rows, study, plots_dir)
    plot_fcr_trace(trace_rows, plots_dir)

    print(f"Saved plots to: {plots_dir.as_posix()}")


if __name__ == "__main__":
    main()
