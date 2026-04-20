"""
Generate only the required IGD/HV bar charts and summary tables.

Charts:
- Best IGD bar chart
- Best HV bar chart
- Mean IGD bar chart
- Mean HV bar chart
- Ablation IGD bar chart
- Ablation HV bar chart

Tables:
- Best IGD/HV statistics table
- Mean IGD/HV statistics table
- Ablation IGD/HV statistics table
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_RESULTS_CSV_PATH = "outputs/benchmark_results.csv"
DEFAULT_SUMMARY_CSV_PATH = "outputs/benchmark_summary.csv"
DEFAULT_ABLATION_SUMMARY_CSV_PATH = "outputs/benchmark_summary_ablation.csv"

DEFAULT_HV_BEST_OUTPUT_PATH = "outputs/benchmark_hv_best_bar.png"
DEFAULT_IGD_BEST_OUTPUT_PATH = "outputs/benchmark_igd_best_bar.png"
DEFAULT_HV_MEAN_OUTPUT_PATH = "outputs/benchmark_hv_mean_bar.png"
DEFAULT_IGD_MEAN_OUTPUT_PATH = "outputs/benchmark_igd_mean_bar.png"
DEFAULT_HV_ABLATION_OUTPUT_PATH = "outputs/ablation_hv_bar.png"
DEFAULT_IGD_ABLATION_OUTPUT_PATH = "outputs/ablation_igd_bar.png"

DEFAULT_BEST_TABLE_PATH = "outputs/table_best_hv_igd.csv"
DEFAULT_MEAN_TABLE_PATH = "outputs/table_mean_hv_igd.csv"
DEFAULT_ABLATION_TABLE_PATH = "outputs/table_ablation_hv_igd.csv"
DEFAULT_BEST_TABLE_MD_PATH = "outputs/table_best_hv_igd.md"
DEFAULT_MEAN_TABLE_MD_PATH = "outputs/table_mean_hv_igd.md"
DEFAULT_ABLATION_TABLE_MD_PATH = "outputs/table_ablation_hv_igd.md"

DEFAULT_ABLATION_PREFIX = "improved_nsga2_ablation_"
DEFAULT_FULL_ALGORITHM = "improved_nsga2"
DEFAULT_LABEL_EXPONENT = -1


def _format_scaled_label(value: float, digits: int = 2, exponent: int = DEFAULT_LABEL_EXPONENT) -> str:
    scaled = value / (10 ** exponent)
    return f"{scaled:.{digits}f}"


def _load_results_csv(path: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with open(path, "r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            rows.append(
                {
                    "problem": row["problem"],
                    "algorithm": row["algorithm"],
                    "hv": float(row["hv"]),
                    "igd": float(row["igd"]),
                }
            )
    return rows


def _load_summary_csv(path: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with open(path, "r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            rows.append(
                {
                    "problem": row["problem"],
                    "algorithm": row["algorithm"],
                    "runs": int(row.get("runs", "0")),
                    "hv_mean": float(row["hv_mean"]),
                    "hv_std": float(row["hv_std"]),
                    "igd_mean": float(row["igd_mean"]),
                    "igd_std": float(row["igd_std"]),
                }
            )
    return rows


def _write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        output_path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format_markdown_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _write_markdown_table(path: str, rows: List[Dict[str, object]], title: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        output_path.write_text("No rows\n", encoding="utf-8")
        return

    headers = list(rows[0].keys())
    lines: List[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for row in rows:
        row_values = [_format_markdown_value(row[h]) for h in headers]
        lines.append("| " + " | ".join(row_values) + " |")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _short_algorithm_label(algorithm: str) -> str:
    known = {
        "improved_nsga2": "full model",
        "improved_nsga2_ablation_sbx_only": "SBX only",
        "improved_nsga2_ablation_no_adaptive_de": "no adaptive DE",
        "improved_nsga2_ablation_no_obl_init": "no OBL init",
        "improved_nsga2_ablation_no_periodic_obl": "no periodic OBL",
        "improved_nsga2_ablation_no_restart": "no restart",
        "pymoo_nsga2": "NSGA-II",
    }
    if algorithm in known:
        return known[algorithm]

    label = algorithm
    for prefix in (
        "improved_nsga2_ablation_",
        "improved_nsga2_incremental_",
        "improved_nsga2_",
        "pymoo_",
    ):
        if label.startswith(prefix):
            label = label[len(prefix) :]
            break

    return label.replace("_", " ")


def _build_best_table(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str], Dict[str, float]] = {}
    for row in rows:
        key = (str(row["problem"]), str(row["algorithm"]))
        current = grouped.get(key)
        hv = float(row["hv"])
        igd = float(row["igd"])

        if current is None:
            grouped[key] = {
                "hv_best": hv,
                "igd_best": igd,
            }
        else:
            current["hv_best"] = max(current["hv_best"], hv)
            current["igd_best"] = min(current["igd_best"], igd)

    table: List[Dict[str, object]] = []
    for (problem, algorithm), values in sorted(grouped.items()):
        table.append(
            {
                "problem": problem,
                "algorithm": algorithm,
                "hv_best": float(values["hv_best"]),
                "igd_best": float(values["igd_best"]),
            }
        )
    return table


def _build_mean_table(summary_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    table: List[Dict[str, object]] = []
    for row in sorted(summary_rows, key=lambda r: (str(r["problem"]), str(r["algorithm"]))):
        table.append(
            {
                "problem": str(row["problem"]),
                "algorithm": str(row["algorithm"]),
                "runs": int(row["runs"]),
                "hv_mean": float(row["hv_mean"]),
                "hv_std": float(row["hv_std"]),
                "igd_mean": float(row["igd_mean"]),
                "igd_std": float(row["igd_std"]),
            }
        )
    return table


def _build_ablation_table(
    summary_rows: List[Dict[str, object]],
    full_algorithm: str,
    ablation_prefix: str,
) -> List[Dict[str, object]]:
    lookup: Dict[Tuple[str, str], Dict[str, object]] = {
        (str(row["problem"]), str(row["algorithm"])): row for row in summary_rows
    }

    problems = sorted({str(row["problem"]) for row in summary_rows})
    algorithms = sorted({str(row["algorithm"]) for row in summary_rows})
    ablation_algorithms = [algo for algo in algorithms if algo.startswith(ablation_prefix)]

    table: List[Dict[str, object]] = []
    for problem in problems:
        full_key = (problem, full_algorithm)
        if full_key not in lookup:
            continue

        full_row = lookup[full_key]
        full_hv = float(full_row["hv_mean"])
        full_igd = float(full_row["igd_mean"])

        for algo in ablation_algorithms:
            key = (problem, algo)
            if key not in lookup:
                continue

            abl_row = lookup[key]
            hv_mean = float(abl_row["hv_mean"])
            igd_mean = float(abl_row["igd_mean"])
            hv_delta = hv_mean - full_hv
            igd_delta = igd_mean - full_igd

            table.append(
                {
                    "problem": problem,
                    "algorithm": algo,
                    "hv_mean": hv_mean,
                    "igd_mean": igd_mean,
                    "delta_hv_vs_full": hv_delta,
                    "delta_igd_vs_full": igd_delta,
                }
            )

    return sorted(table, key=lambda r: (str(r["problem"]), str(r["algorithm"])))


def _build_grouped_metric_map(
    table_rows: List[Dict[str, object]],
    metric_key: str,
) -> Tuple[List[str], List[str], Dict[Tuple[str, str], float]]:
    problems = sorted({str(row["problem"]) for row in table_rows})
    algorithms = sorted({str(row["algorithm"]) for row in table_rows})

    values: Dict[Tuple[str, str], float] = {}
    for row in table_rows:
        values[(str(row["problem"]), str(row["algorithm"]))] = float(row[metric_key])

    return problems, algorithms, values


def _build_ablation_aggregate_map(
    table_rows: List[Dict[str, object]],
    metric_key: str,
) -> Tuple[List[str], Dict[str, float]]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for row in table_rows:
        grouped[str(row["algorithm"])].append(float(row[metric_key]))

    algorithms = sorted(grouped.keys())
    values = {algo: float(np.mean(grouped[algo])) for algo in algorithms}
    return algorithms, values


def _plot_grouped_problem_bar(
    problems: List[str],
    algorithms: List[str],
    values: Dict[Tuple[str, str], float],
    title: str,
    y_label: str,
    output_path: str,
    ui_scale: float,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12.0 * ui_scale, 5.2 * ui_scale), constrained_layout=True)
    cmap = plt.get_cmap("tab20")

    x = np.arange(len(problems), dtype=float)
    n_algorithms = max(1, len(algorithms))
    total_width = 0.82
    bar_width = total_width / n_algorithms
    start = -total_width / 2.0 + bar_width / 2.0

    for idx, algorithm in enumerate(algorithms):
        series = [values.get((problem, algorithm), 0.0) for problem in problems]
        positions = x + start + idx * bar_width
        bars = ax.bar(
            positions,
            series,
            width=bar_width,
            label=algorithm,
            color=cmap(idx % cmap.N),
            alpha=0.9,
        )
        labels = [_format_scaled_label(v, digits=2, exponent=DEFAULT_LABEL_EXPONENT) for v in series]
        ax.bar_label(bars, labels=labels, padding=3, fontsize=max(7, int(8 * ui_scale)))

    ax.set_title(title, fontsize=13 * ui_scale)
    ax.set_xlabel("Problem", fontsize=11 * ui_scale)
    ax.set_ylabel(y_label, fontsize=11 * ui_scale)
    ax.set_xticks(x)
    ax.set_xticklabels(problems, fontsize=10 * ui_scale)
    ax.tick_params(axis="y", labelsize=10 * ui_scale)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8 * ui_scale)
    ax.text(
        0.995,
        0.99,
        rf"Bar labels scaled by $\times 10^{{{DEFAULT_LABEL_EXPONENT}}}$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8 * ui_scale,
        color="dimgray",
    )

    fig.savefig(output, dpi=300)
    plt.close(fig)


def _plot_ablation_bar(
    algorithms: List[str],
    values: Dict[str, float],
    title: str,
    y_label: str,
    output_path: str,
    ui_scale: float,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10.5 * ui_scale, 5.0 * ui_scale), constrained_layout=True)
    cmap = plt.get_cmap("tab20")
    short_labels = [_short_algorithm_label(name) for name in algorithms]

    x = np.arange(len(algorithms), dtype=float)
    series = [values[algo] for algo in algorithms]
    bars = ax.bar(
        x,
        series,
        width=0.66,
        color=[cmap(i % cmap.N) for i in range(len(algorithms))],
        alpha=0.9,
    )
    labels = [_format_scaled_label(v, digits=2, exponent=DEFAULT_LABEL_EXPONENT) for v in series]
    ax.bar_label(bars, labels=labels, padding=3, fontsize=max(7, int(8 * ui_scale)))

    ax.set_title(title, fontsize=13 * ui_scale)
    ax.set_xlabel("Ablation Algorithm", fontsize=11 * ui_scale)
    ax.set_ylabel(y_label, fontsize=11 * ui_scale)
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=20, ha="right", fontsize=9 * ui_scale)
    ax.tick_params(axis="y", labelsize=10 * ui_scale)
    ax.grid(axis="y", alpha=0.25)
    ax.text(
        0.995,
        0.99,
        rf"Bar labels scaled by $\times 10^{{{DEFAULT_LABEL_EXPONENT}}}$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8 * ui_scale,
        color="dimgray",
    )

    fig.savefig(output, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate requested IGD/HV charts and tables only")
    parser.add_argument("results_csv", nargs="?", default=DEFAULT_RESULTS_CSV_PATH, help="Path to benchmark detailed CSV")
    parser.add_argument("--summary-csv", default=DEFAULT_SUMMARY_CSV_PATH, help="Path to benchmark summary CSV")
    parser.add_argument(
        "--ablation-summary-csv",
        default=DEFAULT_ABLATION_SUMMARY_CSV_PATH,
        help="Path to ablation summary CSV",
    )
    parser.add_argument("--hv-best-output", default=DEFAULT_HV_BEST_OUTPUT_PATH)
    parser.add_argument("--igd-best-output", default=DEFAULT_IGD_BEST_OUTPUT_PATH)
    parser.add_argument("--hv-mean-output", default=DEFAULT_HV_MEAN_OUTPUT_PATH)
    parser.add_argument("--igd-mean-output", default=DEFAULT_IGD_MEAN_OUTPUT_PATH)
    parser.add_argument("--hv-ablation-output", default=DEFAULT_HV_ABLATION_OUTPUT_PATH)
    parser.add_argument("--igd-ablation-output", default=DEFAULT_IGD_ABLATION_OUTPUT_PATH)
    parser.add_argument("--best-table-output", default=DEFAULT_BEST_TABLE_PATH)
    parser.add_argument("--mean-table-output", default=DEFAULT_MEAN_TABLE_PATH)
    parser.add_argument("--ablation-table-output", default=DEFAULT_ABLATION_TABLE_PATH)
    parser.add_argument("--best-table-md-output", default=DEFAULT_BEST_TABLE_MD_PATH)
    parser.add_argument("--mean-table-md-output", default=DEFAULT_MEAN_TABLE_MD_PATH)
    parser.add_argument("--ablation-table-md-output", default=DEFAULT_ABLATION_TABLE_MD_PATH)
    parser.add_argument("--ablation-prefix", default=DEFAULT_ABLATION_PREFIX)
    parser.add_argument("--full-algorithm", default=DEFAULT_FULL_ALGORITHM)
    parser.add_argument("--ui-scale", type=float, default=1.0)
    args = parser.parse_args()

    results_path = Path(args.results_csv)
    summary_path = Path(args.summary_csv)
    ablation_summary_path = Path(args.ablation_summary_csv)

    if not results_path.exists():
        raise FileNotFoundError(f"Results CSV not found: {results_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary CSV not found: {summary_path}")

    result_rows = _load_results_csv(str(results_path))
    summary_rows = _load_summary_csv(str(summary_path))
    ablation_summary_rows: List[Dict[str, object]] = []
    has_ablation_summary = ablation_summary_path.exists()
    if has_ablation_summary:
        ablation_summary_rows = _load_summary_csv(str(ablation_summary_path))
    else:
        warnings.warn(
            f"Ablation summary CSV not found: {ablation_summary_path}. "
            "Ablation charts and ablation tables will be skipped.",
            stacklevel=1,
        )

    if not result_rows:
        raise ValueError("Detailed benchmark CSV is empty")
    if not summary_rows:
        raise ValueError("Summary benchmark CSV is empty")
    if has_ablation_summary and not ablation_summary_rows:
        raise ValueError("Ablation summary CSV is empty")

    best_table = _build_best_table(result_rows)
    mean_table = _build_mean_table(summary_rows)
    ablation_table: List[Dict[str, object]] = []
    if has_ablation_summary:
        ablation_table = _build_ablation_table(
            ablation_summary_rows,
            full_algorithm=args.full_algorithm,
            ablation_prefix=args.ablation_prefix,
        )
        if not ablation_table:
            raise ValueError("No ablation rows found. Check --ablation-prefix and --full-algorithm")

    _write_csv(args.best_table_output, best_table)
    _write_csv(args.mean_table_output, mean_table)
    if ablation_table:
        _write_csv(args.ablation_table_output, ablation_table)
    _write_markdown_table(args.best_table_md_output, best_table, "Best HV and IGD Table")
    _write_markdown_table(args.mean_table_md_output, mean_table, "Mean HV and IGD Table")
    if ablation_table:
        _write_markdown_table(args.ablation_table_md_output, ablation_table, "Ablation HV and IGD Table")

    best_problems, best_algorithms, best_hv_map = _build_grouped_metric_map(best_table, "hv_best")
    _, _, best_igd_map = _build_grouped_metric_map(best_table, "igd_best")

    mean_problems, mean_algorithms, mean_hv_map = _build_grouped_metric_map(mean_table, "hv_mean")
    _, _, mean_igd_map = _build_grouped_metric_map(mean_table, "igd_mean")

    ablation_algorithms_hv: List[str] = []
    ablation_hv_map: Dict[str, float] = {}
    ablation_algorithms_igd: List[str] = []
    ablation_igd_map: Dict[str, float] = {}
    if ablation_table:
        ablation_algorithms_hv, ablation_hv_map = _build_ablation_aggregate_map(ablation_table, "hv_mean")
        ablation_algorithms_igd, ablation_igd_map = _build_ablation_aggregate_map(ablation_table, "igd_mean")

    _plot_grouped_problem_bar(
        problems=best_problems,
        algorithms=best_algorithms,
        values=best_igd_map,
        title="Best IGD of comparison algorithms",
        y_label="Best IGD",
        output_path=args.igd_best_output,
        ui_scale=args.ui_scale,
    )
    _plot_grouped_problem_bar(
        problems=best_problems,
        algorithms=best_algorithms,
        values=best_hv_map,
        title="Best HV of comparison algorithms",
        y_label="Best HV",
        output_path=args.hv_best_output,
        ui_scale=args.ui_scale,
    )
    _plot_grouped_problem_bar(
        problems=mean_problems,
        algorithms=mean_algorithms,
        values=mean_igd_map,
        title="Mean IGD of comparison algorithms across repeated runs",
        y_label="Mean IGD",
        output_path=args.igd_mean_output,
        ui_scale=args.ui_scale,
    )
    _plot_grouped_problem_bar(
        problems=mean_problems,
        algorithms=mean_algorithms,
        values=mean_hv_map,
        title="Mean HV of comparison algorithms across repeated runs",
        y_label="Mean HV",
        output_path=args.hv_mean_output,
        ui_scale=args.ui_scale,
    )
    if ablation_table:
        _plot_ablation_bar(
            algorithms=ablation_algorithms_igd,
            values=ablation_igd_map,
            title="Ablation IGD (average across benchmark problems)",
            y_label="Mean IGD",
            output_path=args.igd_ablation_output,
            ui_scale=args.ui_scale,
        )
        _plot_ablation_bar(
            algorithms=ablation_algorithms_hv,
            values=ablation_hv_map,
            title="Ablation HV (average across benchmark problems)",
            y_label="Mean HV",
            output_path=args.hv_ablation_output,
            ui_scale=args.ui_scale,
        )

    print(f"Saved best table: {args.best_table_output}")
    print(f"Saved mean table: {args.mean_table_output}")
    if ablation_table:
        print(f"Saved ablation table: {args.ablation_table_output}")
    print(f"Saved best markdown table: {args.best_table_md_output}")
    print(f"Saved mean markdown table: {args.mean_table_md_output}")
    if ablation_table:
        print(f"Saved ablation markdown table: {args.ablation_table_md_output}")
    print(f"Saved best IGD chart: {args.igd_best_output}")
    print(f"Saved best HV chart: {args.hv_best_output}")
    print(f"Saved mean IGD chart: {args.igd_mean_output}")
    print(f"Saved mean HV chart: {args.hv_mean_output}")
    if ablation_table:
        print(f"Saved ablation IGD chart: {args.igd_ablation_output}")
        print(f"Saved ablation HV chart: {args.hv_ablation_output}")


if __name__ == "__main__":
    main()
