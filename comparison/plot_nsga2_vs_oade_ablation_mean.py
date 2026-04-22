"""
Plot HV/IGD comparison figures for NSGA-II vs OADE-NSGA2 and ablations.

Supports:
- Mean mode: uses benchmark summary CSV (hv_mean, igd_mean).
- Best mode: uses benchmark results CSV and computes per (problem, algorithm):
  hv_best = max(hv), igd_best = min(igd).

Creates (for selected mode):
1) NSGA-II vs full OADE-NSGA2.
2) One pair of figures per ablation mechanism (HV + IGD), each comparing:
   NSGA-II vs full OADE-NSGA2 vs OADE-NSGA2 with one mechanism removed.

Example:
    .\\.venv\\Scripts\\python.exe comparison\\plot_nsga2_vs_oade_ablation_mean.py --plot-mode both
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_SUMMARY_CSV = "outputs/benchmark_summary_ablation.csv"
DEFAULT_RESULTS_CSV = "outputs/benchmark_results_ablation.csv"
DEFAULT_OUTPUT_DIR = "outputs/ablation_oade"
DEFAULT_NS_GA2 = "pymoo_nsga2"
DEFAULT_OADE_FULL = "OADE_NSGA2"
DEFAULT_ABLATION_PREFIX = "OADE_NSGA2_ablation_"


def _normalize_algorithm_name(name: str) -> str:
    if name.startswith("improved_nsga2"):
        return name.replace("improved_nsga2", "OADE_NSGA2", 1)
    return name


def _read_summary(path: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with open(path, "r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            rows.append(
                {
                    "problem": row["problem"],
                    "algorithm": _normalize_algorithm_name(row["algorithm"]),
                    "hv_mean": float(row["hv_mean"]),
                    "igd_mean": float(row["igd_mean"]),
                }
            )
    return rows


def _read_results(path: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with open(path, "r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            rows.append(
                {
                    "problem": row["problem"],
                    "algorithm": _normalize_algorithm_name(row["algorithm"]),
                    "hv": float(row["hv"]),
                    "igd": float(row["igd"]),
                }
            )
    return rows


def _build_best_metric_maps_from_results(
    rows: List[Dict[str, object]],
) -> Tuple[List[str], List[str], Dict[tuple[str, str], float], Dict[tuple[str, str], float]]:
    problems = sorted({str(r["problem"]) for r in rows})
    algorithms = sorted({str(r["algorithm"]) for r in rows})

    hv_best_map: Dict[tuple[str, str], float] = {}
    igd_best_map: Dict[tuple[str, str], float] = {}

    for row in rows:
        key = (str(row["problem"]), str(row["algorithm"]))
        hv_value = float(row["hv"])
        igd_value = float(row["igd"])

        if key not in hv_best_map or hv_value > hv_best_map[key]:
            hv_best_map[key] = hv_value
        if key not in igd_best_map or igd_value < igd_best_map[key]:
            igd_best_map[key] = igd_value

    return problems, algorithms, hv_best_map, igd_best_map


def _build_metric_map(rows: List[Dict[str, object]], metric: str) -> Dict[tuple[str, str], float]:
    metric_map: Dict[tuple[str, str], float] = {}
    for row in rows:
        metric_map[(str(row["problem"]), str(row["algorithm"]))] = float(row[metric])
    return metric_map


def _merge_rows_without_algorithm_overlap(
    primary_rows: List[Dict[str, object]],
    reference_rows: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    # Reference data is used to add missing algorithms (for example full OADE)
    # without overriding or inflating algorithms already present in primary data.
    primary_algorithms = {str(row["algorithm"]) for row in primary_rows}
    merged = list(primary_rows)
    merged.extend(
        row for row in reference_rows if str(row["algorithm"]) not in primary_algorithms
    )
    return merged


def _algorithm_label(name: str) -> str:
    if name == "pymoo_nsga2":
        return "NSGA-II"
    if name == "OADE_NSGA2":
        return "OADE-NSGA2"

    if name == "OADE_NSGA2_incremental_baseline":
        return "NSGA-II baseline"

    if name.startswith("OADE_NSGA2_incremental_plus_"):
        mechanism = name.replace("OADE_NSGA2_incremental_plus_", "")
        return f"NSGA-II + {mechanism.replace('_', ' ')}"

    if name.startswith("OADE_NSGA2_ablation_no_"):
        mechanism = name.replace("OADE_NSGA2_ablation_no_", "")
        return f"OADE no {mechanism.replace('_', ' ')}"

    if name.startswith("OADE_NSGA2_ablation_"):
        tail = name.replace("OADE_NSGA2_ablation_", "")
        return f"OADE {tail.replace('_', ' ')}"

    return name.replace("_", " ")


def _safe_filename(text: str) -> str:
    keep = []
    for ch in text.lower():
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        elif ch in (" ", "/"):
            keep.append("_")
    cleaned = "".join(keep).strip("_")
    return cleaned or "plot"


def _mechanism_file_prefix(ablation_prefix: str) -> str:
    return "incremental" if "incremental" in ablation_prefix.lower() else "ablation"


def _plot_grouped_bars(
    problems: List[str],
    algorithms: List[str],
    metric_map: Dict[tuple[str, str], float],
    metric_name: str,
    title: str,
    output_path: Path,
    y_label_suffix: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(problems), dtype=float)
    n = len(algorithms)
    total_width = 0.8
    bar_width = total_width / max(1, n)
    offset_start = -total_width / 2 + bar_width / 2

    fig, ax = plt.subplots(figsize=(11, 5.2), constrained_layout=True)
    cmap = plt.get_cmap("Set2")

    for idx, algo in enumerate(algorithms):
        values = [metric_map.get((problem, algo), np.nan) for problem in problems]
        positions = x + offset_start + idx * bar_width
        bars = ax.bar(
            positions,
            values,
            width=bar_width,
            color=cmap(idx % cmap.N),
            alpha=0.95,
            label=_algorithm_label(algo),
        )

        # Keep labels compact to reduce overlap on small values.
        labels = [f"{v:.4f}" if np.isfinite(v) else "NA" for v in values]
        ax.bar_label(bars, labels=labels, padding=2, fontsize=8)

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Problem")
    ax.set_ylabel(f"{metric_name} {y_label_suffix}")
    ax.set_xticks(x)
    ax.set_xticklabels(problems)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=9)

    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _collect_metric_rows(
    problems: List[str],
    algorithms: List[str],
    hv_map: Dict[tuple[str, str], float],
    igd_map: Dict[tuple[str, str], float],
    hv_key: str,
    igd_key: str,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for problem in problems:
        for algorithm in algorithms:
            hv_value = hv_map.get((problem, algorithm), np.nan)
            igd_value = igd_map.get((problem, algorithm), np.nan)
            if not (np.isfinite(hv_value) and np.isfinite(igd_value)):
                continue

            rows.append(
                {
                    "problem": problem,
                    "algorithm": algorithm,
                    hv_key: float(hv_value),
                    igd_key: float(igd_value),
                }
            )
    return rows


def _rows_to_markdown_table(rows: List[Dict[str, object]], headers: List[str], title: str) -> str:
    lines: List[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for row in rows:
        values: List[str] = []
        for header in headers:
            value = row[header]
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")

    lines.append("")
    return "\n".join(lines)


def _write_markdown_table(path: Path, rows: List[Dict[str, object]], headers: List[str], title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    markdown_text = _rows_to_markdown_table(rows=rows, headers=headers, title=title)
    path.write_text(markdown_text, encoding="utf-8")


def create_plots(
    summary_csv: str,
    reference_summary_csv: str | None,
    output_dir: str,
    nsga2_name: str,
    oade_full_name: str,
    ablation_prefix: str,
    ablation_vs_full_only: bool,
) -> None:
    rows = _read_summary(summary_csv)
    if reference_summary_csv:
        rows = _merge_rows_without_algorithm_overlap(rows, _read_summary(reference_summary_csv))
    if not rows:
        raise ValueError("Input summary CSV is empty")

    problems = sorted({str(r["problem"]) for r in rows})
    algorithms = sorted({str(r["algorithm"]) for r in rows})

    if nsga2_name not in algorithms:
        raise ValueError(f"Algorithm not found in CSV: {nsga2_name}")
    if oade_full_name not in algorithms:
        raise ValueError(f"Algorithm not found in CSV: {oade_full_name}")

    ablations = sorted([a for a in algorithms if a.startswith(ablation_prefix) and a != oade_full_name])
    if not ablations:
        raise ValueError(f"No ablation algorithms found with prefix: {ablation_prefix}")

    hv_map = _build_metric_map(rows, "hv_mean")
    igd_map = _build_metric_map(rows, "igd_mean")

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    mechanism_prefix = _mechanism_file_prefix(ablation_prefix)

    # 1) NSGA-II vs full OADE
    pair_algorithms = [nsga2_name, oade_full_name]
    _plot_grouped_bars(
        problems=problems,
        algorithms=pair_algorithms,
        metric_map=hv_map,
        metric_name="HV",
        title="Mean HV: NSGA-II vs OADE-NSGA2",
        output_path=out_root / "mean_hv_nsga2_vs_oade.png",
        y_label_suffix="mean",
    )
    _plot_grouped_bars(
        problems=problems,
        algorithms=pair_algorithms,
        metric_map=igd_map,
        metric_name="IGD",
        title="Mean IGD: NSGA-II vs OADE-NSGA2",
        output_path=out_root / "mean_igd_nsga2_vs_oade.png",
        y_label_suffix="mean",
    )

    # 2) One pair per ablation mechanism
    for ablation_algo in ablations:
        compare_algorithms = [ablation_algo, oade_full_name] if ablation_vs_full_only else [nsga2_name, oade_full_name, ablation_algo]
        mechanism_name = ablation_algo.replace(ablation_prefix, "")
        mechanism_slug = _safe_filename(mechanism_name)
        title_suffix = f"{_algorithm_label(ablation_algo)} vs OADE-NSGA2" if ablation_vs_full_only else f"NSGA-II vs OADE-NSGA2 vs { _algorithm_label(ablation_algo) }"

        _plot_grouped_bars(
            problems=problems,
            algorithms=compare_algorithms,
            metric_map=hv_map,
            metric_name="HV",
            title=f"Mean HV: {title_suffix}",
            output_path=out_root / f"mean_hv_{mechanism_prefix}_{mechanism_slug}.png",
            y_label_suffix="mean",
        )

        _plot_grouped_bars(
            problems=problems,
            algorithms=compare_algorithms,
            metric_map=igd_map,
            metric_name="IGD",
            title=f"Mean IGD: {title_suffix}",
            output_path=out_root / f"mean_igd_{mechanism_prefix}_{mechanism_slug}.png",
            y_label_suffix="mean",
        )

    selected_algorithms = [nsga2_name, oade_full_name] + ablations
    mean_rows = _collect_metric_rows(
        problems=problems,
        algorithms=selected_algorithms,
        hv_map=hv_map,
        igd_map=igd_map,
        hv_key="hv_mean",
        igd_key="igd_mean",
    )
    mean_md_path = out_root / "table_mean_hv_igd_compare.md"
    _write_markdown_table(
        path=mean_md_path,
        rows=mean_rows,
        headers=["problem", "algorithm", "hv_mean", "igd_mean"],
        title="Mean HV and IGD Comparison Table",
    )

    print(f"Saved plots to: {out_root}")
    print("Base comparison files:")
    print(f"- {out_root / 'mean_hv_nsga2_vs_oade.png'}")
    print(f"- {out_root / 'mean_igd_nsga2_vs_oade.png'}")
    print("Mechanism comparison files (HV + IGD per mechanism) created.")
    print(f"Mean markdown table: {mean_md_path}")


def create_best_plots(
    results_csv: str,
    reference_results_csv: str | None,
    output_dir: str,
    nsga2_name: str,
    oade_full_name: str,
    ablation_prefix: str,
    ablation_vs_full_only: bool,
) -> None:
    rows = _read_results(results_csv)
    if reference_results_csv:
        rows = _merge_rows_without_algorithm_overlap(rows, _read_results(reference_results_csv))
    if not rows:
        raise ValueError("Input results CSV is empty")

    problems, algorithms, hv_map, igd_map = _build_best_metric_maps_from_results(rows)

    if nsga2_name not in algorithms:
        raise ValueError(f"Algorithm not found in CSV: {nsga2_name}")
    if oade_full_name not in algorithms:
        raise ValueError(f"Algorithm not found in CSV: {oade_full_name}")

    ablations = sorted([a for a in algorithms if a.startswith(ablation_prefix) and a != oade_full_name])
    if not ablations:
        raise ValueError(f"No ablation algorithms found with prefix: {ablation_prefix}")

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    mechanism_prefix = _mechanism_file_prefix(ablation_prefix)

    pair_algorithms = [nsga2_name, oade_full_name]
    _plot_grouped_bars(
        problems=problems,
        algorithms=pair_algorithms,
        metric_map=hv_map,
        metric_name="HV",
        title="Best HV: NSGA-II vs OADE-NSGA2",
        output_path=out_root / "best_hv_nsga2_vs_oade.png",
        y_label_suffix="best",
    )
    _plot_grouped_bars(
        problems=problems,
        algorithms=pair_algorithms,
        metric_map=igd_map,
        metric_name="IGD",
        title="Best IGD: NSGA-II vs OADE-NSGA2",
        output_path=out_root / "best_igd_nsga2_vs_oade.png",
        y_label_suffix="best",
    )

    for ablation_algo in ablations:
        compare_algorithms = [ablation_algo, oade_full_name] if ablation_vs_full_only else [nsga2_name, oade_full_name, ablation_algo]
        mechanism_name = ablation_algo.replace(ablation_prefix, "")
        mechanism_slug = _safe_filename(mechanism_name)
        title_suffix = f"{_algorithm_label(ablation_algo)} vs OADE-NSGA2" if ablation_vs_full_only else f"NSGA-II vs OADE-NSGA2 vs { _algorithm_label(ablation_algo) }"

        _plot_grouped_bars(
            problems=problems,
            algorithms=compare_algorithms,
            metric_map=hv_map,
            metric_name="HV",
            title=f"Best HV: {title_suffix}",
            output_path=out_root / f"best_hv_{mechanism_prefix}_{mechanism_slug}.png",
            y_label_suffix="best",
        )

        _plot_grouped_bars(
            problems=problems,
            algorithms=compare_algorithms,
            metric_map=igd_map,
            metric_name="IGD",
            title=f"Best IGD: {title_suffix}",
            output_path=out_root / f"best_igd_{mechanism_prefix}_{mechanism_slug}.png",
            y_label_suffix="best",
        )

    selected_algorithms = [nsga2_name, oade_full_name] + ablations
    best_rows = _collect_metric_rows(
        problems=problems,
        algorithms=selected_algorithms,
        hv_map=hv_map,
        igd_map=igd_map,
        hv_key="hv_best",
        igd_key="igd_best",
    )
    best_md_path = out_root / "table_best_hv_igd_compare.md"
    _write_markdown_table(
        path=best_md_path,
        rows=best_rows,
        headers=["problem", "algorithm", "hv_best", "igd_best"],
        title="Best HV and IGD Comparison Table",
    )

    print(f"Saved plots to: {out_root}")
    print("Base comparison files:")
    print(f"- {out_root / 'best_hv_nsga2_vs_oade.png'}")
    print(f"- {out_root / 'best_igd_nsga2_vs_oade.png'}")
    print("Mechanism comparison files (HV + IGD per mechanism) created.")
    print(f"Best markdown table: {best_md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot mean/best HV-IGD comparison for NSGA-II vs OADE-NSGA2 and one-by-one ablation variants"
        )
    )
    parser.add_argument("--summary-csv", default=DEFAULT_SUMMARY_CSV, help="Input benchmark summary CSV")
    parser.add_argument("--results-csv", default=DEFAULT_RESULTS_CSV, help="Input benchmark results CSV")
    parser.add_argument("--reference-summary-csv", default=None, help="Optional extra summary CSV to merge")
    parser.add_argument("--reference-results-csv", default=None, help="Optional extra results CSV to merge")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory for PNG files")
    parser.add_argument("--nsga2-name", default=DEFAULT_NS_GA2, help="Algorithm name for NSGA-II in CSV")
    parser.add_argument("--oade-full-name", default=DEFAULT_OADE_FULL, help="Algorithm name for full OADE in CSV")
    parser.add_argument(
        "--plot-mode",
        choices=("mean", "best", "both"),
        default="mean",
        help="Choose which charts to generate",
    )
    parser.add_argument(
        "--ablation-prefix",
        default=DEFAULT_ABLATION_PREFIX,
        help="Prefix to identify ablation algorithms in CSV",
    )
    parser.add_argument(
        "--ablation-vs-full-only",
        action="store_true",
        help="For each mechanism figure, compare only (mechanism variant vs full OADE)",
    )
    args = parser.parse_args()

    if args.plot_mode in ("mean", "both"):
        summary_path = Path(args.summary_csv)
        if not summary_path.exists():
            raise FileNotFoundError(f"Summary CSV not found: {summary_path}")

        create_plots(
            summary_csv=str(summary_path),
            reference_summary_csv=args.reference_summary_csv,
            output_dir=args.output_dir,
            nsga2_name=args.nsga2_name,
            oade_full_name=args.oade_full_name,
            ablation_prefix=args.ablation_prefix,
            ablation_vs_full_only=args.ablation_vs_full_only,
        )

    if args.plot_mode in ("best", "both"):
        results_path = Path(args.results_csv)
        if not results_path.exists():
            raise FileNotFoundError(f"Results CSV not found: {results_path}")

        create_best_plots(
            results_csv=str(results_path),
            reference_results_csv=args.reference_results_csv,
            output_dir=args.output_dir,
            nsga2_name=args.nsga2_name,
            oade_full_name=args.oade_full_name,
            ablation_prefix=args.ablation_prefix,
            ablation_vs_full_only=args.ablation_vs_full_only,
        )


if __name__ == "__main__":
    main()
