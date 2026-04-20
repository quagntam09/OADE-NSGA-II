"""
Create an incremental chain report for mechanism-by-mechanism analysis.

Default chain:
baseline -> +OBL init -> +DE fixed -> +DE adaptive -> +periodic OBL -> +restart

Inputs:
- benchmark_summary_incremental.csv

Outputs:
- incremental_chain_detail.csv
- incremental_chain_overall.csv
- incremental_chain.md
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


DEFAULT_SUMMARY_CSV = "outputs/benchmark_summary_incremental.csv"
DEFAULT_OUTPUT_DIR = "outputs"
DEFAULT_OUTPUT_PREFIX = "incremental_chain"
DEFAULT_CHAIN: Sequence[str] = (
    "improved_nsga2_incremental_baseline",
    "improved_nsga2_incremental_plus_obl_init",
    "improved_nsga2_incremental_plus_de_fixed",
    "improved_nsga2_incremental_plus_de_adaptive",
    "improved_nsga2_incremental_plus_periodic_obl",
    "improved_nsga2_incremental_plus_restart",
)


def _safe_pct(delta: float, base: float) -> float:
    if np.isclose(base, 0.0):
        return float("nan")
    return float(100.0 * delta / base)


def _nanmean_or_nan(values: np.ndarray) -> float:
    valid = values[~np.isnan(values)]
    if valid.size == 0:
        return float("nan")
    return float(np.mean(valid))


def _parse_chain(chain_csv: str) -> List[str]:
    chain = [item.strip() for item in chain_csv.split(",") if item.strip()]
    if len(chain) < 2:
        raise ValueError("Chain must contain at least two algorithms")
    return chain


def _load_summary_rows(path: str) -> List[Dict[str, object]]:
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
                    "runtime_mean": float(row.get("runtime_mean", "nan")),
                }
            )
    return rows


def _build_lookup(rows: List[Dict[str, object]]) -> Dict[tuple[str, str], Dict[str, object]]:
    return {(str(r["problem"]), str(r["algorithm"])): r for r in rows}


def build_chain_tables(rows: List[Dict[str, object]], chain: Sequence[str]) -> tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    problems = sorted({str(r["problem"]) for r in rows})
    lookup = _build_lookup(rows)

    detail_rows: List[Dict[str, object]] = []

    for problem in problems:
        if not all((problem, algo) in lookup for algo in chain):
            continue

        baseline_row = lookup[(problem, chain[0])]
        baseline_hv = float(baseline_row["hv_mean"])
        baseline_igd = float(baseline_row["igd_mean"])
        baseline_runtime = float(baseline_row["runtime_mean"])

        prev_hv = baseline_hv
        prev_igd = baseline_igd
        prev_runtime = baseline_runtime

        for step_idx, algo in enumerate(chain):
            current = lookup[(problem, algo)]
            hv = float(current["hv_mean"])
            igd = float(current["igd_mean"])
            runtime = float(current["runtime_mean"])

            if step_idx == 0:
                step_delta_hv = 0.0
                step_delta_igd = 0.0
                step_delta_runtime = 0.0
            else:
                step_delta_hv = hv - prev_hv
                step_delta_igd = igd - prev_igd
                step_delta_runtime = runtime - prev_runtime

            cum_delta_hv = hv - baseline_hv
            cum_delta_igd = igd - baseline_igd
            cum_delta_runtime = runtime - baseline_runtime

            detail_rows.append(
                {
                    "problem": problem,
                    "step_index": step_idx,
                    "algorithm": algo,
                    "hv_mean": hv,
                    "igd_mean": igd,
                    "runtime_mean": runtime,
                    "step_delta_hv_abs": step_delta_hv,
                    "step_delta_hv_pct": _safe_pct(step_delta_hv, prev_hv),
                    "step_hv_improvement_abs": step_delta_hv,
                    "step_hv_improvement_pct": _safe_pct(step_delta_hv, prev_hv),
                    "step_delta_igd_abs": step_delta_igd,
                    "step_delta_igd_pct": _safe_pct(step_delta_igd, prev_igd),
                    "step_igd_improvement_abs": -step_delta_igd,
                    "step_igd_improvement_pct": _safe_pct(-step_delta_igd, prev_igd),
                    "step_delta_runtime_abs": step_delta_runtime,
                    "step_delta_runtime_pct": _safe_pct(step_delta_runtime, prev_runtime),
                    "cum_delta_hv_abs": cum_delta_hv,
                    "cum_delta_hv_pct": _safe_pct(cum_delta_hv, baseline_hv),
                    "cum_hv_improvement_abs": cum_delta_hv,
                    "cum_hv_improvement_pct": _safe_pct(cum_delta_hv, baseline_hv),
                    "cum_delta_igd_abs": cum_delta_igd,
                    "cum_delta_igd_pct": _safe_pct(cum_delta_igd, baseline_igd),
                    "cum_igd_improvement_abs": -cum_delta_igd,
                    "cum_igd_improvement_pct": _safe_pct(-cum_delta_igd, baseline_igd),
                    "cum_delta_runtime_abs": cum_delta_runtime,
                    "cum_delta_runtime_pct": _safe_pct(cum_delta_runtime, baseline_runtime),
                }
            )

            prev_hv = hv
            prev_igd = igd
            prev_runtime = runtime

    if not detail_rows:
        raise ValueError("No complete problems found where all algorithms in chain are present")

    overall_rows: List[Dict[str, object]] = []
    for step_idx, algo in enumerate(chain):
        step_rows = [r for r in detail_rows if int(r["step_index"]) == step_idx and str(r["algorithm"]) == algo]
        if not step_rows:
            continue

        def arr(key: str) -> np.ndarray:
            return np.array([float(r[key]) for r in step_rows], dtype=float)

        overall_rows.append(
            {
                "step_index": step_idx,
                "algorithm": algo,
                "n_problems": len(step_rows),
                "step_hv_improvement_abs_mean": float(np.mean(arr("step_hv_improvement_abs"))),
                "step_hv_improvement_pct_mean": _nanmean_or_nan(arr("step_hv_improvement_pct")),
                "step_igd_improvement_abs_mean": float(np.mean(arr("step_igd_improvement_abs"))),
                "step_igd_improvement_pct_mean": _nanmean_or_nan(arr("step_igd_improvement_pct")),
                "cum_hv_improvement_abs_mean": float(np.mean(arr("cum_hv_improvement_abs"))),
                "cum_hv_improvement_pct_mean": _nanmean_or_nan(arr("cum_hv_improvement_pct")),
                "cum_igd_improvement_abs_mean": float(np.mean(arr("cum_igd_improvement_abs"))),
                "cum_igd_improvement_pct_mean": _nanmean_or_nan(arr("cum_igd_improvement_pct")),
                "cum_delta_runtime_abs_mean": float(np.mean(arr("cum_delta_runtime_abs"))),
                "cum_delta_runtime_pct_mean": _nanmean_or_nan(arr("cum_delta_runtime_pct")),
            }
        )

    return detail_rows, overall_rows


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _to_markdown_table(rows: List[Dict[str, object]]) -> str:
    if not rows:
        return "No rows"

    headers = [
        "step_index",
        "algorithm",
        "n_problems",
        "step_hv_improvement_abs_mean",
        "step_hv_improvement_pct_mean",
        "step_igd_improvement_abs_mean",
        "step_igd_improvement_pct_mean",
        "cum_hv_improvement_abs_mean",
        "cum_hv_improvement_pct_mean",
        "cum_igd_improvement_abs_mean",
        "cum_igd_improvement_pct_mean",
        "cum_delta_runtime_abs_mean",
        "cum_delta_runtime_pct_mean",
    ]

    lines: List[str] = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for row in rows:
        values: List[str] = []
        for h in headers:
            val = row[h]
            if isinstance(val, float):
                values.append(f"{val:.6f}")
            else:
                values.append(str(val))
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def create_incremental_chain_report(
    summary_csv: str,
    output_dir: str,
    output_prefix: str,
    chain: Sequence[str],
) -> None:
    rows = _load_summary_rows(summary_csv)
    detail_rows, overall_rows = build_chain_tables(rows, chain)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_prefix = output_prefix.strip() or DEFAULT_OUTPUT_PREFIX
    detail_path = out_dir / f"{safe_prefix}_detail.csv"
    overall_path = out_dir / f"{safe_prefix}_overall.csv"
    markdown_path = out_dir / f"{safe_prefix}.md"

    _write_csv(detail_path, detail_rows)
    _write_csv(overall_path, overall_rows)

    markdown_text = "# Incremental Chain Report\n\n"
    markdown_text += "## Chain\n\n"
    markdown_text += " -> ".join(chain)
    markdown_text += "\n\n## Overall (mean across problems)\n\n"
    markdown_text += _to_markdown_table(overall_rows)
    markdown_text += "\n"
    markdown_path.write_text(markdown_text, encoding="utf-8")

    print(f"Saved chain detail CSV: {detail_path}")
    print(f"Saved chain overall CSV: {overall_path}")
    print(f"Saved chain markdown: {markdown_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create incremental chain tables from benchmark summary CSV")
    parser.add_argument("summary_csv", nargs="?", default=DEFAULT_SUMMARY_CSV, help="Path to benchmark summary CSV")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX, help="Output file prefix")
    parser.add_argument(
        "--chain",
        default=",".join(DEFAULT_CHAIN),
        help="Comma-separated chain order",
    )
    args = parser.parse_args()

    chain = _parse_chain(args.chain)
    create_incremental_chain_report(
        summary_csv=args.summary_csv,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        chain=chain,
    )


if __name__ == "__main__":
    main()
