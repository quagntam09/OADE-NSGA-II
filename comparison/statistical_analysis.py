"""
Run statistical analysis over benchmark result CSV.

Outputs:
- friedman_results.csv
- average_ranks.csv
- wilcoxon_holm_vs_reference.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    from scipy.stats import friedmanchisquare, rankdata, wilcoxon
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "scipy is required for statistical analysis. Install with: pip install scipy"
    ) from exc


DEFAULT_RESULTS_CSV = "outputs/benchmark_results.csv"
DEFAULT_OUTPUT_DIR = "outputs/statistics"
DEFAULT_REFERENCE_ALGORITHM = "OADE_NSGA2"


def load_results(csv_path: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            rows.append(
                {
                    "problem": row["problem"],
                    "algorithm": row["algorithm"],
                    "seed": int(row["seed"]),
                    "hv": float(row["hv"]),
                    "igd": float(row["igd"]),
                    "igd_plus": float(row.get("igd_plus", row["igd"])),
                    "epsilon_additive": float(row.get("epsilon_additive", row["igd"])),
                    "spacing": float(row.get("spacing", 0.0)),
                }
            )
    return rows


def _to_problem_algorithm_mean(
    rows: List[Dict[str, object]], metric: str
) -> Dict[str, Dict[str, float]]:
    grouped: Dict[Tuple[str, str], List[float]] = {}
    for row in rows:
        key = (str(row["problem"]), str(row["algorithm"]))
        grouped.setdefault(key, []).append(float(row[metric]))

    table: Dict[str, Dict[str, float]] = {}
    for (problem, algorithm), values in grouped.items():
        table.setdefault(problem, {})[algorithm] = float(np.mean(values))

    return table


def _common_algorithms(table: Dict[str, Dict[str, float]]) -> List[str]:
    algorithms_per_problem = [set(algo_values.keys()) for algo_values in table.values() if algo_values]
    if not algorithms_per_problem:
        return []
    return sorted(set.intersection(*algorithms_per_problem))


def _build_matrix(
    table: Dict[str, Dict[str, float]], algorithms: Sequence[str]
) -> Tuple[List[str], np.ndarray]:
    problems = sorted(table.keys())
    matrix = np.array(
        [[table[problem][algorithm] for algorithm in algorithms] for problem in problems],
        dtype=float,
    )
    return problems, matrix


def _average_ranks(matrix: np.ndarray, lower_is_better: bool) -> np.ndarray:
    # rankdata gives rank 1 to smallest value, so invert for higher-is-better metrics.
    prepared = matrix if lower_is_better else -matrix
    ranks = np.array([rankdata(row, method="average") for row in prepared], dtype=float)
    return np.mean(ranks, axis=0)


def _holm_adjusted(p_values: List[float], alpha: float = 0.05) -> Tuple[List[float], List[bool]]:
    if not p_values:
        return [], []

    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    m = len(p_values)
    adjusted = [0.0] * m
    reject = [False] * m

    prev_adj = 0.0
    for rank, (orig_idx, p_val) in enumerate(indexed, start=1):
        adj = min(1.0, (m - rank + 1) * p_val)
        adj = max(adj, prev_adj)
        prev_adj = adj
        adjusted[orig_idx] = adj

    for orig_idx, p_adj in enumerate(adjusted):
        reject[orig_idx] = p_adj <= alpha

    return adjusted, reject


def analyze_statistics(
    rows: List[Dict[str, object]],
    output_dir: str,
    reference_algorithm: str,
) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_configs = [
        ("hv", False),
        ("igd", True),
    ]

    friedman_rows: List[Dict[str, object]] = []
    rank_rows: List[Dict[str, object]] = []
    wilcoxon_rows: List[Dict[str, object]] = []

    for metric, lower_is_better in metric_configs:
        table = _to_problem_algorithm_mean(rows, metric)
        algorithms = _common_algorithms(table)
        if len(algorithms) < 2:
            continue

        problems, matrix = _build_matrix(table, algorithms)
        avg_ranks = _average_ranks(matrix, lower_is_better=lower_is_better)

        for algo, rank in zip(algorithms, avg_ranks):
            rank_rows.append(
                {
                    "metric": metric,
                    "algorithm": algo,
                    "average_rank": float(rank),
                    "n_problems": len(problems),
                }
            )

        if len(algorithms) >= 3 and len(problems) >= 2:
            samples = [matrix[:, idx] for idx in range(matrix.shape[1])]
            friedman_stat, friedman_p = friedmanchisquare(*samples)
            friedman_rows.append(
                {
                    "metric": metric,
                    "n_algorithms": len(algorithms),
                    "n_problems": len(problems),
                    "friedman_statistic": float(friedman_stat),
                    "p_value": float(friedman_p),
                }
            )

        if reference_algorithm in algorithms and len(problems) >= 2:
            ref_idx = algorithms.index(reference_algorithm)
            ref_values = matrix[:, ref_idx]

            pairwise_p_values: List[float] = []
            pairwise_meta: List[Tuple[str, np.ndarray]] = []
            for algo_idx, algo in enumerate(algorithms):
                if algo == reference_algorithm:
                    continue
                cmp_values = matrix[:, algo_idx]

                if np.allclose(ref_values, cmp_values):
                    p_val = 1.0
                else:
                    _, p_val = wilcoxon(ref_values, cmp_values, zero_method="wilcox", alternative="two-sided")

                pairwise_p_values.append(float(p_val))
                pairwise_meta.append((algo, cmp_values))

            adjusted, reject = _holm_adjusted(pairwise_p_values, alpha=0.05)

            for idx, (algo, cmp_values) in enumerate(pairwise_meta):
                if lower_is_better:
                    win = int(np.sum(ref_values < cmp_values))
                    tie = int(np.sum(np.isclose(ref_values, cmp_values)))
                    loss = int(np.sum(ref_values > cmp_values))
                else:
                    win = int(np.sum(ref_values > cmp_values))
                    tie = int(np.sum(np.isclose(ref_values, cmp_values)))
                    loss = int(np.sum(ref_values < cmp_values))

                wilcoxon_rows.append(
                    {
                        "metric": metric,
                        "reference": reference_algorithm,
                        "comparator": algo,
                        "n_problems": len(problems),
                        "reference_mean": float(np.mean(ref_values)),
                        "comparator_mean": float(np.mean(cmp_values)),
                        "p_value_raw": pairwise_p_values[idx],
                        "p_value_holm": adjusted[idx],
                        "reject_h0_0_05": reject[idx],
                        "win": win,
                        "tie": tie,
                        "loss": loss,
                    }
                )

    _write_csv(out_dir / "friedman_results.csv", friedman_rows)
    _write_csv(out_dir / "average_ranks.csv", rank_rows)
    _write_csv(out_dir / "wilcoxon_holm_vs_reference.csv", wilcoxon_rows)


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Statistical analysis for benchmark results")
    parser.add_argument("results_csv", nargs="?", default=DEFAULT_RESULTS_CSV, help="Path to benchmark_results.csv")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory for statistical CSV files")
    parser.add_argument(
        "--reference-algorithm",
        default=DEFAULT_REFERENCE_ALGORITHM,
        help="Reference algorithm for pairwise Wilcoxon-Holm",
    )
    args = parser.parse_args()

    rows = load_results(args.results_csv)
    if not rows:
        raise ValueError("Input benchmark CSV is empty")

    analyze_statistics(
        rows,
        args.output_dir,
        args.reference_algorithm,
    )
    print(f"Saved statistical outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()


