"""Plot Wilcoxon rank-sum benchmark results."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RESULT_COLORS = {
    "comparison_better": "#d95f02",
    "no_significant_difference": "#bdbdbd",
    "reference_better": "#1b9e77",
}


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def ordered_unique(values) -> list[str]:
    out = []
    for value in values:
        if value not in out:
            out.append(value)
    return out


def lookup(rows: list[dict], problem: str, metric: str, comparison: str) -> dict | None:
    for row in rows:
        if (
            row["problem"] == problem
            and row["metric"] == metric
            and row["comparison_algorithm"] == comparison
        ):
            return row
    return None


def plot_pvalue_heatmap(rows: list[dict], out_dir: Path) -> None:
    problems = ordered_unique(row["problem"] for row in rows)
    comparisons = ordered_unique(row["comparison_algorithm"] for row in rows)

    for metric in ordered_unique(row["metric"] for row in rows):
        matrix = np.full((len(problems), len(comparisons)), np.nan)
        labels = [["" for _ in comparisons] for _ in problems]

        for i, problem in enumerate(problems):
            for j, comparison in enumerate(comparisons):
                row = lookup(rows, problem, metric, comparison)
                if row is None:
                    continue
                p_value = float(row["p_value"])
                matrix[i, j] = -math.log10(max(p_value, 1e-300))
                labels[i][j] = f"{p_value:.2g}"

        fig, ax = plt.subplots(figsize=(1.8 * len(comparisons) + 4, 0.8 * len(problems) + 3))
        im = ax.imshow(matrix, cmap="viridis", aspect="auto")
        ax.set_title(f"Wilcoxon rank-sum p-values ({metric})")
        ax.set_xlabel("comparison algorithm")
        ax.set_ylabel("problem")
        ax.set_xticks(range(len(comparisons)))
        ax.set_xticklabels(comparisons)
        ax.set_yticks(range(len(problems)))
        ax.set_yticklabels([p.upper() for p in problems])

        for i in range(len(problems)):
            for j in range(len(comparisons)):
                ax.text(j, i, labels[i][j], ha="center", va="center", color="white", fontsize=8)

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("-log10(p-value)")
        fig.tight_layout()
        fig.savefig(out_dir / f"wilcoxon_{metric.lower()}_pvalue_heatmap.png", dpi=220)
        plt.close(fig)


def plot_result_heatmap(rows: list[dict], out_dir: Path) -> None:
    problems = ordered_unique(row["problem"] for row in rows)
    comparisons = ordered_unique(row["comparison_algorithm"] for row in rows)
    result_order = ["comparison_better", "no_significant_difference", "reference_better"]
    result_to_num = {name: idx for idx, name in enumerate(result_order)}

    for metric in ordered_unique(row["metric"] for row in rows):
        matrix = np.full((len(problems), len(comparisons)), 1)
        labels = [["" for _ in comparisons] for _ in problems]

        for i, problem in enumerate(problems):
            for j, comparison in enumerate(comparisons):
                row = lookup(rows, problem, metric, comparison)
                if row is None:
                    continue
                result = row["result"]
                matrix[i, j] = result_to_num[result]
                labels[i][j] = {
                    "reference_better": "OADE",
                    "comparison_better": comparison,
                    "no_significant_difference": "n.s.",
                }[result]

        colors = [RESULT_COLORS[name] for name in result_order]
        cmap = plt.matplotlib.colors.ListedColormap(colors)
        norm = plt.matplotlib.colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

        fig, ax = plt.subplots(figsize=(1.8 * len(comparisons) + 4, 0.8 * len(problems) + 3))
        ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
        ax.set_title(f"Wilcoxon rank-sum result ({metric})")
        ax.set_xlabel("comparison algorithm")
        ax.set_ylabel("problem")
        ax.set_xticks(range(len(comparisons)))
        ax.set_xticklabels(comparisons)
        ax.set_yticks(range(len(problems)))
        ax.set_yticklabels([p.upper() for p in problems])

        for i in range(len(problems)):
            for j in range(len(comparisons)):
                ax.text(j, i, labels[i][j], ha="center", va="center", color="white", fontsize=8)

        handles = [
            plt.Rectangle((0, 0), 1, 1, color=RESULT_COLORS["reference_better"], label="OADE better"),
            plt.Rectangle((0, 0), 1, 1, color=RESULT_COLORS["comparison_better"], label="Comparison better"),
            plt.Rectangle(
                (0, 0),
                1,
                1,
                color=RESULT_COLORS["no_significant_difference"],
                label="Not significant",
            ),
        ]
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0))
        fig.tight_layout()
        fig.savefig(out_dir / f"wilcoxon_{metric.lower()}_result_heatmap.png", dpi=220)
        plt.close(fig)


def plot_result_counts(rows: list[dict], out_dir: Path) -> None:
    comparisons = ordered_unique(row["comparison_algorithm"] for row in rows)
    metrics = ordered_unique(row["metric"] for row in rows)
    result_order = ["reference_better", "comparison_better", "no_significant_difference"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(7 * len(metrics), 5), squeeze=False)
    for idx, metric in enumerate(metrics):
        ax = axes[0][idx]
        x = np.arange(len(comparisons))
        bottom = np.zeros(len(comparisons))

        for result in result_order:
            counts = []
            for comparison in comparisons:
                counts.append(
                    sum(
                        1
                        for row in rows
                        if row["metric"] == metric
                        and row["comparison_algorithm"] == comparison
                        and row["result"] == result
                    )
                )
            ax.bar(x, counts, bottom=bottom, color=RESULT_COLORS[result], label=result)
            bottom += np.asarray(counts)

        ax.set_title(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(comparisons)
        ax.set_ylabel("problem count")
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    axes[0][-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle("Wilcoxon rank-sum result counts")
    fig.tight_layout()
    fig.savefig(out_dir / "wilcoxon_result_counts.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="results_zdt/all_problems_wilcoxon_rank_sum.csv")
    parser.add_argument("--out-dir", type=str, default="results_zdt/wilcoxon_plots")
    args = parser.parse_args()

    rows = read_csv(Path(args.input))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_pvalue_heatmap(rows, out_dir)
    plot_result_heatmap(rows, out_dir)
    plot_result_counts(rows, out_dir)

    print(f"Saved Wilcoxon plots to: {out_dir.as_posix()}")


if __name__ == "__main__":
    main()
