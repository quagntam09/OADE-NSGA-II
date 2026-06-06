"""
Run parameter sensitivity experiments for OADE-NSGA-II.

This script is intentionally separate from the main benchmark config and plot
script. It writes raw per-run metrics, summary tables, statistical tests, and
F/CR traces into sensitivity_analysis/results by default.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.problems import get_problem
from scipy.stats import friedmanchisquare, wilcoxon

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark_pymoo_compare import load_yaml_config, make_initial_population, non_dominated
from main_src import OADE_NSGAII, ProblemWrapper


RAW_FIELDNAMES = [
    "study",
    "parameter",
    "parameter_label",
    "problem",
    "n_var",
    "run_idx",
    "seed",
    "pop_size",
    "n_gen",
    "stagnation_patience",
    "initial_mean_F",
    "initial_mean_CR",
    "final_mean_F",
    "final_mean_CR",
    "mean_F_gen30",
    "mean_CR_gen30",
    "mean_F_gen50",
    "mean_CR_gen50",
    "IGD",
    "HV",
    "runtime_seconds",
    "restart_count",
]

SUMMARY_FIELDNAMES = [
    "study",
    "parameter",
    "parameter_label",
    "problem",
    "n_var",
    "runs",
    "IGD_mean",
    "IGD_std",
    "IGD_best",
    "HV_mean",
    "HV_std",
    "HV_best",
    "runtime_mean",
    "restart_count_mean",
    "final_mean_F_mean",
    "final_mean_CR_mean",
]

STATS_FIELDNAMES = [
    "study",
    "problem",
    "metric",
    "test",
    "baseline_label",
    "comparison_label",
    "statistic",
    "p_value",
    "p_adjusted_holm",
    "alpha",
    "significant",
    "result",
]

TRACE_FIELDNAMES = [
    "study",
    "parameter_label",
    "problem",
    "run_idx",
    "seed",
    "generation",
    "mean_F",
    "mean_CR",
    "prob_de",
]


@dataclass(frozen=True)
class Experiment:
    study: str
    parameter: str
    label: str
    spec: Any


class TrackingOADE_NSGAII(OADE_NSGAII):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.restart_count = 0

    def _partial_restart(self) -> None:
        self.restart_count += 1
        super()._partial_restart()


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.std(np.asarray(values, dtype=float), ddof=1))


def metric_direction(metric: str) -> str:
    if metric == "IGD":
        return "lower"
    if metric == "HV":
        return "higher"
    raise ValueError(f"Unsupported metric: {metric}")


def compare_result(baseline_values: list[float], comparison_values: list[float], metric: str, significant: bool) -> str:
    if not significant:
        return "no_significant_difference"
    baseline_mean = float(np.mean(baseline_values))
    comparison_mean = float(np.mean(comparison_values))
    if metric_direction(metric) == "lower":
        return "baseline_better" if baseline_mean < comparison_mean else "comparison_better"
    return "baseline_better" if baseline_mean > comparison_mean else "comparison_better"


def holm_adjust(p_values: list[float]) -> list[float]:
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [1.0 for _ in p_values]
    previous = 0.0
    total = len(p_values)
    for rank, (original_idx, p_value) in enumerate(indexed):
        adjusted_value = min(1.0, (total - rank) * p_value)
        adjusted_value = max(previous, adjusted_value)
        previous = adjusted_value
        adjusted[original_idx] = adjusted_value
    return adjusted


def build_experiments(cfg: dict) -> tuple[list[Experiment], dict[str, str], dict[str, list[str]]]:
    experiments = []
    baselines = {}
    study_labels = {}

    for study_cfg in cfg["studies"]:
        study = str(study_cfg["name"])
        parameter = str(study_cfg["parameter"])
        baselines[study] = str(study_cfg["baseline"])
        study_labels[study] = []

        for value in study_cfg["values"]:
            label = str(value.get("label")) if isinstance(value, dict) else str(value)
            experiments.append(Experiment(study=study, parameter=parameter, label=label, spec=value))
            study_labels[study].append(label)

    return experiments, baselines, study_labels


def resolve_settings(cfg: dict, experiment: Experiment, seed: int) -> tuple[int, dict, float, float]:
    baseline = dict(cfg["baseline"])
    oade_cfg = {
        "stagnation_patience": int(baseline["stagnation_patience"]),
        "stagnation_tolerance": float(baseline["stagnation_tolerance"]),
        "restart_elite_ratio": float(baseline["restart_elite_ratio"]),
        "prob_de": float(baseline["prob_de"]),
        "n_neighbors": int(baseline["n_neighbors"]),
        "mean_F": float(baseline["mean_F"]),
        "mean_CR": float(baseline["mean_CR"]),
    }
    pop_size = int(baseline["pop_size"])

    if experiment.parameter == "pop_size":
        pop_size = int(experiment.spec)
    elif experiment.parameter == "stagnation_patience":
        oade_cfg["stagnation_patience"] = int(experiment.spec)
    elif experiment.parameter == "f_cr_init":
        spec = dict(experiment.spec)
        rng = np.random.default_rng(seed + 100_000)
        if "mean_F_range" in spec:
            low, high = [float(v) for v in spec["mean_F_range"]]
            oade_cfg["mean_F"] = float(rng.uniform(low, high))
        else:
            oade_cfg["mean_F"] = float(spec["mean_F"])

        if "mean_CR_range" in spec:
            low, high = [float(v) for v in spec["mean_CR_range"]]
            oade_cfg["mean_CR"] = float(rng.uniform(low, high))
        else:
            oade_cfg["mean_CR"] = float(spec["mean_CR"])
    else:
        raise ValueError(f"Unsupported sensitivity parameter: {experiment.parameter}")

    return pop_size, oade_cfg, oade_cfg["mean_F"], oade_cfg["mean_CR"]


def apply_oade_config(solver: TrackingOADE_NSGAII, oade_cfg: dict) -> None:
    solver.stagnation_patience = int(oade_cfg["stagnation_patience"])
    solver.stagnation_tolerance = float(oade_cfg["stagnation_tolerance"])
    solver.restart_elite_ratio = float(oade_cfg["restart_elite_ratio"])
    solver.prob_de = float(oade_cfg["prob_de"])
    solver.n_neighbors = int(oade_cfg["n_neighbors"])
    solver.mean_F = float(oade_cfg["mean_F"])
    solver.mean_CR = float(oade_cfg["mean_CR"])


def trace_value(trace: list[dict], key: str, generation: int) -> float | str:
    if len(trace) < generation:
        return ""
    return float(trace[generation - 1][key])


def run_experiment(
    cfg: dict,
    experiment: Experiment,
    problem_name: str,
    n_var: int | None,
    run_idx: int,
    seed: int,
) -> tuple[dict, list[dict]]:
    problem_kwargs = {}
    if n_var is not None:
        problem_kwargs["n_var"] = n_var
    problem = get_problem(problem_name, **problem_kwargs)

    pf = np.asarray(problem.pareto_front())
    igd_indicator = IGD(pf)
    hv_indicator = HV(ref_point=np.max(pf, axis=0) + 0.1)

    pop_size, oade_cfg, initial_mean_f, initial_mean_cr = resolve_settings(cfg, experiment, seed)
    n_gen = int(cfg["global"]["n_gen"])
    initial_x = make_initial_population(problem, pop_size, seed)

    np.random.seed(seed)
    random.seed(seed)
    solver = TrackingOADE_NSGAII(ProblemWrapper(problem), pop_size=pop_size, n_gen=n_gen)
    apply_oade_config(solver, oade_cfg)

    trace = []

    def record_trace(generation: int, _n_gen: int) -> None:
        trace.append(
            {
                "generation": generation,
                "mean_F": float(solver.mean_F),
                "mean_CR": float(solver.mean_CR),
                "prob_de": float(solver.prob_de),
            }
        )

    start = time.perf_counter()
    front = non_dominated(solver.run(initial_x=initial_x, progress_callback=record_trace))
    runtime = time.perf_counter() - start

    raw_row = {
        "study": experiment.study,
        "parameter": experiment.parameter,
        "parameter_label": experiment.label,
        "problem": problem_name,
        "n_var": n_var,
        "run_idx": run_idx,
        "seed": seed,
        "pop_size": pop_size,
        "n_gen": n_gen,
        "stagnation_patience": int(oade_cfg["stagnation_patience"]),
        "initial_mean_F": initial_mean_f,
        "initial_mean_CR": initial_mean_cr,
        "final_mean_F": float(solver.mean_F),
        "final_mean_CR": float(solver.mean_CR),
        "mean_F_gen30": trace_value(trace, "mean_F", 30),
        "mean_CR_gen30": trace_value(trace, "mean_CR", 30),
        "mean_F_gen50": trace_value(trace, "mean_F", 50),
        "mean_CR_gen50": trace_value(trace, "mean_CR", 50),
        "IGD": float(igd_indicator(front)),
        "HV": float(hv_indicator(front)),
        "runtime_seconds": float(runtime),
        "restart_count": int(solver.restart_count),
    }

    trace_rows = [
        {
            "study": experiment.study,
            "parameter_label": experiment.label,
            "problem": problem_name,
            "run_idx": run_idx,
            "seed": seed,
            **item,
        }
        for item in trace
    ]

    return raw_row, trace_rows


def summarize(raw_rows: list[dict]) -> list[dict]:
    grouped = {}
    for row in raw_rows:
        key = (row["study"], row["parameter"], row["parameter_label"], row["problem"], row["n_var"])
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    for (study, parameter, label, problem, n_var), rows in grouped.items():
        igd = [float(row["IGD"]) for row in rows]
        hv = [float(row["HV"]) for row in rows]
        runtime = [float(row["runtime_seconds"]) for row in rows]
        restart_count = [float(row["restart_count"]) for row in rows]
        final_mean_f = [float(row["final_mean_F"]) for row in rows]
        final_mean_cr = [float(row["final_mean_CR"]) for row in rows]
        summary_rows.append(
            {
                "study": study,
                "parameter": parameter,
                "parameter_label": label,
                "problem": problem,
                "n_var": n_var,
                "runs": len(rows),
                "IGD_mean": float(np.mean(igd)),
                "IGD_std": std(igd),
                "IGD_best": float(np.min(igd)),
                "HV_mean": float(np.mean(hv)),
                "HV_std": std(hv),
                "HV_best": float(np.max(hv)),
                "runtime_mean": float(np.mean(runtime)),
                "restart_count_mean": float(np.mean(restart_count)),
                "final_mean_F_mean": float(np.mean(final_mean_f)),
                "final_mean_CR_mean": float(np.mean(final_mean_cr)),
            }
        )
    return summary_rows


def values_by_run(rows: list[dict], study: str, problem: str, label: str, metric: str) -> dict[int, float]:
    return {
        int(row["run_idx"]): float(row[metric])
        for row in rows
        if row["study"] == study and row["problem"] == problem and row["parameter_label"] == label
    }


def statistical_tests(
    raw_rows: list[dict],
    baselines: dict[str, str],
    study_labels: dict[str, list[str]],
    alpha: float,
) -> list[dict]:
    stats_rows = []
    problems = sorted({str(row["problem"]) for row in raw_rows})

    for study, labels in study_labels.items():
        baseline_label = baselines[study]
        for problem in problems:
            for metric in ("IGD", "HV"):
                label_values = []
                common_runs = None
                for label in labels:
                    value_map = values_by_run(raw_rows, study, problem, label, metric)
                    common_runs = set(value_map) if common_runs is None else common_runs & set(value_map)
                    label_values.append(value_map)

                common_run_list = sorted(common_runs or [])
                arrays = [[value_map[i] for i in common_run_list] for value_map in label_values]
                if len(arrays) >= 3 and common_run_list:
                    try:
                        test_result = friedmanchisquare(*arrays)
                        statistic = float(test_result.statistic)
                        p_value = float(test_result.pvalue)
                    except ValueError:
                        statistic = 0.0
                        p_value = 1.0
                    stats_rows.append(
                        {
                            "study": study,
                            "problem": problem,
                            "metric": metric,
                            "test": "friedman",
                            "baseline_label": baseline_label,
                            "comparison_label": "all",
                            "statistic": statistic,
                            "p_value": p_value,
                            "p_adjusted_holm": "",
                            "alpha": alpha,
                            "significant": p_value < alpha,
                            "result": "parameter_effect_detected" if p_value < alpha else "no_parameter_effect_detected",
                        }
                    )

                baseline_map = values_by_run(raw_rows, study, problem, baseline_label, metric)
                pair_rows = []
                p_values = []
                for label in labels:
                    if label == baseline_label:
                        continue
                    comparison_map = values_by_run(raw_rows, study, problem, label, metric)
                    paired_runs = sorted(set(baseline_map) & set(comparison_map))
                    baseline_values = [baseline_map[i] for i in paired_runs]
                    comparison_values = [comparison_map[i] for i in paired_runs]
                    if not paired_runs:
                        continue
                    try:
                        test_result = wilcoxon(baseline_values, comparison_values, zero_method="wilcox")
                        statistic = float(test_result.statistic)
                        p_value = float(test_result.pvalue)
                    except ValueError:
                        statistic = 0.0
                        p_value = 1.0
                    pair_rows.append((label, statistic, p_value, baseline_values, comparison_values))
                    p_values.append(p_value)

                adjusted_values = holm_adjust(p_values)
                for (label, statistic, p_value, baseline_values, comparison_values), adjusted in zip(
                    pair_rows, adjusted_values
                ):
                    significant = adjusted < alpha
                    stats_rows.append(
                        {
                            "study": study,
                            "problem": problem,
                            "metric": metric,
                            "test": "wilcoxon_signed_rank_vs_baseline",
                            "baseline_label": baseline_label,
                            "comparison_label": label,
                            "statistic": statistic,
                            "p_value": p_value,
                            "p_adjusted_holm": adjusted,
                            "alpha": alpha,
                            "significant": significant,
                            "result": compare_result(baseline_values, comparison_values, metric, significant),
                        }
                    )

    return stats_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(Path(__file__).with_name("config.yaml")))
    args = parser.parse_args()

    cfg = load_yaml_config(Path(args.config))
    out_dir = Path(cfg["global"]["out_dir"])
    runs = int(cfg["global"]["runs"])
    seed_base = int(cfg["global"]["seed_base"])
    alpha = float(cfg["global"]["alpha"])
    experiments, baselines, study_labels = build_experiments(cfg)

    raw_rows = []
    trace_rows = []

    for experiment in experiments:
        for problem_item in cfg["problems"]:
            problem_name = str(problem_item["name"])
            n_var = problem_item.get("n_var")
            for run_idx in range(runs):
                seed = seed_base + run_idx
                raw_row, new_trace_rows = run_experiment(cfg, experiment, problem_name, n_var, run_idx, seed)
                raw_rows.append(raw_row)
                trace_rows.extend(new_trace_rows)
                print(
                    f"[{experiment.study}={experiment.label}] "
                    f"{problem_name} run {run_idx + 1}/{runs} done (seed={seed})."
                )

    summary_rows = summarize(raw_rows)
    stats_rows = statistical_tests(raw_rows, baselines, study_labels, alpha)

    write_csv(out_dir / "sensitivity_raw_runs.csv", raw_rows, RAW_FIELDNAMES)
    write_csv(out_dir / "sensitivity_summary.csv", summary_rows, SUMMARY_FIELDNAMES)
    write_csv(out_dir / "sensitivity_statistics.csv", stats_rows, STATS_FIELDNAMES)
    write_csv(out_dir / "sensitivity_fcr_trace.csv", trace_rows, TRACE_FIELDNAMES)

    print(f"Saved: {(out_dir / 'sensitivity_raw_runs.csv').as_posix()}")
    print(f"Saved: {(out_dir / 'sensitivity_summary.csv').as_posix()}")
    print(f"Saved: {(out_dir / 'sensitivity_statistics.csv').as_posix()}")
    print(f"Saved: {(out_dir / 'sensitivity_fcr_trace.csv').as_posix()}")
    print("Plot with: python sensitivity_analysis/plot_parameter_sensitivity.py")


if __name__ == "__main__":
    main()
