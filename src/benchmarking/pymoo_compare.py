"""
Benchmark OADE-NSGA-II vs NSGA2, R-NSGA-II, D-NSGA-II from pymoo.

- Reads all settings from a separate YAML config file.
- Runs all configured problems (default: full ZDT set except ZDT5).
- Uses the same initial population seed for all algorithms in each run.
- Computes IGD and HV using pymoo indicators.
- Applies Wilcoxon rank-sum tests at the configured significance level.
- Exports per-problem and combined CSV summaries.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
from pymoo.algorithms.moo.dnsga2 import DNSGA2
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.rnsga2 import RNSGA2
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from scipy.stats import ranksums

from src.oade_nsga2 import OADE_NSGAII, ProblemWrapper
from src.utils.config import load_yaml_config
from src.utils.csv_io import write_csv


@dataclass
class RunMetrics:
    igd: float
    hv: float


SUPPORTED_ALGORITHMS = ("OADE_NSGAII", "NSGA2", "RNSGA2", "DNSGA2")
DEFAULT_CONFIG_PATH = Path("config") / "benchmark.yaml"
WILCOXON_FIELDNAMES = [
    "problem",
    "n_var",
    "metric",
    "test",
    "reference_algorithm",
    "comparison_algorithm",
    "reference_mean",
    "comparison_mean",
    "statistic",
    "p_value",
    "alpha",
    "significant",
    "result",
]

def non_dominated(F: np.ndarray) -> np.ndarray:
    nds = NonDominatedSorting()
    front = nds.do(F, only_non_dominated_front=True)
    return F[front]


def make_initial_population(problem, pop_size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    xl = np.asarray(problem.xl)
    xu = np.asarray(problem.xu)
    return rng.uniform(xl, xu, size=(pop_size, problem.n_var))


def validate_algorithms(algorithms: List[str]) -> List[str]:
    unknown = [name for name in algorithms if name not in SUPPORTED_ALGORITHMS]
    if unknown:
        raise ValueError(
            f"Unsupported algorithms in config: {unknown}. "
            f"Supported: {list(SUPPORTED_ALGORITHMS)}"
        )
    return algorithms


def make_rnsga2_ref_points(problem, initial_x: np.ndarray) -> np.ndarray:
    """
    Tạo aspiration points cho R-NSGA-II từ objective values của quần thể ban đầu.
    Không dùng Pareto front thật để tránh lợi thế oracle khi benchmark.
    """
    F0 = np.asarray(problem.evaluate(initial_x))
    if F0.ndim != 2:
        raise ValueError("problem.evaluate(initial_x) must return a 2D objective matrix")

    f_min = F0.min(axis=0)
    f_max = F0.max(axis=0)
    span = np.where((f_max - f_min) < 1e-12, 1.0, f_max - f_min)
    n_obj = F0.shape[1]

    if n_obj == 1:
        return (f_min + 0.1 * span)[None, :]

    rp1 = f_min + 0.2 * span
    rp2 = f_min + 0.2 * span
    rp1[0] = f_min[0] + 0.05 * span[0]
    rp2[0] = f_min[0] + 0.45 * span[0]
    rp1[-1] = f_min[-1] + 0.45 * span[-1]
    rp2[-1] = f_min[-1] + 0.05 * span[-1]

    return np.vstack([rp1, rp2])


def run_oade(problem, pop_size: int, n_gen: int, initial_x: np.ndarray, seed: int, oade_cfg: dict) -> np.ndarray:
    np.random.seed(seed)
    random.seed(seed)
    solver = OADE_NSGAII(ProblemWrapper(problem), pop_size=pop_size, n_gen=n_gen)

    solver.stagnation_patience = int(oade_cfg["stagnation_patience"])
    solver.stagnation_tolerance = float(oade_cfg["stagnation_tolerance"])
    solver.restart_elite_ratio = float(oade_cfg["restart_elite_ratio"])
    solver.prob_de = float(oade_cfg["prob_de"])
    solver.n_neighbors = int(oade_cfg["n_neighbors"])
    solver.mean_F = float(oade_cfg["mean_F"])
    solver.mean_CR = float(oade_cfg["mean_CR"])

    F = solver.run(initial_x=initial_x)
    return non_dominated(F)


def run_nsga2(problem, pop_size: int, n_gen: int, initial_x: np.ndarray, seed: int) -> np.ndarray:
    algorithm = NSGA2(pop_size=pop_size, sampling=initial_x)
    res = minimize(problem, algorithm, termination=("n_gen", n_gen), seed=seed, verbose=False)
    return non_dominated(res.F)


def run_rnsga2(problem, pop_size: int, n_gen: int, initial_x: np.ndarray, seed: int, ref_points: np.ndarray) -> np.ndarray:
    algorithm = RNSGA2(ref_points=ref_points, pop_size=pop_size, sampling=initial_x)
    res = minimize(problem, algorithm, termination=("n_gen", n_gen), seed=seed, verbose=False)
    return non_dominated(res.F)


def run_dnsga2(problem, pop_size: int, n_gen: int, initial_x: np.ndarray, seed: int) -> np.ndarray:
    algorithm = DNSGA2(pop_size=pop_size, sampling=initial_x)
    res = minimize(problem, algorithm, termination=("n_gen", n_gen), seed=seed, verbose=False)
    return non_dominated(res.F)


def summarize(metrics: Dict[str, List[RunMetrics]]) -> Dict[str, Dict[str, float]]:
    out = {}
    for name, values in metrics.items():
        igd_values = np.array([m.igd for m in values], dtype=float)
        hv_values = np.array([m.hv for m in values], dtype=float)
        if len(values) >= 2:
            igd_std = float(np.std(igd_values, ddof=1))
            hv_std = float(np.std(hv_values, ddof=1))
        else:
            igd_std = 0.0
            hv_std = 0.0
        out[name] = {
            "igd_mean": float(np.mean(igd_values)),
            "igd_std": igd_std,
            "igd_best": float(np.min(igd_values)),
            "hv_mean": float(np.mean(hv_values)),
            "hv_std": hv_std,
            "hv_best": float(np.max(hv_values)),
        }
    return out


def metric_values(metrics: Dict[str, List[RunMetrics]], algorithm: str, metric: str) -> np.ndarray:
    if metric == "IGD":
        return np.array([m.igd for m in metrics[algorithm]], dtype=float)
    if metric == "HV":
        return np.array([m.hv for m in metrics[algorithm]], dtype=float)
    raise ValueError(f"Unsupported metric: {metric}")


def better_label(reference_values: np.ndarray, comparison_values: np.ndarray, metric: str, significant: bool) -> str:
    if not significant:
        return "no_significant_difference"

    reference_mean = float(np.mean(reference_values))
    comparison_mean = float(np.mean(comparison_values))
    if metric == "IGD":
        return "reference_better" if reference_mean < comparison_mean else "comparison_better"
    if metric == "HV":
        return "reference_better" if reference_mean > comparison_mean else "comparison_better"
    raise ValueError(f"Unsupported metric: {metric}")


def wilcoxon_rank_sum_rows(
    problem_name: str,
    n_var: int | None,
    metrics: Dict[str, List[RunMetrics]],
    algorithms: List[str],
    cfg: dict,
) -> List[dict]:
    stat_cfg = cfg.get("statistics", {})
    alpha = float(stat_cfg.get("alpha", 0.05))
    reference_algorithm = str(stat_cfg.get("reference_algorithm", "OADE_NSGAII"))

    if reference_algorithm not in algorithms:
        raise ValueError(f"statistics.reference_algorithm not in algorithms: {reference_algorithm}")

    rows = []
    for comparison_algorithm in algorithms:
        if comparison_algorithm == reference_algorithm:
            continue

        for metric in ("IGD", "HV"):
            reference_values = metric_values(metrics, reference_algorithm, metric)
            comparison_values = metric_values(metrics, comparison_algorithm, metric)
            test_result = ranksums(reference_values, comparison_values)
            p_value = float(test_result.pvalue)
            significant = p_value < alpha

            rows.append(
                {
                    "problem": problem_name,
                    "n_var": n_var,
                    "metric": metric,
                    "test": "wilcoxon_rank_sum",
                    "reference_algorithm": reference_algorithm,
                    "comparison_algorithm": comparison_algorithm,
                    "reference_mean": float(np.mean(reference_values)),
                    "comparison_mean": float(np.mean(comparison_values)),
                    "statistic": float(test_result.statistic),
                    "p_value": p_value,
                    "alpha": alpha,
                    "significant": significant,
                    "result": better_label(reference_values, comparison_values, metric, significant),
                }
            )

    return rows


def run_one_problem(problem_name: str, n_var: int | None, cfg: dict, out_dir: Path) -> tuple[list[dict], list[dict], list[dict]]:
    problem_kwargs = {}
    if n_var is not None:
        problem_kwargs["n_var"] = n_var
    problem = get_problem(problem_name, **problem_kwargs)

    pf = np.asarray(problem.pareto_front())
    if pf.ndim != 2:
        raise ValueError(f"pareto_front() invalid for {problem_name}")

    ref_point = np.max(pf, axis=0) + 0.1
    igd_indicator = IGD(pf)
    hv_indicator = HV(ref_point=ref_point)

    runs = int(cfg["global"]["runs"])
    pop_size = int(cfg["global"]["pop_size"])
    n_gen = int(cfg["global"]["n_gen"])
    seed_base = int(cfg["global"]["seed_base"])
    algorithms = validate_algorithms(list(cfg["algorithms"]))

    metrics: Dict[str, List[RunMetrics]] = {name: [] for name in algorithms}

    for run_idx in range(runs):
        seed = seed_base + run_idx
        initial_x = make_initial_population(problem, pop_size, seed)
        rnsga2_ref_points = make_rnsga2_ref_points(problem, initial_x)

        for name in algorithms:
            if name == "OADE_NSGAII":
                front = run_oade(problem, pop_size, n_gen, initial_x, seed, cfg["oade"])
            elif name == "NSGA2":
                front = run_nsga2(problem, pop_size, n_gen, initial_x, seed)
            elif name == "RNSGA2":
                front = run_rnsga2(problem, pop_size, n_gen, initial_x, seed, rnsga2_ref_points)
            elif name == "DNSGA2":
                front = run_dnsga2(problem, pop_size, n_gen, initial_x, seed)
            else:
                raise ValueError(f"Unsupported algorithm: {name}")

            metrics[name].append(RunMetrics(igd=float(igd_indicator(front)), hv=float(hv_indicator(front))))

        print(f"[{problem_name}] run {run_idx + 1}/{runs} done (seed={seed}).")

    stats = summarize(metrics)
    wilcoxon_rows = wilcoxon_rank_sum_rows(problem_name, n_var, metrics, algorithms, cfg)

    mean_std_rows = []
    best_rows = []
    for name in algorithms:
        mean_std_rows.append(
            {
                "problem": problem_name,
                "n_var": n_var,
                "algorithm": name,
                "IGD_mean": stats[name]["igd_mean"],
                "IGD_std": stats[name]["igd_std"],
                "HV_mean": stats[name]["hv_mean"],
                "HV_std": stats[name]["hv_std"],
            }
        )
        best_rows.append(
            {
                "problem": problem_name,
                "n_var": n_var,
                "algorithm": name,
                "IGD_best": stats[name]["igd_best"],
                "HV_best": stats[name]["hv_best"],
            }
        )

    write_csv(
        out_dir / f"{problem_name}_igd_hv_mean_std.csv",
        mean_std_rows,
        ["problem", "n_var", "algorithm", "IGD_mean", "IGD_std", "HV_mean", "HV_std"],
    )
    write_csv(
        out_dir / f"{problem_name}_igd_hv_best.csv",
        best_rows,
        ["problem", "n_var", "algorithm", "IGD_best", "HV_best"],
    )
    write_csv(
        out_dir / f"{problem_name}_wilcoxon_rank_sum.csv",
        wilcoxon_rows,
        WILCOXON_FIELDNAMES,
    )

    return mean_std_rows, best_rows, wilcoxon_rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args(argv)

    cfg_path = Path(args.config)
    cfg = load_yaml_config(cfg_path)

    out_dir = Path(cfg["global"]["out_dir"])
    all_mean_std = []
    all_best = []
    all_wilcoxon = []

    for item in cfg["problems"]:
        problem_name = item["name"]
        n_var = item.get("n_var")
        mean_rows, best_rows, wilcoxon_rows = run_one_problem(problem_name, n_var, cfg, out_dir)
        all_mean_std.extend(mean_rows)
        all_best.extend(best_rows)
        all_wilcoxon.extend(wilcoxon_rows)

    write_csv(
        out_dir / "all_problems_igd_hv_mean_std.csv",
        all_mean_std,
        ["problem", "n_var", "algorithm", "IGD_mean", "IGD_std", "HV_mean", "HV_std"],
    )
    write_csv(
        out_dir / "all_problems_igd_hv_best.csv",
        all_best,
        ["problem", "n_var", "algorithm", "IGD_best", "HV_best"],
    )
    write_csv(
        out_dir / "all_problems_wilcoxon_rank_sum.csv",
        all_wilcoxon,
        WILCOXON_FIELDNAMES,
    )

    print(f"Saved: {(out_dir / 'all_problems_igd_hv_mean_std.csv').as_posix()}")
    print(f"Saved: {(out_dir / 'all_problems_igd_hv_best.csv').as_posix()}")
    print(f"Saved: {(out_dir / 'all_problems_wilcoxon_rank_sum.csv').as_posix()}")


if __name__ == "__main__":
    main()
