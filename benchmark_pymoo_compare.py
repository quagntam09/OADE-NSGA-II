"""
Benchmark OADE-NSGA-II (main_src) vs NSGA2, R-NSGA-II, D-NSGA-II from pymoo.

- Reads all settings from a separate JSON config file.
- Runs all configured problems (default: full ZDT set except ZDT5).
- Uses the same initial population seed for all algorithms in each run.
- Computes IGD and HV using pymoo indicators.
- Exports per-problem and combined CSV summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
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

from main_src import OADE_NSGAII, ProblemWrapper


@dataclass
class RunMetrics:
    igd: float
    hv: float


def non_dominated(F: np.ndarray) -> np.ndarray:
    nds = NonDominatedSorting()
    front = nds.do(F, only_non_dominated_front=True)
    return F[front]


def make_initial_population(problem, pop_size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    xl = np.asarray(problem.xl)
    xu = np.asarray(problem.xu)
    return rng.uniform(xl, xu, size=(pop_size, problem.n_var))


def run_oade(problem, pop_size: int, n_gen: int, initial_x: np.ndarray, seed: int, oade_cfg: dict) -> np.ndarray:
    np.random.seed(seed)
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


def run_rnsga2(problem, pop_size: int, n_gen: int, initial_x: np.ndarray, seed: int, pf: np.ndarray) -> np.ndarray:
    ref_points = np.vstack([pf[0], pf[len(pf) // 2]])
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
        out[name] = {
            "igd_mean": float(np.mean(igd_values)),
            "igd_std": float(np.std(igd_values, ddof=1)),
            "igd_best": float(np.min(igd_values)),
            "hv_mean": float(np.mean(hv_values)),
            "hv_std": float(np.std(hv_values, ddof=1)),
            "hv_best": float(np.max(hv_values)),
        }
    return out


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_one_problem(problem_name: str, n_var: int | None, cfg: dict, out_dir: Path) -> tuple[list[dict], list[dict]]:
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
    algorithms = list(cfg["algorithms"])

    metrics: Dict[str, List[RunMetrics]] = {name: [] for name in algorithms}

    for run_idx in range(runs):
        seed = seed_base + run_idx
        initial_x = make_initial_population(problem, pop_size, seed)

        fronts = {
            "OADE_NSGAII": run_oade(problem, pop_size, n_gen, initial_x, seed, cfg["oade"]),
            "NSGA2": run_nsga2(problem, pop_size, n_gen, initial_x, seed),
            "RNSGA2": run_rnsga2(problem, pop_size, n_gen, initial_x, seed, pf),
            "DNSGA2": run_dnsga2(problem, pop_size, n_gen, initial_x, seed),
        }

        for name, front in fronts.items():
            metrics[name].append(RunMetrics(igd=float(igd_indicator(front)), hv=float(hv_indicator(front))))

        print(f"[{problem_name}] run {run_idx + 1}/{runs} done (seed={seed}).")

    stats = summarize(metrics)

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

    return mean_std_rows, best_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="benchmark_config.json")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    out_dir = Path(cfg["global"]["out_dir"])
    all_mean_std = []
    all_best = []

    for item in cfg["problems"]:
        problem_name = item["name"]
        n_var = item.get("n_var")
        mean_rows, best_rows = run_one_problem(problem_name, n_var, cfg, out_dir)
        all_mean_std.extend(mean_rows)
        all_best.extend(best_rows)

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

    print(f"Saved: {(out_dir / 'all_problems_igd_hv_mean_std.csv').as_posix()}")
    print(f"Saved: {(out_dir / 'all_problems_igd_hv_best.csv').as_posix()}")


if __name__ == "__main__":
    main()
