"""
Benchmark OADE-NSGA-II (main_src) vs NSGA2, R-NSGA-II, D-NSGA-II from pymoo.

- Does NOT modify main_src.
- Uses the same initial population seed for all algorithms in each run.
- Computes IGD and HV using pymoo indicators.
- Exports mean/std and best tables across n independent runs.
"""

from __future__ import annotations

import argparse
import csv
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
    """Return rank-0 front from objective matrix F."""
    nds = NonDominatedSorting()
    front = nds.do(F, only_non_dominated_front=True)
    return F[front]


def make_initial_population(problem, pop_size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    xl = np.asarray(problem.xl)
    xu = np.asarray(problem.xu)
    return rng.uniform(xl, xu, size=(pop_size, problem.n_var))


def run_oade(problem, pop_size: int, n_gen: int, initial_x: np.ndarray, seed: int) -> np.ndarray:
    np.random.seed(seed)
    solver = OADE_NSGAII(ProblemWrapper(problem), pop_size=pop_size, n_gen=n_gen)

    solver.stagnation_patience = 20
    solver.stagnation_tolerance = 1e-4
    solver.restart_elite_ratio = 0.30
    solver.prob_de = 0.5
    solver.n_neighbors = 5
    solver.mean_F = 0.65
    solver.mean_CR = 0.5

    F = solver.run(initial_x=initial_x)
    return non_dominated(F)


def run_nsga2(problem, pop_size: int, n_gen: int, initial_x: np.ndarray, seed: int) -> np.ndarray:
    algorithm = NSGA2(pop_size=pop_size, sampling=initial_x)
    res = minimize(problem, algorithm, termination=("n_gen", n_gen), seed=seed, verbose=False)
    return non_dominated(res.F)


def run_rnsga2(problem, pop_size: int, n_gen: int, initial_x: np.ndarray, seed: int, pf: np.ndarray) -> np.ndarray:
    # R-NSGA-II requires aspiration/reference points.
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str, default="zdt1", help="pymoo problem name, e.g. zdt1, zdt2, dtlz2")
    parser.add_argument("--n-var", type=int, default=None, help="number of variables when supported by problem")
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--pop-size", type=int, default=100)
    parser.add_argument("--n-gen", type=int, default=200)
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="results")
    args = parser.parse_args()

    problem_kwargs = {}
    if args.n_var is not None:
        problem_kwargs["n_var"] = args.n_var
    problem = get_problem(args.problem, **problem_kwargs)

    pf = problem.pareto_front()
    if pf is None:
        raise ValueError(f"Problem '{args.problem}' does not provide an analytical pareto_front(), cannot compute IGD reliably.")

    pf = np.asarray(pf)
    if pf.ndim != 2:
        raise ValueError("pareto_front() must return a 2D array.")

    ref_point = np.max(pf, axis=0) + 0.1
    igd_indicator = IGD(pf)
    hv_indicator = HV(ref_point=ref_point)

    algorithms = ["OADE_NSGAII", "NSGA2", "RNSGA2", "DNSGA2"]
    metrics: Dict[str, List[RunMetrics]] = {name: [] for name in algorithms}

    for run_idx in range(args.runs):
        seed = args.seed_base + run_idx
        initial_x = make_initial_population(problem, args.pop_size, seed)

        fronts = {
            "OADE_NSGAII": run_oade(problem, args.pop_size, args.n_gen, initial_x, seed),
            "NSGA2": run_nsga2(problem, args.pop_size, args.n_gen, initial_x, seed),
            "RNSGA2": run_rnsga2(problem, args.pop_size, args.n_gen, initial_x, seed, pf),
            "DNSGA2": run_dnsga2(problem, args.pop_size, args.n_gen, initial_x, seed),
        }

        for name, front in fronts.items():
            metrics[name].append(
                RunMetrics(
                    igd=float(igd_indicator(front)),
                    hv=float(hv_indicator(front)),
                )
            )

        print(f"Run {run_idx + 1}/{args.runs} done (seed={seed}).")

    stats = summarize(metrics)

    mean_std_rows = []
    best_rows = []
    for name in algorithms:
        mean_std_rows.append(
            {
                "algorithm": name,
                "IGD_mean": stats[name]["igd_mean"],
                "IGD_std": stats[name]["igd_std"],
                "HV_mean": stats[name]["hv_mean"],
                "HV_std": stats[name]["hv_std"],
            }
        )
        best_rows.append(
            {
                "algorithm": name,
                "IGD_best": stats[name]["igd_best"],
                "HV_best": stats[name]["hv_best"],
            }
        )

    out_dir = Path(args.out_dir)
    write_csv(
        out_dir / "igd_hv_mean_std.csv",
        mean_std_rows,
        ["algorithm", "IGD_mean", "IGD_std", "HV_mean", "HV_std"],
    )
    write_csv(
        out_dir / "igd_hv_best.csv",
        best_rows,
        ["algorithm", "IGD_best", "HV_best"],
    )

    print("\n=== Mean/Std Table ===")
    for row in mean_std_rows:
        print(
            f"{row['algorithm']:12s} | "
            f"IGD mean={row['IGD_mean']:.6e}, std={row['IGD_std']:.6e} | "
            f"HV mean={row['HV_mean']:.6e}, std={row['HV_std']:.6e}"
        )

    print("\n=== Best Table ===")
    for row in best_rows:
        print(
            f"{row['algorithm']:12s} | "
            f"IGD best={row['IGD_best']:.6e} | "
            f"HV best={row['HV_best']:.6e}"
        )

    print(f"\nSaved: {(out_dir / 'igd_hv_mean_std.csv').as_posix()}")
    print(f"Saved: {(out_dir / 'igd_hv_best.csv').as_posix()}")


if __name__ == "__main__":
    main()
