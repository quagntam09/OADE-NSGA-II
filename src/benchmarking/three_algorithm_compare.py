"""Benchmark OADE-NSGA-II against the two standalone algorithms on ZDT."""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.problems import get_problem

from algorithms.ce_moea import CEMOEA
from algorithms.nsga2_sdr_ols import NSGAIISDROLS
from algorithms.nsga2_sdr_ols import ProblemAdapter as SDROLSProblemAdapter
from src.benchmarking.pymoo_compare import make_initial_population, non_dominated, run_oade
from src.utils.config import load_yaml_config
from src.utils.csv_io import write_csv


DEFAULT_CONFIG_PATH = Path("config") / "three_algorithms.yaml"
SUPPORTED_ALGORITHMS = ("OADE_NSGAII", "NSGAII_SDR_OLS", "CE_MOEA")

RAW_FIELDNAMES = [
    "problem",
    "n_var",
    "run_idx",
    "seed",
    "algorithm",
    "IGD",
    "HV",
]

SUMMARY_FIELDNAMES = [
    "problem",
    "n_var",
    "algorithm",
    "IGD_mean",
    "IGD_std",
    "IGD_best",
    "HV_mean",
    "HV_std",
    "HV_best",
]


@dataclass
class RunMetrics:
    igd: float
    hv: float


Runner = Callable[[object, int, int, np.ndarray, int, dict], np.ndarray]


def validate_algorithms(names: list[str]) -> list[str]:
    unknown = [name for name in names if name not in SUPPORTED_ALGORITHMS]
    if unknown:
        raise ValueError(f"Unsupported algorithms: {unknown}. Supported: {list(SUPPORTED_ALGORITHMS)}")
    return names


def run_nsga2_sdr_ols(problem, pop_size: int, n_gen: int, initial_x: np.ndarray, seed: int, cfg: dict) -> np.ndarray:
    np.random.seed(seed)
    random.seed(seed)
    max_neighbors_factor = cfg.get("local_search_max_neighbors_factor")
    max_neighbors = None if max_neighbors_factor is None else int(max_neighbors_factor) * pop_size

    solver = NSGAIISDROLS(
        SDROLSProblemAdapter(problem),
        pop_size=pop_size,
        n_gen=n_gen,
        mu=float(cfg.get("mu", 0.0)),
        sigma=float(cfg.get("sigma", 0.1)),
        seed=seed,
        local_search_max_neighbors=max_neighbors,
        crossover_prob=float(cfg.get("crossover_prob", 0.9)),
        mutation_prob=cfg.get("mutation_prob"),
        eta_c=float(cfg.get("eta_c", 20.0)),
        eta_m=float(cfg.get("eta_m", 20.0)),
    )
    solver.run(initial_x=initial_x)
    return non_dominated(solver.result_F())


def run_ce_moea(problem, pop_size: int, n_gen: int, initial_x: np.ndarray, seed: int, cfg: dict) -> np.ndarray:
    solver = CEMOEA(
        problem,
        pop_size=pop_size,
        n_gen=n_gen,
        F_DE=float(cfg.get("F_DE", 0.7)),
        CR=float(cfg.get("CR", 0.5)),
        p_m=float(cfg.get("p_m", 0.02)),
        eta_m=float(cfg.get("eta_m", 20.0)),
        seed=seed,
    )
    _, F = solver.run(initial_x=initial_x)
    return non_dominated(F)


def runner_for(name: str) -> Runner:
    if name == "OADE_NSGAII":
        return lambda problem, pop_size, n_gen, initial_x, seed, cfg: run_oade(
            problem, pop_size, n_gen, initial_x, seed, cfg
        )
    if name == "NSGAII_SDR_OLS":
        return run_nsga2_sdr_ols
    if name == "CE_MOEA":
        return run_ce_moea
    raise ValueError(f"Unsupported algorithm: {name}")


def summarize(metrics: dict[str, list[RunMetrics]]) -> dict[str, dict[str, float]]:
    summary = {}
    for algorithm, values in metrics.items():
        igd = np.asarray([item.igd for item in values], dtype=float)
        hv = np.asarray([item.hv for item in values], dtype=float)
        summary[algorithm] = {
            "IGD_mean": float(np.mean(igd)),
            "IGD_std": float(np.std(igd, ddof=1)) if len(igd) > 1 else 0.0,
            "IGD_best": float(np.min(igd)),
            "HV_mean": float(np.mean(hv)),
            "HV_std": float(np.std(hv, ddof=1)) if len(hv) > 1 else 0.0,
            "HV_best": float(np.max(hv)),
        }
    return summary


def run_one_problem(problem_name: str, n_var: int | None, cfg: dict) -> tuple[list[dict], list[dict]]:
    problem_kwargs = {}
    if n_var is not None:
        problem_kwargs["n_var"] = n_var
    problem = get_problem(problem_name, **problem_kwargs)

    pf = np.asarray(problem.pareto_front())
    if pf.ndim != 2:
        raise ValueError(f"Invalid pareto_front() for {problem_name}")

    igd_indicator = IGD(pf)
    hv_indicator = HV(ref_point=np.max(pf, axis=0) + 0.1)

    runs = int(cfg["global"]["runs"])
    pop_size = int(cfg["global"]["pop_size"])
    n_gen = int(cfg["global"]["n_gen"])
    seed_base = int(cfg["global"]["seed_base"])
    algorithms = validate_algorithms(list(cfg["algorithms"]))

    metrics: dict[str, list[RunMetrics]] = {name: [] for name in algorithms}
    raw_rows: list[dict] = []

    for run_idx in range(runs):
        seed = seed_base + run_idx
        initial_x = make_initial_population(problem, pop_size, seed)

        for algorithm in algorithms:
            front = runner_for(algorithm)(
                problem,
                pop_size,
                n_gen,
                initial_x,
                seed,
                cfg.get(algorithm_config_key(algorithm), {}),
            )
            metric = RunMetrics(igd=float(igd_indicator(front)), hv=float(hv_indicator(front)))
            metrics[algorithm].append(metric)
            raw_rows.append(
                {
                    "problem": problem_name,
                    "n_var": n_var,
                    "run_idx": run_idx,
                    "seed": seed,
                    "algorithm": algorithm,
                    "IGD": metric.igd,
                    "HV": metric.hv,
                }
            )

        print(f"[{problem_name}] run {run_idx + 1}/{runs} done (seed={seed}).")

    stats = summarize(metrics)
    summary_rows = [
        {
            "problem": problem_name,
            "n_var": n_var,
            "algorithm": algorithm,
            **stats[algorithm],
        }
        for algorithm in algorithms
    ]

    return raw_rows, summary_rows


def algorithm_config_key(algorithm: str) -> str:
    return {
        "OADE_NSGAII": "oade",
        "NSGAII_SDR_OLS": "nsga2_sdr_ols",
        "CE_MOEA": "ce_moea",
    }[algorithm]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args(argv)

    cfg = load_yaml_config(Path(args.config))
    out_dir = Path(cfg["global"]["out_dir"])
    all_raw_rows: list[dict] = []
    all_summary_rows: list[dict] = []

    for item in cfg["problems"]:
        raw_rows, summary_rows = run_one_problem(str(item["name"]), item.get("n_var"), cfg)
        all_raw_rows.extend(raw_rows)
        all_summary_rows.extend(summary_rows)

    write_csv(out_dir / "three_algorithms_raw_runs.csv", all_raw_rows, RAW_FIELDNAMES)
    write_csv(out_dir / "three_algorithms_igd_hv_summary.csv", all_summary_rows, SUMMARY_FIELDNAMES)

    print(f"Saved: {(out_dir / 'three_algorithms_raw_runs.csv').as_posix()}")
    print(f"Saved: {(out_dir / 'three_algorithms_igd_hv_summary.csv').as_posix()}")
    print("Plot with: python main.py plot-three-algorithms")


if __name__ == "__main__":
    main()

