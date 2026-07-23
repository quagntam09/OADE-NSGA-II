"""Unified benchmark pipeline for local and pymoo multi-objective algorithms."""

from __future__ import annotations

import argparse
import importlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.optimize import minimize
from pymoo.problems import get_problem

from algorithms.ce_moea import CEMOEA
from algorithms.cgde_nsga2 import CGDE_NSGAII
from algorithms.cgde_nsga2 import ProblemAdapter as CGDEProblemAdapter
from algorithms.nsga2_sdr_ols import NSGAIISDROLS
from algorithms.nsga2_sdr_ols import ProblemAdapter as SDROLSProblemAdapter
from src.benchmarking.pymoo_compare import make_initial_population, make_rnsga2_ref_points, non_dominated, run_oade
from src.utils.config import load_yaml_config
from src.utils.csv_io import write_csv


DEFAULT_CONFIG_PATH = Path("config") / "algorithm_benchmark.yaml"
RAW_CSV = "algorithm_benchmark_raw_runs.csv"
SUMMARY_CSV = "algorithm_benchmark_igd_hv_summary.csv"

RAW_FIELDNAMES = [
    "problem",
    "n_var",
    "run_idx",
    "seed",
    "algorithm",
    "runner",
    "IGD",
    "HV",
]

SUMMARY_FIELDNAMES = [
    "problem",
    "n_var",
    "algorithm",
    "runner",
    "IGD_mean",
    "IGD_std",
    "IGD_best",
    "HV_mean",
    "HV_std",
    "HV_best",
]


@dataclass(frozen=True)
class AlgorithmSpec:
    name: str
    runner: str
    params: dict[str, Any]
    class_path: str | None = None
    sampling: str | None = None
    auto_ref_points: bool = False


@dataclass(frozen=True)
class RunMetrics:
    igd: float
    hv: float


def load_algorithm_specs(paths: list[str]) -> list[AlgorithmSpec]:
    specs = []
    for path_text in paths:
        cfg = load_yaml_config(Path(path_text))
        params = cfg.get("params") or {}
        if not isinstance(params, dict):
            params = {}
        specs.append(
            AlgorithmSpec(
                name=str(cfg["name"]),
                runner=str(cfg["runner"]),
                params=dict(params),
                class_path=cfg.get("class_path"),
                sampling=cfg.get("sampling"),
                auto_ref_points=bool(cfg.get("auto_ref_points", False)),
            )
        )
    return specs


def import_class(class_path: str):
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def run_pymoo_algorithm(
    spec: AlgorithmSpec,
    problem,
    pop_size: int,
    n_gen: int,
    initial_x: np.ndarray,
    seed: int,
) -> np.ndarray:
    if not spec.class_path:
        raise ValueError(f"{spec.name} uses runner=pymoo but has no class_path")

    algorithm_class = import_class(spec.class_path)
    params = dict(spec.params)
    params.setdefault("pop_size", pop_size)
    if spec.sampling == "initial_population":
        params.setdefault("sampling", initial_x)
    if spec.auto_ref_points:
        params.setdefault("ref_points", make_rnsga2_ref_points(problem, initial_x))

    algorithm = algorithm_class(**params)
    result = minimize(problem, algorithm, termination=("n_gen", n_gen), seed=seed, verbose=False)
    return non_dominated(result.F)


def run_nsga2_sdr_ols(
    spec: AlgorithmSpec,
    problem,
    pop_size: int,
    n_gen: int,
    initial_x: np.ndarray,
    seed: int,
) -> np.ndarray:
    np.random.seed(seed)
    random.seed(seed)
    cfg = spec.params
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


def run_ce_moea(
    spec: AlgorithmSpec,
    problem,
    pop_size: int,
    n_gen: int,
    initial_x: np.ndarray,
    seed: int,
) -> np.ndarray:
    cfg = spec.params
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


def run_cgde_nsga2(
    spec: AlgorithmSpec,
    problem,
    pop_size: int,
    n_gen: int,
    initial_x: np.ndarray,
    seed: int,
) -> np.ndarray:
    cfg = spec.params
    solver = CGDE_NSGAII(
        CGDEProblemAdapter(problem),
        pop_size=pop_size,
        n_gen=n_gen,
        F=float(cfg.get("F", 0.5)),
        CR=float(cfg.get("CR", 0.9)),
        pbest_ratio=float(cfg.get("pbest_ratio", 0.2)),
        archive_size_factor=float(cfg.get("archive_size_factor", 1.0)),
        mutation_prob=cfg.get("mutation_prob"),
        eta_m=float(cfg.get("eta_m", 20.0)),
        seed=seed,
    )
    _, F = solver.run(initial_x=initial_x)
    return non_dominated(F)


def run_algorithm(
    spec: AlgorithmSpec,
    problem,
    pop_size: int,
    n_gen: int,
    initial_x: np.ndarray,
    seed: int,
) -> np.ndarray:
    if spec.runner == "pymoo":
        return run_pymoo_algorithm(spec, problem, pop_size, n_gen, initial_x, seed)
    if spec.runner == "local_oade":
        return run_oade(problem, pop_size, n_gen, initial_x, seed, spec.params)
    if spec.runner == "local_cgde_nsga2":
        return run_cgde_nsga2(spec, problem, pop_size, n_gen, initial_x, seed)
    if spec.runner == "local_nsga2_sdr_ols":
        return run_nsga2_sdr_ols(spec, problem, pop_size, n_gen, initial_x, seed)
    if spec.runner == "local_ce_moea":
        return run_ce_moea(spec, problem, pop_size, n_gen, initial_x, seed)
    raise ValueError(f"Unsupported runner for {spec.name}: {spec.runner}")


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


def run_one_problem(problem_name: str, n_var: int | None, cfg: dict, specs: list[AlgorithmSpec]) -> tuple[list[dict], list[dict]]:
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

    metrics: dict[str, list[RunMetrics]] = {spec.name: [] for spec in specs}
    raw_rows: list[dict] = []

    for run_idx in range(runs):
        seed = seed_base + run_idx
        initial_x = make_initial_population(problem, pop_size, seed)

        for spec in specs:
            front = run_algorithm(spec, problem, pop_size, n_gen, initial_x, seed)
            metric = RunMetrics(igd=float(igd_indicator(front)), hv=float(hv_indicator(front)))
            metrics[spec.name].append(metric)
            raw_rows.append(
                {
                    "problem": problem_name,
                    "n_var": n_var,
                    "run_idx": run_idx,
                    "seed": seed,
                    "algorithm": spec.name,
                    "runner": spec.runner,
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
            "algorithm": spec.name,
            "runner": spec.runner,
            **stats[spec.name],
        }
        for spec in specs
    ]
    return raw_rows, summary_rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args(argv)

    cfg = load_yaml_config(Path(args.config))
    specs = load_algorithm_specs([str(path) for path in cfg["algorithm_configs"]])
    out_dir = Path(cfg["global"]["out_dir"])
    all_raw_rows: list[dict] = []
    all_summary_rows: list[dict] = []

    for item in cfg["problems"]:
        raw_rows, summary_rows = run_one_problem(str(item["name"]), item.get("n_var"), cfg, specs)
        all_raw_rows.extend(raw_rows)
        all_summary_rows.extend(summary_rows)

    write_csv(out_dir / RAW_CSV, all_raw_rows, RAW_FIELDNAMES)
    write_csv(out_dir / SUMMARY_CSV, all_summary_rows, SUMMARY_FIELDNAMES)

    print(f"Saved: {(out_dir / RAW_CSV).as_posix()}")
    print(f"Saved: {(out_dir / SUMMARY_CSV).as_posix()}")
    print("Plot with: python main.py plot-algorithm-benchmark")


if __name__ == "__main__":
    main()
