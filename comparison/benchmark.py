"""
Run multi-objective benchmark experiments on ZDT problems using pymoo baselines
and the existing OADE_NSGA2 implementation.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.termination import get_termination
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from algorithm_src.algorithm import MechanismConfig, NSGA2ImprovedSmart
from algorithm_src.core import ProblemWrapper


DEFAULT_PROBLEM_NAMES: Sequence[str] = ("zdt1", "zdt2", "zdt3", "zdt4", "zdt6")
DEFAULT_SEEDS: Sequence[int] = tuple(range(30))
DEFAULT_POP_SIZE: int = 100
DEFAULT_N_GEN: int = 200
DEFAULT_OUTPUT_DIR: str = "outputs"
DEFAULT_CSV_PATH: str = "outputs/benchmark_results.csv"
DEFAULT_SUMMARY_CSV_PATH: str = "outputs/benchmark_summary.csv"
DEFAULT_ALGORITHMS: Sequence[str] = (
    "OADE_NSGA2",
    "pymoo_nsga2",
    "pymoo_rnsga2",
    "pymoo_dnsga2",
    "pymoo_mopso",
    "pymoo_moead",
    "pymoo_sms_emoa",
)
DEFAULT_ABLATION_ALGORITHMS: Sequence[str] = (
    "OADE_NSGA2_ablation_sbx_only",
    "OADE_NSGA2_ablation_no_adaptive_de",
    "OADE_NSGA2_ablation_no_obl_init",
    "OADE_NSGA2_ablation_no_periodic_obl",
    "OADE_NSGA2_ablation_no_restart",
)
DEFAULT_INCREMENTAL_ALGORITHMS: Sequence[str] = (
    "OADE_NSGA2_incremental_baseline",
    "OADE_NSGA2_incremental_plus_obl_init",
    "OADE_NSGA2_incremental_plus_de_fixed",
    "OADE_NSGA2_incremental_plus_de_adaptive",
    "OADE_NSGA2_incremental_plus_periodic_obl",
    "OADE_NSGA2_incremental_plus_restart",
)

DEFAULT_PROBLEM_N_VARS: Dict[str, int] = {
    "zdt1": 30,
    "zdt2": 30,
    "zdt3": 30,
    "zdt4": 10,
    "zdt6": 10,
}


class AlgorithmUnavailableError(RuntimeError):
    """Raised when an algorithm cannot be instantiated in the current environment."""


@dataclass(frozen=True)
class BenchmarkResult:
    problem_name: str
    algorithm_name: str
    n_var: int
    seed: int
    runtime_seconds: float
    front: np.ndarray
    population: np.ndarray
    ideal_point: np.ndarray
    nadir_point: np.ndarray
    hv: float
    igd: float
    igd_plus: float
    epsilon_additive: float
    spacing: float

    @property
    def front_size(self) -> int:
        return int(len(self.front))


@dataclass(frozen=True)
class PymooAlgorithmSpec:
    name: str
    factory: Callable[[Any, int], Any]


def get_benchmark_problem(name: str, n_var: Optional[int] = None) -> Any:
    if n_var is None:
        n_var = DEFAULT_PROBLEM_N_VARS.get(name.lower(), 30)
    return get_problem(name, n_var=n_var)


def get_default_benchmarks() -> Dict[str, Any]:
    return {
        name: get_benchmark_problem(name, n_var=DEFAULT_PROBLEM_N_VARS.get(name, 30))
        for name in DEFAULT_PROBLEM_NAMES
    }


def _build_reference_points(n_obj: int) -> np.ndarray:
    eye = np.eye(n_obj, dtype=float)
    center = np.full((1, n_obj), 1.0 / max(1, n_obj), dtype=float)
    return np.vstack([eye, center])


def _make_pymoo_nsga2(_: Any, pop_size: int) -> Any:
    return NSGA2(pop_size=pop_size)


def _make_pymoo_rnsga2(problem: Any, pop_size: int) -> Any:
    try:
        from pymoo.algorithms.moo.rnsga2 import RNSGA2
    except ImportError as exc:
        raise AlgorithmUnavailableError("RNSGA2 is not available in this pymoo version") from exc

    ref_points = _build_reference_points(int(problem.n_obj))
    try:
        return RNSGA2(ref_points=ref_points, pop_size=pop_size)
    except TypeError:
        return RNSGA2(pop_size=pop_size)


def _make_pymoo_dnsga2(_: Any, pop_size: int) -> Any:
    try:
        from pymoo.algorithms.moo.dnsga2 import DNSGA2
    except ImportError as exc:
        raise AlgorithmUnavailableError("DNSGA2 is not available in this pymoo version") from exc

    try:
        return DNSGA2(pop_size=pop_size)
    except TypeError:
        return DNSGA2(version="A", pop_size=pop_size)


def _make_pymoo_mopso(_: Any, pop_size: int) -> Any:
    # Prefer OMOPSO when available. Fallback to SMPSO as a close MOPSO-family baseline.
    try:
        from pymoo.algorithms.moo.omopso import OMOPSO

        return OMOPSO(pop_size=pop_size)
    except ImportError:
        pass

    try:
        from pymoo.algorithms.moo.smpso import SMPSO

        return SMPSO(pop_size=pop_size)
    except ImportError as exc:
        raise AlgorithmUnavailableError("Neither OMOPSO nor SMPSO is available") from exc


def _make_pymoo_moead(problem: Any, pop_size: int) -> Any:
    try:
        from pymoo.algorithms.moo.moead import MOEAD
        from pymoo.util.ref_dirs import get_reference_directions
    except ImportError as exc:
        raise AlgorithmUnavailableError("MOEAD is not available in this pymoo version") from exc

    n_obj = int(problem.n_obj)
    # Keep MOEAD budget aligned with requested pop_size as closely as possible.
    try:
        ref_dirs = get_reference_directions("energy", n_obj, n_points=max(2, int(pop_size)), seed=1)
    except Exception:
        if n_obj == 2:
            ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=max(1, int(pop_size) - 1))
        else:
            ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=max(12, n_obj * 6))
    neighborhood = min(20, max(2, len(ref_dirs) - 1))

    try:
        return MOEAD(ref_dirs=ref_dirs, n_neighbors=neighborhood, prob_neighbor_mating=0.9)
    except TypeError:
        return MOEAD(ref_dirs=ref_dirs)


def _make_pymoo_sms_emoa(_: Any, pop_size: int) -> Any:
    try:
        from pymoo.algorithms.moo.sms import SMSEMOA
    except ImportError as exc:
        raise AlgorithmUnavailableError("SMSEMOA is not available in this pymoo version") from exc

    return SMSEMOA(pop_size=pop_size)


def get_pymoo_algorithm_specs() -> Dict[str, PymooAlgorithmSpec]:
    specs = [
        PymooAlgorithmSpec("pymoo_nsga2", _make_pymoo_nsga2),
        PymooAlgorithmSpec("pymoo_rnsga2", _make_pymoo_rnsga2),
        PymooAlgorithmSpec("pymoo_dnsga2", _make_pymoo_dnsga2),
        PymooAlgorithmSpec("pymoo_mopso", _make_pymoo_mopso),
        PymooAlgorithmSpec("pymoo_moead", _make_pymoo_moead),
        PymooAlgorithmSpec("pymoo_sms_emoa", _make_pymoo_sms_emoa),
    ]
    return {spec.name: spec for spec in specs}


def _run_pymoo_algorithm(
    problem: Any,
    seed: int,
    pop_size: int,
    n_gen: int,
    factory: Callable[[Any, int], Any],
) -> tuple[np.ndarray, np.ndarray]:
    algorithm = factory(problem, pop_size)
    result = minimize(
        problem,
        algorithm,
        get_termination("n_gen", n_gen),
        seed=seed,
        verbose=False,
    )

    population = _extract_population_from_result(result)
    front = _extract_front(population)
    return front, population


def _extract_population_from_result(result: Any) -> np.ndarray:
    if getattr(result, "pop", None) is not None:
        try:
            pop_f = result.pop.get("F")
            population = np.asarray(pop_f)
            if population.size > 0:
                if population.ndim == 1:
                    return population.reshape(1, -1)
                return population
        except Exception as exc:
            raise RuntimeError("Unable to extract population objectives from pymoo result") from exc

    raise RuntimeError(
        "pymoo result does not expose final population (result.pop). "
        "Cannot ensure fair comparison on a shared population semantic."
    )


def run_OADE_NSGA2(
    problem: Any,
    seed: int,
    pop_size: int,
    n_gen: int,
    mechanism_config: Optional[MechanismConfig] = None,
) -> tuple[np.ndarray, np.ndarray]:
    np.random.seed(seed)
    random.seed(seed)

    wrapped_problem = ProblemWrapper(problem)
    solver = NSGA2ImprovedSmart(
        wrapped_problem,
        pop_size=pop_size,
        n_gen=n_gen,
        mechanism_config=mechanism_config,
    )
    population = solver.run()
    front = _extract_front(population)
    return front, population


def get_oade_ablation_configs() -> Dict[str, MechanismConfig]:
    return {
        "OADE_NSGA2": MechanismConfig(),
        "OADE_NSGA2_ablation_sbx_only": MechanismConfig(
            use_de_operator=False,
            use_adaptive_de=False,
        ),
        "OADE_NSGA2_ablation_no_adaptive_de": MechanismConfig(
            use_adaptive_de=False,
        ),
        "OADE_NSGA2_ablation_no_obl_init": MechanismConfig(
            use_obl_init=False,
        ),
        "OADE_NSGA2_ablation_no_periodic_obl": MechanismConfig(
            use_periodic_obl_injection=False,
        ),
        "OADE_NSGA2_ablation_no_restart": MechanismConfig(
            use_stagnation_restart=False,
        ),
    }


def get_oade_incremental_configs() -> Dict[str, MechanismConfig]:
    """
    Incremental variants from a NSGA2-style internal baseline.

    Baseline here means all extra mechanisms are off:
    - random init (no OBL init)
    - SBX only (no DE)
    - no adaptive DE
    - no periodic OBL injection
    - no stagnation restart
    """
    baseline = MechanismConfig(
        use_obl_init=False,
        use_de_operator=False,
        use_adaptive_de=False,
        use_periodic_obl_injection=False,
        use_stagnation_restart=False,
    )

    return {
        "OADE_NSGA2_incremental_baseline": baseline,
        "OADE_NSGA2_incremental_plus_obl_init": MechanismConfig(
            use_obl_init=True,
            use_de_operator=False,
            use_adaptive_de=False,
            use_periodic_obl_injection=False,
            use_stagnation_restart=False,
        ),
        "OADE_NSGA2_incremental_plus_de_fixed": MechanismConfig(
            use_obl_init=False,
            use_de_operator=True,
            use_adaptive_de=False,
            use_periodic_obl_injection=False,
            use_stagnation_restart=False,
        ),
        "OADE_NSGA2_incremental_plus_de_adaptive": MechanismConfig(
            use_obl_init=False,
            use_de_operator=True,
            use_adaptive_de=True,
            use_periodic_obl_injection=False,
            use_stagnation_restart=False,
        ),
        "OADE_NSGA2_incremental_plus_periodic_obl": MechanismConfig(
            use_obl_init=False,
            use_de_operator=False,
            use_adaptive_de=False,
            use_periodic_obl_injection=True,
            use_stagnation_restart=False,
        ),
        "OADE_NSGA2_incremental_plus_restart": MechanismConfig(
            use_obl_init=False,
            use_de_operator=False,
            use_adaptive_de=False,
            use_periodic_obl_injection=False,
            use_stagnation_restart=True,
        ),
    }


def build_algorithm_runners(selected_names: Optional[Sequence[str]] = None) -> Dict[str, Callable[..., tuple[np.ndarray, np.ndarray]]]:
    pymoo_specs = get_pymoo_algorithm_specs()
    ablation_configs = get_oade_ablation_configs()
    incremental_configs = get_oade_incremental_configs()

    runners: Dict[str, Callable[..., tuple[np.ndarray, np.ndarray]]] = {}

    for ablation_name, ablation_cfg in ablation_configs.items():
        def _ablation_runner(
            problem: Any,
            seed: int,
            pop_size: int,
            n_gen: int,
            _cfg=ablation_cfg,
        ) -> tuple[np.ndarray, np.ndarray]:
            return run_OADE_NSGA2(
                problem=problem,
                seed=seed,
                pop_size=pop_size,
                n_gen=n_gen,
                mechanism_config=_cfg,
            )

        runners[ablation_name] = _ablation_runner

    for incremental_name, incremental_cfg in incremental_configs.items():
        def _incremental_runner(
            problem: Any,
            seed: int,
            pop_size: int,
            n_gen: int,
            _cfg=incremental_cfg,
        ) -> tuple[np.ndarray, np.ndarray]:
            return run_OADE_NSGA2(
                problem=problem,
                seed=seed,
                pop_size=pop_size,
                n_gen=n_gen,
                mechanism_config=_cfg,
            )

        runners[incremental_name] = _incremental_runner

    for name, spec in pymoo_specs.items():
        def _runner(problem: Any, seed: int, pop_size: int, n_gen: int, _factory=spec.factory) -> tuple[np.ndarray, np.ndarray]:
            return _run_pymoo_algorithm(problem, seed, pop_size, n_gen, _factory)

        runners[name] = _runner

    if selected_names is None:
        selected_names = DEFAULT_ALGORITHMS

    available = sorted(runners.keys())
    unknown = [name for name in selected_names if name not in runners]
    if unknown:
        raise ValueError(f"Unknown algorithm(s): {unknown}. Available: {available}")

    return {name: runners[name] for name in selected_names}


def run_benchmark_suite(
    problems: Optional[Mapping[str, Any]] = None,
    problem_n_vars: Optional[Mapping[str, int]] = None,
    algorithms: Optional[Mapping[str, Callable[..., tuple[np.ndarray, np.ndarray]]]] = None,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    pop_size: int = DEFAULT_POP_SIZE,
    n_gen: int = DEFAULT_N_GEN,
) -> List[BenchmarkResult]:
    problem_map = dict(problems or get_default_benchmarks())
    n_var_map = dict(problem_n_vars or DEFAULT_PROBLEM_N_VARS)
    algorithm_map = dict(algorithms or build_algorithm_runners())

    results: List[BenchmarkResult] = []
    skipped: List[str] = []

    for problem_name, problem in problem_map.items():
        for algorithm_name, algorithm_runner in algorithm_map.items():
            skip_current_algorithm = False
            for seed in seeds:
                try:
                    result = run_single_benchmark(
                        problem=problem,
                        algorithm_runner=algorithm_runner,
                        algorithm_name=algorithm_name,
                        n_var=int(getattr(problem, "n_var", n_var_map.get(problem_name, 30))),
                        seed=seed,
                        pop_size=pop_size,
                        n_gen=n_gen,
                        problem_name=problem_name,
                    )
                    results.append(result)
                except AlgorithmUnavailableError as exc:
                    message = f"Skipped {algorithm_name} on {problem_name}: {exc}"
                    if message not in skipped:
                        skipped.append(message)
                    skip_current_algorithm = True
                    break
            if skip_current_algorithm:
                continue

    for item in skipped:
        print(item)

    if not results:
        raise RuntimeError("No benchmark results were generated. Check algorithm availability and configuration.")

    return results


def run_single_benchmark(
    problem: Any,
    algorithm_runner: Callable[..., tuple[np.ndarray, np.ndarray]],
    algorithm_name: str,
    n_var: int,
    seed: int,
    pop_size: int,
    n_gen: int,
    problem_name: Optional[str] = None,
) -> BenchmarkResult:
    start = time.perf_counter()
    front, population = algorithm_runner(
        problem=problem,
        seed=seed,
        pop_size=pop_size,
        n_gen=n_gen,
    )
    runtime_seconds = time.perf_counter() - start

    true_pf = _get_true_pareto_front(problem)
    ref_point = _get_reference_point(true_pf)

    if front.size == 0:
        hv_value = 0.0
        igd_value = float("inf")
        igd_plus_value = float("inf")
        epsilon_additive_value = float("inf")
        spacing_value = float("inf")
        ideal_point = np.full(problem.n_obj, np.nan)
        nadir_point = np.full(problem.n_obj, np.nan)
    else:
        hv_value = float(HV(ref_point=ref_point)(front))
        igd_value = float(IGD(true_pf)(front))
        igd_plus_value = _compute_igd_plus(true_pf, front)
        epsilon_additive_value = _compute_additive_epsilon(true_pf, front)
        spacing_value = _compute_spacing(front)
        ideal_point = front.min(axis=0)
        nadir_point = front.max(axis=0)

    return BenchmarkResult(
        problem_name=problem_name or getattr(problem, "name", "unknown_problem"),
        algorithm_name=algorithm_name,
        n_var=n_var,
        seed=seed,
        runtime_seconds=runtime_seconds,
        front=front,
        population=population,
        ideal_point=ideal_point,
        nadir_point=nadir_point,
        hv=hv_value,
        igd=igd_value,
        igd_plus=igd_plus_value,
        epsilon_additive=epsilon_additive_value,
        spacing=spacing_value,
    )


def summarize_results(results: Sequence[BenchmarkResult]) -> List[Dict[str, object]]:
    summary: List[Dict[str, object]] = []
    for item in results:
        summary.append(
            {
                "problem": item.problem_name,
                "algorithm": item.algorithm_name,
                "n_var": item.n_var,
                "seed": item.seed,
                "runtime_seconds": item.runtime_seconds,
                "front_size": item.front_size,
                "hv": item.hv,
                "igd": item.igd,
                "igd_plus": item.igd_plus,
                "epsilon_additive": item.epsilon_additive,
                "spacing": item.spacing,
                "ideal_point": item.ideal_point.tolist(),
                "nadir_point": item.nadir_point.tolist(),
                "front": item.front.tolist(),
            }
        )
    return summary


def export_results_to_csv(results: Sequence[BenchmarkResult], csv_path: str) -> None:
    summary = summarize_results(results)
    if not summary:
        return

    fieldnames = list(summary[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, tuple, dict)) else value
                    for key, value in row.items()
                }
            )


def export_summary_to_csv(results: Sequence[BenchmarkResult], csv_path: str) -> None:
    summary_rows = summarize_results(results)
    if not summary_rows:
        return

    grouped: Dict[tuple[str, str], List[Dict[str, object]]] = {}
    for row in summary_rows:
        key = (str(row["problem"]), str(row["algorithm"]))
        grouped.setdefault(key, []).append(row)

    rows_to_write: List[Dict[str, object]] = []
    for (problem, algorithm), rows in grouped.items():
        hv_values = np.array([float(r["hv"]) for r in rows], dtype=float)
        igd_values = np.array([float(r["igd"]) for r in rows], dtype=float)
        igd_plus_values = np.array([float(r["igd_plus"]) for r in rows], dtype=float)
        epsilon_values = np.array([float(r["epsilon_additive"]) for r in rows], dtype=float)
        spacing_values = np.array([float(r["spacing"]) for r in rows], dtype=float)
        runtime_values = np.array([float(r["runtime_seconds"]) for r in rows], dtype=float)
        front_sizes = np.array([float(r["front_size"]) for r in rows], dtype=float)
        n_var_values = np.array([float(r["n_var"]) for r in rows], dtype=float)

        rows_to_write.append(
            {
                "problem": problem,
                "algorithm": algorithm,
                "n_var": int(np.round(np.mean(n_var_values))),
                "runs": len(rows),
                "runtime_mean": float(np.mean(runtime_values)),
                "runtime_std": float(np.std(runtime_values, ddof=0)),
                "front_size_mean": float(np.mean(front_sizes)),
                "front_size_std": float(np.std(front_sizes, ddof=0)),
                "hv_mean": float(np.mean(hv_values)),
                "hv_std": float(np.std(hv_values, ddof=0)),
                "hv_median": float(np.median(hv_values)),
                "hv_iqr": _iqr(hv_values),
                "igd_mean": float(np.mean(igd_values)),
                "igd_std": float(np.std(igd_values, ddof=0)),
                "igd_median": float(np.median(igd_values)),
                "igd_iqr": _iqr(igd_values),
                "igd_plus_mean": float(np.mean(igd_plus_values)),
                "igd_plus_std": float(np.std(igd_plus_values, ddof=0)),
                "igd_plus_median": float(np.median(igd_plus_values)),
                "igd_plus_iqr": _iqr(igd_plus_values),
                "epsilon_additive_mean": float(np.mean(epsilon_values)),
                "epsilon_additive_std": float(np.std(epsilon_values, ddof=0)),
                "epsilon_additive_median": float(np.median(epsilon_values)),
                "epsilon_additive_iqr": _iqr(epsilon_values),
                "spacing_mean": float(np.mean(spacing_values)),
                "spacing_std": float(np.std(spacing_values, ddof=0)),
                "spacing_median": float(np.median(spacing_values)),
                "spacing_iqr": _iqr(spacing_values),
            }
        )

    fieldnames = list(rows_to_write[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_to_write)


def run_configured_comparison(
    problem_names: Sequence[str] = DEFAULT_PROBLEM_NAMES,
    algorithms: Sequence[str] = DEFAULT_ALGORITHMS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    pop_size: int = DEFAULT_POP_SIZE,
    n_gen: int = DEFAULT_N_GEN,
    csv_path: str = DEFAULT_CSV_PATH,
    summary_csv_path: str = DEFAULT_SUMMARY_CSV_PATH,
    problem_n_vars: Optional[Mapping[str, int]] = None,
) -> List[BenchmarkResult]:
    if pop_size <= 0:
        raise ValueError(f"pop_size must be > 0, got {pop_size}")
    if n_gen <= 0:
        raise ValueError(f"n_gen must be > 0, got {n_gen}")
    if not seeds:
        raise ValueError("seeds must not be empty")

    n_var_map = dict(problem_n_vars or DEFAULT_PROBLEM_N_VARS)
    problems = {
        name: get_benchmark_problem(name, n_var=n_var_map.get(name, 30))
        for name in problem_names
    }
    algorithm_runners = build_algorithm_runners(algorithms)

    results = run_benchmark_suite(
        problems=problems,
        problem_n_vars=n_var_map,
        algorithms=algorithm_runners,
        seeds=seeds,
        pop_size=pop_size,
        n_gen=n_gen,
    )

    for row in summarize_results(results):
        print(
            f"{row['problem']} | {row['algorithm']} | seed={row['seed']} | "
            f"runtime={row['runtime_seconds']:.4f}s | front_size={row['front_size']} | "
            f"HV={row['hv']:.6f} | IGD={row['igd']:.6f}"
        )

    output_dir = Path(DEFAULT_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    if csv_path:
        export_results_to_csv(results, csv_path)
        print(f"CSV saved to: {csv_path}")

    if summary_csv_path:
        export_summary_to_csv(results, summary_csv_path)
        print(f"Summary CSV saved to: {summary_csv_path}")

    return results


def _extract_front(population: np.ndarray) -> np.ndarray:
    if population.size == 0:
        return population

    front_indices = NonDominatedSorting().do(population, only_non_dominated_front=True)
    front = population[front_indices]
    if front.ndim == 1:
        front = front.reshape(1, -1)
    return front


def _get_true_pareto_front(problem: Any) -> np.ndarray:
    true_pf = problem.pareto_front()
    if true_pf is None:
        raise ValueError(f"Problem {getattr(problem, 'name', '<unknown>')} does not provide a Pareto front")

    true_pf = np.asarray(true_pf)
    if true_pf.ndim == 1:
        true_pf = true_pf.reshape(1, -1)
    return true_pf


def _get_reference_point(true_pf: np.ndarray) -> np.ndarray:
    pf_max = np.max(true_pf, axis=0)
    pf_min = np.min(true_pf, axis=0)
    spread = np.where(pf_max - pf_min == 0.0, 1.0, pf_max - pf_min)
    return pf_max + 0.1 * spread + 1e-6


def _compute_igd_plus(true_pf: np.ndarray, front: np.ndarray) -> float:
    """
    IGD+ variant where only positive deviations are penalized.
    Smaller is better.
    """
    distances = []
    for z in true_pf:
        # max(front - z, 0) ensures only dominated-side errors are counted.
        shifted = np.maximum(front - z, 0.0)
        d = np.sqrt(np.sum(shifted * shifted, axis=1))
        distances.append(np.min(d))
    return float(np.mean(distances)) if distances else float("inf")


def _compute_additive_epsilon(true_pf: np.ndarray, front: np.ndarray) -> float:
    """
    Additive epsilon indicator for minimization problems.
    Smaller is better.
    """
    eps_values = []
    for z in true_pf:
        eps_to_front = np.max(front - z, axis=1)
        eps_values.append(np.min(eps_to_front))
    return float(np.max(eps_values)) if eps_values else float("inf")


def _compute_spacing(front: np.ndarray) -> float:
    """
    Spacing metric based on nearest-neighbor distances in objective space.
    Smaller is better (more uniform spread).
    """
    if len(front) <= 2:
        return 0.0

    diff = front[:, None, :] - front[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    nearest = np.min(dist, axis=1)
    return float(np.std(nearest, ddof=0))


def _iqr(values: np.ndarray) -> float:
    q75, q25 = np.percentile(values, [75, 25])
    return float(q75 - q25)


def _parse_csv_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_csv_int_list(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _validate_cli_inputs(problem_names: Sequence[str], algorithm_names: Sequence[str], seeds: Sequence[int]) -> None:
    if not problem_names:
        raise ValueError("--problems cannot be empty")
    if not algorithm_names:
        raise ValueError("--algorithms cannot be empty")
    if not seeds:
        raise ValueError("--seeds cannot be empty")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ZDT benchmark comparison for multiple algorithms")
    parser.add_argument(
        "--problems",
        default=",".join(DEFAULT_PROBLEM_NAMES),
        help="Comma-separated benchmark names, for example zdt1,zdt2,zdt3,zdt4,zdt6",
    )
    parser.add_argument(
        "--algorithms",
        default=",".join(DEFAULT_ALGORITHMS),
        help=(
            "Comma-separated algorithm names. Available: "
            "OADE_NSGA2,pymoo_nsga2,pymoo_rnsga2,pymoo_dnsga2,pymoo_mopso,pymoo_moead,pymoo_sms_emoa"
        ),
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(s) for s in DEFAULT_SEEDS),
        help="Comma-separated seed list, for example 0,1,2,3,4",
    )
    parser.add_argument("--pop-size", type=int, default=DEFAULT_POP_SIZE, help="Population size")
    parser.add_argument("--n-gen", type=int, default=DEFAULT_N_GEN, help="Number of generations")
    parser.add_argument("--csv", default=DEFAULT_CSV_PATH, help="Path to detailed CSV output")
    parser.add_argument("--summary-csv", default=DEFAULT_SUMMARY_CSV_PATH, help="Path to summary CSV output")
    parser.add_argument("--list-algorithms", action="store_true", help="Print supported algorithm names and exit")
    parser.add_argument("--list-ablations", action="store_true", help="Print ablation variants for OADE_NSGA2 and exit")
    parser.add_argument("--list-incrementals", action="store_true", help="Print incremental variants from baseline and exit")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.list_algorithms:
        print("Supported algorithm names:")
        for name in sorted(build_algorithm_runners().keys()):
            print(f"- {name}")
        return

    if args.list_ablations:
        print("OADE_NSGA2 ablation variants:")
        for name in sorted(get_oade_ablation_configs().keys()):
            print(f"- {name}")
        return

    if args.list_incrementals:
        print("OADE_NSGA2 incremental variants from baseline:")
        for name in sorted(get_oade_incremental_configs().keys()):
            print(f"- {name}")
        return

    problem_names = _parse_csv_list(args.problems)
    algorithm_names = _parse_csv_list(args.algorithms)
    seeds = _parse_csv_int_list(args.seeds)
    _validate_cli_inputs(problem_names, algorithm_names, seeds)

    run_configured_comparison(
        problem_names=problem_names,
        algorithms=algorithm_names,
        seeds=seeds,
        pop_size=args.pop_size,
        n_gen=args.n_gen,
        csv_path=args.csv,
        summary_csv_path=args.summary_csv,
    )


if __name__ == "__main__":
    main()


