"""Single command entry point for the project."""

from __future__ import annotations

import argparse
import sys
from typing import Callable

from src.benchmarking import algorithm_benchmark, pymoo_compare
from src.plotting import algorithm_benchmark as algorithm_benchmark_plots
from src.plotting import all_problems, parameter_sensitivity, wilcoxon_results
from src.sensitivity import parameter_analysis


Command = Callable[[list[str] | None], None]


COMMANDS: dict[str, Command] = {
    "benchmark": algorithm_benchmark.main,
    "benchmark-algorithms": algorithm_benchmark.main,
    "benchmark-three-algorithms": algorithm_benchmark.main,
    "benchmark-legacy": pymoo_compare.main,
    "plot-all": all_problems.main,
    "plot-algorithm-benchmark": algorithm_benchmark_plots.main,
    "plot-three-algorithms": algorithm_benchmark_plots.main,
    "plot-wilcoxon": wilcoxon_results.main,
    "sensitivity": parameter_analysis.main,
    "plot-sensitivity": parameter_sensitivity.main,
}


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description="OADE-NSGA-II project runner")
    parser.add_argument("command", choices=COMMANDS)

    if not argv or argv[0] in {"-h", "--help"}:
        parser.parse_args(argv)
        return

    command = argv[0]
    if command not in COMMANDS:
        parser.parse_args([command])
        return

    COMMANDS[command](argv[1:])


if __name__ == "__main__":
    main()
