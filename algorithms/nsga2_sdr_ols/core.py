"""Core data structures for the standalone NSGA-II/SDR-OLS implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Individual:
    """One candidate solution."""

    X: np.ndarray
    F: np.ndarray | None = None
    rank: int | None = None
    crowding_dist: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def copy(self) -> "Individual":
        return Individual(
            X=self.X.copy(),
            F=None if self.F is None else self.F.copy(),
            rank=self.rank,
            crowding_dist=self.crowding_dist,
            metadata=dict(self.metadata),
        )


class ProblemAdapter:
    """
    Small adapter for pymoo-like minimization problems.

    The wrapped object must expose n_var, n_obj, xl, xu, and evaluate(X).
    """

    def __init__(self, problem: Any) -> None:
        self.problem = problem
        self.n_var = int(problem.n_var)
        self.n_obj = int(problem.n_obj)
        self.xl = np.asarray(problem.xl, dtype=float)
        self.xu = np.asarray(problem.xu, dtype=float)

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        values = np.asarray(self.problem.evaluate(np.asarray(X, dtype=float)), dtype=float)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        return values


def make_individual(x: np.ndarray, f: np.ndarray | None = None, **metadata: Any) -> Individual:
    ind = Individual(np.asarray(x, dtype=float).copy())
    if f is not None:
        ind.F = np.asarray(f, dtype=float).reshape(-1).copy()
    ind.metadata.update(metadata)
    return ind


def evaluate_unevaluated(problem: ProblemAdapter, population: list[Individual]) -> None:
    unevaluated = [ind for ind in population if ind.F is None]
    if not unevaluated:
        return

    X = np.asarray([ind.X for ind in unevaluated], dtype=float)
    F = problem.evaluate(X)
    for ind, f in zip(unevaluated, F):
        ind.F = np.asarray(f, dtype=float).reshape(-1).copy()

