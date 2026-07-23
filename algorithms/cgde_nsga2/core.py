"""Core helpers for CGDE-NSGA-II."""

from __future__ import annotations

from typing import Any

import numpy as np


class ProblemAdapter:
    """Adapter for pymoo-like minimization problems."""

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


def resize_initial_population(
    initial_x: np.ndarray,
    pop_size: int,
    xl: np.ndarray,
    xu: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    X = np.clip(np.asarray(initial_x, dtype=float), xl, xu)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if len(X) > pop_size:
        return X[:pop_size].copy()
    if len(X) < pop_size:
        missing = pop_size - len(X)
        extra = rng.uniform(xl, xu, size=(missing, X.shape[1]))
        X = np.vstack([X, extra])
    return X

