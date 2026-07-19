"""Continuous Encoding MOEA based on NSGA-II and DE."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .nsga2 import binary_tournament_selection, environmental_selection, rank_and_crowding
from .operators import differential_evolution_operator


class CEMOEA:
    """
    CE-MOEA for continuous encoded community detection.

    The problem object must expose n_var, n_obj, xl, xu, and evaluate(X). Use
    AttributedNetworkProblem for the graph neural network encoding described
    in the paper.
    """

    def __init__(
        self,
        problem,
        pop_size: int = 100,
        n_gen: int = 200,
        F_DE: float = 0.7,
        CR: float = 0.5,
        p_m: float = 0.02,
        eta_m: float = 20.0,
        seed: int | None = None,
    ) -> None:
        self.problem = problem
        self.pop_size = int(pop_size)
        self.n_gen = int(n_gen)
        self.F_DE = float(F_DE)
        self.CR = float(CR)
        self.p_m = float(p_m)
        self.eta_m = float(eta_m)
        self.rng = np.random.default_rng(seed)

        self.population: np.ndarray | None = None
        self.objectives: np.ndarray | None = None
        self.ranks: np.ndarray | None = None
        self.crowding: np.ndarray | None = None
        self.history: list[np.ndarray] = []

    def run(
        self,
        initial_x: np.ndarray | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if initial_x is None:
            P = self.rng.uniform(self.problem.xl, self.problem.xu, size=(self.pop_size, self.problem.n_var))
        else:
            P = np.clip(np.asarray(initial_x, dtype=float), self.problem.xl, self.problem.xu)
            if P.ndim == 1:
                P = P.reshape(1, -1)
            if P.shape[1] != self.problem.n_var:
                raise ValueError(f"expected {self.problem.n_var} variables, got {P.shape[1]}")
            if len(P) != self.pop_size:
                P = self._resize_initial_population(P)

        F = self.problem.evaluate(P)
        ranks, crowding, _ = rank_and_crowding(F)
        self.history.clear()

        for generation in range(1, self.n_gen):
            parents = binary_tournament_selection(P, ranks, crowding, self.rng, size=self.pop_size)
            offspring = self._make_offspring(parents)
            offspring_F = self.problem.evaluate(offspring)

            combined_P = np.vstack([P, offspring])
            combined_F = np.vstack([F, offspring_F])
            P, F, ranks, crowding = environmental_selection(combined_P, combined_F, self.pop_size)

            self.history.append(F.copy())
            if progress_callback is not None:
                progress_callback(generation + 1, self.n_gen)

        self.population = P
        self.objectives = F
        self.ranks = ranks
        self.crowding = crowding
        return P, F

    def pareto_set(self) -> np.ndarray:
        self._require_run()
        return self.population.copy()

    def pareto_front(self) -> np.ndarray:
        self._require_run()
        return self.objectives.copy()

    def decoded_partitions(self) -> list:
        self._require_run()
        if not hasattr(self.problem, "decode"):
            raise TypeError("problem does not provide a decode(x) method")
        return [self.problem.decode(x) for x in self.population]

    def _make_offspring(self, parents: np.ndarray) -> np.ndarray:
        offspring = []
        n = len(parents)
        for j in range(n):
            x1 = parents[j]
            candidates = [idx for idx in range(n) if idx != j]
            replace = len(candidates) < 2
            chosen = self.rng.choice(candidates if candidates else list(range(n)), size=2, replace=replace)
            x2 = parents[int(chosen[0])]
            x3 = parents[int(chosen[1])]
            child = differential_evolution_operator(
                x1,
                x2,
                x3,
                self.F_DE,
                self.CR,
                self.p_m,
                self.eta_m,
                self.problem.xl,
                self.problem.xu,
                self.rng,
            )
            offspring.append(child)
        return np.asarray(offspring, dtype=float)

    def _resize_initial_population(self, P: np.ndarray) -> np.ndarray:
        if len(P) > self.pop_size:
            return P[: self.pop_size].copy()
        missing = self.pop_size - len(P)
        extra = self.rng.uniform(self.problem.xl, self.problem.xu, size=(missing, self.problem.n_var))
        return np.vstack([P, extra])

    def _require_run(self) -> None:
        if self.population is None or self.objectives is None:
            raise RuntimeError("run() must be called first")

