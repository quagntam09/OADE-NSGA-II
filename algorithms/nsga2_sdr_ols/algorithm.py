"""Standalone NSGA-II/SDR-OLS algorithm."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .core import Individual, ProblemAdapter, evaluate_unevaluated, make_individual
from .operators import (
    local_search,
    make_ga_offspring,
    opposition_based_learning,
    random_population,
)
from .sdr import environmental_selection_sdr, fast_non_dominated_sort_sdr


class NSGAIISDROLS:
    """
    NSGA-II/SDR-OLS from Zhang, Wang, and Wang (Mathematics 2023).

    This implementation is standalone and does not depend on the existing
    src/oade_nsga2 package. All objectives are treated as minimization objectives.
    """

    def __init__(
        self,
        problem: ProblemAdapter,
        pop_size: int = 100,
        n_gen: int = 100,
        mu: float = 0.0,
        sigma: float = 0.1,
        seed: int | None = None,
        local_search_max_neighbors: int | None = None,
        crossover_prob: float = 0.9,
        mutation_prob: float | None = None,
        eta_c: float = 20.0,
        eta_m: float = 20.0,
    ) -> None:
        self.problem = problem
        self.pop_size = int(pop_size)
        self.n_gen = int(n_gen)
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.local_search_max_neighbors = local_search_max_neighbors
        self.crossover_prob = float(crossover_prob)
        self.mutation_prob = mutation_prob
        self.eta_c = float(eta_c)
        self.eta_m = float(eta_m)
        self.rng = np.random.default_rng(seed)

        self.population: list[Individual] = []
        self.last_combined_population: list[Individual] = []
        self.history: list[np.ndarray] = []

    def run(
        self,
        initial_x: np.ndarray | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[Individual]:
        """Run the algorithm and return the final selected population."""

        self.population = self._initial_population(initial_x)
        fast_non_dominated_sort_sdr(self.population)
        self.history.clear()

        for gen in range(self.n_gen):
            search_neighbors = local_search(
                self.problem,
                self.population,
                self.mu,
                self.sigma,
                self.rng,
                max_neighbors=self.local_search_max_neighbors,
            )

            combined = self.population + search_neighbors
            combined = environmental_selection_sdr(combined, max(self.pop_size, len(self.population)), self.problem.n_obj)

            offspring = make_ga_offspring(
                self.problem,
                combined,
                self.pop_size,
                self.rng,
                crossover_prob=self.crossover_prob,
                mutation_prob=self.mutation_prob,
                eta_c=self.eta_c,
                eta_m=self.eta_m,
            )

            self.last_combined_population = combined + offspring
            self.population = environmental_selection_sdr(
                self.last_combined_population,
                self.pop_size,
                self.problem.n_obj,
            )
            self.history.append(np.asarray([ind.F for ind in self.population], dtype=float))

            if progress_callback is not None:
                progress_callback(gen + 1, self.n_gen)

        return self.population

    def result_X(self) -> np.ndarray:
        return np.asarray([ind.X for ind in self.population], dtype=float)

    def result_F(self) -> np.ndarray:
        return np.asarray([ind.F for ind in self.population], dtype=float)

    def _initial_population(self, initial_x: np.ndarray | None) -> list[Individual]:
        if initial_x is None:
            population = random_population(self.problem, self.pop_size, self.rng)
        else:
            X = np.clip(np.asarray(initial_x, dtype=float), self.problem.xl, self.problem.xu)
            population = [make_individual(x, source="initial_x") for x in X]
            evaluate_unevaluated(self.problem, population)

        return opposition_based_learning(self.problem, population, self.pop_size)

