"""
Triển khai NSGA-II cơ bản để làm mốc so sánh với bản cải tiến.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np

try:
    from .core import Individual, ProblemWrapper
    from .operators import sbx_crossover_mutation, tournament_selection
    from .selection import environmental_selection, remove_duplicates
except ImportError:  # pragma: no cover - fallback for direct script-style imports
    from core import Individual, ProblemWrapper
    from operators import sbx_crossover_mutation, tournament_selection
    from selection import environmental_selection, remove_duplicates


class NSGA2Baseline:
    """
    NSGA-II cơ bản theo luồng cổ điển:
    - Khởi tạo ngẫu nhiên trong miền tìm kiếm
    - Chọn lọc bằng tournament
    - Lai ghép SBX + polynomial mutation
    - Chọn lọc môi trường bằng non-dominated sorting + crowding distance

    Không dùng DE, OBL hay partial restart.
    """

    def __init__(self, problem: ProblemWrapper, pop_size: int = 100, n_gen: int = 100) -> None:
        self.problem = problem
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.n_var = problem.n_var
        self.n_obj = problem.n_obj
        self.xl = problem.xl
        self.xu = problem.xu

        self.pc = 0.9
        self.pm = 1.0 / (self.n_var * np.log(max(pop_size, 2)))
        self.eta_c = 20.0
        self.eta_m = 20.0

        self.population: List[Individual] = []
        self.history: List[np.ndarray] = []

    def run(
        self,
        initial_x: Optional[np.ndarray] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> np.ndarray:
        """
        Chạy NSGA-II cơ bản và trả về ma trận F của quần thể cuối.
        """
        self.population = self._build_initial_population(initial_x)
        self.history.clear()

        for gen in range(self.n_gen):
            offspring = self._generate_offspring()
            self._evaluate_unevaluated(offspring)

            combined = remove_duplicates(self.population + offspring)
            combined = self._fill_if_too_small(combined)
            self.population = environmental_selection(combined, self.pop_size, self.n_obj)
            self.history.append(np.array([ind.F for ind in self.population]))

            if progress_callback is not None:
                progress_callback(gen + 1, self.n_gen)

        return np.array([ind.F for ind in self.population])

    def _build_initial_population(self, initial_x: Optional[np.ndarray]) -> List[Individual]:
        if initial_x is not None:
            return self._evaluate_and_build(initial_x)

        x_init = self.xl + np.random.rand(self.pop_size, self.n_var) * (self.xu - self.xl)
        return self._evaluate_and_build(x_init)

    def _evaluate_and_build(self, x_batch: np.ndarray) -> List[Individual]:
        f_batch = self.problem.evaluate(x_batch)
        population: List[Individual] = []
        for x, f in zip(x_batch, f_batch):
            ind = Individual()
            ind.X = x.copy()
            ind.F = f.flatten().copy()
            population.append(ind)
        return environmental_selection(population, self.pop_size, self.n_obj)

    def _generate_offspring(self) -> List[Individual]:
        offspring: List[Individual] = []
        for _ in range(self.pop_size):
            parent1 = tournament_selection(self.population)
            parent2 = tournament_selection(self.population)
            offspring.append(
                sbx_crossover_mutation(
                    parent1,
                    parent2,
                    self.xl,
                    self.xu,
                    self.n_var,
                    self.pc,
                    self.pm,
                    self.eta_c,
                    self.eta_m,
                )
            )
        return offspring

    def _evaluate_unevaluated(self, offspring: List[Individual]) -> None:
        unevaluated = [ind for ind in offspring if ind.F is None]
        if not unevaluated:
            return

        x_batch = np.array([ind.X for ind in unevaluated])
        f_batch = self.problem.evaluate(x_batch)
        for ind, f in zip(unevaluated, f_batch):
            ind.F = f.flatten().copy()

    def _fill_if_too_small(self, population: List[Individual]) -> List[Individual]:
        missing = self.pop_size - len(population)
        if missing <= 0:
            return population

        x_fill = self.xl + np.random.rand(missing, self.n_var) * (self.xu - self.xl)
        f_fill = self.problem.evaluate(x_fill)
        extras: List[Individual] = []
        for x, f in zip(x_fill, f_fill):
            ind = Individual()
            ind.X = x.copy()
            ind.F = f.flatten().copy()
            extras.append(ind)
        return population + extras