"""
Vòng lặp tiến hoá chính của NSGA-II cải tiến.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

try:
    from .core import CreationMode, Individual, ProblemWrapper
    from .operators import (
        _make_evaluated_individual,
        de_mutation,
        generate_obl_offspring,
        get_neighborhood_indices,
        initialize_from_data,
        initialize_obl,
        sbx_crossover_mutation,
        tournament_selection,
    )
    from .selection import environmental_selection, remove_duplicates
except ImportError:  # pragma: no cover - fallback for direct script-style imports
    from core import CreationMode, Individual, ProblemWrapper
    from operators import (
        _make_evaluated_individual,
        de_mutation,
        generate_obl_offspring,
        get_neighborhood_indices,
        initialize_from_data,
        initialize_obl,
        sbx_crossover_mutation,
        tournament_selection,
    )
    from selection import environmental_selection, remove_duplicates


@dataclass(frozen=True)
class MechanismConfig:
    """Bật/tắt từng cơ chế cải tiến để làm ablation study."""

    use_obl_init: bool = True
    use_de_operator: bool = True
    use_adaptive_de: bool = True
    use_periodic_obl_injection: bool = True
    obl_injection_period: Optional[int] = None
    use_stagnation_restart: bool = True


class OADENSGA2:
    """
    NSGA-II cải tiến với DE thích nghi, OBL và partial restart khi trì trệ.

    Cải tiến so với NSGA-II gốc (Deb et al., 2002)
    ------------------------------------------------
    - Khởi tạo OBL + Sobol/LHS thay vì phân phối đều thuần tuý
    - Đột biến DE/current-to-pbest/1 hướng láng giềng thay SBX đơn thuần
    - Tham số F, CR tự thích nghi theo Lehmer-mean (SaDE-style)
    - Phun đa dạng định kỳ qua OBL mỗi N thế hệ (N = kích thước quần thể)
    - Phát hiện trì trệ và partial restart khi ideal point ngừng cải thiện
    """

    def __init__(
        self,
        problem: ProblemWrapper,
        pop_size: int = 100,
        n_gen: int = 100,
        mechanism_config: Optional[MechanismConfig] = None,
    ) -> None:
        self.problem  = problem
        self.pop_size = pop_size
        self.n_gen    = n_gen
        self.n_var    = problem.n_var
        self.n_obj    = problem.n_obj
        self.xl       = problem.xl
        self.xu       = problem.xu
        self.mechanism_config = mechanism_config or MechanismConfig()

        # Khởi tạo ADE theo cấu hình: F ~ U[0.4, 0.9], CR ~ U[0.1, 0.9]
        self.mean_F  = float(np.random.uniform(0.4, 0.9))
        self.mean_CR = float(np.random.uniform(0.1, 0.9))
        self.prob_de = 0.5 if self.mechanism_config.use_de_operator else 0.0

        # K-láng giềng cho ADE
        self.n_neighbors = 5

        # Chu kỳ phun OBL: mặc định mỗi N thế hệ (N = pop_size)
        cfg_period = self.mechanism_config.obl_injection_period
        self.obl_injection_period = self.pop_size if cfg_period is None else max(1, int(cfg_period))

        # Tham số SBX và Polynomial Mutation
        self.pc    = 0.9                                           # xác suất lai ghép
        self.pm    = 1.0 / (self.n_var * np.log(max(pop_size, 2)))  # xác suất đột biến mỗi gene
        self.eta_c = 20.0
        self.eta_m = 20.0

        # Ngưỡng phát hiện trì trệ
        self.stagnation_patience   = 20          # số thế hệ không cải thiện trước khi restart
        self.stagnation_tolerance  = 1e-4        # cải thiện nhỏ hơn ngưỡng này bị coi là trì trệ
        self.restart_elite_ratio   = 0.3         # tỉ lệ cá thể tốt giữ lại sau restart

        # Trạng thái runtime
        self.population: List[Individual] = []
        self.history:    List[np.ndarray] = []  # lịch sử F mỗi thế hệ

    # Vòng lặp chính
    def run(
        self,
        initial_x: Optional[np.ndarray] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> np.ndarray:
        """
        Chạy thuật toán và trả về ma trận F của quần thể cuối, shape (pop_size, n_obj).
        Nếu initial_x được cung cấp, dùng nó làm quần thể khởi đầu (warm-start).
        Nếu không, khởi tạo bằng OBL + Sobol.
        """
        self.population = self._build_initial_population(initial_x)
        self.history.clear()

        last_ideal         = None
        stagnation_counter = 0

        for gen in range(self.n_gen):
            stagnation_counter, last_ideal = self._check_and_handle_stagnation(
                stagnation_counter, last_ideal
            )

            offspring = self._generate_offspring(gen)
            self._evaluate_unevaluated(offspring)

            combined        = remove_duplicates(self.population + offspring)
            combined        = self._fill_if_too_small(combined)
            self.population = environmental_selection(combined, self.pop_size, self.n_obj)

            self._update_adaptive_parameters(offspring)
            self.history.append(np.array([ind.F for ind in self.population]))

            if progress_callback is not None:
                progress_callback(gen + 1, self.n_gen)

        return np.array([ind.F for ind in self.population])

    # Khởi tạo
    def _build_initial_population(self, initial_x: Optional[np.ndarray]) -> List[Individual]:
        """Khởi tạo từ dữ liệu có sẵn hoặc bằng OBL + Sobol nếu không có dữ liệu."""
        if initial_x is not None:
            pop = initialize_from_data(self.problem, initial_x)
            return environmental_selection(pop, self.pop_size, self.n_obj)
        if self.mechanism_config.use_obl_init:
            return initialize_obl(self.problem, self.pop_size)
        return self._initialize_random_population(self.pop_size)

    def _initialize_random_population(self, size: int) -> List[Individual]:
        x_init = self.xl + np.random.rand(size, self.n_var) * (self.xu - self.xl)
        f_init = self.problem.evaluate(x_init)
        pop = [_make_evaluated_individual(x_init[i], f_init[i]) for i in range(size)]
        return environmental_selection(pop, self.pop_size, self.n_obj)

    # Sinh con lai
    def _generate_offspring(self, gen: int) -> List[Individual]:
        """Sinh toàn bộ con lai cho thế hệ này: DE hoặc SBX theo prob_de, thêm OBL theo chu kỳ cấu hình."""
        neighbor_indices = get_neighborhood_indices(self.population, self.n_neighbors)

        offspring = [
            self._create_one_child(i, neighbor_indices)
            for i in range(self.pop_size)
        ]

        if (
            self.mechanism_config.use_periodic_obl_injection
            and self.obl_injection_period > 0
            and gen % self.obl_injection_period == 0
        ):
            offspring += generate_obl_offspring(self.population, self.problem, self.xl, self.xu)

        return offspring

    def _create_one_child(self, idx: int, neighbor_indices: np.ndarray) -> Individual:
        """Tạo một con lai bằng DE (theo prob_de) hoặc SBX."""
        if self.mechanism_config.use_de_operator and np.random.rand() < self.prob_de:
            return de_mutation(
                idx, self.population, neighbor_indices,
                self.xl, self.xu, self.n_var, self.mean_F, self.mean_CR,
            )
        return sbx_crossover_mutation(
            tournament_selection(self.population),
            tournament_selection(self.population),
            self.xl, self.xu, self.n_var, self.pc, self.pm, self.eta_c, self.eta_m,
        )

    def _evaluate_unevaluated(self, offspring: List[Individual]) -> None:
        """Đánh giá batch tất cả con lai chưa có F. OBL offspring đã được đánh giá trước."""
        unevaluated = [ind for ind in offspring if ind.F is None]
        if not unevaluated:
            return

        x_batch = np.array([ind.X for ind in unevaluated])
        f_batch = self.problem.evaluate(x_batch)
        for ind, f in zip(unevaluated, f_batch):
            ind.F = f.flatten().copy()

    # Trì trệ và partial restart
    def _check_and_handle_stagnation(
        self, stagnation_counter: int, last_ideal: Optional[np.ndarray]
    ) -> tuple[int, np.ndarray]:
        """Đếm thế hệ trì trệ; restart khi đủ kiên nhẫn. Trả về (counter, ideal_point) mới."""
        current_ideal = np.min([ind.F for ind in self.population], axis=0)

        if last_ideal is not None:
            improved       = np.linalg.norm(current_ideal - last_ideal) >= self.stagnation_tolerance
            stagnation_counter = 0 if improved else stagnation_counter + 1

        if self.mechanism_config.use_stagnation_restart and stagnation_counter >= self.stagnation_patience:
            self._partial_restart()
            stagnation_counter = 0

        return stagnation_counter, current_ideal

    def _partial_restart(self) -> None:
        """Giữ lại elite, thay thế phần còn lại bằng cá thể ngẫu nhiên, reset tham số DE."""
        n_keep   = int(self.pop_size * self.restart_elite_ratio)
        elite    = self.population[:n_keep]

        n_new    = self.pop_size - n_keep
        x_new    = self.xl + np.random.rand(n_new, self.n_var) * (self.xu - self.xl)
        f_new    = self.problem.evaluate(x_new)
        new_inds = [_make_evaluated_individual(x_new[i], f_new[i]) for i in range(n_new)]

        self.population = environmental_selection(elite + new_inds, self.pop_size, self.n_obj)
        self.mean_F = float(np.random.uniform(0.4, 0.9))
        self.mean_CR = float(np.random.uniform(0.1, 0.9))

    def _fill_if_too_small(self, population: List[Individual]) -> List[Individual]:
        """Bổ sung cá thể ngẫu nhiên nếu remove_duplicates làm quần thể thiếu hụt."""
        missing = self.pop_size - len(population)
        if missing <= 0:
            return population

        x_fill = self.xl + np.random.rand(missing, self.n_var) * (self.xu - self.xl)
        f_fill = self.problem.evaluate(x_fill)
        extras = [_make_evaluated_individual(x_fill[i], f_fill[i]) for i in range(missing)]
        return population + extras

    # Cập nhật tham số thích nghi
    def _update_adaptive_parameters(self, offspring: List[Individual]) -> None:
        """
        Cập nhật prob_de, mean_F, mean_CR dựa trên con lai đạt Pareto front 1.

        prob_de : EMA của tỉ lệ DE thành công trong offspring
        mean_F  : Lehmer mean của F — ưu giá trị lớn, khuyến khích khám phá
        mean_CR : trung bình cộng của CR
        """
        if not self.mechanism_config.use_adaptive_de:
            return

        de_offspring  = [ind for ind in offspring if ind.creation_mode == CreationMode.DE]
        sbx_offspring = [ind for ind in offspring if ind.creation_mode == CreationMode.SBX]
        successful_de = [ind for ind in de_offspring if ind.rank == 1]

        total = len(de_offspring) + len(sbx_offspring)
        if total > 0:
            de_ratio     = len(de_offspring) / total
            self.prob_de = float(np.clip(0.9 * self.prob_de + 0.1 * de_ratio, 0.2, 0.8))

        if successful_de:
            f_values     = np.array([ind.used_F  for ind in successful_de])
            cr_values    = np.array([ind.used_CR for ind in successful_de])
            lehmer_mean_F = float(np.mean(f_values ** 2) / np.mean(f_values))
            self.mean_F  = 0.9 * self.mean_F  + 0.1 * lehmer_mean_F
            self.mean_CR = 0.9 * self.mean_CR + 0.1 * float(np.mean(cr_values))


