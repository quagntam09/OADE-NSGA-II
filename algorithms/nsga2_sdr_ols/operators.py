"""OBL, local search, and GA operators for NSGA-II/SDR-OLS."""

from __future__ import annotations

import numpy as np

from .core import Individual, ProblemAdapter, evaluate_unevaluated, make_individual
from .sdr import environmental_selection_sdr


def random_population(problem: ProblemAdapter, size: int, rng: np.random.Generator) -> list[Individual]:
    X = rng.uniform(problem.xl, problem.xu, size=(size, problem.n_var))
    F = problem.evaluate(X)
    return [make_individual(x, f, source="random") for x, f in zip(X, F)]


def opposition_based_learning(
    problem: ProblemAdapter,
    population: list[Individual],
    target_size: int,
) -> list[Individual]:
    X = np.asarray([ind.X for ind in population], dtype=float)
    opposite_X = np.clip(problem.xl + problem.xu - X, problem.xl, problem.xu)
    opposite_F = problem.evaluate(opposite_X)
    opposite = [make_individual(x, f, source="obl") for x, f in zip(opposite_X, opposite_F)]
    return environmental_selection_sdr(population + opposite, target_size, problem.n_obj)


def local_search(
    problem: ProblemAdapter,
    population: list[Individual],
    mu: float,
    sigma: float,
    rng: np.random.Generator,
    max_neighbors: int | None = None,
) -> list[Individual]:
    """
    Generate omega+ and omega- neighbors from Equations (14) and (15).

    By default all decision variables are visited. For large-scale experiments,
    max_neighbors can cap the number of evaluated neighbors per generation.
    """

    neighbors: list[Individual] = []
    n_var = problem.n_var
    budget = np.inf if max_neighbors is None else max(0, int(max_neighbors))

    for i, ind in enumerate(population):
        if len(neighbors) >= budget:
            break
        variable_order = rng.permutation(n_var)
        for k in variable_order:
            if len(neighbors) >= budget:
                break

            u_idx, v_idx = rng.choice(len(population), size=2, replace=len(population) < 2)
            diff = population[u_idx].X[k] - population[v_idx].X[k]
            c = float(rng.normal(mu, sigma))

            x_plus = ind.X.copy()
            x_minus = ind.X.copy()
            x_plus[k] = x_plus[k] + c * diff
            x_minus[k] = x_minus[k] - c * diff

            neighbors.append(make_individual(np.clip(x_plus, problem.xl, problem.xu), source="ls", parent=i, variable=int(k)))
            if len(neighbors) < budget:
                neighbors.append(make_individual(np.clip(x_minus, problem.xl, problem.xu), source="ls", parent=i, variable=int(k)))

    evaluate_unevaluated(problem, neighbors)
    return neighbors


def tournament_selection(population: list[Individual], rng: np.random.Generator) -> Individual:
    a_idx, b_idx = rng.choice(len(population), size=2, replace=len(population) < 2)
    a = population[int(a_idx)]
    b = population[int(b_idx)]

    a_rank = a.rank if a.rank is not None else np.inf
    b_rank = b.rank if b.rank is not None else np.inf
    if a_rank != b_rank:
        return a if a_rank < b_rank else b
    return a if a.crowding_dist >= b.crowding_dist else b


def make_ga_offspring(
    problem: ProblemAdapter,
    population: list[Individual],
    offspring_size: int,
    rng: np.random.Generator,
    crossover_prob: float = 0.9,
    mutation_prob: float | None = None,
    eta_c: float = 20.0,
    eta_m: float = 20.0,
) -> list[Individual]:
    mutation_prob = 1.0 / problem.n_var if mutation_prob is None else mutation_prob
    offspring: list[Individual] = []

    while len(offspring) < offspring_size:
        p1 = tournament_selection(population, rng)
        p2 = tournament_selection(population, rng)
        child_x = p1.X.copy()
        if rng.random() <= crossover_prob:
            child_x = sbx_crossover(p1.X, p2.X, problem.xl, problem.xu, eta_c, rng)
        child_x = polynomial_mutation(child_x, problem.xl, problem.xu, mutation_prob, eta_m, rng)
        offspring.append(make_individual(child_x, source="ga"))

    evaluate_unevaluated(problem, offspring)
    return offspring


def sbx_crossover(
    x1: np.ndarray,
    x2: np.ndarray,
    xl: np.ndarray,
    xu: np.ndarray,
    eta_c: float,
    rng: np.random.Generator,
) -> np.ndarray:
    child = x1.copy()
    for j in range(len(x1)):
        if rng.random() > 0.5 or abs(x1[j] - x2[j]) <= 1e-14:
            continue

        y1 = min(x1[j], x2[j])
        y2 = max(x1[j], x2[j])
        rand = rng.random()

        beta = 1.0 + (2.0 * (y1 - xl[j]) / (y2 - y1))
        alpha = 2.0 - beta ** (-(eta_c + 1.0))
        if rand <= 1.0 / alpha:
            betaq = (rand * alpha) ** (1.0 / (eta_c + 1.0))
        else:
            betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta_c + 1.0))

        child[j] = 0.5 * ((y1 + y2) - betaq * (y2 - y1))

    return np.clip(child, xl, xu)


def polynomial_mutation(
    x: np.ndarray,
    xl: np.ndarray,
    xu: np.ndarray,
    mutation_prob: float,
    eta_m: float,
    rng: np.random.Generator,
) -> np.ndarray:
    mutated = x.copy()
    for j in range(len(mutated)):
        if rng.random() > mutation_prob or xu[j] <= xl[j]:
            continue

        y = mutated[j]
        delta1 = (y - xl[j]) / (xu[j] - xl[j])
        delta2 = (xu[j] - y) / (xu[j] - xl[j])
        rand = rng.random()
        mut_pow = 1.0 / (eta_m + 1.0)

        if rand <= 0.5:
            xy = 1.0 - delta1
            val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta_m + 1.0))
            deltaq = val ** mut_pow - 1.0
        else:
            xy = 1.0 - delta2
            val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta_m + 1.0))
            deltaq = 1.0 - val ** mut_pow

        mutated[j] = y + deltaq * (xu[j] - xl[j])

    return np.clip(mutated, xl, xu)

