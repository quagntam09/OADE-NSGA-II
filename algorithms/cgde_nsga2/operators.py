"""Cross-generation differential evolution operators."""

from __future__ import annotations

import numpy as np


def cgde_offspring(
    target_idx: int,
    population: np.ndarray,
    ranks: np.ndarray,
    crowding: np.ndarray,
    archive: np.ndarray,
    F: float,
    CR: float,
    pbest_ratio: float,
    xl: np.ndarray,
    xu: np.ndarray,
    rng: np.random.Generator,
    mutation_prob: float | None = None,
    eta_m: float = 20.0,
) -> np.ndarray:
    """Create one offspring using current-to-pbest with cross-generation archive."""

    target = population[target_idx]
    pbest = _choose_pbest(population, ranks, crowding, pbest_ratio, rng)
    r1 = _choose_current(population, target_idx, rng)
    r2 = _choose_archive_or_population(archive, population, rng)

    mutant = target + F * (pbest - target) + F * (r1 - r2)
    trial = binomial_crossover(target, mutant, CR, rng)
    trial = np.clip(trial, xl, xu)
    if mutation_prob is not None and mutation_prob > 0.0:
        trial = polynomial_mutation(trial, xl, xu, mutation_prob, eta_m, rng)
    return np.clip(trial, xl, xu)


def binomial_crossover(target: np.ndarray, mutant: np.ndarray, CR: float, rng: np.random.Generator) -> np.ndarray:
    mask = rng.random(target.size) <= CR
    mask[int(rng.integers(target.size))] = True
    return np.where(mask, mutant, target)


def polynomial_mutation(
    x: np.ndarray,
    xl: np.ndarray,
    xu: np.ndarray,
    mutation_prob: float,
    eta_m: float,
    rng: np.random.Generator,
) -> np.ndarray:
    mutated = x.copy()
    for i in range(mutated.size):
        if rng.random() > mutation_prob or xu[i] <= xl[i]:
            continue
        y = mutated[i]
        delta1 = (y - xl[i]) / (xu[i] - xl[i])
        delta2 = (xu[i] - y) / (xu[i] - xl[i])
        rand = rng.random()
        mut_pow = 1.0 / (eta_m + 1.0)
        if rand <= 0.5:
            xy = 1.0 - delta1
            val = 2.0 * rand + (1.0 - 2.0 * rand) * xy ** (eta_m + 1.0)
            delta_q = val ** mut_pow - 1.0
        else:
            xy = 1.0 - delta2
            val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * xy ** (eta_m + 1.0)
            delta_q = 1.0 - val ** mut_pow
        mutated[i] = y + delta_q * (xu[i] - xl[i])
    return mutated


def _choose_pbest(
    population: np.ndarray,
    ranks: np.ndarray,
    crowding: np.ndarray,
    pbest_ratio: float,
    rng: np.random.Generator,
) -> np.ndarray:
    order = np.lexsort((-crowding, ranks))
    top_count = max(1, int(np.ceil(len(population) * pbest_ratio)))
    chosen = int(rng.choice(order[:top_count]))
    return population[chosen]


def _choose_current(population: np.ndarray, target_idx: int, rng: np.random.Generator) -> np.ndarray:
    candidates = [idx for idx in range(len(population)) if idx != target_idx]
    if not candidates:
        return population[target_idx]
    return population[int(rng.choice(candidates))]


def _choose_archive_or_population(
    archive: np.ndarray,
    population: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    pool = archive if len(archive) > 0 else population
    return pool[int(rng.integers(len(pool)))]

