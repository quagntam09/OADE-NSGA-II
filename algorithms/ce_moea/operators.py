"""Differential evolution and polynomial mutation operators."""

from __future__ import annotations

import numpy as np


def differential_evolution_operator(
    x1: np.ndarray,
    x2: np.ndarray,
    x3: np.ndarray,
    F_DE: float,
    CR: float,
    p_m: float,
    eta_m: float,
    xl: np.ndarray,
    xu: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    x1 = np.asarray(x1, dtype=float)
    mutant = x1 + float(F_DE) * (np.asarray(x2, dtype=float) - np.asarray(x3, dtype=float))
    mask = rng.random(x1.size) <= float(CR)
    y = np.where(mask, mutant, x1)
    y = repair_bounds(y, xl, xu)
    y = polynomial_mutation(y, xl, xu, p_m, eta_m, rng)
    return repair_bounds(y, xl, xu)


def repair_bounds(x: np.ndarray, xl: np.ndarray, xu: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=float), np.asarray(xl, dtype=float), np.asarray(xu, dtype=float))


def polynomial_mutation(
    x: np.ndarray,
    xl: np.ndarray,
    xu: np.ndarray,
    p_m: float,
    eta_m: float,
    rng: np.random.Generator,
) -> np.ndarray:
    y = np.asarray(x, dtype=float).copy()
    xl = np.asarray(xl, dtype=float)
    xu = np.asarray(xu, dtype=float)

    for i in range(y.size):
        if rng.random() >= p_m or xu[i] <= xl[i]:
            continue

        delta1 = (y[i] - xl[i]) / (xu[i] - xl[i])
        delta2 = (xu[i] - y[i]) / (xu[i] - xl[i])
        rand = rng.random()
        mut_pow = 1.0 / (eta_m + 1.0)

        if rand < 0.5:
            xy = 1.0 - delta1
            val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta_m + 1.0))
            delta_q = val ** mut_pow - 1.0
        else:
            xy = 1.0 - delta2
            val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta_m + 1.0))
            delta_q = 1.0 - val ** mut_pow

        y[i] += delta_q * (xu[i] - xl[i])

    return y

