"""Strengthened dominance relation (SDR) sorting and survival."""

from __future__ import annotations

import math

import numpy as np

from .core import Individual

EPS = 1e-12


def normalized_objectives(population: list[Individual]) -> np.ndarray:
    F = np.asarray([ind.F for ind in population], dtype=float)
    f_min = F.min(axis=0)
    f_max = F.max(axis=0)
    span = np.where((f_max - f_min) <= EPS, 1.0, f_max - f_min)
    return (F - f_min) / span


def acute_angle_matrix(F: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(F, axis=1)
    denom = np.outer(norms, norms)
    cosine = np.ones((len(F), len(F)), dtype=float)
    valid = denom > EPS
    cosine[valid] = np.sum(F[:, None, :] * F[None, :, :], axis=2)[valid] / denom[valid]
    cosine = np.clip(cosine, -1.0, 1.0)
    angles = np.arccos(cosine)
    np.fill_diagonal(angles, np.inf)
    return angles


def adaptive_theta(angles: np.ndarray) -> float:
    """Equation (9): floor(|P| / 2)-th smallest nearest-neighbor angle."""

    finite = np.where(np.isfinite(angles), angles, np.inf)
    nearest = finite.min(axis=1)
    nearest = nearest[np.isfinite(nearest)]
    if len(nearest) == 0:
        return 0.0

    kth = max(0, min(len(nearest) - 1, math.floor(len(nearest) / 2) - 1))
    return float(np.partition(nearest, kth)[kth])


def sdr_dominance_matrix(population: list[Individual]) -> np.ndarray:
    """
    Return matrix D where D[i, j] is True if i dominates j by SDR.

    SDR uses convergence Con(x)=sum normalized objectives and the acute angle between
    normalized objective vectors. All objectives are assumed to be minimized.
    """

    if not population:
        return np.empty((0, 0), dtype=bool)

    F = normalized_objectives(population)
    con = F.sum(axis=1)
    angles = acute_angle_matrix(F)
    theta = adaptive_theta(angles)

    better_con = con[:, None] < con[None, :]
    same_niche = angles <= theta
    if theta <= EPS:
        dominates = better_con
    else:
        cross_niche = (con[:, None] * angles / theta) < con[None, :]
        dominates = (better_con & same_niche) | (cross_niche & ~same_niche)

    np.fill_diagonal(dominates, False)
    return dominates


def fast_non_dominated_sort_sdr(population: list[Individual]) -> list[list[Individual]]:
    if not population:
        return []

    dominates = sdr_dominance_matrix(population)
    domination_count = dominates.sum(axis=0).astype(int)
    dominated_by = [np.where(dominates[i])[0].tolist() for i in range(len(population))]

    fronts: list[list[Individual]] = []
    current = np.where(domination_count == 0)[0].tolist()
    rank = 1

    while current:
        for idx in current:
            population[idx].rank = rank
        fronts.append([population[idx] for idx in current])

        next_front: list[int] = []
        for p in current:
            for q in dominated_by[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)

        current = next_front
        rank += 1

    return fronts


def calculate_crowding_distance(front: list[Individual], n_obj: int) -> None:
    if not front:
        return

    for ind in front:
        ind.crowding_dist = 0.0

    if len(front) <= 2:
        for ind in front:
            ind.crowding_dist = float("inf")
        return

    for obj in range(n_obj):
        front.sort(key=lambda ind: float(ind.F[obj]))
        front[0].crowding_dist = float("inf")
        front[-1].crowding_dist = float("inf")

        f_min = float(front[0].F[obj])
        f_max = float(front[-1].F[obj])
        if abs(f_max - f_min) <= EPS:
            continue

        for idx in range(1, len(front) - 1):
            if math.isinf(front[idx].crowding_dist):
                continue
            front[idx].crowding_dist += (front[idx + 1].F[obj] - front[idx - 1].F[obj]) / (f_max - f_min)


def environmental_selection_sdr(population: list[Individual], target_size: int, n_obj: int) -> list[Individual]:
    fronts = fast_non_dominated_sort_sdr(population)
    selected: list[Individual] = []

    for front in fronts:
        calculate_crowding_distance(front, n_obj)
        remaining = target_size - len(selected)
        if remaining <= 0:
            break
        if len(front) <= remaining:
            selected.extend(front)
        else:
            front.sort(key=lambda ind: ind.crowding_dist, reverse=True)
            selected.extend(front[:remaining])
            break

    selected.sort(key=lambda ind: (ind.rank if ind.rank is not None else math.inf, -ind.crowding_dist))
    return selected

