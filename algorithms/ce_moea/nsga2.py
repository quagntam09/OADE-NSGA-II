"""NSGA-II ranking, crowding distance, and selection utilities."""

from __future__ import annotations

import numpy as np


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a <= b) and np.any(a < b))


def fast_non_dominated_sort(F: np.ndarray) -> list[list[int]]:
    F = np.asarray(F, dtype=float)
    n = len(F)
    dominated_sets: list[list[int]] = [[] for _ in range(n)]
    domination_count = np.zeros(n, dtype=int)
    fronts: list[list[int]] = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(F[p], F[q]):
                dominated_sets[p].append(q)
            elif dominates(F[q], F[p]):
                domination_count[p] += 1
        if domination_count[p] == 0:
            fronts[0].append(p)

    rank = 0
    while fronts[rank]:
        next_front: list[int] = []
        for p in fronts[rank]:
            for q in dominated_sets[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)
        rank += 1
        fronts.append(next_front)

    return fronts[:-1]


def crowding_distance(F: np.ndarray, front: list[int]) -> dict[int, float]:
    if not front:
        return {}
    if len(front) <= 2:
        return {idx: float("inf") for idx in front}

    F = np.asarray(F, dtype=float)
    distances = {idx: 0.0 for idx in front}
    n_obj = F.shape[1]

    for obj in range(n_obj):
        ordered = sorted(front, key=lambda idx: F[idx, obj])
        distances[ordered[0]] = float("inf")
        distances[ordered[-1]] = float("inf")
        f_min = F[ordered[0], obj]
        f_max = F[ordered[-1], obj]
        if abs(f_max - f_min) <= 1e-12:
            continue
        for pos in range(1, len(ordered) - 1):
            idx = ordered[pos]
            if np.isinf(distances[idx]):
                continue
            distances[idx] += float((F[ordered[pos + 1], obj] - F[ordered[pos - 1], obj]) / (f_max - f_min))

    return distances


def rank_and_crowding(F: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
    fronts = fast_non_dominated_sort(F)
    ranks = np.full(len(F), fill_value=np.inf, dtype=float)
    crowding = np.zeros(len(F), dtype=float)
    for rank, front in enumerate(fronts):
        for idx in front:
            ranks[idx] = rank
        distances = crowding_distance(F, front)
        for idx, distance in distances.items():
            crowding[idx] = distance
    return ranks, crowding, fronts


def binary_tournament_selection(
    P: np.ndarray,
    ranks: np.ndarray,
    crowding: np.ndarray,
    rng: np.random.Generator,
    size: int | None = None,
) -> np.ndarray:
    size = len(P) if size is None else int(size)
    selected = []
    for _ in range(size):
        a, b = rng.choice(len(P), size=2, replace=len(P) < 2)
        if ranks[a] < ranks[b]:
            winner = a
        elif ranks[b] < ranks[a]:
            winner = b
        else:
            winner = a if crowding[a] >= crowding[b] else b
        selected.append(P[int(winner)].copy())
    return np.asarray(selected, dtype=float)


def environmental_selection(P: np.ndarray, F: np.ndarray, target_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ranks, crowding, fronts = rank_and_crowding(F)
    selected_indices: list[int] = []

    for front in fronts:
        slots = target_size - len(selected_indices)
        if slots <= 0:
            break
        if len(front) <= slots:
            selected_indices.extend(front)
        else:
            ordered = sorted(front, key=lambda idx: crowding[idx], reverse=True)
            selected_indices.extend(ordered[:slots])
            break

    selected_indices_array = np.asarray(selected_indices, dtype=int)
    selected_P = P[selected_indices_array].copy()
    selected_F = F[selected_indices_array].copy()
    selected_ranks, selected_crowding, _ = rank_and_crowding(selected_F)
    return selected_P, selected_F, selected_ranks, selected_crowding

