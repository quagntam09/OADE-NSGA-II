"""Objective functions for attributed network community detection."""

from __future__ import annotations

from typing import Literal

import numpy as np

EPS = 1e-12


def modularity(network, labels: np.ndarray) -> float:
    """Newman-Girvan modularity Q for an undirected view of the network."""

    adjacency = np.asarray(network.adjacency, dtype=bool)
    undirected = np.logical_or(adjacency, adjacency.T)
    np.fill_diagonal(undirected, False)

    degrees = undirected.sum(axis=1).astype(float)
    total_edges = float(np.triu(undirected, k=1).sum())
    if total_edges <= EPS:
        return 0.0

    q_value = 0.0
    for community in np.unique(labels):
        idx = np.flatnonzero(labels == community)
        if len(idx) == 0:
            continue
        subgraph = undirected[np.ix_(idx, idx)]
        internal_edges = float(np.triu(subgraph, k=1).sum())
        degree_sum = float(degrees[idx].sum())
        q_value += internal_edges / total_edges - (degree_sum / (2.0 * total_edges)) ** 2
    return float(q_value)


def attribute_objective(
    attributes: np.ndarray,
    labels: np.ndarray,
    mode: Literal["single", "multi"],
    multi_attribute_mode: Literal["distance", "paper_cosine"] = "distance",
) -> float:
    if mode == "single":
        return single_attribute_distance(attributes, labels)
    if mode == "multi":
        return multi_attribute_distance(attributes, labels, multi_attribute_mode)
    raise ValueError(f"unsupported attribute mode: {mode}")


def single_attribute_distance(attributes: np.ndarray, labels: np.ndarray) -> float:
    values = np.asarray(attributes, dtype=float)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]

    numerator = 0.0
    denominator = 0
    for community in np.unique(labels):
        idx = np.flatnonzero(labels == community)
        r = len(idx)
        if r < 2:
            continue
        denominator += r * (r - 1)
        community_values = values[idx]
        diffs = np.abs(community_values[:, None] - community_values[None, :])
        numerator += float(np.triu(diffs, k=1).sum())

    return 0.0 if denominator == 0 else float(numerator / denominator)


def multi_attribute_distance(
    attributes: np.ndarray,
    labels: np.ndarray,
    mode: Literal["distance", "paper_cosine"] = "distance",
) -> float:
    values = np.asarray(attributes, dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1, 1)

    numerator = 0.0
    denominator = 0
    for community in np.unique(labels):
        idx = np.flatnonzero(labels == community)
        r = len(idx)
        if r < 2:
            continue
        denominator += r * (r - 1)
        community_values = values[idx]
        norms = np.linalg.norm(community_values, axis=1)

        for a in range(r):
            for b in range(a + 1, r):
                denom = norms[a] * norms[b]
                cosine = 0.0 if denom <= EPS else float(np.dot(community_values[a], community_values[b]) / denom)
                numerator += cosine if mode == "paper_cosine" else 1.0 - cosine

    return 0.0 if denominator == 0 else float(numerator / denominator)

