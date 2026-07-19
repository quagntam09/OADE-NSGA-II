"""Graph neural network encoding for CE-MOEA."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EncodedPartition:
    """Decoded community assignment from one continuous genotype."""

    labels: np.ndarray
    selected_targets: np.ndarray
    selected_edges: list[tuple[int, int]]


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return
        if self.rank[root_a] < self.rank[root_b]:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        if self.rank[root_a] == self.rank[root_b]:
            self.rank[root_a] += 1


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x)


def graph_neural_network_encoding(x: np.ndarray, network) -> EncodedPartition:
    """
    Decode a continuous vector to communities using sigmoid-softmax-argmax.

    For each node Vi, the sub-vector xi corresponds to its adjacent nodes Di.
    The selected argmax neighbor creates a locus edge, then connected components
    of the selected locus graph become communities.
    """

    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size != network.n_var:
        raise ValueError(f"expected vector length {network.n_var}, got {x.size}")

    uf = _UnionFind(network.n_nodes)
    selected_targets = np.arange(network.n_nodes, dtype=int)
    selected_edges: list[tuple[int, int]] = []

    offset = 0
    for node, neighbors in enumerate(network.neighbors):
        degree = len(neighbors)
        if degree == 0:
            continue

        node_values = x[offset : offset + degree]
        offset += degree

        hidden = sigmoid(node_values)
        probabilities = softmax(hidden)
        target = int(neighbors[int(np.argmax(probabilities))])
        selected_targets[node] = target
        selected_edges.append((node, target))
        uf.union(node, target)

    labels = np.asarray([uf.find(i) for i in range(network.n_nodes)], dtype=int)
    _, labels = np.unique(labels, return_inverse=True)
    return EncodedPartition(labels=labels, selected_targets=selected_targets, selected_edges=selected_edges)

