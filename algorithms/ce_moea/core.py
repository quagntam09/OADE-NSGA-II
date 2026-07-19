"""Core attributed-network problem model for CE-MOEA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .encoding import EncodedPartition, graph_neural_network_encoding
from .objectives import attribute_objective, modularity

AttributeMode = Literal["auto", "single", "multi"]
MultiAttributeMode = Literal["distance", "paper_cosine"]


@dataclass(frozen=True)
class AttributedNetwork:
    """
    Attributed graph used by CE-MOEA.

    The continuous genotype contains one value for each non-zero adjacency entry
    e_ij. For an undirected symmetric adjacency matrix, this naturally gives one
    value per node-neighbor relation.
    """

    adjacency: np.ndarray
    attributes: np.ndarray

    def __post_init__(self) -> None:
        adjacency = np.asarray(self.adjacency, dtype=bool)
        attributes = np.asarray(self.attributes, dtype=float)
        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError("adjacency must be a square matrix")
        if attributes.shape[0] != adjacency.shape[0]:
            raise ValueError("attributes must have one row/value per node")

        object.__setattr__(self, "adjacency", adjacency.copy())
        object.__setattr__(self, "attributes", attributes.copy())

    @classmethod
    def from_edges(
        cls,
        n_nodes: int,
        edges: list[tuple[int, int]],
        attributes: np.ndarray,
        undirected: bool = True,
    ) -> "AttributedNetwork":
        adjacency = np.zeros((n_nodes, n_nodes), dtype=bool)
        for u, v in edges:
            adjacency[u, v] = True
            if undirected:
                adjacency[v, u] = True
        return cls(adjacency=adjacency, attributes=np.asarray(attributes, dtype=float))

    @property
    def n_nodes(self) -> int:
        return int(self.adjacency.shape[0])

    @property
    def neighbors(self) -> list[np.ndarray]:
        return [np.flatnonzero(self.adjacency[i]) for i in range(self.n_nodes)]

    @property
    def n_var(self) -> int:
        return int(np.count_nonzero(self.adjacency))


class AttributedNetworkProblem:
    """
    CE-MOEA objective wrapper.

    Objective vector is minimized as F(x) = (-Q(x), fs(x)) for single attributes
    or F(x) = (-Q(x), fm(x)) for multi attributes.
    """

    def __init__(
        self,
        network: AttributedNetwork,
        attribute_mode: AttributeMode = "auto",
        multi_attribute_mode: MultiAttributeMode = "distance",
    ) -> None:
        self.network = network
        self.attribute_mode = self._resolve_attribute_mode(attribute_mode)
        self.multi_attribute_mode = multi_attribute_mode
        self.n_var = network.n_var
        self.n_obj = 2
        self.xl = np.zeros(self.n_var, dtype=float)
        self.xu = np.ones(self.n_var, dtype=float)

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.n_var:
            raise ValueError(f"expected {self.n_var} decision variables, got {X.shape[1]}")

        values = []
        for x in X:
            partition = self.decode(x)
            q_value = modularity(self.network, partition.labels)
            attr_value = attribute_objective(
                self.network.attributes,
                partition.labels,
                self.attribute_mode,
                multi_attribute_mode=self.multi_attribute_mode,
            )
            values.append([-q_value, attr_value])
        return np.asarray(values, dtype=float)

    def decode(self, x: np.ndarray) -> EncodedPartition:
        return graph_neural_network_encoding(np.asarray(x, dtype=float), self.network)

    def _resolve_attribute_mode(self, mode: AttributeMode) -> Literal["single", "multi"]:
        if mode != "auto":
            return mode
        attrs = self.network.attributes
        return "single" if attrs.ndim == 1 or attrs.shape[1] == 1 else "multi"

