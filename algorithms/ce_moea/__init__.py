"""Continuous Encoding MOEA for attributed network community detection."""

from .algorithm import CEMOEA
from .core import AttributedNetwork, AttributedNetworkProblem
from .encoding import EncodedPartition, graph_neural_network_encoding
from .operators import differential_evolution_operator

__all__ = [
    "AttributedNetwork",
    "AttributedNetworkProblem",
    "CEMOEA",
    "EncodedPartition",
    "differential_evolution_operator",
    "graph_neural_network_encoding",
]

