"""Backward-compatible wrapper package that points to algorithm_src."""

from .algorithm import MechanismConfig, OADENSGA2
from .core import CreationMode, Individual, ProblemWrapper

__all__ = [
    "OADENSGA2",
    "MechanismConfig",
    "ProblemWrapper",
    "Individual",
    "CreationMode",
]