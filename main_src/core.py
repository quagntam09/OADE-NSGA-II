"""Backward-compatible wrapper that re-exports algorithm_src.core."""

from algorithm_src.core import CreationMode, Individual, ProblemWrapper

__all__ = ["ProblemWrapper", "Individual", "CreationMode"]