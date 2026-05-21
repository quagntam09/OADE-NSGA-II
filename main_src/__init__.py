"""
Package thuật toán Improved NSGA-II.

Dùng nhanh
----------
    from nsga2 import ProblemWrapper, OADE_NSGAII
    solver = OADE_NSGAII(problem, pop_size=100, n_gen=200)
    front  = solver.run()
"""

from .algorithm import OADE_NSGAII
from .core import CreationMode, Individual, ProblemWrapper

__all__ = ["OADE_NSGAII", "ProblemWrapper", "Individual", "CreationMode"]