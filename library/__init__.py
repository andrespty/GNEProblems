"""
Generalized Nash Equilibrium Problems Solver
---------------------------------------------


"""
from .GNESolverBounded import *
from .GNESolverUnbounded import *
from .core import check_NE, get_problem, get_initial_point, summary

__all__ = [
    "check_NE",
    "GNEP_Solver_Unbounded",
    "GNEP_Solver_Bounded",
    "get_problem",
    "get_initial_point",
    "summary",
]


__version__ = "0.1.0"
__author__ = "Andres Ho"