"""
Generalized Nash Equilibrium Problems Solver
---------------------------------------------


"""
from .Problems_Bounded.ProblemA6 import A6

from .Problems_Unbounded.ProblemA1U import A1U
from .Problems_Unbounded.ProblemA2U import A2U
from .Problems_Unbounded.ProblemA3U import A3U
from .Problems_Unbounded.ProblemA4U import A4U
from .Problems_Unbounded.ProblemA5U import A5U
from .Problems_Unbounded.ProblemA7U import A7U
from .Problems_Unbounded.ProblemA8U import A8U
from .Problems_Unbounded.ProblemA9aU import A9aU
from .Problems_Unbounded.ProblemA9bU import A9bU
from .Problems_Unbounded.ProblemA10aU import A10aU
from .Problems_Unbounded.ProblemA10bU import A10bU
from .Problems_Unbounded.ProblemA10cU import A10cU
from .Problems_Unbounded.ProblemA10dU import A10dU
from .Problems_Unbounded.ProblemA10eU import A10eU
from .Problems_Bounded import *

__all__ = [
    "A6",

    "A1U",
    'A2U',
    'A3U',
    'A4U',
    'A5U',
    #'A6U', #Not working yet
    'A7U',
    # 'A8U', Not working yet
    'A9aU',
    'A9bU',
    'A10aU',
    'A10bU',
    'A10cU',
    'A10dU',
    'A10eU',
]


__version__ = "0.1.0"
__author__ = "Andres Ho"