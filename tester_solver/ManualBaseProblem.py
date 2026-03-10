from abc import ABC, abstractmethod
import jax.numpy as jnp
from gnep_solver.schema import *
from typing import List
from gnep_solver.Player import Player
from gnep_solver.GeneralizedGame import GeneralizedGame
from .ManualGeneralizedGame import ManualGeneralizedGame
from gnep_solver.validation import validate_problem_functions
from gnep_solver.BaseProblem import BaseProblem

class ManualBaseProblem(BaseProblem):
    def __init__(self, players: List[Player] = None):
        super().__init__(players)
        self.engine = ManualGeneralizedGame(
            self.objectives(),
            self.objectives_der(),
            self.constraints(),
            self.constraints_der(),
            self.players
        )

    @abstractmethod
    @validate_problem_functions(derivative=True)
    def objectives_der(self):
        """Return a list of the objectives of the problem."""
        pass

    @abstractmethod
    @validate_problem_functions(derivative=True)
    def constraints_der(self):
        """Return a list of the constraints of the problem."""
        pass