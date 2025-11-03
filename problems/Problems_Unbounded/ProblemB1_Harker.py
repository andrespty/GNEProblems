
import numpy as np
from typing import List

Vector = np.ndarray
VectorList = List[np.ndarray]

class B1U:
    @staticmethod
    def paper_solution():
        # From the paper: SOL^GNEP = {(5, 9)} U {(t, 15 - t) | 9 <= t <= 10}
        # Representing just the main equilibrium point (5, 9)
        value_1 = [5, 9]
        return [value_1]

    @staticmethod
    def define_players():
        # There are 2 players, each controlling 1 variable
        player_vector_sizes = [1, 1]
        player_objective_functions = [0, 0]  # Player 1 -> f1(x1,x2), Player 2 -> f2(x1,x2)
        # Shared constraint: x1 + x2 ≤ 15
        player_constraints = [[0], [0]]  # Both share the same constraint
        return [player_vector_sizes, player_objective_functions, player_constraints]

    # === Objective Functions ===
    @staticmethod
    def objective_functions():
        return [B1U.obj_func]

    @staticmethod
    def objective_function_derivatives():
        return [B1U.obj_func_der]

    # === Constraints ===
    @staticmethod
    def constraints():
        return [B1U.g_shared]

    @staticmethod
    def constraint_derivatives():
        return [B1U.g_shared_der]

    # === Player 1 and Player 2 objective functions ===
    @staticmethod
    def obj_func(x: VectorList) -> Vector:
        """
        f1(x1, x2) = x1^2 + (8/3)x1x2 - 34x1
        f2(x1, x2) = x2^2 + (5/4)x1x2 - 24.25x2
        """
        x1 = x[0]
        x2 = x[1]

        f1 = x1**2 + (8/3) * x1 * x2 - 34 * x1
        f2 = x2**2 + (5/4) * x1 * x2 - 24.25 * x2
        return np.array([f1, f2]).reshape(-1, 1)

    @staticmethod
    def obj_func_der(x: VectorList) -> Vector:
        """
        Gradient of each player's objective with respect to their own variable:
        ∂f1/∂x1 = 2x1 + (8/3)x2 - 34
        ∂f2/∂x2 = 2x2 + (5/4)x1 - 24.25
        """
        x1 = x[0]
        x2 = x[1]

        df1_dx1 = 2 * x1 + (8 / 3) * x2 - 34
        df2_dx2 = 2 * x2 + (5 / 4) * x1 - 24.25
        return np.array([df1_dx1, df2_dx2]).reshape(-1, 1)

    # === Shared constraint x1 + x2 ≤ 15 ===
    @staticmethod
    def g_shared(x: VectorList) -> Vector:
        x1 = x[0]
        x2 = x[1]
        return np.array([x1 + x2 - 15]).reshape(-1, 1)

    @staticmethod
    def g_shared_der(x: VectorList) -> Vector:
        # derivative of (x1 + x2 - 15) wrt [x1, x2] = [1, 1]
        return np.array([[1, 1]]).reshape(-1, 1)


