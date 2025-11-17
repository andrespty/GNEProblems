import numpy as np
from typing import List

Vector = np.ndarray
VectorList = List[np.ndarray]

class B2bU:
    @staticmethod
    def paper_solution():
        """
        Full GNE set for Example 2:
            (x1, x2) = (t, sqrt(1 - t^2)),   0 <= t <= 4/5

        We return a discretized set of points along this curve.
        """
        # Sample t in [0, 4/5]
        t_vals = np.linspace(0.0, 4.0 / 5.0, 41)  # 41 points
        sol_list = [[float(t), float(np.sqrt(1.0 - t**2))] for t in t_vals]
        return sol_list

    @staticmethod
    def define_players():
        # There are 2 players, each controlling 1 variable (x1 and x2)
        player_vector_sizes = [1, 1]
        player_objective_functions = [0, 0]  # Player 1 -> f1, Player 2 -> f2

        # Shared quadratic constraint: x1^2 + x2^2 ≤ 1
        player_constraints = [[0], [0]]  # Both share the same constraint
        return [player_vector_sizes, player_objective_functions, player_constraints]

    # === Objective Functions ===
    @staticmethod
    def objective_functions():
        return [B2bU.obj_func]

    @staticmethod
    def objective_function_derivatives():
        return [B2bU.obj_func_der]

    # === Constraints ===
    @staticmethod
    def constraints():
        return [B2bU.g_shared]

    @staticmethod
    def constraint_derivatives():
        return [B2bU.g_shared_der]

    # === Player 1 and Player 2 objective functions (same as Example 1) ===
    @staticmethod
    def obj_func(x: VectorList) -> Vector:
        """
        f1(x1, x2) = x1^2 - x1*x2 - x1
        f2(x1, x2) = x2^2 - (1/2)*x1*x2 - 2*x2
        """
        x1 = x[0]
        x2 = x[1]

        f1 = x1**2 - x1 * x2 - x1
        f2 = x2**2 - 0.5 * x1 * x2 - 2.0 * x2
        return np.array([f1, f2]).reshape(-1, 1)

    @staticmethod
    def obj_func_der(x: VectorList) -> Vector:
        """
        Gradient of each player's objective with respect to their own variable:

        ∂f1/∂x1 = 2x1 - x2 - 1
        ∂f2/∂x2 = 2x2 - (1/2)x1 - 2
        """
        x1 = x[0]
        x2 = x[1]

        df1_dx1 = 2.0 * x1 - x2 - 1.0
        df2_dx2 = 2.0 * x2 - 0.5 * x1 - 2.0
        return np.array([df1_dx1, df2_dx2]).reshape(-1, 1)

    # === Shared quadratic constraint x1^2 + x2^2 ≤ 1 ===
    @staticmethod
    def g_shared(x: VectorList) -> Vector:
        x1 = x[0]
        x2 = x[1]
        return np.array([x1**2 + x2**2 - 1.0]).reshape(-1, 1)

    @staticmethod
    def g_shared_der(x: VectorList) -> Vector:
        # derivative of (x1^2 + x2^2 - 1) wrt [x1, x2] = [2x1, 2x2]
        x1 = x[0]
        x2 = x[1]
        return np.array([[2.0 * x1, 2.0 * x2]]).reshape(-1, 1)

