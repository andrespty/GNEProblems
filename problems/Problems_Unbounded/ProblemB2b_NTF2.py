import numpy as np
from typing import List

Vector = np.ndarray
VectorList = List[np.ndarray]

class B2bU:
    @staticmethod
    def paper_solution():
        """
        Entire GNE set:
            (x1, x2) = (t, sqrt(1 - t^2)),   0 ≤ t ≤ 4/5

        We discretize t on this interval to represent the continuum.
        """
        t_vals = np.linspace(0.0, 4.0/5.0, 41)  # 41 points
        sol_list = [[float(t), float(np.sqrt(1 - t**2))] for t in t_vals]
        return sol_list

    @staticmethod
    def define_players():
        """
        Two players controlling x1 and x2.

        Constraints:
        g0: x1^2 + x2^2 ≤ 1 (shared)
        g1: -x1 ≤ 0  (x1 ≥ 0)
        g2: -x2 ≤ 0  (x2 ≥ 0)
        """
        player_vector_sizes = [1, 1]
        player_objective_functions = [0, 0]
        player_constraints = [
            [0, 1],  # Player 1: shared + own lower bound
            [0, 2]   # Player 2: shared + own lower bound
        ]
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
        return [B2bU.g0, B2bU.g1, B2bU.g2]

    @staticmethod
    def constraint_derivatives():
        return [B2bU.g0_der, B2bU.g1_der, B2bU.g2_der]

    # === Objective functions (same as Example 1) ===
    @staticmethod
    def obj_func(x: VectorList) -> Vector:
        """
        f1(x1, x2) = x1^2 - x1*x2 - x1
        f2(x1, x2) = x2^2 - (1/2)*x1*x2 - 2*x2
        """
        x1 = x[0]
        x2 = x[1]

        f1 = x1**2 - x1 * x2 - x1
        f2 = x2**2 - 0.5 * x1 * x2 - 2 * x2
        return np.array([f1, f2]).reshape(-1, 1)

    @staticmethod
    def obj_func_der(x: VectorList) -> Vector:
        """
        ∂f1/∂x1 = 2x1 - x2 - 1
        ∂f2/∂x2 = 2x2 - (1/2)x1 - 2
        """
        x1 = x[0]
        x2 = x[1]

        df1_dx1 = 2 * x1 - x2 - 1
        df2_dx2 = 2 * x2 - 0.5 * x1 - 2

        return np.array([df1_dx1, df2_dx2]).reshape(-1, 1)

    # === Shared constraint: x1^2 + x2^2 ≤ 1 ===
    @staticmethod
    def g0(x: VectorList) -> Vector:
        x1 = x[0]
        x2 = x[1]
        return np.array([x1**2 + x2**2 - 1.0]).reshape(-1, 1)

    @staticmethod
    def g1(x: VectorList) -> Vector:
        # x1 ≥ 0 → -x1 ≤ 0
        x1 = x[0]
        return np.array([-x1]).reshape(-1, 1)

    @staticmethod
    def g2(x: VectorList) -> Vector:
        # x2 ≥ 0 → -x2 ≤ 0
        x2 = x[1]
        return np.array([-x2]).reshape(-1, 1)

    # === Constraint Derivatives ===
    @staticmethod
    def g0_der(x: VectorList) -> Vector:
        # derivative of x1^2 + x2^2 - 1 wrt [x1, x2] is [2x1, 2x2]^T
        x1 = x[0]
        x2 = x[1]
        return np.array([[2.0 * x1], [2.0 * x2]])

    @staticmethod
    def g1_der(x: VectorList) -> Vector:
        return np.array([[-1.0], [0.0]])

    @staticmethod
    def g2_der(x: VectorList) -> Vector:
        return np.array([[0.0], [-1.0]])
