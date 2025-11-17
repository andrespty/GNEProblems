
import numpy as np
from typing import List

Vector = np.ndarray
VectorList = List[np.ndarray]

class B2aU:
    @staticmethod
    def paper_solution():
        """
        From the paper's VI solution:
        x* = (4/11, 7/11)^T
        """
        value_1 = [4.0 / 11.0, 7.0 / 11.0]
        return [value_1]

    @staticmethod
    def define_players():
        """
        Two players, each controlling one variable:
        Player 1: x1
        Player 2: x2

        Constraints:
        g0: x1 + x2 ≤ 1      (shared)
        g1: x1 ≥ 0 → -x1 ≤ 0
        g2: x2 ≥ 0 → -x2 ≤ 0
        """
        player_vector_sizes = [1, 1]
        player_objective_functions = [0, 0]  # both use obj_func()
        player_constraints = [
            [0, 1],  # Player 1: g0, g1
            [0, 2]   # Player 2: g0, g2
        ]
        return [player_vector_sizes, player_objective_functions, player_constraints]

    # === Objective Functions (Shared block) ===
    @staticmethod
    def objective_functions():
        return [B2aU.obj_func]

    @staticmethod
    def objective_function_derivatives():
        return [B2aU.obj_func_der]

    # === Constraints ===
    @staticmethod
    def constraints():
        return [B2aU.g0, B2aU.g1, B2aU.g2]

    @staticmethod
    def constraint_derivatives():
        return [B2aU.g0_der, B2aU.g1_der, B2aU.g2_der]

    # === Objective Functions ===
    @staticmethod
    def obj_func(x: VectorList) -> Vector:
        r"""
        Example 1 from the problem:

        Player 1:
            f1(x1,x2) = x1^2 - x1 x2 - x1

        Player 2:
            f2(x1,x2) = x2^2 - (1/2)x1 x2 - 2x2
        """
        x1 = x[0]
        x2 = x[1]

        f1 = x1**2 - x1 * x2 - x1
        f2 = x2**2 - 0.5 * x1 * x2 - 2.0 * x2
        return np.array([f1, f2]).reshape(-1, 1)

    @staticmethod
    def obj_func_der(x: VectorList) -> Vector:
        r"""
        Gradients:

        ∂f1/∂x1 = 2x1 - x2 - 1
        ∂f2/∂x2 = 2x2 - (1/2)x1 - 2
        """
        x1 = x[0]
        x2 = x[1]

        df1_dx1 = 2.0 * x1 - x2 - 1.0
        df2_dx2 = 2.0 * x2 - 0.5 * x1 - 2.0

        return np.array([df1_dx1, df2_dx2]).reshape(-1, 1)

    # === Constraints ===
    @staticmethod
    def g0(x: VectorList) -> Vector:
        """
        Shared constraint:
            x1 + x2 ≤ 1  → x1 + x2 - 1 ≤ 0
        """
        x1 = x[0]
        x2 = x[1]
        return np.array([x1 + x2 - 1.0]).reshape(-1, 1)

    @staticmethod
    def g1(x: VectorList) -> Vector:
        """
        Player 1 nonnegativity:
            x1 ≥ 0 → -x1 ≤ 0
        """
        x1 = x[0]
        return np.array([-x1]).reshape(-1, 1)

    @staticmethod
    def g2(x: VectorList) -> Vector:
        """
        Player 2 nonnegativity:
            x2 ≥ 0 → -x2 ≤ 0
        """
        x2 = x[1]
        return np.array([-x2]).reshape(-1, 1)

    # === Constraint Derivatives ===
    @staticmethod
    def g0_der(x: VectorList) -> Vector:
        # ∂(x1 + x2 - 1)/∂[x1, x2] = [1, 1]^T
        return np.array([[1.0], [1.0]])

    @staticmethod
    def g1_der(x: VectorList) -> Vector:
        # ∂(-x1)/∂[x1, x2] = [-1, 0]^T
        return np.array([[-1.0], [0.0]])

    @staticmethod
    def g2_der(x: VectorList) -> Vector:
        # ∂(-x2)/∂[x1, x2] = [0, -1]^T
        return np.array([[0.0], [-1.0]])
