
import numpy as np
from typing import List

Vector = np.ndarray
VectorList = List[np.ndarray]


class B2aU:
    @staticmethod
    def paper_solution():
        # One GNE / VI solution reported in the paper:
        # x* = (4/11, 7/11)^T
        value_1 = [4.0 / 11.0, 7.0 / 11.0]
        return [value_1]

    @staticmethod
    def define_players():
        # Two players, each with a scalar decision variable
        n = 2
        player_vector_sizes = [1 for _ in range(n)]
        # single objective mapping F(x) used for both players (index 0)
        player_objective_functions = [0 for _ in range(n)]
        # constraints indices for each player:
        # g0: x1 + x2 - 1 <= 0  (shared)
        # g1: -x1 <= 0          (x1 >= 0, player 1)
        # g2: -x2 <= 0          (x2 >= 0, player 2)
        # g3: dummy always-satisfied constraint
        player_constraints = [
            [0, 1],  # player 1 has g0, g1
            [0, 2]   # player 2 has g0, g2
        ]
        return [player_vector_sizes, player_objective_functions, player_constraints]

    @staticmethod
    def objective_functions():
        # Mapping F(x) = (2x1 - x2 - 1,  -1/2 x1 + 2x2 - 2)^T
        return [B2aU.obj_func]

    @staticmethod
    def objective_function_derivatives():
        # Diagonal of Jacobian of F with respect to each player's own variable
        return [B2aU.obj_func_der]

    @staticmethod
    def constraints():
        # g0: x1 + x2 - 1 <= 0  (shared)
        # g1: -x1 <= 0          (x1 >= 0)
        # g2: -x2 <= 0          (x2 >= 0)
        # g3: dummy constraint (always inactive)
        return [B2aU.g0,
                B2aU.g1,
                B2aU.g2,
                B2aU.g3]

    @staticmethod
    def constraint_derivatives():
        return [B2aU.g0_der,
                B2aU.g1_der,
                B2aU.g2_der,
                B2aU.g3_der]

    # === Objective Mapping F(x) ===
    @staticmethod
    def obj_func(x: VectorList) -> Vector:
        # x is a list of player vectors; concatenate to [x1, x2]^T
        x_full = np.concatenate(x).reshape(-1, 1)
        x1 = x_full[0, 0]
        x2 = x_full[1, 0]

        F1 = 2.0 * x1 - x2 - 1.0
        F2 = -0.5 * x1 + 2.0 * x2 - 2.0

        return np.array([[F1], [F2]]).reshape(-1, 1)

    @staticmethod
    def obj_func_der(x: VectorList) -> Vector:
        # For many algorithms only dF_i/dx_i is used:
        # ∂F1/∂x1 = 2, ∂F2/∂x2 = 2
        return np.array([[2.0], [2.0]]).reshape(-1, 1)

    # === Constraint Functions ===
    @staticmethod
    def g0(x: VectorList) -> Vector:
        # Shared coupling constraint: x1 + x2 <= 1  → x1 + x2 - 1 <= 0
        x_full = np.concatenate(x).reshape(-1, 1)
        x1 = x_full[0, 0]
        x2 = x_full[1, 0]
        return np.array([[x1 + x2 - 1.0]]).reshape(-1, 1)

    @staticmethod
    def g1(x: VectorList) -> Vector:
        # Nonnegativity of x1: -x1 <= 0
        x_full = np.concatenate(x).reshape(-1, 1)
        x1 = x_full[0, 0]
        return np.array([[-x1]]).reshape(-1, 1)

    @staticmethod
    def g2(x: VectorList) -> Vector:
        # Nonnegativity of x2: -x2 <= 0
        x_full = np.concatenate(x).reshape(-1, 1)
        x2 = x_full[1, 0]
        return np.array([[-x2]]).reshape(-1, 1)

    @staticmethod
    def g3(x: VectorList) -> Vector:
        # Dummy constraint, always strictly satisfied: -1 <= 0
        return np.array([[-1.0]]).reshape(-1, 1)

    # === Constraint Derivatives (w.r.t. each player's own variable) ===
    @staticmethod
    def g0_der(x: VectorList) -> Vector:
        # d(x1 + x2 - 1)/dx_i = 1 for each player
        return np.array([[1.0]]).reshape(-1, 1)

    @staticmethod
    def g1_der(x: VectorList) -> Vector:
        # d(-x1)/dx1 = -1
        return np.array([[-1.0]]).reshape(-1, 1)

    @staticmethod
    def g2_der(x: VectorList) -> Vector:
        # d(-x2)/dx2 = -1
        return np.array([[-1.0]]).reshape(-1, 1)

    @staticmethod
    def g3_der(x: VectorList) -> Vector:
        # derivative of dummy constraint is zero
        return np.array([[0.0]]).reshape(-1, 1)


