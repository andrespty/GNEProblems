
import numpy as np
from typing import List

Vector = np.ndarray
VectorList = List[np.ndarray]


import numpy as np
from typing import List

Vector = np.ndarray
VectorList = List[np.ndarray]


import numpy as np
from typing import List

Vector = np.ndarray
VectorList = List[np.ndarray]


class B2aU:
    @staticmethod
    def paper_solution():
        # From the paper (Example 1):
        # GNE set: {(t, 1 - t) | 0 ≤ t ≤ 2/3}
        # The VI(F, X) solution is uniquely x* = (4/11, 7/11)^T,
        # which is also one of the GNEs. We return this representative point.
        value_1 = [4.0 / 11.0, 7.0 / 11.0]
        return [value_1]

    @staticmethod
    def define_players():
        # There are 2 players, each controlling 1 variable
        player_vector_sizes = [1, 1]          # x1 and x2
        player_objective_functions = [0, 0]   # Player 1 -> f1(x1,x2), Player 2 -> f2(x1,x2)

        # Shared constraint: x1 + x2 ≤ 1
        # (nonnegativity x1 ≥ 0, x2 ≥ 0 is not encoded here, mirroring the B1U style)
        player_constraints = [[0], [0]]       # Both share the same constraint
        return [player_vector_sizes, player_objective_functions, player_constraints]

    # === Objective Functions ===
    @staticmethod
    def objective_functions():
        return [B2aU.obj_func]

    @staticmethod
    def objective_function_derivatives():
        return [B2aU.obj_func_der]

    # === Constraints ===
    @staticmethod
    def constraints():
        return [B2aU.g_shared]

    @staticmethod
    def constraint_derivatives():
        return [B2aU.g_shared_der]

    # === Player 1 and Player 2 objective functions ===
    @staticmethod
    def obj_func(x: VectorList) -> Vector:
        """
        Example 1 objectives:

        Player 1:
            f1(x1, x2) = x1^2 - x1*x2 - x1

        Player 2:
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

    # === Shared constraint x1 + x2 ≤ 1 ===
    @staticmethod
    def g_shared(x: VectorList) -> Vector:
        x1 = x[0]
        x2 = x[1]
        return np.array([x1 + x2 - 1.0]).reshape(-1, 1)

    @staticmethod
    def g_shared_der(x: VectorList) -> Vector:
        # derivative of (x1 + x2 - 1) wrt [x1, x2] = [1, 1]
        return np.array([[1.0, 1.0]]).reshape(-1, 1)



