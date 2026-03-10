import jax.numpy as jnp
from gnep_solver import Vector, VectorList
from gnep_solver.BaseProblem import BaseProblem
from gnep_solver.Player import Player

class ProblemA15(BaseProblem):
    def known_solution(self):
        value_1 =  [46.66150692423980, 32.15293850189938, 15.00419467998705, 22.10485810522063, 12.34076570922471, 12.34076570922471]
        return value_1

    def define_players(self):
        player_vector_sizes = [1, 1, 1, 1, 1, 1]
        player_objective_functions = [0, 1, 1, 2, 2, 2]
        player_constraints = [[None] for _ in range(6)]
        bounds = [(0, 80), (0, 80), (0, 50), (0, 55), (0, 30), (0, 40)]
        return Player.batch_create(
            player_vector_sizes,
            player_objective_functions,
            player_constraints,
            bounds
        )

    def objectives(self):
        def obj_func_1(x: VectorList) -> jnp.ndarray:
            x1 = x[0]
            x2 = x[1]
            x3 = x[2]
            x4 = x[3]
            x5 = x[4]
            x6 = x[5]

            S = 2 * (x1 + x2 + x3 + x4 + x5 + x6) - 378.4
            c1 = 0.04
            d1 = 2.0
            e1 = 0.0
            obj = S * x1 + (0.5 * c1 * x1 ** 2 + d1 * x1 + e1)
            return obj

        def obj_func_2(x: VectorList) -> jnp.ndarray:
            x1 = x[0]
            x2 = x[1]
            x3 = x[2]
            x4 = x[3]
            x5 = x[4]
            x6 = x[5]

            S = 2 * (x1 + x2 + x3 + x4 + x5 + x6) - 378.4
            v2 = jnp.concatenate([x2.ravel(), x3.ravel()]).reshape(-1, 1)
            c2 = jnp.array([0.035, 0.125]).reshape(-1,1)
            d2 = jnp.array([1.75, 1]).reshape(-1,1)
            e2 = jnp.array([0.0, 0.0]).reshape(-1,1)
            obj = S * (x2 + x3) + jnp.sum(0.5 * c2 * v2 ** 2 + d2 * v2 + e2)
            return obj

        def obj_func_3(x: VectorList) -> jnp.ndarray:
            x1 = x[0]
            x2 = x[1]
            x3 = x[2]
            x4 = x[3]
            x5 = x[4]
            x6 = x[5]

            S = 2 * (x1 + x2 + x3 + x4 + x5 + x6) - 378.4
            v3 = jnp.concatenate([x4.ravel(), x5.ravel(), x6.ravel()]).reshape(-1, 1)
            c3 = jnp.array([0.0166, 0.05, 0.05]).reshape(-1,1)
            d3 = jnp.array([3.25, 3.0, 3.0]).reshape(-1,1)
            e3 = jnp.array([0.0, 0.0, 0.0]).reshape(-1,1)
            obj = S * (x4 + x5 + x6) + jnp.sum(0.5 * c3 * v3 ** 2 + d3 * v3 + e3)
            return obj

        return [obj_func_1, obj_func_2, obj_func_3]

    def constraints(self):
        return []
