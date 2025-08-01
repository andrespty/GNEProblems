import numpy as np
from scipy.optimize import basinhopping
import timeit


class A1dev:
    @staticmethod
    def paper_solution():
        value_1 = [0.29923815223336,
                   0.06951127617805,
                   0.06951127617805,
                   0.06951127617805,
                   0.06951127617805,
                   0.06951127617805,
                   0.06951127617805,
                   0.06951127617805,
                   0.06951127617805,
                   0.06951127617805]
        return [value_1]

    @staticmethod
    def define_players():
        B = 1
        player_vector_sizes = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        player_objective_functions = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # change to all 0s
        player_constraints = [[1,2], [0,3], [0,3], [0,3], [0,3], [0,3], [0,3], [0,3], [0,3], [0,3]]
        return [player_vector_sizes, player_objective_functions, player_constraints]

    @staticmethod
    def objective_functions():
        return [A1dev.obj_func]

    @staticmethod
    def objective_function_derivatives():
        return [A1dev.obj_func_der]

    @staticmethod
    def constraints():
        return [A1dev.g0, A1dev.g1, A1dev.g2, A1dev.g3]

    @staticmethod
    def constraint_derivatives():
        return [A1dev.g0_der, A1dev.g1_der, A1dev.g2_der, A1dev.g3_der]

    @staticmethod
    def obj_func(x):
        # x: numpy array (N, 1)
        # B: constant
        S = sum(x)
        B=1
        obj = (-x / S) * (1 - S / B)
        return obj

    @staticmethod
    def obj_func_der(x):
        # x: numpy array (N,1)
        # B: constant
        B=1
        x = np.concatenate(x).reshape(-1, 1)
        S = np.sum(x)
        obj = ((x - S) / (S ** 2)) + (1 / B)
        return obj

    # === Constraint Functions ===
    @staticmethod
    def g0(x):
        # x: numpy array (N,1)
        # B: constant
        B=1
        x = np.concatenate(x).reshape(-1, 1)
        return x.sum() - B

    @staticmethod
    def g1(x):
        return 0.3 - x[0]

    @staticmethod
    def g2(x):
        return x[0] - 0.5

    @staticmethod
    def g3(x):
        x = np.concatenate(x).reshape(-1, 1)
        return 0.01 - x[1:]

    @staticmethod
    def g0_der(x):
        return 1

    @staticmethod
    def g1_der(x):
        return -1

    @staticmethod
    def g2_der(x):
        return 1

    @staticmethod
    def g3_der(x):
        return -1

def A1devrun(vars):
    players = np.array(vars[:10]).reshape(-1,1)
    dual = np.array(vars[10:]).reshape(-1,1)
    constraints = A1dev.constraints()
    # Gradients
    grad_obj = A1dev.obj_func_der(players)  # (10,1)
    grad_cons_0 = A1dev.g0_der(players) * np.hstack(([0], np.ones(9, int))).reshape(-1, 1) * dual[0]
    grad_cons_1 = A1dev.g1_der(players) * np.hstack(([1], np.zeros(9, int))).reshape(-1, 1) * dual[1]
    grad_cons_2 = A1dev.g2_der(players) * np.hstack(([1], np.zeros(9, int))).reshape(-1, 1) * dual[2]
    grad_cons_3 = A1dev.g3_der(players) * np.vstack((np.zeros((1, 9), dtype=int), np.identity(9, dtype=int))) * dual[3]
    grad = (grad_obj + grad_cons_0 + grad_cons_1 + grad_cons_2 + np.sum(grad_cons_3, axis=1).reshape(-1, 1))

    eng = grad.T @ grad

    # Dual player
    grad_dual = []
    for jdx, constraint in enumerate(constraints):
        g = -constraint(players)
        g = (dual[jdx] ** 2 / (1 + dual[jdx] ** 2)) * (g ** 2 / (1 + g ** 2)) + np.exp(-dual[jdx] ** 2) * (
                    np.maximum(0, -g) ** 2 / (1 + np.maximum(0, -g) ** 2))
        grad_dual.append(g.flatten())
    g_dual = np.concatenate(grad_dual).reshape(-1, 1)

    return np.sum(eng) + np.sum(g_dual)

run = True
if run:
    minimizer_kwargs = dict(method="L-BFGS-B")
    player_vars = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
    constraint_vars = [1, 1, 1, 1]
    ip1=player_vars + constraint_vars
    start = timeit.default_timer()
    res1=basinhopping(A1devrun, ip1, stepsize=0.01, niter=1000, minimizer_kwargs=minimizer_kwargs, interval=1, niter_success=100, disp = True)
    stop = timeit.default_timer()

    print("Result: ", res1.x[:10])
    print("Time: ", stop - start)
    print("Constraints: ", res1.x[10:])

    print("Energy: ", A1devrun(res1.x))
