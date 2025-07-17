import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import basinhopping
import timeit
from typing import List, Tuple, Dict, Optional, Callable
import numpy.typing as npt

from library.ProblemA7_BL import A7_sig

class A1_BL:
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
        player_constraints = [[None], [0], [0], [0], [0], [0], [0], [0], [0], [0]]
        bounds = [(0.3, 0.5), (0.01, B), (0.01, B), (0.01, B), (0.01, B), (0.01, B), (0.01, B), (0.01, B), (0.01, B),
                  (0.01, B), (0, 10)]
        bounds_training = [(0.3, 0.5), (0.01, B), (0.01, B), (0.01, B), (0.01, B), (0.01, B), (0.01, B), (0.01, B),
                           (0.01, B), (0.01, B), (0, 10)]
        return [player_vector_sizes, player_objective_functions, player_constraints, bounds, bounds_training]

    @staticmethod
    def objective_functions():
        return [A1_BL.obj_func]

    @staticmethod
    def objective_function_derivatives():
        return [A1_BL.obj_func_der]

    @staticmethod
    def constraints():
        return [A1_BL.g0]

    @staticmethod
    def constraint_derivatives():
        return [A1_BL.g0_der]

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
        print(x)
        S = sum(x)
        print("S: ", S)
        obj = ((x - S) / (S ** 2)) + (1 / B)
        return obj

    # === Constraint Functions ===
    @staticmethod
    def g0(x):
        # x: numpy array (N,1)
        # B: constant
        B=1
        return x.sum() - B

    @staticmethod
    def g1(x):
        return 0.3 - x[0]
    @staticmethod
    def g2(x):
        return x[0] - 0.5
    @staticmethod
    def g3(x):
        mask = np.ones(x.shape, dtype=bool)
        mask[0] = False
        return 0.01 - x[mask]

    @staticmethod
    def g0_der(x):
        return 1

def get_engval(myvar, gradval, mylb, myub):
  if gradval<=0:
    engval=(myub-myvar)*np.log(1-gradval)
  else:
    engval=(myvar-mylb)*np.log(1+gradval)
  return engval

def A1_ex(vars):
    players = np.array(vars[:10])
    dual = vars[10:]
    lb = [0.3,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]
    ub = [0.5,1,1,1,1,1,1,1,1,1]
    cost = A1_BL.obj_func_der(players)
    cost[1:] += dual[0]

    eng = [get_engval(player, cost[i], lb[i], ub[i]) for i, player in enumerate(players)]
    cost_dual = -A1_BL.g0(players)
    eng_dual = get_engval(dual[0], cost_dual, 0, 100)
    return sum(eng) + eng_dual

def A1_eng(vars):
    players = np.array(vars[:10]).reshape(-1,1)
    dual = np.array(vars[10:]).reshape(-1,1)
    # Gradients
    grad, grad_dual = get_grad(vars) # grad = (10,1), grad_dual = (4,1)
    print(grad)
    eng = grad**2
    # eng_dual = np.where(
    #     grad_dual <= 0,
    #     grad_dual**2,
    #     grad_dual**2 * np.tanh(dual)**2
    # )
    # eng_dual = (dual**2/(1+dual**2)) * (grad_dual**2/(1+grad_dual**2)) + np.exp(-dual**2) * (np.maximum(0,-grad_dual)**2/(1+np.maximum(0,-grad_dual)**2))
    # eng_dual = dual**2 * grad_dual**2 + np.maximum(0,-grad_dual)**2

    return np.sum(eng) + np.sum(grad_dual)


def get_grad(vars):
    players = np.array(vars[:10]).reshape(-1, 1)
    dual = np.array(vars[10:]).reshape(-1, 1)  # (12,1)

    print(players, dual)
    grad_obj = A1_BL.obj_func_der(players)  # (10,1)
    print(grad_obj)
    grad_cons_0 = np.hstack(([0], np.ones(9, int))).reshape(-1, 1) * dual[0]
    grad_cons_1 = np.hstack(([-1], np.zeros(9, int))).reshape(-1, 1) * dual[1]
    grad_cons_2 = np.hstack(([1], np.zeros(9, int))).reshape(-1, 1) * dual[2]
    grad_cons_3 = np.vstack((np.zeros((1,9), dtype=int), -np.identity(9, dtype=int))) * dual[3]
    grad = (grad_obj + grad_cons_0 + grad_cons_1 + grad_cons_2 + np.sum(grad_cons_3, axis=1).reshape(-1,1))

    grad_dual = []
    constraints = [A1_BL.g0, A1_BL.g1, A1_BL.g2, A1_BL.g3]
    for jdx, constraint in enumerate(constraints):
        g = -constraint(players)
        g = (dual[jdx]**2/(1+dual[jdx]**2)) * (g**2/(1+g**2)) + np.exp(-dual[jdx]**2) * (np.maximum(0,-g)**2/(1+np.maximum(0,-g)**2))
        grad_dual.append(g.flatten())
    g_dual = np.concatenate(grad_dual).reshape(-1, 1)
    return grad, g_dual


player_vars = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
constraint_vars = [0, 0, 0, 0]
# constraint_vars = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]


ip1 = player_vars + constraint_vars
print(A1_eng(ip1))
minimizer_kwargs = dict(method="L-BFGS-B")

optimize = False

if optimize:
    minimizer_kwargs = dict(method="L-BFGS-B")
    player_vars = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
    constraint_vars = [0, 0, 0, 0]#, 0, 0, 0, 0, 0, 0, 0, 0]
    ip1=player_vars + constraint_vars
    start = timeit.default_timer()
    res1=basinhopping(A1_eng, ip1, stepsize=0.01, niter=1000, minimizer_kwargs=minimizer_kwargs, interval=1, niter_success=100, disp = True)
    stop = timeit.default_timer()

print("Result: ", res1.x[:10])
print("Time: ", stop - start)
print("Constraints: ", res1.x[10:])

print("Energy: ", A1_eng(res1.x))
print("OG Energy: ", A1_ex(res1.x[:12]))
