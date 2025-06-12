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
        S = sum(x)
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

def A1_sig(vars):
    players = np.array(vars[:10]).reshape(-1,1)
    dual = vars[10:][0]
    lb = np.array([0.3,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]).reshape(-1,1)
    ub = np.array([0.5,1,1,1,1,1,1,1,1,1]).reshape(-1,1)

    # Transformed actions
    x = np.clip(players, -500, 500)
    actions = (ub - lb) * (1/(1+np.exp(-x))) + lb     # (10,1)
    dual_actions = 100/(1+np.exp(-dual))               # scalar

    # Gradients
    grad_obj = A1_BL.obj_func_der(actions)                      # (10,1)
    grad_cons = np.hstack(([0], np.ones(9, int))).reshape(-1,1) * dual_actions
    grad = grad_obj + grad_cons #* (actions * (1-actions) * (ub-lb)) # (10,1)
    # eng = (grad**2)/(1+grad**2)
    eng = np.abs(grad)

    grad_dual = -A1_BL.g0(actions) #* (dual_actions * (1-dual_actions)) * 100
    # eng_dual = (grad_dual**2)/(1+grad_dual**2)
    eng_dual = np.abs(grad_dual)
    return np.sum(eng) + eng_dual

def get_grad(vars, translate=False):
    players = np.array(vars[:10]).reshape(-1, 1)
    dual = vars[10:][0]
    lb = np.array([0.3, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]).reshape(-1, 1)
    ub = np.array([0.5, 1, 1, 1, 1, 1, 1, 1, 1, 1]).reshape(-1, 1)
    if translate:
        x = np.clip(players, -500, 500)
        players = (ub - lb) * (1 / (1 + np.exp(-x))) + lb  # (10,1)
        dual = 100 / (1 + np.exp(-dual))
    grad_obj = A1_BL.obj_func_der(players)  # (10,1)
    grad_cons = np.hstack(([0], np.ones(9, int))).reshape(-1, 1) * dual
    grad = (grad_obj + grad_cons)

    grad_dual = -A1_BL.g0(players)
    return grad, grad_dual


ip1=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0]
print(A1_sig(ip1))

isTransform = True

if isTransform:
    minimizer_kwargs = dict(method="L-BFGS-B")
    ip1 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
    start = timeit.default_timer()
    res1 = basinhopping(A1_sig, ip1, stepsize=0.0001, niter=1000, minimizer_kwargs=minimizer_kwargs, interval=1 ,
                        niter_success=100, disp=True)
    stop = timeit.default_timer()
    lb_t = np.array([0.3, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]).reshape(-1, 1)
    ub_t = np.array([0.5, 1, 1, 1, 1, 1, 1, 1, 1, 1]).reshape(-1, 1)
    x_t = np.clip(np.array(res1.x[:10]).reshape(-1, 1), -500, 500)
    xd_t = np.clip(res1.x[10:], -500, 500)
    actions_primal = (ub_t - lb_t) * (1 / (1 + np.exp(-x_t))) + lb_t
    actions_dual = 100 / (1 + np.exp(-xd_t))
    print('Result translated: ', actions_primal)
    print(actions_dual)

else:
    bounds = Bounds([0.3,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0], [0.5,1,1,1,1,1,1,1,1,1,10])
    minimizer_kwargs = dict(method="SLSQP", bounds=bounds)
    ip1=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0]
    start = timeit.default_timer()
    res1=basinhopping(A1_ex, ip1, stepsize=0.01, niter=1000, minimizer_kwargs=minimizer_kwargs, interval=1, niter_success=100, disp = True)
    stop = timeit.default_timer()


print("Result: ", res1.x)
if isTransform and actions_primal.any() and actions_dual.any():
    print('Primal Translated: ', actions_primal)
    print('Dual Translated: ', actions_dual)
print("Energy: ", A1_sig(res1.x) if isTransform else A1_ex(res1.x))
print("Gradients: ", get_grad(res1.x, translate=isTransform))
