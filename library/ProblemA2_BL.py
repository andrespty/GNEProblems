import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import basinhopping
import timeit
from typing import List, Tuple, Dict, Optional, Callable
import numpy.typing as npt


class A2_BL:
    @staticmethod
    def paper_solution():
        value_1 = [0.29962894677774, 0.00997828224734, 0.00997828224734,
                   0.00997828224734, 0.59852469355630, 0.02187270661760,
                   0.00999093169361, 0.00999093169361, 0.00999093169361,
                   0.00999093169361]

        value_2 = [0.29962898846513, 0.00997828313762, 0.00997828313762,
                   0.00997828313762, 0.59745624992082, 0.02220301920403,
                   0.01013441012117, 0.01013441012117, 0.01013441012117,
                   0.01013441012117]

        return [value_1, value_2]

    @staticmethod
    def define_players():
        B = 1
        player_vector_sizes = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        player_objective_functions = [0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        player_constraints = [[None], [0], [0], [0], [0, 1], [0, 1], [0], [0], [0], [0]]
        bounds = [(0.3, 0.5), (0.01, B), (0.01, B), (0.01, B), (0.01, B), (0.01, B), (0.01, B), (0.01, B), (0.01, 0.06),
                  (0.01, 0.05), (0, 10), (0, 10)]
        bounds_training = [(0.3, 0.5), (0.01, B), (0.01, B), (0.01, B), (0.01, B), (0.01, B), (0.01, B), (0.01, B),
                           (0.01, 0.06), (0.01, 0.05), (0, 10), (0, 10)]
        return [player_vector_sizes, player_objective_functions, player_constraints, bounds, bounds_training]

    @staticmethod
    def objective_functions():
        return [A2_BL.obj_func_1, A2_BL.obj_func_2]

    @staticmethod
    def objective_function_derivatives():
        return [A2_BL.obj_func_der_1, A2_BL.obj_func_der_2]

    @staticmethod
    def constraints():
        return [A2_BL.g0, A2_BL.g1]

    @staticmethod
    def constraint_derivatives():
        return [A2_BL.g0_der, A2_BL.g1_der]

    @staticmethod
    def obj_func_1(x):
        # x: numpy array (N, 1)
        B = 1
        S = sum(x)
        obj = (-x / S) * (1 - S / B)
        return obj

    @staticmethod
    def obj_func_2(x):
        # x: numpy array (N,1)
        # B: constant
        B = 1
        S = sum(x)
        obj = (-x / S) * (1 - S / B) ** 2
        return obj

    @staticmethod
    def obj_func_der_1(x):
        # x: numpy array (N,1)
        B = 1
        S = sum(x)
        obj = (x - S) / S ** 2 + 1 / B
        return obj

    @staticmethod
    def obj_func_der_2(x):
        # x: numpy array (N,1)
        B = 1
        S = sum(x)
        obj = (2 * B * (S ** 2) - (B ** 2) * S - S ** 3 - x * (S ** 2) + x * (B ** 2)) / (S ** 2)
        return obj

    @staticmethod
    def g0(x):
        # x: numpy array (N,1)
        B = 1
        return x.sum() - B

    @staticmethod
    def g1(x):
        # x: numpy array (N,1)
        B = 1
        return 0.99 - x.sum()

    @staticmethod
    def g0_der(x):
        return 1

    @staticmethod
    def g1_der(x):
        return -1

def get_engval(myvar, gradval, mylb, myub):
  if gradval<=0:
    engval=(myub-myvar)*np.log(1-gradval)
  else:
    engval=(myvar-mylb)*np.log(1+gradval)
  return engval

def A2_ex(vars):
    players = np.array(vars[:10]).reshape(-1,1)
    dual = vars[10:]
    lb = np.array([0.3,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]).reshape(-1,1)
    ub = np.array([0.5,1,1,1,1,1,1,1,0.06,0.05]).reshape(-1,1)
    cost1 = A2_BL.obj_func_der_1(players)
    cost2 = A2_BL.obj_func_der_2(players)
    grad = cost1
    grad[1:5] = cost2[1:5]
    grad.reshape(-1, 1)
    grad_cons_1 = np.hstack(([0], np.ones(9, int))).reshape(-1, 1) * dual[0]
    grad_cons_2 = np.hstack((np.zeros(4), np.ones(2, int), np.zeros(4))).reshape(-1, 1) * -dual[1]
    grad += grad_cons_1 + grad_cons_2
    eng = np.where(
        grad<=0,
        (ub-players) * np.log(1-grad),
        (players-lb) * np.log(1+grad)
    )

    cost_dual_1 = -A2_BL.g0(players)
    cost_dual_2 = -A2_BL.g1(players)
    eng_dual_1 = get_engval(dual[0], cost_dual_1, 0, 100)
    eng_dual_2 = get_engval(dual[1], cost_dual_2, 0, 100)
    return sum(eng) + eng_dual_1 + eng_dual_2

def A2_sig(vars):
    players = np.array(vars[:10]).reshape(-1,1)
    dual = np.array(vars[10:]).reshape(-1,1)
    lb = np.array([0.3,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]).reshape(-1,1)
    ub = np.array([0.5,1,1,1,1,1,1,1,0.06,0.05]).reshape(-1,1)

    # Transformed actions
    x = np.clip(players, -500, 500)
    xd = np.clip(dual, -500, 500)
    actions = (ub - lb) * (1/(1+np.exp(-x))) + lb     # (10,1)
    dual_actions = 100/(1+np.exp(-xd))               # (2,1)

    # Gradients
    grad_obj1 = A2_BL.obj_func_der_1(actions)                      # (10,1)
    grad_obj2 = A2_BL.obj_func_der_2(actions)                      # (10,1)
    grad_obj = np.copy(grad_obj1)
    grad_obj[1:5] += grad_obj2[1:5]
    grad_cons_1 = np.hstack(([0], np.ones(9, int))).reshape(-1,1) * dual_actions[0]
    grad_cons_2 = np.hstack((np.zeros(4), np.ones(2, int), np.zeros(4))).reshape(-1, 1) * -dual_actions[1]
    grad = grad_obj + grad_cons_1 + grad_cons_2
    # eng = (grad**2)/(1+grad**2)
    eng = np.abs(grad)

    grad_dual_1 = -A2_BL.g0(actions) #* (dual_actions * (1-dual_actions)) * 100
    grad_dual_2 = -A2_BL.g1(actions) #* (dual_actions * (1-dual_actions)) * 100
    # eng_dual = (grad_dual**2)/(1+grad_dual**2)
    eng_dual_1 = np.abs(grad_dual_1)
    eng_dual_2 = np.abs(grad_dual_2)
    return np.sum(eng) + eng_dual_1 + eng_dual_2

def get_grad(vars, translate=False):
    players = np.array(vars[:10]).reshape(-1, 1)
    dual = np.array(vars[10:]).reshape(-1,1)
    lb = np.array([0.3, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]).reshape(-1, 1)
    ub = np.array([0.5,1,1,1,1,1,1,1,0.06, 0.05]).reshape(-1, 1)
    if translate:
        x = np.clip(players, -500, 500)
        players = (ub - lb) * (1 / (1 + np.exp(-x))) + lb  # (10,1)
        dual = 100 / (1 + np.exp(-dual))
    grad_obj1 = A2_BL.obj_func_der_1(players)  # (10,1)
    grad_obj2 = A2_BL.obj_func_der_2(players)  # (10,1)
    grad_obj = np.copy(grad_obj1)
    grad_obj[1:5] += grad_obj2[1:5]
    grad_cons_1 = np.hstack(([0], np.ones(9, int))).reshape(-1, 1) * dual[0]
    grad_cons_2 = np.hstack((np.zeros(4), np.ones(2, int), np.zeros(4))).reshape(-1, 1) * -dual[1]
    grad = grad_obj + grad_cons_1 + grad_cons_2

    grad_dual_1 = -A2_BL.g0(players)  # * (dual_actions * (1-dual_actions)) * 100
    grad_dual_2 = -A2_BL.g1(players)
    return grad, grad_dual_1, grad_dual_2

ip1=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 10, 10]
print(A2_ex(ip1))

isTransform = True

if isTransform:
    minimizer_kwargs = dict(method="L-BFGS-B")
    ip1 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
    start = timeit.default_timer()
    res1 = basinhopping(A2_sig, ip1, stepsize=0.0001, niter=1000, minimizer_kwargs=minimizer_kwargs, interval=1 ,
                        niter_success=100, disp=True)
    stop = timeit.default_timer()
    lb_t = np.array([0.3, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]).reshape(-1, 1)
    ub_t = np.array([0.5,1,1,1,1,1,1,1,0.06, 0.05]).reshape(-1, 1)
    x_t = np.clip(np.array(res1.x[:10]).reshape(-1, 1), -500, 500)
    xd_t = np.clip(res1.x[10:], -500, 500)
    actions_primal = (ub_t - lb_t) * (1 / (1 + np.exp(-x_t))) + lb_t
    actions_dual = 100 / (1 + np.exp(-xd_t))
    print('Result translated: ', actions_primal)
    print(actions_dual)

else:
    bounds = Bounds([0.3,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0, 0], [0.5,1,1,1,1,1,1,1,0.06,0.05,100, 100])
    minimizer_kwargs = dict(method="SLSQP", bounds=bounds)
    ip1=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0, 0]
    start = timeit.default_timer()
    res1=basinhopping(A2_ex, ip1, stepsize=0.01, niter=1000, minimizer_kwargs=minimizer_kwargs, interval=1, niter_success=100, disp = True)
    stop = timeit.default_timer()


print("Result: ", res1.x)
if isTransform and actions_primal.any() and actions_dual.any():
    print('Primal Translated: ', actions_primal)
    print('Dual Translated: ', actions_dual)
    print(sum(actions_primal))
    # print(A2_BL.g0(actions_primal))
    # print(A2_BL.g1(actions_primal))
print("Energy: ", A2_sig(res1.x) if isTransform else A2_ex(res1.x))
print("Gradients: ", get_grad(res1.x, translate=isTransform))
