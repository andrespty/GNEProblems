import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import basinhopping
import timeit
from typing import List, Tuple, Dict, Optional, Callable
import numpy.typing as npt

from GNESolver5 import check_nash_equillibrium


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
        return sum(x) - B

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
    def g4(x):
        # x: numpy array (N,1)
        B = 1
        return 0.99 - sum(x)
    @staticmethod
    def g5(x):
        return x[8] - 0.06
    @staticmethod
    def g6(x):
        return x[9] - 0.05

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

def fisher_func(a,b, epsilon=1e-6):
    return a+b - np.sqrt(a**2 + b**2)

def A2_fb(vars):
    players = np.array(vars[:10]).reshape(-1,1)                 # (10,1)
    dual_constraints = np.array(vars[10:]).reshape(-1,1)        # (7,1)
    player_constraints = [[1, 2], [0, 3], [0, 3], [0, 3], [0, 3, 4], [0, 3, 4], [0, 3], [0, 3], [0, 3, 5], [0, 3, 6]]
    grad_constraints = np.array([1, -1, 1, -1, -1, 1, 1]).reshape(-1, 1)
    constraints = [A2_BL.g0, A2_BL.g1, A2_BL.g2, A2_BL.g3,A2_BL.g4,A2_BL.g5,A2_BL.g6]
    # Gradients
    grad_obj1 = A2_BL.obj_func_der_1(players)                      # (10,1)
    grad_obj2 = A2_BL.obj_func_der_2(players)                      # (10,1)
    grad_obj = np.copy(grad_obj1)
    grad_obj[1:5] = grad_obj2[1:5]
    print(grad_obj)
    for idx, player_const in enumerate(player_constraints):
        for jdx in player_const:
            grad_obj[idx] += dual_constraints[jdx] * grad_constraints[jdx]
    eng = grad_obj.T @ grad_obj

    fisher_res = []
    for jdx, constraint in enumerate(constraints):
        fisher = fisher_func(-constraint(players), dual_constraints[jdx])
        fisher_res.append(fisher.flatten())
    fisher_res = np.concatenate(fisher_res).reshape(-1, 1)
    eng_fisher = fisher_res.T @ fisher_res

    return eng + eng_fisher

def A2_eng(vars):
    players = np.array(vars[:10]).reshape(-1,1)                 # (10,1)
    dual_constraints = np.array(vars[10:]).reshape(-1,1)
    player_constraints = [[1, 2], [0, 3], [0, 3], [0, 3], [0, 3, 4], [0, 3, 4], [0, 3], [0, 3], [0, 3, 5], [0, 3, 6]]
    grad_constraints = np.array([1, -1, 1, -1, -1, 1, 1]).reshape(-1, 1)
    constraints = [A2_BL.g0, A2_BL.g1, A2_BL.g2, A2_BL.g3, A2_BL.g4, A2_BL.g5, A2_BL.g6]

    grad_obj1 = A2_BL.obj_func_der_1(players)  # (10,1)
    grad_obj2 = A2_BL.obj_func_der_2(players)  # (10,1)
    grad_obj = np.copy(grad_obj1)
    grad_obj[1:5] = grad_obj2[1:5]
    print(grad_obj)
    for idx, player_const in enumerate(player_constraints):
        for jdx in player_const:
            grad_obj[idx] += dual_constraints[jdx] * grad_constraints[jdx]
    eng = grad_obj.T @ grad_obj

    grad_dual = []
    for jdx, constraint in enumerate(constraints):
        g = -constraint(players)
        # g = np.where(
        #     g <= 0,
        #     g**2,
        #     g**2 * np.tanh(dual_constraints[jdx])
        # )
        g = (dual_constraints[jdx] ** 2 / (1 + dual_constraints[jdx] ** 2)) * (g ** 2 / (1 + g ** 2)) + np.exp(-dual_constraints[jdx] ** 2) * (
                    np.maximum(0, -g) ** 2 / (1 + np.maximum(0, -g) ** 2))
        grad_dual.append(g.flatten())
    g_dual = np.concatenate(grad_dual).reshape(-1, 1)

    return eng + np.sum(g_dual)

def obj(x):
    players = np.array(x).reshape(-1,1)
    obj1 = A2_BL.obj_func_1(players)  # (10,1)
    obj2 = A2_BL.obj_func_2(players)  # (10,1)
    o = np.copy(obj1)
    o[1:5] = obj2[1:5]
    return o

def wrap_func(func, fixed_x, idx):
    def w_obj(p_var):
        player_var_opt = np.array(p_var).reshape(-1, 1).item()
        new_vars = np.array(fixed_x[:idx] + [player_var_opt] + fixed_x[idx + 1:]).reshape(-1, 1)
        return func(new_vars)
    return w_obj

def wrap_obj_func(func, fixed_x, idx):
    def w_obj(p_var):
        player_var_opt = np.array(p_var).reshape(-1, 1).item()
        new_vars = np.array(fixed_x[:idx] + [player_var_opt] + fixed_x[idx + 1:]).reshape(-1, 1)
        return func(new_vars)[idx]
    return w_obj


def NE_check(x_star):
    # x_star is a list of only player actions
    print('NE check')
    conclusion = np.zeros_like(x_star).reshape(-1, 1)
    C = [A2_BL.g0, A2_BL.g4]
    C_i = [[None], [0], [0], [0], [0, 1], [0, 1], [0], [0], [0], [0]]
    print("x_star: ", x_star)
    for idx, player in enumerate(x_star):
        x_i = 0
        optimization_constraints = []
        for cdx in C_i[idx]:
            if cdx is not None:
                optimization_constraints.append(
                    {'type': 'ineq', 'fun': lambda x: wrap_func(C[cdx], x_star, idx)(x)}
                )

        wrapped_obj = wrap_obj_func(obj, x_star, idx)
        bounds = [(0.3, 0.5), (0.01, 1.0), (0.01, 1.0), (0.01, 1.0), (0.01, 1.0), (0.01, 1.0), (0.01, 1.0), (0.01, 1.0),
                  (0.01, 0.06), (0.01, 0.05)]
        player_bounds = [bounds[idx]]
        result = minimize(
            wrapped_obj,
            x_i,
            method='SLSQP',
            bounds=player_bounds,
            constraints=optimization_constraints,
            options={
                'disp': False,
                'maxiter': 1000,
                'ftol': 1e-5,
            }
        )
        # Report
        fixed_vars = x_star[:idx] + x_star[idx + 1:]
        opt_actions = np.array(result.x).reshape(-1, 1).item()
        new_vars = fixed_vars[:idx] + [opt_actions] + fixed_vars[idx:]
        opt_obj_func = obj(new_vars)
        comp_obj_func = obj(x_star)
        print(f"Player {idx + 1}")
        print(f"Computed Actions: {x_star[idx]}")
        print(f"Optimized Actions: {result.x}\nOptimized Objective Function: {opt_obj_func[idx]}")
        print('Computed Objective Functions: ', np.array(comp_obj_func).reshape(-1, 1)[idx])
        print('Optimized Constraints: ', constraints_check(new_vars, idx))
        print('Computed Constraints: ', constraints_check(x_star, idx))

        print()
        conclusion[idx] = np.array(comp_obj_func).reshape(-1, 1)[idx] - opt_obj_func[idx] > 1e-3
    print(conclusion)
    return

def constraints_check(x_star, idx):
    C = [A2_BL.g0, A2_BL.g4]
    C_i = [[None], [0], [0], [0], [0, 1], [0, 1], [0], [0], [0], [0]]
    result = []
    for cdx in C_i[idx]:
        if cdx is not None:
            result.append(C[cdx](x_star))

    return result

ip1=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
players_ip1 = [0, 0, 0, 0, 0, 0, 0]

ip1 = ip1 + players_ip1
print("Initial point: ",ip1)
print('Energy: ',A2_eng(ip1))

optimize = False
if optimize:
    minimizer_kwargs = dict(method="L-BFGS-B")
    start = timeit.default_timer()
    res1 = basinhopping(A2_fb, ip1, stepsize=0.0001, niter=1000, minimizer_kwargs=minimizer_kwargs, interval=1 ,
                        niter_success=100, disp=True)
    stop = timeit.default_timer()

    print("Result: ", res1.x[:10])
    print("Time: ", stop - start)
    print("Constraints: ", res1.x[10:])
    # print("Constraints Upper Bounds: ", res1.x[22:24])
    NE_check(res1.x[:10].tolist())

    print("Energy: ", A2_fb(res1.x))
