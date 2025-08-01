

import numpy as np

from scipy.optimize import Bounds

from scipy.optimize import minimize

from scipy.optimize import LinearConstraint

from scipy.optimize import basinhopping

import timeit

from typing import List, Tuple, Dict, Optional, Callable

import numpy.typing as npt

from library.misc import construct_vectors


class A15dev:
  @staticmethod
  def paper_solution():
    value_1 = [46.66150692423980, 32.15293850189938, 15.00419467998705, 22.10485810522063, 12.34076570922471, 12.34076570922471]
    return [value_1]

  @staticmethod
  def define_players():
    player_vector_sizes = [1, 2, 3]
    player_objective_functions = [0, 1, 2]
    player_constraints = [[None]], [[None], [None]]
    bounds = [(0.0, 80), (0.0, 80), (0.0, 50), (0.0, 55), (0.0, 30), (0.0, 40)]
    bounds_training = [(0.0, 80), (0.0, 80), (0.0, 50), (0.0, 55), (0.0, 30), (0.0, 40)]
    return [player_vector_sizes, player_objective_functions, player_constraints, bounds, bounds_training]

  @staticmethod
  def objective_functions():
    return [A15dev.obj_func_1, A15dev.obj_func_2, A15dev.obj_func_3]

  @staticmethod
  def objective_function_derivatives():
    return [A15dev.obj_func_der_1, A15dev.obj_func_der_2, A15dev.obj_func_der_3]
  
  @staticmethod
  def constraints():
    return [A15dev.g0, A15dev.g1, A15dev.g2, A15dev.g3, A15dev.g4, A15dev.g5, A15dev.g6]
  
  @staticmethod
  def constraint_derivatives():
     return [A15dev.g0_der, A15dev.g1_der, A15dev.g2_der, A15dev.g3_der, A15dev.g4_der, A15dev.g5_der, A15dev.g6_der]


  @staticmethod
  def obj_func(
      x: npt.NDArray[np.float64],
      i: npt.NDArray[np.float64],
      c1: npt.NDArray[np.float64],
      c2: npt.NDArray[np.float64],
      c3: npt.NDArray[np.float64],
      S: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:

    x_selected = x[i]
    obj = (2 * S - 378.4) * np.sum(x_selected) + np.sum(0.5 * c1 * x_selected**2 + c2 * x_selected + c3)
    return obj


  @staticmethod
  def obj_func_1(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
      x1 = x[0]
      x2 = x[1]
      x3 = x[2]
      x4 = x[3]
      x5 = x[4]
      x6 = x[5]

      S = x1 + x2 + x3 + x4 + x5 + x6
      i = np.array([0])
      c1 = 0.04
      c2 = 2.0
      c3 = 0.0
      return A15dev.obj_func(x, i, c1, c2, c3, S)


  @staticmethod
  def obj_func_2(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
      x1 = x[0]
      x2 = x[1]
      x3 = x[2]
      x4 = x[3]
      x5 = x[4]
      x6 = x[5]
      S = x1 + x2 + x3 + x4 + x5 + x6
      i = np.array([1, 2]).reshape(-1,1)
      c1 = np.array([0.035, 0.125]).reshape(-1,1)
      c2 = np.array([1.75, 1]).reshape(-1,1)
      c3 = np.array([0.0, 0.0]).reshape(-1,1)
      return A15dev.obj_func(x, i, c1, c2, c3, S)


  @staticmethod
  def obj_func_3(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:

      x1 = x[0]
      x2 = x[1]
      x3 = x[2]
      x4 = x[3]
      x5 = x[4]
      x6 = x[5]

      S = x1 + x2 + x3 + x4 + x5 + x6
      i = np.array([3, 4, 5]).reshape(-1,1)
      c1 = np.array([0.0166, 0.05, 0.05]).reshape(-1,1)
      c2 = np.array([3.25, 3.0, 3.0]).reshape(-1,1)
      c3 = np.array([0.0, 0.0, 0.0]).reshape(-1,1)
      return A15dev.obj_func(x, i, c1, c2, c3, S)

  @staticmethod
  def obj_func_der_1(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
      x1 = x[0]
      x2 = x[1]
      x3 = x[2]

      S = np.sum(x1) + np.sum(x2) + np.sum(x3)
      c1 = 0.04
      d1 = 2.0
      return 2 * x1 + (2 * S - 378.4) + c1 * x1 + d1

  @staticmethod
  def obj_func_der_2(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
      x1 = x[0]
      x2 = x[1]
      x3 = x[2]

      S = np.sum(x1) + np.sum(x2) + np.sum(x3)
      c2 = np.array([0.035, 0.125]).reshape(-1,1)
      d2 = np.array([1.75, 1]).reshape(-1,1)
      return 2 * np.sum(x2) + (2 * S - 378.4) + c2*x2 + d2

  @staticmethod
  def obj_func_der_3(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
      x1 = x[0]
      x2 = x[1]
      x3 = x[2]

      S = np.sum(x1) + np.sum(x2) + np.sum(x3)
      c3 = np.array([0.0166, 0.05, 0.05]).reshape(-1,1)
      d3 = np.array([3.25, 3.0, 3.0]).reshape(-1,1)
      return 2 * np.sum(x3) + (2 * S - 378.4) + c3*x3 + d3

  @staticmethod
  def g0(x):
      x1 = x[0]
      return x1 - 80

  @staticmethod
  def g1(x):
      x2 = x[1]
      return x2 - 80

  @staticmethod
  def g2(x):
      x3 = x[2]
      return x3 - 50

  @staticmethod
  def g3(x):
      x4 = x[3]
      return x4 - 55

  @staticmethod
  def g4(x):
      x5 = x[4]
      return x5 - 30

  @staticmethod
  def g5(x):
      x6 = x[5]
      return x6 - 40

  @staticmethod
  def g6(x):
      x = x.reshape(-1, 1)
      return 0 - x

  @staticmethod
  def g0_der(x):
      return 1

  @staticmethod
  def g1_der(x):
      return 1

  @staticmethod
  def g2_der(x):
      return 1

  @staticmethod
  def g3_der(x):
      return 1

  @staticmethod
  def g4_der(x):
      return 1

  @staticmethod
  def g5_der(x):
      return 1

  @staticmethod
  def g6_der(x):
      return -1

def A15devrun(x):
    x_p = np.array(x[:6]).reshape(-1, 1)
    players = construct_vectors(x_p, [1, 2, 3] )  # two player each with one variable
    dual = np.array(x[6:]).reshape(-1, 1)  # lower and upper bound
    constraints = A15dev.constraints()

    grad1 = A15dev.obj_func_der_1(players)
    grad2 = A15dev.obj_func_der_2(players)
    grad3 = A15dev.obj_func_der_3(players)
    grad = np.vstack((grad1, grad2, grad3)).reshape(-1,1)

    ub_duals = dual[:6].reshape(-1, 1)
    grad += ub_duals
    grad -= dual[6]

    eng = grad.T @ grad

    # Dual player
    grad_dual = []
    for jdx, constraint in enumerate(constraints):
        g = -constraint(x_p)
        # g = np.where(

        #     g <= 0,
        #     g**2,
        #     g**2 * np.tanh(dual[jdx])
        # )
        g = (dual[jdx] ** 2 / (1 + dual[jdx] ** 2)) * (g ** 2 / (1 + g ** 2)) + np.exp(-dual[jdx] ** 2) * (
                np.maximum(0, -g) ** 2 / (1 + np.maximum(0, -g) ** 2))
        grad_dual.append(g.flatten())
    g_dual = np.concatenate(grad_dual).reshape(-1, 1)

    return eng + np.sum(g_dual)



run = True
if run:
    minimizer_kwargs = dict(method="L-BFGS-B")
    primal = [1, 2, 0, 0, 0, 4]
    dual_ip = [1, 2, 3, 4, 5, 6, 7]
    ip1 = primal + dual_ip
    A15devrun(ip1)
    start = timeit.default_timer()
    res1 = basinhopping(A15devrun, ip1, stepsize=0.01, niter=1000, minimizer_kwargs=minimizer_kwargs, niter_success=100,
                        interval=1, disp=True)
    stop = timeit.default_timer()

    print('Time: ', stop - start)
    print(res1.x)