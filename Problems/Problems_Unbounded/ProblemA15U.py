

import numpy as np

from scipy.optimize import Bounds

from scipy.optimize import minimize

from scipy.optimize import LinearConstraint

from scipy.optimize import basinhopping

import timeit

from typing import List, Tuple, Dict, Optional, Callable

import numpy.typing as npt


class A15U:
  @staticmethod
  def paper_solution():
    value_1 = [46.66150692423980, 32.15293850189938, 15.00419467998705, 22.10485810522063, 12.34076570922471, 12.34076570922471]
    return [value_1]

  @staticmethod
  def define_players():
    player_vector_sizes = [1, 2, 3]
    player_objective_functions = [0, 1, 2]
    player_constraints = [[0,1] for _ in range(3)]
    return [player_vector_sizes, player_objective_functions, player_constraints]

  @staticmethod
  def objective_functions():
    return [A15U.obj_func_1, A15U.obj_func_2, A15U.obj_func_3]

  @staticmethod
  def objective_function_derivatives():
    return [A15U.obj_func_der_1, A15U.obj_func_der_2, A15U.obj_func_der_3]
  
  @staticmethod
  def constraints():
    return [A15U.g0, A15U.g1]
  
  @staticmethod
  def constraint_derivatives():
     return [A15U.g0_der, A15U.g1_der]


  @staticmethod
  def obj_func(
      x_i: npt.NDArray[np.float64],
      c_i: npt.NDArray[np.float64],
      d_i: npt.NDArray[np.float64],
      e_i: npt.NDArray[np.float64],
      S: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    obj = (2 * S - 378.4) * np.sum(x_i) + np.sum(0.5 * c_i * x_i**2 + d_i * x_i+ e_i)
    return obj

  @staticmethod
  def obj_func_1(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
      x1 = x[0]
      x2 = x[1]
      x3 = x[2]
      S = np.sum(x1) + np.sum(x2) + np.sum(x3)
      c1 = 0.04
      d1 = 2.0
      e1 = 0.0
      return A15U.obj_func(x1, c1, d1, e1, S)

  @staticmethod
  def obj_func_2(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
      x1 = x[0]
      x2 = x[1]
      x3 = x[2]
      S = np.sum(x1) + np.sum(x2) + np.sum(x3)
      c2 = np.array([0.035, 0.125]).reshape(-1,1)
      d2 = np.array([1.75, 1]).reshape(-1,1)
      e2 = np.array([0.0, 0.0]).reshape(-1,1)
      return A15U.obj_func(x2, c2, d2, e2, S)

  @staticmethod
  def obj_func_3(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
      x1 = x[0]
      x2 = x[1]
      x3 = x[2]
      S = np.sum(x1) + np.sum(x2) + np.sum(x3)
      c3 = np.array([0.0166, 0.05, 0.05]).reshape(-1,1)
      d3 = np.array([3.25, 3.0, 3.0]).reshape(-1,1)
      e3 = np.array([0.0, 0.0, 0.0]).reshape(-1,1)
      return A15U.obj_func(x3, c3, d3, e3, S)


  @staticmethod
  def obj_func_der(x_i: npt.NDArray[np.float64],
      c_i: npt.NDArray[np.float64],
      d_i: npt.NDArray[np.float64],
      S: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    obj = 2 * x_i + (2 * S - 378.4) + c_i * x_i+ d_i
    return obj


  @staticmethod
  def obj_func_der_1(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
      x1 = x[0].reshape(-1,1)
      x2 = x[1].reshape(-1,1)
      x3 = x[2].reshape(-1,1)
      S = np.sum(x1) + np.sum(x2) + np.sum(x3)
      c1 = 0.04
      d1 = 2.0
      return A15U.obj_func_der(x1, c1, d1, S)


  @staticmethod
  def obj_func_der_2(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
      x1 = x[0].reshape(-1,1)
      x2 = x[1].reshape(-1,1)
      x3 = x[2].reshape(-1,1)
      S = np.sum(x1) + np.sum(x2) + np.sum(x3)
      c2 = np.array([0.035, 0.125]).reshape(-1,1)
      d2 = np.array([1.75, 1]).reshape(-1,1)
      return A15U.obj_func_der(x2, c2, d2, S)


  @staticmethod
  def obj_func_der_3(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
      x1 = x[0].reshape(-1,1)
      x2 = x[1].reshape(-1,1)
      x3 = x[2].reshape(-1,1)
      S = np.sum(x1) + np.sum(x2) + np.sum(x3)
      c3 = np.array([0.0166, 0.05, 0.05]).reshape(-1,1)
      d3 = np.array([3.25, 3.0, 3.0]).reshape(-1,1)
      return A15U.obj_func_der(x3, c3, d3, S)

  @staticmethod
  def g0(x):
      x = np.concatenate(x)
      return 0 - x

  @staticmethod
  def g1(x):
      x = np.concatenate(x).reshape(-1,1)
      upper_bounds = np.array([80, 80, 50, 55, 30, 40]).reshape(-1, 1)
      return x - upper_bounds

  @staticmethod
  def g0_der(x):
      return -1

  @staticmethod
  def g1_der(x):
      return 1


