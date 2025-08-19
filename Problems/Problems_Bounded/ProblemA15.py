

import numpy as np

from scipy.optimize import Bounds

from scipy.optimize import minimize

from scipy.optimize import LinearConstraint

from scipy.optimize import basinhopping

import timeit

from typing import List, Tuple, Dict, Optional, Callable

import numpy.typing as npt


class A15:

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
    return [A15.obj_func_1, A15.obj_func_2, A15.obj_func_3]


  @staticmethod
  def objective_function_derivatives():
    return [A15.obj_func_der_1, A15.obj_func_der_2, A15.obj_func_der_3]
  
  @staticmethod
  def constraints():
    return []
  
  @staticmethod
  def constraint_derivatives():
     return []


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
      return A15.obj_func(x, i, c1, c2, c3, S)


  @staticmethod
  def obj_func_2(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
      x1 = x[0]
      x2 = x[1]
      x3 = x[2]
      x4 = x[3]
      x5 = x[4]
      x6 = x[5]
      S = x1 + x2 + x3 + x4 + x5 + x6
      i = np.array([1, 2])
      c1 = np.array([0.035, 0.125])
      c2 = np.array([1.75, 1])
      c3 = np.array([0.0, 0.0])
      return A15.obj_func(x, i, c1, c2, c3, S)


  @staticmethod
  def obj_func_3(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:

      x1 = x[0]
      x2 = x[1]
      x3 = x[2]
      x4 = x[3]
      x5 = x[4]
      x6 = x[5]

      S = x1 + x2 + x3 + x4 + x5 + x6
      i = np.array([3, 4, 5])
      c1 = np.array([0.0166, 0.05, 0.05])
      c2 = np.array([3.25, 3.0, 3.0])
      c3 = np.array([0.0, 0.0, 0.0])
      return A15.obj_func(x, i, c1, c2, c3, S)


  @staticmethod

  def obj_func_der(x: npt.NDArray[np.float64], 

      i: npt.NDArray[np.float64],
      c1: npt.NDArray[np.float64],
      c2: npt.NDArray[np.float64],
      S: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    x_selected = x[i]
    obj = 2 * x_selected + (2 * S - 378.4) + c1 * x_selected + c2
    return obj


  @staticmethod

  def obj_func_der_1(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:

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
      return A15.obj_func_der(x, i, c1, c2, S)


  @staticmethod

  def obj_func_der_2(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:

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
      return A15.obj_func_der(x, i, c1, c2, S)


  @staticmethod

  def obj_func_der_3(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:

      x1 = x[0]
      x2 = x[1]
      x3 = x[2]
      x4 = x[3]
      x5 = x[4]
      x6 = x[5]
      S = x1 + x2 + x3 + x4 + x5 + x6
      i = np.array([3, 4, 5])
      c1 = np.array([0.0166, 0.05, 0.05])
      c2 = np.array([3.25, 3.0, 3.0])
      return A15.obj_func_der(x, i, c1, c2, S)
