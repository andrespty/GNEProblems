import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import basinhopping
import timeit
from typing import List, Tuple, Dict, Optional, Callable
import numpy.typing as npt

class A8_BL:
    @staticmethod
    def paper_solution():
        value_1 = [
            0.62503131162143,
            0.37500031253875,
            0.93754579549990
        ]

        value_2 = [
            0.62510047245551,
            0.37500003126256,
            0.9376505914769
        ]
        return [value_1, value_2]

    @staticmethod
    def define_players():
        player_vector_sizes = [1, 1, 1]
        player_objective_functions = [0, 1, 2]
        player_constraints = [[0, 1], [0, 1], [None]]
        bounds = [(0, 100), (0, 100), (0, 2), (0, 100), (0, 100)]
        bounds_training = [(0, 100), (0, 100), (0, 2), (0, 100), (0, 100)]
        return [player_vector_sizes, player_objective_functions, player_constraints, bounds, bounds_training]

    @staticmethod
    def objective_functions():
        return [A8_BL.obj_func_1, A8_BL.obj_func_2, A8_BL.obj_func_3]

    @staticmethod
    def objective_function_derivatives():
        return [A8_BL.obj_func_der_1, A8_BL.obj_func_der_2, A8_BL.obj_func_der_3]

    @staticmethod
    def constraints():
        return [A8_BL.g0, A8_BL.g1]

    @staticmethod
    def constraint_derivatives():
        return [A8_BL.g0_der, A8_BL.g1_der]

    @staticmethod
    def obj_func_1(x: npt.NDArray[np.float64]):
      x1 = x[0]
      x2 = x[1]
      x3 = x[2]
      return -x1

    @staticmethod
    def obj_func_2(x: npt.NDArray[np.float64]):
      x1 = x[0]
      x2 = x[1]
      x3 = x[2]
      return (x2-0.5)**2

    @staticmethod
    def obj_func_3(x: npt.NDArray[np.float64]):
      x1 = x[0]
      x2 = x[1]
      x3 = x[2]
      return (x3 - 1.5*x1)**2

    @staticmethod
    def obj_func_der_1(x: npt.NDArray[np.float64]):
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        return -1

    @staticmethod
    def obj_func_der_2(x: npt.NDArray[np.float64]):
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        return 2 * x2 - 1

    @staticmethod
    def obj_func_der_3(x: npt.NDArray[np.float64]):
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        return 2 * x3 - 3 * x1

    @staticmethod
    def g0(x):
        x1, x2, x3 = x
        return x1 + x2 - 1

    @staticmethod
    def g1(x):
        x1, x2, x3 = x
        return x3 - x1 - x2

    @staticmethod
    def g0_der(x1):
        return np.array([[1]]).reshape(-1, 1)

    @staticmethod
    def g1_der(x1):
        return np.array([[-1]]).reshape(-1, 1)


def A8_ex(myinput):
  players = myinput[0:3]
  dual = myinput[3:]

  # costGradp1=-1-d1+d2 # Differentiation of the objective
  costGradp1 = A8_BL.obj_func_der_1(players) + dual[0] * A8_BL.g0_der(players) + dual[1] * A8_BL.g1_der(players)
  p1eng = costGradp1**2

  # costGradp2=2*(p2-0.5)-d1+d2 # Differentiation of the objective
  costGradp2 = A8_BL.obj_func_der_2(players) + dual[0] * A8_BL.g0_der(players) + dual[1] * A8_BL.g1_der(players)
  p2eng = costGradp2**2

  costGradp3 = A8_BL.obj_func_der_3(players)
  p3eng = costGradp3**2

  # Dual Players
  costGradd1= -A8_BL.g0(players)
  d1eng = costGradd1**2

  costGradd2= -A8_BL.g1(players)
  d2eng = costGradd2**2

  return (p1eng+p2eng+p3eng+d1eng+d2eng)

# minimizer_kwargs = dict(method="SLSQP")
# ip1=[0,0,0,0,0]
# start = timeit.default_timer()
# res1=basinhopping(A8_ex, ip1, stepsize=0.0001, niter=5000, minimizer_kwargs=minimizer_kwargs, niter_success=1000, interval=1, disp = True)
#
# stop = timeit.default_timer()
#
# print('Time: ', stop - start)
# print(res1.x)