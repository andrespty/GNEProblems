import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import basinhopping
import timeit
from typing import List, Tuple, Dict, Optional, Callable
import numpy.typing as npt

class A8:
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
    @staticmethod
    def obj_func_1(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
      x1 = x[0]
      x2 = x[1]
      x3 = x[2]
      return -x1

    @staticmethod
    def obj_func_2(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
      x1 = x[0]
      x2 = x[1]
      x3 = x[2]
      return (x2-0.5)**2

    @staticmethod
    def obj_func_3(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
      x1 = x[0]
      x2 = x[1]
      x3 = x[2]
      return (x3 - 1.5*x1)**2

    @staticmethod
    def obj_func_der_1(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        return -1 + 0 * x1

    @staticmethod
    def obj_func_der_2(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        return 2 * x2 - 1

    @staticmethod
    def obj_func_der_3(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        return 2 * x3 - 3 * x1

    @staticmethod
    def g0(x):
        x1, x2, x3 = x
        return (x1 + x2 - 1)[0]

    @staticmethod
    def g1(x):
        x1, x2, x3 = x
        return (x3 - x1 - x2)[0]

    @staticmethod
    def g0_der(x1):
        return np.array([[1]]).reshape(-1, 1)

    @staticmethod
    def g1_der(x1):
        return np.array([[-1]]).reshape(-1, 1)