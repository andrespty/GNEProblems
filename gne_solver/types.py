from typing import List, Tuple, Dict, Optional, Callable, Union
from numpy.typing import NDArray
import numpy as np

# Inputs
Vector= NDArray[np.float64] #(n,1) column vector
Matrix = NDArray[np.float64]
VectorList = List[Vector] # Player actions vectors

# Functions
ObjFunction = Callable[[VectorList], Vector]
ObjFunctionGrad = Callable[[VectorList], Vector] # Can return a single float or a vector
ConsFunction = Callable[[VectorList], Vector]
ConsFunctionGrad = Callable[[VectorList], Vector] # Should always return either a scalar or a vector of same size as the number of actions

PlayerConstraint = Union[List[int], None, List[None]]

WrappedFunction = Callable[[List[float]], Vector]