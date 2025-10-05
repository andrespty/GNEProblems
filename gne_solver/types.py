from typing import List, Tuple, Dict, Optional, Callable, Union
from numpy.typing import NDArray
import numpy as np
from dataclasses import dataclass

# Inputs
class Types:
    """
    Common type definitions.

    Attributes
    ----------
    Vector : NDArray[np.float64]
        (n,1) column vector.
    Matrix : NDArray[np.float64]
        General 2D matrix of floats.
    VectorList : list of Vector
        List of player action vectors.
    PlayerConstraint : list[int] | None
        Holds the player constraints.
    ObjFunction : Callable Function
        Takes in a list of vectors as the argument and returns a single vector.
    ObjFunctionGrad : Callable Function
        Takes in a list of vectors as the argument and returns a single vector or a float.
    ConsFunction : Callable Function
        A constraint function that takes in a vector list and returns a single vector.
    ConsFunctionGrad : Callable Function
        A constraint function that takes in a vector list and returns a single vector or a float.
    WrappedFunction : Callable Function
        A wrapper function that takes in a vector list and returns a single vector.
    """
Vector: NDArray[np.float64]
Matrix: NDArray[np.float64]
VectorList: List[NDArray[np.float64]]
PlayerConstraint: Union[List[int], None, List[None]]

# Functions
#Does grad mean gradient or does it just mean it can return a float too?
ObjFunction = Callable[[VectorList], Vector]
ObjFunctionGrad = Callable[[VectorList], Vector] # Can return a single float or a vector
#For ObjFunctionGrad do we need = Callable[[VectorList], Union[float, Vector]]
ConsFunction = Callable[[VectorList], Vector]
#Does Cons stand for constraint
ConsFunctionGrad = Callable[[VectorList], Vector] # Should always return either a scalar or a vector of same size as the number of actions
WrappedFunction = Callable[[List[float]], Vector]