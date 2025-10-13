from typing import List, Tuple, Dict, Optional, Callable, Union
from numpy.typing import NDArray
import numpy as np
from dataclasses import dataclass

class Types:
    """Common type definitions.

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
    ObjFunction : Callable
        Takes in a list of vectors and returns a single vector.
    ObjFunctionGrad : Callable
        Takes in a list of vectors and returns a single vector or a float.
    ConsFunction : Callable
        A constraint function that takes in a vector list and returns a single vector.
    ConsFunctionGrad : Callable
        A constraint function that takes in a vector list and returns a single vector or a float.
    WrappedFunction : Callable
        A wrapper function that takes in a vector list and returns a single vector.
    """

Vector = NDArray[np.float64]
Matrix = NDArray[np.float64]
VectorList = List[NDArray[np.float64]]
PlayerConstraint = Union[List[int], None, List[None]]

# Functions
#Does grad mean gradient or does it just mean it can return a float too?
ObjFunction = Callable[[VectorList], Vector]
ObjFunctionGrad = Callable[[VectorList], Vector] # Can return a single float or a vector
#For ObjFunctionGrad do we need = Callable[[VectorList], Union[float, Vector]]
ConsFunction = Callable[[VectorList], Vector]
#Does Cons stand for constraint
ConsFunctionGrad = Callable[[VectorList], Vector] # Should always return either a scalar or a vector of same size as the number of actions
WrappedFunction = Callable[[List[float]], Vector]


__all__ = [
    "Vector",
    "Matrix",
    "VectorList",
    "PlayerConstraint",
    "ObjFunction",
    "ObjFunctionGrad",
    "ConsFunction",
    "ConsFunctionGrad",
    "WrappedFunction",
]