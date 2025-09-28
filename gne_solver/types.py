from typing import List, Tuple, Dict, Optional, Callable, Union
from numpy.typing import NDArray
import numpy as np
from dataclasses import dataclass

# Inputs
class Types:

Vector= NDArray[np.float64] #(n,1) column vector
"""
   Attributes
    ----------
    Vector : NDArray[np.float64]
        (n,1) column vector.
   
"""

Matrix = NDArray[np.float64]
"""
    Attributes
     ----------
    Matrix : NDArray[np.float64]
        General 2D matrix of floats.


"""
VectorList = List[Vector] # Player actions vectors
"""
    Attributes
     ----------
    VectorList : list of Vector
        List of player action vectors.

"""
PlayerConstraint = Union[List[int], None, List[None]]
"""
    Attributes
    ----------
    PlayerConstraint : list of ints or None
    Holds the player contraints.
    
    """

# Functions
ObjFunction = Callable[[VectorList], Vector]
ObjFunctionGrad = Callable[[VectorList], Vector] # Can return a single float or a vector
ConsFunction = Callable[[VectorList], Vector]
ConsFunctionGrad = Callable[[VectorList], Vector] # Should always return either a scalar or a vector of same size as the number of actions


WrappedFunction = Callable[[List[float]], Vector]