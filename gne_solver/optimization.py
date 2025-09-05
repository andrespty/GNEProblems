from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
from typing import List, Tuple, Dict, Optional, Callable, Union
from .types import *
from .utils import *
import numpy as np

def optimize_obj_func(
        player_idx: int,
        fixed_vars: VectorList,
        obj_func: ObjFunction,
        constraints: List[ConsFunction],
        method: str,
        single_obj_vector=True
):
    # A single vector contains multiple objective functions
    if single_obj_vector:
        wrapped_p_objective = create_wrapped_function_single(obj_func, fixed_vars, player_idx)

    # Each objective function returns value for each player
    else:
        wrapped_p_objective = create_wrapped_function(obj_func, fixed_vars, player_idx)


    optimization_constraints: List[Dict[str, Union[str, any]]] = [
        {'type': 'ineq', 'fun': lambda x: create_wrapped_function(constraint, fixed_vars, player_idx)(x).ravel()}
        for constraint in constraints
    ]

    p_var_0 = np.zeros_like(fixed_vars[player_idx]).reshape(-1).tolist()
    min_result = minimize(
        wrapped_p_objective,
        p_var_0,
        method=method,
        constraints=optimization_constraints,
        options={
            'disp': False,
            'maxiter': 1000,
            # 'ftol': 1e-5
        }
    )

    return min_result