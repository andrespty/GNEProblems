import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import basinhopping
import timeit
from typing import List, Tuple, Dict, Optional, Callable
import numpy.typing as npt

def flatten_variables(vectors, scalars):
    """
    Flattens a list of vectors and scalars into a single list for optimization.
    """
    return np.hstack([v.flatten() for v in vectors] + [scalars])

def construct_vectors(actions: npt.NDArray[np.float64], action_sizes: List[int]) -> List[npt.NDArray[np.float64]]:
    """
    Input:
      actions: np.array of all players' actions. Shape (sum(all actions), 1)
      action_sizes: list of sizes of each player's action vector
    Output:
      python list of 2d np.arrays
    """
    value_array = np.array(actions)
    indices = np.cumsum(action_sizes)
    return np.split(value_array, indices[:-1])

def deconstruct_vectors(vectors: List[npt.NDArray[np.float64]]) -> npt.NDArray[np.float64]:
    """
    Input:
        vectors: list of 2d np.arrays
    Output:
        np.array of all players' actions in a vector. Shape (sum(all actions), 1)
    """
    return np.concatenate(vectors)

def create_wrapped_function(
    original_func:Callable[[npt.NDArray[np.float64]],npt.NDArray[np.float64]], 
    vars: List[npt.NDArray[np.float64]], 
    player_idx: int
    ):
    player_var = vars[player_idx]
    fixed_vars = vars[:player_idx] + vars[player_idx+1:] # list of np vectors
    def wrap_func(player_var_opt):
        player_var_opt = np.array(player_var_opt).reshape(-1,1)
        new_vars = fixed_vars[:player_idx] + [player_var_opt] + fixed_vars[player_idx:]
        return original_func(new_vars)

    return wrap_func
    

def objective_check(objective_functions: List[Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]], actions: List[npt.NDArray[np.float64]]):
    objective_values = []
    for objective in objective_functions:
        o = objective(actions)
        objective_values.append(o)
    return objective_values
      
def constraint_check(constraints: List[Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]], actions: List[npt.NDArray[np.float64]], epsilon=1e-3):
    for constraint in constraints:
        c = constraint(actions)
        if not np.all(np.ravel(c) <= epsilon):
            print(f"CONSTRAINT VIOLATION: {c}")
            return False
    print("All constraints satisfied")
    return True

def print_table(vec1, vec2, header1="Vector 1", header2="Vector 2"):
    print(f"{header1:^10} | {header2:^10}")  # Header
    print("-" * 23)
    for v1, v2 in zip(vec1, vec2):
        # Extract scalar value from NumPy arrays
        v1_scalar = v1.item() if isinstance(v1, np.ndarray) else v1
        v2_scalar = v2.item() if isinstance(v2, np.ndarray) else v2
        print(f"{v1_scalar:^10.4f} | {v2_scalar:^10.4f}")  # Align values with 4 decimal places


def compare_solutions(
    computed_solution: List[float], 
    paper_solution: List[float],
    action_sizes: List[int], 
    objective_functions: List[Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]],
    solution_name = ['Computed', 'Paper']
  ):
    computed_NE = np.array(computed_solution).reshape(-1,1)
    paper_NE = np.array(paper_solution).reshape(-1,1)

    computed_NE_vectors = construct_vectors(computed_NE, action_sizes)
    paper_NE_vectors = construct_vectors(paper_NE, action_sizes)

    computed_NE_obj_func = objective_check(objective_functions, computed_NE_vectors)
    paper_NE_obj_func = objective_check(objective_functions, paper_NE_vectors)

    difference = np.array(computed_NE_obj_func) - np.array(paper_NE_obj_func)
#     print(f"Average difference between {solution_name[0]} and {solution_name[1]}: ", np.mean(difference))
    print("Objective Functions")
    print_table(computed_NE_obj_func, paper_NE_obj_func, solution_name[0], solution_name[1])
    return difference.reshape(-1,1)

def check_nash_equillibrium(
    result: List[np.float64], 
    action_sizes: List[int], 
    player_objective_function: List[int],
    objective_functions: List[Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]],
    constraints: List[Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]],
    player_constraints: List[List[int]],
    bounds: List[Tuple[float, float]],
    paper_res: List[float] = None,
    epsilon=1e-3
  ):
    computed_NE = np.array(result).reshape(-1,1)
    computed_NE_vectors = construct_vectors(computed_NE, action_sizes)
    print('Computed Solution: \n',computed_NE_vectors)
    action_splits = np.cumsum(np.insert(action_sizes, 0, 0) )

    # Get the obj function values at the current NE
    computed_NE_obj_func = objective_check(objective_functions, computed_NE_vectors)
    print('Computed Objective Function: \n', np.array(computed_NE_obj_func).reshape(-1,1))

    # Check that constraints are being satisfied
    if not constraint_check(constraints, computed_NE_vectors):
        return

    if paper_res is not None:
        compare_solutions(
          computed_NE, 
          paper_res, 
          action_sizes, 
          objective_functions,
        )
    print('--------------------------------------')
  # Optimize each player by fixing the opponents
    for player_idx, p_o_idx in enumerate(player_objective_function):
        p_var = computed_NE_vectors[player_idx]
        p_objective = objective_functions[p_o_idx] # this is a function
        p_constraints = [constraints[c_idx] for c_idx in player_constraints[p_o_idx] if c_idx ]

        wrapped_p_objective = create_wrapped_function(p_objective, computed_NE_vectors, player_idx)

        optimization_constraints = [
            {'type': 'ineq', 'fun': lambda x: create_wrapped_function(constraint, computed_NE_vectors, player_idx)(x)  } for constraint in p_constraints
        ]

        p_var_0 = np.zeros_like(p_var).flatten()
        # p_var_0 = p_var.flatten()
        player_bounds = bounds[action_splits[player_idx]:action_splits[player_idx+1]]
        result = minimize(
            wrapped_p_objective,
            p_var_0,
            method='SLSQP',
            bounds=player_bounds,
            constraints=optimization_constraints,
            options={
                'disp': False,
                'maxiter': 1000,
                'ftol': 1e-5
            }
        )
        # Report
        fixed_vars = computed_NE_vectors[:player_idx] + computed_NE_vectors[player_idx+1:]
        opt_actions = np.array(result.x).reshape(-1,1)
        new_vars = fixed_vars[:player_idx] + [opt_actions] + fixed_vars[player_idx:]
        opt_obj_func = objective_check(objective_functions, new_vars)
        print(f"Player {player_idx+1}")
        print(f"Optimized Actions: {result.x}\nOptimized Function value: {result.fun}")
        print('Optimized Objective Functions: \n', np.array(opt_obj_func).reshape(-1,1))
        constraint_check(constraints, new_vars)
        difference = compare_solutions(
          computed_NE, 
          deconstruct_vectors(new_vars), 
          action_sizes, 
          objective_functions,
          solution_name = ['Computed', 'Optimized']
        )
        print(f"Difference between Objective Functions of Player {player_idx + 1}: {difference[p_o_idx]}")
        if difference[p_o_idx] > epsilon:
            print('Not at the NE')
        else: 
            print(f"Computed Solution is at the NE for Player {player_idx + 1}")
        print('--------------------------------------')
    return

class GeneralizedNashEquilibriumSolver:
    def __init__(self,
                 obj_funcs:                     List[Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]],
                 derivative_obj_funcs:          List[Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]],
                 constraints:                   List[Callable[[npt.NDArray[np.float64]], np.float64]],
                 derivative_constraints:        List[Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]],
                 player_obj_func:               List[int],
                 player_constraints:            List[List[int]],
                 bounds:                        List[Tuple[float, float]],
                 player_vector_sizes:           List[int] = None,
                 useBounds:                     bool = True,
                 energy_select:                 str = 'abs'
                 ):
        self.objective_functions =              obj_funcs                        # list of functions
        self.objective_function_derivatives =   derivative_obj_funcs             # list of functions
        self.constraints =                      constraints                      # list of functions
        self.constraint_derivatives =           derivative_constraints           # list of functions
        self.player_objective_function =        np.array(player_obj_func)        # which obj function is used for each player
        max_length =                            max(len(sublist) for sublist in player_constraints)         # Player constraints is a list of lists. The lists are not always the same size.
        homogeneous_matrix =                    np.array([sublist + [None] * (max_length - len(sublist)) for sublist in player_constraints])# To address this, we create a homogeneous matrix by padding with None
        self.player_constraints =               np.array(homogeneous_matrix)     # which constraints are used for each player
        self.action_sizes =                     np.array(player_vector_sizes)    # size of each player's action vector
        self.bounds =                           np.array(bounds)                 # bounds of each player
        self.N =                                len(player_vector_sizes)         # number of players
        self.useBounds =                        useBounds                        # whether to use bounds or not

    def repeat_items(self, items, sizes):
        """
        Input:
          items: list of items     [1,2,3,4,5]
          sizes: list of sizes     [2,3,1,4,1]
        Output:
          list of repeated items   [1,1,2,2,2,3,4,4,4,4,5]
        """
        return np.array(np.repeat(items, sizes, axis=0).tolist())

    def vectorized_sigmoid(self, x: npt.NDArray[np.float64], bounds: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Input:
          x: 2d np.array (x,1)
          bounds: 2d np.array (x, 1)
        Output:
          2d np.array (x,1)
        """
        x = np.clip(x, -500, 500)
        lb = bounds[:, 0].reshape(-1,1)
        ub = bounds[:, 1].reshape(-1,1)
        sigmoid = 1 / (1 + np.exp(-x))
        return sigmoid * (ub - lb) + lb

    def wrapper(self, initial_actions: List[float]) -> float:
        """
        Input:
          initial_actions: python list of all players' actions
        Output:
          total energy: float value
        """
        actions_count = sum(self.action_sizes)
        actions = np.array(initial_actions[:actions_count]).reshape(-1,1)
        dual_actions = np.array(initial_actions[actions_count:]).reshape(-1,1)
        if self.useBounds:
            return self.energy_function(actions, dual_actions)
        bounds_primal = self.repeat_items(self.bounds[:self.N], self.action_sizes)
        bounds_dual = self.bounds[self.N:]
        scaled_actions = self.vectorized_sigmoid(actions, bounds_primal)           # shape(N, 1)
        scaled_dual_actions = self.vectorized_sigmoid(dual_actions, bounds_dual)
        return self.energy_function(scaled_actions, scaled_dual_actions)


    def energy_function(self, actions: npt.NDArray[np.float64], dual_actions: npt.NDArray[np.float64]) -> float:
        """
        Input:
          actions: 2d np.array shape (sum(number of actions),1)   i.e. [[1], [2], [3], ..., [number of actions]]
          dual_actions: 2d np.array shape (N_d,1)                 i.e. [[1], [2], [3], ..., [N_d]]
        Output:
          total energy: float value
        """
        primal_players_energy = self.primal_energy_function(actions, dual_actions)
        dual_players_energy = self.dual_energy_function(actions, dual_actions)
        return sum(primal_players_energy) + sum(dual_players_energy)

    def energy_handler(self, gradient: npt.NDArray[np.float64], actions: npt.NDArray[np.float64], isDual=False) -> float:
        """
        Input:
          gradient: 2d np.array shape (sum(number of actions),1)
        Output:
          total energy: float value
        """
        if self.useBounds:
            if isDual:
                # print('DUAL GRADIENT: ',gradient)
                bounds = self.bounds[self.N:]
            else:
                # print('PRIMAL GRADIENT: ',gradient)
                bounds = self.repeat_items(self.bounds[:self.N], self.action_sizes)
            lb = bounds[:, 0].reshape(-1,1)
            ub = bounds[:, 1].reshape(-1,1)
            engval = np.where(
                gradient <=0,
                (ub - actions) * np.log(1 - gradient),
                (actions - lb) * np.log(1 + gradient)
              )
            return engval
        else:
          # if isDual:
          #   print('DUAL GRADIENT: ',gradient)
          # else:
          #   print('PRIMAL GRADIENT: ',gradient)
            return np.abs(gradient)

    def primal_energy_function(self, actions: npt.NDArray[np.float64], dual_actions: npt.NDArray[np.float64]) -> float:
        """
        Input:
          actions: 2d np.array shape (sum(number of actions),1)
          dual_actions: 2d np.array shape (N_d,1)
        Output:
          (sum(number of actions),1) vector with the energy of each players' action
        """
        gradient = self.calculate_gradient(actions, dual_actions)
        return self.energy_handler(gradient, actions)
        # return np.abs(gradient) # returns a vector with shape (N, 1)

    def dual_energy_function(self, actions: npt.NDArray[np.float64], dual_actions: npt.NDArray[np.float64]) -> float:
        """
        Input:
          actions: 2d np.array shape (sum(number of actions),1)
          dual_actions: 2d np.array shape (N_d,1
        """
        gradient = self.calculate_gradient_dual(actions, dual_actions)
        return self.energy_handler(gradient, dual_actions, isDual=True)
        # return np.abs(gradient) # returns a vector with shape (N_d, 1)

    # Gradient of primal player
    def calculate_gradient(self, actions: npt.NDArray[np.float64], dual_actions: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Input:
          actions: 2d np.array shape (sum(number of actions),1)
          dual_actions: 2d np.array shape (N_d,1)
        Output:
          (sum(number of actions),1) vector
        """
        result = np.zeros_like(actions)
        # Track action indices per player
        action_splits = np.cumsum(np.insert(self.action_sizes, 0, 0) )  # Start and end indices for each player
        for func_idx in np.unique(self.player_objective_function):
            mask = (self.player_objective_function == func_idx)
            player_indices = np.where(mask)[0]
            o = self.objective_function_derivatives[func_idx](construct_vectors(actions, self.action_sizes))
            # Assign correct gradients to result
            for player in player_indices:
                start_idx, end_idx = action_splits[player], action_splits[player + 1]
                result[start_idx:end_idx] = o
        # Handle constraints
        for player, c_indexes in enumerate(self.player_constraints):
            start_idx, end_idx = action_splits[player], action_splits[player + 1]
            for c_idx in c_indexes:
                if c_idx is not None:
                    dual_var = dual_actions[c_idx]
                    grad_constraint = dual_var * self.constraint_derivatives[c_idx](actions)
                    result[start_idx:end_idx] += grad_constraint

        return result


    # Gradient of dual player
    def calculate_gradient_dual(self, actions: npt.NDArray[np.float64], dual_actions: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Input:
          actions: 2d np.array shape (sum(number of actions),1)
          dual_actions: 2d np.array shape (N_d,1)
        Output:
          (N_d,1) vector
        """
        actions_vectors = construct_vectors(actions, self.action_sizes)
        dual_objective_values = np.array([
            -constraint(actions_vectors) for constraint in self.constraints
        ])

        return dual_objective_values.reshape(-1,1)

    def solve_game(self, initial_guess: List[float], bounds: List[Tuple[float, float]], disp=True):
        """
        Input:
          initial_guess: python list of all players' actions
          bounds: python list of bounds for each player
        Output:
          result: scipy.optimize.optimize.OptimizeResult object
          time: float
        """
        if self.useBounds:
            minimizer_kwargs = dict(method="SLSQP", bounds=bounds)
        else:
            minimizer_kwargs = dict(method="SLSQP")
        start = timeit.default_timer()
        result = basinhopping(
            self.wrapper,
            initial_guess,
            stepsize=0.01,
            niter=1000,
            minimizer_kwargs=minimizer_kwargs,
            interval=1,
            niter_success=100,
            disp=disp,
            # callback=stopping_criterion
        )
        stop = timeit.default_timer()
        elapsed_time = stop - start
        self.result = result
        self.time = elapsed_time
        return result, elapsed_time

    def translate_solution(self, solution):
        actions = np.array(solution[:sum(self.action_sizes)]).reshape(-1,1)
        dual_actions = np.array(solution[sum(self.action_sizes):]).reshape(-1,1)
        bounds_primal = self.repeat_items(self.bounds[:self.N], self.action_sizes)
        bounds_dual = self.bounds[self.N:]

        scaled_actions = self.vectorized_sigmoid(actions, bounds_primal)           # shape(N, 1)
        scaled_dual_actions = self.vectorized_sigmoid(dual_actions, bounds_dual)   # shape(N_d,1)
        return np.vstack((scaled_actions, scaled_dual_actions))

    def calculate_main_objective(self, actions):
        objective_values_matrix = [
            self.objective_functions[idx](actions) for idx in self.player_objective_function
        ]
        return np.array(deconstruct_vectors(objective_values_matrix))
    def summary(self, paper_res=None):
        if self.result:
          # res = self.translate_solution(self.result.x)
            if self.useBounds:
                translated_solution = self.result.x
            else:
                translated_solution = self.translate_solution(self.result.x)
            print('Time: ', self.time)
            print('Iterations: ', self.result.nit)
            if not self.useBounds:
                print('Translated Solution: \n', translated_solution)
            if paper_res:
                print('Paper Result: \n', paper_res)
            print('Solution: \n', self.result.x)
            print('Total Energy: ', self.wrapper(self.result.x))
            if paper_res:
                paper = np.array(paper_res).reshape(-1,1)
                computed_actions = np.array(translated_solution[:sum(self.action_sizes)]).reshape(-1,1)
                calculated_obj = self.calculate_main_objective(construct_vectors(computed_actions, self.action_sizes))
                paper_obj = self.calculate_main_objective(construct_vectors(paper, self.action_sizes))
                print('Difference: ', sum(deconstruct_vectors(calculated_obj)) - sum(deconstruct_vectors(paper_obj)))

    def nash_check(self, epsilon=1e-3):
        if not self.result:
            print('No solution found')
            return
        print("Checking Nash Equilibrium")
        computed_NE = np.array(self.result.x[:sum(self.action_sizes)]).reshape(-1,1)
        check_nash_equillibrium(
          computed_NE, 
          self.action_sizes, 
          self.player_objective_function,
          self.objective_functions,
          self.constraints,
          self.player_constraints,
          self.bounds,
          paper_res=self.result.x[:sum(self.action_sizes)] if self.result.x is not None else None
        )
        print('Check finished')
        return 