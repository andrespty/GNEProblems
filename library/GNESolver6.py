import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import basinhopping
import timeit
from typing import List, Tuple, Dict, Optional, Callable
import numpy.typing as npt
from library.misc import *

class GNEP_Solver:
    def __init__(self,
                 obj_funcs:                     List[Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]],
                 derivative_obj_funcs:          List[Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]],
                 constraints:                   List[Callable[[npt.NDArray[np.float64]], np.float64]],
                 derivative_constraints:        List[Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]],
                 player_obj_func:               List[int],
                 player_constraints:            List[List[int]],
                 bounds:                        List[Tuple[float, float]],
                 player_vector_sizes:           List[int] = None,
                 ):
        self.objective_functions =              obj_funcs                        # list of functions
        self.objective_function_derivatives =   derivative_obj_funcs             # list of functions
        self.constraints =                      constraints                      # list of functions
        self.constraint_derivatives =           derivative_constraints           # list of functions
        self.player_objective_function =        np.array(player_obj_func)        # which obj function is used for each player
        self.player_constraints =               one_hot_encoding(player_constraints, player_vector_sizes, len(derivative_constraints))     # which constraints are used for each player
        self.action_sizes =                     np.array(player_vector_sizes)    # size of each player's action vector
        self.bounds =                           np.array(bounds)                 # bounds of each player
        self.N =                                len(player_obj_func)

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
        return self.energy_function(actions, dual_actions)

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
        engval=gradient**2
        # if isDual:
        #     # print('DUAL GRADIENT: ',gradient)
        #     bounds = self.bounds[self.N:]
        # else:
        #     # print('PRIMAL GRADIENT: ',gradient)
        #     bounds = repeat_items(self.bounds[:self.N], self.action_sizes)
        # lb = bounds[:, 0].reshape(-1, 1)
        # ub = bounds[:, 1].reshape(-1, 1)
        # engval = np.where(
        #     gradient <= 0,
        #     (ub - actions) * np.log(1 - gradient),
        #     (actions - lb) * np.log(1 + gradient)
        # )
        return engval

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

    def dual_energy_function(self, actions: npt.NDArray[np.float64], dual_actions: npt.NDArray[np.float64]) -> float:
        """
        Input:
          actions: 2d np.array shape (sum(number of actions),1)
          dual_actions: 2d np.array shape (N_d,1
        """
        gradient = self.calculate_gradient_dual(actions, dual_actions)
        return self.energy_handler(gradient, actions, isDual=True)

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
        # Add constraints
        for c_idx, p_vector in enumerate(self.player_constraints.T):
            result += p_vector.reshape(-1,1) * dual_actions[c_idx] * self.constraint_derivatives[c_idx](actions)
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
        bounds_primal = repeat_items(self.bounds[:self.N], self.action_sizes)
        bounds_dual = self.bounds[self.N:]

        scaled_actions = vectorized_sigmoid(actions, bounds_primal)           # shape(N, 1)
        scaled_dual_actions = vectorized_sigmoid(dual_actions, bounds_dual)   # shape(N_d,1)
        return np.vstack((scaled_actions, scaled_dual_actions))

    def calculate_main_objective(self, actions):
        objective_values_matrix = [
            self.objective_functions[idx](actions) for idx in self.player_objective_function
        ]
        return np.array(deconstruct_vectors(objective_values_matrix))
    def summary(self, paper_res=None):
        if self.result:
            print(self.result.x)
            translated_solution = self.translate_solution(self.result.x).tolist()
            print('Time: ', self.time)
            print('Iterations: ', self.result.nit)
            print('Translated Solution: \n', translated_solution)
            if paper_res:
                print('Paper Result: \n', paper_res)
            print('Solution: \n', translated_solution)
            print('Total Energy: ', self.wrapper(translated_solution))
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