from .utils import *
from .types import *
from scipy.optimize import basinhopping
import timeit

class GNEP_Solver_Unbounded:

    def __init__(self,
                 obj_funcs:                     List[ObjFunction],
                 derivative_obj_funcs:          List[ObjFunctionGrad],
                 constraints:                   List[ConsFunction],
                 derivative_constraints:        List[ConsFunctionGrad],
                 player_obj_func:               List[int],
                 player_constraints:            List[PlayerConstraint],
                 player_vector_sizes:           List[int]
                 ):
        self.objective_functions =              obj_funcs                        # list of functions
        self.player_obj_func =                  one_hot_encoding(player_obj_func, player_vector_sizes, len(derivative_obj_funcs))
        self.objective_function_derivatives =   derivative_obj_funcs             # list of functions
        self.constraints =                      constraints                      # list of functions
        self.constraint_derivatives =           derivative_constraints           # list of functions
        self.player_objective_function =        np.array(player_obj_func, dtype=int)        # which obj function is used for each player
        self.player_constraints =               one_hot_encoding(player_constraints, player_vector_sizes, len(derivative_constraints))     # which constraints are used for each player
        self.action_sizes =                     player_vector_sizes    # size of each player's action vector
        self.N =                                len(player_obj_func)

    def wrapper(self, initial_actions: List[float]) -> float:
        """
        Input:
          initial_actions: python list of all players' actions
        Output:
          total energy: float value
        """
        actions_count = sum(self.action_sizes)
        actions = np.array(initial_actions[:actions_count], dtype=np.float64).reshape(-1,1)
        dual_actions = np.array(initial_actions[actions_count:], dtype=np.float64).reshape(-1,1)
        return self.energy_function(actions, dual_actions)

    def energy_function( self, actions: Vector, dual_actions: Vector) -> float:
        """
        Input:
          actions: 2d np.array shape (sum(number of actions),1)   i.e. [[1], [2], [3], ..., [number of actions]]
          dual_actions: 2d np.array shape (N_d,1)                 i.e. [[1], [2], [3], ..., [N_d]]
        Output:
          total energy: float value
        """
        primal_players_energy = self.primal_energy_function(actions, dual_actions)
        dual_players_energy = self.dual_energy_function(actions, dual_actions)
        return float(np.sum(primal_players_energy, axis=0) + np.sum(dual_players_energy, axis=0))

    @staticmethod
    def energy_handler(gradient: Vector, actions: Vector, isDual=False) -> Vector:
        """
        Input:
          gradient: 2d np.array shape (sum(number of actions),1)
        Output:
          total energy: float value
        """
        if isDual:
            return (actions**2/(1+actions**2)) * (gradient**2/(1+gradient**2)) + np.exp(-actions**2) * (np.maximum(0,-gradient)**2/(1+np.maximum(0,-gradient)**2))
        else:
            return np.square(gradient)

    def primal_energy_function(self, actions: Vector, dual_actions: Vector) -> Vector:
        """
        Input:
          actions: 2d np.array shape (sum(number of actions),1)
          dual_actions: 2d np.array shape (N_d,1)
        Output:
          (sum(number of actions),1) vector with the energy of each players' action
        """
        gradient = self.calculate_gradient(actions, dual_actions)
        return self.energy_handler(gradient, actions)

    def dual_energy_function(self, actions: Vector, dual_actions: Vector) -> Vector:
        """
        Input:
          actions: 2d np.array shape (sum(number of actions),1)
          dual_actions: 2d np.array shape (N_d,1
        """
        eng_dual = self.calculate_gradient_dual(actions, dual_actions)
        return eng_dual

    # Gradient of primal player
    def calculate_gradient(self, actions: Vector, dual_actions: Vector) -> Vector:
        """
        Input:
          actions: 2d np.array shape (sum(number of actions),1)
          dual_actions: 2d np.array shape (N_d,1)
        Output:
          (sum(number of actions),1) vector
        """
        result = np.zeros_like(actions) # shape (-1,1)
        # Track action indices per player
        action_splits = np.cumsum(np.insert(self.action_sizes, 0, 0) )  # Start and end indices for each player
        vector_actions = construct_vectors(actions, self.action_sizes)
        for obj_idx, mask in enumerate(self.player_obj_func.T):
            mask: NDArray[bool]
            o = self.objective_function_derivatives[obj_idx](vector_actions)
            # Case 1: objective gradient returns full (sum(number of actions),1) vector
            if o.shape == result.shape:
                result += mask.reshape(-1,1) * o.reshape(-1,1)

            # Case 2: objective returns gradient only for its actions (actions,1) vector
            else:
                offset = 0
                # mask: NDArray[np.bool_] = (self.player_objective_function == obj_idx)
                mask = np.equal(self.player_objective_function, obj_idx)
                player_indices = np.where(mask)[0]
                for player in player_indices:
                    start_idx, end_idx = action_splits[player], action_splits[player + 1]
                    size = end_idx - start_idx
                    result[start_idx:end_idx] = o#[offset:offset + size]
                    offset += size

        # Adding constraints, should be vectors with same size of result
        for c_idx, p_vector in enumerate(self.player_constraints.T):
            p_vector: NDArray[int]
            result += p_vector.reshape(-1,1) * dual_actions[c_idx] * self.constraint_derivatives[c_idx](vector_actions)
        return result

    # Gradient of dual player
    def calculate_gradient_dual(self, actions: Vector, dual_actions: Vector) -> Vector:
        """
        Input:
          actions: 2d np.array shape (sum(number of actions),1)
          dual_actions: 2d np.array shape (N_d,1)
        Output:
          (N_d,1) vector
        """
        grad_dual = []
        actions_vectors = construct_vectors(actions, self.action_sizes)
        for jdx, constraint in enumerate(self.constraints):
            g = -constraint(actions_vectors)
            g = self.energy_handler(g, dual_actions[jdx], isDual=True).astype(np.float64)
            grad_dual.append(g.flatten())
        g_dual = np.concatenate(grad_dual).reshape(-1, 1)
        return g_dual

    def solve_game(self, initial_guess: List[float], disp: bool=True):
        """
        Input:
          initial_guess: python list of all players' actions
        Output:
          result: scipy.optimize.optimize.OptimizeResult object
          time: float
        """
        minimizer_kwargs = dict(method="L-BFGS-B")
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
