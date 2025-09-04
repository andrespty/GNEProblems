from .misc import *
from .types import *

def construct_vectors(actions: Vector, action_sizes: List[int]) -> VectorList:
    """
    Split a concatenated action array into separate action vectors for each player.

    Parameters
    ----------
    actions : numpy.ndarray of shape (sum(action_sizes), 1)
        A 2D NumPy array containing all players' actions stacked vertically.
        The number of rows must equal the sum of all entries in ``action_sizes``.
    action_sizes : list of int
        A list specifying the length of each player's action vector.
        The sum of these sizes must match the number of rows in ``actions``.

    Returns
    -------
    list of numpy.ndarray
        A list of 2D NumPy arrays, each corresponding to one player's action vector.
        The arrays are in the same order as the players in ``action_sizes``.

    Examples
    --------
    >> actions = np.array([[1.0], [2.0], [3.0], [4.0]])
    >> action_sizes = [2, 2]
    >> construct_vectors(actions, action_sizes)
    [array([[1.],
            [2.]]),
     array([[3.],
            [4.]])]
    """
    value_array = np.array(actions).reshape(-1,1)
    indices = np.cumsum(action_sizes)
    return np.split(value_array, indices[:-1])

def one_hot_encoding(funcs_idx: List[Union[int, PlayerConstraint]], sizes: List[int], num_functions: int) -> Matrix:
    assert len(funcs_idx) == len(sizes), "funcs_idx and sizes must match in length"

    total_vars = sum(sizes)
    M = np.zeros((total_vars, num_functions), dtype=int)

    # Row offsets per variable set
    offsets = np.cumsum([0] + sizes[:-1])

    for var_idx, funcs in enumerate(funcs_idx):
        # Treat [None] as "uses no functions"
        print(funcs)
        if funcs is None or funcs == [None]:
            continue
        start = offsets[var_idx]
        end = start + sizes[var_idx]
        M[start:end, funcs] = 1

    return M

def create_wrapped_function(original_func: ObjFunction, actions: VectorList, player_idx: int) -> WrappedFunction:
    fixed_vars = actions[:player_idx] + actions[player_idx + 1:]

    def wrap_func(player_var_opt: List[float]) -> Vector:
        player_var_opt = np.array(player_var_opt).reshape(-1, 1)
        new_vars = fixed_vars[:player_idx] + [player_var_opt] + fixed_vars[player_idx:]
        return original_func(new_vars)

    return wrap_func

def create_wrapped_function_single(original_func: ObjFunction,actions: VectorList,player_idx: int) -> WrappedFunction:
    fixed_vars = actions[:player_idx] + actions[player_idx + 1:]  # list of np vectors

    def wrap_func(player_var_opt):
        player_var_opt = np.array(player_var_opt).reshape(-1, 1)
        new_vars = fixed_vars[:player_idx] + [player_var_opt] + fixed_vars[player_idx:]
        return original_func(new_vars)[player_idx]

    return wrap_func

def objective_check(objective_functions: List[ObjFunction], actions: VectorList) -> Vector:
    objective_values = []
    for objective in objective_functions:
        o = objective(actions)
        objective_values.append(o)
    return np.concatenate(objective_values).reshape(-1, 1)

def constraint_check(constraints: List[ConsFunction], actions: VectorList, epsilon: float = 1e-3) -> Tuple[VectorList, List[bool]]:
    constraint_values = []
    constraint_satisfaction = []
    for c_idx, constraint in enumerate(constraints):
        c = constraint(actions)
        if not np.all(np.ravel(c) <= epsilon):
            print(f"CONSTRAINT VIOLATION: {c_idx}, {c}")
            constraint_values.append(c)
            constraint_satisfaction.append(False)
        else:
            constraint_values.append(c)
            constraint_satisfaction.append(True)
    return constraint_values, constraint_satisfaction

def compare_solutions(
    computed_solution: List[float],
    paper_solution: List[float],
    action_sizes: List[int],
    objective_functions: List[ObjFunction],
    solution_name: List[str] = None
) -> Vector:
    if solution_name is None:
        solution_name=['Computed', 'Paper']
    computed_res = np.array(computed_solution).reshape(-1, 1)
    paper_res = np.array(paper_solution).reshape(-1, 1)

    computed_res_vectors = construct_vectors(computed_res, action_sizes)
    paper_res_vectors = construct_vectors(paper_res, action_sizes)

    computed_res_obj_func = objective_check(objective_functions, computed_res_vectors)
    paper_res_obj_func = objective_check(objective_functions, paper_res_vectors)

    difference = np.array(computed_res_obj_func) - np.array(paper_res_obj_func)
    print("Objective Functions")
    print_table(computed_res_obj_func, paper_res_obj_func, solution_name[0], solution_name[1])
    return difference.reshape(-1, 1)


"______________Recheck_____________"
def calculate_main_objective(self, actions):
    objective_values_matrix = [
        self.objective_functions[idx](actions) for idx in self.player_objective_function
    ]
    return np.array(deconstruct_vectors(objective_values_matrix))

def summary(result, time, wrapper, action_sizes, paper_res=None):
    print(result.x)
    print('Time: ', time)
    print('Iterations: ', result.nit)
    if paper_res:
        print('Paper Result: \n', paper_res)
    print('Solution: \n', result.x)
    print('Total Energy: ', wrapper(result.x))
    if paper_res:
        paper = np.array(paper_res).reshape(-1,1)
        computed_actions = np.array(result.x[:sum(action_sizes)]).reshape(-1,1)
        calculated_obj = calculate_main_objective(construct_vectors(computed_actions, action_sizes))
        paper_obj = calculate_main_objective(construct_vectors(paper, action_sizes))
        print('Difference: ', sum(deconstruct_vectors(calculated_obj)) - sum(deconstruct_vectors(paper_obj)))