from .utils import *
from .misc import *

def check_NE(
        result: List[np.float64],
        action_sizes: List[int],
        player_objective_function: List[int],
        objective_functions: List[Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]],
        constraints: List[Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]],
        player_constraints: List[List[int]],
        paper_res: List[float] = None,
        epsilon=1e-3
):
    res = np.array(result).reshape(-1, 1)
    res_vectors = construct_vectors(res, action_sizes)
    print('Computed Solution: \n', res_vectors)

    action_splits = np.cumsum(np.insert(action_sizes, 0, 0))

    # Get the obj function values at the current NE
    obj_func_vals = objective_check(objective_functions, res_vectors) # vector with values of each objective function
    print('Computed Objective Function: \n', obj_func_vals)

    # Check that constraints are being satisfied
    constraint_vals, constraint_sat = constraint_check(constraints, res_vectors)
    if all(constraint_sat):
        print('Constraints Satisfied')
    else:
        print('Constraints Not Satisfied')

    if paper_res is not None:
        compare_solutions(
            res,
            paper_res,
            action_sizes,
            objective_functions,
        )
    print('--------------------------------------')

    # Optimize each player by fixing the opponents
    for player_idx, p_o_idx in enumerate(player_objective_function):
        print(f"Player {player_idx + 1}")
        p_var = res_vectors[player_idx]
        p_objective = objective_functions[p_o_idx]  # this is a function
        p_constraints = [constraints[c_idx] for c_idx in player_constraints[p_o_idx] if c_idx]

        wrapped_p_objective = create_wrapped_function(p_objective, res_vectors, player_idx)

        optimization_constraints = [
            {'type': 'ineq', 'fun': lambda x: create_wrapped_function(constraint, res_vectors, player_idx)(x).ravel()}
            for constraint in p_constraints
        ]

        p_var_0 = np.zeros_like(p_var).flatten()

        # p_var_0 = p_var.flatten()
        result = minimize(
            wrapped_p_objective,
            p_var_0,
            method='SLSQP',
            constraints=optimization_constraints,
            options={
                'disp': False,
                'maxiter': 1000,
                'ftol': 1e-5
            }
        )
        # Report
        fixed_vars = res_vectors[:player_idx] + res_vectors[player_idx + 1:]
        opt_actions = np.array(result.x).reshape(-1, 1)
        new_vars = fixed_vars[:player_idx] + [opt_actions] + fixed_vars[player_idx:]
        opt_obj_func = objective_check(objective_functions, new_vars)
        opt_constraint_vals, opt_constraint_sat = constraint_check(constraints, new_vars)


        print(f"Optimized Actions: {result.x}\nOptimized Function value: {result.fun}")
        print('Optimized Objective Functions: \n', np.array(opt_obj_func).reshape(-1, 1))
        if all(opt_constraint_sat):
            print('Constraints Satisfied')
        else:
            print('Constraints Not Satisfied')

        difference = compare_solutions(
            res,
            deconstruct_vectors(new_vars),
            action_sizes,
            objective_functions,
            solution_name=['Computed', 'Optimized']
        )
        print(f"Difference between Objective Functions of Player {player_idx + 1}: {difference[p_o_idx]}")
        if difference[p_o_idx] > epsilon:
            print('Not at the NE')
        else:
            print(f"Computed Solution is at the NE for Player {player_idx + 1}")
        print('--------------------------------------')
    return