from .utils import *
from .misc import *
from .types import *
from scipy.optimize import minimize

def check_NE(
        result: List[float],
        action_sizes: List[int],
        player_objective_function: List[int],
        objective_functions: List[ObjFunction],
        constraints: List[ConsFunction],
        player_constraints: List[PlayerConstraint],
        paper_res: List[float] = None,
        epsilon=1e-3,
        single_obj_vector=False
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
            result,
            paper_res,
            action_sizes,
            objective_functions,
        )
    print('--------------------------------------')

    conclusion = []
    # Optimize each player by fixing the opponents
    for player_idx, p_o_idx in enumerate(player_objective_function):
        print(f"Player {player_idx + 1}")
        p_var = res_vectors[player_idx]
        p_objective = objective_functions[p_o_idx]  # this is a function
        p_constraints = [constraints[c_idx] for c_idx in player_constraints[p_o_idx] if c_idx]

        if single_obj_vector:
            wrapped_p_objective = create_wrapped_function_single(p_objective, res_vectors, player_idx)
        else:
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
            conclusion.append(True)
            continue

        difference = compare_solutions(
            result,
            deconstruct_vectors(new_vars),
            action_sizes,
            objective_functions,
            solution_name=['Computed', 'Optimized']
        )
        print(f"Difference between Objective Functions of Player {player_idx + 1}: {difference[p_o_idx]}")
        if difference[p_o_idx] > epsilon:
            print('Not at the NE')
            conclusion.append(False)
        else:
            print(f"Computed Solution is at the NE for Player {player_idx + 1}")
            conclusion.append(True)
        print('--------------------------------------')
    return conclusion

def get_problem(problem):
    # Define the problem
    obj = problem.objective_functions()
    obj_der = problem.objective_function_derivatives()
    c = problem.constraints()
    c_der = problem.constraint_derivatives()

    # Describe Players responsibilities
    p = problem.define_players()
    return dict(obj_funcs=obj, obj_ders=obj_der, constraints=c, constraint_ders=c_der, players=p)

def get_initial_point(action_sizes, player_constraints, primal_ip=0.01, dual_ip=10):
    length = len(player_constraints)
    primal = [np.reshape(np.ones(size, dtype=np.float64) * primal_ip , [-1,1]) for size in action_sizes]
    dual = [dual_ip for _ in range(length)]
    return primal, dual


