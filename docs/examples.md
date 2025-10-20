# Examples

## Problem A.1 
### **Problem Description**
This test problem is a variant of the internet switching model introduced by **Kesselman et al.** and further analyzed by **Facchinei et al**. There are $N$ players, each player having a single variable $x^{\nu} \in \mathbb{R}$. The objective function are given by $$ \theta_{\nu}(\mathbf{x}) := \frac{-x^{\nu}}{x^{1}+\ldots+x^{N}}\left(1-\frac{x^{1}+\ldots+x^{N}}{B}\right) \quad \forall \nu=1, \ldots, N $$ for some constant $B$. We set $N=10$ and $B=1$. The constraints of the first player are $0.3 \leq x^{1} \leq 0.5$, while the remaining players' constraints are $$ x^{1}+\ldots+x^{N} \leq B, \quad x^{\nu} \geq 0.01. $$ Note that the objective functions are not continuous at $x=0$ which, however, is an infeasible point. This variant of the basic problem, described in Example A.14, gives the first player a "privileged status".

### **Python Implementation**
This class defines the **Problem A.1** setup for an Unbounded Generalized Nash Equilibrium Problem (GNEP).
Each method defines one component of the mathematical formulation:


```python
class A1U:

    @staticmethod
    def define_players():
        n = 10
        player_vector_sizes = [1 for _ in range(n)]
        player_objective_functions = [0 for _ in range(n)]  # change to all 0s
        player_constraints = [[1,2], [0,3], [0,3], [0,3], [0,3], [0,3], [0,3], [0,3], [0,3], [0,3]]
        return [player_vector_sizes, player_objective_functions, player_constraints]

    @staticmethod
    def objective_functions():
        return [A1U.obj_func]

    @staticmethod
    def objective_function_derivatives():
        return [A1U.obj_func_der]

    @staticmethod
    def constraints():
        return [A1U.g0, A1U.g1, A1U.g2, A1U.g3]

    @staticmethod
    def constraint_derivatives():
        return [A1U.g0_der, A1U.g1_der, A1U.g2_der, A1U.g3_der]

    @staticmethod
    def obj_func(x: VectorList) -> Vector:
        x = np.concatenate(x).reshape(-1,1)
        s = np.sum(x, axis=0)
        b = 1.0
        obj = -(x / s) * (1.0 - s / b)
        return obj.reshape(-1,1)

    @staticmethod
    def obj_func_der(x: VectorList) -> Vector:
        x = np.concatenate(x).reshape(-1, 1)
        b = 1.0
        s = np.sum(x, axis=0)
        obj = ((x - s) / s ** 2) + (1 / b)
        return obj.reshape(-1,1)

    # === Constraint Functions ===
    @staticmethod
    def g0(x: VectorList) -> Vector:
        x = np.concatenate(x).reshape(-1, 1)
        b = 1.0
        return np.array([np.sum(x, axis=0) - b]).reshape(-1,1)

    @staticmethod
    def g1(x: VectorList) -> Vector:
        # lower bound
        return (0.3 - x[0]).reshape(-1, 1)

    @staticmethod
    def g2(x: VectorList) -> Vector:
        #  upper bound
        return (x[0] - 0.5).reshape(-1,1)

    @staticmethod
    def g3(x: VectorList) -> Vector:
        t = np.vstack([s.reshape(-1,1) for s in x[1:]])
        return (0.01 - t).reshape(-1,1)

    @staticmethod
    def g0_der(x: VectorList) -> Vector:
        return np.array([1]).reshape(-1,1)

    @staticmethod
    def g1_der(x: VectorList) -> Vector:
        return np.array([-1]).reshape(-1,1)

    @staticmethod
    def g2_der(x: VectorList) -> Vector:
        return np.array([1]).reshape(-1,1)

    @staticmethod
    def g3_der(x: VectorList) -> Vector:
        return np.array([-1]).reshape(-1,1)
```
### **Methods**
1. `paper_solution()`  

    Returns the analytical or reference solution reported in the literature.

    **Returns**  

    `list`: A list containing the known equilibrium vector.

2. `define_players()`  

    Defines the number of players, their variable sizes, objective functions, and assigned constraints.  

    **Returns**  

    `list`: A list containing:  
            - `player_vector_sizes`: Dimensions of each player’s decision vector.  
            - `player_objective_functions`: List of indices for objective functions.  
            - `player_constraints`: List mapping each player to its constraints.  

3. `objective_functions()`  

    Returns the objective function(s) used by all players.

    **Returns**  
    
    `list`: A list containing the function `A1U.obj_func`.

4. `objective_function_derivatives()`  

    Returns the derivatives of the player objective functions.  

    **Returns**  
    
    `list`: A list containing the derivative function `A1U.obj_func_der`.

5. `constraints()`  

    Returns all constraint functions used in the model.

    **Returns**  
    
    `list`: A list of functions `[A1U.g0, A1U.g1, A1U.g2, A1U.g3]`.

6. `constraint_derivatives()`  

    Returns the derivatives of each constraint function.

    **Returns**  
    
    `list`: A list containing `[A1U.g0_der, A1U.g1_der, A1U.g2_der, A1U.g3_der]`  

### **Constraints**
| Function | Description                                | Formula                            |
| -------- | ------------------------------------------ |------------------------------------|
| `g0`     | Coupling constraint (shared among players) | $\sum_i x_i - 1 = 0$               |
| `g1`     | Lower bound constraint for player 1        | $0.3 - x_1 \le 0$                  |
| `g2`     | Upper bound constraint for player 1        | $x_1 - 0.5 \le 0$                  |
| `g3`     | Lower bound for all other players          | $0.01 - x_i \le 0,   i = 2,...,10$ |

### **Constraint Derivatives**
| Function | Derivative | Description                              |
| -------- |------------| ---------------------------------------- |
| `g0_der` | 1          | Derivative of shared constraint          |
| `g1_der` | -1         | Lower bound derivative                   |
| `g2_der` | 1          | Upper bound derivative                   |
| `g3_der` | -1         | Lower bound derivative for other players |

### **Solver Integration**
This section shows how the problem setup integrates into the GNEP solver.

```python
    else:
        problem = get_problem(problem_n)
        (player_vector_sizes,
         player_objective_functions,
         player_constraints) = problem['players']

        solver1 = GNEP_Solver_Unbounded(
            problem['obj_funcs'],
            problem['obj_ders'],
            problem['constraints'],
            problem['constraint_ders'],
            player_objective_functions,
            player_constraints,
            player_vector_sizes,
        )
```


