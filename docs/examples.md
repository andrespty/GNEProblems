# Examples

## Example A.1 
### **Problem Description**
This test problem is a variant of the internet switching model introduced by **Kesselman et al.** and further analyzed by **Facchinei et al**. There are $N$ players, each player having a single variable $x^{\nu} \in \mathbb{R}$. The objective function are given by $$ \theta_{\nu}(\mathbf{x}) := \frac{-x^{\nu}}{x^{1}+\ldots+x^{N}}\left(1-\frac{x^{1}+\ldots+x^{N}}{B}\right) \quad \forall \nu=1, \ldots, N $$ for some constant $B$. We set $N=10$ and $B=1$. The constraints of the first player are $0.3 \leq x^{1} \leq 0.5$, while the remaining players' constraints are $$ x^{1}+\ldots+x^{N} \leq B, \quad x^{\nu} \geq 0.01. $$ Note that the objective functions are not continuous at $x=0$ which, however, is an infeasible point. This variant of the basic problem, described in Example A.14, gives the first player a "privileged status".

### **Python Implementation**
This class implements the Example A.1 problem for a Generalized Nash Equilibrium Problem (GNEP).
Each method defines one component of the mathematical formulation:

<details>
  <summary>Click to see code</summary>

```python
class A1U:
    @staticmethod
    def paper_solution():
        value_1 = [0.29923815223336,
                   0.06951127617805,
                   0.06951127617805,
                   0.06951127617805,
                   0.06951127617805,
                   0.06951127617805,
                   0.06951127617805,
                   0.06951127617805,
                   0.06951127617805,
                   0.06951127617805]
        return [value_1]

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
</details>

---
| Method                           | Description                                                                         | Returns                                                              |
|----------------------------------|-------------------------------------------------------------------------------------|----------------------------------------------------------------------|
| `paper_solution()`               | Returns the analytical solution reported in the reference paper.                    | `list`: A list containing the known equilibrium vector.
| `define_players()`               | Defines the number of players, their objective indices, and constraint assignments. | `list`: <ol><li>`player_vector_sizes`: dimensions of each player’s decision vector (all 1)</li><li>`player_objective_functions`: list of indices for objective functions</li><li>`player_constraints`: mapping of each player to its constraint set</li></ol>                     | 
| `obj_func()`                     | Implements the player objective \( \theta_{\nu}(\mathbf{x}) \).                     | `list`: A list containing the function `A1U.obj_func`
| `objective_function_derivatives()` | Computes the derivative of the player objective.| `lis`t: A list containing the derivative function `A1U.obj_func_der`.
| `constraints()`| Lists all constraint functions| `list`: A list of functions `[A1U.g0, A1U.g1, A1U.g2, A1U.g3]`.
| `constraint_derivatives()`| Returns the derivatives of the constraint functions| `list`: A list containing `[A1U.g0_der, A1U.g1_der, A1U.g2_der, A1U.g3_der]`.
| `g0–g3()`                        | Define the shared and individual inequality constraints.                            | 
| `g0_der–g3_der()`                | Return the gradient of each constraint.                                             | 













