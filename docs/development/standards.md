# 📖 Docstring Guidelines (NumPy Style)

This document defines the **standard docstring format** to be used across the project.  
We follow the **NumPy style** conventions, as supported by [mkdocstrings](https://mkdocstrings.github.io) and rendered by [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/).

---

## 1. General Principles

- Every **public function, class, method, and module** must have a docstring.  
- Use **NumPy style sections** (`Parameters`, `Returns`, `Raises`, `Examples`, etc.).  
- Start with a **short, one-line summary** in imperative mood (e.g., *"Compute equilibrium strategies"*, not *"This function computes..."*).  
- Follow with an **extended description** if necessary.  
- Keep line length reasonable (~79–100 chars).  
- **Private functions** (starting with `_`) may have shorter docstrings, but should still explain purpose if non-trivial.  

---

## 2. Function & Method Docstrings

### Template

```python
def function_name(param1: int, param2: str = "default") -> float:
    """
    One-line summary of the function.

    Extended description if necessary. Can span multiple sentences to
    explain the context, usage, or algorithm.

    Parameters
    ----------
    param1 : int
        Description of the parameter. Include units if applicable.
    param2 : str, optional
        Description of the parameter. State the default if not obvious
        from the signature.

    Returns
    -------
    float
        Description of the return value.

    Raises
    ------
    ValueError
        Explanation of the condition that causes this exception.

    Examples
    --------
    >>> result = function_name(5, "option")
    >>> print(result)
    3.14
    """
```

## 3. Function & Method Docstrings
### Template

```python
class Solver:
    """
    Generalized Nash Equilibrium solver.

    Provides algorithms to compute equilibria in generalized Nash games.

    Parameters
    ----------
    tol : float, optional
        Convergence tolerance. Default is ``1e-6``.
    max_iter : int, optional
        Maximum number of iterations. Default is ``1000``.

    Attributes
    ----------
    tol : float
        Convergence tolerance.
    max_iter : int
        Maximum number of iterations.

    See Also
    --------
    create_game : Utility to create game instances.
    flatten_variables : Function used internally to reshape variables.
    """

```

## Module Docstrings
At the top of every Python file:

```
"""
Utilities for game creation.

This module provides helper functions to build and initialize
generalized Nash equilibrium problems.
"""

```

## Sections to Use

| Section        | When to Use                      | Notes                                               |
| -------------- | -------------------------------- | --------------------------------------------------- |
| **Parameters** | Always                           | Document type, shape, and description.              |
| **Returns**    | If function returns something    | Be explicit about types and shapes.                 |
| **Raises**     | If function can raise exceptions | Name the exception and explain why.                 |
| **Attributes** | For classes                      | List important public attributes.                   |
| **See Also**   | Optional                         | Cross-reference related functions/classes.          |
| **Examples**   | Strongly encouraged              | Use `>>>` doctest style. Keep minimal but runnable. |

## Examples Style

* Always use >>> prompt.
* Show inputs and outputs.
* Keep examples small, self-contained, and copy-pasteable.
* Example blocks must be valid Python so they can be tested via pytest --doctest-modules.

### Good Example:
```commandline
Examples
--------
>>> import numpy as np
>>> vectors = [np.array([[1.0], [2.0]]), np.array([[3.0]])]
>>> scalars = [4.0, 5.0]
>>> flatten_variables(vectors, scalars)
[1.0, 2.0, 3.0, 4.0, 5.0]

```

## Do's and Don'ts

✅ Do:

* Use backticks for code (e.g., "matrix").
* Be explicit with shapes (array of shape (n, m)).
* Keep summaries short and imperative.

❌ Don’t:

* Repeat type hints unnecessarily: "param1 : int, an integer" is redundant.
* Write personal notes or informal language.
* Omit examples for important public functions.