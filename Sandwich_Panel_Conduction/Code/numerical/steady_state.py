"""
steady_state.py

Numerical solvers for steady-state linear systems.
"""

import numpy as np


def solve_tridiagonal_system(lower, diagonal, upper, rhs):
    """
    Solve a tridiagonal linear system using the Thomas algorithm.

The system has the form:

    lower[i-1] * T[i-1]
    + diagonal[i] * T[i]
    + upper[i] * T[i+1]
    = rhs[i]

    INPUTS:
    ----------
    lower : array
        Lower diagonal coefficients. Length N-1.
    diagonal : array
        Main diagonal coefficients. Length N.
    upper : array
        Upper diagonal coefficients. Length N-1.
    rhs : array
        Right-hand side vector. Length N.

    OUTPUTS:
    -------
    numpy.ndarray
        Solution vector.
    """

    lower = np.asarray(lower, dtype=float).copy()
    diagonal = np.asarray(diagonal, dtype=float).copy()
    upper = np.asarray(upper, dtype=float).copy()
    rhs = np.asarray(rhs, dtype=float).copy()

    N = len(diagonal)

    # --------------------------------------------------------
    # Input validation
    # --------------------------------------------------------

    if N < 2:
        raise ValueError("At least 2 equations are required.")

    if len(lower) != N - 1 or len(upper) != N - 1:
        raise ValueError("Invalid tridiagonal matrix dimensions.")

    if len(rhs) != N:
        raise ValueError("The RHS vector must have length N.")

    if not all(
        np.all(np.isfinite(array))
        for array in (lower, diagonal, upper, rhs)
    ):
        raise ValueError("All input arrays must contain finite values.")

    # --------------------------------------------------------
    # Forward elimination
    # --------------------------------------------------------

    for i in range(1, N):

        if np.isclose(diagonal[i - 1], 0.0):
            raise ZeroDivisionError(
                "Zero pivot encountered during forward elimination."
            )

        factor = lower[i - 1] / diagonal[i - 1]

        diagonal[i] -= factor * upper[i - 1]
        rhs[i] -= factor * rhs[i - 1]

    # --------------------------------------------------------
    # Back substitution
    # --------------------------------------------------------

    if np.isclose(diagonal[-1], 0.0):
        raise ZeroDivisionError(
            "Zero pivot encountered during back substitution."
        )

    solution = np.zeros(N)

    solution[-1] = rhs[-1] / diagonal[-1]

    for i in range(N - 2, -1, -1):

        solution[i] = (
            rhs[i] - upper[i] * solution[i + 1]
        ) / diagonal[i]

    return solution