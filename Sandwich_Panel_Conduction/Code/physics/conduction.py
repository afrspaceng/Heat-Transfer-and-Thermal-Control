
"""
conduction.py

Analytical and numerical solutions for
one-dimensional steady-state heat conduction.
"""
import numpy as np

def analytical_solution(x, q_dot_v, k_eff, L, T_b1, T_b2):
    """
    analytical_solution computes the analytical temperature distribution
    for one-dimensional steady-state heat conduction with uniform
    volumetric heat generation.

    INPUTS
    ----------
    x : array-like
        Spatial coordinates along the panel [m].

    q_dot_v : float
        Uniform volumetric heat generation rate [W/m³].

    k_eff : float
        Effective thermal conductivity of the panel [W/(m K)].

    L : float
        Total length of the panel [m].

    T_b1 : float
        Prescribed temperature at the left boundary (x = 0) [K].

    T_b2 : float
        Prescribed temperature at the right boundary (x = L) [K].

    OUTPUTS
    ----------
    T : ndarray
        Analytical temperature distribution along the panel [K].
    """

    T = (
        T_b1
        + (T_b2 - T_b1) * x / L
        + q_dot_v * x * (L - x) / (2 * k_eff)
    )

    return T


def numerical_solution(N, q_dot_v, k_eff, L, T_b1, T_b2):
    """
    numerical_solution computes the temperature distribution
    for one-dimensional steady-state heat conduction with uniform
    volumetric heat generation using the finite difference method.

    INPUTS
    ----------
    N : int
        Number of spatial discretization nodes.

    q_dot_v : float
        Uniform volumetric heat generation rate [W/m³].

    k_eff : float
        Effective thermal conductivity of the panel [W/(m K)].

    L : float
        Total length of the panel [m].

    T_b1 : float
        Prescribed temperature at the left boundary (x = 0) [K].

    T_b2 : float
        Prescribed temperature at the right boundary (x = L) [K].

    OUTPUTS
    ----------
    x : ndarray of shape (N,)
        Spatial coordinates of the discretization nodes [m].

    T : ndarray of shape (N,)
        Numerical temperature distribution along the panel [K].
    """

    if N < 3:
        raise ValueError("At least 3 nodes are required.")

    # Spatial discretization
    x = np.linspace(0, L, N)

    dx = L / (N - 1)

    # Finite difference coefficient
    C = -q_dot_v * dx**2 / k_eff

    # Matrix construction
    A = np.zeros((N - 2, N - 2))

    b = np.full(N - 2, C)

    # Boundary conditions
    b[0] -= T_b1
    b[-1] -= T_b2

    # Internal node equations
    for i in range(N - 2):

        A[i, i] = -2

        if i > 0:
            A[i, i - 1] = 1

        if i < N - 3:
            A[i, i + 1] = 1

    # Solve the linear system
    T_internal = np.linalg.solve(A, b)

    # Reconstruct the complete temperature vector
    T = np.concatenate((
        [T_b1],
        T_internal,
        [T_b2]
    ))

    return x, T
