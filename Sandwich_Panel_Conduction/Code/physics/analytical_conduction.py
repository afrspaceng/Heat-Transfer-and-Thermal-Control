"""
analytical_conduction.py

Analytical solutions for steady-state heat conduction.
"""

import numpy as np


def analytical_solution_localized_generation(
    x,
    q_dot_v_ebox,
    q_dot_v_heater,
    k_eff,
    x_b,
    x_c,
    x_d,
    T_b1
):
    """
    analytical_solution_localized_generation computes the analytical
    temperature distribution for one-dimensional steady-state heat
    conduction with localized volumetric heat generation.

    INPUTS
    ----------
    x : array-like
        Spatial coordinates along the panel [m].

    q_dot_v_ebox : float
        Volumetric heat generation rate in the EBOX region [W/m³].

    q_dot_v_heater : float
        Volumetric heat generation rate in the heater region [W/m³].

    k_eff : float
        Effective thermal conductivity of the panel [W/(m K)].

    x_b : float
        Position of the beginning of the EBOX region [m].

    x_c : float
        Position of the beginning of the heater region [m].

    x_d : float
        Position of the panel center, where the symmetry condition
        is imposed [m].

    T_b1 : float
        Prescribed temperature at the left boundary of the panel [K].

    OUTPUTS
    ----------
    T : ndarray
        Analytical temperature distribution along the panel [K].
        The temperature is calculated separately for the three
        regions, taking into account their respective volumetric
        heat generation rates and the symmetry condition at
        the panel center.
    """

    # --------------------------------------------------------
    # Integration constants
    # --------------------------------------------------------

    C2 = T_b1

    C5 = (
        (q_dot_v_ebox + q_dot_v_heater) / k_eff
    ) * x_d

    C3 = (
        -q_dot_v_heater / k_eff * x_c
        + C5
    )

    C1 = (
        -q_dot_v_ebox / k_eff * x_b
        + C3
    )

    C4 = (
        C1 * x_b
        + C2
        - C3 * x_b
        + q_dot_v_ebox / (2 * k_eff) * x_b**2
    )

    C6 = (
        C3 * x_c
        + C4
        + q_dot_v_heater / (2 * k_eff) * x_c**2
        - C5 * x_c
    )

    # --------------------------------------------------------
    # Temperature distribution
    # --------------------------------------------------------

    T = np.zeros_like(x, dtype=float)

    region_1 = x < x_b

    region_2 = (
        (x >= x_b)
        & (x < x_c)
    )

    region_3 = (
        (x >= x_c)
        & (x <= x_d)
    )

    T[region_1] = (
        C1 * x[region_1] + C2
    )

    T[region_2] = (
        -q_dot_v_ebox / (2 * k_eff)
        * x[region_2]**2
        + C3 * x[region_2]
        + C4
    )

    T[region_3] = (
        -(q_dot_v_ebox + q_dot_v_heater)
        / (2 * k_eff)
        * x[region_3]**2
        + C5 * x[region_3]
        + C6
    )

    return T