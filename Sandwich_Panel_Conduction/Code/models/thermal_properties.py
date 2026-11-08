
"""
thermal_properties.py

Calculation of equivalent thermal properties
for the sandwich panel.
"""

# Geometry of the different parts of the structure
from models.geometry import (
    L_X_PANEL,
    L_Y_PANEL,
    HONEYCOMB_THICKNESS,
    CFRP_THICKNESS,
    PANEL_THICKNESS
)

# Material conductivities
from models.materials import (
    K_X_HONEYCOMB,
    K_X_CFRP
)

# EBOX-panel thermal coupling
EBOX_PANEL_CONDUCTANCE = 200  # [W/(m^2 K)]
EBOX_PANEL_EFFECTIVE_CONDUCTANCE = EBOX_PANEL_CONDUCTANCE / 2
EBOX_THERMAL_CAPACITY = 1900  # [J/K]


def calculate_effective_conductivity():
    """
    Calculate the equivalent thermal conductivity
    along the X direction of the sandwich panel.

    OUTPUTS:
    -------
        float: Equivalent thermal conductivity [W/(m K)]
    """

    k_eff = (
        K_X_HONEYCOMB * HONEYCOMB_THICKNESS
        + 2 * K_X_CFRP * CFRP_THICKNESS
    ) / PANEL_THICKNESS

    return k_eff


def calculate_volumetric_heat_generation(ebox_dissipation, heater_dissipation):
    """
    Calculate the uniform volumetric heat generation
    across the entire sandwich panel.

    INPUTS:
    ----------
        ebox_dissipation (float): EBOX heat dissipation [W]
        heater_dissipation (float): Heater heat dissipation [W]

    OUTPUTS:
    -------
        float: Volumetric heat generation [W/m^3]
    """

    panel_volume = L_X_PANEL * L_Y_PANEL * PANEL_THICKNESS

    q_dot_v = (ebox_dissipation + heater_dissipation) / panel_volume

    return q_dot_v


def calculate_localized_heat_generation(
    dissipation,
    length,
    width,
    thickness
):
    """
    Calculate volumetric heat generation for a localized
    heat source.

    INPUTS:
    ----------
    dissipation : float
        Total dissipated power [W].
    length : float
        Heat source length [m].
    width : float
        Effective heat source width [m].
    thickness : float
        Panel thickness [m].

    OUTPUTS:
    -------
    float
        Volumetric heat generation [W/m^3].
    """

    volume = length * width * thickness

    q_dot_v = dissipation / volume

    return q_dot_v

