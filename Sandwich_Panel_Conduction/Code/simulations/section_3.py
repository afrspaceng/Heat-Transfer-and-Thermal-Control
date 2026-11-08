"""
section_3.py

Steady-state thermal analysis of a panel coupled to an EBOX.
"""

import numpy as np

from config.common import (
    EBOX_DISSIPATION,
    T_B1,
    T_B2,
)

from config.section_3 import (
    SECTION_3_NUMERICAL_PARTITION_PARAMETER,
)

from models.geometry import (
    L_X_PANEL,
    L_X_EBOX,
    L_Y_EBOX,
    PANEL_THICKNESS,
)

from models.thermal_properties import (
    calculate_effective_conductivity,
    EBOX_PANEL_CONDUCTANCE,
    EBOX_PANEL_EFFECTIVE_CONDUCTANCE,
)

from numerical.discretization import build_ebox_panel_system

from postprocessing.plotting import plot_section_3


# -------------------------------------------------------------------------
# GEOMETRY AND THERMAL PROPERTIES
# -------------------------------------------------------------------------

k_eff = calculate_effective_conductivity()


# -------------------------------------------------------------------------
# NUMERICAL MESH
# -------------------------------------------------------------------------

n = SECTION_3_NUMERICAL_PARTITION_PARAMETER

i_numerical = 2 * n - 1

number_of_nodes = 7 + 6 * i_numerical

x = np.linspace(0, L_X_PANEL, number_of_nodes)

dx = L_X_PANEL / (number_of_nodes - 1)


# -------------------------------------------------------------------------
# EBOX-PANEL COUPLED SYSTEM
# -------------------------------------------------------------------------

ebox_start_x = (L_X_PANEL - L_X_EBOX) / 2

A, b = build_ebox_panel_system(
    number_of_nodes=number_of_nodes,
    dx=dx,
    effective_conductance=EBOX_PANEL_EFFECTIVE_CONDUCTANCE,
    effective_conductivity=k_eff,
    panel_thickness=PANEL_THICKNESS,
    ebox_conductance=EBOX_PANEL_CONDUCTANCE,
    ebox_dissipation=EBOX_DISSIPATION,
    ebox_width=L_Y_EBOX,
    boundary_temperature_1=T_B1,
    boundary_temperature_2=T_B2,
    ebox_start_x=ebox_start_x,
)


# -------------------------------------------------------------------------
# SOLUTION
# -------------------------------------------------------------------------

solution = np.linalg.solve(A, b)

T_panel = solution[:-1]
T_ebox = solution[-1]


# -------------------------------------------------------------------------
# RESULTS
# -------------------------------------------------------------------------

print("\n========== SECTION 3: STEADY-STATE SOLUTION ==========")

print(f"Number of panel nodes: {number_of_nodes}")
print(f"Total number of unknowns: {len(solution)}")
print(f"Spatial step: {dx:.8f} m")

print(f"\nMaximum panel temperature: {np.max(T_panel) - 273.15:.4f} °C")
print(f"Minimum panel temperature: {np.min(T_panel) - 273.15:.4f} °C")
print(f"EBOX temperature: {T_ebox - 273.15:.4f} °C")


# -------------------------------------------------------------------------
# PLOT
# -------------------------------------------------------------------------

plot_section_3(x, T_panel, T_ebox)