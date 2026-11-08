
import numpy as np

# Nodes for both analytical and numerical calculations
from config.section_1 import (
    SECTION_1_N_VALUES,
    SECTION_1_ANALYTICAL_POINTS,
)

# Heater and EBOX dissipation and boundary conditions
from config.common import (
    EBOX_DISSIPATION,
    HEATER_DISSIPATION,
    T_B1, # Boundary condition for 1st point of panel
    T_B2 # Boundary condition for 2nd point of panel
)

# Dimensiones of the panel. Being this section about studying 
# one-dimensional problem, only one dimension is needed.
from models.geometry import L_X_PANEL

# Within this module, both analytical and numerical problem are solved.
from physics.conduction import (
    analytical_solution,
    numerical_solution
)

# Thermal properties calculation for the overall and complete structure.
from models.thermal_properties import (
    calculate_effective_conductivity,
    calculate_volumetric_heat_generation
)

# In order to plot the results
from postprocessing.plotting import plot_temperature_distribution

# ============================================================
# THERMAL PROPERTIES
# ============================================================

# Effective conductivity of the panel [W/(mK)]
k_eff = calculate_effective_conductivity()

# Volumetric heat generation of both devices that generate heat flux [W/(m3)]
q_dot_v = calculate_volumetric_heat_generation(
    EBOX_DISSIPATION,
    HEATER_DISSIPATION
)


# ============================================================
# NUMERICAL AND ANALYTICAL SOLUTIONS
# ============================================================
numerical_results = []
for N in SECTION_1_N_VALUES:

    print(f"\n========== N = {N} nodes ==========")

    # Numerical solution
    x, T_numerical = numerical_solution(
        N,
        q_dot_v,
        k_eff,
        L_X_PANEL,
        T_B1,
        T_B2
    )

    # Analytical solution
    T_analytical = analytical_solution(
        x,
        q_dot_v,
        k_eff,
        L_X_PANEL,
        T_B1,
        T_B2
    )

    numerical_results.append((x, T_numerical, N))

    # Comparison
    max_difference = np.max(
        np.abs(T_analytical - T_numerical)
    )

    print(f"Maximum absolute difference: {max_difference:.3e} K")
    print(f"Maximum numerical temperature: {np.max(T_numerical) - 273.15:.2f} °C")
    print(f"Maximum analytical temperature: {np.max(T_analytical) - 273.15:.2f} °C")

# ============================================================
# TEMPERATURE DISTRIBUTION PLOT
# ============================================================

x_analytical = np.linspace(0, L_X_PANEL, SECTION_1_ANALYTICAL_POINTS)

T_analytical = analytical_solution(
    x_analytical,
    q_dot_v,
    k_eff,
    L_X_PANEL,
    T_B1,
    T_B2
)

plot_temperature_distribution(
    x_analytical,
    T_analytical,
    numerical_results
)