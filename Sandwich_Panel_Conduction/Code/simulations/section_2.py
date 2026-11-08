"""
section_2.py

Simulation of one-dimensional steady-state heat conduction
with localized volumetric heat generation.
"""

import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================

from config.common import (
    T_B1,
    EBOX_DISSIPATION,
    HEATER_DISSIPATION
)

from config.section_2 import (
    SECTION_2_ANALYTICAL_PARTITION_PARAMETER,
    SECTION_2_NUMERICAL_PARTITION_PARAMETER
)


# ============================================================
# MODELS
# ============================================================

from models.geometry import (
    L_X_PANEL,
    L_X_EBOX,
    L_X_HEATER,
    L_Y_PANEL,
    PANEL_THICKNESS
)

from models.thermal_properties import (
    calculate_effective_conductivity,
    calculate_localized_heat_generation
)


# ============================================================
# PHYSICS
# ============================================================

from physics.analytical_conduction import (
    analytical_solution_localized_generation
)


# ============================================================
# NUMERICAL METHODS
# ============================================================

from numerical.discretization import (
    build_localized_generation_system
)

from numerical.steady_state import (
    solve_tridiagonal_system
)


# ============================================================
# POSTPROCESSING
# ============================================================

from postprocessing.plotting import (
    plot_section_2_half_panel,
    plot_section_2_full_panel
)

# ============================================================
# GEOMETRICAL BOUNDARIES
# ============================================================

X_A = 0.0

X_B = L_X_PANEL / 2 - L_X_EBOX / 2

X_C = L_X_PANEL / 2 - L_X_HEATER / 2

X_D = L_X_PANEL / 2


# ============================================================
# THERMAL PROPERTIES
# ============================================================

K_EFF = calculate_effective_conductivity()

Q_DOT_V_EBOX = calculate_localized_heat_generation(
    EBOX_DISSIPATION,
    L_X_EBOX,
    L_Y_PANEL,
    PANEL_THICKNESS
)

Q_DOT_V_HEATER = calculate_localized_heat_generation(
    HEATER_DISSIPATION,
    L_X_HEATER,
    L_Y_PANEL,
    PANEL_THICKNESS
)

# ============================================================
# MESH GENERATION
# ============================================================

def create_half_panel_mesh(partition_parameter):
    """
    create_half_panel_mesh generates a uniform spatial mesh
    along half of the sandwich panel.

    INPUTS
    ----------
    partition_parameter : int
        Partition parameter used to determine the total number
        of discretization nodes according to the original
        MATLAB formulation. Must be a positive integer.

    OUTPUTS
    ----------
    x : ndarray
        Spatial coordinates of the discretization nodes along
        half of the panel, from x = 0 to x = X_D [m].
    """

    if not isinstance(partition_parameter, (int, np.integer)):
        raise TypeError("The partition parameter must be an integer.")

    if partition_parameter < 1:
        raise ValueError("The partition parameter must be positive.")

    i = 2 * partition_parameter - 1

    N = 7 + 6 * i

    x = np.linspace(0, X_D, N)

    return x


# Analytical mesh
x_analytical = create_half_panel_mesh(
    SECTION_2_ANALYTICAL_PARTITION_PARAMETER
)

# Numerical mesh
x_numerical = create_half_panel_mesh(
    SECTION_2_NUMERICAL_PARTITION_PARAMETER
)

# ============================================================
# ANALYTICAL SOLUTION
# ============================================================

T_analytical = analytical_solution_localized_generation(
    x_analytical,
    Q_DOT_V_EBOX,
    Q_DOT_V_HEATER,
    K_EFF,
    X_B,
    X_C,
    X_D,
    T_B1
)

# ============================================================
# NUMERICAL SOLUTION
# ============================================================

# Build the tridiagonal system
lower, diagonal, upper, rhs = build_localized_generation_system(
    x_numerical,
    x_numerical[1] - x_numerical[0],
    Q_DOT_V_EBOX,
    Q_DOT_V_HEATER,
    K_EFF,
    X_B,
    X_C,
    T_B1
)

# Solve the system using the Thomas algorithm
T_numerical = solve_tridiagonal_system(
    lower,
    diagonal,
    upper,
    rhs
)

# ============================================================
# ANALYTICAL SOLUTION VERIFICATION
# ============================================================

print("\n========== Analytical solution ==========")

print(
    f"Maximum temperature: "
    f"{np.max(T_analytical) - 273.15:.4f} °C"
)

print(
    f"Temperature at panel boundary: "
    f"{T_analytical[0] - 273.15:.4f} °C"
)

print(
    f"Temperature at panel center: "
    f"{T_analytical[-1] - 273.15:.4f} °C"
)

# ============================================================
# NUMERICAL SOLUTION VERIFICATION
# ============================================================

print("\n========== Numerical solution ==========")

print(f"Number of nodes: {len(T_numerical)}")

print(
    f"Maximum temperature: "
    f"{np.max(T_numerical) - 273.15:.4f} °C"
)

print(
    f"Temperature at panel boundary: "
    f"{T_numerical[0] - 273.15:.4f} °C"
)

print(
    f"Temperature at panel center: "
    f"{T_numerical[-1] - 273.15:.4f} °C"
)

# ============================================================
# FULL PANEL TEMPERATURE DISTRIBUTION
# ============================================================

# Analytical solution
x_analytical_full = np.concatenate((
    x_analytical,
    L_X_PANEL - x_analytical[-2::-1]
))

T_analytical_full = np.concatenate((
    T_analytical,
    T_analytical[-2::-1]
))


# Numerical solution
x_numerical_full = np.concatenate((
    x_numerical,
    L_X_PANEL - x_numerical[-2::-1]
))

T_numerical_full = np.concatenate((
    T_numerical,
    T_numerical[-2::-1]
))

# ============================================================
# SYMMETRY VERIFICATION
# ============================================================

analytical_symmetry_error = np.max(
    np.abs(T_analytical_full - T_analytical_full[::-1])
)

numerical_symmetry_error = np.max(
    np.abs(T_numerical_full - T_numerical_full[::-1])
)


# ============================================================
# FULL PANEL VERIFICATION
# ============================================================

print("\n========== Full panel ==========")

print(f"Analytical nodes: {len(x_analytical_full)}")
print(f"Numerical nodes: {len(x_numerical_full)}")

print(
    f"Analytical panel length: "
    f"{x_analytical_full[-1]:.4f} m"
)

print(
    f"Numerical panel length: "
    f"{x_numerical_full[-1]:.4f} m"
)

# --------------------------------------------------------
# Analytical solution evaluated on numerical mesh
# --------------------------------------------------------

T_analytical_on_numerical_mesh = (
    analytical_solution_localized_generation(
        x_numerical,
        Q_DOT_V_EBOX,
        Q_DOT_V_HEATER,
        K_EFF,
        X_B,
        X_C,
        X_D,
        T_B1
    )
)

absolute_error = np.abs(
    T_numerical - T_analytical_on_numerical_mesh
)

max_error_index = np.argmax(absolute_error)

dx_numerical = x_numerical[1] - x_numerical[0]

print(
    f"Maximum absolute difference: "
    f"{np.max(absolute_error):.6e} K"
)

print(
    f"Maximum error location: "
    f"{x_numerical[max_error_index]:.6f} m"
)

print(
    f"Error at X_B: "
    f"{absolute_error[round(X_B / dx_numerical)]:.6e} K"
)

print(
    f"Error at X_C: "
    f"{absolute_error[round(X_C / dx_numerical)]:.6e} K"
)

print(
    f"Error at X_D: "
    f"{absolute_error[-1]:.6e} K"
)

# ============================================================
# MESH VERIFICATION
# ============================================================

for mesh_name, x in [
    ("Analytical", x_analytical),
    ("Numerical", x_numerical)
]:

    dx = x[1] - x[0]

    print(f"\n========== {mesh_name} mesh ==========")

    print(f"Number of nodes: {len(x)}")
    print(f"Spatial step: {dx:.8f} m")

    for boundary_name, boundary_position in [
        ("X_B", X_B),
        ("X_C", X_C),
        ("X_D", X_D)
    ]:

        node_index = round(boundary_position / dx)

        is_aligned = np.isclose(
            x[node_index],
            boundary_position
        )

        print(
            f"{boundary_name}: "
            f"x = {x[node_index]:.6f} m, "
            f"node = {node_index}, "
            f"aligned = {is_aligned}"
        )


# ============================================================
# THERMAL PROPERTY VERIFICATION
# ============================================================

print("\n========== Thermal properties ==========")

print(f"Effective conductivity: {K_EFF:.4f} W/(m K)")
print(f"EBOX heat generation: {Q_DOT_V_EBOX:.2f} W/m^3")
print(f"Heater heat generation: {Q_DOT_V_HEATER:.2f} W/m^3")

# ============================================================
# PLOTS
# ============================================================

plot_section_2_half_panel(
    x_analytical,
    T_analytical,
    x_numerical,
    T_numerical,
    X_B,
    X_C,
    X_D
)

plot_section_2_full_panel(
    x_analytical_full,
    T_analytical_full,
    x_numerical_full,
    T_numerical_full,
    X_B,
    X_C,
    X_D,
    L_X_PANEL
)