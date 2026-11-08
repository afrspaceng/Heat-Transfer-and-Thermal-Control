"""
section_4.py

Transient thermal simulation of a sandwich panel coupled to an EBOX.
The explicit Euler method is used for time integration.
"""

import numpy as np
from pathlib import Path

from config.common import (
    EBOX_DISSIPATION,
    HEATER_DISSIPATION,
    T_B1,
    T_B2,
)

from config.section_4 import (
    SECTION_4_NUMERICAL_PARTITION_PARAMETER,
    SECTION_4_TOTAL_TIME,
    SECTION_4_STEFAN_BOLTZMANN_CONSTANT,
    SECTION_4_EMISSIVITY,
    SECTION_4_RADIATION_TEMPERATURE,
    SECTION_4_HEATER_SWITCH_OFF_TEMPERATURE,
    SECTION_4_INITIAL_TEMPERATURE,
)

from models.geometry import (
    L_X_PANEL,
    L_Y_PANEL,
    PANEL_THICKNESS,
    L_X_EBOX,
    L_Y_EBOX,
    L_X_HEATER,
    L_Y_HEATER,
    HONEYCOMB_THICKNESS,
    CFRP_THICKNESS,
)

from models.materials import (
    RHO_HONEYCOMB,
    RHO_CFRP,
    CP_HONEYCOMB,
    CP_CFRP,
)

from models.thermal_properties import (
    calculate_effective_conductivity,
    EBOX_PANEL_CONDUCTANCE,
    EBOX_THERMAL_CAPACITY,
)

from postprocessing.plotting import plot_section_4


# -------------------------------------------------------------------------
# THERMAL PROPERTIES
# -------------------------------------------------------------------------

k_eff = calculate_effective_conductivity()

rho_effective = (
    2 * RHO_CFRP * CFRP_THICKNESS
    + RHO_HONEYCOMB * HONEYCOMB_THICKNESS
) / PANEL_THICKNESS

c_effective = (
    2 * RHO_CFRP * CP_CFRP * L_Y_PANEL * L_X_PANEL * CFRP_THICKNESS
    + RHO_HONEYCOMB * CP_HONEYCOMB
    * L_Y_PANEL * L_X_PANEL * HONEYCOMB_THICKNESS
    + EBOX_THERMAL_CAPACITY
) / (
    PANEL_THICKNESS
    * L_X_PANEL
    * L_Y_PANEL
    * rho_effective
)

thermal_diffusivity = k_eff / (rho_effective * c_effective)


# -------------------------------------------------------------------------
# NUMERICAL MESH
# -------------------------------------------------------------------------

n = SECTION_4_NUMERICAL_PARTITION_PARAMETER

i_numerical = 2 * n - 1

number_of_nodes = 7 + 6 * i_numerical

x = np.linspace(0, L_X_PANEL, number_of_nodes)

dx = L_X_PANEL / (number_of_nodes - 1)

cell_volume = dx * L_Y_PANEL * PANEL_THICKNESS


# -------------------------------------------------------------------------
# EBOX AND HEATER CONTACT NODES
# -------------------------------------------------------------------------

ebox_start_idx = int(np.floor((L_X_PANEL - L_X_EBOX) / 2 / dx))
ebox_end_idx = number_of_nodes - 1 - ebox_start_idx

heater_start_idx = int(np.floor((L_X_PANEL - L_X_HEATER) / 2 / dx))
heater_end_idx = number_of_nodes - 1 - heater_start_idx


# Contact weights: half contribution at each end.
ebox_weights = np.zeros(number_of_nodes)
ebox_weights[ebox_start_idx:ebox_end_idx + 1] = 1.0
ebox_weights[ebox_start_idx] = 0.5
ebox_weights[ebox_end_idx] = 0.5

heater_weights = np.zeros(number_of_nodes)
heater_weights[heater_start_idx:heater_end_idx + 1] = 1.0
heater_weights[heater_start_idx] = 0.5
heater_weights[heater_end_idx] = 0.5


# -------------------------------------------------------------------------
# RADIATION WEIGHTS
# -------------------------------------------------------------------------

radiation_weights = np.zeros(number_of_nodes)

radiation_weights[1:ebox_start_idx] = 2.0
radiation_weights[ebox_start_idx] = 1.5

radiation_weights[ebox_start_idx + 1:ebox_end_idx] = 1.0
radiation_weights[ebox_end_idx] = 1.5

radiation_weights[ebox_end_idx + 1:number_of_nodes - 1] = 2.0


# -------------------------------------------------------------------------
# TIME DISCRETIZATION
# -------------------------------------------------------------------------

dt = dx / 2

number_of_time_steps = int(np.floor(SECTION_4_TOTAL_TIME / dt))

time = np.arange(number_of_time_steps + 1) * dt


# -------------------------------------------------------------------------
# INITIAL TEMPERATURE FIELD
# -------------------------------------------------------------------------

# The final unknown is the EBOX temperature.
T = np.full(number_of_nodes + 1, SECTION_4_INITIAL_TEMPERATURE)

T[0] = T_B1
T[number_of_nodes - 1] = T_B2


# -------------------------------------------------------------------------
# STORAGE OF RESULTS
# -------------------------------------------------------------------------

# MATLAB rows 10000, 200000 and 500000 correspond to these
# numbers of completed time steps.
snapshot_steps = {
    9999: 10000,
    199999: 200000,
    499999: 500000,
}

temperature_profiles = {}

ebox_temperature_history = np.zeros(number_of_time_steps + 1)
ebox_temperature_history[0] = T[-1]


# -------------------------------------------------------------------------
# CONSTANT HEATER POWER DISTRIBUTION
# -------------------------------------------------------------------------

heater_power_per_node = (
    HEATER_DISSIPATION
    / (L_Y_HEATER * PANEL_THICKNESS * L_X_HEATER)
    * dx
    * PANEL_THICKNESS
    * L_Y_PANEL
    * heater_weights
)


# -------------------------------------------------------------------------
# TRANSIENT SIMULATION: EXPLICIT EULER
# -------------------------------------------------------------------------

for step in range(number_of_time_steps):

    # Current temperatures.
    T_panel = T[:number_of_nodes]
    T_ebox = T[-1]

    # Initialize heat flow contributions.
    panel_heat_flow = np.zeros(number_of_nodes)

    # ---------------------------------------------------------------------
    # PANEL CONDUCTION
    # ---------------------------------------------------------------------

    panel_heat_flow[1:-1] += (
        k_eff
        / dx**2
        * cell_volume
        * (
            T_panel[:-2]
            + T_panel[2:]
            - 2 * T_panel[1:-1]
        )
    )

    # ---------------------------------------------------------------------
    # EBOX-TO-PANEL CONDUCTION
    # ---------------------------------------------------------------------

    ebox_to_panel = (
        EBOX_PANEL_CONDUCTANCE
        * L_Y_EBOX
        * dx
        * ebox_weights
        * (T_ebox - T_panel)
    )

    panel_heat_flow += ebox_to_panel

    # ---------------------------------------------------------------------
    # RADIATION TO THE PANEL
    # ---------------------------------------------------------------------

    panel_radiation = (
        SECTION_4_STEFAN_BOLTZMANN_CONSTANT
        * SECTION_4_EMISSIVITY
        * dx
        * L_Y_PANEL
        * radiation_weights
        * (
            SECTION_4_RADIATION_TEMPERATURE**4
            - T_panel**4
        )
    )

    panel_heat_flow += panel_radiation

    # ---------------------------------------------------------------------
    # HEATER-TO-PANEL HEAT TRANSFER
    # ---------------------------------------------------------------------

    if T_ebox < SECTION_4_HEATER_SWITCH_OFF_TEMPERATURE:
        heater_to_panel = heater_power_per_node
    else:
        heater_to_panel = np.zeros(number_of_nodes)

    panel_heat_flow += heater_to_panel

    # ---------------------------------------------------------------------
    # INTERNAL EBOX HEAT GENERATION
    # ---------------------------------------------------------------------

    if T_ebox >= SECTION_4_HEATER_SWITCH_OFF_TEMPERATURE:
        internal_ebox_heat = EBOX_DISSIPATION
    else:
        internal_ebox_heat = 0.0

    # ---------------------------------------------------------------------
    # PANEL-TO-EBOX CONDUCTION
    # ---------------------------------------------------------------------

    panel_to_ebox = -np.sum(ebox_to_panel)

    # ---------------------------------------------------------------------
    # RADIATION TO THE EBOX
    # ---------------------------------------------------------------------

    ebox_radiation = (
        SECTION_4_STEFAN_BOLTZMANN_CONSTANT
        * SECTION_4_EMISSIVITY
        * L_X_EBOX
        * L_Y_EBOX
        * (
            SECTION_4_RADIATION_TEMPERATURE**4
            - T_ebox**4
        )
    )

    # ---------------------------------------------------------------------
    # EXPLICIT EULER TEMPERATURE UPDATE
    # ---------------------------------------------------------------------

    T_next = T.copy()

    T_next[1:number_of_nodes - 1] = (
        T_panel[1:-1]
        + dt
        / (c_effective * rho_effective * cell_volume)
        * panel_heat_flow[1:-1]
    )

    T_next[-1] = (
        T_ebox
        + dt / EBOX_THERMAL_CAPACITY
        * (
            internal_ebox_heat
            + panel_to_ebox
            + ebox_radiation
        )
    )

    T = T_next

    # Store EBOX temperature history.
    ebox_temperature_history[step + 1] = T[-1]

    # Store requested panel temperature profiles.
    completed_steps = step + 1

    if completed_steps in snapshot_steps:
        matlab_row = snapshot_steps[completed_steps]
        temperature_profiles[matlab_row] = T[:number_of_nodes].copy()


# -------------------------------------------------------------------------
# FINAL STEADY-STATE APPROXIMATION
# -------------------------------------------------------------------------

T_steady_state = T.copy()

temperature_profiles[number_of_time_steps + 1] = (
    T_steady_state[:number_of_nodes].copy()
)

# Store the final state for subsequent sections.
results_directory = Path("results")
results_directory.mkdir(exist_ok=True)

np.savez(
    results_directory / "section_4_steady_state.npz",
    temperature=T_steady_state,
    x=x,
    time=SECTION_4_TOTAL_TIME,
)


# -------------------------------------------------------------------------
# RESULTS
# -------------------------------------------------------------------------

print("\n========== SECTION 4: TRANSIENT SOLUTION ==========")

print(f"Number of panel nodes: {number_of_nodes}")
print(f"Total number of unknowns: {number_of_nodes + 1}")
print(f"Spatial step: {dx:.8f} m")
print(f"Time step: {dt:.8f} s")
print(f"Number of time steps: {number_of_time_steps}")
print(f"Total simulation time: {time[-1]:.2f} s")

print(
    f"\nMaximum final panel temperature: "
    f"{np.max(T_steady_state[:-1]) - 273.15:.4f} °C"
)

print(
    f"Minimum final panel temperature: "
    f"{np.min(T_steady_state[:-1]) - 273.15:.4f} °C"
)

print(f"Final EBOX temperature: {T_steady_state[-1] - 273.15:.4f} °C")


# -------------------------------------------------------------------------
# PLOTS
# -------------------------------------------------------------------------

plot_section_4(
    x,
    temperature_profiles,
    ebox_temperature_history,
    time,
)