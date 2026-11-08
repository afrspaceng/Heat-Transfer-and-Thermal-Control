"""
section_5.py

Transient thermal simulation of a sandwich panel coupled to an EBOX.
The heater is controlled by a thermostat with hysteresis.
"""

import numpy as np

from pathlib import Path

from config.common import (
    HEATER_DISSIPATION,
    T_B1,
    T_B2,
)

from config.section_5 import (
    SECTION_5_NUMERICAL_PARTITION_PARAMETER,
    SECTION_5_TOTAL_TIME,
    SECTION_5_STEFAN_BOLTZMANN_CONSTANT,
    SECTION_5_EMISSIVITY,
    SECTION_5_RADIATION_TEMPERATURE,
    SECTION_5_HEATER_SWITCH_ON_TEMPERATURE,
    SECTION_5_HEATER_SWITCH_OFF_TEMPERATURE,
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

from postprocessing.plotting import plot_section_5


# -------------------------------------------------------------------------
# INITIAL CONDITION: SECTION 4 FINAL STATE
# -------------------------------------------------------------------------

results_directory = Path("results")

section_4_file = results_directory / "section_4_steady_state.npz"

with np.load(section_4_file) as section_4_results:
    T = section_4_results["temperature"].copy()


# -------------------------------------------------------------------------
# THERMAL PROPERTIES
# -------------------------------------------------------------------------

k_eff = calculate_effective_conductivity()

rho_effective = (
    2 * RHO_CFRP * CFRP_THICKNESS
    + RHO_HONEYCOMB * HONEYCOMB_THICKNESS
) / PANEL_THICKNESS

c_effective = (
    2 * RHO_CFRP * CP_CFRP
    * L_Y_PANEL * L_X_PANEL * CFRP_THICKNESS
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

n = SECTION_5_NUMERICAL_PARTITION_PARAMETER

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


# -------------------------------------------------------------------------
# CONTACT WEIGHTS
# -------------------------------------------------------------------------

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

dt = dx**2 / (2 * thermal_diffusivity)

number_of_time_steps = int(np.floor(SECTION_5_TOTAL_TIME / dt))

time = np.arange(number_of_time_steps + 1) * dt


# -------------------------------------------------------------------------
# INITIAL TEMPERATURE FIELD
# -------------------------------------------------------------------------

# The initial condition is the final state obtained in Section 4.
# The last element corresponds to the EBOX temperature.

T[0] = T_B1
T[number_of_nodes - 1] = T_B2


# -------------------------------------------------------------------------
# STORAGE OF RESULTS
# -------------------------------------------------------------------------

ebox_temperature_history = np.zeros(number_of_time_steps + 1)

heater_power_history = np.zeros(number_of_time_steps + 1)

ebox_temperature_history[0] = T[-1]


# MATLAB column 25 corresponds to Python index 24.
heater_monitoring_node = 24


# -------------------------------------------------------------------------
# HEATER POWER DISTRIBUTION
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
# THERMOSTAT INITIAL STATE
# -------------------------------------------------------------------------

# 0: heater OFF
# 1: heater ON

heater_on = False


# -------------------------------------------------------------------------
# TRANSIENT SIMULATION: EXPLICIT EULER
# -------------------------------------------------------------------------

for step in range(number_of_time_steps):

    # Current temperatures.
    T_panel = T[:number_of_nodes]
    T_ebox = T[-1]

    # ---------------------------------------------------------------------
    # THERMOSTAT CONTROL WITH HYSTERESIS
    # ---------------------------------------------------------------------

    if T_ebox <= SECTION_5_HEATER_SWITCH_ON_TEMPERATURE:
        heater_on = True

    elif T_ebox >= SECTION_5_HEATER_SWITCH_OFF_TEMPERATURE:
        heater_on = False

    # Otherwise, retain the previous heater state.

    # ---------------------------------------------------------------------
    # PANEL CONDUCTION
    # ---------------------------------------------------------------------

    panel_heat_flow = np.zeros(number_of_nodes)

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
        SECTION_5_STEFAN_BOLTZMANN_CONSTANT
        * SECTION_5_EMISSIVITY
        * dx
        * L_Y_PANEL
        * radiation_weights
        * (
            SECTION_5_RADIATION_TEMPERATURE**4
            - T_panel**4
        )
    )

    panel_heat_flow += panel_radiation

    # ---------------------------------------------------------------------
    # HEATER-TO-PANEL HEAT TRANSFER
    # ---------------------------------------------------------------------

    if heater_on:
        heater_to_panel = heater_power_per_node
    else:
        heater_to_panel = np.zeros(number_of_nodes)

    panel_heat_flow += heater_to_panel

    # ---------------------------------------------------------------------
    # INTERNAL EBOX HEAT GENERATION
    # ---------------------------------------------------------------------

    internal_ebox_heat = 0.0

    # ---------------------------------------------------------------------
    # PANEL-TO-EBOX CONDUCTION
    # ---------------------------------------------------------------------

    panel_to_ebox = -np.sum(ebox_to_panel)

    # ---------------------------------------------------------------------
    # RADIATION TO THE EBOX
    # ---------------------------------------------------------------------

    ebox_radiation = (
        SECTION_5_STEFAN_BOLTZMANN_CONSTANT
        * SECTION_5_EMISSIVITY
        * L_X_EBOX
        * L_Y_EBOX
        * (
            SECTION_5_RADIATION_TEMPERATURE**4
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

    # ---------------------------------------------------------------------
    # STORE TIME HISTORIES
    # ---------------------------------------------------------------------

    ebox_temperature_history[step + 1] = T[-1]
    # Store heater power at the current time step.
    if heater_on:
        heater_power_history[step] = heater_power_per_node[
            heater_monitoring_node
        ]


# -------------------------------------------------------------------------
# FINAL STATE
# -------------------------------------------------------------------------

T_steady_state = T.copy()

# Preserve the last heater power value at the final time instant...
heater_power_history[-1] = heater_power_history[-2]

np.savez(
    results_directory / "section_5_final_state.npz",
    temperature=T_steady_state,
    x=x,
    time=time[-1],
    heater_on=heater_on,
)


# -------------------------------------------------------------------------
# RESULTS
# -------------------------------------------------------------------------

print("\n========== SECTION 5: TRANSIENT SOLUTION ==========")

print(f"Number of panel nodes: {number_of_nodes}")
print(f"Total number of unknowns: {number_of_nodes + 1}")
print(f"Spatial step: {dx:.8f} m")
print(f"Thermal diffusivity: {thermal_diffusivity:.6e} m^2/s")
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

print(f"Final heater state: {'ON' if heater_on else 'OFF'}")


# -------------------------------------------------------------------------
# PLOTS
# -------------------------------------------------------------------------

plot_section_5(
    time,
    ebox_temperature_history,
    heater_power_history,
)