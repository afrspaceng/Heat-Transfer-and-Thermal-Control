"""
section_6.py

Transient thermal simulation of the sandwich panel and EBOX
using a proportional-integral (PI) heater controller.
"""

from pathlib import Path
from postprocessing.plotting import plot_section_6

import numpy as np

from config.common import T_B1, T_B2, HEATER_DISSIPATION

from config.section_6 import (
    SECTION_6_NUMERICAL_PARTITION_PARAMETER,
    SECTION_6_TOTAL_TIME,
    SECTION_6_STEFAN_BOLTZMANN_CONSTANT,
    SECTION_6_EMISSIVITY,
    SECTION_6_RADIATION_TEMPERATURE,
    SECTION_6_DESIRED_TEMPERATURE,
    SECTION_6_PROPORTIONAL_GAIN,
    SECTION_6_INTEGRAL_GAIN,
    SECTION_6_INTEGRAL_START_FRACTION,
)

from models.geometry import (
    L_X_PANEL,
    L_Y_PANEL,
    PANEL_THICKNESS,
    HONEYCOMB_THICKNESS,
    CFRP_THICKNESS,
    L_X_EBOX,
    L_Y_EBOX,
    L_X_HEATER,
    L_Y_HEATER,
)

from models.materials import (
    RHO_HONEYCOMB,
    CP_HONEYCOMB,
    RHO_CFRP,
    CP_CFRP,
)

from models.thermal_properties import (
    calculate_effective_conductivity,
    EBOX_PANEL_CONDUCTANCE,
    EBOX_THERMAL_CAPACITY,
)


def run_section_6():

    # -------------------------------------------------------------------------
    # 1. Initial temperature distribution
    # -------------------------------------------------------------------------

    results_directory = Path("results")

    section_4_file = results_directory / "section_4_steady_state.npz"

    with np.load(section_4_file) as section_4_results:
        T = section_4_results["temperature"].copy()


    # -------------------------------------------------------------------------
    # 2. Spatial discretization
    # -------------------------------------------------------------------------

    n_numerical = SECTION_6_NUMERICAL_PARTITION_PARAMETER

    i_numerical = 2 * n_numerical - 1

    number_of_nodes = 7 + 6 * i_numerical

    x = np.linspace(0.0, L_X_PANEL, number_of_nodes)

    dx = L_X_PANEL / (number_of_nodes - 1)


    # -------------------------------------------------------------------------
    # 3. Thermal properties
    # -------------------------------------------------------------------------

    k_effective = calculate_effective_conductivity()

    effective_density = (
        2 * RHO_CFRP * CFRP_THICKNESS
        + RHO_HONEYCOMB * HONEYCOMB_THICKNESS
    ) / PANEL_THICKNESS

    effective_specific_heat = (
        2 * RHO_CFRP * CP_CFRP * L_Y_PANEL * L_X_PANEL * CFRP_THICKNESS
        + RHO_HONEYCOMB * CP_HONEYCOMB * L_Y_PANEL * L_X_PANEL * HONEYCOMB_THICKNESS
        + EBOX_THERMAL_CAPACITY
    ) / (
        PANEL_THICKNESS
        * L_X_PANEL
        * L_Y_PANEL
        * effective_density
    )

    thermal_diffusivity = k_effective / (
        effective_density * effective_specific_heat
    )

    cell_volume = dx * L_Y_PANEL * PANEL_THICKNESS


    # -------------------------------------------------------------------------
    # 4. Temporal discretization
    # -------------------------------------------------------------------------

    # Preserve the time-step definition used in the MATLAB implementation.
    dt = dx / 2

    number_of_time_steps = int(
        np.floor(SECTION_6_TOTAL_TIME / dt)
    )

    time = np.arange(number_of_time_steps + 1) * dt


    # -------------------------------------------------------------------------
    # 5. EBOX and heater node locations
    # -------------------------------------------------------------------------

    # MATLAB node numbers are converted to Python zero-based indices.
    ebox_start_node = int(np.floor(0.2 / dx + 1)) - 1
    ebox_end_node = number_of_nodes - ebox_start_node - 1

    heater_start_node = int(np.floor(0.25 / dx + 1)) - 1
    heater_end_node = number_of_nodes - heater_start_node - 1


    # -------------------------------------------------------------------------
    # 6. Initial conditions and histories
    # -------------------------------------------------------------------------

    # Enforce the panel boundary temperatures.
    T[0] = T_B1
    T[number_of_nodes - 1] = T_B2

    # The last temperature entry corresponds to the EBOX.
    ebox_temperature_history = np.zeros(number_of_time_steps + 1)
    ebox_temperature_history[0] = T[-1]

    # Controller histories to use them in order to be controlled.
    error_history = np.zeros(number_of_time_steps)
    proportional_control_history = np.zeros(number_of_time_steps)
    integral_control_history = np.zeros(number_of_time_steps)
    total_control_history = np.zeros(number_of_time_steps)

    heater_power_history = np.zeros(number_of_time_steps)


    # -------------------------------------------------------------------------
    # 7. Controller initialization
    # -------------------------------------------------------------------------

    accumulated_error = 0.0

    integral_start_step = int(
        number_of_time_steps * SECTION_6_INTEGRAL_START_FRACTION
    )

    # -------------------------------------------------------------------------
    # 8. Transient thermal simulation
    # -------------------------------------------------------------------------

    for step in range(number_of_time_steps):

        # ---------------------------------------------------------------------
        # 8.1. PI controller
        # ---------------------------------------------------------------------

        T_ebox = T[-1]

        error = T_ebox - SECTION_6_DESIRED_TEMPERATURE

        error_history[step] = error

        proportional_control = (
            SECTION_6_PROPORTIONAL_GAIN * abs(error)
        )

        # Start integrating the error after the first 10% of the simulation.
        if step + 1 > integral_start_step:
            accumulated_error += error

        integral_control = (
            SECTION_6_INTEGRAL_GAIN * accumulated_error
        )

        total_control = proportional_control + integral_control

        proportional_control_history[step] = proportional_control
        integral_control_history[step] = integral_control
        total_control_history[step] = total_control

        # The heater operates only when the EBOX is below the desired temperature.
        if error < 0:
            z_2 = total_control
        else:
            z_2 = 0.0


        # ---------------------------------------------------------------------
        # 8.2. Initialize heat transfer contributions
        # ---------------------------------------------------------------------

        conduction_panel = np.zeros(number_of_nodes)
        conduction_ebox = np.zeros(number_of_nodes)
        radiation_panel = np.zeros(number_of_nodes)
        heater_to_panel = np.zeros(number_of_nodes)

        heat_ebox_internal = 0.0
        panel_to_ebox = 0.0


        # ---------------------------------------------------------------------
        # 8.3. Heat transfer along the panel
        # ---------------------------------------------------------------------

        for node in range(1, number_of_nodes - 1):

            conduction_panel[node] = (
                k_effective / dx**2
                * cell_volume
                * (
                    T[node - 1]
                    + T[node + 1]
                    - 2 * T[node]
                )
            )


        # ---------------------------------------------------------------------
        # 8.4. Heat transfer between the EBOX and the panel
        # ---------------------------------------------------------------------

        for node in range(1, number_of_nodes - 1):

            if ebox_start_node < node < ebox_end_node:

                contact_factor = 1.0

            elif node == ebox_start_node or node == ebox_end_node:

                contact_factor = 0.5

            else:

                contact_factor = 0.0

            conduction_ebox[node] = (
                contact_factor
                * EBOX_PANEL_CONDUCTANCE
                * L_Y_EBOX
                * dx
                * (T[-1] - T[node])
            )


        # ---------------------------------------------------------------------
        # 8.5. Radiation from the panel to the environment
        # ---------------------------------------------------------------------

        sigma = SECTION_6_STEFAN_BOLTZMANN_CONSTANT
        emissivity = SECTION_6_EMISSIVITY
        radiation_temperature = SECTION_6_RADIATION_TEMPERATURE

        for node in range(1, number_of_nodes - 1):

            if node < ebox_start_node:

                radiation_factor = 2.0

            elif node == ebox_start_node or node == ebox_end_node:

                radiation_factor = 1.5

            elif ebox_start_node < node < ebox_end_node:

                radiation_factor = 1.0

            else:

                radiation_factor = 2.0

            radiation_panel[node] = (
                radiation_factor
                * sigma
                * emissivity
                * dx
                * L_Y_PANEL
                * (
                    radiation_temperature**4
                    - T[node]**4
                )
            )


        # ---------------------------------------------------------------------
        # 8.6. Heat transfer from the heater to the panel
        # ---------------------------------------------------------------------

        for node in range(1, number_of_nodes - 1):

            if node == heater_start_node or node == heater_end_node:

                heater_factor = 0.5

            elif heater_start_node < node < heater_end_node:

                heater_factor = 1.0

            else:

                heater_factor = 0.0

            heater_to_panel[node] = (
                z_2
                * heater_factor
                * HEATER_DISSIPATION
                / (
                    L_Y_HEATER
                    * PANEL_THICKNESS
                    * L_X_HEATER
                )
                * dx
                * PANEL_THICKNESS
                * L_Y_PANEL
            )


        # ---------------------------------------------------------------------
        # 8.7. Internal heat generation in the EBOX
        # ---------------------------------------------------------------------

        heat_ebox_internal = 0.0


        # ---------------------------------------------------------------------
        # 8.8. Heat transfer from the panel to the EBOX
        # ---------------------------------------------------------------------

        panel_to_ebox = -np.sum(conduction_ebox)


        # ---------------------------------------------------------------------
        # 8.9. Radiation from the EBOX to the environment
        # ---------------------------------------------------------------------

        radiation_ebox = (
            sigma
            * emissivity
            * L_X_EBOX
            * L_Y_EBOX
            * (
                radiation_temperature**4
                - T[-1]**4
            )
        )


        # ---------------------------------------------------------------------
        # 8.10. Explicit Euler temperature update
        # ---------------------------------------------------------------------

        T_new = T.copy()

        # Update panel temperatures.
        T_new[1:-2] = T[1:-2] + (
            dt / (effective_specific_heat * effective_density * cell_volume)
            * (
                conduction_panel[1:-1]
                + conduction_ebox[1:-1]
                + radiation_panel[1:-1]
                + heater_to_panel[1:-1]
            )
        )

        # Update EBOX temperature.
        T_new[-1] = T[-1] + (
            dt / EBOX_THERMAL_CAPACITY
        ) * (
            heat_ebox_internal
            + panel_to_ebox
            + radiation_ebox
        )

        T = T_new

        # Store EBOX temperature at the new time instant.
        ebox_temperature_history[step + 1] = T[-1]

        # Store total heater power distributed over the panel.
        heater_power_history[step] = np.sum(heater_to_panel)

    # -------------------------------------------------------------------------
    # 9. Save simulation results
    # -------------------------------------------------------------------------

    T_final = T.copy()

    np.savez(
        results_directory / "section_6_final_state.npz",
        temperature=T_final,
        x=x,
        time=time,
        ebox_temperature_history=ebox_temperature_history,
        heater_power_history=heater_power_history,
        final_time=time[-1],
    )

    # -------------------------------------------------------------------------
    # 10. Print simulation results
    # -------------------------------------------------------------------------

    print("\n========== Section 6 results ==========")
    print(f"Number of panel nodes: {number_of_nodes}")
    print(f"Total number of unknowns: {len(T)}")
    print(f"Spatial step: {dx:.8f} m")
    print(f"Time step: {dt:.8f} s")
    print(f"Number of time steps: {number_of_time_steps}")
    print(f"Total simulation time: {time[-1]:.2f} s")
    print(f"Final EBOX temperature: {T[-1] - 273.15:.4f} °C")
    print(f"Maximum panel temperature: {np.max(T[:-1]) - 273.15:.4f} °C")
    print(f"Minimum panel temperature: {np.min(T[:-1]) - 273.15:.4f} °C")

    plot_section_6(time, ebox_temperature_history)

if __name__ == "__main__":
    run_section_6()