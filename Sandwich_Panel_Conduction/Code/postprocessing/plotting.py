
"""
plotting.py

Plotting functions for thermal analysis results.
"""

import matplotlib.pyplot as plt


# ============================================================
# PLOT CONFIGURATION
# ============================================================

plt.rcParams["text.usetex"] = False
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"


def plot_temperature_distribution(
    x_analytical,
    T_analytical,
    numerical_results
):
    """
    Plot analytical and numerical temperature distributions.

    INPUTS:
        x_analytical (array): Analytical spatial coordinates [m].
        T_analytical (array): Analytical temperatures [K].
        numerical_results (list): Numerical solutions for
                                  different discretizations.
    """

    # Create figure
    plt.figure(figsize=(10, 6))

    # Analytical solution
    plt.plot(
        x_analytical * 1000,
        T_analytical - 273.15,
        label=r"$\mathrm{Analytical\ solution}$",
        linewidth=2
    )

    # Numerical solutions
    for x, T, N in numerical_results:

        plt.plot(
            x * 1000,
            T - 273.15,
            marker="o",
            linestyle="--",
            label=rf"$\mathrm{{Numerical\ solution}}\;(N={N})$"
        )

    # Axis labels
    plt.xlabel(
        r"$x\;[\mathrm{mm}]$",
        fontsize=12
    )

    plt.ylabel(
        r"$T\;[^\circ\mathrm{C}]$",
        fontsize=12
    )

    # Title
    plt.title(
        r"$\mathrm{Temperature\ Distribution}$",
        fontsize=14
    )

    # Format axis tick labels using mathematical notation
    plt.gca().xaxis.set_major_formatter(
        plt.FormatStrFormatter(r"$%g$")
    )

    plt.gca().yaxis.set_major_formatter(
        plt.FormatStrFormatter(r"$%g$")
    )

    # Legend
    plt.legend()

    # Grid
    plt.grid(True)

    # Display plot
    plt.tight_layout()
    plt.show()


def plot_section_2_half_panel(
    x_analytical,
    T_analytical,
    x_numerical,
    T_numerical,
    x_b,
    x_c,
    x_d
):
    """
    Plot analytical and numerical temperature distributions
    along half of the sandwich panel.
    """

    plt.figure(figsize=(10, 6))

    plt.plot(
        x_analytical * 1000,
        T_analytical - 273.15,
        label=r"$\mathrm{Analytical\ solution}$",
        linewidth=2
    )

    plt.plot(
        x_numerical * 1000,
        T_numerical - 273.15,
        label=r"$\mathrm{Numerical\ solution}$",
        linewidth=1.5,
        linestyle="--"
    )

        # Region boundaries
    plt.axvline(
        x_b * 1000,
        linestyle=":",
        linewidth=1,
        label=r"$X_B$"
    )

    plt.axvline(
        x_c * 1000,
        linestyle=":",
        linewidth=1,
        label=r"$X_C$"
    )

    plt.axvline(
        x_d * 1000,
        linestyle=":",
        linewidth=1,
        label=r"$X_D$"
    )

    plt.xlabel(r"$x\;[\mathrm{mm}]$", fontsize=12)
    plt.ylabel(r"$T\;[^\circ\mathrm{C}]$", fontsize=12)

    plt.title(
        r"$\mathrm{Temperature\ Distribution\ -\ Half\ Panel}$",
        fontsize=14
    )

    plt.gca().xaxis.set_major_formatter(
        plt.FormatStrFormatter(r"$%g$")
    )

    plt.gca().yaxis.set_major_formatter(
        plt.FormatStrFormatter(r"$%g$")
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()


def plot_section_2_full_panel(
    x_analytical,
    T_analytical,
    x_numerical,
    T_numerical,
    x_b,
    x_c,
    x_d,
    panel_length
):
    """
    Plot analytical and numerical temperature distributions
    along the full sandwich panel.
    """

    plt.figure(figsize=(10, 6))

    plt.plot(
        x_analytical * 1000,
        T_analytical - 273.15,
        label=r"$\mathrm{Analytical\ solution}$",
        linewidth=2
    )

    plt.plot(
        x_numerical * 1000,
        T_numerical - 273.15,
        label=r"$\mathrm{Numerical\ solution}$",
        linewidth=1.5,
        linestyle="--"
    )

           # Region boundaries
    boundaries = [
        (x_b, r"$X_B$"),
        (x_c, r"$X_C$"),
        (x_d, r"$X_D$"),
        (panel_length - x_d, r"$L_x-X_D$"),
        (panel_length - x_c, r"$L_x-X_C$"),
        (panel_length - x_b, r"$L_x-X_B$")
    ]

    for boundary, label in boundaries:
        plt.axvline(
            boundary * 1000,
            linestyle=":",
            linewidth=1,
            label=label
        )

    plt.xlabel(r"$x\;[\mathrm{mm}]$", fontsize=12)
    plt.ylabel(r"$T\;[^\circ\mathrm{C}]$", fontsize=12)

    plt.title(
        r"$\mathrm{Temperature\ Distribution\ -\ Full\ Panel}$",
        fontsize=14
    )

    plt.gca().xaxis.set_major_formatter(
        plt.FormatStrFormatter(r"$%g$")
    )

    plt.gca().yaxis.set_major_formatter(
        plt.FormatStrFormatter(r"$%g$")
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.show()

def plot_section_3(x, T_panel, T_ebox):
    """
    Plot the steady-state panel temperature profile and EBOX temperature.
    """

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 13,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 9,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "text.latex.preamble": r"\usepackage{amsmath}",
    })

    plt.figure(figsize=(9, 5))

    plt.plot(
        x * 1000,
        T_panel - 273.15,
        "k",
        linewidth=0.8,
        label=r"$\mathrm{Steady\text{-}state\ panel\ temperature}$",
    )

    plt.axhline(
        T_ebox - 273.15,
        color="red",
        linewidth=0.8,
        label=r"$\mathrm{EBOX\ temperature}$",
    )

    plt.xlabel(r"$x\ [\mathrm{mm}]$")
    plt.ylabel(r"$T\ [^\circ\mathrm{C}]$")

    plt.title(r"$\mathrm{Steady\text{-}state\ thermal\ profile}$")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.show()



def plot_section_4(x, temperature_profiles, ebox_temperature_history, time):
    """
    Plot the transient panel temperature profiles and EBOX temperature history.
    """

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 13,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 9,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })

    # ---------------------------------------------------------------------
    # FIGURE 1: PANEL TEMPERATURE PROFILES
    # ---------------------------------------------------------------------

    plt.figure(figsize=(9, 5))

    profile_styles = {
        10000: "--",
        200000: "-.",
        500000: ":",
        len(time): "-",
    }

    for row, temperatures in temperature_profiles.items():

        if row == len(time):
            label = r"$t = 8000\ \mathrm{s}$"
        else:
            label = rf"$t = {((row - 1) * (time[1] - time[0])):.2f}\ \mathrm{{s}}$"

        plt.plot(
            x * 1000,
            temperatures - 273.15,
            color="black",
            linestyle=profile_styles[row],
            linewidth=0.8,
            label=label,
        )

    plt.xlabel(r"$x\ [\mathrm{mm}]$")
    plt.ylabel(r"$T\ [^\circ\mathrm{C}]$")
    plt.title(r"$\mathrm{Transient\ panel\ temperature\ profiles}$")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # ---------------------------------------------------------------------
    # FIGURE 2: EBOX TEMPERATURE HISTORY
    # ---------------------------------------------------------------------

    plt.figure(figsize=(9, 5))

    plt.plot(
        time,
        ebox_temperature_history - 273.15,
        color="black",
        linewidth=0.8,
    )

    plt.xlabel(r"$t\ [\mathrm{s}]$")
    plt.ylabel(r"$T_{\mathrm{EBOX}}\ [^\circ\mathrm{C}]$")
    plt.title(r"$\mathrm{EBOX\ temperature\ evolution}$")

    plt.grid(True)
    plt.tight_layout()

    plt.show()


def plot_section_5(time, ebox_temperature_history, heater_power_history):
    """
    Plot the EBOX temperature and heater power histories.
    """

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 13,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 9,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })

    # ---------------------------------------------------------------------
    # FIGURE 1: EBOX TEMPERATURE AND THERMOSTAT THRESHOLDS
    # ---------------------------------------------------------------------

    plt.figure(figsize=(9, 5))

    plt.plot(
        time,
        ebox_temperature_history - 273.15,
        color="black",
        linewidth=0.8,
        label=r"$T_{\mathrm{EBOX}}$",
    )

    plt.axhline(
        5,
        color="red",
        linewidth=0.8,
        linestyle="--",
        label=r"$T_{\mathrm{ON}} = 5\,^\circ\mathrm{C}$",
    )

    plt.axhline(
        8.5,
        color="red",
        linewidth=0.8,
        linestyle="--",
        label=r"$T_{\mathrm{OFF}} = 8.5\,^\circ\mathrm{C}$",
    )

    plt.xlabel(r"$t\ [\mathrm{s}]$")
    plt.ylabel(r"$T\ [^\circ\mathrm{C}]$")

    plt.title(r"$\mathrm{EBOX\ temperature\ and\ thermostat\ thresholds}$")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # ---------------------------------------------------------------------
    # FIGURE 2: EBOX TEMPERATURE AND HEATER POWER
    # ---------------------------------------------------------------------

    fig, ax1 = plt.subplots(figsize=(9, 5))

    line1, = ax1.plot(
        time,
        ebox_temperature_history - 273.15,
        color="black",
        linewidth=0.8,
        label=r"$T_{\mathrm{EBOX}}$",
    )

    ax1.set_xlabel(r"$t\ [\mathrm{s}]$")
    ax1.set_ylabel(r"$T\ [^\circ\mathrm{C}]$")

    ax2 = ax1.twinx()

    line2, = ax2.plot(
        time,
        heater_power_history,
        color="red",
        linewidth=0.75,
        label=r"$W_{\mathrm{HT}}$",
    )

    ax2.set_ylabel(r"$W_{\mathrm{HT}}\ [\mathrm{W}]$", color="red")
    ax2.set_ylim(0, 35)

    ax2.spines["right"].set_color("red")
    ax2.tick_params(axis="y", colors="red")

    ax1.tick_params(axis="y", labelcolor="black")
    ax2.tick_params(axis="y", labelcolor="red")

    ax1.legend(
        handles=[line1, line2],
        labels=[
            r"$\mathrm{EBOX\ temperature}$",
            r"$\mathrm{Heater\ power}$",
        ],
    )

    ax1.set_title(r"$\mathrm{EBOX\ temperature\ and\ heater\ operation}$")

    ax1.grid(True)
    fig.tight_layout()

    plt.show()


def plot_section_6(time, ebox_temperature_history):
    """
    Plot the EBOX temperature history and desired temperature.
    """

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 13,
    })

    plt.figure(4)

    plt.plot(
        time,
        ebox_temperature_history - 273.15,
        "-",
        color="black",
        label=r"$\mathrm{Temperatura\ de\ la\ EBOX}$"
    )

    plt.axhline(
        7.5,
        color="red",
        linewidth=0.8,
        linestyle="--"
    )

    plt.xlabel(r"$t$ [s]")
    plt.ylabel(r"$T$ [$^o$C]")

    plt.legend(
        [
            r"$\mathrm{Temperatura\ de\ la\ EBOX}$",
            r"$\mathrm{Temperatura\ deseada}$"
        ],
        fontsize=9
    )

    plt.grid(True)

    plt.tick_params(axis="both", labelsize=13)

    plt.tight_layout()
    plt.show()
