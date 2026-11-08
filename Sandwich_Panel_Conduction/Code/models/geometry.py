
"""
geometry.py

Geometrical definition of the sandwich panel, EBOX and heater.
"""

# ============================================================
# PANEL GEOMETRY
# ============================================================

# Panel dimensions [m]
L_X_PANEL = 600e-3
L_Y_PANEL = 400e-3

# Sandwich panel thicknesses [m]
HONEYCOMB_THICKNESS = 12e-3
CFRP_THICKNESS = 3e-3

# Total sandwich panel thickness [m]
PANEL_THICKNESS = (
    2 * CFRP_THICKNESS
    + HONEYCOMB_THICKNESS
)


# ============================================================
# EBOX GEOMETRY
# ============================================================

# EBOX dimensions [m]
L_X_EBOX = 200e-3
L_Y_EBOX = 200e-3

# EBOX contact area [m^2]
EBOX_AREA = L_X_EBOX * L_Y_EBOX


# ============================================================
# HEATER GEOMETRY
# ============================================================

# Heater dimensions [m]
L_X_HEATER = 100e-3
L_Y_HEATER = 100e-3
