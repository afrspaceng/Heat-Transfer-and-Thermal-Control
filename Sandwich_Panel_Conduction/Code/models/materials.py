
"""
materials.py

Thermal properties of the materials used
in the sandwich panel.
"""

# ============================================================
# HONEYCOMB CORE PROPERTIES
# ============================================================

RHO_HONEYCOMB = 40       # Density [kg/m^3]
CP_HONEYCOMB = 890       # Specific heat capacity [J/(kg K)]

K_X_HONEYCOMB = 0.7      # Thermal conductivity in X [W/(m K)]
K_Y_HONEYCOMB = 1.5      # Thermal conductivity in Y [W/(m K)]
K_Z_HONEYCOMB = 0.7      # Thermal conductivity in Z [W/(m K)]


# ============================================================
# CFRP PROPERTIES
# ============================================================

RHO_CFRP = 1650          # Density [kg/m^3]
CP_CFRP = 930            # Specific heat capacity [J/(kg K)]

K_X_CFRP = 130            # Thermal conductivity in X [W/(m K)]
K_Y_CFRP = 1              # Thermal conductivity in Y [W/(m K)]
K_Z_CFRP = 130            # Thermal conductivity in Z [W/(m K)]
