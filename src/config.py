"""Shared constants and paths for the whole project.

i kept everything that more than one script needs in here so the numbers
dont drift apart between files. if you want to study another site or change
the turbine, this is the one place to edit.
"""

import os

# --- paths -------------------------------------------------------------
HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data", "la_guajira_wind.csv")
RESULTS = os.path.join(HERE, "..", "results")

# --- site --------------------------------------------------------------
# La Guajira wind resource area, near the Jepirachi / Guajira I wind farms.
LATITUDE = 12.0
LONGITUDE = -71.6
START_DATE = "2021-01-01"
END_DATE = "2023-12-31"

# measurement heights of the two wind speed columns (Open Meteo levels)
HEIGHT_LOW = 10.0    # m, wind_speed_10m_ms
HEIGHT_HIGH = 100.0  # m, wind_speed_100m_ms
HUB_HEIGHT = 100.0   # m, where the turbine sits in this study

# --- air ---------------------------------------------------------------
# fallback density for a warm coastal climate. analysis.py can also work it
# out from the measured temperature, which is a bit more honest.
AIR_DENSITY = 1.18   # kg/m^3
R_SPECIFIC = 287.05  # J/(kg K), dry air gas constant
SEA_LEVEL_PRESSURE = 101325.0  # Pa, no station pressure in the dataset

# --- plotting ----------------------------------------------------------
ACCENT = "#1f4e79"
HIGHLIGHT = "#c0392b"


def air_density_from_temp(temp_c, pressure_pa=SEA_LEVEL_PRESSURE):
    """Air density [kg/m^3] from temperature [C], ideal gas law, sea level pressure."""
    import numpy as np

    t_kelvin = np.asarray(temp_c, dtype=float) + 273.15
    return pressure_pa / (R_SPECIFIC * t_kelvin)
