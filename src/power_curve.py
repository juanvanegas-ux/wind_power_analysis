"""A simple, generic wind-turbine power curve used to convert wind speed into
electrical power for the resource assessment and the forecasting target.

Parameters loosely match a multi-MW onshore turbine; the shape (cut-in, cubic
ramp, rated plateau, cut-out) is what matters for the analysis.
"""

import numpy as np

CUT_IN = 3.0       # m/s
RATED = 12.0       # m/s
CUT_OUT = 25.0     # m/s
RATED_POWER = 2.0  # MW


def power_output(wind_speed):
    """Electrical power [MW] for a given wind speed [m/s] (array-safe)."""
    v = np.asarray(wind_speed, dtype=float)
    p = np.zeros_like(v)

    ramp = (v >= CUT_IN) & (v < RATED)
    p[ramp] = RATED_POWER * (v[ramp] ** 3 - CUT_IN**3) / (RATED**3 - CUT_IN**3)

    rated = (v >= RATED) & (v <= CUT_OUT)
    p[rated] = RATED_POWER

    return p
