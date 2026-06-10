import numpy as np

import small_wind as sw
from loads import (ct_column, ct_at, thrust, thrust_mppt, thrust_fixed,
                   overturning_moment)


def test_ct_column_maps_cp_to_ct():
    assert ct_column("Cp_Smart") == "Ct_Smart"
    assert ct_column("Cp_Comercial") == "Ct_Comercial"


def test_thrust_scales_with_velocity_squared():
    # double the wind -> 4x the thrust (F ~ V^2)
    f1 = thrust(5.0, 2.5, 0.8)
    f2 = thrust(10.0, 2.5, 0.8)
    assert np.isclose(f2 / f1, 4.0)


def test_thrust_mppt_furls_and_zeros_outside_band():
    cp = sw.load_cp()
    lam = cp["lambda"].values
    _, lopt = sw.cp_peak(cp, lam, "Cp_Smart")
    ct_op = ct_at(cp, lam, "Ct_Smart", lopt)
    V = np.array([sw.CUT_IN - 0.5, 8.0, sw.V_RATED, sw.V_RATED + 4,
                  sw.CUT_OUT + 1])
    f = thrust_mppt(V, 2.5, ct_op)
    assert f[0] == 0.0                                   # below cut in
    assert f[-1] == 0.0                                  # above cut out
    f_rated = thrust(sw.V_RATED, 2.5, ct_op)
    assert np.all(f <= f_rated + 1e-6)                   # furl caps the load


def test_overturning_moment_grows_linearly_with_height():
    f = 1000.0
    assert np.isclose(overturning_moment(f, 30) / overturning_moment(f, 15),
                      2.0)


def test_smart_blade_has_lower_load_per_power():
    # the headline: the bend twist smart blade sheds thrust, so its Ct/Cp
    # (load per unit power) is below the comercial blade's
    cp = sw.load_cp()
    lam = cp["lambda"].values
    ratios = {}
    for name, t in sw.TURBINES.items():
        cmax, lopt = sw.cp_peak(cp, lam, t["col"])
        ct_op = ct_at(cp, lam, ct_column(t["col"]), lopt)
        ratios[name] = ct_op / cmax
    assert ratios["Smart blade"] < ratios["Comercial blade"]


def test_fixed_speed_thrust_is_finite_everywhere():
    cp = sw.load_cp()
    lam = cp["lambda"].values
    V = np.linspace(0.1, sw.CUT_OUT, 100)
    f = thrust_fixed(V, 2.275, cp, lam, "Ct_Comercial")
    assert np.all(np.isfinite(f))