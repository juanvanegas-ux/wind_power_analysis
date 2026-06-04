import numpy as np
import pandas as pd

from small_wind import (load_cp, cp_peak, swept_area, power_mppt, power_fixed,
                        rated_power_of, net_loss_factor, annual_energy_mwh,
                        hub_wind, HOURS_PER_YEAR, CUT_IN, CUT_OUT, V_RATED,
                        TURBINES)


def test_cp_peak_matches_known_design():
    cp = load_cp()
    lam = cp["lambda"].values
    cmax_s, lopt_s = cp_peak(cp, lam, "Cp_Smart")
    cmax_c, lopt_c = cp_peak(cp, lam, "Cp_Comercial")
    # from the BEM study: smart peaks higher and at a higher lambda
    assert 0.46 < cmax_s < 0.49
    assert 0.45 < cmax_c < 0.48
    assert lopt_s > lopt_c


def test_swept_area():
    assert np.isclose(swept_area(2.5), np.pi * 2.5 ** 2)


def test_mppt_zero_outside_band_and_capped():
    cp = load_cp()
    cmax, _ = cp_peak(cp, cp["lambda"].values, "Cp_Smart")
    rated = rated_power_of(2.5, cmax)
    V = np.array([CUT_IN - 0.5, 8.0, V_RATED, CUT_OUT + 1.0])
    p = power_mppt(V, 2.5, cmax, rated)
    assert p[0] == 0.0           # below cut in
    assert p[-1] == 0.0          # above cut out
    assert p.max() <= rated + 1e-6


def test_mppt_is_upper_bound_on_fixed_speed():
    # Cp at any fixed speed is <= Cp_max, so MPPT must dominate everywhere
    cp = load_cp()
    lam = cp["lambda"].values
    for name, t in TURBINES.items():
        cmax, _ = cp_peak(cp, lam, t["col"])
        rated = rated_power_of(t["R"], cmax)
        V = np.linspace(0, CUT_OUT, 120)
        p_vs = power_mppt(V, t["R"], cmax, rated)
        p_fs = power_fixed(V, t["R"], cp, lam, t["col"], rated)
        assert np.all(p_fs <= p_vs + 1e-6)


def test_rated_power_positive_and_ordered():
    cp = load_cp()
    lam = cp["lambda"].values
    cmax_s, _ = cp_peak(cp, lam, "Cp_Smart")
    cmax_c, _ = cp_peak(cp, lam, "Cp_Comercial")
    rs = rated_power_of(2.5, cmax_s)
    rc = rated_power_of(2.275, cmax_c)
    assert rs > 0 and rc > 0
    assert rs > rc   # bigger rotor, more rated power


def test_rated_power_scales_with_cube_of_rated_speed():
    cp = load_cp()
    cmax, _ = cp_peak(cp, cp["lambda"].values, "Cp_Smart")
    r10 = rated_power_of(2.5, cmax, v_rated=10.0)
    r20 = rated_power_of(2.5, cmax, v_rated=20.0)
    # double the rated wind speed -> 8x the rated power (P ~ v^3)
    assert np.isclose(r20 / r10, 8.0)


def test_net_loss_factor_in_range():
    loss = net_loss_factor()
    assert 0.0 < loss < 0.2          # a sane single turbine loss stack
    # multiplicative, so strictly less than the naive sum
    assert loss < (0.04 + 0.02 + 0.02)


def test_annual_energy_matches_mean_power():
    # constant 1 kW for the whole record -> 1 kW * 8760 h = 8.76 MWh
    p = np.full(500, 1000.0)
    assert np.isclose(annual_energy_mwh(p), 1000.0 / 1e6 * HOURS_PER_YEAR)


def _fake_wind(n=400):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "wind_speed_10m_ms": rng.uniform(3, 10, n),
        "wind_speed_100m_ms": rng.uniform(5, 14, n),
    })


def test_hub_wind_rises_with_height():
    df = _fake_wind()
    # with positive shear, a taller hub sees more wind on average
    assert hub_wind(df, height=30).mean() > hub_wind(df, height=15).mean()
