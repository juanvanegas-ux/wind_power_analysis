import numpy as np

import config
from wind_shear import shear_exponent, extrapolate
from energy_yield import total_loss_factor, power_curve, yield_for_turbine, TURBINES
from analysis import power_class


def test_air_density_decreases_with_heat():
    cold = config.air_density_from_temp(10.0)
    hot = config.air_density_from_temp(35.0)
    assert hot < cold
    # ballpark sane numbers for near sea level air
    assert 1.0 < hot < 1.3
    assert 1.0 < cold < 1.3


def test_shear_exponent_known_value():
    # if the wind doubles from 10 m to 100 m, alpha = ln(2)/ln(10)
    a = shear_exponent(5.0, 10.0, z_low=10.0, z_high=100.0)
    assert np.isclose(a, np.log(2) / np.log(10))


def test_shear_exponent_calm_is_nan():
    a = shear_exponent(0.1, 0.2)
    assert np.isnan(a)


def test_extrapolate_round_trip():
    # push 10 m up to 100 m with some alpha, then the ratio should match
    a = 0.2
    v100 = extrapolate(6.0, a, 10.0, 100.0)
    assert np.isclose(v100, 6.0 * 10.0 ** a)


def test_loss_factor_is_multiplicative():
    losses = {"a": 0.1, "b": 0.1}
    # not 0.2, it is 1 - 0.9*0.9 = 0.19
    assert np.isclose(total_loss_factor(losses), 1 - 0.81)


def test_loss_factor_in_range():
    assert 0.0 < total_loss_factor() < 0.3


def test_turbine_yield_positive_and_capped():
    v = np.array([3.0, 8.0, 12.0, 20.0, 30.0])
    for params in TURBINES.values():
        g_aep, n_aep, g_cf, n_cf = yield_for_turbine(v, params)
        assert g_aep > 0
        assert 0 < n_aep <= g_aep
        assert 0 < n_cf <= g_cf <= 1.0


def test_power_curve_caps_at_rated():
    p = power_curve(np.array([100.0]), 3.0, 12.0, 25.0, 2.0)
    assert p[0] == 0.0  # past cut out


def test_power_class_monotonic():
    assert power_class(50) <= power_class(250) <= power_class(700)
    assert power_class(700) == 7
