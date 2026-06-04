import numpy as np

from power_curve import power_output, CUT_IN, RATED, CUT_OUT, RATED_POWER


def test_below_cut_in_is_zero():
    assert power_output(CUT_IN - 0.1) == 0.0
    assert power_output(0.0) == 0.0


def test_above_cut_out_is_zero():
    # the turbine shuts down in a storm
    assert power_output(CUT_OUT + 1.0) == 0.0


def test_rated_plateau():
    assert np.isclose(power_output(RATED), RATED_POWER)
    assert np.isclose(power_output(RATED + 5.0), RATED_POWER)
    # never makes more than rated
    speeds = np.linspace(0, CUT_OUT, 200)
    assert power_output(speeds).max() <= RATED_POWER + 1e-9


def test_ramp_is_monotonic():
    v = np.linspace(CUT_IN, RATED, 50)
    p = power_output(v)
    assert np.all(np.diff(p) >= -1e-12)


def test_array_safe():
    out = power_output([0.0, 5.0, 12.0, 30.0])
    assert out.shape == (4,)
