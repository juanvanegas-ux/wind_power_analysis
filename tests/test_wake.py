import numpy as np

import energy_yield
import wake


def test_no_deficit_upstream_or_beside():
    # a turbine upstream (x_down <= 0) causes no deficit
    assert wake.pair_deficit(-100.0, 0.0) == 0.0
    assert wake.pair_deficit(0.0, 0.0) == 0.0


def test_deficit_decreases_with_distance():
    near = wake.pair_deficit(3 * wake.D, 0.0)
    far = wake.pair_deficit(9 * wake.D, 0.0)
    assert near > far > 0.0


def test_deficit_decreases_with_crosswind_offset():
    centred = wake.pair_deficit(5 * wake.D, 0.0)
    offset = wake.pair_deficit(5 * wake.D, 1.5 * wake.D)
    assert centred > offset >= 0.0


def test_circle_overlap_limits():
    # identical circles fully overlap, far apart circles do not overlap
    assert np.isclose(wake._circle_overlap(wake.R, wake.R, 0.0),
                      np.pi * wake.R ** 2)
    assert wake._circle_overlap(wake.R, wake.R, 10 * wake.R) == 0.0


def test_higher_ct_makes_a_deeper_wake():
    low = wake.pair_deficit(5 * wake.D, 0.0, ct=0.5)
    high = wake.pair_deficit(5 * wake.D, 0.0, ct=0.9)
    assert high > low


def test_computed_wake_loss_is_a_sensible_fraction():
    df = energy_yield.load()
    pos = wake.layout()
    d = wake.deficit_by_direction(pos)
    loss = wake.computed_wake_loss(d, df)
    assert 0.0 < loss < 0.5


def test_shallow_layout_beats_deep_layout():
    # for a one directional wind, wide and shallow loses less than deep and narrow
    df = energy_yield.load()
    shallow = wake.computed_wake_loss(
        wake.deficit_by_direction(wake.layout(n_along=2, n_across=8)), df)
    deep = wake.computed_wake_loss(
        wake.deficit_by_direction(wake.layout(n_along=8, n_across=2)), df)
    assert shallow < deep