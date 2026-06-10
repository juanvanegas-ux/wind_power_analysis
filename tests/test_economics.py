import numpy as np

from economics import (crf, annuity_factor, lcoe_usd_per_kwh,
                       simple_payback_years)


def test_crf_known_value():
    # CRF at 10% over 20 years is a standard ~0.1175
    assert np.isclose(crf(0.10, 20), 0.117459, atol=1e-5)


def test_crf_and_annuity_are_reciprocal():
    assert np.isclose(crf(0.09, 20) * annuity_factor(0.09, 20), 1.0)


def test_lcoe_falls_when_energy_rises():
    # same machine, more annual energy -> cheaper per kWh
    a = lcoe_usd_per_kwh(1500, 2000, 40, 6_000_000)
    b = lcoe_usd_per_kwh(1500, 2000, 40, 9_000_000)
    assert b < a


def test_lcoe_rises_with_capex():
    cheap = lcoe_usd_per_kwh(1000, 2000, 40, 7_000_000)
    dear = lcoe_usd_per_kwh(2000, 2000, 40, 7_000_000)
    assert dear > cheap


def test_lcoe_only_depends_on_capacity_factor_not_size():
    # for a fixed $/kW, opex/kW and capacity factor, LCOE is size invariant
    # (this is why both blades land on the same LCOE in the writeup)
    cf = 0.40
    hours = 8760.0
    small_aep = 5.0 * cf * hours          # 5 kW machine
    big_aep = 2000.0 * cf * hours         # 2 MW machine
    small = lcoe_usd_per_kwh(4000, 5.0, 39, small_aep)
    big = lcoe_usd_per_kwh(4000, 2000.0, 39, big_aep)
    assert np.isclose(small, big)


def test_payback_infinite_when_revenue_negative():
    assert simple_payback_years(10000, -50) == float("inf")