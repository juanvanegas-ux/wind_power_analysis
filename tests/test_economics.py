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


def test_blade_costs_smart_beats_comercial():
    # the BEM -> money result: the smart blade has lower LCOE and higher NPV,
    # locked by its lower Ct/Cp (cheaper structure per kW) plus more energy
    from economics import blade_full_costs
    b = blade_full_costs()
    s, c = b["Smart blade"], b["Comercial blade"]
    assert s["ct_cp"] < c["ct_cp"]          # smart sheds load (bend twist)
    assert c["per_kw"] > s["per_kw"]        # so comercial costs more per kW
    assert s["lcoe"] < c["lcoe"]            # lower LCOE
    assert s["npv"] > c["npv"]              # better investment


def test_blade_capex_split_adds_up():
    from economics import blade_full_costs
    b = blade_full_costs()
    for name, x in b.items():
        assert np.isclose(x["struct"] + x["nonstruct"], x["total"])
        assert x["struct"] > 0 and x["nonstruct"] > 0