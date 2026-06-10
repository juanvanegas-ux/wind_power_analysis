"""The money side: what the energy is actually worth at La Guajira.

Every other script in this repo sticks to things you can measure from the wind
data. economics is different, every input here is an *assumption*, so the rule
is to source the few external numbers (not recall them), put them in one visible
block, and let the result fall out. all the cost/price anchors below are dated
June 2026 and cited in the comments.

What it computes:
  * LCOE (levelised cost of energy) for the small turbines (from small_wind.py)
    and the utility machines (from energy_yield.py), on the same net AEP basis
  * the two relevant electricity prices, because they are NOT the same number:
      - utility scale wind is valued at the wholesale / contract price (it sells
        into the grid)
      - small distributed wind is valued at the avoided retail tariff (it offsets
        what the owner would otherwise buy, behind the meter)
  * whether a taller tower pays for itself, by putting a price on the extra AEP
    the hub height sweep in small_wind.py already produced

The honest headline it lands on: utility wind here is firmly below the wholesale
price (very bankable), while small wind sits above wholesale and only makes sense
behind the meter against the retail tariff, where this site's unusually high
capacity factor drags it to roughly break even. that conclusion holds across the
whole small wind capex range, which is why there is a sensitivity, not a single
bar.

Sources (all June 2026):
  * FX ~3,600 COP/USD                         (market rate, June 2026)
  * wholesale/contract ~250-300 COP/kWh       (XM bolsa 2025 avg 246, contract
                                               avg 299; 2026 ytd ~213)
  * retail residential ~841 COP/kWh           (GlobalPetrolPrices / CREG, Sep 25)
  * utility onshore wind ~1,041 USD/kW,
    LCOE 0.034 USD/kWh                        (IRENA Renewable Power Gen Costs 2024)
  * distributed wind capex 1,990-6,971 USD/kW,
    opex 39 USD/kW/yr                         (NREL Cost of Wind Energy 2024 / ATB)

Outputs (saved to ../results):
  - econ_lcoe.png          LCOE of every machine against the price it can earn
  - econ_tower_payback.png does a taller small wind tower pay for itself

Run:  python src/economics.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np

import config
import energy_yield
import small_wind

# --- finance -----------------------------------------------------------
COP_PER_USD = 3600.0     # June 2026 market rate
DISCOUNT_RATE = 0.09     # WACC, reasonable for a Colombian renewable project
LIFE_YEARS = 20          # project life

# --- electricity prices (two different numbers, on purpose) ------------
# utility wind sells into the grid -> wholesale/contract price
WHOLESALE_COP_KWH = 275.0    # between the 2025 bolsa (246) and contract (299)
# small wind offsets a retail purchase -> avoided retail tariff
RETAIL_COP_KWH = 841.0       # GlobalPetrolPrices / CREG residential, Sep 2025
IRENA_LCOE_USD_KWH = 0.034   # IRENA 2024 global onshore wind benchmark

WHOLESALE_USD_KWH = WHOLESALE_COP_KWH / COP_PER_USD   # ~0.076
RETAIL_USD_KWH = RETAIL_COP_KWH / COP_PER_USD         # ~0.234

# --- capex / opex ------------------------------------------------------
# utility onshore, IRENA global is 1,041 USD/kW, Latin America runs a bit
# higher, so 1,300 central with a note. opex a standard ~40 USD/kW/yr.
CAPEX_UTILITY_USD_KW = 1300.0
OPEX_UTILITY_USD_KW_YR = 40.0

# small/distributed wind is far pricier per kW. NREL range is 1,990-6,971,
# residential turbines sit high, so 5,000 central with the full range shown.
CAPEX_SWT_USD_KW = 5000.0
CAPEX_SWT_RANGE = (3000.0, 7000.0)
OPEX_SWT_USD_KW_YR = 39.0    # NREL assumption

# the softest assumption in the file: incremental tower cost per extra metre
# of hub height for a ~6 kW machine. flagged loudly, and the tower figure is
# drawn at two energy prices so you can see how much it depends on this.
TOWER_USD_PER_M = 400.0
TOWER_BASE_HEIGHT = 18.0     # the height we compare taller towers against


def crf(rate=DISCOUNT_RATE, years=LIFE_YEARS):
    """Capital recovery factor: turns an up front capex into a level annual
    payment over the project life."""
    return rate * (1 + rate) ** years / ((1 + rate) ** years - 1)


def annuity_factor(rate=DISCOUNT_RATE, years=LIFE_YEARS):
    """Present value of 1 per year for `years` years. it is just 1/crf, used to
    discount a stream of annual energy savings back to today."""
    return (1 - (1 + rate) ** -years) / rate


def lcoe_usd_per_kwh(capex_per_kw, rated_kw, opex_per_kw_yr, aep_kwh,
                     rate=DISCOUNT_RATE, years=LIFE_YEARS):
    """Levelised cost of energy [USD/kWh].

    LCOE = (annualised capex + annual opex) / annual energy. it is the price the
    energy has to fetch, averaged over the life, for the project to break even.
    """
    capex = capex_per_kw * rated_kw
    annual_cost = crf(rate, years) * capex + opex_per_kw_yr * rated_kw
    return annual_cost / aep_kwh


def simple_payback_years(capex_total, annual_net_revenue):
    """Undiscounted payback. crude but the number everyone asks for first."""
    if annual_net_revenue <= 0:
        return float("inf")
    return capex_total / annual_net_revenue


# --------------------------------------------------------------------------
# gather the energy numbers from the other modules (net AEP, same basis)
# --------------------------------------------------------------------------

def utility_machines():
    """Net AEP and rated kW for each utility class from energy_yield.py."""
    df = energy_yield.load()
    v = df["wind_speed_100m_ms"].values
    out = {}
    for name, params in energy_yield.TURBINES.items():
        _, net_aep_mwh, _, net_cf = energy_yield.yield_for_turbine(v, params)
        rated_kw = params[3] * 1000.0
        out[name] = dict(rated_kw=rated_kw, aep_kwh=net_aep_mwh * 1000.0,
                         net_cf=net_cf)
    return out


def swt_machines():
    """Net AEP and rated kW for each small blade from small_wind.py."""
    cp = small_wind.load_cp()
    lam = cp["lambda"].values
    v_hub = small_wind.hub_wind(small_wind.load_wind())
    loss = small_wind.net_loss_factor()
    out = {}
    for name, t in small_wind.TURBINES.items():
        cmax, _ = small_wind.cp_peak(cp, lam, t["col"])
        rated_w = small_wind.rated_power_of(t["R"], cmax)
        p = small_wind.power_mppt(v_hub, t["R"], cmax, rated_w)
        net_aep_kwh = p.mean() / 1000.0 * small_wind.HOURS_PER_YEAR * (1 - loss)
        out[name] = dict(rated_kw=rated_w / 1000.0, aep_kwh=net_aep_kwh)
    return out


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

def plot_lcoe(util, swt):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    # ---- utility, valued at the wholesale price ----
    names = list(util.keys())
    short = [n.split(" (")[0] for n in names]
    lc = [lcoe_usd_per_kwh(CAPEX_UTILITY_USD_KW, util[n]["rated_kw"],
                           OPEX_UTILITY_USD_KW_YR, util[n]["aep_kwh"])
          for n in names]
    x = np.arange(len(names))
    axes[0].bar(x, lc, color=config.ACCENT, alpha=0.85)
    axes[0].axhline(WHOLESALE_USD_KWH, color=config.HIGHLIGHT, lw=2,
                    label=f"wholesale price ~${WHOLESALE_USD_KWH:.03f}/kWh")
    axes[0].axhline(IRENA_LCOE_USD_KWH, color="grey", ls="--", lw=1.5,
                    label=f"IRENA 2024 global ${IRENA_LCOE_USD_KWH:.03f}/kWh")
    for xi, v in zip(x, lc):
        axes[0].text(xi, v + 0.002, f"${v:.03f}", ha="center", fontsize=8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(short, rotation=20, ha="right", fontsize=8)
    axes[0].set_ylabel("LCOE  [USD/kWh]")
    axes[0].set_title("Utility scale: LCOE vs the wholesale price it earns")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, axis="y")

    # ---- small wind, valued at the retail tariff, with capex range bars ----
    snames = list(swt.keys())
    base = [lcoe_usd_per_kwh(CAPEX_SWT_USD_KW, swt[n]["rated_kw"],
                             OPEX_SWT_USD_KW_YR, swt[n]["aep_kwh"])
            for n in snames]
    lo = [lcoe_usd_per_kwh(CAPEX_SWT_RANGE[0], swt[n]["rated_kw"],
                           OPEX_SWT_USD_KW_YR, swt[n]["aep_kwh"])
          for n in snames]
    hi = [lcoe_usd_per_kwh(CAPEX_SWT_RANGE[1], swt[n]["rated_kw"],
                           OPEX_SWT_USD_KW_YR, swt[n]["aep_kwh"])
          for n in snames]
    xs = np.arange(len(snames))
    yerr = [np.array(base) - np.array(lo), np.array(hi) - np.array(base)]
    axes[1].bar(xs, base, color=config.ACCENT, alpha=0.85,
                yerr=yerr, capsize=6,
                label=f"LCOE @ ${CAPEX_SWT_USD_KW:,.0f}/kW (bars = "
                      f"${CAPEX_SWT_RANGE[0]:,.0f}-{CAPEX_SWT_RANGE[1]:,.0f})")
    axes[1].axhline(RETAIL_USD_KWH, color=config.HIGHLIGHT, lw=2,
                    label=f"retail tariff ~${RETAIL_USD_KWH:.03f}/kWh")
    axes[1].axhline(WHOLESALE_USD_KWH, color="grey", ls="--", lw=1.5,
                    label=f"wholesale ~${WHOLESALE_USD_KWH:.03f}/kWh")
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(snames, fontsize=9)
    axes[1].set_ylabel("LCOE  [USD/kWh]")
    axes[1].set_title("Small wind: only pays behind the meter (vs retail)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, axis="y")

    fig.suptitle("What the energy costs to make vs what it can earn")
    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS, "econ_lcoe.png"), dpi=130)


def plot_tower_payback(swt):
    """Does a taller tower pay? value the extra AEP the hub height sweep buys,
    over the project life, against the extra steel. drawn at both the retail
    and the wholesale price so you can see how much the answer depends on what
    the energy is worth."""
    cp = small_wind.load_cp()
    lam = cp["lambda"].values
    df = small_wind.load_wind()
    loss = small_wind.net_loss_factor()
    t = small_wind.TURBINES["Smart blade"]
    cmax, _ = small_wind.cp_peak(cp, lam, t["col"])
    rated = small_wind.rated_power_of(t["R"], cmax)   # nameplate fixed

    heights = np.arange(TOWER_BASE_HEIGHT, 41, 2.0)
    base_aep = (small_wind.annual_energy_mwh(
        small_wind.power_mppt(small_wind.hub_wind(df, TOWER_BASE_HEIGHT),
                              t["R"], cmax, rated)) * (1 - loss) * 1000.0)

    af = annuity_factor()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for price, label, color in [
            (RETAIL_USD_KWH, "valued at retail tariff", config.ACCENT),
            (WHOLESALE_USD_KWH, "valued at wholesale", config.HIGHLIGHT)]:
        npvs = []
        for h in heights:
            aep = (small_wind.annual_energy_mwh(
                small_wind.power_mppt(small_wind.hub_wind(df, h),
                                      t["R"], cmax, rated))
                   * (1 - loss) * 1000.0)
            extra_value = (aep - base_aep) * price * af   # PV of extra energy
            extra_cost = TOWER_USD_PER_M * (h - TOWER_BASE_HEIGHT)
            npvs.append(extra_value - extra_cost)
        ax.plot(heights, npvs, marker="o", ms=3, color=color, label=label)

    ax.axhline(0, color="black", lw=1)
    ax.axvline(small_wind.HUB_HEIGHT, color="grey", ls=":", lw=1)
    ax.text(small_wind.HUB_HEIGHT + 0.3, ax.get_ylim()[1] * 0.9,
            "baseline study\nhub 24 m", fontsize=8)
    ax.set_xlabel("Hub height  [m]")
    ax.set_ylabel(f"NPV of the tower upgrade vs {TOWER_BASE_HEIGHT:.0f} m  [USD]")
    ax.set_title("Does a taller tower pay? (Smart blade, 20 yr, 9% discount)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS, "econ_tower_payback.png"), dpi=130)


# --------------------------------------------------------------------------

def summary(util, swt):
    print("=" * 78)
    print("Economics summary (assumptions sourced in the file header, June 2026)")
    print("=" * 78)
    print(f"FX {COP_PER_USD:,.0f} COP/USD | discount {DISCOUNT_RATE:.0%} | "
          f"life {LIFE_YEARS} yr | CRF {crf():.4f}")
    print(f"wholesale ~{WHOLESALE_COP_KWH:.0f} COP/kWh "
          f"(${WHOLESALE_USD_KWH:.3f}) | retail ~{RETAIL_COP_KWH:.0f} COP/kWh "
          f"(${RETAIL_USD_KWH:.3f})")
    print("-" * 78)

    print("UTILITY SCALE (sells at wholesale):")
    print(f"  {'machine':<28}{'net CF':>8}{'LCOE $/kWh':>12}"
          f"{'margin $/kWh':>14}")
    for name, m in util.items():
        lc = lcoe_usd_per_kwh(CAPEX_UTILITY_USD_KW, m["rated_kw"],
                              OPEX_UTILITY_USD_KW_YR, m["aep_kwh"])
        margin = WHOLESALE_USD_KWH - lc
        print(f"  {name.split(' (')[0]:<28}{m['net_cf']:>7.1%}{lc:>12.3f}"
              f"{margin:>+14.3f}")

    print("\nSMALL WIND (offsets retail behind the meter):")
    print(f"  {'blade':<18}{'rated kW':>9}{'LCOE $/kWh':>12}"
          f"{'payback retail':>16}{'payback wholesale':>19}")
    for name, m in swt.items():
        lc = lcoe_usd_per_kwh(CAPEX_SWT_USD_KW, m["rated_kw"],
                              OPEX_SWT_USD_KW_YR, m["aep_kwh"])
        capex = CAPEX_SWT_USD_KW * m["rated_kw"]
        opex = OPEX_SWT_USD_KW_YR * m["rated_kw"]
        pb_r = simple_payback_years(capex, m["aep_kwh"] * RETAIL_USD_KWH - opex)
        pb_w = simple_payback_years(capex,
                                    m["aep_kwh"] * WHOLESALE_USD_KWH - opex)
        pb_w_s = f"{pb_w:.1f} yr" if np.isfinite(pb_w) and pb_w < 100 \
            else "never"
        print(f"  {name:<18}{m['rated_kw']:>9.2f}{lc:>12.3f}"
              f"{pb_r:>13.1f} yr{pb_w_s:>19}")
    print("-" * 78)
    print("note: small wind LCOE > wholesale, so it only makes sense offsetting"
          " the\nretail tariff. utility LCOE < wholesale, so it is bankable as a"
          " grid seller.")


def main():
    os.makedirs(config.RESULTS, exist_ok=True)
    util = utility_machines()
    swt = swt_machines()
    summary(util, swt)
    plot_lcoe(util, swt)
    plot_tower_payback(swt)
    print(f"Saved figures to {os.path.abspath(config.RESULTS)}")


if __name__ == "__main__":
    main()
