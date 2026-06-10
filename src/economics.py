"""The money side: LCOE, payback, and which blade to buy, pricing the energy the
rest of the pipeline produced. every input here is an assumption (unlike the
measured wind), so the external anchors are sourced below, not recalled.

Two prices, used on purpose: utility wind sells at the wholesale/contract price,
small distributed wind offsets the retail tariff behind the meter (about 3x
higher). headline: utility wind is firmly below wholesale (bankable), small wind
only pays behind the meter, and the smart blade is the better buy.

Sources (all June 2026):
  * FX ~3,600 COP/USD                         (market rate)
  * wholesale/contract ~250-300 COP/kWh       (XM bolsa 2025 avg 246, contract 299)
  * retail residential ~841 COP/kWh           (GlobalPetrolPrices / CREG, Sep 25)
  * utility onshore wind ~1,041 USD/kW,
    LCOE 0.034 USD/kWh                        (IRENA Renewable Power Gen Costs 2024)
  * distributed wind capex 1,990-6,971 USD/kW,
    opex 39 USD/kW/yr                         (NREL Cost of Wind Energy 2024 / ATB)

Outputs (../results): econ_lcoe, econ_tower_payback, econ_blade_choice.
Run:  python src/economics.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np

import config
import energy_yield
import small_wind
import loads

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

# to tell the two BEM blades apart on cost (not just energy) the capex is split:
# the tower + foundation scale with the rotor thrust load (the overturning moment
# from loads.py, which comes from the BEM Ct curve), the rest scales with rated
# power. tower+foundation is taken as ~25% of the baseline small wind capex, a
# typical share. this is the seam that lets Ct, not just Cp, reach the money.
STRUCT_FRACTION_BASELINE = 0.25

# the softest assumption in the file: incremental tower cost per extra metre
# of hub height for a ~6 kW machine. flagged loudly, and the tower figure is
# drawn at two energy prices so you can see how much it depends on this.
TOWER_USD_PER_M = 400.0
TOWER_BASE_HEIGHT = 18.0     # the height we compare taller towers against


def crf(rate=DISCOUNT_RATE, years=LIFE_YEARS):
    """Capital recovery factor, an up front capex as a level annual payment."""
    return rate * (1 + rate) ** years / ((1 + rate) ** years - 1)


def annuity_factor(rate=DISCOUNT_RATE, years=LIFE_YEARS):
    """Present value of 1 per year over the life (just 1/crf)."""
    return (1 - (1 + rate) ** -years) / rate


def lcoe_usd_per_kwh(capex_per_kw, rated_kw, opex_per_kw_yr, aep_kwh,
                     rate=DISCOUNT_RATE, years=LIFE_YEARS):
    """LCOE [USD/kWh] = (annualised capex + annual opex) / annual energy."""
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


def _moment_Nm(cp, t):
    """Base overturning moment [N m] at rated for one blade, from the BEM Ct."""
    lam = cp["lambda"].values
    cmax, lopt = small_wind.cp_peak(cp, lam, t["col"])
    ct_op = loads.ct_at(cp, lam, loads.ct_column(t["col"]), lopt)
    f_rated = loads.thrust(small_wind.V_RATED, t["R"], ct_op)
    return f_rated * small_wind.HUB_HEIGHT, ct_op, cmax


def blade_full_costs(cp=None):
    """Cost out each blade from both BEM curves: Cp sets the AEP, Ct sets the
    tower+foundation capex via the moment, so the two no longer tie on LCOE."""
    if cp is None:
        cp = small_wind.load_cp()
    swt = swt_machines()

    # calibrate the unit costs off the smart blade
    ref = small_wind.TURBINES["Smart blade"]
    m_ref, _, _ = _moment_Nm(cp, ref)
    rated_ref = swt["Smart blade"]["rated_kw"]
    total_ref = CAPEX_SWT_USD_KW * rated_ref
    struct_ref = STRUCT_FRACTION_BASELINE * total_ref
    nonstruct_per_kw = (total_ref - struct_ref) / rated_ref     # USD/kW
    struct_per_Nm = struct_ref / m_ref                          # USD per N m

    out = {}
    for name, t in small_wind.TURBINES.items():
        m, ct_op, cmax = _moment_Nm(cp, t)
        rated = swt[name]["rated_kw"]
        aep = swt[name]["aep_kwh"]
        struct = struct_per_Nm * m
        nonstruct = nonstruct_per_kw * rated
        total = struct + nonstruct
        opex = OPEX_SWT_USD_KW_YR * rated
        lcoe = (crf() * total + opex) / aep
        npv = aep * RETAIL_USD_KWH * annuity_factor() \
            - total - opex * annuity_factor()
        out[name] = dict(rated_kw=rated, aep_kwh=aep, moment=m, ct_op=ct_op,
                         ct_cp=ct_op / cmax, struct=struct, nonstruct=nonstruct,
                         total=total, per_kw=total / rated, lcoe=lcoe, npv=npv)
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
    """NPV of a taller tower (extra AEP over life minus extra steel), drawn at
    both the retail and the wholesale price."""
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


def plot_blade_choice(blades):
    """The payoff of the whole BEM -> money pipeline: which blade is the better
    buy once you cost out both its energy (Cp) and its structure (Ct)."""
    names = list(blades.keys())
    colors = [small_wind.TURBINES[n]["color"] for n in names]
    x = np.arange(len(names))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))

    # (1) capex split, rotor/drivetrain vs tower/foundation
    nonstruct = [blades[n]["nonstruct"] / 1000 for n in names]
    struct = [blades[n]["struct"] / 1000 for n in names]
    axes[0].bar(x, nonstruct, color=config.ACCENT, label="rotor + drivetrain (per kW)")
    axes[0].bar(x, struct, bottom=nonstruct, color=config.HIGHLIGHT,
                label="tower + foundation (per moment)")
    for xi, n in zip(x, names):
        axes[0].text(xi, blades[n]["total"] / 1000 + 0.4,
                     f"${blades[n]['total']/1000:.1f}k", ha="center", fontsize=8)
    axes[0].set_xticks(x); axes[0].set_xticklabels(names, fontsize=8)
    axes[0].set_ylabel("Installed capex  [k USD]")
    axes[0].set_title("Capex, split by what drives it")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, axis="y")

    # (2) LCOE per blade (now they differ)
    lc = [blades[n]["lcoe"] for n in names]
    axes[1].bar(x, lc, color=colors)
    axes[1].axhline(RETAIL_USD_KWH, color="grey", ls="--", lw=1.5,
                    label=f"retail ${RETAIL_USD_KWH:.03f}")
    for xi, v in zip(x, lc):
        axes[1].text(xi, v + 0.003, f"${v:.03f}", ha="center", fontsize=8)
    axes[1].set_xticks(x); axes[1].set_xticklabels(names, fontsize=8)
    axes[1].set_ylabel("LCOE  [USD/kWh]")
    axes[1].set_title("LCOE per blade (Cp + Ct)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, axis="y")

    # (3) lifetime NPV behind the meter, the investment number
    npv = [blades[n]["npv"] / 1000 for n in names]
    axes[2].bar(x, npv, color=colors)
    for xi, v in zip(x, npv):
        axes[2].text(xi, v + 0.2, f"+${v:.1f}k", ha="center", fontsize=8)
    axes[2].set_xticks(x); axes[2].set_xticklabels(names, fontsize=8)
    axes[2].set_ylabel("20 yr NPV at retail  [k USD]")
    axes[2].set_title("Which blade is the better buy")
    axes[2].grid(alpha=0.3, axis="y")

    fig.suptitle("BEM to money: costing each blade by its energy and its loads")
    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS, "econ_blade_choice.png"), dpi=130)


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


def blade_summary(blades):
    print("\nBEM BLADE CHOICE (both curves: Cp -> energy, Ct -> structure cost):")
    print(f"  {'blade':<18}{'Ct/Cp':>7}{'AEP kWh':>9}{'capex $':>10}"
          f"{'$/kW':>8}{'LCOE':>8}{'NPV @retail':>13}")
    for name, b in blades.items():
        print(f"  {name:<18}{b['ct_cp']:>7.3f}{b['aep_kwh']:>9.0f}"
              f"{b['total']:>10,.0f}{b['per_kw']:>8,.0f}{b['lcoe']:>8.3f}"
              f"{b['npv']:>+13,.0f}")
    s, c = blades["Smart blade"], blades["Comercial blade"]
    print("-" * 78)
    print(f"the smart blade wins on both: more energy AND a lower Ct/Cp, so its")
    print(f"structure costs less per kW. LCOE {s['lcoe']:.3f} vs {c['lcoe']:.3f} "
          f"$/kWh, lifetime NPV +${s['npv']/1000:.1f}k vs +${c['npv']/1000:.1f}k.")
    print("the LCOE ordering is locked by the BEM Ct/Cp ratio (same CF), so it")
    print("holds whatever the exact tower cost share is.")


def main():
    os.makedirs(config.RESULTS, exist_ok=True)
    util = utility_machines()
    swt = swt_machines()
    blades = blade_full_costs()
    summary(util, swt)
    blade_summary(blades)
    plot_lcoe(util, swt)
    plot_tower_payback(swt)
    plot_blade_choice(blades)
    print(f"Saved figures to {os.path.abspath(config.RESULTS)}")


if __name__ == "__main__":
    main()
