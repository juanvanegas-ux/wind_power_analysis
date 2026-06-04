"""Small wind turbine (SWT) analysis for La Guajira, using the Cp(lambda)
curves from the companion BEM solver.

This ties the two projects together. the BEM repo
(https://github.com/juanvanegas-ux/wind_turbine_bem) designs two ~2.3-2.5 m
rotors for a small machine, a smart (bend twist coupled) blade and a comercial
one, and spits out their power coefficient Cp as a function of tip speed ratio
lambda. here those exact curves are dropped onto the La Guajira wind resource to
see what they would actually make.

Two things make this a SWT study and not the multi MW one in the rest of the
repo:
  * the rotors are tiny (a few kW), so Cp comes from the real blade design, not a
    generic curve
  * small turbines sit on short towers, so the wind is taken at ~24 m, not 100 m.
    the 10 m measurement is pushed up to hub height with the measured shear.

Two control strategies are modelled from the same Cp curve:
  * variable speed (MPPT): the rotor tracks lambda_opt so Cp sits at its peak
    across the whole productive range, then the generator caps the power. this is
    what a modern small turbine does.
  * fixed speed: the rotor spins at a constant rpm (the 250 rpm the BEM study
    used), so lambda slides along the curve as the wind changes and Cp falls off
    either side of the design point. this is the old school stall regulated way
    and it shows why variable speed wins.

On top of the headline numbers there are a few studies a small wind buyer
actually cares about:
  * where the energy comes from (the power curve folded onto the wind
    distribution at hub height)
  * how much the tower height is worth (AEP vs hub height, using the measured
    shear, with the generator nameplate held fixed)
  * how to size the generator (sweeping the rated wind speed, i.e. the specific
    power in W/m2, and watching capacity factor trade against annual energy)

A note on the capacity factor. ETA below is the electromechanical conversion
(rotor shaft to AC out). on top of that a real machine loses energy to downtime,
dirty blades and the wiring to the point of use, so the script reports both an
idealized (gross) number and a net one on the same loss basis as energy_yield.py
(minus the wake loss, since a single small turbine has no neighbours to steal
from). even the net figure is optimistic for small wind, which in the field runs
lower because of turbulence near the ground, short towers and rough siting. the
La Guajira resource is doing the heavy lifting here.

Outputs (saved to ../results):
  - swt_cp_curves.png        the BEM Cp(lambda) curves
  - swt_power_curves.png     derived power curves, both blades, both strategies
  - swt_compare.png          capacity factor and annual energy
  - swt_energy_distribution.png  where the annual energy actually comes from
  - swt_hub_height.png       what a taller tower is worth
  - swt_specific_power.png   sizing the generator (rated speed / specific power)

Run:  python src/small_wind.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from wind_shear import shear_exponent, extrapolate

HOURS_PER_YEAR = 8760.0
CP_DATA = os.path.join(os.path.dirname(__file__), "..", "data",
                       "bem_cp_lambda.csv")

# small turbine assumptions (edit these for another machine)
HUB_HEIGHT = 24.0   # m, short tower for a small turbine
ETA = 0.90          # drivetrain + generator + inverter efficiency
CUT_IN = 3.0        # m/s
V_RATED = 11.0      # m/s, where the generator reaches its cap
CUT_OUT = 20.0      # m/s, small turbines furl/brake earlier than big ones
FIXED_RPM = 250.0   # rpm for the fixed speed case (the BEM reference speed)

# net losses for a single small turbine. no wake (it has no neighbours), but it
# still loses to downtime, dirty blades and the wiring run. same idea and basis
# as energy_yield.py, just without the wake bucket. stacked multiplicatively.
SWT_LOSSES = {
    "availability": 0.04,          # downtime, more relatively than a big machine
    "soiling": 0.02,               # dust/salt on the blades drops Cp
    "downstream electrical": 0.02, # wiring to the point of use
}

# the two blades from the BEM study (radius in m, Cp column in the csv)
TURBINES = {
    "Smart blade":     {"R": 2.500, "col": "Cp_Smart",     "color": config.ACCENT},
    "Comercial blade": {"R": 2.275, "col": "Cp_Comercial", "color": config.HIGHLIGHT},
}


def load_cp() -> pd.DataFrame:
    return pd.read_csv(CP_DATA)


def cp_peak(cp, lam, col):
    """Peak Cp and the tip speed ratio it happens at, for one blade."""
    i = int(np.argmax(cp[col].values))
    return float(cp[col].values[i]), float(lam[i])


def swept_area(R):
    return np.pi * R ** 2


def net_loss_factor(losses=SWT_LOSSES):
    """Combine the small turbine losses multiplicatively (same as the farm
    loss stack in energy_yield.py, just without the wake bucket)."""
    keep = 1.0
    for f in losses.values():
        keep *= (1.0 - f)
    return 1.0 - keep


def power_mppt(V, R, cp_max, rated_power, rho=config.AIR_DENSITY):
    """Variable speed power curve [W]: the rotor tracks lambda_opt so Cp is
    pinned at its peak, then the generator caps the output."""
    V = np.asarray(V, dtype=float)
    p = ETA * 0.5 * rho * swept_area(R) * cp_max * V ** 3
    p = np.clip(p, 0.0, rated_power)
    p[(V < CUT_IN) | (V > CUT_OUT)] = 0.0
    return p


def power_fixed(V, R, cp, lam, col, rated_power, rpm=FIXED_RPM,
                rho=config.AIR_DENSITY):
    """Fixed speed power curve [W]: constant rpm, so lambda = omega*R/V slides
    along the Cp curve and Cp drops away from the design point."""
    V = np.asarray(V, dtype=float)
    omega = rpm * 2.0 * np.pi / 60.0
    with np.errstate(divide="ignore"):
        lam_local = omega * R / V
    # interpolate Cp from the curve, zero outside the measured lambda range
    cp_local = np.interp(lam_local, lam, cp[col].values, left=0.0, right=0.0)
    p = ETA * 0.5 * rho * swept_area(R) * cp_local * V ** 3
    p = np.clip(p, 0.0, rated_power)
    p[(V < CUT_IN) | (V > CUT_OUT)] = 0.0
    return p


def hub_wind(df, height=HUB_HEIGHT) -> np.ndarray:
    """Bring the 10 m wind up to a hub height with the per hour power law shear
    measured between 10 m and 100 m."""
    alpha = shear_exponent(df["wind_speed_10m_ms"].values,
                           df["wind_speed_100m_ms"].values)
    alpha = np.where(np.isnan(alpha), np.nanmedian(alpha), alpha)
    return extrapolate(df["wind_speed_10m_ms"].values, alpha,
                       config.HEIGHT_LOW, height)


def rated_power_of(R, cp_max, v_rated=V_RATED, rho=config.AIR_DENSITY):
    """Generator cap, sized as the MPPT power at the rated wind speed."""
    return ETA * 0.5 * rho * swept_area(R) * cp_max * v_rated ** 3


def annual_energy_mwh(power_w):
    """Mean power [W] over the record -> annual energy [MWh/yr]."""
    return power_w.mean() / 1e6 * HOURS_PER_YEAR


def analyse(cp):
    lam = cp["lambda"].values
    v_hub = hub_wind(load_wind())
    loss = net_loss_factor()
    print(f"Hub height wind ({HUB_HEIGHT:.0f} m): mean {v_hub.mean():.2f} m/s")
    print(f"Net loss factor (no wake): {loss:.1%}  "
          f"[{', '.join(f'{k} {v:.0%}' for k, v in SWT_LOSSES.items())}]")
    print("=" * 92)
    print(f"{'turbine':<18}{'R [m]':>7}{'Cp_max':>8}{'lam_opt':>9}"
          f"{'rated kW':>10}{'W/m2':>7}{'CF gross':>10}{'CF net':>9}"
          f"{'AEP net':>10}")
    print("-" * 92)

    results = {}
    for name, t in TURBINES.items():
        cmax, lopt = cp_peak(cp, lam, t["col"])
        rated = rated_power_of(t["R"], cmax)
        spec_power = rated / swept_area(t["R"])   # W/m2

        p_vs = power_mppt(v_hub, t["R"], cmax, rated)
        p_fs = power_fixed(v_hub, t["R"], cp, lam, t["col"], rated)

        cf_vs = p_vs.mean() / rated
        cf_fs = p_fs.mean() / rated
        aep_vs = annual_energy_mwh(p_vs)               # gross MWh/yr
        aep_fs = annual_energy_mwh(p_fs)

        results[name] = dict(
            cmax=cmax, lopt=lopt, rated=rated, spec_power=spec_power,
            cf_vs=cf_vs, cf_fs=cf_fs,
            cf_vs_net=cf_vs * (1 - loss), cf_fs_net=cf_fs * (1 - loss),
            aep_vs=aep_vs, aep_fs=aep_fs,
            aep_vs_net=aep_vs * (1 - loss), aep_fs_net=aep_fs * (1 - loss),
        )
        r = results[name]
        print(f"{name:<18}{t['R']:>7.3f}{cmax:>8.3f}{lopt:>9.2f}"
              f"{rated/1000:>10.2f}{spec_power:>7.0f}{cf_vs:>9.1%}"
              f"{r['cf_vs_net']:>9.1%}{r['aep_vs_net']:>10.1f}")

    print("-" * 92)
    print(f"rated/mean wind ratio (sizing): {V_RATED/v_hub.mean():.2f}  "
          f"(low ratio -> high CF, this is why the numbers look so good)")
    print("CF/AEP columns are variable speed (MPPT). gross = ETA only, "
          "net = after the loss stack above.")
    return v_hub, results


# --------------------------------------------------------------------------
# data + plots
# --------------------------------------------------------------------------

def load_wind() -> pd.DataFrame:
    df = pd.read_csv(config.DATA, parse_dates=["timestamp"])
    return df.dropna(subset=["wind_speed_10m_ms", "wind_speed_100m_ms"])


def plot_cp(cp):
    lam = cp["lambda"].values
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, t in TURBINES.items():
        cmax, lopt = cp_peak(cp, lam, t["col"])
        ax.plot(lam, cp[t["col"]].values, marker="o", ms=3, color=t["color"],
                label=f"{name} (Cp_max {cmax:.3f} @ {lopt:.1f})")
    ax.set_xlabel("Tip speed ratio lambda  [-]")
    ax.set_ylabel("Power coefficient Cp  [-]")
    ax.set_title("BEM blade Cp curves (from the companion repo)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS, "swt_cp_curves.png"), dpi=130)


def plot_power_curves(cp):
    lam = cp["lambda"].values
    V = np.linspace(0, CUT_OUT + 1, 200)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for name, t in TURBINES.items():
        cmax, _ = cp_peak(cp, lam, t["col"])
        rated = rated_power_of(t["R"], cmax)
        ax.plot(V, power_mppt(V, t["R"], cmax, rated) / 1000.0,
                color=t["color"], lw=2, label=f"{name}, variable speed")
        ax.plot(V, power_fixed(V, t["R"], cp, lam, t["col"], rated) / 1000.0,
                color=t["color"], lw=1.5, ls="--",
                label=f"{name}, fixed {FIXED_RPM:.0f} rpm")
    ax.set_xlabel("Wind speed at hub  [m/s]")
    ax.set_ylabel("Electrical power  [kW]")
    ax.set_title("Power curves built from the Cp(lambda) data")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS, "swt_power_curves.png"), dpi=130)


def plot_compare(results):
    names = list(results.keys())
    x = np.arange(len(names))
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    axes[0].bar(x - w / 2, [results[n]["cf_vs_net"] * 100 for n in names], w,
                color=config.ACCENT, label="variable speed")
    axes[0].bar(x + w / 2, [results[n]["cf_fs_net"] * 100 for n in names], w,
                color=config.ACCENT, alpha=0.45, label="fixed speed")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names)
    axes[0].set_ylabel("Net capacity factor  [%]")
    axes[0].set_title("Capacity factor (net)")
    axes[0].legend()
    axes[0].grid(alpha=0.3, axis="y")

    axes[1].bar(x - w / 2, [results[n]["aep_vs_net"] for n in names], w,
                color=config.HIGHLIGHT, label="variable speed")
    axes[1].bar(x + w / 2, [results[n]["aep_fs_net"] for n in names], w,
                color=config.HIGHLIGHT, alpha=0.45, label="fixed speed")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names)
    axes[1].set_ylabel("Net annual energy  [MWh/yr]")
    axes[1].set_title("AEP per turbine (net)")
    axes[1].legend()
    axes[1].grid(alpha=0.3, axis="y")

    fig.suptitle("Small wind on the La Guajira resource (24 m hub, net of losses)")
    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS, "swt_compare.png"), dpi=130)


def plot_energy_distribution(cp, v_hub):
    """Fold the power curve onto the wind distribution at hub height to show
    where the annual energy actually comes from. the energy peak sits well
    above the most common wind, because energy goes with v^3."""
    t = TURBINES["Smart blade"]
    lam = cp["lambda"].values
    cmax, _ = cp_peak(cp, lam, t["col"])
    rated = rated_power_of(t["R"], cmax)

    edges = np.arange(0, np.ceil(v_hub.max()) + 1.0, 1.0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    hours, _ = np.histogram(v_hub, bins=edges)
    hours = hours / len(v_hub) * HOURS_PER_YEAR          # hours/yr in each bin

    p_curve = power_mppt(centers, t["R"], cmax, rated) / 1000.0   # kW at bin mid
    energy = p_curve * hours                              # kWh/yr per bin

    fig, ax1 = plt.subplots(figsize=(8.5, 4.8))
    ax1.bar(centers, hours, width=0.9, color=config.ACCENT, alpha=0.35,
            label="hours per year (resource)")
    ax1.bar(centers, energy / energy.max() * hours.max(), width=0.45,
            color=config.HIGHLIGHT, alpha=0.85,
            label="energy per bin (scaled)")
    ax1.set_xlabel("Wind speed at 24 m hub  [m/s]")
    ax1.set_ylabel("Hours per year  /  energy (scaled)")
    ax1.axvline(CUT_IN, color="grey", ls=":", lw=1)
    ax1.axvline(V_RATED, color="black", ls="--", lw=1)
    ax1.text(V_RATED + 0.1, hours.max() * 0.95, "rated", rotation=90,
             va="top", fontsize=8)

    ax2 = ax1.twinx()
    ax2.plot(centers, p_curve, color="black", lw=2, marker="o", ms=3,
             label="power curve [kW]")
    ax2.set_ylabel("Electrical power  [kW]")

    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, fontsize=8, loc="upper right")
    ax1.set_title("Where the year's energy comes from (Smart blade, variable speed)")
    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS, "swt_energy_distribution.png"),
                dpi=130)

    # the modal wind vs the energy weighted wind, a nice honest number
    modal = centers[np.argmax(hours)]
    e_weighted = np.sum(centers * energy) / np.sum(energy)
    print(f"Energy distribution: modal wind {modal:.1f} m/s, but the average "
          f"kWh arrives at {e_weighted:.1f} m/s")


def plot_hub_height(cp, df):
    """What is a taller tower worth. sweep the hub height and recompute AEP,
    holding the generator nameplate fixed (it does not change with the tower),
    so only the wind distribution shifts up."""
    lam = cp["lambda"].values
    heights = np.arange(10, 41, 2.0)
    loss = net_loss_factor()

    fig, ax1 = plt.subplots(figsize=(8, 4.8))
    ax2 = ax1.twinx()
    for name, t in TURBINES.items():
        cmax, _ = cp_peak(cp, lam, t["col"])
        rated = rated_power_of(t["R"], cmax)   # fixed across the sweep
        aeps = []
        for h in heights:
            v = hub_wind(df, height=h)
            aeps.append(annual_energy_mwh(power_mppt(v, t["R"], cmax, rated))
                        * (1 - loss))
        ax1.plot(heights, aeps, marker="o", ms=3, color=t["color"],
                 label=f"{name} net AEP")

    mean_v = [hub_wind(df, height=h).mean() for h in heights]
    ax2.plot(heights, mean_v, color="grey", ls="--", lw=1.5,
             label="mean hub wind")
    ax1.axvline(HUB_HEIGHT, color="black", ls=":", lw=1)
    ax1.text(HUB_HEIGHT + 0.3, ax1.get_ylim()[0], "baseline 24 m",
             rotation=90, va="bottom", fontsize=8)
    ax1.set_xlabel("Hub height  [m]")
    ax1.set_ylabel("Net AEP  [MWh/yr]")
    ax2.set_ylabel("Mean wind at hub  [m/s]")
    ax1.set_title("What a taller tower is worth (nameplate held fixed)")
    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, fontsize=8, loc="upper left")
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS, "swt_hub_height.png"), dpi=130)

    # quick "is the tower worth it" number for the smart blade
    t = TURBINES["Smart blade"]
    cmax, _ = cp_peak(cp, lam, t["col"])
    rated = rated_power_of(t["R"], cmax)
    a18 = annual_energy_mwh(power_mppt(hub_wind(df, 18), t["R"], cmax, rated))
    a30 = annual_energy_mwh(power_mppt(hub_wind(df, 30), t["R"], cmax, rated))
    print(f"Tower height: going 18 m -> 30 m lifts Smart blade AEP "
          f"{(a30/a18 - 1)*100:.0f}% ({a18:.1f} -> {a30:.1f} MWh/yr gross)")


def plot_specific_power(cp, v_hub):
    """Sizing the generator. sweep the rated wind speed, which is the same as
    sweeping the specific power (rated W per m2 of rotor), and watch capacity
    factor trade off against annual energy. a small generator (low specific
    power) gives a gorgeous CF but caps the energy, a big one does the opposite.
    """
    t = TURBINES["Smart blade"]
    lam = cp["lambda"].values
    cmax, _ = cp_peak(cp, lam, t["col"])
    area = swept_area(t["R"])
    loss = net_loss_factor()

    v_rated_grid = np.arange(8.0, 16.1, 0.5)
    spec_power, cfs, aeps = [], [], []
    for vr in v_rated_grid:
        rated = rated_power_of(t["R"], cmax, v_rated=vr)
        p = power_mppt(v_hub, t["R"], cmax, rated)
        spec_power.append(rated / area)
        cfs.append(p.mean() / rated * (1 - loss) * 100)
        aeps.append(annual_energy_mwh(p) * (1 - loss))

    spec_power = np.array(spec_power)
    fig, ax1 = plt.subplots(figsize=(8, 4.8))
    ax2 = ax1.twinx()
    l1 = ax1.plot(spec_power, cfs, marker="o", ms=3, color=config.ACCENT,
                  label="net capacity factor")
    l2 = ax2.plot(spec_power, aeps, marker="s", ms=3, color=config.HIGHLIGHT,
                  label="net AEP")
    # mark the baseline V_rated = 11
    base_sp = rated_power_of(t["R"], cmax, v_rated=V_RATED) / area
    ax1.axvline(base_sp, color="black", ls=":", lw=1)
    ax1.text(base_sp + 5, min(cfs) + 1, f"baseline\nV_rated {V_RATED:.0f}",
             fontsize=8)
    ax1.set_xlabel("Specific power  [W/m2]  (bigger generator ->)")
    ax1.set_ylabel("Net capacity factor  [%]", color=config.ACCENT)
    ax2.set_ylabel("Net AEP  [MWh/yr]", color=config.HIGHLIGHT)
    ax1.set_title("Sizing the generator (Smart blade): CF vs energy trade off")
    lines = l1 + l2
    ax1.legend(lines, [ln.get_label() for ln in lines], loc="center right",
               fontsize=8)
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS, "swt_specific_power.png"), dpi=130)


def main():
    os.makedirs(config.RESULTS, exist_ok=True)
    cp = load_cp()
    df = load_wind()
    v_hub, results = analyse(cp)
    plot_cp(cp)
    plot_power_curves(cp)
    plot_compare(results)
    plot_energy_distribution(cp, v_hub)
    plot_hub_height(cp, df)
    plot_specific_power(cp, v_hub)
    print(f"Saved figures to {os.path.abspath(config.RESULTS)}")


if __name__ == "__main__":
    main()
