"""Energy yield for La Guajira: how much electricity a turbine would actually
make here over a year, and how that changes with losses and with the turbine
you pick.

The resource summary in analysis.py gives a capacity factor, but a developer
cares about energy in MWh/year, gross vs net once you subtract the real world
losses (wakes between turbines, downtime, electrical losses), and which turbine
class fits the site best. that is what this script does.

AEP (annual energy production) is just the average power times the hours in a
year

    AEP = mean_power_MW * 8760

and the net number knocks off a loss factor on top of that.

Outputs (saved to ../results):
  - cf_heatmap.png       capacity factor by month and hour of day
  - turbine_compare.png  AEP and capacity factor for a few turbine classes

Run:  python src/energy_yield.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

HOURS_PER_YEAR = 8760.0

# typical wind farm loss buckets. these are industry rule of thumb numbers,
# a real bankable study would measure each one on site.
LOSSES = {
    "wake": 0.08,          # turbines stealing wind from each other
    "availability": 0.03,  # downtime for maintenance and faults
    "electrical": 0.02,    # cabling and transformer losses
    "other": 0.02,         # blade soiling, curtailment, etc
}


def total_loss_factor(losses=LOSSES):
    """Combine the individual losses multiplicatively, not by adding them.

    losses stack on what is left after the previous one, so 8% then 3% is
    not 11%, it is 1 - 0.92*0.97.
    """
    keep = 1.0
    for f in losses.values():
        keep *= (1.0 - f)
    return 1.0 - keep


# a handful of generic onshore turbines (cut_in, rated, cut_out, rated_MW).
# the point is the shape and the rated speed, not any specific product.
TURBINES = {
    "1.5 MW (class III, high wind)": (3.0, 11.0, 25.0, 1.5),
    "2.0 MW (baseline)":            (3.0, 12.0, 25.0, 2.0),
    "3.0 MW (class II)":            (3.0, 13.0, 25.0, 3.0),
    "4.5 MW (class I, low wind)":   (3.0, 14.0, 25.0, 4.5),
}


def power_curve(wind_speed, cut_in, rated, cut_out, rated_power):
    """Generic turbine power curve [MW], cubic ramp then a flat rated plateau."""
    v = np.asarray(wind_speed, dtype=float)
    p = np.zeros_like(v)
    ramp = (v >= cut_in) & (v < rated)
    p[ramp] = rated_power * (v[ramp] ** 3 - cut_in ** 3) / (rated ** 3 - cut_in ** 3)
    flat = (v >= rated) & (v <= cut_out)
    p[flat] = rated_power
    return p


def load() -> pd.DataFrame:
    df = pd.read_csv(config.DATA, parse_dates=["timestamp"])
    df = df.dropna(subset=["wind_speed_100m_ms"]).reset_index(drop=True)
    df["hour"] = df["timestamp"].dt.hour
    df["month"] = df["timestamp"].dt.month
    return df


def yield_for_turbine(v, params):
    """Return (gross AEP MWh, net AEP MWh, gross CF, net CF) for one turbine."""
    cut_in, rated, cut_out, rated_power = params
    power = power_curve(v, cut_in, rated, cut_out, rated_power)
    mean_power = power.mean()

    gross_aep = mean_power * HOURS_PER_YEAR              # MWh/year
    net_aep = gross_aep * (1.0 - total_loss_factor())
    gross_cf = mean_power / rated_power
    net_cf = gross_cf * (1.0 - total_loss_factor())
    return gross_aep, net_aep, gross_cf, net_cf


def cf_heatmap(df):
    """Capacity factor of the baseline turbine by month x hour."""
    params = TURBINES["2.0 MW (baseline)"]
    rated_power = params[3]
    df = df.copy()
    df["cf"] = power_curve(df["wind_speed_100m_ms"].values, *params) / rated_power
    grid = df.pivot_table(index="month", columns="hour", values="cf",
                          aggfunc="mean")

    fig, ax = plt.subplots(figsize=(10, 4.6))
    im = ax.imshow(grid.values, aspect="auto", origin="lower", cmap="viridis",
                   extent=[0, 24, 0.5, 12.5])
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Month")
    ax.set_yticks(range(1, 13))
    ax.set_title("Capacity factor by month and hour (2 MW turbine)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Capacity factor")
    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS, "cf_heatmap.png"), dpi=130)


def turbine_compare(df):
    v = df["wind_speed_100m_ms"].values
    names, gross_cfs, net_cfs, net_aeps = [], [], [], []
    for name, params in TURBINES.items():
        g_aep, n_aep, g_cf, n_cf = yield_for_turbine(v, params)
        names.append(name)
        gross_cfs.append(g_cf)
        net_cfs.append(n_cf)
        net_aeps.append(n_aep)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    y = np.arange(len(names))

    axes[0].barh(y, gross_cfs, color=config.ACCENT, alpha=0.5, label="gross")
    axes[0].barh(y, net_cfs, color=config.ACCENT, label="net (after losses)")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names)
    axes[0].set_xlabel("Capacity factor")
    axes[0].set_title("Capacity factor by turbine class")
    axes[0].legend()
    axes[0].grid(alpha=0.3, axis="x")

    axes[1].barh(y, np.array(net_aeps) / 1000.0, color=config.HIGHLIGHT)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([])
    axes[1].set_xlabel("Net AEP  [GWh/year per turbine]")
    axes[1].set_title("Net annual energy")
    axes[1].grid(alpha=0.3, axis="x")

    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS, "turbine_compare.png"), dpi=130)


def summary(df):
    v = df["wind_speed_100m_ms"].values
    loss = total_loss_factor()

    print("=" * 60)
    print("Energy yield summary (per turbine, one average year)")
    print("=" * 60)
    print(f"Combined loss factor   : {loss:.1%}")
    print("-" * 60)
    print(f"{'turbine':<32}{'net CF':>9}{'net AEP':>14}")
    print(f"{'':<32}{'':>9}{'[MWh/yr]':>14}")
    for name, params in TURBINES.items():
        g_aep, n_aep, g_cf, n_cf = yield_for_turbine(v, params)
        print(f"{name:<32}{n_cf:>8.1%}{n_aep:>13,.0f}")


def main():
    os.makedirs(config.RESULTS, exist_ok=True)
    df = load()
    summary(df)
    cf_heatmap(df)
    turbine_compare(df)
    print(f"Saved figures to {os.path.abspath(config.RESULTS)}")


if __name__ == "__main__":
    main()
