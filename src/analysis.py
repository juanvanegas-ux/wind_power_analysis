"""Exploratory wind resource analysis for La Guajira, Colombia.

Produces (saved to ../results):
  - weibull_fit.png       wind speed distribution + fitted Weibull
  - patterns.png          diurnal and monthly wind speed patterns
  - wind_rose.png         directional frequency of the wind
  - seasonal_weibull.png  how the Weibull k and c move through the year
  - energy_rose.png       direction weighted by energy, not just by frequency
  - extreme_gust.png      Gumbel fit to the daily maxima for return wind speeds

And prints a resource summary (mean speed, Weibull params, power density both
hardcoded and from the measured temperature, a rough wind power class, and a
50 year extreme wind estimate).

Run:  python src/analysis.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import config
from power_curve import power_output, RATED_POWER

ACCENT = config.ACCENT
HIGHLIGHT = config.HIGHLIGHT


def load() -> pd.DataFrame:
    df = pd.read_csv(config.DATA, parse_dates=["timestamp"])
    df = df.dropna(subset=["wind_speed_100m_ms"]).reset_index(drop=True)
    df["hour"] = df["timestamp"].dt.hour
    df["month"] = df["timestamp"].dt.month
    df["date"] = df["timestamp"].dt.date
    return df


def weibull_plot(df):
    v = df["wind_speed_100m_ms"].values
    # Weibull fit with location fixed at 0 (standard for wind speed).
    k, loc, scale = stats.weibull_min.fit(v, floc=0)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(v, bins=40, density=True, color=ACCENT, alpha=0.55,
            label="Observed")
    x = np.linspace(0, v.max(), 200)
    ax.plot(x, stats.weibull_min.pdf(x, k, loc, scale), color=HIGHLIGHT, lw=2,
            label=f"Weibull (k={k:.2f}, c={scale:.2f} m/s)")
    ax.set_xlabel("Wind speed at 100 m  [m/s]")
    ax.set_ylabel("Probability density")
    ax.set_title("La Guajira wind speed distribution")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS, "weibull_fit.png"), dpi=130)
    return k, scale


def patterns_plot(df):
    by_hour = df.groupby("hour")["wind_speed_100m_ms"].mean()
    by_month = df.groupby("month")["wind_speed_100m_ms"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(by_hour.index, by_hour.values, marker="o", color=ACCENT)
    axes[0].set_xlabel("Hour of day")
    axes[0].set_ylabel("Mean wind speed  [m/s]")
    axes[0].set_title("Diurnal pattern")
    axes[0].grid(alpha=0.3)

    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    axes[1].bar(by_month.index, by_month.values, color=ACCENT, alpha=0.8)
    axes[1].set_xticks(range(1, 13))
    axes[1].set_xticklabels(months, rotation=45)
    axes[1].set_ylabel("Mean wind speed  [m/s]")
    axes[1].set_title("Seasonal (monthly) pattern")
    axes[1].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS, "patterns.png"), dpi=130)


def _direction_fractions(dirs, weights=None):
    bins = np.arange(0, 361, 30)
    if weights is None:
        counts, _ = np.histogram(dirs % 360, bins=bins)
    else:
        counts, _ = np.histogram(dirs % 360, bins=bins, weights=weights)
    frac = counts / counts.sum()
    centers = np.radians(bins[:-1] + 15)
    return centers, frac


def wind_rose(df):
    dirs = df["wind_direction_100m_deg"].dropna().values
    centers, frac = _direction_fractions(dirs)

    fig = plt.figure(figsize=(5.5, 5.5))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.bar(centers, frac, width=np.radians(28), color=ACCENT, alpha=0.8,
           edgecolor="white")
    ax.set_title("Wind direction frequency (100 m)", pad=18)
    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS, "wind_rose.png"), dpi=130)


def energy_rose(df):
    """Same directions, but weight each hour by v^3 instead of counting hours.

    a calm breeze from the south and a howling gale from the east count the
    same on a normal wind rose, which hides where the energy actually comes
    from. weighting by the cube fixes that.
    """
    sub = df.dropna(subset=["wind_direction_100m_deg", "wind_speed_100m_ms"])
    dirs = sub["wind_direction_100m_deg"].values
    energy = sub["wind_speed_100m_ms"].values ** 3

    c_freq, f_freq = _direction_fractions(dirs)
    c_en, f_en = _direction_fractions(dirs, weights=energy)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2),
                             subplot_kw={"projection": "polar"})
    for ax, (centers, frac, title) in zip(
        axes,
        [(c_freq, f_freq, "By frequency (hours)"),
         (c_en, f_en, "By energy (v^3 weighted)")],
    ):
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.bar(centers, frac, width=np.radians(28), color=ACCENT, alpha=0.8,
               edgecolor="white")
        ax.set_title(title, pad=16)

    fig.suptitle("Where the wind comes from vs where the energy comes from")
    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS, "energy_rose.png"), dpi=130)


def seasonal_weibull(df):
    """Fit a Weibull per month and show how the shape k and scale c drift
    through the year. the trade wind season should stand out as a fatter c
    and a tighter (higher k) distribution."""
    ks, cs, months = [], [], []
    for m in range(1, 13):
        v = df.loc[df["month"] == m, "wind_speed_100m_ms"].values
        if len(v) < 50:
            continue
        k, _, c = stats.weibull_min.fit(v, floc=0)
        ks.append(k)
        cs.append(c)
        months.append(m)

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax2 = ax1.twinx()
    l1 = ax1.plot(months, cs, marker="o", color=ACCENT, label="scale c [m/s]")
    l2 = ax2.plot(months, ks, marker="s", color=HIGHLIGHT, label="shape k [-]")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Weibull scale c  [m/s]", color=ACCENT)
    ax2.set_ylabel("Weibull shape k  [-]", color=HIGHLIGHT)
    ax1.set_xticks(range(1, 13))
    ax1.set_title("Seasonal Weibull parameters")
    ax1.grid(alpha=0.3)
    lines = l1 + l2
    ax1.legend(lines, [ln.get_label() for ln in lines], loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS, "seasonal_weibull.png"), dpi=130)


def extreme_value(df):
    """Gumbel fit to the daily maximum wind speed, used to read off the wind
    speed you would expect once every N years.

    block maxima method: take the strongest hour of each day as one sample,
    fit a Gumbel, then invert it for the 1, 10 and 50 year return levels. this
    is the kind of number that drives the turbine class survival load.
    """
    daily_max = df.groupby("date")["wind_speed_100m_ms"].max().values
    loc, scale = stats.gumbel_r.fit(daily_max)

    blocks_per_year = 365.25  # daily blocks
    return_periods = np.array([1, 5, 10, 50])
    # return level x_T = loc - scale * ln(-ln(1 - 1/(T*blocks_per_year)))
    p = 1.0 - 1.0 / (return_periods * blocks_per_year)
    levels = loc - scale * np.log(-np.log(p))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(daily_max, bins=40, density=True, color=ACCENT, alpha=0.55,
            label="Daily max wind")
    x = np.linspace(daily_max.min(), levels[-1] * 1.05, 200)
    ax.plot(x, stats.gumbel_r.pdf(x, loc, scale), color=HIGHLIGHT, lw=2,
            label="Gumbel fit")
    for T, lv in zip(return_periods, levels):
        ax.axvline(lv, color="black", ls="--", lw=1.0, alpha=0.6)
        ax.text(lv, ax.get_ylim()[1] * 0.9, f"{T}y", rotation=90,
                va="top", ha="right", fontsize=8)
    ax.set_xlabel("Wind speed at 100 m  [m/s]")
    ax.set_ylabel("Probability density")
    ax.set_title("Extreme wind: Gumbel fit to daily maxima")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS, "extreme_gust.png"), dpi=130)
    return dict(zip(return_periods.tolist(), levels.tolist()))


def power_class(power_density):
    """Very rough NREL style wind power class from the power density.

    the official classes are defined at 50 m and we are at 100 m, so treat
    this as a label not a certificate. anything past ~600 W/m2 is about as
    good as onshore wind gets.
    """
    edges = [100, 150, 200, 250, 300, 400, 1000]
    for i, e in enumerate(edges, start=1):
        if power_density < e:
            return i
    return 7


def summary(df, k, scale, returns):
    v = df["wind_speed_100m_ms"].values
    mean_speed = v.mean()

    # density two ways: the flat assumption, and from the measured temperature
    rho_fixed = config.AIR_DENSITY
    rho_temp = config.air_density_from_temp(df["temperature_c"].values)
    rho_temp_mean = float(np.nanmean(rho_temp))

    pd_fixed = 0.5 * rho_fixed * (v ** 3).mean()
    pd_temp = 0.5 * rho_temp_mean * (v ** 3).mean()

    power_mw = power_output(v)
    capacity_factor = power_mw.mean() / RATED_POWER

    print("=" * 56)
    print("La Guajira wind resource summary (2021-2023)")
    print("=" * 56)
    print(f"Hours analysed             : {len(df):,}")
    print(f"Mean wind speed (100 m)    : {mean_speed:.2f} m/s")
    print(f"Weibull shape k            : {k:.2f}")
    print(f"Weibull scale c            : {scale:.2f} m/s")
    print(f"Air density (fixed 1.18)   : {rho_fixed:.3f} kg/m^3")
    print(f"Air density (from temp)    : {rho_temp_mean:.3f} kg/m^3")
    print(f"Power density (fixed rho)  : {pd_fixed:.0f} W/m^2")
    print(f"Power density (temp rho)   : {pd_temp:.0f} W/m^2")
    print(f"Rough wind power class      : {power_class(pd_temp)} (of 7)")
    print(f"Capacity factor (2 MW)     : {capacity_factor:.1%}")
    print("-" * 56)
    print("Extreme wind (Gumbel on daily maxima):")
    for T, lv in returns.items():
        print(f"  {T:>2}-year return wind speed : {lv:.1f} m/s")


def main():
    os.makedirs(config.RESULTS, exist_ok=True)
    df = load()
    k, scale = weibull_plot(df)
    patterns_plot(df)
    wind_rose(df)
    energy_rose(df)
    seasonal_weibull(df)
    returns = extreme_value(df)
    summary(df, k, scale, returns)
    print(f"Saved figures to {os.path.abspath(config.RESULTS)}")


if __name__ == "__main__":
    main()
