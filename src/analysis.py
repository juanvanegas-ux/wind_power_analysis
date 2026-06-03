"""Exploratory wind-resource analysis for La Guajira, Colombia.

Produces (saved to ../results):
  - weibull_fit.png     wind-speed distribution + fitted Weibull
  - patterns.png        diurnal and monthly wind-speed patterns
  - wind_rose.png       directional distribution of the wind
And prints a short resource summary (mean speed, Weibull params,
power density, estimated capacity factor).

Run:  python src/analysis.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from power_curve import power_output, RATED_POWER

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data", "la_guajira_wind.csv")
RESULTS = os.path.join(HERE, "..", "results")
AIR_DENSITY = 1.18  # kg/m^3, warm coastal climate

ACCENT = "#1f4e79"


def load() -> pd.DataFrame:
    df = pd.read_csv(DATA, parse_dates=["timestamp"])
    df = df.dropna(subset=["wind_speed_100m_ms"]).reset_index(drop=True)
    df["hour"] = df["timestamp"].dt.hour
    df["month"] = df["timestamp"].dt.month
    return df


def weibull_plot(df):
    v = df["wind_speed_100m_ms"].values
    # Weibull fit with location fixed at 0 (standard for wind speed).
    k, loc, scale = stats.weibull_min.fit(v, floc=0)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(v, bins=40, density=True, color=ACCENT, alpha=0.55,
            label="Observed")
    x = np.linspace(0, v.max(), 200)
    ax.plot(x, stats.weibull_min.pdf(x, k, loc, scale), color="#c0392b", lw=2,
            label=f"Weibull (k={k:.2f}, c={scale:.2f} m/s)")
    ax.set_xlabel("Wind speed at 100 m  [m/s]")
    ax.set_ylabel("Probability density")
    ax.set_title("La Guajira wind-speed distribution")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "weibull_fit.png"), dpi=130)
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
    fig.savefig(os.path.join(RESULTS, "patterns.png"), dpi=130)


def wind_rose(df):
    dirs = df["wind_direction_100m_deg"].dropna().values
    bins = np.arange(0, 361, 30)
    counts, _ = np.histogram(dirs % 360, bins=bins)
    frac = counts / counts.sum()
    centers = np.radians(bins[:-1] + 15)

    fig = plt.figure(figsize=(5.5, 5.5))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.bar(centers, frac, width=np.radians(28), color=ACCENT, alpha=0.8,
           edgecolor="white")
    ax.set_title("Wind direction frequency (100 m)", pad=18)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "wind_rose.png"), dpi=130)


def summary(df, k, scale):
    v = df["wind_speed_100m_ms"].values
    mean_speed = v.mean()
    power_density = 0.5 * AIR_DENSITY * (v**3).mean()  # W/m^2
    power_mw = power_output(v)
    capacity_factor = power_mw.mean() / RATED_POWER

    print("=" * 52)
    print("La Guajira wind-resource summary (2021-2023)")
    print("=" * 52)
    print(f"Hours analysed         : {len(df):,}")
    print(f"Mean wind speed (100 m): {mean_speed:.2f} m/s")
    print(f"Weibull shape k        : {k:.2f}")
    print(f"Weibull scale c        : {scale:.2f} m/s")
    print(f"Mean power density      : {power_density:.0f} W/m^2")
    print(f"Estimated capacity factor (2 MW turbine): {capacity_factor:.1%}")


def main():
    os.makedirs(RESULTS, exist_ok=True)
    df = load()
    k, scale = weibull_plot(df)
    patterns_plot(df)
    wind_rose(df)
    summary(df, k, scale)
    print(f"Saved figures to {os.path.abspath(RESULTS)}")


if __name__ == "__main__":
    main()
