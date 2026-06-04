"""Side by side comparison of the two wind data sources for La Guajira:
Open Meteo (ERA5) and NASA POWER (MERRA-2 satellite).

both describe the same 3 years at the same point, so any difference is the
models disagreeing, not the weather. that disagreement is the whole point: it
tells you how much the headline resource numbers depend on which dataset you
picked, which is exactly the uncertainty a real project has to live with before
a met mast goes up.

To keep it a fair fight everything is computed the same way for both, same
Weibull fit, same power curve, same air density, on the overlapping hours only.

Outputs (saved to ../results):
  - source_comparison.png   six panel side by side
And prints a resource table for each source plus the agreement metrics.

Run:  python src/compare_sources.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import config
from power_curve import power_output, RATED_POWER

HOURS_PER_YEAR = 8760.0

SOURCES = {
    "Open Meteo (ERA5)": os.path.join(os.path.dirname(__file__), "..", "data",
                                      "la_guajira_wind.csv"),
    "NASA POWER (MERRA-2)": os.path.join(os.path.dirname(__file__), "..", "data",
                                         "la_guajira_wind_nasa.csv"),
}
COLORS = {
    "Open Meteo (ERA5)": config.HIGHLIGHT,
    "NASA POWER (MERRA-2)": config.ACCENT,
}


def load_source(path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.dropna(subset=["wind_speed_100m_ms"])
    return df[["timestamp", "wind_speed_100m_ms"]].copy()


def align(frames: dict) -> pd.DataFrame:
    """Inner join every source on the timestamp so we only compare hours that
    exist in all of them."""
    merged = None
    for name, df in frames.items():
        col = df.rename(columns={"wind_speed_100m_ms": name})
        merged = col if merged is None else pd.merge(merged, col, on="timestamp")
    merged["hour"] = merged["timestamp"].dt.hour
    merged["month"] = merged["timestamp"].dt.month
    return merged


def resource_stats(v) -> dict:
    """Same recipe analysis.py uses, so the numbers line up with the rest."""
    k, _, c = stats.weibull_min.fit(v, floc=0)
    power_density = 0.5 * config.AIR_DENSITY * (v ** 3).mean()
    cf = power_output(v).mean() / RATED_POWER
    aep = power_output(v).mean() * HOURS_PER_YEAR  # MWh/yr per 2 MW turbine
    return {
        "mean": v.mean(),
        "k": k,
        "c": c,
        "power_density": power_density,
        "cf": cf,
        "aep": aep,
    }


def agreement(a, b) -> dict:
    diff = a - b
    return {
        "bias": diff.mean(),
        "mae": np.abs(diff).mean(),
        "rmse": np.sqrt((diff ** 2).mean()),
        "corr": np.corrcoef(a, b)[0, 1],
    }


def print_table(merged, names):
    print("=" * 70)
    print("Resource comparison on the overlapping hours")
    print("=" * 70)
    print(f"Overlapping hours: {len(merged):,}")
    print("-" * 70)
    header = f"{'metric':<26}" + "".join(f"{n:>22}" for n in names)
    print(header)
    stats_by = {n: resource_stats(merged[n].values) for n in names}
    rows = [
        ("mean wind 100 m [m/s]", "mean", "{:.2f}"),
        ("Weibull k [-]", "k", "{:.2f}"),
        ("Weibull c [m/s]", "c", "{:.2f}"),
        ("power density [W/m2]", "power_density", "{:.0f}"),
        ("capacity factor [%]", "cf", "{:.1%}"),
        ("AEP per 2 MW [MWh/yr]", "aep", "{:,.0f}"),
    ]
    for label, key, fmt in rows:
        line = f"{label:<26}"
        for n in names:
            line += f"{fmt.format(stats_by[n][key]):>22}"
        print(line)

    print("-" * 70)
    a = merged[names[0]].values
    b = merged[names[1]].values
    ag = agreement(a, b)
    print(f"Agreement ({names[0]} minus {names[1]}):")
    print(f"  bias                : {ag['bias']:+.2f} m/s")
    print(f"  mean abs difference : {ag['mae']:.2f} m/s")
    print(f"  RMSE                : {ag['rmse']:.2f} m/s")
    print(f"  correlation         : {ag['corr']:.3f}")
    return stats_by


def plot(merged, names, stats_by):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))

    # 1) distribution overlay with fitted Weibull
    ax = axes[0, 0]
    xmax = max(merged[n].max() for n in names)
    x = np.linspace(0, xmax, 200)
    for n in names:
        v = merged[n].values
        ax.hist(v, bins=45, density=True, alpha=0.35, color=COLORS[n], label=n)
        k, c = stats_by[n]["k"], stats_by[n]["c"]
        ax.plot(x, stats.weibull_min.pdf(x, k, 0, c), color=COLORS[n], lw=2)
    ax.set_xlabel("Wind speed 100 m  [m/s]")
    ax.set_ylabel("Density")
    ax.set_title("Distribution + Weibull fit")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 2) hourly scatter vs 1:1
    ax = axes[0, 1]
    a, b = merged[names[0]].values, merged[names[1]].values
    ax.scatter(b, a, s=3, alpha=0.06, color=config.ACCENT)
    lims = [0, max(a.max(), b.max())]
    ax.plot(lims, lims, color="black", lw=1.2, label="1:1")
    ax.set_xlabel(names[1])
    ax.set_ylabel(names[0])
    ax.set_title("Hourly agreement")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 3) monthly means
    ax = axes[0, 2]
    by_month = merged.groupby("month")[names].mean()
    for n in names:
        ax.plot(by_month.index, by_month[n], marker="o", color=COLORS[n],
                label=n)
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean wind  [m/s]")
    ax.set_title("Seasonal cycle")
    ax.set_xticks(range(1, 13))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 4) diurnal means
    ax = axes[1, 0]
    by_hour = merged.groupby("hour")[names].mean()
    for n in names:
        ax.plot(by_hour.index, by_hour[n], marker="o", color=COLORS[n], label=n)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean wind  [m/s]")
    ax.set_title("Diurnal cycle")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 5) bias by wind speed regime (does one source run hotter at high wind?)
    ax = axes[1, 1]
    edges = np.arange(0, 22, 2.0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.digitize(b, edges) - 1
    bias = [np.nan if (idx == j).sum() == 0
            else (a[idx == j] - b[idx == j]).mean()
            for j in range(len(centers))]
    ax.axhline(0, color="black", lw=0.8)
    ax.plot(centers, bias, marker="o", color=config.ACCENT)
    ax.set_xlabel(f"{names[1]} wind speed  [m/s]")
    ax.set_ylabel(f"mean ({names[0]} - {names[1]})  [m/s]")
    ax.set_title("Bias across the wind range")
    ax.grid(alpha=0.3)

    # 6) capacity factor bars
    ax = axes[1, 2]
    cfs = [stats_by[n]["cf"] * 100 for n in names]
    ax.bar(range(len(names)), cfs, color=[COLORS[n] for n in names], alpha=0.85)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.split(" (")[0] for n in names])
    ax.set_ylabel("Capacity factor  [%]")
    ax.set_title("Capacity factor (2 MW turbine)")
    ax.set_ylim(0, max(cfs) * 1.18)
    for i, v in enumerate(cfs):
        ax.text(i, v + 1.0, f"{v:.1f}%", ha="center", fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("La Guajira wind: Open Meteo (ERA5) vs NASA POWER (MERRA-2)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS, "source_comparison.png"), dpi=130)


def main():
    os.makedirs(config.RESULTS, exist_ok=True)
    frames = {}
    for name, path in SOURCES.items():
        if not os.path.exists(path):
            print(f"missing {name} at {path}, run its fetch script first")
            return
        frames[name] = load_source(path)

    names = list(SOURCES.keys())
    merged = align(frames)
    stats_by = print_table(merged, names)
    plot(merged, names, stats_by)
    print(f"Saved figure to {os.path.abspath(config.RESULTS)}")


if __name__ == "__main__":
    main()
