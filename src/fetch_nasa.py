"""Download hourly wind data for La Guajira from NASA POWER instead of Open
Meteo, as a second, independent source.

NASA POWER (Prediction Of Worldwide Energy Resources) serves meteorology
derived from the MERRA-2 reanalysis, which assimilates NASA satellite
observations. it is free and needs no API key, same as Open Meteo, but it comes
from a different model so it is a genuine cross check on the resource numbers.

One catch: POWER gives wind at 10 m and 50 m, not at 100 m. that is fine here,
because with two heights you can measure the power law shear exponent and push
the wind up to the 100 m hub height yourself (the same physics as wind_shear.py).
POWER also gives the real surface pressure, so the air density does not have to
be assumed at sea level.

The output CSV uses the exact same columns as the Open Meteo file, so the rest
of the pipeline (analysis, energy_yield, model) runs on it unchanged.

Run:  python src/fetch_nasa.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

import config

POWER_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"
FILL = -999.0  # POWER missing value flag
OUT = os.path.join(os.path.dirname(__file__), "..", "data",
                   "la_guajira_wind_nasa.csv")

# POWER heights for the two wind levels it reports
H_LOW = 10.0
H_MID = 50.0


def fetch() -> pd.DataFrame:
    params = {
        "parameters": "WS10M,WS50M,WD10M,T2M,PS",
        "community": "RE",                 # renewable energy community
        "longitude": config.LONGITUDE,
        "latitude": config.LATITUDE,
        "start": config.START_DATE.replace("-", ""),  # YYYYMMDD
        "end": config.END_DATE.replace("-", ""),
        "format": "JSON",
        "time-standard": "UTC",
    }
    print(f"Requesting NASA POWER {config.START_DATE} to {config.END_DATE} "
          f"for ({config.LATITUDE}, {config.LONGITUDE}) ...")
    r = requests.get(POWER_URL, params=params, timeout=180)
    r.raise_for_status()
    par = r.json()["properties"]["parameter"]

    # each parameter is a dict keyed by YYYYMMDDHH
    df = pd.DataFrame(par)
    df.index = pd.to_datetime(df.index, format="%Y%m%d%H", utc=True)
    df = df.sort_index()

    # POWER timestamps are UTC, Colombia is UTC-5 with no daylight saving, so
    # convert so the diurnal pattern lines up with local time
    df.index = df.index.tz_convert("America/Bogota").tz_localize(None)
    df = df.replace(FILL, np.nan)
    return df


def to_pipeline_schema(df) -> pd.DataFrame:
    """Build the same columns as the Open Meteo file, extrapolating to 100 m."""
    ws10 = df["WS10M"].values
    ws50 = df["WS50M"].values

    # per hour shear exponent from the two heights: alpha = ln(v50/v10)/ln(5)
    good = (ws10 > 0.5) & (ws50 > 0.5)
    alpha = np.full(ws10.shape, np.nan)
    alpha[good] = np.log(ws50[good] / ws10[good]) / np.log(H_MID / H_LOW)

    # fall back to the median alpha where a level was calm/missing
    alpha_fill = np.nanmedian(alpha)
    alpha = np.where(np.isnan(alpha), alpha_fill, alpha)

    # push 50 m up to 100 m with the local power law
    ws100 = ws50 * (config.HUB_HEIGHT / H_MID) ** alpha

    out = pd.DataFrame({
        "timestamp": df.index,
        "wind_speed_100m_ms": np.round(ws100, 2),
        "wind_speed_10m_ms": np.round(ws10, 2),
        "wind_direction_100m_deg": np.round(df["WD10M"].values).astype("float"),
        "temperature_c": df["T2M"].values,
        # extras POWER gives us that Open Meteo did not
        "wind_speed_50m_ms": np.round(ws50, 2),
        "pressure_kpa": df["PS"].values,
        "shear_alpha": np.round(alpha, 3),
    })
    return out


def compare_with_openmeteo(nasa):
    """If the Open Meteo snapshot is around, print a quick side by side so the
    two independent sources can be sanity checked against each other."""
    if not os.path.exists(config.DATA):
        return
    om = pd.read_csv(config.DATA, parse_dates=["timestamp"])
    print("-" * 56)
    print("Cross check vs Open Meteo (ERA5) on the overlap:")
    merged = pd.merge(
        nasa[["timestamp", "wind_speed_100m_ms"]],
        om[["timestamp", "wind_speed_100m_ms"]],
        on="timestamp", suffixes=("_nasa", "_om"),
    ).dropna()
    if merged.empty:
        print("  no overlapping timestamps")
        return
    a = merged["wind_speed_100m_ms_nasa"].values
    b = merged["wind_speed_100m_ms_om"].values
    print(f"  overlapping hours      : {len(merged):,}")
    print(f"  mean speed NASA POWER  : {a.mean():.2f} m/s")
    print(f"  mean speed Open Meteo  : {b.mean():.2f} m/s")
    print(f"  mean abs difference    : {np.mean(np.abs(a - b)):.2f} m/s")
    print(f"  correlation            : {np.corrcoef(a, b)[0, 1]:.3f}")

    # a picture of the agreement: hourly scatter + monthly means
    merged["month"] = merged["timestamp"].dt.to_period("M").dt.to_timestamp()
    monthly = merged.groupby("month")[
        ["wind_speed_100m_ms_nasa", "wind_speed_100m_ms_om"]].mean()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    axes[0].scatter(b, a, s=3, alpha=0.08, color=config.ACCENT)
    lims = [0, max(a.max(), b.max())]
    axes[0].plot(lims, lims, color=config.HIGHLIGHT, lw=1.5, label="1:1")
    axes[0].set_xlabel("Open Meteo / ERA5  [m/s]")
    axes[0].set_ylabel("NASA POWER / MERRA-2  [m/s]")
    axes[0].set_title("Hourly 100 m wind, two sources")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(monthly.index, monthly["wind_speed_100m_ms_nasa"],
                 marker="o", color=config.ACCENT, label="NASA POWER")
    axes[1].plot(monthly.index, monthly["wind_speed_100m_ms_om"],
                 marker="s", color=config.HIGHLIGHT, label="Open Meteo")
    axes[1].set_ylabel("Monthly mean wind  [m/s]")
    axes[1].set_title("Monthly means agree on the shape")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    os.makedirs(config.RESULTS, exist_ok=True)
    fig.savefig(os.path.join(config.RESULTS, "source_compare.png"), dpi=130)


def main():
    df = fetch()
    out = to_pipeline_schema(df)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False)

    print(f"Saved {len(out):,} hourly rows -> {os.path.abspath(OUT)}")
    print(f"Mean wind speed at 100 m (extrapolated): "
          f"{out['wind_speed_100m_ms'].mean():.2f} m/s")
    print(f"Median shear alpha (10 m -> 50 m)      : "
          f"{out['shear_alpha'].median():.3f}")
    compare_with_openmeteo(out)
    print(out.head())


if __name__ == "__main__":
    main()
