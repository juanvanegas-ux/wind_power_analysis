"""Wind shear between 10 m and 100 m for La Guajira.

The dataset has wind speed at two heights and the first version of this repo
only ever used the 100 m one. that felt like a waste, the 10 m column lets you
measure the wind shear directly instead of guessing it.

The power law model is

    V(z) = V_ref * (z / z_ref) ** alpha

and with two heights you can just solve for the exponent alpha at every hour

    alpha = ln(V_high / V_low) / ln(z_high / z_low)

alpha tells you how fast the wind speeds up as you go higher. a small alpha
(~0.1) is a smooth surface like the open sea, a big one (~0.3 or more) means a
rough surface or a very stable night time boundary layer. it is the number you
need to push a measurement up to hub height, and it usually swings a lot between
day and night.

Outputs (saved to ../results):
  - wind_shear.png   alpha distribution + how it changes through the day

Run:  python src/wind_shear.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config


def shear_exponent(v_low, v_high, z_low=config.HEIGHT_LOW, z_high=config.HEIGHT_HIGH):
    """Power law exponent alpha from two wind speeds at two heights.

    array safe. hours where either speed is basically calm are dropped to
    nan because the log ratio gets noisy and meaningless down there.
    """
    v_low = np.asarray(v_low, dtype=float)
    v_high = np.asarray(v_high, dtype=float)
    good = (v_low > 0.5) & (v_high > 0.5)
    alpha = np.full(v_low.shape, np.nan)
    alpha[good] = np.log(v_high[good] / v_low[good]) / np.log(z_high / z_low)
    return alpha


def extrapolate(v_ref, alpha, z_ref, z_target):
    """Push a wind speed from z_ref up (or down) to z_target with the power law."""
    return np.asarray(v_ref, dtype=float) * (z_target / z_ref) ** alpha


def load() -> pd.DataFrame:
    df = pd.read_csv(config.DATA, parse_dates=["timestamp"])
    df = df.dropna(subset=["wind_speed_10m_ms", "wind_speed_100m_ms"])
    df = df.reset_index(drop=True)
    df["hour"] = df["timestamp"].dt.hour
    df["alpha"] = shear_exponent(df["wind_speed_10m_ms"].values,
                                 df["wind_speed_100m_ms"].values)
    return df


def check_extrapolation(df):
    """Sanity check: go from 10 m up to 100 m with the median alpha and see
    how close we land to the real measured 100 m wind. if the power law was
    useless this error would be large."""
    a_med = np.nanmedian(df["alpha"].values)
    pred_100 = extrapolate(df["wind_speed_10m_ms"].values, a_med,
                           config.HEIGHT_LOW, config.HEIGHT_HIGH)
    real_100 = df["wind_speed_100m_ms"].values
    err = pred_100 - real_100
    mae = np.nanmean(np.abs(err))
    bias = np.nanmean(err)
    return a_med, mae, bias


def plot(df):
    alpha = df["alpha"].dropna().values
    by_hour = df.groupby("hour")["alpha"].median()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    axes[0].hist(alpha, bins=50, color=config.ACCENT, alpha=0.7)
    axes[0].axvline(np.median(alpha), color=config.HIGHLIGHT, lw=2,
                    label=f"median = {np.median(alpha):.3f}")
    axes[0].axvline(1.0 / 7.0, color="black", ls="--", lw=1.2,
                    label="1/7 rule (0.143)")
    axes[0].set_xlabel("Shear exponent alpha  [-]")
    axes[0].set_ylabel("Hours")
    axes[0].set_title("Distribution of the shear exponent")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(by_hour.index, by_hour.values, marker="o", color=config.ACCENT)
    axes[1].axhline(1.0 / 7.0, color="black", ls="--", lw=1.2)
    axes[1].set_xlabel("Hour of day")
    axes[1].set_ylabel("Median alpha  [-]")
    axes[1].set_title("Shear through the day")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS, "wind_shear.png"), dpi=130)


def summary(df):
    alpha = df["alpha"].dropna().values
    a_med, mae, bias = check_extrapolation(df)

    print("=" * 52)
    print("Wind shear summary (10 m vs 100 m)")
    print("=" * 52)
    print(f"Hours with valid alpha : {len(alpha):,}")
    print(f"Median alpha           : {np.median(alpha):.3f}")
    print(f"Mean alpha             : {np.mean(alpha):.3f}")
    print(f"10th to 90th pct alpha : {np.percentile(alpha, 10):.3f} "
          f"to {np.percentile(alpha, 90):.3f}")
    print(f"1/7 power law reference : {1.0 / 7.0:.3f}")
    print("-" * 52)
    print("Extrapolating 10 m -> 100 m with the median alpha:")
    print(f"  MAE vs measured 100 m : {mae:.2f} m/s")
    print(f"  mean bias             : {bias:+.2f} m/s")


def main():
    os.makedirs(config.RESULTS, exist_ok=True)
    df = load()
    summary(df)
    plot(df)
    print(f"Saved figure to {os.path.abspath(config.RESULTS)}")


if __name__ == "__main__":
    main()
