"""Forecast wind power several hours ahead for La Guajira from historical
weather.

The model is a from-scratch **ridge regression** (closed-form, NumPy only) with
physics-informed features: because turbine power scales with the cube of wind
speed, polynomial terms of the wind speed let a linear model capture the
strongly non-linear power response. It is compared against a naive persistence
baseline and a plain linear model on a chronological hold-out
(train: 2021-2022, test: 2023).

The forecast horizon is set by HORIZON (hours). At several hours ahead the
naive persistence baseline degrades and the learned model adds clear value.

Outputs (saved to ../results):
  - forecast_timeseries.png   actual vs predicted over a test window
  - feature_importance.png    standardized coefficient magnitudes
  - actual_vs_pred.png        hold-out scatter

Run:  python src/model.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from power_curve import power_output, RATED_POWER

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data", "la_guajira_wind.csv")
RESULTS = os.path.join(HERE, "..", "results")
ACCENT = "#1f4e79"
HORIZON = 6  # forecast horizon in hours


class RidgeRegressor:
    """L2-regularized linear regression solved in closed form.

    Features are standardized internally; the intercept is not penalized.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X, y):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-9
        Xs = np.c_[np.ones(len(X)), (X - self.mean_) / self.std_]
        n = Xs.shape[1]
        penalty = self.alpha * np.eye(n)
        penalty[0, 0] = 0.0  # don't regularize the intercept
        self.coef_ = np.linalg.solve(Xs.T @ Xs + penalty, Xs.T @ y)
        return self

    def predict(self, X):
        Xs = np.c_[np.ones(len(X)), (X - self.mean_) / self.std_]
        return Xs @ self.coef_


def build_features(df: pd.DataFrame):
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["power_mw"] = power_output(df["wind_speed_100m_ms"].values)
    df["target"] = df["power_mw"].shift(-HORIZON)  # power HORIZON hours ahead

    for lag in (1, 2, 3, 6, 24):
        df[f"wind_lag{lag}"] = df["wind_speed_100m_ms"].shift(lag - 1)
        df[f"power_lag{lag}"] = df["power_mw"].shift(lag - 1)
    df["wind_roll3"] = df["wind_speed_100m_ms"].rolling(3).mean()
    # Physics-informed non-linear terms (power ~ wind^3).
    df["wind_sq"] = df["wind_speed_100m_ms"] ** 2
    df["wind_cube"] = df["wind_speed_100m_ms"] ** 3
    df["temp"] = df["temperature_c"]

    next_time = df["timestamp"].shift(-HORIZON)
    hour, month = next_time.dt.hour, next_time.dt.month
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)

    lag_cols = [f"wind_lag{l}" for l in (1, 2, 3, 6, 24)] + \
               [f"power_lag{l}" for l in (1, 2, 3, 6, 24)]
    linear_cols = lag_cols + ["wind_roll3", "temp",
                              "hour_sin", "hour_cos", "month_sin", "month_cos"]
    physics_cols = linear_cols + ["wind_sq", "wind_cube"]

    df = df.dropna(subset=physics_cols + ["target"]).reset_index(drop=True)
    return df, linear_cols, physics_cols


def metrics(name, y_true, y_pred):
    err = y_pred - y_true
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err**2))
    ss_res = np.sum(err**2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    print(f"{name:<28} MAE={mae:.4f} MW  RMSE={rmse:.4f} MW  R2={r2:.3f}")
    return mae, rmse, r2


def main():
    os.makedirs(RESULTS, exist_ok=True)
    df = pd.read_csv(DATA, parse_dates=["timestamp"])
    df, linear_cols, physics_cols = build_features(df)

    split = df["timestamp"] < "2023-01-01"
    train, test = df[split], df[~split]
    y_train = train["target"].values
    y_test = test["target"].values
    print(f"Forecast horizon: {HORIZON} h")
    print(f"Train rows: {len(train):,}   Test rows: {len(test):,}")
    print("-" * 70)

    metrics("Persistence baseline", y_test, test["power_lag1"].values)

    lin = RidgeRegressor(alpha=1.0).fit(train[linear_cols].values, y_train)
    metrics("Ridge (lag/time features)", y_test, lin.predict(test[linear_cols].values))

    model = RidgeRegressor(alpha=1.0).fit(train[physics_cols].values, y_train)
    pred = np.clip(model.predict(test[physics_cols].values), 0.0, RATED_POWER)
    metrics("Ridge + physics features", y_test, pred)
    print("-" * 70)

    # --- Plots -----------------------------------------------------------
    window = slice(0, 24 * 7)  # first week of the test set
    t = test["timestamp"].values[window]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(t, y_test[window], label="Actual", color="black", lw=1.5)
    ax.plot(t, pred[window], label="Forecast (ridge + physics)",
            color="#c0392b", lw=1.5)
    ax.set_ylabel("Power  [MW]")
    ax.set_title(f"{HORIZON}-hour-ahead wind-power forecast vs actual "
                 f"(first test week, 2023)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "forecast_timeseries.png"), dpi=130)

    # Importance via standardized coefficient magnitudes (skip intercept).
    importance = pd.Series(np.abs(model.coef_[1:]), index=physics_cols)
    importance = importance.sort_values()
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh(importance.index, importance.values, color=ACCENT)
    ax.set_xlabel("Standardized coefficient magnitude")
    ax.set_title("Feature importance (ridge + physics)")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "feature_importance.png"), dpi=130)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y_test, pred, s=4, alpha=0.2, color=ACCENT)
    lims = [0, max(y_test.max(), pred.max())]
    ax.plot(lims, lims, color="#c0392b", lw=1.5)
    ax.set_xlabel("Actual power  [MW]")
    ax.set_ylabel("Predicted power  [MW]")
    ax.set_title("Hold-out: actual vs predicted")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "actual_vs_pred.png"), dpi=130)

    print(f"Saved figures to {os.path.abspath(RESULTS)}")


if __name__ == "__main__":
    main()
