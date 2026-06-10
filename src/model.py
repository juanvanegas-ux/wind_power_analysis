"""Forecast wind power a few hours ahead with a from scratch ridge regression
(closed form, NumPy only) and physics informed features (power ~ wind^3). scored
against persistence on a chronological hold out (train 2021-2022, test 2023),
with side studies on skill vs horizon, the ridge penalty, and error by wind speed.

Outputs (../results): forecast_timeseries, feature_importance, actual_vs_pred,
skill_vs_horizon, alpha_sweep, error_by_regime.
Run:  python src/model.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from power_curve import power_output, RATED_POWER

ACCENT = config.ACCENT
HIGHLIGHT = config.HIGHLIGHT
HORIZON = 6           # default forecast horizon in hours
SPLIT_DATE = "2023-01-01"


class RidgeRegressor:
    """L2 regularized linear regression solved in closed form.

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


def build_features(df: pd.DataFrame, horizon: int = HORIZON):
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["power_mw"] = power_output(df["wind_speed_100m_ms"].values)
    df["target"] = df["power_mw"].shift(-horizon)  # power `horizon` hours ahead

    for lag in (1, 2, 3, 6, 24):
        df[f"wind_lag{lag}"] = df["wind_speed_100m_ms"].shift(lag - 1)
        df[f"power_lag{lag}"] = df["power_mw"].shift(lag - 1)
    df["wind_roll3"] = df["wind_speed_100m_ms"].rolling(3).mean()
    # Physics informed non linear terms (power ~ wind^3).
    df["wind_sq"] = df["wind_speed_100m_ms"] ** 2
    df["wind_cube"] = df["wind_speed_100m_ms"] ** 3
    df["temp"] = df["temperature_c"]

    next_time = df["timestamp"].shift(-horizon)
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
    rmse = np.sqrt(np.mean(err ** 2))
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    print(f"{name:<28} MAE={mae:.4f} MW  RMSE={rmse:.4f} MW  R2={r2:.3f}")
    return mae, rmse, r2


def mae_of(y_true, y_pred):
    return float(np.mean(np.abs(y_pred - y_true)))


# --------------------------------------------------------------------------
# extra studies
# --------------------------------------------------------------------------

def skill_vs_horizon(df_raw, horizons=(1, 2, 3, 6, 12, 24), alpha=1.0):
    """Refit at each horizon and track the skill over persistence (1 - MAE ratio)."""
    pers_mae, model_mae, skill = [], [], []
    for h in horizons:
        df, _, physics_cols = build_features(df_raw.copy(), horizon=h)
        split = df["timestamp"] < SPLIT_DATE
        train, test = df[split], df[~split]
        y_tr, y_te = train["target"].values, test["target"].values

        m = RidgeRegressor(alpha=alpha).fit(train[physics_cols].values, y_tr)
        pred = np.clip(m.predict(test[physics_cols].values), 0.0, RATED_POWER)

        p_mae = mae_of(y_te, test["power_lag1"].values)
        m_mae = mae_of(y_te, pred)
        pers_mae.append(p_mae)
        model_mae.append(m_mae)
        skill.append(1.0 - m_mae / p_mae)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(horizons, pers_mae, marker="o", color="black",
                 label="persistence")
    axes[0].plot(horizons, model_mae, marker="o", color=HIGHLIGHT,
                 label="ridge + physics")
    axes[0].set_xlabel("Forecast horizon  [h]")
    axes[0].set_ylabel("MAE  [MW]")
    axes[0].set_title("Error vs horizon")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(horizons, np.array(skill) * 100, marker="o", color=ACCENT)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_xlabel("Forecast horizon  [h]")
    axes[1].set_ylabel("Skill over persistence  [%]")
    axes[1].set_title("Skill score vs horizon")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS, "skill_vs_horizon.png"), dpi=130)
    return horizons, skill


def alpha_sweep(train, physics_cols, alphas=None):
    """Sweep the ridge penalty on the last 20% of the train years (never touches
    the 2023 test set)."""
    if alphas is None:
        alphas = np.logspace(-2, 3, 18)

    cut = int(len(train) * 0.8)
    inner_tr = train.iloc[:cut]
    inner_val = train.iloc[cut:]
    y_tr = inner_tr["target"].values
    y_val = inner_val["target"].values

    val_mae = []
    for a in alphas:
        m = RidgeRegressor(alpha=a).fit(inner_tr[physics_cols].values, y_tr)
        pred = np.clip(m.predict(inner_val[physics_cols].values),
                       0.0, RATED_POWER)
        val_mae.append(mae_of(y_val, pred))

    best = alphas[int(np.argmin(val_mae))]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogx(alphas, val_mae, marker="o", color=ACCENT)
    ax.axvline(best, color=HIGHLIGHT, lw=1.5, label=f"best alpha = {best:.2g}")
    ax.set_xlabel("Ridge penalty alpha")
    ax.set_ylabel("Validation MAE  [MW]")
    ax.set_title("Ridge penalty sweep (chronological validation)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS, "alpha_sweep.png"), dpi=130)
    return best


def error_by_regime(test, y_test, pred):
    """MAE binned by wind speed, to show the error concentrates on the steep
    part of the power curve."""
    wind = test["wind_speed_100m_ms"].values
    edges = np.arange(0, 22, 2.0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.digitize(wind, edges) - 1

    maes, counts = [], []
    for b in range(len(centers)):
        sel = idx == b
        if sel.sum() == 0:
            maes.append(np.nan)
            counts.append(0)
        else:
            maes.append(mae_of(y_test[sel], pred[sel]))
            counts.append(int(sel.sum()))

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax2 = ax1.twinx()
    ax2.bar(centers, counts, width=1.8, color=ACCENT, alpha=0.25)
    ax1.plot(centers, maes, marker="o", color=HIGHLIGHT, lw=2)
    ax1.set_xlabel("Wind speed at 100 m  [m/s]")
    ax1.set_ylabel("MAE  [MW]", color=HIGHLIGHT)
    ax2.set_ylabel("Test hours in bin", color=ACCENT)
    ax1.set_title("Forecast error across the wind speed range")
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS, "error_by_regime.png"), dpi=130)


def main():
    os.makedirs(config.RESULTS, exist_ok=True)
    df_raw = pd.read_csv(config.DATA, parse_dates=["timestamp"])
    df, linear_cols, physics_cols = build_features(df_raw.copy(), horizon=HORIZON)

    split = df["timestamp"] < SPLIT_DATE
    train, test = df[split], df[~split]
    y_train = train["target"].values
    y_test = test["target"].values
    print(f"Forecast horizon: {HORIZON} h")
    print(f"Train rows: {len(train):,}   Test rows: {len(test):,}")
    print("-" * 70)

    metrics("Persistence baseline", y_test, test["power_lag1"].values)

    lin = RidgeRegressor(alpha=1.0).fit(train[linear_cols].values, y_train)
    metrics("Ridge (lag/time features)", y_test,
            lin.predict(test[linear_cols].values))

    model = RidgeRegressor(alpha=1.0).fit(train[physics_cols].values, y_train)
    pred = np.clip(model.predict(test[physics_cols].values), 0.0, RATED_POWER)
    metrics("Ridge + physics features", y_test, pred)
    print("-" * 70)

    # --- core plots ------------------------------------------------------
    window = slice(0, 24 * 7)  # first week of the test set
    t = test["timestamp"].values[window]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(t, y_test[window], label="Actual", color="black", lw=1.5)
    ax.plot(t, pred[window], label="Forecast (ridge + physics)",
            color=HIGHLIGHT, lw=1.5)
    ax.set_ylabel("Power  [MW]")
    ax.set_title(f"{HORIZON}-hour-ahead wind power forecast vs actual "
                 f"(first test week, 2023)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS, "forecast_timeseries.png"), dpi=130)

    importance = pd.Series(np.abs(model.coef_[1:]), index=physics_cols)
    importance = importance.sort_values()
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh(importance.index, importance.values, color=ACCENT)
    ax.set_xlabel("Standardized coefficient magnitude")
    ax.set_title("Feature importance (ridge + physics)")
    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS, "feature_importance.png"), dpi=130)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y_test, pred, s=4, alpha=0.2, color=ACCENT)
    lims = [0, max(y_test.max(), pred.max())]
    ax.plot(lims, lims, color=HIGHLIGHT, lw=1.5)
    ax.set_xlabel("Actual power  [MW]")
    ax.set_ylabel("Predicted power  [MW]")
    ax.set_title("Hold out: actual vs predicted")
    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS, "actual_vs_pred.png"), dpi=130)

    # --- extra studies ---------------------------------------------------
    best_alpha = alpha_sweep(train, physics_cols)
    print(f"Best ridge alpha on validation : {best_alpha:.3g}")
    horizons, skill = skill_vs_horizon(df_raw.copy())
    print("Skill over persistence by horizon:")
    for h, s in zip(horizons, skill):
        print(f"  {h:>2} h : {s * 100:+.1f}%")
    error_by_regime(test, y_test, pred)

    print(f"Saved figures to {os.path.abspath(config.RESULTS)}")


if __name__ == "__main__":
    main()
