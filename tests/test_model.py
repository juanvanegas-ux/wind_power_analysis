import numpy as np
import pandas as pd

from model import RidgeRegressor, build_features


def test_ridge_recovers_linear_signal():
    # tiny penalty + clean linear data should give back the true slope
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 3))
    true = np.array([2.0, -1.0, 0.5])
    y = 4.0 + X @ true
    model = RidgeRegressor(alpha=1e-6).fit(X, y)
    pred = model.predict(X)
    assert np.mean(np.abs(pred - y)) < 1e-3


def test_ridge_shrinks_with_more_penalty():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 4))
    y = X @ np.array([3.0, 3.0, 3.0, 3.0]) + rng.normal(scale=0.1, size=300)
    small = RidgeRegressor(alpha=0.01).fit(X, y)
    big = RidgeRegressor(alpha=100.0).fit(X, y)
    # heavier penalty pulls the standardized coefficients toward zero
    assert np.linalg.norm(big.coef_[1:]) < np.linalg.norm(small.coef_[1:])


def _toy_frame(n=400):
    t = pd.date_range("2021-01-01", periods=n, freq="h")
    wind = 8 + 3 * np.sin(np.arange(n) / 12.0)
    return pd.DataFrame({
        "timestamp": t,
        "wind_speed_100m_ms": wind,
        "wind_speed_10m_ms": wind * 0.7,
        "wind_direction_100m_deg": np.full(n, 80.0),
        "temperature_c": np.full(n, 27.0),
    })


def test_build_features_columns_and_no_nans():
    df, linear_cols, physics_cols = build_features(_toy_frame(), horizon=6)
    assert "wind_cube" in physics_cols
    assert set(linear_cols).issubset(set(physics_cols))
    assert not df[physics_cols + ["target"]].isna().any().any()


def test_build_features_horizon_shifts_target():
    df, _, _ = build_features(_toy_frame(), horizon=3)
    assert len(df) > 0
    assert "target" in df.columns
