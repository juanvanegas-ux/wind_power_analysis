import numpy as np
import pandas as pd

from compare_sources import align, resource_stats, agreement


def _src(mean, n=200):
    idx = pd.date_range("2021-01-01", periods=n, freq="h")
    rng = np.random.default_rng(int(mean * 10))
    return pd.DataFrame({
        "timestamp": idx,
        "wind_speed_100m_ms": rng.normal(mean, 1.0, n),
    })


def test_align_inner_joins_on_timestamp():
    a = _src(9.0, 200)
    b = _src(10.0, 150)  # shorter, so overlap is 150
    merged = align({"A": a, "B": b})
    assert len(merged) == 150
    assert "A" in merged.columns and "B" in merged.columns
    assert "hour" in merged.columns and "month" in merged.columns


def test_resource_stats_keys_and_ranges():
    v = _src(9.0, 1000)["wind_speed_100m_ms"].values
    s = resource_stats(v)
    for key in ["mean", "k", "c", "power_density", "cf", "aep"]:
        assert key in s
    assert 0.0 <= s["cf"] <= 1.0
    assert s["aep"] > 0


def test_agreement_self_is_perfect():
    v = _src(9.0, 500)["wind_speed_100m_ms"].values
    ag = agreement(v, v)
    assert np.isclose(ag["bias"], 0.0)
    assert np.isclose(ag["mae"], 0.0)
    assert np.isclose(ag["corr"], 1.0)


def test_agreement_detects_constant_offset():
    v = _src(9.0, 500)["wind_speed_100m_ms"].values
    ag = agreement(v + 1.5, v)
    assert np.isclose(ag["bias"], 1.5, atol=1e-9)
    assert np.isclose(ag["corr"], 1.0)
