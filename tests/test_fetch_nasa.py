import numpy as np
import pandas as pd

from fetch_nasa import to_pipeline_schema


def _fake_power_frame(n=48):
    idx = pd.date_range("2021-06-01", periods=n, freq="h")
    ws10 = np.full(n, 6.0)
    ws50 = np.full(n, 8.0)   # implies a fixed shear between 10 and 50 m
    return pd.DataFrame({
        "WS10M": ws10,
        "WS50M": ws50,
        "WD10M": np.full(n, 80.0),
        "T2M": np.full(n, 27.0),
        "PS": np.full(n, 100.5),
    }, index=idx)


def test_schema_has_pipeline_columns():
    out = to_pipeline_schema(_fake_power_frame())
    for col in ["timestamp", "wind_speed_100m_ms", "wind_speed_10m_ms",
                "wind_direction_100m_deg", "temperature_c"]:
        assert col in out.columns


def test_extrapolated_100m_above_50m():
    # with positive shear the 100 m wind must exceed the 50 m wind
    out = to_pipeline_schema(_fake_power_frame())
    assert (out["wind_speed_100m_ms"] > out["wind_speed_50m_ms"]).all()


def test_extrapolation_matches_power_law():
    out = to_pipeline_schema(_fake_power_frame())
    alpha = np.log(8.0 / 6.0) / np.log(50.0 / 10.0)
    expected = 8.0 * (100.0 / 50.0) ** alpha
    assert np.allclose(out["wind_speed_100m_ms"].values, round(expected, 2),
                       atol=0.02)
