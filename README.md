# Wind Power Analysis — La Guajira, Colombia

End-to-end analysis of **3 years of hourly wind data (2021–2023)** for La
Guajira, the region with Colombia's best wind-energy resource. The project
pulls real public data, characterizes the wind resource, estimates energy
yield, and trains a **wind-power forecasting model** — all reproducible from a
single command.

**Stack:** Python · pandas · NumPy · SciPy · Matplotlib · requests
(data from the free [Open-Meteo](https://open-meteo.com/) historical archive)

## Key findings

| Metric (2021–2023, 100 m hub height) | Value |
|---|---|
| Hours analysed | 26,280 |
| Mean wind speed | **9.26 m/s** |
| Weibull shape *k* / scale *c* | 3.75 / 10.22 m/s |
| Mean wind power density | 593 W/m² |
| Estimated capacity factor (2 MW turbine) | **~53%** |

A ~53% capacity factor is exceptional — most onshore wind sites sit around
25–40% — confirming why La Guajira is the focus of Colombia's wind build-out.

## Resource assessment

| Wind-speed distribution | Diurnal & seasonal patterns | Wind direction |
|---|---|---|
| ![weibull](results/weibull_fit.png) | ![patterns](results/patterns.png) | ![rose](results/wind_rose.png) |

The wind is strong, remarkably steady (high Weibull *k*), peaks at night and in
the trade-wind season (Jun–Aug), and blows overwhelmingly from the east-northeast.

## Forecasting wind power (6 hours ahead)

A from-scratch **ridge regression** (NumPy only) with lagged, cyclical-time and
**physics-informed** features (turbine power ∝ wind³). Evaluated on a strict
chronological hold-out — trained on 2021–2022, tested on 2023.

| Model | MAE (MW) | RMSE (MW) | R² |
|---|---|---|---|
| Persistence baseline | 0.258 | 0.369 | 0.706 |
| Ridge (lag/time features) | 0.257 | 0.340 | 0.750 |
| **Ridge + physics features** | **0.253** | **0.339** | **0.752** |

The learned model cuts forecast error ~8% below the naive baseline at a 6-hour
horizon (where persistence starts to fail).

| Forecast vs actual | Feature importance | Hold-out fit |
|---|---|---|
| ![ts](results/forecast_timeseries.png) | ![imp](results/feature_importance.png) | ![scatter](results/actual_vs_pred.png) |

## Reproduce it

```bash
pip install -r requirements.txt
python src/fetch_data.py   # downloads data/la_guajira_wind.csv (Open-Meteo)
python src/analysis.py     # resource assessment + figures
python src/model.py        # forecasting model + figures
```

A snapshot of the data is committed under `data/`, so the analysis and model run
without re-fetching. Change `HORIZON` in `src/model.py` to forecast at other
lead times, or the coordinates in `src/fetch_data.py` to study another site.

## Project layout

```
src/fetch_data.py    download hourly wind data (Open-Meteo archive API)
src/power_curve.py   generic turbine power curve (wind speed -> MW)
src/analysis.py      Weibull fit, diurnal/seasonal patterns, wind rose, capacity factor
src/model.py         ridge-regression wind-power forecaster + evaluation
data/                cached dataset (CSV)
results/             generated figures
```

## License

MIT — see [LICENSE](LICENSE). Weather data © Open-Meteo (CC BY 4.0).
