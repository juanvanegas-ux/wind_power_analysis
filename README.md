# Wind Power Analysis, La Guajira (Colombia)

End to end analysis of 3 years of hourly wind data (2021 to 2023) for La Guajira,
the region with the best wind energy resource of Colombia. The project pull real
public data, characterize the wind resource, estimate the energy yield and train
a wind power forecasting model, and everything is reproducible from a single
command.

Stack: Python, pandas, NumPy, SciPy, Matplotlib, requests. The data come from the
free [Open Meteo](https://open-meteo.com/) historical archive.

## Key findings

```
hours analysed                   : 26,280
mean wind speed                  : 9.26 m/s
Weibull shape k / scale c        : 3.75 / 10.22 m/s
mean wind power density          : 593 W/m2
estimated capacity factor (2 MW) : about 53%
```

A capacity factor around 53% is exceptional, most onshore wind sites are around
25 to 40%, and this is the reason why La Guajira is the focus of the wind build
out of Colombia.

## Resource assessment

The wind speed distribution, the diurnal and seasonal patterns, and the wind
direction:

![weibull](results/weibull_fit.png)
![patterns](results/patterns.png)
![rose](results/wind_rose.png)

The wind is strong and remarkably steady (high Weibull k), it peaks at night and
during the trade wind season (June to August), and it blow most of the time from
the east north east.

## Forecasting the wind power (6 hours ahead)

A ridge regression written from scratch (NumPy only) with lagged features,
cyclical time features and physics informed features (the turbine power go with
the cube of the wind speed). It is evaluated on a strict chronological hold out,
trained on 2021 and 2022 and tested on 2023.

```
persistence baseline       : MAE 0.258 MW   RMSE 0.369 MW   R2 0.706
ridge (lag and time)       : MAE 0.257 MW   RMSE 0.340 MW   R2 0.750
ridge + physics features   : MAE 0.253 MW   RMSE 0.339 MW   R2 0.752
```

The learned model cut the forecast error about 8% below the naive baseline at the
6 hour horizon, which is where the persistence start to fail.

![ts](results/forecast_timeseries.png)
![imp](results/feature_importance.png)
![scatter](results/actual_vs_pred.png)

## Reproduce it

```bash
pip install -r requirements.txt
python src/fetch_data.py   # downloads data/la_guajira_wind.csv (Open Meteo)
python src/analysis.py     # resource assessment and figures
python src/model.py        # forecasting model and figures
```

A snapshot of the data is already commited under data/, so the analysis and the
model run without fetching again. You can change HORIZON in src/model.py to
forecast at other lead times, or the coordinates in src/fetch_data.py to study
another site.

## Project layout

```
src/fetch_data.py    download the hourly wind data (Open Meteo archive API)
src/power_curve.py   generic turbine power curve (wind speed to MW)
src/analysis.py      Weibull fit, diurnal and seasonal patterns, wind rose, capacity factor
src/model.py         ridge regression wind power forecaster and evaluation
data/                cached dataset (CSV)
results/             generated figures
```

## License

MIT, see LICENSE. Weather data from Open Meteo (CC BY 4.0).
