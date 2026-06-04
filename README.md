# Wind Power Analysis, La Guajira (Colombia)

End to end analysis of 3 years of hourly wind data (2021 to 2023) for La Guajira,
the region with the best wind energy resource of Colombia. The project pulls real
public data, characterizes the wind resource, works out the wind shear and the
energy yield, and trains a wind power forecasting model. everything is
reproducible from a single command.

Stack: Python, pandas, NumPy, SciPy, Matplotlib, requests. The data come from the
free [Open Meteo](https://open-meteo.com/) historical archive.

## Key findings

```
hours analysed                   : 26,280
mean wind speed (100 m)          : 9.26 m/s
Weibull shape k / scale c        : 3.75 / 10.22 m/s
mean wind power density          : 591 W/m2  (rough class 7 of 7)
median shear exponent (10->100m) : 0.145  (basically the 1/7 law)
estimated capacity factor (2 MW) : about 53%
50 year extreme wind (Gumbel)    : about 36 m/s
```

A capacity factor around 53% is exceptional, most onshore wind sites land around
25 to 40%, and that is the reason La Guajira is the focus of the wind build out
of Colombia.

## Resource assessment

The wind speed distribution, the diurnal and seasonal patterns, and the wind
direction:

![weibull](results/weibull_fit.png)
![patterns](results/patterns.png)
![rose](results/wind_rose.png)

The wind is strong and remarkably steady (high Weibull k), it peaks at night and
during the trade wind season (June to August), and it blows almost always from
the east north east.

If you weight each direction by the energy it carries (the cube of the speed)
instead of just counting hours, the rose gets even tighter on the ENE sector,
which is exactly where you would want to point a row of turbines.

![energy rose](results/energy_rose.png)

The Weibull shape and scale also move through the year, the trade wind months are
both windier (bigger c) and steadier (higher k):

![seasonal weibull](results/seasonal_weibull.png)

## Wind shear (using both heights)

The dataset has wind speed at 10 m and at 100 m, so the shear does not need to be
assumed, you can measure it. fitting the power law exponent alpha at every hour
gives a median of 0.145, which is almost exactly the textbook 1/7 law, with a
clear day night swing (more shear at night when the boundary layer is stable).

Pushing the 10 m wind up to 100 m with that single median alpha lands within
about 1.1 m/s of the real measured 100 m wind, so the power law holds up well
here.

![wind shear](results/wind_shear.png)

## Energy yield

Capacity factor is nice but a developer wants energy in MWh/year and the number
after the real losses (wakes, downtime, electrical). stacking those losses
multiplicatively gives about 14% off the top. comparing a few turbine classes on
the same wind shows the usual trade off, a smaller rated machine has a gorgeous
capacity factor but a bigger one simply makes more energy:

```
turbine                          net CF      net AEP [MWh/yr]
1.5 MW (class III)               53.4%        7,022
2.0 MW (baseline)                45.5%        7,963
3.0 MW (class II)                37.7%        9,894
4.5 MW (class I)                 30.7%       12,106
```

![turbine compare](results/turbine_compare.png)
![cf heatmap](results/cf_heatmap.png)

The heatmap shows the capacity factor is high almost everywhere, with the dip in
the calmer afternoons of the low season (Sep to Nov).

## Forecasting the wind power

A ridge regression written from scratch (NumPy only, closed form) with lagged
features, cyclical time features and physics informed features (the turbine power
goes with the cube of the wind speed). it is evaluated on a strict chronological
hold out, trained on 2021 and 2022 and tested on 2023, at a 6 hour horizon:

```
persistence baseline       : MAE 0.258 MW   RMSE 0.369 MW   R2 0.706
ridge (lag and time)       : MAE 0.257 MW   RMSE 0.340 MW   R2 0.750
ridge + physics features   : MAE 0.253 MW   RMSE 0.339 MW   R2 0.752
```

Being honest about it, on MAE the learned model only edges persistence by about
2% at 6 hours, but on RMSE it is about 8% better, it is cutting the big misses
rather than the typical small ones. the gain is also very horizon dependent: at 1
to 3 hours persistence is basically unbeatable (the wind just does not change
that fast), the model is best in the 6 to 12 hour window, and by 24 hours neither
approach is doing much.

![skill](results/skill_vs_horizon.png)
![ts](results/forecast_timeseries.png)
![imp](results/feature_importance.png)
![scatter](results/actual_vs_pred.png)
![regime](results/error_by_regime.png)

The error by wind speed plot makes the why obvious, almost all the error sits on
the steep part of the power curve (roughly 6 to 11 m/s), where a small wind error
turns into a big power error. above rated the turbine is pinned at 2 MW so it is
trivial to predict.

A penalty sweep on a separate validation slice is included too (alpha_sweep.png).
it barely moves the validation error, this problem is not really limited by
overfitting, it is limited by how predictable the wind is.

## Reproduce it

```bash
pip install -r requirements.txt
python run_all.py            # runs the whole pipeline on the cached data
# or step by step:
python src/fetch_data.py     # downloads data/la_guajira_wind.csv (Open Meteo)
python src/analysis.py       # resource assessment, Weibull, roses, extremes
python src/wind_shear.py     # 10 m vs 100 m shear
python src/energy_yield.py   # AEP, losses, turbine comparison
python src/model.py          # forecasting model and evaluation
```

A snapshot of the data is already committed under data/, so everything runs
without fetching again. tests:

```bash
pip install -r requirements-dev.txt
pytest
```

You can change HORIZON in src/model.py to forecast at other lead times, or the
coordinates in src/config.py to study another site.

## Project layout

```
run_all.py            run the whole pipeline (add --fetch to redownload)
src/config.py         shared constants, paths, air density helper
src/fetch_data.py     download the hourly wind data (Open Meteo archive API)
src/power_curve.py    generic turbine power curve (wind speed to MW)
src/analysis.py       Weibull, patterns, wind/energy rose, seasonal, extremes
src/wind_shear.py     shear exponent from 10 m and 100 m, hub height extrapolation
src/energy_yield.py   AEP with losses, capacity factor heatmap, turbine sizing
src/model.py          ridge regression forecaster, skill vs horizon, error studies
tests/                pytest unit tests
data/                 cached dataset (CSV)
results/              generated figures
```

## Ideas / things i would add next

A running list of where this could go, some are quick, some are projects on their
own:

* pull a real measured power curve for a named turbine instead of the generic
  cubic one, the rated and cut out shape drives every energy number here
* fetch station pressure (or use elevation) so the air density is not assumed at
  sea level
* a proper Weibull goodness of fit, the tails matter for the extreme value side
* gust factor and turbulence intensity if i can get higher frequency data
* try a gradient boosted model and an ARIMA as forecasting baselines, the linear
  model is a floor not a ceiling
* probabilistic forecasts (quantiles) instead of a single point, that is what a
  grid operator actually wants
* spatial analysis across several grid points to see the smoothing you get from
  spreading turbines out
* wake modelling (Jensen) for a small layout so the wake loss is computed not
  guessed
* a tiny dashboard (streamlit) to scrub through the data interactively

## License

MIT, see LICENSE. Weather data from Open Meteo (CC BY 4.0).
