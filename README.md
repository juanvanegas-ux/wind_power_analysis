# Wind Power Analysis, La Guajira (Colombia)

End to end analysis of 3 years of hourly wind data (2021 to 2023) for La Guajira,
the region with the best wind energy resource of Colombia. The project pulls real
public data, characterizes the wind resource, works out the wind shear and the
energy yield, sizes small wind turbines using the Cp curves from my companion
[BEM solver](https://github.com/juanvanegas-ux/wind_turbine_bem), and trains a
wind power forecasting model. everything is reproducible from a single command.

Stack: Python, pandas, NumPy, SciPy, Matplotlib, requests. The data come from two
free, independent public sources, [Open Meteo](https://open-meteo.com/) (ERA5
reanalysis) and [NASA POWER](https://power.larc.nasa.gov/) (MERRA-2, built from
NASA satellite observations), which lets the resource numbers be cross checked.

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

## Small wind turbines (using the BEM blade Cp curves)

This is where this repo shakes hands with the other one. my
[wind turbine BEM toolkit](https://github.com/juanvanegas-ux/wind_turbine_bem)
designs two small rotors (about 2.3 to 2.5 m radius, a few kW) and exports their
power coefficient Cp as a function of tip speed ratio lambda. `src/small_wind.py`
drops those exact curves onto the La Guajira wind to see what they would actually
produce.

Two things make this a small wind study and not the multi MW one above: the Cp
comes from a real blade design instead of a generic curve, and small turbines sit
on short towers, so the wind is taken at a 24 m hub (the 10 m measurement pushed
up with the measured shear, not the 100 m level).

Two control strategies are built from the same Cp curve, variable speed (the
rotor tracks lambda_opt so Cp stays at its peak, then the generator caps the
power) and fixed speed (constant rpm, so lambda slides along the curve and Cp
falls off either side of the design point).

![cp](results/swt_cp_curves.png)
![power](results/swt_power_curves.png)

```
turbine          R [m]  Cp_max  lam_opt  rated kW  W/m2  CF gross  CF net  net AEP [MWh/yr]
Smart blade      2.500   0.474     7.67      6.58   335    42.1%   38.8%        22.4
Comercial blade  2.275   0.469     6.24      5.40   332    42.1%   38.8%        18.3
```

![compare](results/swt_compare.png)

The two blades land on the exact same capacity factor, which looks like a typo
but is not. with MPPT and the rated power defined as the MPPT power at a common
rated wind, the blade factor (area times peak Cp) cancels out of CF, so capacity
factor ends up fixed by the wind distribution and V_rated alone, not by the
blade. the blades part ways on absolute energy (AEP), where that factor stays in,
not on CF.

A word on that capacity factor, because being honest about it matters. the ETA
in the model is the electromechanical conversion (shaft to AC out). gross CF is
after that only, net CF knocks off a small loss stack on the same basis as the
energy yield section (downtime, soiling, wiring), just without the wake bucket
since a lone small turbine has no neighbours to steal from. even the ~39% net
number is on the optimistic side for small wind, which in the real world runs
lower because of turbulence near the ground, short towers and rougher siting. it
looks this good because the rated wind (11 m/s) sits only 1.44x above the mean
hub wind (7.63 m/s), a low rated/mean ratio, and because the La Guajira resource
is doing the heavy lifting.

Two clean takeaways. variable speed beats fixed speed by about 7 to 8% of annual
energy, because the fixed speed rotor is spinning too fast for light wind (lambda
too high, Cp low) and gives up the easy low wind energy. and the smart blade
makes clearly more energy than the comercial one, it has a bigger swept area and
a slightly higher peak Cp, which is the same conclusion the BEM study reached, now
confirmed against a real wind resource.

### Where the energy actually comes from

Folding the power curve onto the wind distribution at hub height shows which
winds pay the bills. the most common wind is around 8.5 m/s, but because energy
goes with the cube of the speed the average kWh actually arrives at about 9.3
m/s, right on the steep part of the curve just below rated. the light winds are
common but worth little, the strong winds are valuable but rare.

![energy distribution](results/swt_energy_distribution.png)

### What a taller tower is worth

Tower height is one of the cheapest knobs a small wind owner has, and the
measured shear turns it into MWh. holding the generator nameplate fixed (the
tower does not change the generator) and sweeping just the hub height, going from
an 18 m to a 30 m tower lifts the smart blade AEP by about 18%. that is a real
decision: more steel and a bigger crane against ~18% more energy every year for
the life of the machine.

![hub height](results/swt_hub_height.png)

### Sizing the generator

The blade is fixed but the generator behind it is a choice, and it is the classic
small wind trade off. a small generator (low specific power, low rated wind)
gives a gorgeous capacity factor but caps the energy early, a big one chases the
peaks at the cost of running part loaded most of the time. sweeping the rated
wind speed (the same as sweeping the specific power in W/m2) lays the trade off
out, the baseline 335 W/m2 sits in the sensible middle, where the AEP curve is
starting to flatten but the CF has not yet fallen off a cliff.

![specific power](results/swt_specific_power.png)

The hub height, efficiency, rated wind and cut out at the top of small_wind.py
are easy to change for a different small machine.

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

## Two data sources (NASA satellite vs ERA5)

The main analysis runs on Open Meteo (ERA5), but the same site can be pulled from
NASA POWER, which is built on the MERRA-2 reanalysis and assimilates NASA
satellite observations. it is free and needs no key, and `src/fetch_nasa.py`
downloads it and writes the exact same CSV columns so the rest of the pipeline
runs on it unchanged.

One wrinkle: NASA POWER reports wind at 10 m and 50 m, not at 100 m. that is
actually handy, you measure the power law shear from the two heights and
extrapolate up to the 100 m hub yourself (same physics as the wind shear
section). POWER also gives the real surface pressure, so the air density does not
have to be assumed.

`src/compare_sources.py` runs a full side by side on the overlapping hours,
computing everything the same way for both (same Weibull fit, same power curve,
same air density) so the only thing that varies is the data:

```
metric                  Open Meteo (ERA5)   NASA POWER (MERRA-2)
mean wind 100 m [m/s]                9.26                  10.47
Weibull k [-]                        3.75                   4.29
Weibull c [m/s]                     10.22                  11.48
power density [W/m2]                   593                    820
capacity factor [%]                  53.0                   66.2
AEP per 2 MW [MWh/yr]               9,291                 11,596

agreement (ERA5 minus MERRA-2):  bias -1.21 m/s,  MAE 1.48 m/s,
                                 RMSE 1.86 m/s,  correlation 0.878
```

![sources](results/source_comparison.png)

Six panels: distribution, hourly scatter, seasonal and diurnal cycle, the bias
across the wind range, and the capacity factor. they correlate well (0.88) and
share the same seasonal shape, but NASA reads about 1.2 m/s windier on average
and the gap widens at the higher wind speeds. that is not a small thing: pushed
through the power curve it moves the estimated capacity factor from ~53% (ERA5)
to ~66% (MERRA-2), and the annual energy by about 2,300 MWh per turbine.

The honest takeaway is that the headline number depends on which reanalysis you
trust, and a real project would calibrate against an on site met mast before
committing to either. reassuringly, the diurnal and seasonal shapes agree and the
shear exponent the two give is basically the same (~0.14), so the physics of the
site is consistent even if the absolute level is not.

## Reproduce it

```bash
pip install -r requirements.txt
python run_all.py            # runs the whole pipeline on the cached data
# or step by step:
python src/fetch_data.py     # downloads data/la_guajira_wind.csv (Open Meteo)
python src/fetch_nasa.py     # optional: same site from NASA POWER (MERRA-2)
python src/compare_sources.py # side by side of the two sources
python src/analysis.py       # resource assessment, Weibull, roses, extremes
python src/wind_shear.py     # 10 m vs 100 m shear
python src/energy_yield.py   # AEP, losses, turbine comparison
python src/small_wind.py     # small turbines using the BEM Cp curves
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
src/fetch_nasa.py     download the same site from NASA POWER (MERRA-2 satellite)
src/compare_sources.py  side by side of the two sources (ERA5 vs MERRA-2)
src/power_curve.py    generic turbine power curve (wind speed to MW)
src/analysis.py       Weibull, patterns, wind/energy rose, seasonal, extremes
src/wind_shear.py     shear exponent from 10 m and 100 m, hub height extrapolation
src/energy_yield.py   AEP with losses, capacity factor heatmap, turbine sizing
src/small_wind.py     small wind turbines from the BEM Cp(lambda) curves
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
* use the NASA POWER surface pressure end to end for the air density instead of
  the sea level assumption on the Open Meteo run (the data is already pulled)
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
* a rough LCOE for the small turbine, the AEP is half the equation, turbine and
  tower cost are the other half and that is what actually decides the hub height
* validate the small wind power curves against a real measured one (a Bergey or
  similar) to see how far the BEM + generic losses sit from a certified machine
* a tiny dashboard (streamlit) to scrub through the data interactively

## License

MIT, see LICENSE. Weather data from Open Meteo (CC BY 4.0) and NASA POWER
(freely available, please credit the NASA Langley POWER project).
