# Weather Prediction Model

Predict the next calendar day's maximum temperature from historical Heathrow
weather observations using Python, pandas and scikit-learn. Compare Ridge
regression against a baseline that predicts tomorrow's maximum will equal today's.

## Run

Use Python 3.12 or later. From the repository root:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

Then install and run:

```bash
python -m pip install -e .
python src/project1/weather_predictor.py
```

Installation also provides the `project1` command. Optional arguments:

```bash
python src/project1/weather_predictor.py --data local_weather.csv --output results/predictions.csv
```

Run the separate linear regression/gradient descent demonstration:

```bash
python src/project1/main.py
```

If you use uv instead, run `uv sync` followed by `uv run project1`.

## Data and missing values

The included CSV identifies station UKM00003772, Heathrow, UK, with records from
1 January 1973 to 24 August 2025. Required columns are DATE, PRCP, TMAX and TMIN.
The original download source, units and redistribution terms need verification;
these details are not established by the CSV itself. Report MAE in source-data
units until temperature units are confirmed.

Approximately 31% of maximum temperatures are missing. Missing calendar dates
are inserted before generating the next-day target. Only observed maximum
temperatures are used for the forecast origin and target; targets are never
forward-filled. Minimum temperatures may be carried forward for at most three
days. Missing precipitation is filled with zero alongside an indicator that
allows the model to distinguish missing readings from observed zero rainfall.

## Features and evaluation

- Current maximum/minimum temperature and precipitation.
- Temperature means over 3, 7 and 14 calendar days and daily temperature changes.
- Sine/cosine day-of-year features for seasonality.
- StandardScaler fitted on each training window, followed by Ridge with alpha=10.
- Ten initial years of eligible data, expanding training windows and evaluation
  blocks of 90 calendar days.

Forecasts use observations available by the end of each origin day. Although
models are refitted every 90 days, forecasts later in a block use that day's
weather. This tests next-day predictions, not a forecast for 90 days at once.

## Results

Verified in a fresh installation with pandas 3.0.6 and scikit-learn 1.9.1:

| Method | Mean absolute error |
| --- | ---: |
| Ridge regression | 3.2902 |
| Persistence baseline | 3.4809 |

Both methods were evaluated on the same 5,968 observations. The model reduced
MAE by approximately 5.48% against the baseline. These values cannot be directly
compared with the original pipeline's MAE because that pipeline scored filled
rather than observed targets on many dates.

Features were revised after inspecting historical performance, so these are
**development backtest results**, not an untouched final test. Reserve a separate
period before further feature selection or parameter tuning. Missing observations,
imputed minimum temperatures and occasional large errors remain limitations.
This is an educational project rather than an operational forecast service.

## Checks

```bash
python -m unittest discover -s tests -v
```

Checks cover missing calendar days, exclusion of missing targets and features
that remain unchanged when future observations change.

## Files

```text
src/project1/weather_predictor.py   Weather pipeline and evaluation
src/project1/main.py                Linear regression from scratch
src/project1/data.csv               Regression demonstration data
local_weather.csv                  Weather observations
tests/test_weather_predictor.py    Data and feature checks
pyproject.toml                     Dependencies and command entry point
```
