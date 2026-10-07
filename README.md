# Weather Prediction Model

A machine learning project built in Python to predict future daily maximum temperatures using historical weather data. The model utilizes Ridge Regression with rolling-window feature engineering and time-series backtesting.

---

## Overview

Predicting time-series weather data requires preventing data leakage while capturing short-term weather momentum and seasonal cycles. This project frames next-day maximum temperature (`temp_max`) as a supervised regression target and implements a rolling backtest over historical records.

---

## Features & Engineering

- **Data Preprocessing**: Handled missing values across core meteorological metrics (`PRCP`, `SNWD`, `TMAX`, `TMIN`) using forward fills and domain-appropriate zero-fills.
- **Rolling Averages**: Computed 3-day and 14-day rolling means to detect short-term heating and cooling trends.
- **Relative Changes**: Calculated percentage deviations between daily readings and their moving averages.
- **Seasonal Baselines**: Generated expanding monthly and day-of-year historical averages to anchor predictions to seasonal norms.
- **Time-Series Backtesting**: Implemented an expanding-window backtesting framework starting after 10 years of historical data to evaluate out-of-sample accuracy without lookahead bias.

---

## Tech Stack

- **Python 3.12**
- **Pandas** (Data manipulation and time-series indexing)
- **Scikit-Learn** (Ridge Regression and Mean Absolute Error evaluation)

---

## Project Structure

```text
├── src/
│   └── project1/
│       └── weather_predictor.py   # Main pipeline script
├── .gitignore                     # Ignores local data, virtual environments, and cache
├── pyproject.toml                 # Project metadata and dependencies
└── README.md                      # Project documentation
