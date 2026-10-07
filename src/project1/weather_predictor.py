import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

# 1. Load data
weather = pd.read_csv("project1/local_weather.csv", index_col="DATE")

# 2. Extract core columns & rename
core_weather = weather[["PRCP", "SNWD", "TMAX", "TMIN"]].copy()
core_weather.columns = ["precip", "snow_depth", "temp_max", "temp_min"]

# 3. Fill missing values
core_weather["precip"] = core_weather["precip"].fillna(0)
core_weather["snow_depth"] = core_weather["snow_depth"].fillna(0)
core_weather = core_weather.ffill()

# 4. Format date index & create target
core_weather.index = pd.to_datetime(core_weather.index)
core_weather["target"] = core_weather.shift(-1)["temp_max"]
core_weather = core_weather.iloc[:-1, :].copy()

# 5. Initialize model
reg = Ridge(alpha=0.1)

# 6. Define the backtest function
def backtest(weather, model, predictors, start=3650, step=90):
    all_predictions = []
    
    for i in range(start, weather.shape[0], step):
        train = weather.iloc[:i, :]
        test = weather.iloc[i:(i+step), :]
        
        model.fit(train[predictors], train["target"])
        
        preds = model.predict(test[predictors])
        preds = pd.Series(preds, index=test.index)
        combined = pd.concat([test["target"], preds], axis=1)
        combined.columns = ["actual", "prediction"]
        combined["diff"] = (combined["prediction"] - combined["actual"]).abs()
        
        all_predictions.append(combined)
        
    return pd.concat(all_predictions)

# 7. Helper functions for rolling features
def pct_diff(old, new):
    return (new - old) / old

def compute_rolling(weather, horizon, col):
    label = f"rolling_{horizon}_{col}"
    weather[label] = weather[col].rolling(horizon).mean()
    weather[f"{label}_pct"] = pct_diff(weather[label], weather[col])
    return weather

rolling_horizons = [3, 14]
for horizon in rolling_horizons:
    for col in ["temp_max", "temp_min", "precip"]:
        core_weather = compute_rolling(core_weather, horizon, col)

# 8. Expanding averages
core_weather["month_avg"] = core_weather["temp_max"].groupby(core_weather.index.month, group_keys=False).apply(lambda x: x.expanding(1).mean())
core_weather["day_of_year_avg"] = core_weather["temp_max"].groupby(core_weather.index.day_of_year, group_keys=False).apply(lambda x: x.expanding(1).mean())

# 9. Clean up NaN / infinite values
core_weather = core_weather.iloc[14:, :].copy()
core_weather = core_weather.fillna(0)
core_weather = core_weather.replace([float("inf"), float("-inf")], 0)

# 10. Run backtest & evaluate
predictors = core_weather.columns[~core_weather.columns.isin(["target"])]
predictions = backtest(core_weather, reg, predictors)

error = mean_absolute_error(predictions["actual"], predictions["prediction"])
print(f"Mean Absolute Error: {error:.2f}")
print("\nTop 10 Largest Prediction Errors:")
print(predictions.sort_values("diff", ascending=False).head(10))