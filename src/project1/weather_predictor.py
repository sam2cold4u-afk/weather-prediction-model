"""Evaluate next-day maximum temperatures using chronological backtesting."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_DATA = Path(__file__).resolve().parents[2] / 'local_weather.csv'

def load_weather(path):
    weather = pd.read_csv(path, index_col='DATE', parse_dates=True).sort_index()
    if not weather.index.is_unique:
        raise ValueError('Expected one weather record per date.')
    return weather.asfreq('D')  # Insert absent dates before shifting the target.

def build_features(weather):
    features = pd.DataFrame(index=weather.index)
    features['temp_max'] = weather.TMAX
    features['temp_min'] = weather.TMIN.ffill(limit=3)
    features['precip_missing'] = weather.PRCP.isna().astype(int)
    features['precip'] = weather.PRCP.fillna(0)
    for column in ['temp_max', 'temp_min']:
        for horizon in [3, 7, 14]:
            features[f'{column}_mean_{horizon}'] = features[column].rolling(horizon, min_periods=1).mean()
        features[f'{column}_change'] = features[column].diff()
    angle = 2 * np.pi * features.index.dayofyear / 365.25
    features['season_sin'] = np.sin(angle)
    features['season_cos'] = np.cos(angle)
    return features

def make_training_data(weather):
    data = build_features(weather)
    data['target'] = weather.TMAX.shift(-1)  # Never impute actual target values.
    return data.dropna()

def backtest(data, initial_years=10, step_days=90):
    if initial_years < 1 or step_days < 1:
        raise ValueError('Window sizes must be positive.')
    if data.empty:
        raise ValueError('No complete observations available.')
    predictors = data.columns.drop('target')
    start = data.index.min() + pd.DateOffset(years=initial_years)
    results = []
    while start <= data.index.max():
        train = data.loc[data.index < start]
        test = data.loc[(data.index >= start) & (data.index < start + pd.Timedelta(days=step_days))]
        if not test.empty:
            model = make_pipeline(StandardScaler(), Ridge(alpha=10))
            model.fit(train[predictors], train.target)
            result = pd.DataFrame({'actual': test.target,
                                   'prediction': model.predict(test[predictors]),
                                   'baseline': test.temp_max}, index=test.index)
            result['absolute_error'] = (result.prediction - result.actual).abs()
            results.append(result)
        start += pd.Timedelta(days=step_days)
    if not results:
        raise ValueError('Insufficient history for the initial training window.')
    return pd.concat(results)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--output', type=Path, help='Optional predictions CSV')
    args = parser.parse_args()
    predictions = backtest(make_training_data(load_weather(args.data)))
    model_error = mean_absolute_error(predictions.actual, predictions.prediction)
    baseline_error = mean_absolute_error(predictions.actual, predictions.baseline)
    print(f'Evaluated observations: {len(predictions):,}')
    print(f'Ridge model MAE: {model_error:.4f}')
    print(f'Persistence baseline MAE: {baseline_error:.4f}')
    if baseline_error > 0:
        print(f'Improvement over baseline: {100 * (1-model_error/baseline_error):.2f}%')
    print('Errors use the temperature units in the source CSV.')
    print('\nLargest prediction errors:')
    print(predictions.nlargest(10, 'absolute_error').to_string())
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(args.output, index_label='forecast_origin_date')
        print(f'Saved predictions to {args.output}')

if __name__ == '__main__':
    main()
