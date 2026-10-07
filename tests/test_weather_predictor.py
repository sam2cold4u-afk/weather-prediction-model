import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from project1.weather_predictor import load_weather, build_features, make_training_data

class WeatherDataTests(unittest.TestCase):
    def data(self):
        return pd.DataFrame({'TMAX': np.arange(20.) + 50,
                             'TMIN': np.arange(20.) + 40,
                             'PRCP': np.zeros(20)},
                            index=pd.date_range('2020-01-01', periods=20))

    def test_absent_day_is_not_next_day(self):
        weather = self.data().drop(pd.Timestamp('2020-01-06'))
        weather.index.name = 'DATE'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'weather.csv'
            weather.to_csv(path)
            loaded = load_weather(path)
        self.assertIn(pd.Timestamp('2020-01-06'), loaded.index)
        self.assertNotIn(pd.Timestamp('2020-01-05'), make_training_data(loaded).index)

    def test_missing_target_is_not_filled(self):
        weather = self.data()
        weather.loc['2020-01-10', 'TMAX'] = np.nan
        training = make_training_data(weather)
        self.assertNotIn(pd.Timestamp('2020-01-09'), training.index)
        self.assertNotIn(pd.Timestamp('2020-01-10'), training.index)
        self.assertEqual(training.loc['2020-01-08', 'target'], 58.)

    def test_features_do_not_use_future(self):
        weather = self.data()
        original = build_features(weather)
        weather.loc['2020-01-11':, ['TMAX', 'TMIN', 'PRCP']] = 999
        changed = build_features(weather)
        pd.testing.assert_frame_equal(original.loc[:'2020-01-10'], changed.loc[:'2020-01-10'])

if __name__ == '__main__':
    unittest.main()
