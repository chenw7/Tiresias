import pandas as pd
import numpy as np
from scipy.stats import zscore, median_abs_deviation
from sklearn.ensemble import IsolationForest
import joblib

class FeatureEngineer:
    def __init__(self, raw_data_path):
        self.raw_data_path = raw_data_path
        self.data = pd.read_csv(raw_data_path, header=1).apply(pd.to_numeric, errors='coerce')
        self.med = self.data.groupby('site').median().T
        self.std = self.data.groupby('site').std().T
        self.z_med = self._standardize(self.med)
        self.z_std = self._standardize(self.std)
        self.features = self._generate_features()

    def _standardize(self, df):
        z = df.apply(zscore, axis=1, nan_policy='omit')
        return pd.DataFrame(z.tolist(), index=df.index, columns=df.columns)

    def _generate_features(self):
        z_med_range = self.z_med.max(axis=1) - self.z_med.min(axis=1)
        z_std_range = self.z_std.max(axis=1) - self.z_std.min(axis=1)
        z_med_iqr = self.z_med.quantile(0.75, axis=1) - self.z_med.quantile(0.25, axis=1)
        z_std_iqr = self.z_std.quantile(0.75, axis=1) - self.z_std.quantile(0.25, axis=1)
        z_med_mad = self.z_med.apply(median_abs_deviation, axis=1)
        z_std_mad = self.z_std.apply(median_abs_deviation, axis=1)
        med_skewness = self.med.skew(axis=1)
        std_skewness = self.std.skew(axis=1)
        med_kurtosis = self.med.kurt(axis=1)
        std_kurtosis = self.std.kurt(axis=1)

        features = pd.concat([
            z_med_range, z_std_range, z_med_iqr, z_std_iqr,
            z_med_mad, z_std_mad, med_skewness, std_skewness,
            med_kurtosis, std_kurtosis
        ], axis=1)

        features.columns = [
            "z_med_range", "z_std_range", "z_med_iqr", "z_std_iqr",
            "z_med_mad", "z_std_mad", "med_skewness", "std_skewness",
            "med_kurtosis", "std_kurtosis"
        ]

        return features

    def get_features(self):
        return self.features


class AnomalyDetector:
    def __init__(self, feature_engineer, model_path='lgbm_raw_noimpute.pkl', pipeline_path='pipeline_noimpute.pkl'):
        self.fe = feature_engineer
        self.model = joblib.load(model_path)
        self.pipeline = joblib.load(pipeline_path)
        self.predictions = self._predict()
        self.anomalies_dict = self._detect_anomalies()

    def _predict(self):
        x_transformed = self.pipeline.transform(self.fe.features)
        return self.model.predict(x_transformed)

    def _detect_anomalies(self):
        df = pd.DataFrame({'LightGBM': self.predictions})
        df = df[df['LightGBM'] == 0]

        # Reindex z_med and z_std
        z_med = self.fe.z_med.copy()
        z_std = self.fe.z_std.copy()
        z_med.index = range(z_med.shape[0])
        z_std.index = range(z_std.shape[0])
        z_med = z_med.loc[df.index]
        z_std = z_std.loc[df.index]
        z_med.columns = [i + 1 for i in range(z_med.shape[1])]
        z_std.columns = [i + 1 for i in range(z_std.shape[1])]

        anomalies_dict = {}

        for idx in z_med.index:
            param_med = z_med.loc[idx]
            param_std = z_std.loc[idx]
            data = pd.DataFrame({'med': param_med, 'std': param_std})

            iso_forest = IsolationForest(contamination=0.0065, random_state=42)
            data['anomaly'] = iso_forest.fit_predict(data[['med', 'std']])
            anomalies = data[data['anomaly'] == -1].index.tolist()
            anomalies_dict[idx] = anomalies

        return anomalies_dict

    def get_anomalies(self):
        return self.anomalies_dict
