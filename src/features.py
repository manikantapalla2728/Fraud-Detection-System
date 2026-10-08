"""
Feature Engineering Pipeline Module.
Implements leak-free temporal transformations, non-linear amount scaling,
and PCA behavioral interactions for anonymized transaction-level data.
"""

from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


def compute_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes time-of-day cyclical sine/cosine representations and inter-arrival time deltas.
    Uses only elapsed seconds in the stream without lookahead bias.
    """
    features = pd.DataFrame(index=df.index)
    time_sec = df["Time"].values.astype(np.float64)
    hour_of_day = (time_sec // 3600) % 24

    features["hour_of_day"] = hour_of_day
    features["hour_sin"] = np.sin(2 * np.pi * hour_of_day / 24.0)
    features["hour_cos"] = np.cos(2 * np.pi * hour_of_day / 24.0)

    delta = df["Time"].diff().fillna(0.0)
    features["time_delta_prev"] = np.maximum(delta.values, 0.0)
    features["log_time_delta_prev"] = np.log1p(features["time_delta_prev"])
    return features


class BehavioralFeatureExtractor:
    """
    Extracts statistical amount deviations and latent PCA interaction signals.
    Parameters (mean, std) are fitted strictly on training data to prevent leakage.
    """

    def __init__(self):
        self.hist_mean_amount: float = 88.0
        self.hist_std_amount: float = 250.0
        self.fitted: bool = False

    def fit(self, df: pd.DataFrame):
        self.hist_mean_amount = float(df["Amount"].mean())
        self.hist_std_amount = float(df["Amount"].std())
        if np.isclose(self.hist_std_amount, 0.0):
            self.hist_std_amount = 1.0
        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        amt = df["Amount"].values.astype(np.float64)

        # Amount transformations
        features["log_amount"] = np.log1p(amt)
        features["is_zero_amount"] = (amt == 0.0).astype(float)
        features["amount_is_round"] = (np.mod(amt, 1.0) == 0.0).astype(float)
        features["amount_zscore"] = (amt - self.hist_mean_amount) / (self.hist_std_amount + 1e-5)
        features["amount_ratio_to_mean"] = amt / (self.hist_mean_amount + 1e-5)

        # PCA behavioral interactions
        pca_cols = [f"V{i}" for i in range(1, 29) if f"V{i}" in df.columns]
        if pca_cols:
            v_matrix = df[pca_cols].values
            features["v_latent_norm"] = np.linalg.norm(v_matrix, axis=1)

            # Polarity along strongest distinguishing PCA dimensions
            v4 = df["V4"].values if "V4" in df.columns else 0.0
            v11 = df["V11"].values if "V11" in df.columns else 0.0
            v2 = df["V2"].values if "V2" in df.columns else 0.0
            v14 = df["V14"].values if "V14" in df.columns else 0.0
            v17 = df["V17"].values if "V17" in df.columns else 0.0
            v12 = df["V12"].values if "V12" in df.columns else 0.0
            v10 = df["V10"].values if "V10" in df.columns else 0.0

            features["v_risk_polarity"] = (v4 + v11 + v2) - (v14 + v17 + v12 + v10)
            features["v_extreme_deviation"] = (
                (np.abs(v14) > 4.0).astype(float)
                + (np.abs(v17) > 4.0).astype(float)
                + (np.abs(v12) > 4.0).astype(float)
            )

        return features


class FeaturePipeline:
    """
    Unified feature engineering pipeline orchestrating temporal and behavioral transformations.
    Fitted strictly on the training partition.
    """

    def __init__(self):
        self.behavioral_extractor = BehavioralFeatureExtractor()
        self.scaler = RobustScaler()
        self.feature_names_: List[str] = []
        self.fitted_: bool = False

    def fit(self, df: pd.DataFrame):
        self.behavioral_extractor.fit(df)
        X_df = self._extract_features(df)
        self.feature_names_ = X_df.columns.tolist()
        self.scaler.fit(X_df)
        self.fitted_ = True
        return self

    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        base_cols = [f"V{i}" for i in range(1, 29) if f"V{i}" in df.columns] + ["Amount"]
        base_df = df[base_cols].copy()
        temp_df = compute_temporal_features(df)
        behav_df = self.behavioral_extractor.transform(df)
        return pd.concat([base_df, temp_df, behav_df], axis=1)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("FeaturePipeline must be fitted before calling transform().")

        combined = self._extract_features(df)
        for col in self.feature_names_:
            if col not in combined.columns:
                combined[col] = 0.0

        return combined[self.feature_names_]

    def transform_scaled(self, df: pd.DataFrame) -> np.ndarray:
        X_df = self.transform(df)
        return self.scaler.transform(X_df)
