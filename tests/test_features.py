"""
Unit tests for temporal, amount, and PCA feature engineering.
"""

import numpy as np
import pandas as pd
import pytest

from src.features import compute_temporal_features, BehavioralFeatureExtractor, FeaturePipeline


def test_temporal_cyclical_bounds():
    df = pd.DataFrame({"Time": [0.0, 3600.0 * 6, 3600.0 * 12, 3600.0 * 18, 3600.0 * 24]})
    t_feats = compute_temporal_features(df)
    assert (-1.0 <= t_feats["hour_sin"]).all() and (t_feats["hour_sin"] <= 1.0).all()
    assert (-1.0 <= t_feats["hour_cos"]).all() and (t_feats["hour_cos"] <= 1.0).all()
    assert (t_feats["time_delta_prev"] >= 0.0).all()


def test_behavioral_features():
    df = pd.DataFrame({
        "Amount": [0.0, 10.0, 50.0, 2500.0],
        **{f"V{i}": [0.5, -0.5, 1.0, -1.0] for i in range(1, 29)},
    })
    extractor = BehavioralFeatureExtractor()
    extractor.fit(df)
    feats = extractor.transform(df)

    assert "log_amount" in feats.columns
    assert "is_zero_amount" in feats.columns
    assert feats.loc[0, "is_zero_amount"] == 1.0
    assert feats.loc[1, "is_zero_amount"] == 0.0
    assert "v_latent_norm" in feats.columns
    assert "v_risk_polarity" in feats.columns
    assert (feats["v_latent_norm"] > 0).all()


def test_pipeline_fit_transform():
    train_df = pd.DataFrame({
        "Time": [10.0, 20.0, 30.0],
        "Amount": [10.0, 50.0, 100.0],
        **{f"V{i}": [0.1, -0.2, 0.3] for i in range(1, 29)},
        "Class": [0, 0, 1],
    })
    val_df = pd.DataFrame({
        "Time": [40.0, 50.0],
        "Amount": [20.0, 80.0],
        **{f"V{i}": [-0.1, 0.4] for i in range(1, 29)},
    })

    pipeline = FeaturePipeline()
    pipeline.fit(train_df)
    assert pipeline.fitted_

    X_train = pipeline.transform(train_df)
    X_val = pipeline.transform(val_df)

    assert X_train.shape[1] == len(pipeline.feature_names_)
    assert X_val.shape[1] == len(pipeline.feature_names_)
    assert not X_train.isnull().any().any()
    assert not X_val.isnull().any().any()
