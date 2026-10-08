"""
Unit tests for Baseline, XGBoost, Anomaly Detector, and Risk Engine.
"""

import numpy as np
import pandas as pd
import pytest

from src.models import BaselineModel, XGBoostFraudModel, AnomalyDetector
from src.risk import RiskEngine, InvestigationTriage


def test_baseline_model_fit_and_predict():
    X = np.random.randn(100, 10)
    y = np.random.choice([0, 1], size=100, p=[0.9, 0.1])
    baseline = BaselineModel()
    baseline.fit(X, y)
    probs = baseline.predict_proba(X)
    assert len(probs) == 100
    assert (0.0 <= probs).all() and (probs <= 1.0).all()


def test_xgboost_fit_and_threshold():
    X = pd.DataFrame(np.random.randn(100, 10), columns=[f"f_{i}" for i in range(10)])
    y = np.random.choice([0, 1], size=100, p=[0.9, 0.1])
    model = XGBoostFraudModel(n_estimators=10, max_depth=2)
    model.fit(X, y)
    probs = model.predict_proba(X)
    assert len(probs) == 100
    assert (0.0 <= probs).all() and (probs <= 1.0).all()


def test_anomaly_detector_normalized_range():
    X = np.random.randn(100, 10)
    detector = AnomalyDetector(n_estimators=20)
    detector.fit(X)
    scores = detector.score_anomaly(X)
    assert len(scores) == 100
    assert (0.0 <= scores).all() and (scores <= 1.0).all()


def test_risk_engine_decision_tiers():
    engine = RiskEngine(low_threshold=35.0, high_threshold=70.0)

    # Low risk
    d_low = engine.evaluate(ml_probability=0.05, anomaly_score=0.10, amount=25.0)
    assert d_low.risk_level == "LOW"
    assert d_low.action == "ALLOW"

    # Medium risk
    d_med = engine.evaluate(ml_probability=0.50, anomaly_score=0.50, amount=100.0)
    assert d_med.risk_level == "MEDIUM"
    assert d_med.action == "FLAG_CHALLENGE"

    # High risk
    d_high = engine.evaluate(ml_probability=0.95, anomaly_score=0.90, amount=2000.0)
    assert d_high.risk_level == "HIGH"
    assert d_high.action == "BLOCK_AND_INVESTIGATE"
    assert d_high.priority_score > d_low.priority_score


def test_investigation_triage_ranking():
    df = pd.DataFrame({
        "risk_score": [90.0, 90.0, 20.0],
        "Amount": [10.0, 1000.0, 50.0],
    })
    ranked = InvestigationTriage.rank_cases(df)
    # The $1000 transaction with high risk score should be ranked first
    assert ranked.iloc[0]["Amount"] == 1000.0
