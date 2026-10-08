"""
Machine Learning Models Module.
Defines Defensible Baseline (Logistic Regression), Primary XGBoost Classifier,
and Unsupervised Anomaly Detector (Isolation Forest).
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    confusion_matrix,
)
import xgboost as xgb


class BaselineModel:
    """Interpretable benchmark using balanced regularized Logistic Regression."""

    def __init__(self, C: float = 0.1, max_iter: int = 1000, random_state: int = 42):
        self.model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            class_weight="balanced",
            random_state=random_state,
            solver="lbfgs",
        )
        self.fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        self.fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("BaselineModel is not fitted.")
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
        probs = self.predict_proba(X)
        preds = (probs >= threshold).astype(int)

        precision_arr, recall_arr, _ = precision_recall_curve(y, probs)
        pr_auc = auc(recall_arr, precision_arr)
        roc_auc = roc_auc_score(y, probs)

        return {
            "pr_auc": float(pr_auc),
            "roc_auc": float(roc_auc),
            "precision": float(precision_score(y, preds, zero_division=0)),
            "recall": float(recall_score(y, preds, zero_division=0)),
            "f1": float(f1_score(y, preds, zero_division=0)),
            "brier_score": float(brier_score_loss(y, probs)),
            "threshold": float(threshold),
        }


class XGBoostFraudModel:
    """Production XGBoost model optimized for extreme imbalance via scale_pos_weight & PR-AUC early stopping."""

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        scale_pos_weight: float = 578.0,
        random_state: int = 42,
        early_stopping_rounds: int = 25,
    ):
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "scale_pos_weight": scale_pos_weight,
            "random_state": random_state,
            "eval_metric": "aucpr",
            "tree_method": "hist",
            "n_jobs": -1,
        }
        self.early_stopping_rounds = early_stopping_rounds
        self.model: Optional[xgb.XGBClassifier] = None
        self.best_threshold: float = 0.5
        self.fitted = False

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
    ):
        eval_set = [(X_val, y_val)] if (X_val is not None and y_val is not None) else None

        self.model = xgb.XGBClassifier(
            **self.params,
            early_stopping_rounds=self.early_stopping_rounds if eval_set else None,
        )

        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        self.fitted = True

        if X_val is not None and y_val is not None:
            val_probs = self.predict_proba(X_val)
            self.best_threshold = self.find_optimal_threshold(y_val, val_probs)

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.fitted or self.model is None:
            raise RuntimeError("XGBoostFraudModel is not fitted.")
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: Optional[float] = None) -> np.ndarray:
        th = threshold if threshold is not None else self.best_threshold
        probs = self.predict_proba(X)
        return (probs >= th).astype(int)

    @staticmethod
    def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        best_idx = np.argmax(f1_scores[:-1])
        return float(thresholds[best_idx])

    def evaluate(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        threshold: Optional[float] = None,
        k_values: Tuple[int, ...] = (100, 200, 500),
    ) -> Dict[str, Any]:
        probs = self.predict_proba(X)
        th = threshold if threshold is not None else self.best_threshold
        preds = (probs >= th).astype(int)

        precision_arr, recall_arr, _ = precision_recall_curve(y, probs)
        pr_auc = auc(recall_arr, precision_arr)
        roc_auc = roc_auc_score(y, probs)

        cm = confusion_matrix(y, preds)
        tn, fp, fn, tp = cm.ravel()

        results = {
            "pr_auc": float(pr_auc),
            "roc_auc": float(roc_auc),
            "precision": float(precision_score(y, preds, zero_division=0)),
            "recall": float(recall_score(y, preds, zero_division=0)),
            "f1": float(f1_score(y, preds, zero_division=0)),
            "brier_score": float(brier_score_loss(y, probs)),
            "threshold_used": float(th),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        }

        # Top-K alert ranking metrics
        sorted_indices = np.argsort(probs)[::-1]
        y_sorted = y[sorted_indices]
        for k in k_values:
            if k <= len(y):
                k_frauds = int(y_sorted[:k].sum())
                total_frauds = int(y.sum())
                results[f"precision_at_{k}"] = float(k_frauds / k)
                results[f"recall_at_{k}"] = float(k_frauds / max(total_frauds, 1))
                results[f"fraud_captured_at_{k}"] = k_frauds

        return results

    def get_feature_importances(self, feature_names: List[str]) -> pd.DataFrame:
        if not self.fitted or self.model is None:
            raise RuntimeError("Model is not fitted.")
        return pd.DataFrame({
            "feature": feature_names,
            "importance": self.model.feature_importances_,
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)


class AnomalyDetector:
    """Unsupervised Isolation Forest anomaly detector scoring uncharacteristic behaviors from 0 to 1."""

    def __init__(self, n_estimators: int = 100, contamination: float = 0.002, random_state: int = 42):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        self.fitted = False
        self.score_min = -0.5
        self.score_max = 0.5

    def fit(self, X: np.ndarray):
        self.model.fit(X)
        raw_scores = self.model.decision_function(X)
        self.score_min = float(np.percentile(raw_scores, 0.5))
        self.score_max = float(np.percentile(raw_scores, 99.5))
        self.fitted = True
        return self

    def score_anomaly(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("AnomalyDetector must be fitted before scoring.")
        raw_scores = self.model.decision_function(X)
        clipped = np.clip(raw_scores, self.score_min, self.score_max)
        span = self.score_max - self.score_min + 1e-10
        return np.clip(1.0 - ((clipped - self.score_min) / span), 0.0, 1.0)
