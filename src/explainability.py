"""
Model Explainability Module using SHAP TreeExplainer.
Provides mathematically exact additive feature attributions for XGBoost predictions.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
import shap
import xgboost as xgb


class ModelExplainer:
    """
    SHAP TreeExplainer providing exact local and global feature attribution.
    Extracts human-understandable risk drivers and mitigating factors for risk investigators.
    """

    def __init__(self, xgb_model: xgb.XGBClassifier, feature_names: List[str]):
        self.xgb_model = xgb_model
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(xgb_model)

    def explain_global(self, X_sample: pd.DataFrame, max_features: int = 15) -> pd.DataFrame:
        """Computes mean absolute SHAP values across a representative sample."""
        shap_values = self.explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        return pd.DataFrame({
            "feature": self.feature_names,
            "mean_abs_shap": mean_abs_shap,
        }).sort_values(by="mean_abs_shap", ascending=False).head(max_features).reset_index(drop=True)

    def explain_local(self, transaction_features: pd.DataFrame, top_k: int = 5) -> Dict[str, Any]:
        """
        Computes local SHAP attributions for an individual transaction.
        Returns top risk-increasing and risk-mitigating features with actual values.
        """
        if len(transaction_features) > 1:
            transaction_features = transaction_features.iloc[[0]]

        shap_values = self.explainer.shap_values(transaction_features)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        row_shap = shap_values[0]
        feature_vals = transaction_features.iloc[0].to_dict()

        factors = []
        for feat_name, s_val in zip(self.feature_names, row_shap):
            factors.append({
                "feature": feat_name,
                "shap_value": float(s_val),
                "feature_value": float(feature_vals.get(feat_name, 0.0)),
                "impact": "INCREASES_RISK" if s_val > 0 else "DECREASES_RISK",
            })

        factors_sorted = sorted(factors, key=lambda x: abs(x["shap_value"]), reverse=True)
        top_positive = [f for f in factors_sorted if f["shap_value"] > 0][:top_k]
        top_negative = [f for f in factors_sorted if f["shap_value"] < 0][:top_k]

        exp_val = self.explainer.expected_value
        base_val = float(exp_val if not isinstance(exp_val, np.ndarray) else exp_val[1])

        return {
            "base_value": base_val,
            "top_risk_drivers": top_positive,
            "top_mitigating_factors": top_negative,
            "all_factors_sorted": factors_sorted[:top_k * 2],
            "disclaimer": "SHAP explains model feature attribution; it does not establish real-world legal causality.",
        }
