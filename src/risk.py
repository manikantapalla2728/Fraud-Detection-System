"""
Risk Decision Engine & Investigation Case Triage Module.
Combines Supervised ML and Unsupervised Anomaly Detection into an operational 0-100 score,
action tiers (ALLOW, FLAG_CHALLENGE, BLOCK_AND_INVESTIGATE), and priority ranking.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd


@dataclass
class RiskDecision:
    risk_score: float
    risk_level: str          # "LOW", "MEDIUM", "HIGH"
    action: str              # "ALLOW", "FLAG_CHALLENGE", "BLOCK_AND_INVESTIGATE"
    ml_probability: float
    anomaly_score: float
    priority_score: float
    primary_reasons: List[str]
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
            "action": self.action,
            "ml_probability": self.ml_probability,
            "anomaly_score": self.anomaly_score,
            "priority_score": self.priority_score,
            "primary_reasons": self.primary_reasons,
            "details": self.details,
        }


class RiskEngine:
    """
    Transparent business risk engine synthesizing supervised and anomaly signals.
    Applies calibrated thresholds and monetary exposure prioritization for human investigators.
    """

    def __init__(
        self,
        weight_ml: float = 0.80,
        weight_anomaly: float = 0.20,
        low_threshold: float = 35.0,
        high_threshold: float = 70.0,
    ):
        self.w_ml = weight_ml
        self.w_anomaly = weight_anomaly
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def evaluate(
        self,
        ml_probability: float,
        anomaly_score: float = 0.0,
        amount: float = 0.0,
    ) -> RiskDecision:
        """
        Computes composite risk score (0-100), policy tier, and investigation priority.
        Formula: 100 * (0.80 * ml_prob + 0.20 * anomaly_score).
        """
        blended = (self.w_ml * ml_probability) + (self.w_anomaly * anomaly_score)
        raw_score = blended * 100.0
        final_score = round(min(100.0, max(0.0, raw_score)), 1)

        # Monetary exposure weighting for triage
        priority_score = round(final_score * float(np.log1p(max(0.0, amount))), 2)

        # Operational decision tiers
        if final_score < self.low_threshold:
            risk_level = "LOW"
            action = "ALLOW"
        elif final_score < self.high_threshold:
            risk_level = "MEDIUM"
            action = "FLAG_CHALLENGE"
        else:
            risk_level = "HIGH"
            action = "BLOCK_AND_INVESTIGATE"

        # Transparent reason codes
        reasons = []
        if ml_probability >= 0.70:
            reasons.append(f"Supervised ML fraud probability elevated ({ml_probability:.1%})")
        if anomaly_score >= 0.75:
            reasons.append(f"Unsupervised behavioral anomaly detected (Isolation Score: {anomaly_score:.2f})")
        if amount > 1000.0 and final_score >= self.low_threshold:
            reasons.append(f"High monetary exposure (${amount:,.2f}) amplifies risk severity")
        if not reasons:
            reasons.append("Transaction within expected behavioral boundaries")

        return RiskDecision(
            risk_score=final_score,
            risk_level=risk_level,
            action=action,
            ml_probability=round(float(ml_probability), 4),
            anomaly_score=round(float(anomaly_score), 4),
            priority_score=priority_score,
            primary_reasons=reasons,
            details={"amount": amount, "w_ml": self.w_ml, "w_anomaly": self.w_anomaly},
        )


class InvestigationTriage:
    """
    Ranks flagged transactions into an operational review queue for risk analysts.
    Prioritizes cases by monetary exposure and composite risk severity.
    """

    @staticmethod
    def rank_cases(df: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
        """Ranks transactions by priority_score descending."""
        if "priority_score" not in df.columns:
            if "risk_score" in df.columns and "Amount" in df.columns:
                df["priority_score"] = df["risk_score"] * np.log1p(df["Amount"].clip(lower=0))
            else:
                return df.head(top_n)

        return df.sort_values(by="priority_score", ascending=False).head(top_n).reset_index(drop=True)
