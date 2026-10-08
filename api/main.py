"""
FastAPI Fraud Risk Intelligence Scoring Microservice.
Provides real-time transaction scoring, multi-signal risk fusion, and local SHAP explainability.
"""

from contextlib import asynccontextmanager
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src import (
    FeaturePipeline,
    XGBoostFraudModel,
    AnomalyDetector,
    ModelExplainer,
    RiskEngine,
    load_artifact,
    load_json,
)


# ============================================================
# Pydantic Request & Response Data Contracts
# ============================================================

class TransactionInput(BaseModel):
    transaction_id: Optional[str] = Field("TX_DEMO_001", description="Unique transaction identifier")
    Time: float = Field(..., ge=0.0, description="Elapsed seconds from epoch / observation window")
    Amount: float = Field(..., ge=0.0, description="Transaction amount in currency units (must be >= 0)")
    # PCA components (V1 to V28)
    V1: float = Field(0.0)
    V2: float = Field(0.0)
    V3: float = Field(0.0)
    V4: float = Field(0.0)
    V5: float = Field(0.0)
    V6: float = Field(0.0)
    V7: float = Field(0.0)
    V8: float = Field(0.0)
    V9: float = Field(0.0)
    V10: float = Field(0.0)
    V11: float = Field(0.0)
    V12: float = Field(0.0)
    V13: float = Field(0.0)
    V14: float = Field(0.0)
    V15: float = Field(0.0)
    V16: float = Field(0.0)
    V17: float = Field(0.0)
    V18: float = Field(0.0)
    V19: float = Field(0.0)
    V20: float = Field(0.0)
    V21: float = Field(0.0)
    V22: float = Field(0.0)
    V23: float = Field(0.0)
    V24: float = Field(0.0)
    V25: float = Field(0.0)
    V26: float = Field(0.0)
    V27: float = Field(0.0)
    V28: float = Field(0.0)


class BatchTransactionRequest(BaseModel):
    transactions: List[TransactionInput]


class ContributingFactor(BaseModel):
    feature: str
    shap_value: float
    feature_value: float
    impact: str


class PredictionResponse(BaseModel):
    transaction_id: str
    risk_score: float = Field(..., description="Unified composite risk score between 0 and 100")
    risk_level: str = Field(..., description="Risk tier: LOW, MEDIUM, or HIGH")
    action: str = Field(..., description="Operational action: ALLOW, FLAG_CHALLENGE, or BLOCK_AND_INVESTIGATE")
    ml_fraud_probability: float
    anomaly_score: float
    priority_score: float
    primary_reasons: List[str]
    top_risk_drivers: List[ContributingFactor]
    latency_ms: float


class BatchPredictionResponse(BaseModel):
    total_processed: int
    predictions: List[PredictionResponse]
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    model_version: str
    loaded_artifacts: List[str]


class ModelInfoResponse(BaseModel):
    model_type: str
    calibrated_threshold: float
    feature_count: int
    feature_names: List[str]
    benchmark_metrics: Dict[str, Any]


# ============================================================
# Inference Service Class
# ============================================================

class FraudInferenceService:
    """Manages model artifacts and executes end-to-end scoring workflows."""

    def __init__(self, artifacts_dir: str = "artifacts"):
        self.artifacts_dir = Path(artifacts_dir)
        self.models_dir = self.artifacts_dir / "models"
        self.reports_dir = self.artifacts_dir / "reports"

        self.feat_pipeline: Optional[FeaturePipeline] = None
        self.xgb_model: Optional[XGBoostFraudModel] = None
        self.anomaly_detector: Optional[AnomalyDetector] = None
        self.explainer: Optional[ModelExplainer] = None
        self.risk_engine = RiskEngine()

        self.eval_summary: Dict[str, Any] = {}
        self.is_ready = False
        self._load_artifacts()

    def _load_artifacts(self):
        try:
            self.feat_pipeline = load_artifact(self.models_dir / "feature_pipeline.joblib")
            self.xgb_model = load_artifact(self.models_dir / "xgboost_model.joblib")
            self.anomaly_detector = load_artifact(self.models_dir / "anomaly_detector.joblib")
            self.explainer = ModelExplainer(self.xgb_model.model, self.feat_pipeline.feature_names_)

            eval_path = self.reports_dir / "evaluation_summary.json"
            if eval_path.exists():
                self.eval_summary = load_json(eval_path)

            self.is_ready = True
        except Exception as e:
            print(f"Error loading model artifacts: {e}")
            self.is_ready = False

    def predict_single(self, tx: TransactionInput) -> PredictionResponse:
        t0 = time.perf_counter()
        raw_dict = tx.model_dump()
        tx_id = raw_dict.get("transaction_id") or "TX_LIVE"
        amount = float(raw_dict.get("Amount", 0.0))

        df_row = pd.DataFrame([raw_dict])
        feats = self.feat_pipeline.transform(df_row)
        scaled_feats = self.feat_pipeline.scaler.transform(feats)

        ml_prob = float(self.xgb_model.predict_proba(feats)[0])
        anomaly_score = float(self.anomaly_detector.score_anomaly(scaled_feats)[0])

        decision = self.risk_engine.evaluate(
            ml_probability=ml_prob,
            anomaly_score=anomaly_score,
            amount=amount,
        )

        explanation = self.explainer.explain_local(feats, top_k=3)
        top_factors = [
            ContributingFactor(
                feature=f["feature"],
                shap_value=f["shap_value"],
                feature_value=f["feature_value"],
                impact=f["impact"],
            )
            for f in explanation["top_risk_drivers"]
        ]

        t1 = time.perf_counter()
        lat_ms = round((t1 - t0) * 1000.0, 2)

        return PredictionResponse(
            transaction_id=tx_id,
            risk_score=decision.risk_score,
            risk_level=decision.risk_level,
            action=decision.action,
            ml_fraud_probability=decision.ml_probability,
            anomaly_score=decision.anomaly_score,
            priority_score=decision.priority_score,
            primary_reasons=decision.primary_reasons,
            top_risk_drivers=top_factors,
            latency_ms=lat_ms,
        )

    def predict_batch(self, transactions: List[TransactionInput]) -> BatchPredictionResponse:
        t0 = time.perf_counter()
        results = [self.predict_single(tx) for tx in transactions]
        t1 = time.perf_counter()
        total_ms = round((t1 - t0) * 1000.0, 2)

        return BatchPredictionResponse(
            total_processed=len(results),
            predictions=results,
            processing_time_ms=total_ms,
        )


# ============================================================
# FastAPI App Definition with Modern Lifespan Handler
# ============================================================

_service_instance: Optional[FraudInferenceService] = None
_start_time: float = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _service_instance, _start_time
    _start_time = time.time()
    _service_instance = FraudInferenceService(artifacts_dir="artifacts")
    yield


app = FastAPI(
    title="Fraud Risk Intelligence & Detection API",
    description="Real-time transaction risk scoring, behavioral anomaly detection, and SHAP explainability.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_service() -> FraudInferenceService:
    if _service_instance is None or not _service_instance.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine is not fully initialized. Ensure artifacts exist.",
        )
    return _service_instance


# ============================================================
# API Endpoints
# ============================================================

@app.get("/health", response_model=HealthResponse, tags=["System Health"])
async def health_check():
    svc = get_service()
    loaded = ["feature_pipeline", "xgboost_model", "anomaly_detector", "shap_explainer"]
    return HealthResponse(
        status="healthy" if svc.is_ready else "unhealthy",
        uptime_seconds=round(time.time() - _start_time, 1),
        model_version="1.0.0",
        loaded_artifacts=loaded,
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model Metadata"])
async def get_model_info():
    svc = get_service()
    metrics = svc.eval_summary.get("comparison_table", {}).get("xgboost", {})
    return ModelInfoResponse(
        model_type="XGBoost + Isolation Forest Anomaly Fusion",
        calibrated_threshold=svc.xgb_model.best_threshold,
        feature_count=len(svc.feat_pipeline.feature_names_),
        feature_names=svc.feat_pipeline.feature_names_,
        benchmark_metrics=metrics,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Scoring"])
async def predict_single(transaction: TransactionInput):
    svc = get_service()
    return svc.predict_single(transaction)


@app.post("/batch-predict", response_model=BatchPredictionResponse, tags=["Scoring"])
async def predict_batch(request: BatchTransactionRequest):
    svc = get_service()
    return svc.predict_batch(request.transactions)
