"""
End-to-End Reproducible Training & Evaluation Pipeline.
Executes data validation, chronological 70/15/15 splitting, leak-free feature engineering,
model training (Baseline vs XGBoost vs Anomaly), threshold calibration,
SHAP explainability, and artifact serialization.
"""

import sys
import time
from pathlib import Path
import yaml
import numpy as np
import pandas as pd

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import DataLoader
from src.features import FeaturePipeline
from src.models import BaselineModel, XGBoostFraudModel, AnomalyDetector
from src.explainability import ModelExplainer
from src.risk import RiskEngine
from src.utils import save_artifact, save_json


def run_pipeline():
    start_time = time.time()
    print("=" * 70)
    print("STARTING REPRODUCIBLE FRAUD INTELLIGENCE TRAINING PIPELINE")
    print("=" * 70)

    # 1. Load Config
    config_path = Path("configs/config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    raw_path = config["paths"]["raw_data"]
    processed_dir = config["paths"]["processed_dir"]
    model_dir = Path(config["paths"]["model_dir"])
    reports_dir = Path(config["paths"]["reports_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 2. Data Validation & Temporal Split
    print("\n[Step 1/6] Ingesting, Validating, and Chronologically Splitting Data...")
    loader = DataLoader(raw_path=raw_path)
    split_summary = loader.process_and_save(
        output_dir=processed_dir,
        train_ratio=config["data"]["temporal_split"]["train_ratio"],
        val_ratio=config["data"]["temporal_split"]["val_ratio"],
        test_ratio=config["data"]["temporal_split"]["test_ratio"],
    )

    print(f"  Train: {split_summary['train_size']:,} records | Fraud rate: {split_summary['train_fraud_rate']:.4%}")
    print(f"  Val:   {split_summary['val_size']:,} records | Fraud rate: {split_summary['val_fraud_rate']:.4%}")
    print(f"  Test:  {split_summary['test_size']:,} records | Fraud rate: {split_summary['test_fraud_rate']:.4%}")

    train_df = pd.read_parquet(split_summary["files"]["train"])
    val_df = pd.read_parquet(split_summary["files"]["val"])
    test_df = pd.read_parquet(split_summary["files"]["test"])

    # 3. Leak-Free Feature Engineering
    print("\n[Step 2/6] Fitting Feature Pipeline on Training Data...")
    feat_pipeline = FeaturePipeline()
    feat_pipeline.fit(train_df)

    print(f"  Total engineered features: {len(feat_pipeline.feature_names_)}")
    print(f"  Feature list: {feat_pipeline.feature_names_[:10]} ... (and {len(feat_pipeline.feature_names_)-10} more)")

    print("  Transforming Train, Validation, and Test feature matrices...")
    X_train = feat_pipeline.transform(train_df)
    y_train = train_df["Class"].values.astype(int)

    X_val = feat_pipeline.transform(val_df)
    y_val = val_df["Class"].values.astype(int)

    X_test = feat_pipeline.transform(test_df)
    y_test = test_df["Class"].values.astype(int)

    X_train_scaled = feat_pipeline.scaler.transform(X_train)
    X_val_scaled = feat_pipeline.scaler.transform(X_val)
    X_test_scaled = feat_pipeline.scaler.transform(X_test)

    # 4. Train Baseline Model (Logistic Regression)
    print("\n[Step 3/6] Training Defensible Baseline (Balanced Regularized Logistic Regression)...")
    baseline = BaselineModel(
        C=config["models"]["baseline"]["C"],
        max_iter=config["models"]["baseline"]["max_iter"],
        random_state=config["system"]["random_seed"],
    )
    baseline.fit(X_train_scaled, y_train)
    baseline_val_metrics = baseline.evaluate(X_val_scaled, y_val)
    baseline_test_metrics = baseline.evaluate(X_test_scaled, y_test)
    print(f"  Baseline Val  PR-AUC: {baseline_val_metrics['pr_auc']:.4f} | ROC-AUC: {baseline_val_metrics['roc_auc']:.4f}")
    print(f"  Baseline Test PR-AUC: {baseline_test_metrics['pr_auc']:.4f} | ROC-AUC: {baseline_test_metrics['roc_auc']:.4f} | F1: {baseline_test_metrics['f1']:.4f}")

    # 5. Train XGBoost Model
    print("\n[Step 4/6] Training Primary Supervised XGBoost Classifier with Early Stopping...")
    xgb_cfg = config["models"]["xgboost"]
    xgb_model = XGBoostFraudModel(
        n_estimators=xgb_cfg["n_estimators"],
        max_depth=xgb_cfg["max_depth"],
        learning_rate=xgb_cfg["learning_rate"],
        subsample=xgb_cfg["subsample"],
        colsample_bytree=xgb_cfg["colsample_bytree"],
        scale_pos_weight=xgb_cfg["scale_pos_weight"],
        random_state=config["system"]["random_seed"],
        early_stopping_rounds=xgb_cfg["early_stopping_rounds"],
    )
    xgb_model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    print(f"  Optimal Threshold Calibrated on Val: {xgb_model.best_threshold:.4f}")
    xgb_val_metrics = xgb_model.evaluate(X_val, y_val)
    xgb_test_metrics = xgb_model.evaluate(X_test, y_test)

    print(f"  XGBoost Val  PR-AUC: {xgb_val_metrics['pr_auc']:.4f} | ROC-AUC: {xgb_val_metrics['roc_auc']:.4f}")
    print(f"  XGBoost Test PR-AUC: {xgb_test_metrics['pr_auc']:.4f} | ROC-AUC: {xgb_test_metrics['roc_auc']:.4f} | F1: {xgb_test_metrics['f1']:.4f}")
    print(f"  XGBoost Test Recall@100: {xgb_test_metrics.get('recall_at_100', 0):.4f} | Prec@100: {xgb_test_metrics.get('precision_at_100', 0):.4f}")
    print(f"  Confusion Matrix: {xgb_test_metrics['confusion_matrix']}")

    # 6. Fit Unsupervised Anomaly Detector
    print("\n[Step 5/6] Fitting Isolation Forest Anomaly Detection...")
    anomaly_detector = AnomalyDetector(
        n_estimators=config["models"]["anomaly"]["n_estimators"],
        contamination=config["models"]["anomaly"]["contamination"],
        random_state=config["system"]["random_seed"],
    )
    anomaly_detector.fit(X_train_scaled)
    anomaly_scores_test = anomaly_detector.score_anomaly(X_test_scaled)
    print(f"  Anomaly scores computed. Mean: {anomaly_scores_test.mean():.4f}, 99th percentile: {np.percentile(anomaly_scores_test, 99):.4f}")

    # 7. SHAP Explainability & Global Importance
    print("\n[Step 6/6] Computing SHAP Global Attributions and Serializing Artifacts...")
    explainer = ModelExplainer(xgb_model.model, feat_pipeline.feature_names_)
    shap_sample = X_test.sample(n=min(500, len(X_test)), random_state=42)
    global_shap_df = explainer.explain_global(shap_sample, max_features=15)

    print("  Top 5 SHAP Global Risk Drivers:")
    for _, row in global_shap_df.head(5).iterrows():
        print(f"    - {row['feature']}: {row['mean_abs_shap']:.4f}")

    # Serialize Models
    save_artifact(feat_pipeline, model_dir / "feature_pipeline.joblib")
    save_artifact(baseline, model_dir / "baseline_model.joblib")
    save_artifact(xgb_model, model_dir / "xgboost_model.joblib")
    save_artifact(anomaly_detector, model_dir / "anomaly_detector.joblib")

    # Serialize Reports
    global_shap_df.to_csv(reports_dir / "shap_global_importance.csv", index=False)
    feature_importance_df = xgb_model.get_feature_importances(feat_pipeline.feature_names_)
    feature_importance_df.to_csv(reports_dir / "xgboost_feature_importance.csv", index=False)

    summary_report = {
        "dataset_split": split_summary,
        "comparison_table": {
            "baseline": {
                "val_pr_auc": baseline_val_metrics["pr_auc"],
                "val_roc_auc": baseline_val_metrics["roc_auc"],
                "test_pr_auc": baseline_test_metrics["pr_auc"],
                "test_roc_auc": baseline_test_metrics["roc_auc"],
                "test_f1": baseline_test_metrics["f1"],
                "test_precision": baseline_test_metrics["precision"],
                "test_recall": baseline_test_metrics["recall"],
                "test_brier": baseline_test_metrics["brier_score"],
            },
            "xgboost": {
                "val_pr_auc": xgb_val_metrics["pr_auc"],
                "val_roc_auc": xgb_val_metrics["roc_auc"],
                "test_pr_auc": xgb_test_metrics["pr_auc"],
                "test_roc_auc": xgb_test_metrics["roc_auc"],
                "test_f1": xgb_test_metrics["f1"],
                "test_precision": xgb_test_metrics["precision"],
                "test_recall": xgb_test_metrics["recall"],
                "test_brier": xgb_test_metrics["brier_score"],
                "test_recall_at_100": xgb_test_metrics.get("recall_at_100", 0),
                "test_precision_at_100": xgb_test_metrics.get("precision_at_100", 0),
                "calibrated_threshold": xgb_model.best_threshold,
                "confusion_matrix": xgb_test_metrics["confusion_matrix"],
            },
        },
        "feature_count": len(feat_pipeline.feature_names_),
    }
    save_json(summary_report, reports_dir / "evaluation_summary.json")

    # Write Markdown comparison
    comparison_md = f"""# Model Performance Benchmark Comparison
Evaluated on Holdout Test Split ({split_summary['test_size']:,} transactions, {split_summary['test_fraud_rate']:.4%} fraud rate)

| Metric | Baseline (Logistic Regression) | Production XGBoost | Relative Improvement |
|---|---|---|---|
| **Validation PR-AUC** | {baseline_val_metrics['pr_auc']:.4f} | **{xgb_val_metrics['pr_auc']:.4f}** | +{(xgb_val_metrics['pr_auc'] - baseline_val_metrics['pr_auc']):.4f} |
| **Test PR-AUC** | {baseline_test_metrics['pr_auc']:.4f} | **{xgb_test_metrics['pr_auc']:.4f}** | +{(xgb_test_metrics['pr_auc'] - baseline_test_metrics['pr_auc']):.4f} |
| **Test ROC-AUC** | {baseline_test_metrics['roc_auc']:.4f} | **{xgb_test_metrics['roc_auc']:.4f}** | +{(xgb_test_metrics['roc_auc'] - baseline_test_metrics['roc_auc']):.4f} |
| **Test Precision** | {baseline_test_metrics['precision']:.4f} | **{xgb_test_metrics['precision']:.4f}** | **{(xgb_test_metrics['precision'] / max(baseline_test_metrics['precision'], 1e-4)):.1f}x** |
| **Test Recall** | {baseline_test_metrics['recall']:.4f} | **{xgb_test_metrics['recall']:.4f}** | Balanced operational trade-off |
| **Test F1 Score** | {baseline_test_metrics['f1']:.4f} | **{xgb_test_metrics['f1']:.4f}** | **{(xgb_test_metrics['f1'] / max(baseline_test_metrics['f1'], 1e-4)):.1f}x** |
| **Brier Score (Calibration)** | {baseline_test_metrics['brier_score']:.4f} | **{xgb_test_metrics['brier_score']:.4f}** | Lower is better |
| **Recall @ Top 100 Alerts** | N/A | **{xgb_test_metrics.get('recall_at_100', 0):.1%}** | High investigation efficiency |

### Test Confusion Matrix
- **True Negatives:** {xgb_test_metrics['confusion_matrix']['tn']:,}
- **False Positives:** {xgb_test_metrics['confusion_matrix']['fp']:,}
- **False Negatives:** {xgb_test_metrics['confusion_matrix']['fn']:,}
- **True Positives:** {xgb_test_metrics['confusion_matrix']['tp']:,}
"""
    with open(reports_dir / "model_comparison.md", "w") as f:
        f.write(comparison_md)

    duration = time.time() - start_time
    print(f"\n[DONE] Pipeline completed successfully in {duration:.1f} seconds.")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()
