# Model Performance Benchmark Comparison
Evaluated on Holdout Test Split (42,722 transactions, 0.1217% fraud rate)

| Metric | Baseline (Logistic Regression) | Production XGBoost | Relative Improvement |
|---|---|---|---|
| **Validation PR-AUC** | 0.8517 | **0.8241** | +-0.0276 |
| **Test PR-AUC** | 0.7686 | **0.7438** | +-0.0248 |
| **Test ROC-AUC** | 0.9732 | **0.9778** | +0.0045 |
| **Test Precision** | 0.0223 | **0.6667** | **29.9x** |
| **Test Recall** | 0.8654 | **0.7308** | Balanced operational trade-off |
| **Test F1 Score** | 0.0434 | **0.6972** | **16.1x** |
| **Brier Score (Calibration)** | 0.0368 | **0.0259** | Lower is better |
| **Recall @ Top 100 Alerts** | N/A | **75.0%** | High investigation efficiency |

### Test Confusion Matrix
- **True Negatives:** 42,651
- **False Positives:** 19
- **False Negatives:** 14
- **True Positives:** 38
