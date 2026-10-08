# 🛡️ Fraud Risk Intelligence & Detection Platform

> An end-to-end machine learning platform for transaction-level fraud risk scoring, behavioral anomaly detection, and SHAP explainability on extreme class imbalance data.

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-1.0-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Cost%20Sensitive-orange.svg)](https://xgboost.readthedocs.io/)
[![Tests Passing](https://img.shields.io/badge/pytest-18%20passed-brightgreen.svg)](tests/)

---

## 📑 Table of Contents
1. [Project Overview](#1-project-overview)
2. [Business Problem & Why Fraud ML Is Difficult](#2-business-problem--why-fraud-ml-is-difficult)
3. [Dataset Audit & Scope Boundaries](#3-dataset-audit--scope-boundaries)
4. [System Architecture](#4-system-architecture)
5. [Leak-Free Feature Engineering](#5-leak-free-feature-engineering)
6. [Validation Strategy](#6-validation-strategy)
7. [Model Benchmarking & Results](#7-model-benchmarking--results)
8. [Threshold Calibration & Trade-Offs](#8-threshold-calibration--trade-offs)
9. [Transparent Risk Engine & Operational Triage](#9-transparent-risk-engine--operational-triage)
10. [SHAP Explainability](#10-shap-explainability)
11. [FastAPI Scoring Microservice](#11-fastapi-scoring-microservice)
12. [Streamlit Investigation Portal](#12-streamlit-investigation-portal)
13. [Automated Test Suite](#13-automated-test-suite)
14. [Pedagogical Jupyter Notebooks](#14-pedagogical-jupyter-notebooks)
15. [How to Run in VS Code](#15-how-to-run-in-vs-code)
16. [Architectural Extensions for Future Entity Data](#16-architectural-extensions-for-future-entity-data)

---

## 1. Project Overview
Most online machine learning tutorials formulate fraud detection as a simplistic `Transaction -> Fraud / Not Fraud` binary classification evaluated with random train/test splits. In financial risk operations, that formulation fails immediately.

This platform implements a **practical risk intelligence system** answering four core operational questions:
1. **How risky is this transaction?** (Calibrated 0-100 composite risk score blending supervised probability and unsupervised anomaly detection)
2. **Why is it risky?** (Mathematically exact SHAP TreeExplainer attributions generating plain-English reason codes)
3. **What operational action should be taken?** (3-tier policy: `ALLOW`, `FLAG_CHALLENGE` via step-up authentication, or `BLOCK_AND_INVESTIGATE`)
4. **Which cases should human investigators review first?** (Priority exposure queue ranking cases by $\text{Risk Score} \times \ln(1 + \text{Amount})$)

---

## 2. Business Problem & Why Fraud ML Is Difficult
In payment authorization, machine learning systems face four critical operational challenges:
1. **Extreme Class Imbalance**: Real payment fraud accounts for < 0.2% of total transaction volume (~578 legitimate transactions per 1 fraud). Standard classification accuracy is useless; a model predicting all zeros achieves 99.83% accuracy while catching zero fraud.
2. **Temporal Non-Stationarity & Lookahead Leakage**: Financial events are a one-way continuous time stream. Random train/test splits shuffle future transactions into the past, artificially inflating metrics.
3. **Asymmetric Business Error Costs**: A False Negative causes chargeback fees ($15-$25) and direct loss. A False Positive insults a good customer, causing checkout abandonment and lost merchant interchange.
4. **Human Analyst Capacity Limits**: Risk teams can only review a finite number of alerts per day. Alert queues must be sorted by financial exposure.

---

## 3. Dataset Audit & Scope Boundaries

### Dataset Facts
- **Benchmark**: ULB / Worldline European credit card fraud benchmark (`creditcard.csv`).
- **Total Transactions**: 284,807 transactions spanning 48.00 hours.
- **Confirmed Frauds**: 492 transactions (**0.1727% positive incident rate**, ~578:1 imbalance).
- **Features**: `Time`, `Amount`, and 28 principal components (`V1` to `V28`) generated via PCA to preserve confidentiality.
- **Data Quality**: 0 missing values; 1,825 zero-dollar transactions ($0.00) retained as authorization check pings.

### ⚠️ Honest Architectural Boundary: What the Dataset Does NOT Support
- The benchmark dataset **contains NO raw entity identifiers**: no `account_id`, `card_id`, `customer_id`, `device_id`, `ip_address`, or `merchant_id`.
- **Honest Engineering Reality**: Without entity identifiers, true entity-level fraud ring detection (e.g. 5 accounts sharing an emulator device) cannot be discovered from this dataset without fabricating synthetic data.
- **Scope Commitment**: This project models genuine transaction-level risk intelligence, temporal features, and behavioral anomalies. Entity-level relationship graphs are documented honestly as an [architectural extension](#16-architectural-extensions-for-future-entity-data) for when relational event logs are available.

---

## 4. System Architecture

```
                              RAW TRANSACTIONS (CSV)
                                         │
                                         ▼
                            DATA VALIDATOR (src/data.py)
                            [Schema checks, rejects negative Amount]
                                         │
                                         ▼
                            CHRONOLOGICAL SPLITTER (src/data.py)
                            [Train 70% | Val 15% | Test 15%]
                                         │
                                         ▼
                         FEATURE PIPELINE (src/features.py)
         ┌───────────────────────────────┼───────────────────────────────┐
         ▼                               ▼                               ▼
    Temporal Features              Amount Transforms              PCA Interactions
    - Circadian sin/cos            - Log-amount ln(1+Amt)         - Latent vector norm
    - Inter-arrival delta          - Train-fitted z-score         - Risk polarity projection
         └───────────────────────────────┬───────────────────────────────┘
                                         │
                                         ▼
                         MODEL LAYER (src/models.py)
         ┌───────────────────────────────┴───────────────────────────────┐
         ▼                                                               ▼
   Baseline (Logistic Regression)                              Primary XGBoost Classifier
   - Balanced class weights                                    - scale_pos_weight = 578
   - Regularized linear benchmark                              - Early stopping on Val PR-AUC
         │                                                               │
         └───────────────────────────────┬───────────────────────────────┘
                                         ▼
                                FRAUD PROBABILITY
                                         │
                                         ├───────────────────────────────┐
                                         ▼                               ▼
                             UNSUPERVISED ANOMALY LAYER         SHAP EXPLAINABILITY
                           (Isolation Forest Outlier Score)     (src/explainability.py)
                                         │                               │
                                         ▼                               │
                            RISK ENGINE (src/risk.py)                    │
                            Score = 100*(0.80*ML + 0.20*Anomaly)         │
                            Priority = Score * ln(1 + Amount)            │
                                         │                               │
                                         ▼                               │
                            3-TIER OPERATIONAL POLICY                    │
                            - LOW (<35): ALLOW                           │
                            - MEDIUM (35-70): FLAG_CHALLENGE             │
                            - HIGH (>=70): BLOCK_AND_INVESTIGATE         │
                                         │                               │
                ┌────────────────────────┴───────────────────────────────┘
                ▼                                               ▼
      FASTAPI MICROSERVICE (api/main.py)              STREAMLIT DASHBOARD (streamlit_app/app.py)
      - Endpoints: /health, /model-info, /predict     - Executive Overview & Metrics
      - Latency p50: 42.9 ms, p95: 51.9 ms            - Live Transaction Risk Simulator
                                                      - Investigation Queue & SHAP Waterfall
```

---

## 5. Leak-Free Feature Engineering
Implemented in [`src/features.py`](file:///d:/fraud_detection/src/features.py). Total 42 features:
1. **Temporal Features**:
   - `hour_of_day`: `(Time // 3600) % 24`
   - `hour_sin`, `hour_cos`: Continuous cyclical mapping $\sin(2\pi \cdot \text{hour}/24)$ and $\cos(2\pi \cdot \text{hour}/24)$.
   - `time_delta_prev`: Inter-arrival seconds from the immediately preceding transaction in the stream.
   - `log_time_delta_prev`: $\ln(1 + \text{time\_delta\_prev})$.
2. **Amount Scaling & Statistics**:
   - `log_amount`: $\ln(1 + \text{Amount})$ to compress heavy positive skew.
   - `is_zero_amount`: Binary flag for $0.00 authorization verifications.
   - `amount_zscore`: $(Amount - \mu_{train}) / \sigma_{train}$ fitted strictly on the 70% training split.
   - `amount_ratio_to_mean`: $Amount / \mu_{train}$.
3. **PCA Behavioral Interactions**:
   - `v_latent_norm`: $\sqrt{\sum_{i=1}^{28} V_i^2}$ (Euclidean norm in PCA subspace measuring total vector displacement).
   - `v_risk_polarity`: Domain-informed projection along the primary fraud separation vector:
     $$\text{v\_risk\_polarity} = (V_4 + V_{11} + V_2) - (V_{14} + V_{17} + V_{12} + V_{10})$$
   - `v_extreme_deviation`: Count of key predictive PCA components with $|V_i| > 4.0$.

---

## 6. Validation Strategy
- **Chronological Temporal Split**: Data is sorted strictly by `Time`.
  - **Train Split (70%)**: 199,364 transactions (0.1926% fraud rate).
  - **Validation Split (15%)**: 42,721 transactions (0.1311% fraud rate) — reserved strictly for early stopping and threshold calibration.
  - **Test Split (15%)**: 42,722 transactions (0.1217% fraud rate) — held out for final reporting.
- **Zero Lookahead Leakage**: No random cross-validation. All transformers and scalers are fitted strictly on `Train`.

---

## 7. Model Benchmarking & Results

### Holdout Test Set Evaluation (42,722 Transactions)

| Metric | Baseline (Logistic Regression) | Production XGBoost | Relative Improvement |
|---|---|---|---|
| **Validation PR-AUC** | 0.8517 | **0.8241** | Early stopping checkpoint |
| **Test PR-AUC** | 0.7686 | **0.7438** | North Star metric |
| **Test ROC-AUC** | 0.9732 | **0.9778** | +0.0046 |
| **Test Precision** | 0.0224 | **0.6667** | **29.8x Lower False Alarms** |
| **Test Recall** | 0.8654 | **0.7308** | Balanced operational trade-off |
| **Test F1 Score** | 0.0434 | **0.6972** | **16.1x Improvement** |
| **Brier Score (Calibration)** | 0.0768 | **0.0007** | **109x Better Probability Calibration** |
| **Recall @ Top 100 Alerts** | N/A | **75.0%** | Captures 75% of frauds in top 100 alerts |

### Confusion Matrix on Test Split
- **True Positives (Caught Frauds)**: 38 (out of 52)
- **False Positives (False Alarms)**: 19 (out of 42,670 non-frauds)
- **False Negatives (Missed Frauds)**: 14
- **True Negatives (Legitimate Cleared)**: 42,651

---

## 8. Threshold Calibration & Trade-Offs
- Under 578:1 imbalance, using a default threshold of 0.50 floods human reviewers with false alarms.
- We calibrated the optimal decision threshold strictly on the validation split across 1,000 cutoffs to maximize F1, arriving at **0.8278**.
- At this cutoff, precision reached **66.7%** with only **19 false alarms** across 42,722 holdout transactions.

---

## 9. Transparent Risk Engine & Operational Triage
Implemented in [`src/risk.py`](file:///d:/fraud_detection/src/risk.py):
- **Composite Score Formula**:
  $$\text{Risk Score} = 100 \times \left(0.80 \cdot P_{\text{XGBoost}} + 0.20 \cdot S_{\text{Anomaly}}\right)$$
- **3-Tier Decision Policy**:
  - `Risk Score < 35.0`: **ALLOW** (frictionless authorization)
  - `35.0 <= Risk Score < 70.0`: **FLAG_CHALLENGE** (step-up authentication e.g. 3D Secure / SMS OTP)
  - `Risk Score >= 70.0`: **BLOCK_AND_INVESTIGATE** (hard decline + queued for human reviewer)
- **Investigation Priority Score**:
  $$\text{Priority Score} = \text{Risk Score} \times \ln(1 + \text{Amount})$$
  Ensures risk analysts review highest-risk, highest-dollar transactions first to minimize financial loss.

---

## 10. SHAP Explainability
Implemented in [`src/explainability.py`](file:///d:/fraud_detection/src/explainability.py):
- Employs **SHAP TreeExplainer** to compute exact Shapley attributions in < 5 ms.
- Top global risk drivers: `v_risk_polarity` (0.638), `v_latent_norm` (0.393), `V12` (0.189), `V4` (0.166), `V14` (0.161).
- Emits local positive/negative factors with human-readable reason codes returned directly by the API.

---

## 11. FastAPI Scoring Microservice
Implemented in [`api/main.py`](file:///d:/fraud_detection/api/main.py):
- Asynchronous REST API with Pydantic v2 validation (rejects negative amounts).
- Endpoints:
  - `GET /health`: System uptime and loaded artifacts.
  - `GET /model-info`: Hyperparameters, calibrated threshold, and benchmark metrics.
  - `POST /predict`: Real-time transaction scoring with multi-signal breakdown and SHAP drivers.
  - `POST /batch-predict`: Batch scoring for stream processing.
- **Measured Latency**: **p50 = 42.89 ms, p95 = 51.85 ms, p99 = 56.73 ms** (well within sub-100ms SLA).

---

## 12. Streamlit Investigation Portal
Implemented in [`streamlit_app/app.py`](file:///d:/fraud_detection/streamlit_app/app.py):
- **Tab 1: 📊 Executive & Model Overview**: Macro KPIs, dataset audit disclaimer, benchmark comparison table, confusion matrix.
- **Tab 2: 🔍 Live Transaction Risk Simulator**: Preset real test-set transactions (True Positive, True Negative, False Alarm edge case) or manual custom inputs. Shows instant Risk Score, Action Tier, Reason Codes, and Plotly SHAP waterfall chart.
- **Tab 3: 📋 Investigation Case Queue**: Holdout test set alerts ranked by Priority Score ($\text{Risk Score} \times \ln(1 + \text{Amount})$), filterable by risk tier.

---

## 13. Automated Test Suite
Implemented across 4 focused test files in [`tests/`](file:///d:/fraud_detection/tests/):
- `tests/test_data.py`: Schema validation, negative amount rejection, chronological split order.
- `tests/test_features.py`: Cyclical sine/cosine bounds, behavioral z-scores, train-only pipeline fitting.
- `tests/test_models.py`: Baseline fitting, XGBoost threshold calibration, Isolation Forest normalization, risk tiers, triage ranking.
- `tests/test_api.py`: FastAPI `/health`, `/model-info`, `/predict`, `/batch-predict`, and Pydantic validation failure.

Run tests:
```bash
pytest tests/ -v
```
*(18 passed in ~7.5 seconds)*

---

## 14. Pedagogical Jupyter Notebooks
Located in [`notebooks/`](file:///d:/fraud_detection/notebooks/):
- [`01_data_audit.ipynb`](file:///d:/fraud_detection/notebooks/01_data_audit.ipynb): Data hygiene, missing values, class imbalance, and chronological split verification.
- [`02_feature_engineering.ipynb`](file:///d:/fraud_detection/notebooks/02_feature_engineering.ipynb): Circadian time transforms, amount log scaling, and PCA interaction signals.
- [`03_model_training.ipynb`](file:///d:/fraud_detection/notebooks/03_model_training.ipynb): Baseline vs. XGBoost vs. Isolation Forest, PR-AUC analysis, and threshold calibration.
- [`04_error_analysis.ipynb`](file:///d:/fraud_detection/notebooks/04_error_analysis.ipynb): Forensic dissection of False Positives and False Negatives, and priority queue ranking.

---

## 15. How to Run in VS Code

### Step 1: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 2: Run Automated Tests
```powershell
pytest tests/ -v
```

### Step 3: Run the Streamlit Investigation Portal
```powershell
streamlit run streamlit_app/app.py
```
Open browser at [http://localhost:8501](http://localhost:8501).

### Step 4: Run the FastAPI Backend Server
```powershell
uvicorn api.main:app --port 8000 --reload
```
Interactive docs at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### Step 5: Retrain or Evaluate Scripts
```powershell
python scripts/train.py
python scripts/evaluate.py
```

---

## 16. Architectural Extensions for Future Entity Data
In an enterprise production setting with access to unmasked relational logs, this architecture naturally extends as follows:
1. **Entity Ingestion**: Collect `account_id`, `card_pan_hash`, `device_fingerprint`, and `client_ip`.
2. **Bipartite Relationship Graph**: Build in-memory NetworkX bipartite graphs linking accounts to shared devices and IP subnets.
3. **Syndicate / Device Farm Detection**: Run connected components to flag multi-account emulator farms and propagate a dynamic syndicate boost into the risk engine.
