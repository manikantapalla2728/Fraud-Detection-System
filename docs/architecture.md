# End-to-End System Architecture

## 1. High-Level System Architecture Diagram

```
                              RAW FINANCIAL TRANSACTIONS (CSV)
                                           │
                                           ▼
                              DATA VALIDATION LAYER (src/data.py)
                              [Schema Integrity, Range Checks, Corruptions vs Extremes]
                                           │
                                           ▼
                              CHRONOLOGICAL SPLITTER (src/data.py)
                              [Train 70% | Val 15% | Test 15%]
                                           │
                                           ▼
                           LEAK-FREE FEATURE PIPELINE (src/features.py)
         ┌─────────────────────────────────┼─────────────────────────────────┐
         ▼                                 ▼                                 ▼
    Temporal Features               Amount Scaling & Stats            PCA Behavioral Deviations
    - Circadian sin/cos             - Log-amount transform            - Latent PCA norm (v_norm)
    - Inter-arrival delta           - Train-fitted z-score            - Risk polarity projection
    - Log inter-arrival             - Zero-dollar auth flag           - Extreme component flags
         └─────────────────────────────────┬─────────────────────────────────┘
                                           │
                                           ▼
                           SUPERVISED & UNSUPERVISED MODELS (src/models.py)
         ┌─────────────────────────────────┴─────────────────────────────────┐
         ▼                                                                   ▼
   Baseline Model (Logistic Regression)                             Primary XGBoost Classifier
   - Balanced Class Weights                                         - scale_pos_weight = 578
   - Regularized linear benchmark                                   - PR-AUC early stopping on Val
         │                                                                   │
         └─────────────────────────────────┬─────────────────────────────────┘
                                           ▼
                                  FRAUD PROBABILITY
                                           │
                                           ├─────────────────────────────────┐
                                           ▼                                 ▼
                               UNSUPERVISED ANOMALY LAYER           SHAP EXPLAINER (src/explainability.py)
                             (Isolation Forest Outlier Score)        - TreeExplainer exact attributions
                                           │                         - Positive/negative reason codes
                                           ▼                                 │
                              BUSINESS RISK ENGINE (src/risk.py)             │
                              Score = 100 * (0.80*ML + 0.20*Anomaly)         │
                              Priority Score = Score * ln(1 + Amount)        │
                                           │                                 │
                                           ▼                                 │
                                OPERATIONAL DECISION POLICY                  │
                                - LOW (<35): ALLOW                           │
                                - MEDIUM (35-70): FLAG_CHALLENGE             │
                                - HIGH (>=70): BLOCK_AND_INVESTIGATE         │
                                           │                                 │
                  ┌────────────────────────┴─────────────────────────────────┘
                  ▼                                                 ▼
        FASTAPI SERVING (api/main.py)                    STREAMLIT PORTAL (streamlit_app/app.py)
        - GET /health                                    - Executive Overview & Metrics
        - GET /model-info                                - Real-time Scoring Simulator
        - POST /predict                                  - Investigation Case Triage Queue
        - POST /batch-predict                            - Interactive SHAP Waterfall
```

---

## 2. Component Design & Responsibilities

### 1. Data Validation Layer (`src/data.py`)
- Distinguishes **Fatal Errors** (negative amounts, missing schema columns, corrupt non-binary labels, infinite values) from **Legitimate Extreme Behaviors** ($0 authorization checks, high-value spending).
- Partitions chronologically into 70% Train, 15% Validation, and 15% Test without temporal shuffling.

### 2. Leak-Free Feature Engineering (`src/features.py`)
- **Temporal**: Models continuous 24-hour cycles via $\sin(2\pi \cdot \text{hour}/24)$ and $\cos(2\pi \cdot \text{hour}/24)$, plus inter-arrival time deltas between successive stream events.
- **Amount Scaling**: Logarithmic compression $\ln(1 + \text{Amount})$ to handle heavy right skew, and $z$-score standardization fitted strictly on training data.
- **PCA Behavioral Polarities**: Computes Euclidean norm in PCA subspace ($\sqrt{\sum V_i^2}$) and targeted polarity projection along highest-separating dimensions:
  $$\text{v\_risk\_polarity} = (V_4 + V_{11} + V_2) - (V_{14} + V_{17} + V_{12} + V_{10})$$

### 3. Machine Learning Models (`src/models.py`)
- **Baseline (`BaselineModel`)**: Balanced regularized Logistic Regression. Demonstrates why naive linear classification floods false alarms under 0.17% imbalance without calibrated thresholds.
- **Primary Supervised Classifier (`XGBoostFraudModel`)**: Cost-sensitive gradient boosted trees configured with `scale_pos_weight = 578.0` and early stopping on validation PR-AUC. The operational threshold is calibrated strictly on holdout validation data.
- **Unsupervised Anomaly Detector (`AnomalyDetector`)**: Isolation Forest model scoring uncharacteristic behaviors between 0.0 (normal) and 1.0 (anomalous).

### 4. Explainability Layer (`src/explainability.py`)
- Employs **SHAP TreeExplainer** to compute exact Shapley values.
- Identifies top risk-increasing and risk-mitigating features, emitting plain-English reason codes for investigators.

### 5. Risk Engine & Operational Triage (`src/risk.py`)
- Synthesizes supervised probability and unsupervised anomaly into a composite 0-100 score.
- Categorizes transactions into `ALLOW`, `FLAG_CHALLENGE`, or `BLOCK_AND_INVESTIGATE`.
- Computes `Priority Score` ($\text{Risk Score} \times \ln(1 + \text{Amount})$) to rank review cases by monetary exposure.

### 6. Serving & User Interface
- **FastAPI (`api/main.py`)**: Asynchronous REST endpoints serving single and batch predictions in < 45 ms.
- **Streamlit (`streamlit_app/app.py`)**: Interactive risk simulator, performance dashboard, and case queue.
