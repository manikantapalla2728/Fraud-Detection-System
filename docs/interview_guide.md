# Fraud Risk Intelligence: Senior ML Engineer Interview Guide

This guide prepares you to defend every architectural decision, mathematical formulation, and operational tradeoff in technical machine learning interviews with complete technical honesty and rigor.

---

### 1. One-Minute Pitch
> "I designed and implemented an end-to-end Fraud Risk Intelligence platform on the benchmark 284K-transaction credit card dataset. Rather than building a naive binary classifier, the system answers: *How risky is this transaction, why is it risky, is the behavior anomalous, and how should human risk analysts prioritize alerts?*
> 
> The architecture combines a strictly leak-free feature pipeline, a cost-sensitive XGBoost classifier (`scale_pos_weight=578`) with validation threshold calibration, an unsupervised Isolation Forest anomaly layer, exact SHAP TreeExplainer attributions, and a 3-tier business decision policy (`ALLOW`, `FLAG_CHALLENGE`, `BLOCK`). The system is served via FastAPI with sub-50ms latency and an interactive Streamlit investigation portal."

---

### 2. Five-Minute Deep Dive
> "When building financial fraud detection, Kaggle-style approaches fail because they ignore four real-world operational realities: extreme class imbalance (578:1), future data leakage, asymmetric business error costs, and human investigator capacity constraints.
>
> In this project, I addressed each:
> 1. **Temporal Validation**: I enforced a strictly chronological 70/15/15 split sorted by transaction timestamp. No future transactions are ever visible to past training records.
> 2. **Leak-Free Transformations**: All scalers and behavioral parameters (mean amount, standard deviation) are fitted strictly on the training partition. Feature transformations include circadian hour encodings ($\sin/\cos$), log-amount scaling, inter-arrival time deltas, and PCA risk polarities.
> 3. **Model Formulation**: I trained a gradient boosted tree ensemble using `scale_pos_weight = 578` with early stopping on validation PR-AUC. The model achieved a **0.7438 Test PR-AUC** (0.9778 ROC-AUC) and **0.6972 F1**, capturing **75% of all frauds in the top 100 alerts**. In contrast, our regularized balanced logistic regression baseline collapsed to an F1 of 0.0434 at the default 0.5 threshold, proving why non-linear interactions and threshold calibration are required.
> 4. **Defense in Depth**: We combine supervised XGBoost with unsupervised Isolation Forest to flag uncharacteristic transactions (zero-day fraud) even if supervised probabilities are borderline.
> 5. **Risk Engine & Operations**: Model probability is fused with the anomaly score into a 0-100 composite score. Alerts are prioritized for investigators by monetary exposure ($\text{Priority Score} = \text{Risk Score} \times \ln(1 + \text{Amount})$)."

---

### 3. Dataset Scope & Honest Architectural Boundaries
- **Interviewer Question**: *"Did your dataset support graph-based fraud ring detection (shared device farms, IP clusters)?"*
- **Your Answer**:
  > *"No, and that is a critical architectural distinction. The benchmark European credit card dataset contains anonymized PCA features ($V_1$–$V_{28}$), `Time`, and `Amount`, but intentionally strips entity identifiers (`account_id`, `device_id`, `ip_address`, `card_id`) to comply with privacy regulations.
  >
  > Therefore, claiming to detect 'real device farm fraud rings' directly from this dataset would be technically inaccurate. Instead, I focused our core engine on transaction-level risk intelligence, temporal features, and anomaly detection.
  >
  > In a production banking environment with entity logs (e.g. Device Fingerprint SDK, IP Geo-location), the architecture naturally extends by constructing a bipartite entity graph (Accounts ↔ Devices ↔ IPs) to detect shared infrastructure rings and propagate risk boosts."*

---

### 4. Why XGBoost Over Deep Learning?
- **Tabular Data Superiority**: Tabular transaction data features heterogeneous distributions, extreme skew, and sharp decision thresholds. Gradient boosted trees consistently outperform deep neural networks on tabular benchmarks.
- **Inference Latency & Determinism**: Tree inference via histogram binning executes in single-digit milliseconds on CPU without requiring GPU infrastructure or complex tensor runtimes.
- **Native Imbalance Handling**: Native loss weighting via `scale_pos_weight` directly scales gradient and hessian penalties for positive instances without generating artificial synthetic samples (SMOTE).

---

### 5. Why Chronological Validation Instead of Random Split?
- **Time Is One-Way in Payments**: In real-world payment networks, you only ever predict on the future using the past.
- **Random Train/Test Splitting Is Leakage**: Random $k$-fold cross-validation allows models to train on transactions from hour 40 and predict on transactions from hour 10. This falsely inflates performance because the model memorizes temporal patterns that had not yet occurred in the past.
- **Our Strategy**: Sorted strictly by timestamp. Train: first 70% of time; Validation: next 15% (used solely for threshold calibration and early stopping); Test: final 15% (unseen holdout).

---

### 6. Why These Specific Engineered Features?
- **Circadian Sine & Cosine**: Hour 23 (11:00 PM) and Hour 0 (midnight) are adjacent in reality but have a difference of 23 in raw numbers. Sine/cosine cyclical encoding maps them continuously on the unit circle.
- **Risk Polarity**:
  $$\text{v\_risk\_polarity} = (V_4 + V_{11} + V_2) - (V_{14} + V_{17} + V_{12} + V_{10})$$
  Identified through empirical correlation audit: directly projects latent embeddings along the optimal fraud-separation hyper-axis. SHAP confirmed this as the single most impactful global feature.
- **Latent Vector Euclidean Norm**: $\|V\| = \sqrt{\sum V_i^2}$. Measures total vector displacement from routine transaction variance.

---

### 7. Why Unsupervised Anomaly Detection (Isolation Forest)?
- **Zero-Day Attack Defense**: Supervised models only recognize historical labeled fraud patterns. If an adversary deploys a novel attack vector, supervised classifiers may assign a lower probability.
- **Isolation Forest Role**: Computes average tree path length to isolate points. Anomalous transactions are isolated in few splits.
- **Defense in Depth**: Blending supervised probability (80%) with anomaly score (20%) ensures anomalous outliers reach the **MEDIUM / CHALLENGE** tier for step-up verification.

---

### 8. How Did You Choose Operating Thresholds?
- **Default 0.5 Threshold Is Flawed**: Under 578:1 imbalance, a standard 0.5 threshold produces severe false positive flooding.
- **F1 Calibration on Validation**: We evaluated the Precision-Recall curve across candidate thresholds strictly on the validation split and calibrated the optimal operating threshold to **0.8278**.
- **Three-Tier Policy**:
  - `Risk Score < 35`: **ALLOW** (frictionless authorization)
  - `35 <= Risk Score < 70`: **FLAG_CHALLENGE** (3D Secure OTP step-up authentication)
  - `Risk Score >= 70`: **BLOCK_AND_INVESTIGATE** (hard decline + queued for human review)

---

### 9. What Were the Biggest Model Errors?
- **False Positives (19 out of 42,670 non-frauds)**:
  - Legitimate transactions exhibiting extreme values in `V14` (< -5.0) and `V4` (> 3.5), overlapping with fraud distributions.
  - Mean amount was $1,379.21 (unusual spending) vs normal $71.90.
- **False Negatives (14 out of 52 test frauds missed)**:
  - Subtle mimicry: Transactions that blended closely with typical cardholder amounts (mean: $172.31) and kept latent components within normal bounds.
  - Mean probability was 0.3344, below the 0.8278 cutoff. In production, these are caught by secondary rules and step-up challenge thresholds.
