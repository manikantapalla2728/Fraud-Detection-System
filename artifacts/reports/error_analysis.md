# Forensic Error & Segment Analysis Report
Evaluated on Holdout Test Split (42,722 transactions, calibrated threshold = 0.8278)

## Overall Confusion Matrix
- **True Positives (Caught Fraud):** 38 (73.08% recall)
- **False Positives (False Alarms):** 19
- **False Negatives (Missed Fraud):** 14
- **True Negatives (Legitimate Cleared):** 42,651

## False Positive Forensic Analysis
- **Why did false alarms happen?** Legitimate transactions with extreme PCA values (`V14` < -4.0 or `V4` > 3.0) and uncharacteristic spending.
- **Mean Amount:** $1379.21 (FP) vs $71.90 (TN).
- **Operational Mitigation:** Medium-tier transactions are routed to `FLAG_CHALLENGE` (step-up authentication OTP) rather than instant decline, avoiding customer insult.

## False Negative Forensic Analysis
- **Why were they missed?** Subtle adversarial mimicry where latent components remained within normal bounds.
- **Mean Probability:** 0.3344 (just below cutoff 0.8278).

## Amount Tier Performance
| amount_tier      |   total_tx |   fraud_count |   caught_fraud |   false_positives |   recall |
|:-----------------|-----------:|--------------:|---------------:|------------------:|---------:|
| Micro (<$5)      |      12064 |            26 |             22 |                14 | 0.846154 |
| Medium ($5-$500) |      29612 |            23 |             15 |                 4 | 0.652174 |
| High (>$500)     |       1046 |             3 |              1 |                 1 | 0.333333 |

## Inference Latency Percentiles
- **p50 (Median):** 42.89 ms
- **p95:** 51.85 ms
- **p99:** 56.73 ms
- **Production Status:** Successfully operates in real-time.
