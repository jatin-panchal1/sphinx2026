# FairScore Model Evaluation Report

Generated: 2026-10-31T09:26:27.561196

---

## Dataset: `australian`

- **File:** `D:\fairscore (1)\fairscore\data\raw\australian.dat`
- **Rows:** 690
- **Features:** 14
- **Target:** `feat_14`
- **Model:** RandomForestClassifier
- **Actual Default Rate:** 44.20%

### Test-Set Performance

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.9236 |
| PR-AUC | 0.8611 |
| Accuracy | 0.8261 |
| Precision | 0.7534 |
| Recall | 0.9016 |
| F1 Score | 0.8209 |

### Approval Rates

| Threshold | Approval Rate |
|-----------|---------------|
| 0.35 | 44.20% |
| 0.5 | 47.10% |
| 0.65 | 59.42% |

### Cross-Validation (5-Fold Stratified)

| Metric | Mean | Std |
|--------|------|-----|
| ROC-AUC | 0.9317 | 0.0209 |
| F1 | 0.8474 | 0.0416 |

### Top Feature Importances

| Feature | Importance |
|---------|------------|
| num__feat_0 | 0.0417 |
| num__feat_1 | 0.0417 |

---

## Dataset: `crx`

- **File:** `D:\fairscore (1)\fairscore\data\raw\crx.data`
- **Rows:** 689
- **Features:** 15
- **Target:** `+`
- **Model:** RandomForestClassifier
- **Actual Default Rate:** 55.80%

### Test-Set Performance

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.9109 |
| PR-AUC | 0.9122 |
| Accuracy | 0.8478 |
| Precision | 0.8889 |
| Recall | 0.8312 |
| F1 Score | 0.8591 |

### Approval Rates

| Threshold | Approval Rate |
|-----------|---------------|
| 0.35 | 40.58% |
| 0.5 | 47.83% |
| 0.65 | 51.45% |

### Cross-Validation (5-Fold Stratified)

| Metric | Mean | Std |
|--------|------|-----|
| ROC-AUC | 0.9270 | 0.0245 |
| F1 | 0.8815 | 0.0356 |

### Top Feature Importances

| Feature | Importance |
|---------|------------|
| num__1.25 | 0.0013 |
| num__feat_0 | 0.0013 |

---

## Dataset: `default_of_credit_card_clients`

- **File:** `D:\fairscore (1)\fairscore\data\raw\default of credit card clients.xls`
- **Rows:** 20,000
- **Features:** 24
- **Target:** `default payment next month`
- **Model:** RandomForestClassifier
- **Actual Default Rate:** 21.88%

### Test-Set Performance

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.7709 |
| PR-AUC | 0.5409 |
| Accuracy | 0.8170 |
| Precision | 0.6544 |
| Recall | 0.3463 |
| F1 Score | 0.4529 |

### Approval Rates

| Threshold | Approval Rate |
|-----------|---------------|
| 0.35 | 80.23% |
| 0.5 | 88.42% |
| 0.65 | 94.23% |

### Cross-Validation (5-Fold Stratified)

| Metric | Mean | Std |
|--------|------|-----|
| ROC-AUC | 0.7730 | 0.0030 |
| F1 | 0.4519 | 0.0162 |

### Top Feature Importances

| Feature | Importance |
|---------|------------|
| num__LIMIT_BAL | 0.0195 |
| num__ID | 0.0195 |

---

## Dataset: `german`

- **File:** `D:\fairscore (1)\fairscore\data\raw\german.data`
- **Rows:** 999
- **Features:** 20
- **Target:** `1.1`
- **Model:** RandomForestClassifier
- **Actual Default Rate:** 30.00%

### Test-Set Performance

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.7852 |
| PR-AUC | 0.6180 |
| Accuracy | 0.7400 |
| Precision | 0.6053 |
| Recall | 0.3833 |
| F1 Score | 0.4694 |

### Approval Rates

| Threshold | Approval Rate |
|-----------|---------------|
| 0.35 | 57.50% |
| 0.5 | 81.00% |
| 0.65 | 95.00% |

### Cross-Validation (5-Fold Stratified)

| Metric | Mean | Std |
|--------|------|-----|
| ROC-AUC | 0.7880 | 0.0408 |
| F1 | 0.4826 | 0.0598 |

### Top Feature Importances

| Feature | Importance |
|---------|------------|
| num__feat_6 | 0.0090 |
| num__feat_1169 | 0.0090 |

---

## Dataset: `southgermancredit`

- **File:** `D:\fairscore (1)\fairscore\data\raw\SouthGermanCredit.asc`
- **Rows:** 1,000
- **Features:** 20
- **Target:** `kredit`
- **Model:** RandomForestClassifier
- **Actual Default Rate:** 30.00%

### Test-Set Performance

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.7839 |
| PR-AUC | 0.6803 |
| Accuracy | 0.7800 |
| Precision | 0.7857 |
| Recall | 0.3667 |
| F1 Score | 0.5000 |

### Approval Rates

| Threshold | Approval Rate |
|-----------|---------------|
| 0.35 | 63.50% |
| 0.5 | 86.00% |
| 0.65 | 95.50% |

### Cross-Validation (5-Fold Stratified)

| Metric | Mean | Std |
|--------|------|-----|
| ROC-AUC | 0.7869 | 0.0320 |
| F1 | 0.4747 | 0.0581 |

### Top Feature Importances

| Feature | Importance |
|---------|------------|
| num__laufkont | 0.0240 |
| num__laufzeit | 0.0240 |

---

## Dataset: `synthetic_alt_credit`

- **File:** `synthetic`
- **Rows:** 5,000
- **Features:** 14
- **Target:** `is_default`
- **Model:** RandomForestClassifier
- **Actual Default Rate:** 27.90%

### Test-Set Performance

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.7743 |
| PR-AUC | 0.5665 |
| Accuracy | 0.7630 |
| Precision | 0.6641 |
| Recall | 0.3047 |
| F1 Score | 0.4177 |

### Approval Rates

| Threshold | Approval Rate |
|-----------|---------------|
| 0.35 | 74.60% |
| 0.5 | 87.20% |
| 0.65 | 96.50% |

### Cross-Validation (5-Fold Stratified)

| Metric | Mean | Std |
|--------|------|-----|
| ROC-AUC | 0.7519 | 0.0143 |
| F1 | 0.4019 | 0.0227 |

### Top Feature Importances

| Feature | Importance |
|---------|------------|
| num__income_volatility | 0.0322 |
| num__monthly_income | 0.0322 |

---
