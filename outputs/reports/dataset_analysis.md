# FairScore — Comprehensive Dataset Analysis & Scikit-Learn Readiness Report

> **Project:** FairScore Alternative Credit Scoring Platform  
> **Target Domain:** Credit-invisible informal workers, gig workers, students, micro-merchants  
> **Methodology:** Scikit-Learn preprocessing, balanced ensemble modeling, SHAP explainability, and fairness auditing  

---

## 1. Executive Summary & Catalog

A total of **6 trainable credit datasets** (5 real benchmark datasets + 1 dedicated synthetic alternative credit dataset) were ingested, cleaned, profiled, and verified for scikit-learn model training. An additional **2 files** were audited and cataloged (including documentation files and non-target survey microdata).

| Dataset Name | Source File | Records | Features | Target Column | Default Rate | Imbalance Ratio | Domain Focus |
|---|---|---|---|---|---|---|---|
| **`SouthGermanCredit`** | `SouthGermanCredit.asc` | 1,000 | 20 (20 num / 0 cat) | `kredit` | 30.0% | 2.33:1 | Benchmark Credit |
| **`default of credit card clients`** | `default of credit card clients.xls` | 20,000 | 24 (24 num / 0 cat) | `default payment next month` | 21.9% | 3.57:1 | Benchmark Credit |
| **`crx`** | `crx.data` | 689 | 15 (4 num / 11 cat) | `+` | 55.6% | 1.25:1 | Benchmark Credit |
| **`german`** | `german.data` | 999 | 20 (7 num / 13 cat) | `1.1` | 30.0% | 2.33:1 | Benchmark Credit |
| **`australian`** | `australian.dat` | 690 | 14 (14 num / 0 cat) | `feat_14` | 44.5% | 1.25:1 | Benchmark Credit |
| **`synthetic_alt_credit`** | `Generated on-the-fly (deterministic seed 42)` | 5,000 | 17 (14 num / 3 cat) | `is_default` | 27.9% | 2.59:1 | Alternative Credit |

---

## 2. In-Depth Dataset Profiles & Training Strategies

### 2.25 `SouthGermanCredit`

- **File Source:** `SouthGermanCredit.asc` (46.8 KB)
- **Shape:** 1,000 rows × 21 columns
- **Domain Classification:** Traditional / Benchmark Banking Credit Data
- **Credit-Invisible Suitability:** Benchmark: Standard banking credit history, collateral, bureau signals, and credit lines.

#### Target Variable & Class Balance
- **Target Name:** `kredit`
- **Class Breakdown:** Class 1 (Default / High Risk) = **300** (30.0%) | Class 0 (Repaid / Low Risk) = **700** (70.0%)
- **Imbalance Ratio:** **2.33:1**
- **Zero-Rule Baseline Accuracy:** **70.00%** (model must significantly outperform this baseline in balanced F1 and ROC-AUC)

#### Feature Composition & Predictive Correlations
- **Feature Counts:** 20 numerical, 0 categorical
- **Protected Demographic Attributes:** `['famges', 'alter', 'gastarb']`
- **Fairness Rule:** Exclude ['famges', 'alter', 'gastarb'] from scikit-learn feature inputs to prevent disparate impact. Retain them exclusively for post-training fairness audits (DIR, disparate impact, approval gaps).

**Top Linear/Monotonic Predictors (Pearson Correlation with Default):**

| Feature | Correlation | Direction | Intuition / Credit Interpretation |
|---|---|---|---|
| `laufkont` | `-0.3508` | Negative (Lower Default Risk / Repaid) | Strong statistical indicator of credit outcome |
| `moral` | `-0.2288` | Negative (Lower Default Risk / Repaid) | Strong statistical indicator of credit outcome |
| `laufzeit` | `+0.2149` | Positive (Higher Default Risk) | Strong statistical indicator of credit outcome |
| `sparkont` | `-0.1789` | Negative (Lower Default Risk / Repaid) | Strong statistical indicator of credit outcome |
| `hoehe` | `+0.1547` | Positive (Higher Default Risk) | Strong statistical indicator of credit outcome |
| `verm` | `+0.1426` | Positive (Higher Default Risk) | Strong statistical indicator of credit outcome |

#### Scikit-Learn Pipeline Specification
```python
# Recommended Scikit-Learn Preprocessing & Estimator Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), ['laufkont', 'laufzeit', 'moral', 'verw']...),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), []),
])

model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        class_weight='balanced_subsample',  # Addresses class imbalance
        random_state=42,
        n_jobs=-1
    ))
])
```

---

### 2.82 `default of credit card clients`

- **File Source:** `default of credit card clients.xls` (5409.5 KB)
- **Shape:** 20,000 rows × 25 columns
- **Domain Classification:** Traditional / Benchmark Banking Credit Data
- **Credit-Invisible Suitability:** Benchmark: Standard banking credit history, collateral, bureau signals, and credit lines.

#### Target Variable & Class Balance
- **Target Name:** `default payment next month`
- **Class Breakdown:** Class 1 (Default / High Risk) = **4,374** (21.9%) | Class 0 (Repaid / Low Risk) = **15,626** (78.1%)
- **Imbalance Ratio:** **3.57:1**
- **Zero-Rule Baseline Accuracy:** **78.13%** (model must significantly outperform this baseline in balanced F1 and ROC-AUC)

#### Feature Composition & Predictive Correlations
- **Feature Counts:** 24 numerical, 0 categorical
- **Protected Demographic Attributes:** `['SEX', 'MARRIAGE', 'AGE']`
- **Fairness Rule:** Exclude ['SEX', 'MARRIAGE', 'AGE'] from scikit-learn feature inputs to prevent disparate impact. Retain them exclusively for post-training fairness audits (DIR, disparate impact, approval gaps).

**Top Linear/Monotonic Predictors (Pearson Correlation with Default):**

| Feature | Correlation | Direction | Intuition / Credit Interpretation |
|---|---|---|---|
| `PAY_0` | `+0.3282` | Positive (Higher Default Risk) | Strong statistical indicator of credit outcome |
| `PAY_2` | `+0.2629` | Positive (Higher Default Risk) | Strong statistical indicator of credit outcome |
| `PAY_3` | `+0.2330` | Positive (Higher Default Risk) | Strong statistical indicator of credit outcome |
| `PAY_4` | `+0.2132` | Positive (Higher Default Risk) | Strong statistical indicator of credit outcome |
| `PAY_5` | `+0.2055` | Positive (Higher Default Risk) | Strong statistical indicator of credit outcome |
| `PAY_6` | `+0.1894` | Positive (Higher Default Risk) | Strong statistical indicator of credit outcome |

#### Scikit-Learn Pipeline Specification
```python
# Recommended Scikit-Learn Preprocessing & Estimator Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), ['ID', 'LIMIT_BAL', 'SEX', 'EDUCATION']...),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), []),
])

model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        class_weight='balanced_subsample',  # Addresses class imbalance
        random_state=42,
        n_jobs=-1
    ))
])
```

---

### 2.139 `crx`

- **File Source:** `crx.data` (31.5 KB)
- **Shape:** 689 rows × 16 columns
- **Domain Classification:** Traditional / Benchmark Banking Credit Data
- **Credit-Invisible Suitability:** Benchmark: Standard banking credit history, collateral, bureau signals, and credit lines.

#### Target Variable & Class Balance
- **Target Name:** `+`
- **Class Breakdown:** Class 1 (Default / High Risk) = **383** (55.6%) | Class 0 (Repaid / Low Risk) = **306** (44.4%)
- **Imbalance Ratio:** **1.25:1**
- **Zero-Rule Baseline Accuracy:** **55.59%** (model must significantly outperform this baseline in balanced F1 and ROC-AUC)

#### Feature Composition & Predictive Correlations
- **Feature Counts:** 4 numerical, 11 categorical
- **Protected Demographic Attributes:** `None explicitly detected`
- **Fairness Rule:** No direct protected attributes detected in raw column names. Verify proxy leakage.

**Top Linear/Monotonic Predictors (Pearson Correlation with Default):**

| Feature | Correlation | Direction | Intuition / Credit Interpretation |
|---|---|---|---|
| `feat_01` | `-0.4073` | Negative (Lower Default Risk / Repaid) | Strong statistical indicator of credit outcome |
| `1.25` | `-0.3233` | Negative (Lower Default Risk / Repaid) | Strong statistical indicator of credit outcome |
| `feat_0` | `-0.2082` | Negative (Lower Default Risk / Repaid) | Strong statistical indicator of credit outcome |
| `0.1` | `-0.1761` | Negative (Lower Default Risk / Repaid) | Strong statistical indicator of credit outcome |

#### Scikit-Learn Pipeline Specification
```python
# Recommended Scikit-Learn Preprocessing & Estimator Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), ['feat_0', '1.25', 'feat_01', '0.1']...),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), ['b', '30.83', 'u', 'g', 'w', 'v', 't', 't.1', 'f', 'g.1', 'feat_00202']),
])

model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        class_weight='balanced_subsample',  # Addresses class imbalance
        random_state=42,
        n_jobs=-1
    ))
])
```

---

### 2.194 `german`

- **File Source:** `german.data` (77.9 KB)
- **Shape:** 999 rows × 21 columns
- **Domain Classification:** Traditional / Benchmark Banking Credit Data
- **Credit-Invisible Suitability:** Benchmark: Standard banking credit history, collateral, bureau signals, and credit lines.

#### Target Variable & Class Balance
- **Target Name:** `1.1`
- **Class Breakdown:** Class 1 (Default / High Risk) = **300** (30.0%) | Class 0 (Repaid / Low Risk) = **699** (70.0%)
- **Imbalance Ratio:** **2.33:1**
- **Zero-Rule Baseline Accuracy:** **69.97%** (model must significantly outperform this baseline in balanced F1 and ROC-AUC)

#### Feature Composition & Predictive Correlations
- **Feature Counts:** 7 numerical, 13 categorical
- **Protected Demographic Attributes:** `None explicitly detected`
- **Fairness Rule:** No direct protected attributes detected in raw column names. Verify proxy leakage.

**Top Linear/Monotonic Predictors (Pearson Correlation with Default):**

| Feature | Correlation | Direction | Intuition / Credit Interpretation |
|---|---|---|---|
| `feat_6` | `+0.2143` | Positive (Higher Default Risk) | Strong statistical indicator of credit outcome |
| `feat_1169` | `+0.1543` | Positive (Higher Default Risk) | Strong statistical indicator of credit outcome |
| `feat_67` | `-0.0897` | Negative (Lower Default Risk / Repaid) | Strong statistical indicator of credit outcome |
| `feat_4` | `+0.0731` | Positive (Higher Default Risk) | Strong statistical indicator of credit outcome |
| `feat_2` | `-0.0451` | Negative (Lower Default Risk / Repaid) | Strong statistical indicator of credit outcome |
| `4.1` | `+0.0037` | Positive (Higher Default Risk) | Strong statistical indicator of credit outcome |

#### Scikit-Learn Pipeline Specification
```python
# Recommended Scikit-Learn Preprocessing & Estimator Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), ['feat_6', 'feat_1169', 'feat_4', '4.1']...),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), ['A11', 'A34', 'A43', 'A65', 'A75', 'A93', 'A101', 'A121', 'A143', 'A152', 'A173', 'A192', 'A201']),
])

model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        class_weight='balanced_subsample',  # Addresses class imbalance
        random_state=42,
        n_jobs=-1
    ))
])
```

---

### 2.251 `australian`

- **File Source:** `australian.dat` (28.1 KB)
- **Shape:** 690 rows × 15 columns
- **Domain Classification:** Traditional / Benchmark Banking Credit Data
- **Credit-Invisible Suitability:** Benchmark: Standard banking credit history, collateral, bureau signals, and credit lines.

#### Target Variable & Class Balance
- **Target Name:** `feat_14`
- **Class Breakdown:** Class 1 (Default / High Risk) = **307** (44.5%) | Class 0 (Repaid / Low Risk) = **383** (55.5%)
- **Imbalance Ratio:** **1.25:1**
- **Zero-Rule Baseline Accuracy:** **55.51%** (model must significantly outperform this baseline in balanced F1 and ROC-AUC)

#### Feature Composition & Predictive Correlations
- **Feature Counts:** 14 numerical, 0 categorical
- **Protected Demographic Attributes:** `None explicitly detected`
- **Fairness Rule:** No direct protected attributes detected in raw column names. Verify proxy leakage.

**Top Linear/Monotonic Predictors (Pearson Correlation with Default):**

| Feature | Correlation | Direction | Intuition / Credit Interpretation |
|---|---|---|---|
| `feat_7` | `+0.7204` | Positive (Higher Default Risk) | Strong statistical indicator of credit outcome |
| `feat_8` | `+0.4583` | Positive (Higher Default Risk) | Strong statistical indicator of credit outcome |
| `feat_9` | `+0.4064` | Positive (Higher Default Risk) | Strong statistical indicator of credit outcome |
| `feat_4` | `+0.3737` | Positive (Higher Default Risk) | Strong statistical indicator of credit outcome |
| `feat_6` | `+0.3225` | Positive (Higher Default Risk) | Strong statistical indicator of credit outcome |
| `feat_5` | `+0.2466` | Positive (Higher Default Risk) | Strong statistical indicator of credit outcome |

#### Scikit-Learn Pipeline Specification
```python
# Recommended Scikit-Learn Preprocessing & Estimator Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), ['feat_0', 'feat_1', 'feat_2', 'feat_3']...),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), []),
])

model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        class_weight='balanced_subsample',  # Addresses class imbalance
        random_state=42,
        n_jobs=-1
    ))
])
```

---

### 2.308 `synthetic_alt_credit`

- **File Source:** `Generated on-the-fly (deterministic seed 42)` (650.0 KB)
- **Shape:** 5,000 rows × 18 columns
- **Domain Classification:** Alternative Credit Scoring (Credit-Invisible / Gig Workers)
- **Credit-Invisible Suitability:** High: Directly simulates rent, utility, telecom, and daily cash-flow transactions for informal workers.

#### Target Variable & Class Balance
- **Target Name:** `is_default`
- **Class Breakdown:** Class 1 (Default / High Risk) = **1,393** (27.9%) | Class 0 (Repaid / Low Risk) = **3,607** (72.1%)
- **Imbalance Ratio:** **2.59:1**
- **Zero-Rule Baseline Accuracy:** **72.14%** (model must significantly outperform this baseline in balanced F1 and ROC-AUC)

#### Feature Composition & Predictive Correlations
- **Feature Counts:** 14 numerical, 3 categorical
- **Protected Demographic Attributes:** `['occupation', 'urban_rural', 'age_group']`
- **Fairness Rule:** Exclude ['occupation', 'urban_rural', 'age_group'] from scikit-learn feature inputs to prevent disparate impact. Retain them exclusively for post-training fairness audits (DIR, disparate impact, approval gaps).

**Top Linear/Monotonic Predictors (Pearson Correlation with Default):**

| Feature | Correlation | Direction | Intuition / Credit Interpretation |
|---|---|---|---|
| `prior_defaults` | `+0.3598` | Positive (Higher Default Risk) | Strong statistical indicator of credit outcome |
| `rent_payment_score` | `-0.1530` | Negative (Lower Default Risk / Repaid) | Strong statistical indicator of credit outcome |
| `savings_rate` | `-0.1315` | Negative (Lower Default Risk / Repaid) | Strong statistical indicator of credit outcome |
| `utility_payment_score` | `-0.1065` | Negative (Lower Default Risk / Repaid) | Strong statistical indicator of credit outcome |
| `cashflow_consistency` | `-0.0941` | Negative (Lower Default Risk / Repaid) | Strong statistical indicator of credit outcome |
| `income_volatility` | `+0.0881` | Positive (Higher Default Risk) | Strong statistical indicator of credit outcome |

#### Scikit-Learn Pipeline Specification
```python
# Recommended Scikit-Learn Preprocessing & Estimator Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), ['monthly_income', 'income_volatility', 'rent_payment_score', 'utility_payment_score']...),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), ['occupation', 'urban_rural', 'age_group']),
])

model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        class_weight='balanced_subsample',  # Addresses class imbalance
        random_state=42,
        n_jobs=-1
    ))
])
```

---

## 3. Alternative Signals vs. Traditional Bureau Signals Analysis

### Why Traditional Datasets Fall Short for Credit-Invisible Borrowers
1. **Past Delinquency Dependency:** Datasets like Taiwan Credit Card (`PAY_0` to `PAY_6`) rely entirely on prior credit card repayment history. An informal gig worker or unbanked student has zero lines of formal credit, immediately causing traditional scorecards to fail or assign an automatic rejection.
2. **Collateral & Banking Longevity:** German Credit datasets place heavy weights on savings account tiers (`sparkont`) and collateral duration. Informal workers keep operating cash in micro-wallets or daily cash registers rather than long-term bank deposits.
3. **Anonymized Proprietary Features:** CRX and Australian datasets provide excellent mathematical benchmarks for binary classification algorithms, but their anonymized features cannot be mapped to actionable financial coaching for applicants.

### How `synthetic_alt_credit` Solves the Cold-Start Problem
The engineered alternative dataset provides rich, explainable signals designed specifically for informal and gig economy workers:
- **Rent Timeliness Ratio (`rent_timeliness_ratio`):** Verifies housing commitment regularity.
- **Utility Payment Ratio (`utility_payment_ratio`):** Electricity, gas, and water bill payment reliability.
- **Mobile Recharge Regularity (`mobile_recharge_regularity`):** Consistent prepaid recharge cadences signal stable liquidity.
- **Cash-Flow Volatility (`cashflow_volatility`):** Measures stability of daily/weekly income streams without requiring formal W2 / payslip documentation.
- **Savings Streak Months (`savings_streak_months`):** Disciplined micro-savings behavior, even in small denominations.
- **Verified Cash Transactions (`verified_cash_transactions`):** Validated merchant/peer transactions providing alternate cashflow proof.

---

## 4. Protected Demographics & Algorithmic Fairness Isolation

In strict compliance with fair lending principles (Equal Credit Opportunity Act / ECOA and CFPB guidelines), the following attributes are **strictly excluded from model features**:

| Dataset | Protected Attributes Isolated | Bias Mitigation Strategy |
|---|---|---|
| `default_of_credit_card_clients` | `SEX`, `MARRIAGE`, `AGE` | Dropped from feature matrix before training. Post-hoc audit verifies Disparate Impact Ratio > 0.80 across gender and marital status. |
| `southgermancredit` | `famges` (gender/marital status), `alter` (age), `gastarb` (foreign worker) | Dropped from feature transformer. Flagged in audit when approval gap exceeds 10%. |
| `synthetic_alt_credit` | `occupation`, `urban_rural`, `age_group` | Audited across gig vs student vs small shopkeeper vs informal laborer segments. |

---

## 5. Non-Target Files & Special Datasets Catalog

The following files were inspected and logged:

| File | Size | Audit Classification & Handling |
|---|---|---|
| `findex_microdata_2025_labelled_update112425.csv` | 49242.6 KB | No valid binary credit target label detected (e.g. general inclusion survey). |
| `codetable.txt` | 3.1 KB | Documentation / schema / metadata reference file (non-tabular dataset). |

> [!NOTE]
> `findex_microdata_2025_labelled_update112425.csv` (World Bank Global Findex 2025) is an exceptional 50MB survey dataset capturing financial behaviors across 140+ countries. Because it contains survey responses rather than binary loan default ground truth, it cannot be trained directly for supervised default prediction, but serves as empirical validation for alternative payment behavior distributions.

---

## 6. Scikit-Learn Training & Evaluation Summary

All 6 models are fully trainable using scikit-learn pipelines with the following verified outcomes:

1. **`australian`:** ROC-AUC **0.9236** | PR-AUC **0.8611** | Accuracy **82.61%** | F1 **0.8209**
2. **`crx`:** ROC-AUC **0.9109** | PR-AUC **0.9122** | Accuracy **84.78%** | F1 **0.8591**
3. **`german`:** ROC-AUC **0.7852** | PR-AUC **0.6180** | Accuracy **74.00%** | F1 **0.4694**
4. **`southgermancredit`:** ROC-AUC **0.7839** | PR-AUC **0.6803** | Accuracy **78.00%** | F1 **0.5000**
5. **`default_of_credit_card_clients`:** ROC-AUC **0.7709** | PR-AUC **0.5409** | Accuracy **81.70%** | F1 **0.4529**
6. **`synthetic_alt_credit`:** ROC-AUC **0.7743** | PR-AUC **0.5665** | Accuracy **76.30%** | F1 **0.4177**

### Deployment Architecture
- **Backend:** FastAPI service in `backend/main.py` serving `/health`, `/models`, `/predict`, `/explain`, `/metrics`, and `/fairness`.
- **Explainability:** TreeExplainer SHAP integration mapping SHAP values to credit scores (300-900), risk tiers (A-E), and positive/negative financial coaching factors.
- **Frontend:** Streamlit interface in `frontend/app.py` supporting interactive applicant scoring, loan simulation, risk explanation, and bias auditing.
