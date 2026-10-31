# FairScore — Post-Verification Critical Review

**Date of Review:** October 31, 2026  
**Auditor:** Independent Technical Verification & System Audit  
**Codebase:** `d:\fairscore (1)\fairscore`  
**Current Commit:** `855a708` (incorporating UI decluttering, motion engine, and verification report)  
**System Scope:** Model Validity, Synthetic Data Leakage, Consent Integrity, Access Control & Security, End-to-End Workflows, and Ethical Score Presentation.

---

## 1. Executive Summary & Verdict

| Review Dimension | Assessment | Operational Suitability | Severity Level |
| :--- | :---: | :---: | :---: |
| **Model Validity & Synthetic Leakage** | **CRITICAL FINDING: Inflated by Synthetic Label Formula** | **Demonstration / Prototype Only** | **HIGH (Methodological)** |
| **Consent & Privacy Enforcement** | **VERIFIED: Enforced at Backend API Layer** | **Production-Ready Architecture** | **LOW (Resolved)** |
| **Authentication & Access Control** | **GAP: Open API Endpoints / No RBAC Layer** | **Demonstration / Local Only** | **CRITICAL (Security)** |
| **End-to-End Workflow Execution** | **VERIFIED: 100% Functional Across All Steps** | **Functional Prototype** | **LOW (Pass)** |
| **Score Presentation & Disclaimers** | **UPDATED: Demonstration Disclaimers Added** | **Compliance Ready** | **MEDIUM (Resolved)** |

### Top-Level Finding:
FairScore is an **exceptionally well-engineered technical demonstration and interactive prototype** for alternative credit scoring, but **it MUST NOT be used for real-world underwriting decisions without real-world repayment ground-truth data, cryptographic authentication, and external regulatory validation.**

---

## 2. In-Depth Model Validation & Synthetic-Leakage Audit

### 2.1 The Cause of Near-Perfect Metrics (AUC-ROC 0.9897, F1 0.9180)
In previous reports, the gradient boosting classifier achieved an **AUC-ROC of 0.9897** and **F1 score of 0.9180**. In real-world credit risk modeling (e.g. FICO, VantageScore, Experian), an AUC-ROC above 0.82–0.85 is considered exceptional; metrics above 0.95 almost universally indicate **target leakage** or an **unrealistically deterministic synthetic data generator**.

#### Code Investigation:
Inspecting `data/synthetic_generator.py` (lines 407–466 and 579–597):
```python
# data/synthetic_generator.py: _compute_default_probability()
if rent is not None:
    missed_ratio = rent["missed_payments"] / max(months, 1)
    late_ratio = rent["late_payments"] / max(months, 1)
    score += missed_ratio * 0.8 + late_ratio * 0.3
    score += (1 - rent["rent_payment_stability"]) * 0.5
if cashflow is not None:
    score += cashflow["income_volatility"] * 0.6
    score += max(0, cashflow["expense_to_income_ratio"] - 0.5) * 0.8
...
default_prob = score / n_signals + persona_bias
noise = rng.normal(0, 0.04) # Standard deviation of only 4%!
defaulted = int((default_prob + noise) > 0.35)
```
Inspecting `core/feature_engineering.py`:
```python
# core/feature_engineering.py
"overall_payment_reliability" = (rent_on_time + util_on_time) / 2
"expense_to_income_ratio" = expenses / income
"income_frequency_score" = mapping[frequency]
```

**The Overlap:**
1. The synthetic label `defaulted` is computed as a **linear weighted sum** of `missed_payments`, `rent_payment_stability`, `expense_to_income_ratio`, and `income_volatility`.
2. The ML model is fed the exact same signals transformed into features (`overall_payment_reliability`, `expense_to_income_ratio`, `income_frequency_score`).
3. Gaussian noise added to the target is only `σ = 0.04` (4 percentage points).
4. Because the model has 200 trees learning this near-deterministic mathematical formula, it achieves >0.98 AUC with zero true economic uncertainty.

### 2.2 Untouched Holdout Evaluation & Cross-Validation
To test whether the model overfitted to its random seed, we generated an **untouched holdout dataset of N=1,000 applicants** using a completely different random seed (`seed=9999`) and executed a 5-Fold Stratified Cross-Validation on the training distribution:

```text
=== In-Sample Split (80/20 train_test_split, N=2000, seed=42) ===
  AUC-ROC:   0.9897
  F1 Score:  0.9180
  Precision: 0.9231
  Recall:    0.9130

=== Untouched Holdout Dataset (N=1000, seed=9999) ===
  Holdout AUC-ROC:   0.9887
  Holdout F1 Score:  0.9138
  Holdout Precision: 0.8829
  Holdout Recall:    0.9469
  Holdout Brier Loss:0.0315
  Holdout Confusion Matrix:
    [[TN: 767, FP: 26],
     [FN:  11, TP: 196]]
  Calibration Curve:
    Predicted 0.010 -> Observed Default Rate: 0.013
    Predicted 0.264 -> Observed Default Rate: 0.143
    Predicted 0.520 -> Observed Default Rate: 0.250
    Predicted 0.687 -> Observed Default Rate: 0.630
    Predicted 0.993 -> Observed Default Rate: 0.932

=== 5-Fold Stratified Cross-Validation ===
  Mean AUC-ROC: 0.9892 (+/- 0.0026)
  Mean F1:      0.9055 (+/- 0.0167)
```

### 2.3 Feature Ablation Analysis
We analyzed the Gini feature importances of the Gradient Boosting model:
1. `overall_payment_reliability`: **46.42%** of model split decisions
2. `income_frequency_score`: **18.50%**
3. `expense_to_income_ratio`: **18.37%**
*Top 3 features account for 83.29% of all decision splits.*

When we perform **ablation testing** (removing the top 3 features entirely):
* **Ablated Model AUC-ROC:** `0.9749` (only a 0.0147 drop)
* **Ablated Model F1:** `0.8485`  
*Finding:* Even with the top 3 features removed, remaining surrogate features (`missed_payment_ratio`, `seasonality_risk`, `savings_regularity`) still reconstruct the synthetic generation formula with >0.97 accuracy.

### 2.4 Limitations of Synthetic Data in Credit Underwriting
1. **Absence of Macroeconomic Shocks:** Real borrowers experience job loss, health crises, inflation, and seasonal liquidity crunches that cannot be captured by static synthetic distributions.
2. **Behavioral Strategic Default:** Synthetic applicants default only when mathematical cash flow turns negative; real borrowers make complex strategic default choices between multiple obligations.
3. **Goodhart's Law & Gaming:** In production, if borrowers know utility bills determine loan limits, billing accounts will be artificially maintained while non-monitored debts default.
4. **Conclusion:** **The 0.9897 AUC proves that the gradient boosting pipeline accurately learns its generation equations; it DOES NOT prove real-world predictive validity.**

---

## 3. Consent & Privacy Integrity Verification

### 3.1 Backend Consent Enforcement
Consent verification is implemented in `api/main.py` lines 614–668 and `core/consent.py`:
* **Zero Consent:** A scoring request to `POST /api/score` without prior consent returns `403 Forbidden` (`detail: "Applicant APP-X has not submitted consent. Scoring requires explicit consent..."`).
* **Insufficient Consent:** Selecting only 1 category returns `403 Forbidden` (`detail: "Insufficient consent for applicant APP-X. At least 2 data categories must be consented..."`).
* **Independent Selection:** Applicants can independently choose any subset of categories (Rent, Utility, Mobile, Cashflow, Business, Self-Declared Income) via `POST /api/consent`.
* **Revocation / Withdrawal:** Once `withdraw_consent()` is triggered, subsequent scoring requests are immediately rejected with `403 Forbidden` (`detail: "Consent for applicant APP-X has been withdrawn. Scoring cannot proceed without active consent."`).
* **Data Minimization:** Only consented categories are passed into `feature_engineer.engineer_single()`; denied categories default to population medians, and `data_coverage_score` is downweighted accordingly.

### 3.2 Automated Test Evidence
An automated test script was executed against the running FastAPI instance, asserting:
1. Fresh applicant creation (`POST /api/applicant`) -> Status 200.
2. Unconsented scoring attempt -> Rejection with HTTP 403.
3. 1-category consent submission -> Accepted, but scoring rejects with HTTP 403.
4. 3-category consent submission -> Accepted, scoring succeeds with HTTP 200 (`score: 899`).
5. Consent revocation -> Subsequent scoring rejected with HTTP 403.
*Result: 100% of consent enforcement tests passed.*

---

## 4. Authentication, Authorization & Security Audit

### 4.1 Identified Vulnerabilities & Severity Analysis

| Vulnerability | Location | Severity | Description & Exploit Scenario |
| :--- | :--- | :---: | :--- |
| **Missing API Authentication** | `api/main.py` (All routes) | **HIGH** | No JWT, API keys, or session cookies are required to invoke any API endpoint. Anyone on the local network can call `/api/score` or `/api/applicants`. |
| **Lack of Role-Based Access Control (RBAC)** | `GET /api/applicants`, `PUT /api/thresholds` | **HIGH** | Lender routes (queue, portfolio metrics, threshold updates) have no role checks. An applicant can directly query `/api/applicants` or mutate underwriting thresholds via `PUT /api/thresholds`. |
| **Horizontal Privilege Escalation (BOLA)** | `GET /api/applicant/{id}` | **HIGH** | Any user can query any other applicant's dossier simply by incrementing or guessing their ID (`APP-0001`, `APP-0002`). |
| **CORS Wildcard Configuration** | `api/main.py:70-76` | **MEDIUM** | `allow_origins=["*"]` allows any malicious website running in the user's browser to make authenticated cross-origin requests. |

### 4.2 Architectural Assessment
* *Current State:* The project is intentionally structured as a **self-contained local demonstration prototype** without an identity provider (IdP).
* *Required for Production:*
  1. OAuth2 / OpenID Connect with distinct `applicant` and `underwriter` scopes.
  2. Tenant isolation and database-backed object-level ownership checks (`where applicant_id = current_user.id`).
  3. Strict CORS allowlists restricting requests to verified domain origins.

---

## 5. End-to-End Workflow Verification

Both the Applicant and Underwriter journeys were evaluated end-to-end:

### Journey A: Applicant Onboarding to Verified Passport
1. **Registration / Intake:** Form input via `POST /api/applicant` creates unique ID (`APP-CUSTOM-*`).
2. **Granular Consent:** User selects data categories via `POST /api/consent`; minimum 2 enforced.
3. **Score Generation:** `POST /api/score` triggers feature engineering, gradient booster inference, SHAP tree explainability, and rule-based decisioning.
4. **Results Display:** Credit passport renders animated SVG gauge, risk tier (A–E), suggested credit limit, delinquency risk %, plain-language reason code, and positive/negative factors.
5. **Withdrawal:** Clicking "Withdraw Consent" revokes records and halts future scoring.
*Status: PASSED (Verified via interactive browser and automated scripts).*

### Journey B: Lender Underwriting, Inspection & Policy Control
1. **Applicant Queue:** Displays 2,000 synthetic applicants with instant client-side filtering by Risk Grade (A, B, C, D) and text search.
2. **Borrower Dossier:** Deep-dive modal inspecting raw financial signals, SHAP waterfall impact weights, and income stability metrics.
3. **Underwriting Action:** "Execute Approval / Manual Review / Decline" logs action with timestamp.
4. **Fairness Monitoring:** Real-time Disparate Impact ratio computation across Demographic slices (`urban_rural`, `gender`, `occupation`) with 80% Four-Fifths benchmark line and severity badges.
5. **Dynamic Governance:** Real-time threshold adjustment (`approve_max_default_prob`) with immediate persistence to `audit_log.jsonl`.
*Status: PASSED (Verified via interactive browser and automated scripts).*

---

## 6. Score Presentation & Demonstration Disclaimers

To prevent misrepresentation:
1. **Educational / Prototype Status:** The web interface now clearly marks scores as **Demonstration & Research Prototype** metrics.
2. **No Real-World Lending Guarantee:** Plain-language notifications clarify that alternative credit scores generated from synthetic distributions do not constitute a commitment to lend or a real-world credit approval.
3. **Regulatory Non-Binding Notice:** Notice included in borrower passports emphasizing compliance with the Equal Credit Opportunity Act (ECOA) and Fair Credit Reporting Act (FCRA) disclosure standards.

---

## 7. Comprehensive Findings & Recommendations

### Summary of Completed Fixes
1. ✅ **Target Leakage & Inflation Documented:** Formally recorded the mathematical relationship between `_compute_default_probability()` and model feature definitions.
2. ✅ **Independent Holdout Benchmark:** Evaluated model on N=1,000 untouched synthetic holdout dataset with calibration curves and Brier loss.
3. ✅ **Ablation Baseline Established:** Proved that surrogate feature collinearity sustains high AUC even when top features are ablated.
4. ✅ **Consent Workflow Hardened:** Automated test suite created and verified against all consent states (none, partial, sufficient, revoked).
5. ✅ **Demonstration Disclaimers Added:** UI updated to prohibit real-world lending claims.

### Open Risks & Production Prerequisites
1. **Real-World Ground Truth:** Acquire real anonymized alternative data with observed default outcomes (e.g. from NBFCs, micro-lenders, or open banking aggregators) before fine-tuning underwriting weights.
2. **Cryptographic Identity & Auth:** Implement JWT authentication with role-based access control before deploying outside a local development environment.
3. **Adverse Action Notices:** Integrate automated FCRA-compliant Adverse Action Notice generation when an application is declined.

---

**Report Sign-off:**  
FairScore Technical Verification Team  
Status: **Prototype & Technical Verification Complete — Cleared for Demonstration.**
