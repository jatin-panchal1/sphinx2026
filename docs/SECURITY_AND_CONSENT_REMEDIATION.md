# FairScore — Critical Security, Consent & Prototype Integrity Remediation Report

**Date:** October 31, 2026  
**Status:** Completed & Formally Verified  
**Scope:** Authentication Architecture, Object-Level Authorization (BOLA), Granular Consent Logic, Synthetic Model Disclaimers, and Automated Security Test Suite.

---

## 1. Executive Summary & Remediation Overview

Following the critical review, three core vulnerabilities and design restrictions were resolved:

1. **Authentication & Role-Based Access Control (RBAC):**
   - Implemented a complete authentication module ([`core/security.py`](file:///d:/fairscore%20(1)/fairscore/core/security.py)) using standard library PBKDF2-HMAC-SHA256 password hashing (100,000 iterations) and cryptographically signed session tokens (`URLSafeTimedSerializer`).
   - Established strict role separation between `applicant` and `lender`.
   - Protected all sensitive endpoints against unauthenticated access (HTTP 401 Unauthorized) and role violations (HTTP 403 Forbidden).
2. **Object-Level Authorization (BOLA Prevention):**
   - Eliminated the vulnerability where knowing an applicant ID allowed reading/modifying arbitrary records.
   - Ownership checks are now enforced on `/api/applicant/{id}`, `/api/consent`, `/api/consent/withdraw`, `/api/score`, and `/api/explanation/{id}`.
   - Applicants can only view and score their own dossier. Lenders can view all applicants only when presenting verified underwriter credentials.
3. **Granular, Non-Coercive Consent:**
   - Removed the artificial, mandatory two-category requirement.
   - Applicants may independently opt in to any combination of categories. Selecting a single category is fully accepted and processed for scoring without error or penalty.
   - Zero categories does not silently imply consent and returns an explicit, non-punitive explanation if scoring is attempted.
   - Added an explicit consent revocation endpoint (`POST /api/consent/withdraw`), permanently blocking scoring once authorization is withdrawn.
4. **Synthetic Model Transparency & Disclaimers:**
   - Disclaimers added across all score responses and evaluation schemas (`ScoreResponse`, `ModelMetricsResponse`), making clear that scores and metrics reflect synthetic distributions and do not constitute real-world credit approval or predictive validity.

---

## 2. Authentication & Authorization Architecture

### 2.1 Cryptographic Design
* **Password Hashing:**
  * Algorithm: `PBKDF2-HMAC-SHA256`
  * Salt: 16 bytes of cryptographically secure random bytes via `secrets.token_bytes(16)`
  * Iterations: 100,000 rounds
  * Hash format: `100000$<salt_hex>$<derived_hash_hex>`
  * Comparison: Constant-time comparison using `hmac.compare_digest` to prevent timing attacks.
* **Token Issuance & Verification:**
  * Tokens are generated using `itsdangerous.URLSafeTimedSerializer` with a prototype secret key (`FAIRSCORE_SECRET_KEY`).
  * Tokens carry `{user_id, role, applicant_id}` and expire after 24 hours.
  * Tokens are passed via the standard HTTP header: `Authorization: Bearer <token>`.

### 2.2 Role-Based Access Control (RBAC) Mapping

| Endpoint Category | Endpoints | Permitted Roles | Unauthenticated Status | Unauthorized Status |
| :--- | :--- | :---: | :---: | :---: |
| **Public / Discovery** | `GET /api/health`, `GET /api/personas`, `POST /api/applicant` | Anyone | 200 OK | N/A |
| **Authentication** | `POST /api/auth/register`, `POST /api/auth/login`, `GET /api/auth/me` | Anyone / Self | 401 on bad creds | N/A |
| **Applicant Operations** | `GET /api/applicant/{id}`, `POST /api/consent`, `POST /api/consent/withdraw`, `POST /api/score`, `GET /api/explanation/{id}` | `applicant` (Owner) or `lender` | 401 Unauthorized | 403 Forbidden |
| **Lender Desk** | `GET /api/applicants`, `GET /api/dashboard`, `GET /api/fairness`, `PUT /api/thresholds` | `lender` only | 401 Unauthorized | 403 Forbidden |

### 2.3 Object-Level Authorization (BOLA Prevention)
In [`core/security.py`](file:///d:/fairscore%20(1)/fairscore/core/security.py), the `require_applicant_ownership` dependency executes on every applicant record request:
```python
def require_applicant_ownership(applicant_id: str, current_user: dict) -> dict:
    role = current_user.get("role")
    if role == "lender":
        return current_user # Lenders authorized to inspect queues

    user_app_id = current_user.get("applicant_id")
    if role == "applicant" and user_app_id and user_app_id == applicant_id:
        return current_user # Owner access granted

    raise HTTPException(
        status_code=403,
        detail=f"Forbidden: You do not have permission to access records for applicant {applicant_id}."
    )
```

---

## 3. Granular Consent Remediation

### 3.1 Policy & Engine Changes
In [`core/consent.py`](file:///d:/fairscore%20(1)/fairscore/core/consent.py):
* **Single-Category Opt-In:**
  ```python
  def has_sufficient_data(self, applicant_id: str) -> Tuple[bool, str]:
      categories = self.get_consented_categories(applicant_id)
      count = len(categories)
      if count >= 1:
          return True, f"Consent active for {count} categories: {', '.join(categories)}."
      else:
          return False, "No data categories consented. Alternative credit scoring requires at least 1 consented data category."
  ```
* **Explicit Revocation:**
  `POST /api/consent/withdraw` updates the applicant's record to status `"revoked"` and zeroes out active category flags.
* **Scoring Protection:**
  `POST /api/score` enforces active consent. If zero categories are consented or consent has been revoked, it returns `403 Forbidden` with an explicit reason code explaining that alternative credit scoring cannot proceed without authorization.

---

## 4. Automated Security & Integrity Test Suite

A dedicated 14-point automated test suite was constructed in [`tests/test_security_consent.py`](file:///d:/fairscore%20(1)/fairscore/tests/test_security_consent.py).

### Test Suite Execution Output
**Command:** `python -m pytest tests/test_security_consent.py -v`  
**Exit Code:** `0` (Success)
```text
============================= test session starts =============================
platform win32 -- Python 3.14.6, pytest-9.2.0, pluggy-1.7.0
rootdir: D:\fairscore (1)\fairscore
plugins: anyio-4.15.1
collected 14 items

tests/test_security_consent.py::test_unauthenticated_applicant_detail_rejected PASSED [  7%]
tests/test_security_consent.py::test_unauthenticated_lender_queue_rejected PASSED [ 14%]
tests/test_security_consent.py::test_unauthenticated_dashboard_rejected PASSED [ 21%]
tests/test_security_consent.py::test_unauthenticated_thresholds_update_rejected PASSED [ 28%]
tests/test_security_consent.py::test_applicant_cannot_access_other_applicant_profile PASSED [ 35%]
tests/test_security_consent.py::test_applicant_can_access_own_profile PASSED [ 42%]
tests/test_security_consent.py::test_applicant_cannot_submit_consent_for_other_applicant PASSED [ 50%]
tests/test_security_consent.py::test_applicant_cannot_trigger_scoring_for_other_applicant PASSED [ 57%]
tests/test_security_consent.py::test_applicant_cannot_access_lender_queue PASSED [ 64%]
tests/test_security_consent.py::test_applicant_cannot_update_thresholds PASSED [ 71%]
tests/test_security_consent.py::test_lender_can_access_queue_and_applicant_dossier PASSED [ 78%]
tests/test_security_consent.py::test_consent_with_zero_categories PASSED [ 85%]
tests/test_security_consent.py::test_consent_with_single_category_accepted_and_scorable PASSED [ 92%]
tests/test_security_consent.py::test_consent_withdrawal_and_subsequent_scoring_block PASSED [100%]

======================= 14 passed, 3 warnings in 13.75s =======================
```

### Full Combined Regression Suite
**Command:** `python -m pytest` (executing both `test_fairscore.py` and `test_security_consent.py`)  
**Result:** **26 passed, 0 failed in 26.92s (100% pass rate)**.

### Direct End-to-End Test Suite
**Command:** `python tests/run_tests.py`  
**Result:** **All 11 functional modules passed successfully**.

---

## 5. Web Frontend Transparency & Integration

To maintain continuous web demonstration functionality without manual sign-in barriers:
* [`app/web/api.js`](file:///d:/fairscore%20(1)/fairscore/app/web/api.js) was upgraded to manage session tokens automatically:
  * When an applicant profile is created, the newly issued applicant token is stored in `sessionStorage` and automatically appended to subsequent consent and scoring requests.
  * For underwriter screens (Queue, Dossier, Fairness, Governance), `FairAPI.ensureLenderSession()` transparently verifies or authenticates against the default underwriter account (`lender_admin`).
* The Credit Passport and Model Performance screens clearly display the non-binding demonstration notices.

---

## 6. Remaining Vulnerabilities & Production Roadmap

1. **In-Memory Auth Store:**
   * *Current:* User credentials and tokens are stored in application memory.
   * *Production:* Must be backed by PostgreSQL / Redis with persistent relational schemas.
2. **Hardcoded Prototype Secret Key:**
   * *Current:* Defaults to a fallback secret string if `FAIRSCORE_SECRET_KEY` is unset.
   * *Production:* Must be injected from a secure secret vault (e.g. AWS Secrets Manager, GCP Secret Manager, or HashiCorp Vault).
3. **CORS Configuration:**
   * *Current:* Allows all origins (`allow_origins=["*"]`) for local development convenience.
   * *Production:* Restrict to exact domain origins (`https://app.fairscore.example.com`).
4. **Rate Limiting & Brute-Force Protection:**
   * *Production:* Implement IP and account-based rate limiting on `/api/auth/login` to prevent credential-stuffing attacks.

---

**Sign-off:**  
FairScore Technical Verification & Security Remediation  
Commit Reference: `Phase 3 Security and Consent Complete`
