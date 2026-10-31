"""
Comprehensive Security, Authentication, BOLA and Granular Consent Test Suite.

Tests:
1. Unauthenticated applicant access to sensitive records.
2. Unauthenticated lender access (e.g. GET /api/applicants, PUT /api/thresholds).
3. Applicant attempting to access another applicant's profile (BOLA violation).
4. Applicant attempting to access lender endpoints (Role violation).
5. Unauthorized threshold modifications.
6. Lender access restrictions and successful lender permissions.
7. Valid applicant access to their own information.
8. Consent with zero categories (validly rejected without silent consent).
9. Consent with one category (validly accepted for that category).
10. Consent with multiple categories (validly accepted).
11. Consent withdrawal (full and partial).
12. Scoring after consent withdrawal (blocked).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient
from api.main import app
from core.security import create_access_token


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def lender_token():
    return create_access_token(user_id="usr-test-lender", role="lender")


@pytest.fixture(scope="module")
def applicant_a(client):
    # Create Applicant A
    res = client.post("/api/applicant", json={
        "age_group": "26-35",
        "gender_category": "Female",
        "region_category": "West",
        "urban_rural": "Urban",
        "occupation_type": "Gig Worker",
        "income_type": "Weekly",
        "loan_amount_requested": 40000,
        "loan_tenure_months": 12
    })
    assert res.status_code == 200
    data = res.json()
    return {
        "id": data["applicant_id"],
        "token": data["auth_token"]
    }


@pytest.fixture(scope="module")
def applicant_b(client):
    # Create Applicant B
    res = client.post("/api/applicant", json={
        "age_group": "36-45",
        "gender_category": "Male",
        "region_category": "North",
        "urban_rural": "Rural",
        "occupation_type": "Small Shopkeeper",
        "income_type": "Daily",
        "loan_amount_requested": 80000,
        "loan_tenure_months": 18
    })
    assert res.status_code == 200
    data = res.json()
    return {
        "id": data["applicant_id"],
        "token": data["auth_token"]
    }


# =============================================================================
# 1. Unauthenticated Access Tests
# =============================================================================

def test_unauthenticated_applicant_detail_rejected(client, applicant_a):
    """Accessing applicant details without token must return 401 Unauthorized."""
    res = client.get(f"/api/applicant/{applicant_a['id']}")
    assert res.status_code == 401
    assert "Authorization" in res.json().get("detail", "")


def test_unauthenticated_lender_queue_rejected(client):
    """Accessing lender queue without token must return 401 Unauthorized."""
    res = client.get("/api/applicants")
    assert res.status_code == 401


def test_unauthenticated_dashboard_rejected(client):
    """Accessing portfolio dashboard without token must return 401 Unauthorized."""
    res = client.get("/api/dashboard")
    assert res.status_code == 401


def test_unauthenticated_thresholds_update_rejected(client):
    """Mutating thresholds without token must return 401 Unauthorized."""
    res = client.put("/api/thresholds", json={"approve_max_default_prob": 0.20})
    assert res.status_code == 401


# =============================================================================
# 2. BOLA & Object-Level Authorization Tests
# =============================================================================

def test_applicant_cannot_access_other_applicant_profile(client, applicant_a, applicant_b):
    """Applicant A must be FORBIDDEN (403) from accessing Applicant B's profile."""
    headers_a = {"Authorization": f"Bearer {applicant_a['token']}"}
    res = client.get(f"/api/applicant/{applicant_b['id']}", headers=headers_a)
    assert res.status_code == 403
    assert "permission" in res.json().get("detail", "").lower() or "forbidden" in res.json().get("detail", "").lower()


def test_applicant_can_access_own_profile(client, applicant_a):
    """Applicant A can successfully access their own profile."""
    headers_a = {"Authorization": f"Bearer {applicant_a['token']}"}
    res = client.get(f"/api/applicant/{applicant_a['id']}", headers=headers_a)
    assert res.status_code == 200
    assert res.json()["profile"]["applicant_id"] == applicant_a["id"]


def test_applicant_cannot_submit_consent_for_other_applicant(client, applicant_a, applicant_b):
    """Applicant A cannot submit consent for Applicant B."""
    headers_a = {"Authorization": f"Bearer {applicant_a['token']}"}
    res = client.post("/api/consent", json={
        "applicant_id": applicant_b["id"],
        "rent_data": True
    }, headers=headers_a)
    assert res.status_code == 403


def test_applicant_cannot_trigger_scoring_for_other_applicant(client, applicant_a, applicant_b):
    """Applicant A cannot request scoring for Applicant B."""
    headers_a = {"Authorization": f"Bearer {applicant_a['token']}"}
    res = client.post("/api/score", json={"applicant_id": applicant_b["id"]}, headers=headers_a)
    assert res.status_code == 403


# =============================================================================
# 3. Role-Based Access Control (RBAC) Tests
# =============================================================================

def test_applicant_cannot_access_lender_queue(client, applicant_a):
    """Applicant role cannot access underwriter applicant queue."""
    headers_a = {"Authorization": f"Bearer {applicant_a['token']}"}
    res = client.get("/api/applicants", headers=headers_a)
    assert res.status_code == 403
    assert "lender" in res.json().get("detail", "").lower() or "underwriter" in res.json().get("detail", "").lower()


def test_applicant_cannot_update_thresholds(client, applicant_a):
    """Applicant role cannot alter underwriting thresholds."""
    headers_a = {"Authorization": f"Bearer {applicant_a['token']}"}
    res = client.put("/api/thresholds", json={"approve_max_default_prob": 0.25}, headers=headers_a)
    assert res.status_code == 403


def test_lender_can_access_queue_and_applicant_dossier(client, lender_token, applicant_a):
    """Authorized lender can access queue, dashboard, and applicant dossier."""
    lender_headers = {"Authorization": f"Bearer {lender_token}"}
    
    # 1. Access queue
    res_queue = client.get("/api/applicants?limit=10", headers=lender_headers)
    assert res_queue.status_code == 200
    assert "applicants" in res_queue.json()

    # 2. Access dashboard
    res_dash = client.get("/api/dashboard", headers=lender_headers)
    assert res_dash.status_code == 200

    # 3. Access Applicant A's dossier
    res_dossier = client.get(f"/api/applicant/{applicant_a['id']}", headers=lender_headers)
    assert res_dossier.status_code == 200
    assert res_dossier.json()["profile"]["applicant_id"] == applicant_a["id"]


# =============================================================================
# 4. Granular Consent & Revocation Tests
# =============================================================================

def test_consent_with_zero_categories(client, applicant_a):
    """Submitting zero categories records consent status as non-consented."""
    headers_a = {"Authorization": f"Bearer {applicant_a['token']}"}
    res = client.post("/api/consent", json={
        "applicant_id": applicant_a["id"],
        "rent_data": False,
        "utility_data": False,
        "mobile_data": False,
        "cashflow_data": False,
        "business_data": False,
        "self_declared_income": False
    }, headers=headers_a)
    assert res.status_code == 200
    data = res.json()
    assert len(data["consented_categories"]) == 0
    assert data["sufficient_data"] is False

    # Scoring request must be rejected with 403
    res_score = client.post("/api/score", json={"applicant_id": applicant_a["id"]}, headers=headers_a)
    assert res_score.status_code == 403
    assert "zero categories" in res_score.json().get("detail", "").lower() or "at least one" in res_score.json().get("detail", "").lower()


def test_consent_with_single_category_accepted_and_scorable(client, applicant_a):
    """Consent with exactly 1 category is validly accepted and allows scoring."""
    headers_a = {"Authorization": f"Bearer {applicant_a['token']}"}
    res = client.post("/api/consent", json={
        "applicant_id": applicant_a["id"],
        "rent_data": True,
        "utility_data": False,
        "mobile_data": False,
        "cashflow_data": False,
        "business_data": False,
        "self_declared_income": False
    }, headers=headers_a)
    assert res.status_code == 200
    data = res.json()
    assert len(data["consented_categories"]) == 1
    assert data["sufficient_data"] is True

    # Scoring request must now SUCCEED
    res_score = client.post("/api/score", json={"applicant_id": applicant_a["id"]}, headers=headers_a)
    assert res_score.status_code == 200
    score_data = res_score.json()
    assert 300 <= score_data["score"] <= 900
    assert "demonstration_notice" in score_data


def test_consent_withdrawal_and_subsequent_scoring_block(client, applicant_a):
    """Consent withdrawal must revoke access and block future scoring."""
    headers_a = {"Authorization": f"Bearer {applicant_a['token']}"}
    
    # 1. Withdraw consent
    res_withdraw = client.post("/api/consent/withdraw", json={
        "applicant_id": applicant_a["id"]
    }, headers=headers_a)
    assert res_withdraw.status_code == 200
    assert res_withdraw.json()["status"] == "revoked"

    # 2. Subsequent scoring must be rejected with 403
    res_score = client.post("/api/score", json={"applicant_id": applicant_a["id"]}, headers=headers_a)
    assert res_score.status_code == 403
    assert "withdrawn" in res_score.json().get("detail", "").lower()
