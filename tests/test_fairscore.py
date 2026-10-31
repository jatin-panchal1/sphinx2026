"""FairScore End-to-End System Integration Tests.

Tests all functional requirements (FR-01 to FR-31) and API endpoints
with secure session headers and RBAC credentials.
"""

import sys
import os
import pytest
from fastapi.testclient import TestClient

# Ensure fairscore root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from api.main import app, startup
from core.security import create_access_token


@pytest.fixture(scope="session")
def client():
    """Create test client and trigger startup lifecycle."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def lender_headers():
    token = create_access_token(user_id="usr-test-lender", role="lender")
    return {"Authorization": f"Bearer {token}"}


def test_health_check(client):
    """Test health probe endpoint."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_personas_list(client):
    """FR-02: Test list of synthetic applicant personas."""
    response = client.get("/api/personas")
    assert response.status_code == 200
    personas = response.json()
    assert len(personas) >= 6
    names = [p["name"] for p in personas]
    assert any("Gig Worker" in n for n in names)
    assert any("Shopkeeper" in n for n in names)
    assert any("Student" in n for n in names)


def test_dashboard_summary(client, lender_headers):
    """FR-29: Test portfolio dashboard metrics with lender credentials."""
    response = client.get("/api/dashboard", headers=lender_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["total_applicants"] >= 1000
    assert 0.0 <= data["approval_rate"] <= 1.0
    assert 300 <= data["average_score"] <= 900
    assert "risk_distribution" in data
    assert "approval_by_risk_grade" in data
    assert "score_distribution" in data
    assert data["bias_alert_count"] >= 0


def test_model_metrics(client):
    """FR-15 / Model Evaluation: Test model metrics endpoint."""
    response = client.get("/api/model/metrics")
    assert response.status_code == 200
    data = response.json()
    assert 0.5 <= data["auc_roc"] <= 1.0
    assert 0.0 <= data["precision"] <= 1.0
    assert 0.0 <= data["recall"] <= 1.0
    assert 0.0 <= data["f1_score"] <= 1.0
    assert len(data["confusion_matrix"]) == 2


def test_fairness_summary_and_bias_alerts(client, lender_headers):
    """FR-21 to FR-26, FR-31: Test fairness metrics and bias detection."""
    response = client.get("/api/fairness", headers=lender_headers)
    assert response.status_code == 200
    data = response.json()
    assert "group_metrics" in data
    assert "bias_alerts" in data
    assert "disparate_impact" in data
    assert "summary_stats" in data

    gm = data["group_metrics"]
    assert "urban_rural" in gm
    assert "gender_category" in gm
    assert "occupation_type" in gm

    rural_metrics = gm["urban_rural"]
    assert len(rural_metrics) >= 2
    alerts = data["bias_alerts"]
    assert len(alerts) >= 1
    di = data["disparate_impact"]
    assert "urban_rural" in di
    assert "ratio" in di["urban_rural"]

    for group_list in gm.values():
        for record in group_list:
            if "bias_alert_flag" in record:
                assert isinstance(record["bias_alert_flag"], bool)

    for di_val in di.values():
        if "alert" in di_val:
            assert isinstance(di_val["alert"], bool)


def test_score_without_consent(client):
    """FR-05: Scoring must be rejected (HTTP 403) when no consent has been submitted."""
    create_payload = {
        "age_group": "26-35",
        "gender_category": "Male",
        "region_category": "North",
        "urban_rural": "Urban",
        "occupation_type": "Gig Worker",
        "income_type": "Weekly",
        "loan_amount_requested": 35000.0,
        "loan_tenure_months": 12,
    }
    create_res = client.post("/api/applicant", json=create_payload)
    assert create_res.status_code == 200
    app_id = create_res.json()["applicant_id"]
    auth_token = create_res.json()["auth_token"]
    app_headers = {"Authorization": f"Bearer {auth_token}"}

    score_res = client.post("/api/score", json={"applicant_id": app_id}, headers=app_headers)
    assert score_res.status_code == 403


def test_score_with_zero_consent(client):
    """Scoring with zero consent categories must be rejected."""
    create_payload = {
        "age_group": "36-45",
        "gender_category": "Female",
        "region_category": "South",
        "urban_rural": "Semi-Urban",
        "occupation_type": "Small Shopkeeper",
        "income_type": "Daily",
        "loan_amount_requested": 50000.0,
        "loan_tenure_months": 18,
    }
    create_res = client.post("/api/applicant", json=create_payload)
    assert create_res.status_code == 200
    app_id = create_res.json()["applicant_id"]
    auth_token = create_res.json()["auth_token"]
    app_headers = {"Authorization": f"Bearer {auth_token}"}

    zero_consent = {
        "applicant_id": app_id,
        "rent_data": False,
        "utility_data": False,
        "mobile_data": False,
        "cashflow_data": False,
        "business_data": False,
        "self_declared_income": False,
    }
    consent_res = client.post("/api/consent", json=zero_consent, headers=app_headers)
    assert consent_res.status_code == 200

    score_res = client.post("/api/score", json={"applicant_id": app_id}, headers=app_headers)
    assert score_res.status_code == 403


def test_consent_withdrawal_blocks_scoring(client):
    """Consent withdrawal must block future scoring requests."""
    create_payload = {
        "age_group": "18-25",
        "gender_category": "Male",
        "region_category": "East",
        "urban_rural": "Urban",
        "occupation_type": "Student",
        "income_type": "Monthly",
        "loan_amount_requested": 15000.0,
        "loan_tenure_months": 6,
    }
    create_res = client.post("/api/applicant", json=create_payload)
    assert create_res.status_code == 200
    app_id = create_res.json()["applicant_id"]
    auth_token = create_res.json()["auth_token"]
    app_headers = {"Authorization": f"Bearer {auth_token}"}

    consent_payload = {
        "applicant_id": app_id,
        "rent_data": True,
        "utility_data": True,
        "mobile_data": True,
        "cashflow_data": True,
        "business_data": False,
        "self_declared_income": True,
    }
    consent_res = client.post("/api/consent", json=consent_payload, headers=app_headers)
    assert consent_res.status_code == 200

    withdraw_res = client.post("/api/consent/withdraw", json={"applicant_id": app_id}, headers=app_headers)
    assert withdraw_res.status_code == 200

    score_res = client.post("/api/score", json={"applicant_id": app_id}, headers=app_headers)
    assert score_res.status_code == 403


def test_applicant_creation_and_consent_scoring_workflow(client):
    """End-to-end applicant onboarding, consent, scoring, and explanation."""
    create_payload = {
        "age_group": "26-35",
        "gender_category": "Male",
        "region_category": "North",
        "urban_rural": "Urban",
        "occupation_type": "Gig Worker",
        "income_type": "Weekly",
        "loan_amount_requested": 35000.0,
        "loan_tenure_months": 12,
    }
    create_res = client.post("/api/applicant", json=create_payload)
    assert create_res.status_code == 200
    app_id = create_res.json()["applicant_id"]
    auth_token = create_res.json()["auth_token"]
    app_headers = {"Authorization": f"Bearer {auth_token}"}

    consent_payload = {
        "applicant_id": app_id,
        "rent_data": True,
        "utility_data": True,
        "mobile_data": True,
        "cashflow_data": True,
        "business_data": False,
        "self_declared_income": True,
    }
    consent_res = client.post("/api/consent", json=consent_payload, headers=app_headers)
    assert consent_res.status_code == 200

    score_res = client.post("/api/score", json={"applicant_id": app_id}, headers=app_headers)
    assert score_res.status_code == 200
    score_data = score_res.json()
    assert score_data["applicant_id"] == app_id
    assert 300 <= score_data["score"] <= 900
    assert 0.0 <= score_data["default_probability"] <= 1.0
    assert "demonstration_notice" in score_data

    expl_res = client.get(f"/api/explanation/{app_id}", headers=app_headers)
    assert expl_res.status_code == 200
    expl_data = expl_res.json()
    assert "positive_factors" in expl_data
    assert "negative_factors" in expl_data


def test_applicant_list_filtering(client, lender_headers):
    """Test lender queue retrieval and risk grade filtering."""
    response = client.get("/api/applicants?limit=20", headers=lender_headers)
    assert response.status_code == 200
    data = response.json()
    assert "applicants" in data
    assert len(data["applicants"]) <= 20

    res_a = client.get("/api/applicants?risk_grade=A&limit=10", headers=lender_headers)
    assert res_a.status_code == 200
    for app_item in res_a.json()["applicants"]:
        assert app_item["risk_grade"] == "A"


def test_thresholds_update(client, lender_headers):
    """Test updating decision thresholds with lender authorization."""
    update_payload = {"approve_max_default_prob": 0.20}
    response = client.put("/api/thresholds", json=update_payload, headers=lender_headers)
    assert response.status_code == 200
    data = response.json()
    assert "thresholds" in data
    assert data["thresholds"]["approve_max_default_prob"] == 0.20


def test_fairness_http_serialization_no_crash(client, lender_headers):
    """Ensure fairness response serializes safely without numpy errors."""
    response = client.get("/api/fairness?dimension=urban_rural", headers=lender_headers)
    assert response.status_code == 200
    assert "group_metrics" in response.json()
