"""FairScore End-to-End System Integration Tests.

Tests all functional requirements (FR-01 to FR-31) and API endpoints.
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


@pytest.fixture(scope="session")
def client():
    """Create test client and trigger startup lifecycle."""
    with TestClient(app) as test_client:
        yield test_client


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


def test_dashboard_summary(client):
    """FR-29: Test portfolio dashboard metrics."""
    response = client.get("/api/dashboard")
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


def test_fairness_summary_and_bias_alerts(client):
    """FR-21 to FR-26, FR-31: Test fairness metrics and bias detection."""
    response = client.get("/api/fairness")
    assert response.status_code == 200
    data = response.json()
    assert "group_metrics" in data
    assert "bias_alerts" in data
    assert "disparate_impact" in data
    assert "summary_stats" in data

    # Check demographic slices
    gm = data["group_metrics"]
    assert "urban_rural" in gm
    assert "gender_category" in gm
    assert "occupation_type" in gm

    # Rural built-in bias check
    rural_metrics = gm["urban_rural"]
    assert len(rural_metrics) >= 2
    # Check that bias alerts were triggered as expected by the built-in scenario
    alerts = data["bias_alerts"]
    assert len(alerts) >= 1
    # Check disparate impact structure
    di = data["disparate_impact"]
    assert "urban_rural" in di
    assert "ratio" in di["urban_rural"]


def test_applicant_creation_and_consent_scoring_workflow(client):
    """FR-01 to FR-20: Full applicant journey (Profile -> Consent -> Score -> Explain)."""
    # 1. Create applicant (Reliable Gig Worker profile)
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
    assert app_id.startswith("APP-")

    # 2. Score attempt BEFORE consent should fail (FR-05)
    score_res_no_consent = client.post("/api/score", json={"applicant_id": app_id})
    assert score_res_no_consent.status_code in (400, 403)

    # 3. Submit insufficient consent (1 category only)
    insuf_consent_payload = {
        "applicant_id": app_id,
        "rent_data": True,
        "utility_data": False,
        "mobile_data": False,
        "cashflow_data": False,
        "business_data": False,
        "self_declared_income": False,
    }
    insuf_res = client.post("/api/consent", json=insuf_consent_payload)
    assert insuf_res.status_code == 200
    assert not insuf_res.json()["sufficient_data"]

    # 4. Submit granular consent (4 categories: rent, utility, mobile, cashflow)
    consent_payload = {
        "applicant_id": app_id,
        "rent_data": True,
        "utility_data": True,
        "mobile_data": True,
        "cashflow_data": True,
        "business_data": False,
        "self_declared_income": True,
    }
    consent_res = client.post("/api/consent", json=consent_payload)
    assert consent_res.status_code == 200
    c_data = consent_res.json()
    assert c_data["sufficient_data"] is True
    assert len(c_data["consented_categories"]) >= 4

    # 5. Score applicant
    score_res = client.post("/api/score", json={"applicant_id": app_id})
    assert score_res.status_code == 200
    score_data = score_res.json()
    assert score_data["applicant_id"] == app_id
    assert 300 <= score_data["score"] <= 900
    assert 0.0 <= score_data["default_probability"] <= 1.0
    assert score_data["risk_grade"] in ["A", "B", "C", "D", "E", "Low", "Medium-Low", "Medium", "Medium-High", "High"]
    assert score_data["recommendation"] in ["Approve", "Approve with Reduced Limit", "Manual Review", "Decline", "Review"]
    assert len(score_data["positive_factors"]) >= 1
    assert len(score_data["summary_text"]) > 10

    # 6. Retrieve SHAP explanation
    expl_res = client.get(f"/api/explanation/{app_id}")
    assert expl_res.status_code == 200
    expl_data = expl_res.json()
    assert "positive_factors" in expl_data
    assert "negative_factors" in expl_data
    assert "summary_text" in expl_data


def test_applicant_list_filtering(client):
    """FR-29: Test listing applicants with risk grade filtering."""
    response = client.get("/api/applicants?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert "applicants" in data
    assert len(data["applicants"]) <= 10
    first = data["applicants"][0]
    assert "applicant_id" in first
    assert "score" in first
    assert "risk_grade" in first
    assert "occupation" in first
    assert "recommendation" in first


def test_thresholds_update(client):
    """FR-17: Test dynamic update of decision thresholds."""
    update_payload = {
        "approve_max_default_prob": 0.18,
        "review_max_default_prob": 0.38,
        "min_data_coverage": 0.35,
    }
    response = client.put("/api/thresholds", json=update_payload)
    assert response.status_code == 200
    data = response.json()
    assert "thresholds" in data
    assert data["thresholds"]["approve_max_default_prob"] == 0.18
