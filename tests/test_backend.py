"""
Backend integration tests for the FairScore standalone FastAPI backend.

Tests all public endpoints using FastAPI TestClient.
"""

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Setup: ensure a model exists before running backend tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def ensure_model_trained():
    """Train a small synthetic model if none exists."""
    from pathlib import Path
    metrics_dir = PROJECT_ROOT / "outputs" / "metrics"
    if not list(metrics_dir.glob("*_metrics.json")):
        import scripts.train_model as tm
        tm.train_all(use_xgboost=False, n_synthetic=1_000)


@pytest.fixture(scope="session")
def client():
    """Return a FastAPI TestClient for the backend."""
    from fastapi.testclient import TestClient
    from backend.main import app
    with TestClient(app) as c:
        yield c


VALID_PAYLOAD = {
    "monthly_income": 25000.0,
    "income_volatility": 0.2,
    "rent_payment_score": 0.8,
    "utility_payment_score": 0.75,
    "mobile_recharge_regularity": 0.7,
    "cashflow_consistency": 0.72,
    "savings_rate": 0.12,
    "verified_cash_payments": 8,
    "months_in_current_job": 24,
    "num_dependants": 2,
    "expense_to_income_ratio": 0.55,
    "prior_defaults": 0,
    "loan_amount_requested": 50000.0,
    "loan_to_income_ratio": 0.17,
}


# ===========================================================================
# Tests: GET /health
# ===========================================================================

class TestHealth:
    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_status_ok(self, client):
        r = client.get("/health")
        assert r.json()["status"] == "ok"

    def test_health_has_timestamp(self, client):
        r = client.get("/health")
        assert "timestamp" in r.json()

    def test_health_has_model_loaded(self, client):
        r = client.get("/health")
        assert "model_loaded" in r.json()


# ===========================================================================
# Tests: GET /models
# ===========================================================================

class TestModels:
    def test_models_returns_200(self, client):
        r = client.get("/models")
        assert r.status_code == 200

    def test_models_has_count(self, client):
        r = client.get("/models")
        assert "count" in r.json()

    def test_models_has_list(self, client):
        r = client.get("/models")
        data = r.json()
        assert "models" in data
        assert isinstance(data["models"], list)


# ===========================================================================
# Tests: GET /metrics
# ===========================================================================

class TestMetrics:
    def test_metrics_returns_200(self, client):
        r = client.get("/metrics")
        assert r.status_code == 200

    def test_metrics_has_active_model(self, client):
        r = client.get("/metrics")
        assert "active_model" in r.json()

    def test_metrics_has_roc_auc(self, client):
        r = client.get("/metrics")
        data = r.json()
        tm = data.get("active_model", {}).get("test_metrics", {})
        assert "roc_auc" in tm
        assert 0.0 <= tm["roc_auc"] <= 1.0

    def test_metrics_has_disclaimer(self, client):
        r = client.get("/metrics")
        data = r.json()
        assert "disclaimer" in data


# ===========================================================================
# Tests: GET /fairness
# ===========================================================================

class TestFairness:
    def test_fairness_returns_200(self, client):
        r = client.get("/fairness")
        assert r.status_code == 200

    def test_fairness_response_is_json(self, client):
        r = client.get("/fairness")
        assert r.headers["content-type"].startswith("application/json")


# ===========================================================================
# Tests: POST /predict
# ===========================================================================

class TestPredict:
    def test_predict_returns_200_valid(self, client):
        r = client.post("/predict", json=VALID_PAYLOAD)
        assert r.status_code == 200

    def test_predict_has_credit_score(self, client):
        r = client.post("/predict", json=VALID_PAYLOAD)
        data = r.json()
        assert "credit_score" in data
        assert 300 <= data["credit_score"] <= 900

    def test_predict_probability_in_range(self, client):
        r = client.post("/predict", json=VALID_PAYLOAD)
        prob = r.json()["default_probability"]
        assert 0.0 <= prob <= 1.0

    def test_predict_has_grade(self, client):
        r = client.post("/predict", json=VALID_PAYLOAD)
        grade = r.json()["risk_grade"]
        assert grade in ("A", "B", "C", "D", "E")

    def test_predict_has_recommendation(self, client):
        r = client.post("/predict", json=VALID_PAYLOAD)
        rec = r.json()["recommendation"]
        assert rec in ("Approve", "Approve with Conditions", "Manual Review", "Decline")

    def test_predict_has_factors(self, client):
        r = client.post("/predict", json=VALID_PAYLOAD)
        data = r.json()
        assert "top_positive_factors" in data
        assert "top_negative_factors" in data

    def test_predict_handles_all_null_fields(self, client):
        """Empty payload should not crash – model should handle missing values."""
        r = client.post("/predict", json={})
        # Should return 200 (imputed) or a meaningful error, not 500
        assert r.status_code in (200, 422, 503)

    def test_predict_handles_invalid_types(self, client):
        """String where float expected should return 422."""
        bad = {**VALID_PAYLOAD, "monthly_income": "not-a-number"}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_predict_has_disclaimer(self, client):
        r = client.post("/predict", json=VALID_PAYLOAD)
        assert "disclaimer" in r.json()


# ===========================================================================
# Tests: POST /explain
# ===========================================================================

class TestExplain:
    def test_explain_returns_200(self, client):
        r = client.post("/explain", json=VALID_PAYLOAD)
        assert r.status_code == 200

    def test_explain_has_factors(self, client):
        r = client.post("/explain", json=VALID_PAYLOAD)
        data = r.json()
        assert "positive_factors" in data
        assert "negative_factors" in data

    def test_explain_has_feature_importance(self, client):
        r = client.post("/explain", json=VALID_PAYLOAD)
        data = r.json()
        assert "feature_importance" in data
        assert isinstance(data["feature_importance"], list)

    def test_explain_has_disclaimer(self, client):
        r = client.post("/explain", json=VALID_PAYLOAD)
        assert "disclaimer" in r.json()

