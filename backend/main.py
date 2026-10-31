"""
FairScore Standalone FastAPI Backend.

Endpoints:
  GET  /health      – liveness check
  GET  /models      – list trained models
  GET  /metrics     – model + dataset metrics
  GET  /fairness    – fairness / bias metrics
  POST /predict     – credit score prediction
  POST /explain     – SHAP / feature importance explanation

Run with:
  uvicorn backend.main:app --reload
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.model_loader import ModelStore, predict, _compute_factors
from backend.schemas import (
    ApplicantInput,
    ExplainResponse,
    HealthResponse,
    MetricsResponse,
    ModelSummary,
    PredictionResponse,
)
from backend.utils import setup_request_logger, utc_now_iso

# ---------------------------------------------------------------------------
log = setup_request_logger("fairscore.backend")

app = FastAPI(
    title="FairScore Alternative Credit Scoring API",
    description=(
        "End-to-end alternative credit scoring using behavioural financial signals. "
        "PROTOTYPE — synthetic data only."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===========================================================================
# Startup
# ===========================================================================

@app.on_event("startup")
async def startup_event() -> None:
    log.info("FairScore backend starting up…")
    ModelStore.load_best()
    if ModelStore.is_loaded:
        log.info("Model loaded successfully.")
    else:
        log.warning("Model NOT loaded – predictions will fail until a model is trained.")


# ===========================================================================
# Middleware: Request Logging
# ===========================================================================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    log.info("→ %s %s", request.method, request.url.path)
    response = await call_next(request)
    log.info("← %s %s [%d]", request.method, request.url.path, response.status_code)
    return response


# ===========================================================================
# Endpoints
# ===========================================================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health() -> dict:
    """Liveness check."""
    return {
        "status": "ok",
        "model_loaded": ModelStore.is_loaded,
        "timestamp": utc_now_iso(),
    }


@app.get("/models", tags=["System"])
async def list_models() -> dict:
    """List all available trained models with metrics summary."""
    summaries = ModelStore.all_model_summaries
    if not summaries:
        # Try reloading
        ModelStore.load_best()
        summaries = ModelStore.all_model_summaries

    return {
        "count": len(summaries),
        "models": summaries,
    }


@app.get("/metrics", tags=["Evaluation"])
async def get_metrics() -> dict:
    """Return model evaluation + dataset + cross-validation metrics."""
    if not ModelStore.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")

    meta = ModelStore.metadata
    test_m = meta.get("test_metrics", {})
    cv_m   = meta.get("cv_metrics", {})

    # Load per-model evaluation if available
    eval_files = list((PROJECT_ROOT / "outputs" / "metrics").glob("*_evaluation.json"))
    evaluations = []
    for ef in eval_files:
        try:
            with open(ef) as fh:
                evaluations.append(json.load(fh))
        except Exception:
            pass

    return {
        "active_model": {
            "name":         meta.get("dataset_name"),
            "model_type":   meta.get("model_type"),
            "n_rows":       meta.get("n_rows"),
            "n_features":   meta.get("n_cols"),
            "timestamp":    meta.get("timestamp"),
            "test_metrics": test_m,
            "cv_metrics":   cv_m,
        },
        "dataset_metrics": {
            "default_rate": meta.get("default_rate"),
            "n_rows": meta.get("n_rows"),
        },
        "all_evaluations": evaluations,
        "disclaimer": "All metrics are from synthetic data and do not reflect real-world performance.",
    }


@app.get("/fairness", tags=["Fairness"])
async def get_fairness() -> dict:
    """Return fairness / bias audit metrics by demographic group."""
    fairness = ModelStore.fairness_data

    if not fairness:
        # Try to load from disk
        fairness_files = list((PROJECT_ROOT / "outputs" / "metrics").glob("*_fairness.json"))
        if fairness_files:
            try:
                with open(fairness_files[0]) as fh:
                    fairness = json.load(fh)
                ModelStore.fairness_data = fairness
            except Exception:
                pass

    if not fairness:
        return {
            "message": "No fairness audit data available. Run scripts/fairness_audit.py first.",
            "attributes": {},
            "alerts": [],
        }

    return fairness


@app.post("/predict", response_model=PredictionResponse, tags=["Scoring"])
async def predict_credit(applicant: ApplicantInput) -> dict:
    """Score an applicant and return credit score, grade, recommendation, and factors."""
    if not ModelStore.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")

    try:
        input_data = applicant.model_dump()
        result = predict(input_data)
        return {
            **result,
            "disclaimer": "PROTOTYPE — Trained on synthetic data. Not a credit decision.",
        }
    except Exception as exc:
        log.error("Prediction error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")


@app.post("/explain", response_model=ExplainResponse, tags=["Explainability"])
async def explain_prediction(applicant: ApplicantInput) -> dict:
    """Return SHAP / feature importance explanation for an applicant."""
    if not ModelStore.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")

    try:
        import pandas as pd
        feature_cols = ModelStore.feature_cols
        input_data   = applicant.model_dump()

        row = {col: input_data.get(col, float("nan")) for col in feature_cols}
        X   = pd.DataFrame([row], columns=feature_cols)

        factors = _compute_factors(ModelStore.pipeline, X, feature_cols)

        # Global feature importance fallback
        global_importance = []
        try:
            clf = ModelStore.pipeline.named_steps["classifier"]
            preprocessor = ModelStore.pipeline.named_steps["preprocessor"]
            try:
                feat_names = list(preprocessor.get_feature_names_out())
            except Exception:
                feat_names = feature_cols
            importances = getattr(clf, "feature_importances_", [])
            pairs = sorted(zip(feat_names, importances), key=lambda x: x[1], reverse=True)
            global_importance = [
                {"feature": str(f), "importance": round(float(v), 5)} for f, v in pairs[:15]
            ]
        except Exception:
            pass

        return {
            "positive_factors":   factors["positive"],
            "negative_factors":   factors["negative"],
            "feature_importance": global_importance,
            "disclaimer": "PROTOTYPE — Trained on synthetic data. Not a credit decision.",
        }
    except Exception as exc:
        log.error("Explanation error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Explanation failed: {exc}")


# ===========================================================================
# Error handlers
# ===========================================================================

@app.exception_handler(422)
async def validation_exception_handler(request: Request, exc) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation error",
            "detail": str(exc),
            "hint": "Check that all numeric fields are valid numbers between their documented ranges.",
        },
    )

