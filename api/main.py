"""
FairScore API - FastAPI Application.

Complete REST API for alternative credit scoring, including applicant management,
consent handling, scoring, explainability, fairness auditing, and dashboards.
"""

import sys
import os
import uuid
import logging
from typing import Optional
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------------------------
# Path setup so sibling packages are importable when running from project root
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from api.schemas import (
    ApplicantCreate,
    ApplicantDetailResponse,
    ApplicantResponse,
    ConsentResponse,
    ConsentSubmit,
    DashboardSummaryResponse,
    ExplanationResponse,
    FairnessSummaryResponse,
    ModelMetricsResponse,
    ScoreRequest,
    ScoreResponse,
    ThresholdUpdate,
)
from data.synthetic_generator import generate_full_dataset, generate_single_applicant
from core.feature_engineering import FeatureEngineer
from core.scoring_model import CreditScoringModel
from core.explainability import ExplainabilityEngine
from core.fairness import FairnessAuditor
from core.decision_engine import DecisionEngine
from core.consent import ConsentManager

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fairscore.api")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="FairScore API",
    description=(
        "Alternative Credit Scoring for the Credit-Invisible. "
        "Uses rent payments, utility bills, mobile usage, cashflow, "
        "and business data to generate fair, explainable credit scores."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global State  (populated on startup)
# ---------------------------------------------------------------------------
dataset: Optional[dict] = None
feature_engineer: Optional[FeatureEngineer] = None
model: Optional[CreditScoringModel] = None
explainer: Optional[ExplainabilityEngine] = None
fairness_auditor: Optional[FairnessAuditor] = None
decision_engine: Optional[DecisionEngine] = None
consent_manager: Optional[ConsentManager] = None
features_df: Optional[pd.DataFrame] = None

predictions_cache: dict = {}        # applicant_id  -> prediction dict
explanations_cache: dict = {}       # applicant_id  -> explanation dict
applicant_store: dict = {}          # applicant_id  -> raw applicant row dict
score_results_cache: dict = {}      # applicant_id  -> full ScoreResponse dict

feature_cols: list[str] = []
score_counter: int = 0

# ---------------------------------------------------------------------------
# Available synthetic personas for /api/personas
# ---------------------------------------------------------------------------
PERSONAS = [
    {
        "key": "gig_worker",
        "label": "Gig Worker",
        "description": "Urban delivery/ride-sharing worker with irregular digital income",
        "defaults": {
            "age_group": "26-35",
            "gender_category": "Male",
            "region_category": "West",
            "urban_rural": "Urban",
            "occupation_type": "Gig Worker",
            "income_type": "Irregular",
            "loan_amount_requested": 50000,
            "loan_tenure_months": 12,
        },
    },
    {
        "key": "small_shopkeeper",
        "label": "Small Shopkeeper",
        "description": "Semi-urban shopkeeper with daily cash income",
        "defaults": {
            "age_group": "36-45",
            "gender_category": "Male",
            "region_category": "North",
            "urban_rural": "Semi-Urban",
            "occupation_type": "Small Shopkeeper",
            "income_type": "Daily",
            "loan_amount_requested": 100000,
            "loan_tenure_months": 18,
        },
    },
    {
        "key": "student",
        "label": "Student",
        "description": "Young student seeking education or emergency loan",
        "defaults": {
            "age_group": "18-25",
            "gender_category": "Female",
            "region_category": "South",
            "urban_rural": "Urban",
            "occupation_type": "Student",
            "income_type": "Monthly",
            "loan_amount_requested": 25000,
            "loan_tenure_months": 6,
        },
    },
    {
        "key": "homemaker",
        "label": "Homemaker",
        "description": "Homemaker managing household finances with utility/rent track record",
        "defaults": {
            "age_group": "36-45",
            "gender_category": "Female",
            "region_category": "Central",
            "urban_rural": "Semi-Urban",
            "occupation_type": "Homemaker",
            "income_type": "Monthly",
            "loan_amount_requested": 30000,
            "loan_tenure_months": 12,
        },
    },
    {
        "key": "freelancer",
        "label": "Freelancer",
        "description": "Urban freelancer with irregular but high digital earnings",
        "defaults": {
            "age_group": "26-35",
            "gender_category": "Non-Binary",
            "region_category": "South",
            "urban_rural": "Urban",
            "occupation_type": "Freelancer",
            "income_type": "Irregular",
            "loan_amount_requested": 75000,
            "loan_tenure_months": 12,
        },
    },
    {
        "key": "daily_wage_worker",
        "label": "Daily Wage Worker",
        "description": "Rural labourer with daily cash income and limited digital footprint",
        "defaults": {
            "age_group": "36-45",
            "gender_category": "Male",
            "region_category": "East",
            "urban_rural": "Rural",
            "occupation_type": "Daily Wage Worker",
            "income_type": "Daily",
            "loan_amount_requested": 15000,
            "loan_tenure_months": 6,
        },
    },
    {
        "key": "street_vendor",
        "label": "Street Vendor",
        "description": "Urban street vendor with daily cash flow and mobile payments",
        "defaults": {
            "age_group": "46-55",
            "gender_category": "Female",
            "region_category": "West",
            "urban_rural": "Urban",
            "occupation_type": "Street Vendor",
            "income_type": "Daily",
            "loan_amount_requested": 20000,
            "loan_tenure_months": 9,
        },
    },
    {
        "key": "artisan",
        "label": "Artisan",
        "description": "Rural artisan with seasonal income from handicrafts",
        "defaults": {
            "age_group": "46-55",
            "gender_category": "Female",
            "region_category": "North",
            "urban_rural": "Rural",
            "occupation_type": "Artisan",
            "income_type": "Irregular",
            "loan_amount_requested": 40000,
            "loan_tenure_months": 12,
        },
    },
]


# =========================================================================
#  Helper utilities
# =========================================================================


def _applicant_id_in_synthetic(applicant_id: str) -> bool:
    """Check whether an applicant_id belongs to the pre-generated dataset."""
    return features_df is not None and applicant_id in features_df["applicant_id"].values


def _dp_to_recommendation(default_probability: float) -> str:
    """Map a default probability to a recommendation string using the DecisionEngine thresholds.

    Falls back to fixed thresholds if the decision engine is not yet initialised.
    """
    if decision_engine is not None:
        approve_thresh = decision_engine.thresholds.get("approve_max_default_prob", 0.25)
        review_thresh = decision_engine.thresholds.get("review_max_default_prob", 0.50)
    else:
        approve_thresh = 0.25
        review_thresh = 0.50

    if default_probability <= approve_thresh:
        return "Approve"
    elif default_probability <= review_thresh:
        return "Manual Review"
    else:
        return "Decline"


def _get_applicant_profile(applicant_id: str) -> dict:
    """Return the raw profile dict for an applicant (synthetic or custom)."""
    if applicant_id in applicant_store:
        return applicant_store[applicant_id]

    if dataset is not None and "applicants" in dataset:
        df = dataset["applicants"]
        row = df.loc[df["applicant_id"] == applicant_id]
        if not row.empty:
            return row.iloc[0].to_dict()

    raise HTTPException(status_code=404, detail=f"Applicant {applicant_id} not found")


def _risk_grade_from_score(score: int) -> str:
    """Map a 300-900 score to a letter risk grade."""
    if score >= 780:
        return "A"
    if score >= 700:
        return "B"
    if score >= 620:
        return "C"
    if score >= 540:
        return "D"
    return "E"


def _confidence_label(data_coverage: float) -> str:
    """Map data coverage fraction to a confidence label."""
    if data_coverage >= 0.7:
        return "High"
    if data_coverage >= 0.4:
        return "Medium"
    return "Low"


def _generate_improvement_suggestions(negative_factors: list[dict]) -> list[str]:
    """Generate actionable improvement suggestions from negative factors."""
    suggestions: list[str] = []
    feature_to_suggestion = {
        "rent": "Build a consistent rent payment history over the next 3-6 months",
        "utility": "Ensure utility bills are paid on time to strengthen your profile",
        "mobile": "Maintain regular mobile recharge patterns for better scoring",
        "cashflow": "Keep a steady savings pattern in your bank or wallet account",
        "business": "Maintain regular business revenue records for improved credibility",
        "income": "Provide verifiable income documentation to improve confidence",
        "payment": "Reduce late payments to improve your credit profile",
        "default": "Avoid defaults on existing obligations",
        "balance": "Try to maintain a positive and stable account balance",
    }
    for factor in negative_factors:
        name = factor.get("feature", "").lower()
        for keyword, suggestion in feature_to_suggestion.items():
            if keyword in name and suggestion not in suggestions:
                suggestions.append(suggestion)
    if not suggestions:
        suggestions.append("Continue maintaining timely payments across all data sources")
    return suggestions


# =========================================================================
#  Startup
# =========================================================================


@app.on_event("startup")
async def startup() -> None:
    """Initialise the full pipeline: data → features → model → explainer."""
    global dataset, feature_engineer, model, explainer, fairness_auditor
    global decision_engine, consent_manager, features_df, predictions_cache
    global feature_cols, explanations_cache, applicant_store, score_results_cache

    logger.info("🚀 FairScore API starting up …")

    # 1. Generate synthetic dataset ----------------------------------------
    logger.info("Generating synthetic dataset (2000 applicants) …")
    dataset = generate_full_dataset(n_applicants=2000)

    # 2. Engineer features -------------------------------------------------
    logger.info("Engineering features …")
    feature_engineer = FeatureEngineer()
    features_df = feature_engineer.engineer_features(
        dataset["applicants"],
        dataset["consents"],
        dataset["rent"],
        dataset["utility"],
        dataset["mobile"],
        dataset["cashflow"],
        dataset["business"],
    )

    # 3. Align features with labels ----------------------------------------
    labels_df = dataset["labels"]
    if "default_label" not in labels_df.columns and "defaulted" in labels_df.columns:
        labels_df["default_label"] = labels_df["defaulted"]
    # Merge on applicant_id so we have matching rows
    merged = features_df.merge(labels_df[["applicant_id", "default_label"]], on="applicant_id", how="inner")
    features_df = merged.drop(columns=["default_label"])
    y = merged["default_label"]

    # 4. Train model -------------------------------------------------------
    logger.info("Training gradient-boosting model …")
    model = CreditScoringModel(model_type="gradient_boosting")
    feature_cols = model.FEATURE_COLUMNS
    # Ensure only known columns are used – fill missing with 0
    for col in feature_cols:
        if col not in features_df.columns:
            features_df[col] = 0
    X = features_df[feature_cols].fillna(0)
    model.train(X, y)

    # 5. Initialise explainer ----------------------------------------------
    logger.info("Initialising SHAP explainer …")
    explainer = ExplainabilityEngine(model.model, model.scaler, feature_cols)
    explainer.initialize(X)

    # 6. Other components --------------------------------------------------
    fairness_auditor = FairnessAuditor()
    decision_engine = DecisionEngine()
    consent_manager = ConsentManager()

    # 7. Pre-compute predictions for synthetic applicants ------------------
    logger.info("Pre-computing predictions for synthetic applicants …")
    for _, row in features_df.iterrows():
        app_id = row["applicant_id"]
        pred = model.predict_single(row[feature_cols].to_dict())
        predictions_cache[app_id] = pred

    logger.info("✅ Startup complete – ready to serve requests.")


# =========================================================================
#  Endpoints
# =========================================================================


# ---- Applicant Management ------------------------------------------------


@app.post("/api/applicant", response_model=ApplicantResponse)
async def create_applicant(applicant: ApplicantCreate) -> ApplicantResponse:
    """Create a new applicant profile and generate synthetic alternative data."""
    global score_counter

    occ = applicant.occupation_type.lower()
    if "gig" in occ:
        persona_str = "reliable_gig_worker"
    elif "shopkeeper" in occ:
        persona_str = "stable_shopkeeper"
    elif "student" in occ:
        persona_str = "student"
    elif "informal" in occ or "wage" in occ:
        persona_str = "informal_worker"
    else:
        persona_str = "random"

    try:
        applicant_data = generate_single_applicant(persona=persona_str)
    except Exception as exc:
        logger.error("Failed to generate applicant: %s", exc)
        raise HTTPException(status_code=500, detail=f"Data generation failed: {exc}")

    # Assign custom ID and override with submitted profile fields
    app_id = f"APP-CUSTOM-{uuid.uuid4().hex[:6].upper()}"
    applicant_data["applicant_id"] = app_id
    if "applicant" in applicant_data and isinstance(applicant_data["applicant"], dict):
        applicant_data["applicant"].update(
            {
                "applicant_id": app_id,
                "age_group": applicant.age_group,
                "gender_category": applicant.gender_category,
                "region_category": applicant.region_category,
                "urban_rural": applicant.urban_rural,
                "occupation_type": applicant.occupation_type,
                "income_type": applicant.income_type,
                "loan_amount_requested": applicant.loan_amount_requested,
                "loan_tenure_months": applicant.loan_tenure_months,
            }
        )
    if "consent" in applicant_data and isinstance(applicant_data["consent"], dict):
        applicant_data["consent"]["applicant_id"] = app_id

    applicant_store[app_id] = applicant_data
    score_counter += 1

    return ApplicantResponse(
        applicant_id=app_id,
        message=f"Applicant {app_id} created successfully. Submit consent to proceed.",
    )


@app.get("/api/applicant/{applicant_id}", response_model=ApplicantDetailResponse)
async def get_applicant_detail(applicant_id: str) -> ApplicantDetailResponse:
    """Return full detail for an applicant: profile, consent, score, explanation."""
    profile = _get_applicant_profile(applicant_id)

    # Consent status
    try:
        consent_status = consent_manager.get_consent_status(applicant_id)  # type: ignore[union-attr]
    except Exception:
        consent_status = {"status": "not_submitted"}

    score_result = score_results_cache.get(applicant_id)
    explanation = explanations_cache.get(applicant_id)

    return ApplicantDetailResponse(
        profile=profile,
        consent_status=consent_status,
        score_result=score_result,
        explanation=explanation,
    )


@app.get("/api/applicants")
async def list_applicants(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    risk_grade: Optional[str] = Query(None, description="Filter by risk grade (A-E)"),
) -> dict:
    """List applicants with pagination and optional risk-grade filter."""
    if features_df is None:
        raise HTTPException(status_code=503, detail="System not initialised yet")

    records: list[dict] = []
    app_occ_map = {}
    if dataset and "applicants" in dataset:
        app_occ_map = dict(zip(dataset["applicants"]["applicant_id"], dataset["applicants"]["occupation_type"]))

    filter_grade = risk_grade.upper() if isinstance(risk_grade, str) else None

    # Synthetic applicants
    for _, row in features_df.iterrows():
        app_id = row["applicant_id"]
        pred = predictions_cache.get(app_id, {})
        score = pred.get("score", 0)
        grade = _risk_grade_from_score(score)

        if filter_grade and grade != filter_grade:
            continue

        dp = pred.get("default_probability", 0.5)
        rec = _dp_to_recommendation(dp)
        records.append(
            {
                "applicant_id": app_id,
                "occupation": app_occ_map.get(app_id, "N/A"),
                "score": score,
                "risk_grade": grade,
                "default_probability": round(dp, 4),
                "recommendation": rec,
            }
        )

    # Custom applicants that have been scored
    for app_id, sr in score_results_cache.items():
        if _applicant_id_in_synthetic(app_id):
            continue
        grade = sr.get("risk_grade", "E")
        if filter_grade and grade != filter_grade:
            continue
        occ = applicant_store.get(app_id, {}).get("applicant", {}).get("occupation_type", "N/A")
        records.append(
            {
                "applicant_id": app_id,
                "occupation": occ,
                "score": sr.get("score", 0),
                "risk_grade": grade,
                "default_probability": round(sr.get("default_probability", 0.0), 4),
                "recommendation": sr.get("recommendation", "Review"),
            }
        )

    total = len(records)
    lim = limit if isinstance(limit, int) else 50
    off = offset if isinstance(offset, int) else 0
    paginated = records[off : off + lim]

    return {"total": total, "limit": lim, "offset": off, "applicants": paginated}


# ---- Consent -------------------------------------------------------------


@app.post("/api/consent", response_model=ConsentResponse)
async def submit_consent(consent: ConsentSubmit) -> ConsentResponse:
    """Submit or update data-sharing consent for an applicant."""
    if consent_manager is None:
        raise HTTPException(status_code=503, detail="System not initialised yet")

    # Verify applicant exists (either synthetic or custom)
    if not _applicant_id_in_synthetic(consent.applicant_id) and consent.applicant_id not in applicant_store:
        raise HTTPException(status_code=404, detail=f"Applicant {consent.applicant_id} not found")

    consent_dict = {
        "rent_data": consent.rent_data,
        "utility_data": consent.utility_data,
        "mobile_data": consent.mobile_data,
        "cashflow_data": consent.cashflow_data,
        "business_data": consent.business_data,
        "self_declared_income": consent.self_declared_income,
    }

    try:
        result = consent_manager.submit_consent(consent.applicant_id, consent_dict)
    except Exception as exc:
        logger.error("Consent submission failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Consent submission error: {exc}")

    consented = [k.replace("_data", "").replace("_", " ").title() for k, v in consent_dict.items() if v]
    sufficient = result.get("sufficient_data", len(consented) >= 2)
    timestamp = result.get("timestamp", datetime.now(timezone.utc).isoformat())

    msg = "Consent recorded." if sufficient else "Consent recorded, but more data categories are recommended for a reliable score."

    return ConsentResponse(
        applicant_id=consent.applicant_id,
        consented_categories=consented,
        consent_timestamp=timestamp,
        sufficient_data=sufficient,
        message=msg,
    )


@app.get("/api/consent/{applicant_id}")
async def get_consent_status(applicant_id: str) -> dict:
    """Get the current consent status for an applicant."""
    if consent_manager is None:
        raise HTTPException(status_code=503, detail="System not initialised yet")

    try:
        status = consent_manager.get_consent_status(applicant_id)
    except Exception:
        status = {"applicant_id": applicant_id, "status": "not_submitted", "categories": []}

    return status


# ---- Scoring & Explanation -----------------------------------------------


@app.post("/api/score", response_model=ScoreResponse)
async def generate_score(request: ScoreRequest) -> ScoreResponse:
    """Generate an alternative credit score for an applicant.

    Pipeline:
    1. Verify consent exists and is sufficient.
    2. Gather / engineer features (respecting consent).
    3. Run model prediction.
    4. Generate SHAP explanation.
    5. Run decision engine for recommendation.
    6. Return full response.
    """
    if model is None or explainer is None or feature_engineer is None or consent_manager is None or decision_engine is None:
        raise HTTPException(status_code=503, detail="System not initialised yet")

    app_id = request.applicant_id

    # --- 1. Check consent -------------------------------------------------
    try:
        consent_status = consent_manager.get_consent_status(app_id)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail=f"No consent found for {app_id}. Submit consent first via POST /api/consent.",
        )

    # Explicit consent gate: status must be "active" and sufficient_data must be True
    _consent_status_val = consent_status.get("status", "not_submitted")
    _sufficient = consent_status.get("sufficient_data", False)

    if _consent_status_val == "not_submitted":
        raise HTTPException(
            status_code=403,
            detail=(
                f"Applicant {app_id} has not submitted consent. "
                "Scoring requires explicit consent with at least 2 data categories. "
                "Submit consent via POST /api/consent first."
            ),
        )

    if _consent_status_val == "revoked":
        raise HTTPException(
            status_code=403,
            detail=(
                f"Consent for applicant {app_id} has been withdrawn. "
                "Scoring cannot proceed without active consent."
            ),
        )

    if not _sufficient:
        raise HTTPException(
            status_code=403,
            detail=(
                f"Insufficient consent for applicant {app_id}. "
                "At least 2 data categories must be consented for alternative credit scoring. "
                "Update consent via POST /api/consent."
            ),
        )

    # Determine data coverage from consent
    consent_categories = consent_status.get("categories", [])
    if isinstance(consent_status, dict):
        consent_flags = {
            k: v
            for k, v in consent_status.items()
            if isinstance(v, bool)
        }
        consented_count = sum(1 for v in consent_flags.values() if v)
    else:
        consented_count = len(consent_categories) if consent_categories else 0
    data_coverage = min(consented_count / 6.0, 1.0)  # 6 possible categories

    # --- 2. Feature engineering -------------------------------------------
    if _applicant_id_in_synthetic(app_id):
        # Use pre-computed features
        row = features_df.loc[features_df["applicant_id"] == app_id].iloc[0]
        feature_values = row[feature_cols].to_dict()
    else:
        # Custom applicant – engineer features from generated data
        if app_id not in applicant_store:
            raise HTTPException(status_code=404, detail=f"Applicant {app_id} not found")
        applicant_data = applicant_store[app_id]
        try:
            feat_res = feature_engineer.engineer_single(applicant_data, consent_status)
            feature_values = feat_res[feature_cols].iloc[0].to_dict() if isinstance(feat_res, pd.DataFrame) else feat_res
        except Exception as exc:
            logger.error("Feature engineering failed for %s: %s", app_id, exc)
            raise HTTPException(status_code=500, detail=f"Feature engineering error: {exc}")

    # --- 3. Predict -------------------------------------------------------
    try:
        pred = model.predict_single(feature_values)
    except Exception as exc:
        logger.error("Prediction failed for %s: %s", app_id, exc)
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")

    predictions_cache[app_id] = pred

    score = int(pred.get("score", 500))
    default_prob = float(pred.get("default_probability", 0.5))
    # Compute grade here so it can be passed to the explainer
    risk_grade = _risk_grade_from_score(score)

    # --- 4. Explain -------------------------------------------------------
    try:
        explanation = explainer.explain_single(feature_values, score=score, risk_grade=risk_grade)
    except Exception as exc:
        logger.warning("Explanation generation failed for %s: %s", app_id, exc)
        explanation = {
            "positive_factors": [],
            "negative_factors": [],
            "feature_contributions": {},
            "summary_text": "Explanation unavailable.",
            "reason_codes": [],
        }

    explanations_cache[app_id] = explanation

    positive_factors = explanation.get("positive_factors", [])
    negative_factors = explanation.get("negative_factors", [])

    # --- 5. Decision engine -----------------------------------------------
    confidence = _confidence_label(data_coverage)
    expense_ratio = float(feature_values.get("expense_to_income_ratio", 0.5))
    
    app_info = applicant_store.get(app_id, {}).get("applicant", {}) if app_id in applicant_store else {}
    cashflow_info = applicant_store.get(app_id, {}).get("cashflow", {}) if app_id in applicant_store else {}
    loan_amount = float(app_info.get("loan_amount_requested", 50000.0))
    monthly_income = float(cashflow_info.get("average_monthly_inflow", 25000.0) if cashflow_info else 25000.0)
    missed_payments = int(round(float(feature_values.get("recent_missed_flag", 0)) * 2))

    try:
        decision = decision_engine.recommend(
            default_probability=default_prob,
            score=score,
            risk_grade=risk_grade,
            confidence=confidence,
            data_coverage=data_coverage,
            expense_ratio=expense_ratio,
            loan_amount=loan_amount,
            monthly_income=monthly_income,
            missed_payments=missed_payments,
            fairness_alert=False,
        )
    except Exception as exc:
        logger.warning("Decision engine failed for %s: %s", app_id, exc)
        decision = {
            "recommendation": "Review",
            "reason": "Decision engine error – manual review required.",
            "suggested_limit": None,
            "flags": [],
        }

    summary_parts = [f"FairScore: {score} (Grade {risk_grade})."]
    if positive_factors:
        top_pos = positive_factors[0].get("feature", "payment history")
        summary_parts.append(f"Strongest factor: {top_pos}.")
    summary_parts.append(f"Recommendation: {decision.get('recommendation', 'Review')}.")
    summary_text = " ".join(summary_parts)

    improvement_suggestions = _generate_improvement_suggestions(negative_factors)

    result = ScoreResponse(
        applicant_id=app_id,
        score=score,
        default_probability=round(default_prob, 4),
        risk_grade=risk_grade,
        confidence=confidence,
        recommendation=decision.get("recommendation", "Review"),
        recommendation_reason=decision.get("reason", ""),
        suggested_limit=decision.get("suggested_limit"),
        flags=decision.get("flags", []),
        improvement_suggestions=improvement_suggestions,
        positive_factors=positive_factors,
        negative_factors=negative_factors,
        summary_text=summary_text,
    )

    score_results_cache[app_id] = result.model_dump()

    # Audit log — append JSON event to data/audit_log.jsonl
    try:
        audit_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "applicant_id": app_id,
            "score": score,
            "default_probability": round(default_prob, 4),
            "risk_grade": risk_grade,
            "recommendation": decision.get("recommendation", "Review"),
            "data_coverage": round(data_coverage, 4),
            "consent_categories": consent_categories,
            "model_version": "1.0.0",
        }
        _audit_log_path = os.path.join(_PROJECT_ROOT, "data", "audit_log.jsonl")
        os.makedirs(os.path.dirname(_audit_log_path), exist_ok=True)
        with open(_audit_log_path, "a", encoding="utf-8") as _af:
            import json as _json
            _af.write(_json.dumps(audit_entry) + "\n")
    except Exception as _ae:
        logger.warning("Audit log write failed: %s", _ae)

    return result


@app.get("/api/explanation/{applicant_id}", response_model=ExplanationResponse)
async def get_explanation(applicant_id: str) -> ExplanationResponse:
    """Get SHAP-based explanation for a previously scored applicant."""
    if explainer is None:
        raise HTTPException(status_code=503, detail="System not initialised yet")

    # Check cache first
    if applicant_id in explanations_cache:
        expl = explanations_cache[applicant_id]
    elif applicant_id in predictions_cache:
        # Generate on the fly for synthetic applicants
        if _applicant_id_in_synthetic(applicant_id):
            row = features_df.loc[features_df["applicant_id"] == applicant_id].iloc[0]  # type: ignore[union-attr]
            feature_values = row[feature_cols].to_dict()
        elif applicant_id in applicant_store:
            try:
                consent_status = consent_manager.get_consent_status(applicant_id)  # type: ignore[union-attr]
            except Exception:
                consent_status = {}
            feat_res = feature_engineer.engineer_single(applicant_store[applicant_id], consent_status)  # type: ignore[union-attr]
            feature_values = feat_res[feature_cols].iloc[0].to_dict() if isinstance(feat_res, pd.DataFrame) else feat_res
        else:
            raise HTTPException(status_code=404, detail=f"Applicant {applicant_id} not found")

        try:
            expl = explainer.explain_single(feature_values)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Explanation error: {exc}")
        explanations_cache[applicant_id] = expl
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Applicant {applicant_id} has not been scored yet. Call POST /api/score first.",
        )

    return ExplanationResponse(
        applicant_id=applicant_id,
        positive_factors=expl.get("positive_factors", []),
        negative_factors=expl.get("negative_factors", []),
        feature_contributions=expl.get("feature_contributions", {}),
        summary_text=expl.get("summary_text", ""),
        reason_codes=expl.get("reason_codes", []),
    )


# ---- Fairness ------------------------------------------------------------


@app.get("/api/fairness", response_model=FairnessSummaryResponse)
async def get_fairness_summary(
    group_by: Optional[str] = Query(None, description="Demographic column to group by"),
) -> FairnessSummaryResponse:
    """Compute fairness metrics across demographic groups."""
    if fairness_auditor is None or dataset is None or features_df is None:
        raise HTTPException(status_code=503, detail="System not initialised yet")

    applicants_df = dataset["applicants"]

    # Build a predictions dataframe aligned with applicants
    pred_records = []
    for _, row in applicants_df.iterrows():
        app_id = row["applicant_id"]
        pred = predictions_cache.get(app_id)
        if pred is not None:
            pred_records.append(
                {
                    "applicant_id": app_id,
                    "score": pred.get("score", 0),
                    "default_probability": pred.get("default_probability", 0.5),
                }
            )
    if not pred_records:
        raise HTTPException(status_code=500, detail="No predictions available for fairness analysis")

    preds_df = pd.DataFrame(pred_records)
    labels_df = dataset["labels"]
    if "default_label" not in labels_df.columns and "defaulted" in labels_df.columns:
        labels_df["default_label"] = labels_df["defaulted"]

    try:
        summary = fairness_auditor.get_fairness_summary(applicants_df, preds_df, labels_df)
        group_metrics = summary.get("group_metrics", {})
        bias_alerts = summary.get("bias_alerts", [])
        disparate_impact = summary.get("disparate_impact", {})
        summary_stats = summary.get("summary_stats", {})

        feature_impact = {}
        for col in fairness_auditor.DEMOGRAPHIC_COLUMNS:
            if col in applicants_df.columns:
                fi_df = fairness_auditor.get_feature_impact_by_group(features_df, applicants_df, col)
                if not fi_df.empty:
                    feature_impact[col] = fi_df.round(3).to_dict()

        if group_by and group_by in group_metrics:
            group_metrics = {group_by: group_metrics[group_by]}
            bias_alerts = [a for a in bias_alerts if a.get("group_column") == group_by]
            if group_by in disparate_impact:
                disparate_impact = {group_by: disparate_impact[group_by]}
            if group_by in feature_impact:
                feature_impact = {group_by: feature_impact[group_by]}
    except Exception as exc:
        logger.error("Fairness computation failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Fairness analysis error: {exc}")

    return FairnessSummaryResponse(
        group_metrics=group_metrics,
        bias_alerts=bias_alerts,
        disparate_impact=disparate_impact,
        summary_stats=summary_stats,
        feature_impact=feature_impact,
    )


# ---- Dashboard -----------------------------------------------------------


@app.get("/api/dashboard", response_model=DashboardSummaryResponse)
async def get_dashboard_summary() -> DashboardSummaryResponse:
    """Portfolio-level dashboard metrics across all scored applicants."""
    if not predictions_cache:
        raise HTTPException(status_code=503, detail="No predictions available yet")

    scores = [p.get("score", 0) for p in predictions_cache.values()]
    default_probs = [p.get("default_probability", 0.0) for p in predictions_cache.values()]

    total = len(scores)
    avg_score = float(np.mean(scores)) if scores else 0.0
    avg_default = float(np.mean(default_probs)) if default_probs else 0.0

    # Risk distribution
    risk_dist: dict[str, int] = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0}
    approval_by_grade: dict[str, float] = {}
    for grade in ["A", "B", "C", "D", "E"]:
        grade_preds = [p for p in predictions_cache.values() if _risk_grade_from_score(p.get("score", 0)) == grade]
        if grade_preds:
            # Use decision-engine threshold for consistency
            _approve_thresh = decision_engine.thresholds.get("approve_max_default_prob", 0.25) if decision_engine else 0.25
            appr = sum(1 for p in grade_preds if p.get("default_probability", 0.5) <= _approve_thresh) / len(grade_preds)
            approval_by_grade[grade] = round(appr * 100, 1)
        else:
            approval_by_grade[grade] = 0.0

    for s in scores:
        grade = _risk_grade_from_score(s)
        risk_dist[grade] += 1

    # Approval / review / decline counts — use decision-engine thresholds
    _approve_thresh = decision_engine.thresholds.get("approve_max_default_prob", 0.25) if decision_engine else 0.25
    _review_thresh = decision_engine.thresholds.get("review_max_default_prob", 0.50) if decision_engine else 0.50
    approve_count = 0
    review_count = 0
    decline_count = 0
    for p in predictions_cache.values():
        dp = p.get("default_probability", 0.5)
        if dp <= _approve_thresh:
            approve_count += 1
        elif dp <= _review_thresh:
            review_count += 1
        else:
            decline_count += 1

    approval_rate = approve_count / total if total > 0 else 0.0

    # Consent completion rate
    if consent_manager is not None:
        try:
            all_consents = consent_manager.get_all_consents()
            consent_completion = len(all_consents) / total if total > 0 else 0.0
        except Exception:
            consent_completion = 0.0
    else:
        consent_completion = 0.0

    # Bias alert count
    bias_count = 0
    if fairness_auditor is not None:
        try:
            applicants_df = dataset["applicants"]  # type: ignore[index]
            pred_records = []
            for _, row in applicants_df.iterrows():
                app_id = row["applicant_id"]
                pred = predictions_cache.get(app_id)
                if pred:
                    pred_records.append(
                        {
                            "applicant_id": app_id,
                            "score": pred.get("score", 0),
                            "default_probability": pred.get("default_probability", 0.5),
                        }
                    )
            if pred_records:
                preds_df = pd.DataFrame(pred_records)
                labels_df = dataset["labels"]
                if "default_label" not in labels_df.columns and "defaulted" in labels_df.columns:
                    labels_df["default_label"] = labels_df["defaulted"]
                gm = fairness_auditor.compute_all_group_metrics(applicants_df, preds_df, labels_df)
                alerts = fairness_auditor.generate_bias_alerts(gm)
                bias_count = len(alerts) if isinstance(alerts, list) else len(alerts.get("alerts", []))
        except Exception:
            bias_count = 0

    return DashboardSummaryResponse(
        total_applicants=total,
        approval_rate=round(approval_rate, 4),
        average_score=round(avg_score, 1),
        risk_distribution=risk_dist,
        approval_by_risk_grade=approval_by_grade,
        score_distribution=scores,
        predicted_default_rate=round(avg_default, 4),
        manual_review_count=review_count,
        consent_completion_rate=round(consent_completion, 4),
        bias_alert_count=bias_count,
    )


# ---- Model Metrics -------------------------------------------------------


@app.get("/api/model/metrics", response_model=ModelMetricsResponse)
async def get_model_metrics() -> ModelMetricsResponse:
    """Return model evaluation metrics (computed on training data)."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not trained yet")

    try:
        report = model.get_evaluation_report()
    except Exception as exc:
        logger.error("Model metrics error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Metrics error: {exc}")

    scores = [p.get("score", 0) for p in predictions_cache.values()]
    default_probs = [p.get("default_probability", 0.0) for p in predictions_cache.values()]
    approval_rate = sum(1 for dp in default_probs if dp <= 0.3) / len(default_probs) if default_probs else 0.0
    predicted_default_rate = float(np.mean(default_probs)) if default_probs else 0.0

    return ModelMetricsResponse(
        auc_roc=round(report.get("auc_roc", report.get("roc_auc", 0.0)), 4),
        precision=round(report.get("precision", 0.0), 4),
        recall=round(report.get("recall", 0.0), 4),
        f1_score=round(report.get("f1_score", report.get("f1", 0.0)), 4),
        confusion_matrix=report.get("confusion_matrix", []),
        approval_rate=round(approval_rate, 4),
        predicted_default_rate=round(predicted_default_rate, 4),
    )


# ---- Thresholds ----------------------------------------------------------


@app.put("/api/thresholds")
async def update_thresholds(thresholds: ThresholdUpdate) -> dict:
    """Update the decision engine thresholds at runtime."""
    if decision_engine is None:
        raise HTTPException(status_code=503, detail="Decision engine not initialised yet")

    updated: dict[str, float] = {}

    if thresholds.approve_max_default_prob is not None:
        decision_engine.thresholds["approve_max_default_prob"] = thresholds.approve_max_default_prob
        updated["approve_max_default_prob"] = thresholds.approve_max_default_prob

    if thresholds.review_max_default_prob is not None:
        decision_engine.thresholds["review_max_default_prob"] = thresholds.review_max_default_prob
        updated["review_max_default_prob"] = thresholds.review_max_default_prob

    if thresholds.min_data_coverage is not None:
        decision_engine.thresholds["min_data_coverage"] = thresholds.min_data_coverage
        updated["min_data_coverage"] = thresholds.min_data_coverage

    if not updated:
        return {"message": "No thresholds updated (no values provided)."}

    return {"message": "Thresholds updated successfully.", "thresholds": updated, "updated": updated}


# ---- Personas ------------------------------------------------------------


@app.get("/api/personas")
async def list_personas() -> list:
    """List available synthetic personas for quick applicant creation."""
    for p in PERSONAS:
        if "name" not in p:
            p["name"] = p["label"]
    return PERSONAS


# =========================================================================
#  Health check
# =========================================================================


@app.get("/api/health")
async def health_check() -> dict:
    """Liveness probe."""
    return {
        "status": "ok",
        "version": "1.0.0",
        "model_loaded": model is not None,
        "dataset_size": len(predictions_cache),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
