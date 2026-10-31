"""
Pydantic schemas for the FairScore standalone backend.
"""

from typing import Any, Optional
from pydantic import BaseModel, Field


class ApplicantInput(BaseModel):
    """Input features for a credit-scoring prediction request."""

    monthly_income: Optional[float] = Field(None, ge=0, description="Monthly income in INR")
    income_volatility: Optional[float] = Field(None, ge=0, le=1)
    rent_payment_score: Optional[float] = Field(None, ge=0, le=1)
    utility_payment_score: Optional[float] = Field(None, ge=0, le=1)
    mobile_recharge_regularity: Optional[float] = Field(None, ge=0, le=1)
    cashflow_consistency: Optional[float] = Field(None, ge=0, le=1)
    savings_rate: Optional[float] = Field(None, ge=0, le=1)
    verified_cash_payments: Optional[int] = Field(None, ge=0)
    months_in_current_job: Optional[int] = Field(None, ge=0)
    num_dependants: Optional[int] = Field(None, ge=0)
    expense_to_income_ratio: Optional[float] = Field(None, ge=0, le=1)
    prior_defaults: Optional[int] = Field(None, ge=0)
    loan_amount_requested: Optional[float] = Field(None, ge=0)
    loan_to_income_ratio: Optional[float] = Field(None, ge=0)

    model_config = {"extra": "allow"}


class PredictionResponse(BaseModel):
    default_probability: float
    credit_score: int
    risk_grade: str
    recommendation: str
    top_positive_factors: list[dict[str, Any]]
    top_negative_factors: list[dict[str, Any]]
    disclaimer: str = (
        "PROTOTYPE — Trained on synthetic data. Not a credit decision."
    )


class ExplainResponse(BaseModel):
    positive_factors: list[dict[str, Any]]
    negative_factors: list[dict[str, Any]]
    feature_importance: list[dict[str, Any]]
    disclaimer: str = (
        "PROTOTYPE — Trained on synthetic data. Not a credit decision."
    )


class MetricsResponse(BaseModel):
    models: list[dict[str, Any]]
    dataset_metrics: dict[str, Any]
    fairness_metrics: dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str


class ModelSummary(BaseModel):
    name: str
    path: str
    roc_auc: float
    f1: float
    accuracy: float
    n_rows: int
    model_type: str
    timestamp: str

