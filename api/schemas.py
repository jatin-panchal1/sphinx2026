"""
FairScore API - Pydantic v2 Request/Response Schemas.

Defines all data models used for API input validation and output serialization,
including authentication, consent, scoring, fairness, and governance.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


# =============================================================================
# Auth Models
# =============================================================================

class AuthRegisterRequest(BaseModel):
    """User registration request."""
    username: str = Field(..., min_length=3, max_length=50, description="Username")
    password: str = Field(..., min_length=6, description="User password")
    role: str = Field(default="applicant", description="Role: 'applicant' or 'lender'")
    applicant_id: Optional[str] = Field(default=None, description="Optional applicant ID mapping")


class AuthLoginRequest(BaseModel):
    """User login request."""
    username: str = Field(..., description="Username")
    password: str = Field(..., description="User password")


class AuthTokenResponse(BaseModel):
    """Authentication token response."""
    access_token: str
    token_type: str = "Bearer"
    user_id: str
    username: str
    role: str
    applicant_id: Optional[str] = None
    message: str


# =============================================================================
# Request Models
# =============================================================================


class ApplicantCreate(BaseModel):
    """Request body for creating a new applicant profile."""

    age_group: str = Field(
        ...,
        description="Age group: 18-25, 26-35, 36-45, 46-55, 56+",
        examples=["26-35"],
    )
    gender_category: str = Field(
        ...,
        description="Gender: Male, Female, Non-Binary",
        examples=["Female"],
    )
    region_category: str = Field(
        ...,
        description="Region: North, South, East, West, Central",
        examples=["South"],
    )
    urban_rural: str = Field(
        ...,
        description="Urban, Semi-Urban, or Rural",
        examples=["Urban"],
    )
    occupation_type: str = Field(
        ...,
        description="Gig Worker, Small Shopkeeper, Student, Homemaker, Freelancer, Daily Wage Worker, Street Vendor, Artisan",
        examples=["Gig Worker"],
    )
    income_type: str = Field(
        ...,
        description="Income frequency: Weekly, Monthly, Daily, Irregular",
        examples=["Monthly"],
    )
    loan_amount_requested: float = Field(
        ...,
        ge=5000,
        le=500000,
        description="Loan amount requested in INR (5,000 - 5,00,000)",
        examples=[50000.0],
    )
    loan_tenure_months: int = Field(
        ...,
        description="Loan tenure in months: 3, 6, 9, 12, 18, or 24",
        examples=[12],
    )


class ConsentSubmit(BaseModel):
    """Request body for submitting data-sharing consent."""

    applicant_id: str = Field(..., description="Unique applicant identifier")
    rent_data: bool = Field(default=False, description="Consent to use rent payment history")
    utility_data: bool = Field(default=False, description="Consent to use utility bill payment data")
    mobile_data: bool = Field(default=False, description="Consent to use mobile usage patterns")
    cashflow_data: bool = Field(default=False, description="Consent to use bank/wallet cashflow data")
    business_data: bool = Field(default=False, description="Consent to use business revenue data")
    self_declared_income: bool = Field(default=False, description="Consent to use self-declared income data")


class ConsentWithdrawRequest(BaseModel):
    """Request body for revoking consent."""

    applicant_id: str
    category: Optional[str] = Field(default=None, description="Category to revoke, or None for all")


class ScoreRequest(BaseModel):
    """Request body for generating a credit score."""

    applicant_id: str = Field(..., description="Unique applicant identifier")


class ExplanationRequest(BaseModel):
    """Request body for fetching a score explanation."""

    applicant_id: str = Field(..., description="Unique applicant identifier")


class FairnessSummaryRequest(BaseModel):
    """Request body for fairness summary (used as query param in practice)."""

    group_by: Optional[str] = Field(
        default=None,
        description="Demographic column to group by (e.g., gender_category, region_category, age_group)",
    )


class ThresholdUpdate(BaseModel):
    """Request body for updating decision engine thresholds."""

    approve_max_default_prob: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Maximum default probability for auto-approval",
    )
    review_max_default_prob: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Maximum default probability for manual review (above this = decline)",
    )
    min_data_coverage: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum fraction of data categories required for scoring",
    )


# =============================================================================
# Response Models
# =============================================================================


class ApplicantResponse(BaseModel):
    """Response after creating a new applicant."""

    applicant_id: str
    message: str
    auth_token: Optional[str] = None


class ConsentResponse(BaseModel):
    """Response after submitting consent."""

    applicant_id: str
    consented_categories: list[str]
    consent_timestamp: str
    sufficient_data: bool
    message: str


class ScoreResponse(BaseModel):
    """Full scoring response including score, explanation, and recommendation."""

    applicant_id: str
    score: int = Field(..., ge=300, le=900, description="Alternative credit score (300-900)")
    default_probability: float = Field(..., ge=0.0, le=1.0)
    risk_grade: str = Field(..., description="A, B, C, D, or E")
    confidence: str = Field(..., description="High, Medium, or Low")
    recommendation: str = Field(..., description="Approve, Review, or Decline")
    recommendation_reason: str
    suggested_limit: Optional[float] = None
    flags: list[str] = []
    improvement_suggestions: list[str] = []
    positive_factors: list[dict]
    negative_factors: list[dict]
    summary_text: str
    demonstration_notice: str = "PROTOTYPE DEMONSTRATION ONLY: Score generated from synthetic behavioral model. Not a commitment or guarantee to lend."


class ExplanationResponse(BaseModel):
    """SHAP-based explanation for a scored applicant."""

    applicant_id: str
    positive_factors: list[dict]
    negative_factors: list[dict]
    feature_contributions: dict
    summary_text: str
    reason_codes: list[str]


class FairnessSummaryResponse(BaseModel):
    """Fairness audit results across demographic groups."""

    group_metrics: dict
    bias_alerts: list[dict]
    disparate_impact: dict
    summary_stats: dict
    feature_impact: Optional[dict] = None


class DashboardSummaryResponse(BaseModel):
    """Portfolio-level dashboard summary metrics."""

    total_applicants: int
    approval_rate: float
    average_score: float
    risk_distribution: dict
    approval_by_risk_grade: Optional[dict] = None
    score_distribution: Optional[list] = None
    predicted_default_rate: float
    manual_review_count: int
    consent_completion_rate: float
    bias_alert_count: int


class ModelMetricsResponse(BaseModel):
    """Model evaluation metrics."""

    auc_roc: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: list
    approval_rate: float
    predicted_default_rate: float
    model_validation_status: str = "SYNTHETIC PROTOTYPE: Metrics reflect synthetic generation distribution and do not constitute real-world credit risk predictive validity."


class ApplicantDetailResponse(BaseModel):
    """Full detail for a single applicant."""

    profile: dict
    consent_status: dict
    score_result: Optional[dict] = None
    explanation: Optional[dict] = None
