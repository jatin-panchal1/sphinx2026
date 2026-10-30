"""
FairScore Synthetic Data Generator
===================================

Generates realistic synthetic datasets for alternative credit scoring.
Includes built-in bias scenarios and persona-specific patterns to test
fairness-aware ML models.

Produces ~2000 applicant profiles with:
- Applicant demographics & loan details
- Consent records (full / partial / minimal)
- Rent, utility, mobile, cash-flow, and business payment records
- Deterministic default labels based on financial signals
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AGE_GROUPS = ["18-25", "26-35", "36-45", "46-55", "56+"]
GENDER_CATEGORIES = ["Male", "Female", "Non-Binary"]
REGION_CATEGORIES = ["North", "South", "East", "West", "Central"]
URBAN_RURAL = ["Urban", "Semi-Urban", "Rural"]
OCCUPATION_TYPES = [
    "Gig Worker",
    "Small Shopkeeper",
    "Student",
    "Informal Worker",
    "Freelancer",
    "Daily Wage Worker",
]
INCOME_TYPES = ["Weekly", "Monthly", "Daily", "Irregular"]
LOAN_TENURES = [3, 6, 9, 12, 18, 24]
CONSENT_LEVELS = ["full", "partial", "minimal"]
UTILITY_TYPES = ["Electricity", "Water", "Gas", "Internet"]
RECHARGE_FREQUENCIES = ["weekly", "biweekly", "monthly"]
INCOME_FREQUENCIES = ["daily", "weekly", "biweekly", "monthly"]
SALES_TRENDS = ["growing", "stable", "declining"]

CONSENT_FIELDS = [
    "rent_data_consent",
    "utility_data_consent",
    "mobile_data_consent",
    "cashflow_data_consent",
    "business_data_consent",
    "self_declared_income_consent",
]

PERSONA_NAMES = [
    "reliable_gig_worker",
    "high_risk_gig_worker",
    "stable_shopkeeper",
    "seasonal_shopkeeper",
    "student",
    "informal_worker",
    "thin_file_low_risk",
    "high_risk",
]


# ---------------------------------------------------------------------------
# Persona defaults
# ---------------------------------------------------------------------------

def get_persona_defaults() -> dict:
    """Return parameter ranges for each persona type.

    Each persona maps to a dict of field names → (low, high) tuples or
    fixed values that shape the synthetic profile.
    """
    return {
        "reliable_gig_worker": {
            "occupation_type": "Gig Worker",
            "income_type": "Weekly",
            "on_time_ratio": (0.85, 0.98),
            "income_volatility": (0.05, 0.20),
            "expense_to_income_ratio": (0.35, 0.60),
            "zero_income_weeks": (0, 1),
            "savings_deposit_frequency": (2, 4),
            "rent_payment_stability": (0.80, 0.98),
            "bill_stability": (0.80, 0.98),
            "default_bias": -0.25,
        },
        "high_risk_gig_worker": {
            "occupation_type": "Gig Worker",
            "income_type": "Irregular",
            "on_time_ratio": (0.30, 0.60),
            "income_volatility": (0.50, 0.90),
            "expense_to_income_ratio": (0.80, 1.15),
            "zero_income_weeks": (4, 10),
            "savings_deposit_frequency": (0, 1),
            "rent_payment_stability": (0.20, 0.50),
            "bill_stability": (0.20, 0.50),
            "default_bias": 0.25,
        },
        "stable_shopkeeper": {
            "occupation_type": "Small Shopkeeper",
            "income_type": "Daily",
            "on_time_ratio": (0.85, 0.98),
            "income_volatility": (0.05, 0.15),
            "expense_to_income_ratio": (0.40, 0.65),
            "zero_income_weeks": (0, 1),
            "savings_deposit_frequency": (2, 4),
            "rent_payment_stability": (0.85, 0.98),
            "bill_stability": (0.85, 0.98),
            "sales_trend": "growing",
            "seasonality_index": (0.05, 0.25),
            "default_bias": -0.30,
        },
        "seasonal_shopkeeper": {
            "occupation_type": "Small Shopkeeper",
            "income_type": "Monthly",
            "on_time_ratio": (0.50, 0.70),
            "income_volatility": (0.40, 0.70),
            "expense_to_income_ratio": (0.65, 0.95),
            "zero_income_weeks": (3, 8),
            "savings_deposit_frequency": (0, 2),
            "rent_payment_stability": (0.40, 0.65),
            "bill_stability": (0.40, 0.65),
            "sales_trend": "declining",
            "seasonality_index": (0.60, 0.95),
            "default_bias": 0.15,
        },
        "student": {
            "occupation_type": "Student",
            "income_type": "Monthly",
            "on_time_ratio": (0.80, 0.95),
            "income_volatility": (0.10, 0.30),
            "expense_to_income_ratio": (0.50, 0.75),
            "zero_income_weeks": (0, 3),
            "savings_deposit_frequency": (1, 3),
            "rent_payment_stability": (0.75, 0.95),
            "bill_stability": (0.75, 0.95),
            "default_bias": -0.15,
        },
        "informal_worker": {
            "occupation_type": "Informal Worker",
            "income_type": "Daily",
            "on_time_ratio": (0.55, 0.80),
            "income_volatility": (0.30, 0.60),
            "expense_to_income_ratio": (0.55, 0.85),
            "zero_income_weeks": (1, 6),
            "savings_deposit_frequency": (0, 2),
            "rent_payment_stability": (0.50, 0.75),
            "bill_stability": (0.50, 0.75),
            "default_bias": 0.05,
        },
        "thin_file_low_risk": {
            "occupation_type": "Freelancer",
            "income_type": "Monthly",
            "consent_status": "minimal",
            "on_time_ratio": (0.80, 0.95),
            "income_volatility": (0.10, 0.25),
            "expense_to_income_ratio": (0.40, 0.65),
            "zero_income_weeks": (0, 2),
            "savings_deposit_frequency": (1, 3),
            "rent_payment_stability": (0.75, 0.92),
            "bill_stability": (0.75, 0.92),
            "default_bias": -0.10,
        },
        "high_risk": {
            "occupation_type": "Daily Wage Worker",
            "income_type": "Irregular",
            "on_time_ratio": (0.15, 0.45),
            "income_volatility": (0.60, 0.95),
            "expense_to_income_ratio": (0.85, 1.20),
            "zero_income_weeks": (5, 12),
            "savings_deposit_frequency": (0, 0),
            "rent_payment_stability": (0.10, 0.35),
            "bill_stability": (0.10, 0.35),
            "default_bias": 0.35,
        },
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_id(index: int) -> str:
    """Format an applicant ID as ``APP-XXXX``."""
    return f"APP-{index:04d}"


def _uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(rng.uniform(low, high))


def _randint(rng: np.random.Generator, low: int, high: int) -> int:
    return int(rng.integers(low, high + 1))


def _choice(rng: np.random.Generator, options: list):
    return options[int(rng.integers(0, len(options)))]


def _clamp(val: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, val))


# ---------------------------------------------------------------------------
# Consent generation
# ---------------------------------------------------------------------------

def _generate_consent(
    rng: np.random.Generator,
    consent_status: str,
    is_shopkeeper: bool,
) -> dict:
    """Generate consent flags honouring the consent_status level."""
    fields = list(CONSENT_FIELDS)
    if consent_status == "full":
        consents = {f: True for f in fields}
    elif consent_status == "partial":
        n_true = _randint(rng, 3, 5)
        selected = list(rng.choice(len(fields), size=n_true, replace=False))
        consents = {f: (i in selected) for i, f in enumerate(fields)}
    else:  # minimal
        n_true = _randint(rng, 1, 2)
        selected = list(rng.choice(len(fields), size=n_true, replace=False))
        consents = {f: (i in selected) for i, f in enumerate(fields)}

    # Business consent only relevant for shopkeepers
    if not is_shopkeeper:
        consents["business_data_consent"] = False

    return consents


# ---------------------------------------------------------------------------
# Record generators
# ---------------------------------------------------------------------------

def _generate_rent(
    rng: np.random.Generator,
    on_time_ratio: tuple[float, float],
    stability_range: tuple[float, float],
    urban_rural: str,
) -> Optional[dict]:
    """Generate a rent payment record.  Returns ``None`` if data is missing."""
    # Rural applicants have ~15% chance of missing rent data
    if urban_rural == "Rural" and rng.random() < 0.15:
        return None

    months_observed = _randint(rng, 6, 36)
    ratio = _uniform(rng, *on_time_ratio)
    on_time = int(round(months_observed * ratio))
    remaining = months_observed - on_time
    late = _randint(rng, 0, remaining)
    missed = remaining - late

    return {
        "monthly_rent_amount": _randint(rng, 2000, 25000),
        "months_observed": months_observed,
        "on_time_payments": on_time,
        "late_payments": late,
        "missed_payments": missed,
        "average_days_late": round(_uniform(rng, 0, 30) * (1 - ratio), 1),
        "rent_payment_stability": round(_uniform(rng, *stability_range), 3),
    }


def _generate_utility(
    rng: np.random.Generator,
    on_time_ratio: tuple[float, float],
    stability_range: tuple[float, float],
    urban_rural: str,
) -> Optional[dict]:
    """Generate a utility payment record.  Returns ``None`` if data is missing."""
    # Built-in bias: rural 25% missing vs urban 5%
    missing_prob = 0.25 if urban_rural == "Rural" else 0.05
    if rng.random() < missing_prob:
        return None

    months_observed = _randint(rng, 6, 36)
    ratio = _uniform(rng, *on_time_ratio)
    on_time = int(round(months_observed * ratio))
    late = months_observed - on_time

    return {
        "utility_type": _choice(rng, UTILITY_TYPES),
        "monthly_bill_amount": _randint(rng, 200, 5000),
        "months_observed": months_observed,
        "on_time_payments": on_time,
        "late_payments": late,
        "disconnection_events": _randint(rng, 0, 3) if ratio < 0.6 else 0,
        "bill_stability": round(_uniform(rng, *stability_range), 3),
    }


def _generate_mobile(
    rng: np.random.Generator,
    on_time_ratio: tuple[float, float],
) -> dict:
    """Generate mobile recharge records."""
    ratio = _uniform(rng, *on_time_ratio)
    tenure = _randint(rng, 6, 60)
    missed = _randint(rng, 0, min(6, int(tenure * (1 - ratio) * 0.3)))
    failed = _randint(rng, 0, min(10, int(tenure * (1 - ratio) * 0.5)))
    return {
        "recharge_frequency": _choice(rng, RECHARGE_FREQUENCIES),
        "average_recharge_amount": _randint(rng, 50, 1000),
        "missed_recharge_months": missed,
        "failed_payment_count": failed,
        "mobile_tenure_months": tenure,
        "prepaid_or_postpaid": _choice(rng, ["Prepaid", "Postpaid"]),
    }


def _generate_cashflow(
    rng: np.random.Generator,
    volatility_range: tuple[float, float],
    expense_ratio_range: tuple[float, float],
    zero_weeks_range: tuple[int, int],
    savings_range: tuple[int, int],
    occupation_type: str,
) -> dict:
    """Generate cash-flow records."""
    # Cash-heavy users get lower inflow numbers (less digital visibility)
    if occupation_type in ("Daily Wage Worker", "Informal Worker"):
        avg_inflow = _randint(rng, 3000, 30000)
    elif occupation_type == "Student":
        avg_inflow = _randint(rng, 3000, 15000)
    else:
        avg_inflow = _randint(rng, 8000, 100000)

    freq = _choice(rng, INCOME_FREQUENCIES)
    volatility = round(_uniform(rng, *volatility_range), 3)
    zero_weeks = _randint(rng, *zero_weeks_range)
    savings_freq = round(_uniform(rng, float(savings_range[0]), float(savings_range[1])), 2)
    expense_ratio = round(_uniform(rng, *expense_ratio_range), 3)

    return {
        "average_monthly_inflow": avg_inflow,
        "income_frequency": freq,
        "income_volatility": volatility,
        "zero_income_weeks": zero_weeks,
        "expense_to_income_ratio": expense_ratio,
        "positive_balance_days": _randint(rng, max(0, int(30 * (1 - expense_ratio))), 30),
        "savings_deposit_frequency": savings_freq,
    }


def _generate_business(
    rng: np.random.Generator,
    persona_defaults: dict,
) -> dict:
    """Generate business records for shopkeepers."""
    trend = persona_defaults.get("sales_trend", _choice(rng, SALES_TRENDS))
    seas_range = persona_defaults.get("seasonality_index", (0.1, 0.5))
    stability = persona_defaults.get("on_time_ratio", (0.6, 0.9))

    avg_daily = _randint(rng, 500, 20000)
    active_days = _randint(rng, 3, 7)
    seasonality = round(_uniform(rng, *seas_range), 3)
    repeat = round(_uniform(rng, 0.1, 0.9), 3)
    consistency = round(_uniform(rng, *stability), 3)

    return {
        "average_daily_sales": avg_daily,
        "active_transaction_days_per_week": active_days,
        "sales_trend": trend,
        "seasonality_index": seasonality,
        "repeat_customer_ratio": repeat,
        "collection_consistency": consistency,
    }


# ---------------------------------------------------------------------------
# Default-label formula
# ---------------------------------------------------------------------------

def _compute_default_probability(
    rent: Optional[dict],
    utility: Optional[dict],
    cashflow: Optional[dict],
    business: Optional[dict],
    mobile: Optional[dict],
    persona_bias: float,
    gender: str,
) -> float:
    """Deterministic default probability from financial signals.

    Factors that *increase* risk:
        - missed_payments, late_payments
        - income_volatility
        - expense_to_income_ratio
        - zero_income_weeks
        - low savings_deposit_frequency
        - low rent_payment_stability
        - low bill_stability

    Built-in bias: Women get a slight discipline bonus (-0.03).
    """
    score = 0.0
    n_signals = 0

    # --- Rent signals ---
    if rent is not None:
        months = rent["months_observed"]
        missed_ratio = rent["missed_payments"] / max(months, 1)
        late_ratio = rent["late_payments"] / max(months, 1)
        score += missed_ratio * 0.8 + late_ratio * 0.3
        score += (1 - rent["rent_payment_stability"]) * 0.5
        n_signals += 3
    else:
        # Missing rent data adds small uncertainty penalty
        score += 0.15
        n_signals += 1

    # --- Utility signals ---
    if utility is not None:
        score += (1 - utility["bill_stability"]) * 0.4
        score += utility["disconnection_events"] * 0.12
        n_signals += 2
    else:
        score += 0.12
        n_signals += 1

    # --- Cash-flow signals ---
    if cashflow is not None:
        score += cashflow["income_volatility"] * 0.6
        score += max(0, cashflow["expense_to_income_ratio"] - 0.5) * 0.8
        score += (cashflow["zero_income_weeks"] / 12) * 0.5
        score += (1 - min(cashflow["savings_deposit_frequency"] / 4, 1.0)) * 0.4
        n_signals += 4
    else:
        score += 0.20
        n_signals += 1

    # --- Mobile signals ---
    if mobile is not None:
        score += (mobile["missed_recharge_months"] / 6) * 0.2
        score += (mobile["failed_payment_count"] / 10) * 0.15
        n_signals += 2

    # --- Business signals ---
    if business is not None:
        trend_penalty = {"growing": -0.10, "stable": 0.0, "declining": 0.15}
        score += trend_penalty.get(business["sales_trend"], 0)
        score += business["seasonality_index"] * 0.15
        score += (1 - business["collection_consistency"]) * 0.2
        n_signals += 3

    # Normalise by number of signals so the probability stays in [0, 1]
    if n_signals > 0:
        prob = score / n_signals

    # Apply persona bias
    prob += persona_bias

    # Gender bias: women slightly better payment discipline
    if gender == "Female":
        prob -= 0.03

    return _clamp(prob, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Single-applicant generator
# ---------------------------------------------------------------------------

def generate_single_applicant(
    persona: str = "random",
    rng: Optional[np.random.Generator] = None,
    applicant_index: int = 0,
) -> dict:
    """Generate a single applicant with all records.

    Parameters
    ----------
    persona : str
        One of the predefined persona names, or ``"random"`` to pick one
        at random.
    rng : numpy.random.Generator, optional
        Random generator instance.  Created internally if not supplied.
    applicant_index : int
        Numeric index used for ``APP-XXXX`` ID formatting.

    Returns
    -------
    dict
        Keys: ``applicant``, ``consent``, ``rent``, ``utility``, ``mobile``,
        ``cashflow``, ``business``, ``label``.  Values that are ``None``
        indicate missing / not-applicable records.
    """
    if rng is None:
        rng = np.random.default_rng()

    personas = get_persona_defaults()
    if isinstance(persona, dict):
        p = persona
        persona = "custom"
    else:
        if persona == "random":
            persona = _choice(rng, PERSONA_NAMES)
        if persona not in personas:
            persona = "reliable_gig_worker"
        p = personas[persona]

    app_id = _make_id(applicant_index)
    occupation = p.get("occupation_type", _choice(rng, OCCUPATION_TYPES))
    income_type = p.get("income_type", _choice(rng, INCOME_TYPES))
    is_shopkeeper = occupation == "Small Shopkeeper"
    gender = _choice(rng, GENDER_CATEGORIES)
    urban_rural_val = _choice(rng, URBAN_RURAL)
    consent_status = p.get("consent_status", _choice(rng, CONSENT_LEVELS))

    applicant = {
        "applicant_id": app_id,
        "age_group": _choice(rng, AGE_GROUPS),
        "gender_category": gender,
        "region_category": _choice(rng, REGION_CATEGORIES),
        "urban_rural": urban_rural_val,
        "occupation_type": occupation,
        "income_type": income_type,
        "loan_amount_requested": _randint(rng, 5000, 200000),
        "loan_tenure_months": _choice(rng, LOAN_TENURES),
        "consent_status": consent_status,
    }

    consent = _generate_consent(rng, consent_status, is_shopkeeper)
    consent["applicant_id"] = app_id

    on_time_ratio = p.get("on_time_ratio", (0.5, 0.9))
    stability = p.get("rent_payment_stability", (0.4, 0.9))
    bill_stab = p.get("bill_stability", (0.4, 0.9))

    # Rent
    rent = None
    if consent.get("rent_data_consent", False):
        rent = _generate_rent(rng, on_time_ratio, stability, urban_rural_val)
        if rent is not None:
            rent["applicant_id"] = app_id

    # Utility
    utility = None
    if consent.get("utility_data_consent", False):
        utility = _generate_utility(rng, on_time_ratio, bill_stab, urban_rural_val)
        if utility is not None:
            utility["applicant_id"] = app_id

    # Mobile
    mobile = None
    if consent.get("mobile_data_consent", False):
        mobile = _generate_mobile(rng, on_time_ratio)
        mobile["applicant_id"] = app_id

    # Cash-flow
    cashflow = None
    if consent.get("cashflow_data_consent", False):
        cashflow = _generate_cashflow(
            rng,
            volatility_range=p.get("income_volatility", (0.1, 0.5)),
            expense_ratio_range=p.get("expense_to_income_ratio", (0.4, 0.9)),
            zero_weeks_range=p.get("zero_income_weeks", (0, 6)),
            savings_range=p.get("savings_deposit_frequency", (0, 3)),
            occupation_type=occupation,
        )
        cashflow["applicant_id"] = app_id

    # Business
    business = None
    if is_shopkeeper and consent.get("business_data_consent", False):
        business = _generate_business(rng, p)
        business["applicant_id"] = app_id

    # Default label
    default_prob = _compute_default_probability(
        rent=rent,
        utility=utility,
        cashflow=cashflow,
        business=business,
        mobile=mobile,
        persona_bias=p.get("default_bias", 0.0),
        gender=gender,
    )
    # Deterministic threshold with small per-applicant noise for realism
    noise = rng.normal(0, 0.04)
    defaulted = int((default_prob + noise) > 0.35)

    label = {
        "applicant_id": app_id,
        "default_probability": round(default_prob, 4),
        "defaulted": defaulted,
        "default_label": defaulted,
    }

    return {
        "applicant": applicant,
        "consent": consent,
        "rent": rent,
        "utility": utility,
        "mobile": mobile,
        "cashflow": cashflow,
        "business": business,
        "label": label,
    }


# ---------------------------------------------------------------------------
# Full dataset generator
# ---------------------------------------------------------------------------

def generate_full_dataset(
    n_applicants: int = 2000,
    random_seed: int = 42,
) -> dict:
    """Generate the complete synthetic dataset.

    Parameters
    ----------
    n_applicants : int
        Number of applicant profiles to generate.
    random_seed : int
        Seed for reproducibility.

    Returns
    -------
    dict
        Keys ``'applicants'``, ``'consents'``, ``'rent'``, ``'utility'``,
        ``'mobile'``, ``'cashflow'``, ``'business'``, ``'labels'`` — each a
        :class:`pandas.DataFrame`.
    """
    rng = np.random.default_rng(random_seed)

    # Distribute personas across applicants for realistic variety
    # Roughly: 20% reliable_gig, 10% high_risk_gig, 15% stable_shop,
    #          8% seasonal_shop, 12% student, 15% informal, 10% thin_file, 10% high_risk
    persona_weights = np.array([0.20, 0.10, 0.15, 0.08, 0.12, 0.15, 0.10, 0.10])
    persona_weights = persona_weights / persona_weights.sum()
    persona_choices = rng.choice(
        PERSONA_NAMES, size=n_applicants, p=persona_weights
    )

    applicants_list: list[dict] = []
    consents_list: list[dict] = []
    rent_list: list[dict] = []
    utility_list: list[dict] = []
    mobile_list: list[dict] = []
    cashflow_list: list[dict] = []
    business_list: list[dict] = []
    labels_list: list[dict] = []

    for i in range(n_applicants):
        result = generate_single_applicant(
            persona=persona_choices[i],
            rng=rng,
            applicant_index=i + 1,
        )
        applicants_list.append(result["applicant"])
        consents_list.append(result["consent"])
        if result["rent"] is not None:
            rent_list.append(result["rent"])
        if result["utility"] is not None:
            utility_list.append(result["utility"])
        if result["mobile"] is not None:
            mobile_list.append(result["mobile"])
        if result["cashflow"] is not None:
            cashflow_list.append(result["cashflow"])
        if result["business"] is not None:
            business_list.append(result["business"])
        labels_list.append(result["label"])

    dataset = {
        "applicants": pd.DataFrame(applicants_list),
        "consents": pd.DataFrame(consents_list),
        "rent": pd.DataFrame(rent_list) if rent_list else pd.DataFrame(),
        "utility": pd.DataFrame(utility_list) if utility_list else pd.DataFrame(),
        "mobile": pd.DataFrame(mobile_list) if mobile_list else pd.DataFrame(),
        "cashflow": pd.DataFrame(cashflow_list) if cashflow_list else pd.DataFrame(),
        "business": pd.DataFrame(business_list) if business_list else pd.DataFrame(),
        "labels": pd.DataFrame(labels_list),
    }

    return dataset


# ---------------------------------------------------------------------------
# Quick sanity check when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating FairScore synthetic dataset …")
    data = generate_full_dataset(n_applicants=2000, random_seed=42)

    for name, df in data.items():
        print(f"  {name:12s} → {len(df):>5} rows  ×  {len(df.columns)} cols")

    labels = data["labels"]
    default_rate = labels["defaulted"].mean()
    print(f"\n  Overall default rate: {default_rate:.2%}")

    # Bias check: rural vs urban utility coverage
    merged = data["applicants"].merge(
        data["utility"][["applicant_id"]].drop_duplicates().assign(has_utility=True),
        on="applicant_id",
        how="left",
    )
    merged["has_utility"] = merged["has_utility"].fillna(False)
    coverage = merged.groupby("urban_rural")["has_utility"].mean()
    print("\n  Utility data coverage by area:")
    for area, cov in coverage.items():
        print(f"    {area:12s}: {cov:.2%}")
