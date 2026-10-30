"""
FairScore Feature Engineering Pipeline
=======================================

Transforms raw alternative-data DataFrames (from ``synthetic_generator``)
into a model-ready feature matrix.

Feature groups
--------------
1. **Payment Reliability** – rent & utility on-time ratios, missed-payment
   ratio, normalised days-late.
2. **Cash-flow Stability** – income stability, zero-income ratio,
   positive-balance ratio, savings regularity.
3. **Expense Burden** – expense-to-income, rent-to-income, bill-to-income
   ratios.
4. **Business Continuity** – consistency, trend, repeat-customer score,
   seasonality risk (shopkeepers only, 0 otherwise).
5. **Income Regularity** – frequency score, mobile tenure, recharge
   regularity.
6. **Risk Signals** – binary flags for disconnection, high expense, zero
   savings, recent missed payments.
7. **Data Completeness** – coverage score, total data-point count.

Missing-data strategy
---------------------
If consent is denied or data is absent, features derived from that source
are set to a neutral/median placeholder and the shortfall is reflected in
``data_coverage_score``.  The pipeline **never crashes** on missing data.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants & mappings
# ---------------------------------------------------------------------------

_INCOME_FREQ_SCORES = {
    "daily": 1.0,
    "weekly": 0.8,
    "biweekly": 0.6,
    "monthly": 0.4,
    "irregular": 0.2,
}

_RECHARGE_FREQ_SCORES = {
    "weekly": 1.0,
    "biweekly": 0.7,
    "monthly": 0.4,
}

_SALES_TREND_SCORES = {
    "growing": 1.0,
    "stable": 0.5,
    "declining": 0.0,
}

# Data categories used for coverage calculation
_DATA_CATEGORIES = ["rent", "utility", "mobile", "cashflow", "business"]

# Neutral fill values (roughly population medians) for features derived
# from missing sources.
_NEUTRAL_FILLS: dict[str, float] = {
    "rent_on_time_ratio": 0.5,
    "utility_on_time_ratio": 0.5,
    "overall_payment_reliability": 0.5,
    "missed_payment_ratio": 0.1,
    "average_days_late_norm": 0.3,
    "income_stability": 0.5,
    "zero_income_ratio": 0.15,
    "positive_balance_ratio": 0.5,
    "savings_regularity": 0.25,
    "expense_to_income_ratio": 0.65,
    "rent_to_income_ratio": 0.25,
    "bill_to_income_ratio": 0.10,
    "business_consistency": 0.0,
    "sales_trend_score": 0.0,
    "repeat_customer_score": 0.0,
    "seasonality_risk": 0.0,
    "income_frequency_score": 0.4,
    "mobile_tenure_score": 0.3,
    "recharge_regularity": 0.5,
    "disconnection_flag": 0.0,
    "high_expense_flag": 0.0,
    "zero_savings_flag": 0.0,
    "recent_missed_flag": 0.0,
}

# Ordered list of every feature this pipeline produces (excluding applicant_id).
FEATURE_NAMES: list[str] = [
    # Payment Reliability
    "rent_on_time_ratio",
    "utility_on_time_ratio",
    "overall_payment_reliability",
    "missed_payment_ratio",
    "average_days_late_norm",
    # Cash-flow Stability
    "income_stability",
    "zero_income_ratio",
    "positive_balance_ratio",
    "savings_regularity",
    # Expense Burden
    "expense_to_income_ratio",
    "rent_to_income_ratio",
    "bill_to_income_ratio",
    # Business Continuity
    "business_consistency",
    "sales_trend_score",
    "repeat_customer_score",
    "seasonality_risk",
    # Income Regularity
    "income_frequency_score",
    "mobile_tenure_score",
    "recharge_regularity",
    # Risk Signals
    "disconnection_flag",
    "high_expense_flag",
    "zero_savings_flag",
    "recent_missed_flag",
    # Data Completeness
    "data_coverage_score",
    "total_data_points",
]


# ---------------------------------------------------------------------------
# Feature descriptions
# ---------------------------------------------------------------------------

_FEATURE_DESCRIPTIONS: dict[str, str] = {
    "rent_on_time_ratio": "Fraction of rent payments made on time (0–1). 0 if no rent data.",
    "utility_on_time_ratio": "Fraction of utility payments made on time (0–1).",
    "overall_payment_reliability": "Weighted average of rent & utility on-time ratios.",
    "missed_payment_ratio": "Missed payments / total observed months across rent & utility.",
    "average_days_late_norm": "Average days late for rent normalised to 0–1 (dividing by 30).",
    "income_stability": "1 − income_volatility.  Higher is more stable.",
    "zero_income_ratio": "Fraction of observed weeks with zero income.",
    "positive_balance_ratio": "Fraction of the month (30 days) with a positive balance.",
    "savings_regularity": "Savings-deposit frequency normalised by max of 4 per month (capped at 1).",
    "expense_to_income_ratio": "Monthly expenses / monthly income (can exceed 1).",
    "rent_to_income_ratio": "Monthly rent / average monthly inflow.",
    "bill_to_income_ratio": "Monthly utility bill / average monthly inflow.",
    "business_consistency": "(active_days / 7) × collection_consistency.  0 for non-shopkeepers.",
    "sales_trend_score": "Growing=1, Stable=0.5, Declining=0.  0 for non-shopkeepers.",
    "repeat_customer_score": "Repeat-customer ratio for shopkeepers.  0 otherwise.",
    "seasonality_risk": "Seasonality index for shopkeepers (higher = riskier).  0 otherwise.",
    "income_frequency_score": "Score derived from income frequency: daily=1 … irregular=0.2.",
    "mobile_tenure_score": "Mobile tenure (months) / 60, capped at 1.",
    "recharge_regularity": "Regularity score from recharge frequency: weekly=1, monthly=0.4.",
    "disconnection_flag": "1 if any utility disconnection events recorded, else 0.",
    "high_expense_flag": "1 if expense_to_income_ratio > 0.8, else 0.",
    "zero_savings_flag": "1 if savings_deposit_frequency == 0, else 0.",
    "recent_missed_flag": "1 if total missed payments > 2 (proxy for recent misses), else 0.",
    "data_coverage_score": "Fraction of 5 data categories with consent AND available data (0–1).",
    "total_data_points": "Count of non-null feature values for this applicant.",
}


# ---------------------------------------------------------------------------
# FeatureEngineer class
# ---------------------------------------------------------------------------

class FeatureEngineer:
    """Transform raw alternative-data tables into model-ready features.

    Usage
    -----
    >>> fe = FeatureEngineer()
    >>> features_df = fe.engineer_features(
    ...     applicants_df, consents_df, rent_df, utility_df,
    ...     mobile_df, cashflow_df, business_df,
    ... )
    """

    def __init__(self) -> None:
        self.feature_names: list[str] = list(FEATURE_NAMES)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def engineer_features(
        self,
        applicants_df: pd.DataFrame,
        consents_df: pd.DataFrame,
        rent_df: pd.DataFrame,
        utility_df: pd.DataFrame,
        mobile_df: pd.DataFrame,
        cashflow_df: pd.DataFrame,
        business_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build the full feature matrix for every applicant.

        Parameters
        ----------
        applicants_df .. business_df : pd.DataFrame
            DataFrames produced by
            :func:`data.synthetic_generator.generate_full_dataset`.

        Returns
        -------
        pd.DataFrame
            One row per applicant, columns =
            ``['applicant_id'] + self.feature_names``.
        """
        # Index helper DataFrames by applicant_id for fast lookup
        rent_map = self._to_dict_map(rent_df)
        utility_map = self._to_dict_map(utility_df)
        mobile_map = self._to_dict_map(mobile_df)
        cashflow_map = self._to_dict_map(cashflow_df)
        business_map = self._to_dict_map(business_df)
        consent_map = self._to_dict_map(consents_df)

        rows: list[dict] = []
        for _, app in applicants_df.iterrows():
            aid = app["applicant_id"]
            occupation = app.get("occupation_type", "")
            income_type = app.get("income_type", "")

            rent = rent_map.get(aid)
            utility = utility_map.get(aid)
            mobile = mobile_map.get(aid)
            cashflow = cashflow_map.get(aid)
            business = business_map.get(aid)
            consent = consent_map.get(aid, {})

            feat = self._build_features(
                rent=rent,
                utility=utility,
                mobile=mobile,
                cashflow=cashflow,
                business=business,
                consent=consent,
                occupation=occupation,
                income_type=income_type,
            )
            feat["applicant_id"] = aid
            rows.append(feat)

        features_df = pd.DataFrame(rows)
        # Ensure column order
        cols = ["applicant_id"] + self.feature_names
        for c in cols:
            if c not in features_df.columns:
                features_df[c] = 0.0
        features_df = features_df[cols]

        return self.handle_missing(features_df)

    def engineer_single(self, applicant_data: dict, consent_status: Optional[dict] = None) -> pd.DataFrame:
        """Engineer features for a single applicant.

        Parameters
        ----------
        applicant_data : dict
            Dict returned by
            :func:`data.synthetic_generator.generate_single_applicant`
            with keys ``applicant``, ``consent``, ``rent``, ``utility``,
            ``mobile``, ``cashflow``, ``business``.
        consent_status : dict, optional
            Active consent flags/status to override default consent.

        Returns
        -------
        pd.DataFrame
            Single-row DataFrame with ``applicant_id`` + all features.
        """
        app = applicant_data.get("applicant", applicant_data)
        consent = dict(applicant_data.get("consent", {}))
        if consent_status:
            # Map consent categories or dict
            if isinstance(consent_status, dict):
                consents_sub = consent_status.get("consents", consent_status)
                for k, v in consents_sub.items():
                    norm_k = k if k.endswith("_consent") else f"{k}_consent"
                    consent[norm_k] = bool(v)

        feat = self._build_features(
            rent=applicant_data.get("rent"),
            utility=applicant_data.get("utility"),
            mobile=applicant_data.get("mobile"),
            cashflow=applicant_data.get("cashflow"),
            business=applicant_data.get("business"),
            consent=consent,
            occupation=app.get("occupation_type", ""),
            income_type=app.get("income_type", ""),
        )
        feat["applicant_id"] = app.get("applicant_id", "APP-0000")
        df = pd.DataFrame([feat])

        cols = ["applicant_id"] + self.feature_names
        for c in cols:
            if c not in df.columns:
                df[c] = 0.0
        df = df[cols]
        return self.handle_missing(df)

    @staticmethod
    def get_feature_descriptions() -> dict[str, str]:
        """Return human-readable description of each feature."""
        return dict(_FEATURE_DESCRIPTIONS)

    def handle_missing(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the feature matrix.

        Strategy
        --------
        * Numeric columns: fill NaN with the neutral/median value from
          ``_NEUTRAL_FILLS``, falling back to 0.
        * The ``data_coverage_score`` and ``total_data_points`` columns are
          never filled — they are always computed.

        Parameters
        ----------
        features_df : pd.DataFrame
            Feature matrix possibly containing NaNs.

        Returns
        -------
        pd.DataFrame
            Clean feature matrix with no NaN values.
        """
        df = features_df.copy()
        for col in self.feature_names:
            if col in df.columns and df[col].isna().any():
                fill = _NEUTRAL_FILLS.get(col, 0.0)
                df[col] = df[col].fillna(fill)
        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_dict_map(df: pd.DataFrame) -> dict[str, dict]:
        """Convert a DataFrame to ``{applicant_id: row_dict}``."""
        if df is None or df.empty:
            return {}
        records = df.to_dict(orient="records")
        return {r["applicant_id"]: r for r in records}

    def _build_features(
        self,
        rent: Optional[dict],
        utility: Optional[dict],
        mobile: Optional[dict],
        cashflow: Optional[dict],
        business: Optional[dict],
        consent: dict,
        occupation: str,
        income_type: str,
    ) -> dict:
        """Compute all feature values for one applicant."""
        feat: dict[str, float] = {}

        # ---- Payment Reliability ----
        feat.update(self._payment_reliability(rent, utility))

        # ---- Cash-flow Stability ----
        feat.update(self._cashflow_stability(cashflow))

        # ---- Expense Burden ----
        feat.update(self._expense_burden(rent, utility, cashflow))

        # ---- Business Continuity ----
        feat.update(self._business_continuity(business, occupation))

        # ---- Income Regularity ----
        feat.update(self._income_regularity(mobile, cashflow, income_type))

        # ---- Risk Signals ----
        feat.update(self._risk_signals(rent, utility, cashflow))

        # ---- Data Completeness ----
        feat.update(
            self._data_completeness(rent, utility, mobile, cashflow, business, consent)
        )

        return feat

    # ---- group builders ----

    @staticmethod
    def _payment_reliability(
        rent: Optional[dict], utility: Optional[dict]
    ) -> dict[str, float]:
        """Compute payment-reliability features."""
        f: dict[str, float] = {}

        # Rent on-time ratio
        if rent is not None:
            total_rent = rent.get("months_observed", 0)
            f["rent_on_time_ratio"] = (
                rent.get("on_time_payments", 0) / total_rent if total_rent > 0 else 0.0
            )
            f["average_days_late_norm"] = min(
                rent.get("average_days_late", 0) / 30.0, 1.0
            )
        else:
            f["rent_on_time_ratio"] = _NEUTRAL_FILLS["rent_on_time_ratio"]
            f["average_days_late_norm"] = _NEUTRAL_FILLS["average_days_late_norm"]

        # Utility on-time ratio
        if utility is not None:
            total_util = utility.get("months_observed", 0)
            f["utility_on_time_ratio"] = (
                utility.get("on_time_payments", 0) / total_util
                if total_util > 0
                else 0.0
            )
        else:
            f["utility_on_time_ratio"] = _NEUTRAL_FILLS["utility_on_time_ratio"]

        # Overall payment reliability (weighted: rent 60 %, utility 40 %)
        has_rent = rent is not None
        has_util = utility is not None
        if has_rent and has_util:
            f["overall_payment_reliability"] = (
                0.6 * f["rent_on_time_ratio"] + 0.4 * f["utility_on_time_ratio"]
            )
        elif has_rent:
            f["overall_payment_reliability"] = f["rent_on_time_ratio"]
        elif has_util:
            f["overall_payment_reliability"] = f["utility_on_time_ratio"]
        else:
            f["overall_payment_reliability"] = _NEUTRAL_FILLS[
                "overall_payment_reliability"
            ]

        # Missed-payment ratio (across rent + utility)
        total_months = 0
        total_missed = 0
        if rent is not None:
            total_months += rent.get("months_observed", 0)
            total_missed += rent.get("missed_payments", 0)
        if utility is not None:
            total_months += utility.get("months_observed", 0)
            total_missed += utility.get("late_payments", 0)  # late counted as missed-proxy
        f["missed_payment_ratio"] = (
            total_missed / total_months if total_months > 0 else _NEUTRAL_FILLS["missed_payment_ratio"]
        )

        return f

    @staticmethod
    def _cashflow_stability(cashflow: Optional[dict]) -> dict[str, float]:
        """Compute cash-flow stability features."""
        if cashflow is None:
            return {
                "income_stability": _NEUTRAL_FILLS["income_stability"],
                "zero_income_ratio": _NEUTRAL_FILLS["zero_income_ratio"],
                "positive_balance_ratio": _NEUTRAL_FILLS["positive_balance_ratio"],
                "savings_regularity": _NEUTRAL_FILLS["savings_regularity"],
            }

        volatility = cashflow.get("income_volatility", 0.5)
        zero_weeks = cashflow.get("zero_income_weeks", 0)
        # Assume roughly 48 observed weeks (12 months)
        total_weeks = 48
        pos_days = cashflow.get("positive_balance_days", 15)
        savings_freq = cashflow.get("savings_deposit_frequency", 0)

        return {
            "income_stability": round(1.0 - volatility, 4),
            "zero_income_ratio": round(zero_weeks / total_weeks, 4),
            "positive_balance_ratio": round(pos_days / 30.0, 4),
            "savings_regularity": round(min(savings_freq / 4.0, 1.0), 4),
        }

    @staticmethod
    def _expense_burden(
        rent: Optional[dict],
        utility: Optional[dict],
        cashflow: Optional[dict],
    ) -> dict[str, float]:
        """Compute expense-burden features."""
        f: dict[str, float] = {}

        inflow = (
            cashflow.get("average_monthly_inflow", 0) if cashflow is not None else 0
        )
        f["expense_to_income_ratio"] = (
            cashflow.get("expense_to_income_ratio", _NEUTRAL_FILLS["expense_to_income_ratio"])
            if cashflow is not None
            else _NEUTRAL_FILLS["expense_to_income_ratio"]
        )

        if rent is not None and inflow > 0:
            f["rent_to_income_ratio"] = round(
                rent.get("monthly_rent_amount", 0) / inflow, 4
            )
        else:
            f["rent_to_income_ratio"] = _NEUTRAL_FILLS["rent_to_income_ratio"]

        if utility is not None and inflow > 0:
            f["bill_to_income_ratio"] = round(
                utility.get("monthly_bill_amount", 0) / inflow, 4
            )
        else:
            f["bill_to_income_ratio"] = _NEUTRAL_FILLS["bill_to_income_ratio"]

        return f

    @staticmethod
    def _business_continuity(
        business: Optional[dict], occupation: str
    ) -> dict[str, float]:
        """Compute business-continuity features (0 for non-shopkeepers)."""
        if business is None or occupation != "Small Shopkeeper":
            return {
                "business_consistency": 0.0,
                "sales_trend_score": 0.0,
                "repeat_customer_score": 0.0,
                "seasonality_risk": 0.0,
            }
        active_days = business.get("active_transaction_days_per_week", 0)
        consistency = business.get("collection_consistency", 0)
        trend = business.get("sales_trend", "stable")
        repeat = business.get("repeat_customer_ratio", 0)
        seasonality = business.get("seasonality_index", 0)

        return {
            "business_consistency": round((active_days / 7.0) * consistency, 4),
            "sales_trend_score": _SALES_TREND_SCORES.get(trend, 0.5),
            "repeat_customer_score": round(float(repeat), 4),
            "seasonality_risk": round(float(seasonality), 4),
        }

    @staticmethod
    def _income_regularity(
        mobile: Optional[dict],
        cashflow: Optional[dict],
        income_type: str,
    ) -> dict[str, float]:
        """Compute income-regularity features."""
        f: dict[str, float] = {}

        # Income frequency score (from applicant's income_type mapped via
        # cashflow frequency if available, else from income_type directly)
        if cashflow is not None:
            freq = cashflow.get("income_frequency", income_type.lower())
        else:
            freq = income_type.lower()
        f["income_frequency_score"] = _INCOME_FREQ_SCORES.get(
            freq, _NEUTRAL_FILLS["income_frequency_score"]
        )

        # Mobile tenure score
        if mobile is not None:
            tenure = mobile.get("mobile_tenure_months", 0)
            f["mobile_tenure_score"] = round(min(tenure / 60.0, 1.0), 4)
        else:
            f["mobile_tenure_score"] = _NEUTRAL_FILLS["mobile_tenure_score"]

        # Recharge regularity
        if mobile is not None:
            recharge_freq = mobile.get("recharge_frequency", "monthly")
            f["recharge_regularity"] = _RECHARGE_FREQ_SCORES.get(
                recharge_freq, _NEUTRAL_FILLS["recharge_regularity"]
            )
        else:
            f["recharge_regularity"] = _NEUTRAL_FILLS["recharge_regularity"]

        return f

    @staticmethod
    def _risk_signals(
        rent: Optional[dict],
        utility: Optional[dict],
        cashflow: Optional[dict],
    ) -> dict[str, float]:
        """Compute binary risk-signal flags."""
        f: dict[str, float] = {}

        # Disconnection flag
        if utility is not None:
            f["disconnection_flag"] = (
                1.0 if utility.get("disconnection_events", 0) > 0 else 0.0
            )
        else:
            f["disconnection_flag"] = 0.0

        # High-expense flag
        if cashflow is not None:
            f["high_expense_flag"] = (
                1.0
                if cashflow.get("expense_to_income_ratio", 0) > 0.8
                else 0.0
            )
        else:
            f["high_expense_flag"] = 0.0

        # Zero-savings flag
        if cashflow is not None:
            f["zero_savings_flag"] = (
                1.0
                if cashflow.get("savings_deposit_frequency", 0) == 0
                else 0.0
            )
        else:
            f["zero_savings_flag"] = 0.0

        # Recent-missed flag (proxy: total missed > 2)
        total_missed = 0
        if rent is not None:
            total_missed += rent.get("missed_payments", 0)
        if utility is not None:
            total_missed += utility.get("late_payments", 0)
        f["recent_missed_flag"] = 1.0 if total_missed > 2 else 0.0

        return f

    @staticmethod
    def _data_completeness(
        rent: Optional[dict],
        utility: Optional[dict],
        mobile: Optional[dict],
        cashflow: Optional[dict],
        business: Optional[dict],
        consent: dict,
    ) -> dict[str, float]:
        """Compute data-coverage score and total data-point count."""
        categories_present = 0
        total_categories = len(_DATA_CATEGORIES)  # 5

        # A category counts as "present" only if consent was given AND data
        # exists (some consented categories may still have missing data due
        # to bias scenarios).
        if consent.get("rent_data_consent", False) and rent is not None:
            categories_present += 1
        if consent.get("utility_data_consent", False) and utility is not None:
            categories_present += 1
        if consent.get("mobile_data_consent", False) and mobile is not None:
            categories_present += 1
        if consent.get("cashflow_data_consent", False) and cashflow is not None:
            categories_present += 1
        if consent.get("business_data_consent", False) and business is not None:
            categories_present += 1

        coverage = round(categories_present / total_categories, 4)

        # Total data points: count non-None values across all source dicts
        total_pts = 0
        for src in (rent, utility, mobile, cashflow, business):
            if src is not None:
                total_pts += sum(
                    1 for v in src.values() if v is not None and str(v) != ""
                )

        return {
            "data_coverage_score": coverage,
            "total_data_points": float(total_pts),
        }


# ---------------------------------------------------------------------------
# Quick sanity check when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import os

    # Allow sibling-package import when running as script
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    from data.synthetic_generator import generate_full_dataset

    print("Generating dataset …")
    data = generate_full_dataset(n_applicants=500, random_seed=42)

    fe = FeatureEngineer()
    print("Engineering features …")
    features = fe.engineer_features(
        data["applicants"],
        data["consents"],
        data["rent"],
        data["utility"],
        data["mobile"],
        data["cashflow"],
        data["business"],
    )

    print(f"\nFeature matrix shape: {features.shape}")
    print(f"Columns: {list(features.columns)}")
    print(f"\nSample (first 5 rows):\n{features.head()}")
    print(f"\nDescriptive stats:\n{features.describe().T[['mean', 'std', 'min', 'max']]}")
    print(f"\nNull counts:\n{features.isnull().sum().to_string()}")
    print(f"\nFeature descriptions:")
    for name, desc in fe.get_feature_descriptions().items():
        print(f"  {name:35s} – {desc}")
