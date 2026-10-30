"""
SHAP-based explainability engine for FairScore credit scoring.

Provides local (per-applicant) and global (across population) explanations of
model predictions, translating SHAP values into human-readable reason codes
and plain-language summaries.
"""

import shap
import numpy as np
import pandas as pd
from typing import Optional


class ExplainabilityEngine:
    """SHAP-based model explainability for credit scoring."""

    # Human-readable feature name mapping
    FEATURE_LABELS: dict[str, str] = {
        "rent_on_time_ratio": "Rent Payment Timeliness",
        "utility_on_time_ratio": "Utility Payment Timeliness",
        "overall_payment_reliability": "Overall Payment Reliability",
        "missed_payment_ratio": "Missed Payment Rate",
        "average_days_late_norm": "Average Payment Delay",
        "income_stability": "Income Stability",
        "zero_income_ratio": "Periods Without Income",
        "positive_balance_ratio": "Positive Balance Frequency",
        "savings_regularity": "Savings Regularity",
        "expense_to_income_ratio": "Expense-to-Income Ratio",
        "rent_to_income_ratio": "Rent Burden",
        "bill_to_income_ratio": "Bill Burden",
        "business_consistency": "Business Consistency",
        "sales_trend_score": "Sales Trend",
        "repeat_customer_score": "Repeat Customer Base",
        "seasonality_risk": "Seasonal Business Risk",
        "income_frequency_score": "Income Regularity",
        "mobile_tenure_score": "Mobile Account Tenure",
        "recharge_regularity": "Mobile Recharge Regularity",
        "disconnection_flag": "Utility Disconnection History",
        "high_expense_flag": "High Expense Warning",
        "zero_savings_flag": "No Savings Warning",
        "recent_missed_flag": "Recent Missed Payments",
        "data_coverage_score": "Data Completeness",
        "total_data_points": "Available Data Points",
    }

    # Human-readable reason code templates (positive contributions = lower risk)
    POSITIVE_REASONS: dict[str, str] = {
        "rent_on_time_ratio": "Strong rent payment history with {:.0%} on-time payments",
        "utility_on_time_ratio": "Consistent utility bill payments ({:.0%} on-time)",
        "overall_payment_reliability": "High overall payment reliability score",
        "income_stability": "Stable and predictable income pattern",
        "positive_balance_ratio": "Frequently maintains positive balance",
        "savings_regularity": "Regular savings deposits",
        "business_consistency": "Consistent business operations",
        "sales_trend_score": "Positive business sales trend",
        "repeat_customer_score": "Strong repeat customer base",
        "income_frequency_score": "Regular income frequency",
        "mobile_tenure_score": "Long-standing mobile account",
        "recharge_regularity": "Regular mobile recharge pattern",
        "data_coverage_score": "Comprehensive data available for assessment",
    }

    # Templates for negative contributions = higher risk
    NEGATIVE_REASONS: dict[str, str] = {
        "missed_payment_ratio": "History of missed payments ({:.0%} of total)",
        "average_days_late_norm": "Payments frequently delayed",
        "expense_to_income_ratio": "High expense-to-income ratio ({:.0%})",
        "rent_to_income_ratio": "High rent burden relative to income",
        "bill_to_income_ratio": "High utility bill burden relative to income",
        "zero_income_ratio": "Periods with zero income recorded",
        "seasonality_risk": "Seasonal income variation affects stability",
        "disconnection_flag": "History of utility disconnections",
        "high_expense_flag": "Expenses exceed recommended threshold",
        "zero_savings_flag": "No regular savings pattern detected",
        "recent_missed_flag": "Recent missed payments detected",
        "income_stability": "Inconsistent income pattern",
    }

    def __init__(self, model, scaler, feature_columns: list[str]):
        """Initialise the explainability engine.

        Args:
            model: A trained scikit-learn classifier with ``predict_proba``.
            scaler: A fitted ``StandardScaler`` used during training.
            feature_columns: Ordered list of feature column names.
        """
        self.model = model
        self.scaler = scaler
        self.feature_columns = feature_columns
        self.explainer: Optional[shap.Explainer] = None
        self.background_data: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _predict_fn(self, X: np.ndarray) -> np.ndarray:
        """Wrapper that scales raw features and returns P(default)."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def initialize(self, background_data: pd.DataFrame, max_samples: int = 100) -> None:
        """Initialise the SHAP explainer with background (reference) data.

        Args:
            background_data: DataFrame with feature columns used for training.
            max_samples: Maximum rows to keep as the background dataset
                (sub-sampled for speed).
        """
        bg = background_data[self.feature_columns].copy()
        if len(bg) > max_samples:
            bg = bg.sample(n=max_samples, random_state=42)
        self.background_data = bg

        # Use the KernelExplainer-compatible shap.Explainer with a masker
        self.explainer = shap.Explainer(
            self._predict_fn,
            self.background_data,
            feature_names=self.feature_columns,
        )

    # ------------------------------------------------------------------
    # Local (single-applicant) explanation
    # ------------------------------------------------------------------

    def explain_single(self, features) -> dict:
        """Explain a single applicant's prediction.

        Args:
            features: A single-row DataFrame, Series, or dict with feature columns.

        Returns:
            Dictionary with keys:
                - ``shap_values``: 1-D array of SHAP values per feature.
                - ``feature_values``: dict mapping feature name → raw value.
                - ``positive_factors``: list of dicts for the top-5 features
                  that *reduce* default risk (negative SHAP value).
                - ``negative_factors``: list of dicts for the top-5 features
                  that *increase* default risk (positive SHAP value).
                - ``summary_text``: plain-language explanation string.
        """
        if self.explainer is None:
            raise RuntimeError("Explainer not initialised. Call initialize() first.")

        if isinstance(features, dict):
            df = pd.DataFrame([features])
            for col in self.feature_columns:
                if col not in df.columns:
                    df[col] = 0.0
            row = df[self.feature_columns].iloc[[0]]
        elif isinstance(features, pd.Series):
            df = pd.DataFrame([features.to_dict()])
            for col in self.feature_columns:
                if col not in df.columns:
                    df[col] = 0.0
            row = df[self.feature_columns].iloc[[0]]
        else:
            row = features[self.feature_columns].iloc[[0]]

        shap_values_obj = self.explainer(row)

        # shap_values_obj.values is shape (1, n_features)
        sv = shap_values_obj.values[0]
        feature_vals = row.iloc[0].to_dict()

        # Positive factors: features whose SHAP value pushes prediction DOWN
        # (negative SHAP → lower default prob → good for applicant)
        # Negative factors: features whose SHAP value pushes prediction UP
        positive_indices = np.where(sv < 0)[0]
        negative_indices = np.where(sv > 0)[0]

        # Sort by absolute impact descending
        pos_sorted = sorted(positive_indices, key=lambda i: abs(sv[i]), reverse=True)[:5]
        neg_sorted = sorted(negative_indices, key=lambda i: abs(sv[i]), reverse=True)[:5]

        positive_factors = [
            self._build_factor(self.feature_columns[i], sv[i], feature_vals, is_positive=True)
            for i in pos_sorted
        ]
        negative_factors = [
            self._build_factor(self.feature_columns[i], sv[i], feature_vals, is_positive=False)
            for i in neg_sorted
        ]

        summary_text = self.generate_summary_text(
            score=700,
            risk_grade="Estimated",
            positive_factors=positive_factors,
            negative_factors=negative_factors,
        )

        return {
            "shap_values": sv.tolist() if hasattr(sv, "tolist") else sv,
            "feature_values": feature_vals,
            "positive_factors": positive_factors,
            "negative_factors": negative_factors,
            "summary_text": summary_text,
        }

    def _build_factor(
        self, feature: str, shap_value: float, feature_vals: dict, is_positive: bool
    ) -> dict:
        """Construct a factor dict for a single feature."""
        value = feature_vals.get(feature, 0.0)
        label = self.FEATURE_LABELS.get(feature, feature)
        reason = self._build_reason(feature, value, is_positive)
        return {
            "feature": feature,
            "label": label,
            "impact": float(shap_value),
            "value": float(value),
            "reason": reason,
        }

    def _build_reason(self, feature: str, value: float, is_positive: bool) -> str:
        """Generate a human-readable reason string for a factor."""
        if is_positive:
            template = self.POSITIVE_REASONS.get(feature)
            if template and "{" in template:
                try:
                    return template.format(value)
                except (ValueError, IndexError):
                    return template
            elif template:
                return template
            return f"{self.FEATURE_LABELS.get(feature, feature)} contributed positively"
        else:
            template = self.NEGATIVE_REASONS.get(feature)
            if template and "{" in template:
                try:
                    return template.format(value)
                except (ValueError, IndexError):
                    return template
            elif template:
                return template
            return f"{self.FEATURE_LABELS.get(feature, feature)} contributed negatively"

    # ------------------------------------------------------------------
    # Global explanation
    # ------------------------------------------------------------------

    def explain_global(self, features_df: pd.DataFrame) -> dict:
        """Compute global feature importance across many applicants.

        Args:
            features_df: DataFrame with multiple applicant rows.

        Returns:
            Dictionary with:
                - ``feature_importance``: list of dicts sorted by descending
                  mean |SHAP|.
                - ``top_features``: list of the top-10 feature names.
        """
        if self.explainer is None:
            raise RuntimeError("Explainer not initialised. Call initialize() first.")

        X = features_df[self.feature_columns]
        shap_values_obj = self.explainer(X)
        sv = shap_values_obj.values  # (n_samples, n_features)

        mean_abs = np.mean(np.abs(sv), axis=0)

        importance = []
        for i, feat in enumerate(self.feature_columns):
            importance.append(
                {
                    "feature": feat,
                    "label": self.FEATURE_LABELS.get(feat, feat),
                    "mean_abs_shap": float(mean_abs[i]),
                }
            )

        importance.sort(key=lambda d: d["mean_abs_shap"], reverse=True)
        top_features = [d["feature"] for d in importance[:10]]

        return {
            "feature_importance": importance,
            "top_features": top_features,
        }

    # ------------------------------------------------------------------
    # Summary & reason codes
    # ------------------------------------------------------------------

    def generate_summary_text(
        self,
        score: int,
        risk_grade: str,
        positive_factors: list[dict],
        negative_factors: list[dict],
    ) -> str:
        """Generate a plain-language explanation of the applicant's score.

        Args:
            score: Credit score (300–900).
            risk_grade: Risk grade string.
            positive_factors: List of positive factor dicts from
                :pymeth:`explain_single`.
            negative_factors: List of negative factor dicts from
                :pymeth:`explain_single`.

        Returns:
            A human-readable paragraph summarising the score drivers.
        """
        pos_reasons = [f["reason"] for f in positive_factors[:3]]
        neg_reasons = [f["reason"] for f in negative_factors[:3]]

        parts = [f"The applicant received a score of {score} ({risk_grade} risk)"]

        if pos_reasons:
            parts.append(
                " primarily due to " + ", ".join(pos_reasons[:-1])
                + (f", and {pos_reasons[-1]}" if len(pos_reasons) > 1 else pos_reasons[0])
            )

        if neg_reasons:
            parts.append(
                ". The score was reduced due to " + ", ".join(neg_reasons[:-1])
                + (f", and {neg_reasons[-1]}" if len(neg_reasons) > 1 else neg_reasons[0])
            )

        text = "".join(parts) + "."
        # Clean up double periods
        text = text.replace("..", ".")
        return text

    def get_reason_codes(
        self, positive_factors: list[dict], negative_factors: list[dict]
    ) -> list[str]:
        """Return a flat list of human-readable reason code strings.

        Positive reasons are prefixed with ``(+)`` and negative with ``(-)``.
        """
        codes: list[str] = []
        for f in positive_factors:
            codes.append(f"(+) {f['reason']}")
        for f in negative_factors:
            codes.append(f"(-) {f['reason']}")
        return codes
