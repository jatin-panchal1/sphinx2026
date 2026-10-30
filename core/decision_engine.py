"""
Decision recommendation engine for FairScore.

Combines model output (default probability, confidence) with business rules
(expense ratios, loan-to-income, data coverage) to produce an actionable
recommendation: Approve, Approve with Reduced Limit, Manual Review, or Decline.
"""

from typing import Optional


class DecisionEngine:
    """Decision support engine for credit scoring recommendations."""

    DEFAULT_THRESHOLDS: dict = {
        "approve_max_default_prob": 0.15,
        "review_max_default_prob": 0.35,
        "min_data_coverage": 0.4,
        "high_confidence_coverage": 0.7,
        "max_expense_ratio": 0.8,
        "max_loan_to_income_ratio": 5.0,
    }

    def __init__(self, thresholds: Optional[dict] = None):
        """Initialise the decision engine.

        Args:
            thresholds: Override default thresholds. Only keys present in the
                dict are overridden; the rest keep their defaults.
        """
        self.thresholds = dict(self.DEFAULT_THRESHOLDS)
        if thresholds:
            self.thresholds.update(thresholds)

    # ------------------------------------------------------------------
    # Main recommendation logic
    # ------------------------------------------------------------------

    def recommend(
        self,
        default_probability: float = 0.2,
        score: int = 650,
        risk_grade: str = "Medium",
        confidence: str = "Medium",
        data_coverage: float = 0.5,
        expense_ratio: float = 0.5,
        loan_amount: float = 50000.0,
        monthly_income: float = 25000.0,
        missed_payments: int = 0,
        fairness_alert: bool = False,
        **kwargs,
    ) -> dict:
        """Produce a lending recommendation.

        Logic (evaluated in order):
            1. **Decline** if ``default_prob > review_max_default_prob``.
            2. **Manual Review** if ``default_prob`` is between approve and
               review thresholds, or data coverage is low, or confidence is
               ``Low``, or a fairness alert is active.
            3. **Approve with Reduced Limit** if ``default_prob`` is below
               the approve threshold but the file is thin, loan-to-income
               is high, or confidence is ``Medium``.
            4. **Approve** otherwise.

        Returns:
            Dictionary with ``recommendation``, ``reason``,
            ``suggested_limit``, ``flags``, ``manual_review_triggers``, and
            ``improvement_suggestions``.
        """
        t = self.thresholds
        flags: list[str] = []
        manual_triggers: list[str] = []
        improvement: list[str] = []

        loan_to_income = (
            loan_amount / monthly_income if monthly_income > 0 else float("inf")
        )

        # Collect flags
        if data_coverage < t["min_data_coverage"]:
            flags.append("Thin file: limited alternative data available")
        if expense_ratio > t["max_expense_ratio"]:
            flags.append(f"High expense ratio ({expense_ratio:.0%})")
        if loan_to_income > t["max_loan_to_income_ratio"]:
            flags.append(
                f"Loan-to-income ratio is high ({loan_to_income:.1f}x monthly income)"
            )
        if missed_payments > 0:
            flags.append(f"{missed_payments} missed payment(s) on record")
        if fairness_alert:
            flags.append("Fairness audit alert triggered for this applicant's demographic group")

        # ------ Decision path ------

        # 1. Decline
        if default_probability > t["review_max_default_prob"]:
            improvement = self._get_decline_suggestions(
                default_probability, expense_ratio, data_coverage, missed_payments
            )
            return {
                "recommendation": "Decline",
                "reason": (
                    f"Default probability ({default_probability:.1%}) exceeds the "
                    f"maximum review threshold ({t['review_max_default_prob']:.0%}). "
                    f"Risk grade: {risk_grade}."
                ),
                "suggested_limit": None,
                "flags": flags,
                "manual_review_triggers": [],
                "improvement_suggestions": improvement,
            }

        # 2. Manual Review
        needs_review = False

        if default_probability > t["approve_max_default_prob"]:
            manual_triggers.append(
                f"Default probability ({default_probability:.1%}) is above auto-approve "
                f"threshold ({t['approve_max_default_prob']:.0%})"
            )
            needs_review = True

        if confidence == "Low":
            manual_triggers.append("Model confidence is Low — prediction may be unreliable")
            needs_review = True

        if data_coverage < t["min_data_coverage"]:
            manual_triggers.append(
                f"Data coverage ({data_coverage:.0%}) is below minimum ({t['min_data_coverage']:.0%})"
            )
            needs_review = True

        if fairness_alert:
            manual_triggers.append(
                "Fairness audit flagged potential bias for this demographic group"
            )
            needs_review = True

        if expense_ratio > t["max_expense_ratio"] and default_probability > t["approve_max_default_prob"]:
            manual_triggers.append(
                f"High expense ratio ({expense_ratio:.0%}) combined with elevated default risk"
            )
            needs_review = True

        if needs_review:
            improvement = self._get_review_suggestions(
                default_probability, expense_ratio, data_coverage, missed_payments
            )
            return {
                "recommendation": "Manual Review",
                "reason": (
                    f"Application requires manual review due to: "
                    + "; ".join(manual_triggers[:3]) + "."
                ),
                "suggested_limit": None,
                "flags": flags,
                "manual_review_triggers": manual_triggers,
                "improvement_suggestions": improvement,
            }

        # 3. Approve with Reduced Limit
        thin_file = data_coverage < t["high_confidence_coverage"]
        high_lti = loan_to_income > t["max_loan_to_income_ratio"] * 0.7
        medium_confidence = confidence == "Medium"

        if thin_file or high_lti or medium_confidence:
            suggested = loan_amount * (1 - default_probability) * 0.7
            reasons_parts: list[str] = []
            if thin_file:
                reasons_parts.append("thin data file")
            if high_lti:
                reasons_parts.append(f"elevated loan-to-income ratio ({loan_to_income:.1f}x)")
            if medium_confidence:
                reasons_parts.append("medium confidence prediction")

            improvement = self._get_reduced_suggestions(
                data_coverage, loan_to_income, expense_ratio
            )
            return {
                "recommendation": "Approve with Reduced Limit",
                "reason": (
                    f"Approved at a reduced limit due to: {', '.join(reasons_parts)}. "
                    f"Score {score} ({risk_grade} risk), default probability {default_probability:.1%}."
                ),
                "suggested_limit": round(suggested, 2),
                "flags": flags,
                "manual_review_triggers": [],
                "improvement_suggestions": improvement,
            }

        # 4. Full Approve
        return {
            "recommendation": "Approve",
            "reason": (
                f"Application approved. Score {score} ({risk_grade} risk), "
                f"default probability {default_probability:.1%}, "
                f"confidence {confidence}."
            ),
            "suggested_limit": loan_amount,
            "flags": flags,
            "manual_review_triggers": [],
            "improvement_suggestions": [],
        }

    # ------------------------------------------------------------------
    # Improvement suggestions
    # ------------------------------------------------------------------

    def get_improvement_suggestions(self, features: dict, recommendation: str) -> list[str]:
        """Generate specific improvement suggestions based on feature values.

        Args:
            features: Dictionary of feature name → value.
            recommendation: The recommendation string returned by
                :pymeth:`recommend`.

        Returns:
            List of actionable suggestion strings.
        """
        suggestions: list[str] = []

        if recommendation == "Approve":
            return suggestions

        rent = features.get("rent_on_time_ratio", 1.0)
        utility = features.get("utility_on_time_ratio", 1.0)
        expense = features.get("expense_to_income_ratio", 0.0)
        savings = features.get("savings_regularity", 1.0)
        coverage = features.get("data_coverage_score", 1.0)
        missed = features.get("missed_payment_ratio", 0.0)
        income_stability = features.get("income_stability", 1.0)
        zero_savings = features.get("zero_savings_flag", 0.0)
        recent_missed = features.get("recent_missed_flag", 0.0)

        if rent < 0.8:
            suggestions.append(
                "Maintain consistent rent payments for 3+ months to improve "
                "payment reliability score"
            )
        if utility < 0.8:
            suggestions.append(
                "Pay utility bills on time consistently to strengthen bill "
                "payment discipline"
            )
        if expense > 0.8:
            suggestions.append(
                "Reduce expense-to-income ratio below 80% by managing "
                "discretionary spending"
            )
        if savings < 0.3 or zero_savings > 0.5:
            suggestions.append(
                "Build a savings habit with regular small deposits to "
                "demonstrate financial discipline"
            )
        if coverage < 0.6:
            suggestions.append(
                "Provide additional data categories (utility, mobile, business) "
                "to improve data coverage and scoring confidence"
            )
        if missed > 0.2:
            suggestions.append(
                "Address outstanding missed payments and maintain a clean "
                "payment record for at least 3 months"
            )
        if income_stability < 0.5:
            suggestions.append(
                "Demonstrate stable income over multiple months to improve "
                "income stability assessment"
            )
        if recent_missed > 0.5:
            suggestions.append(
                "Clear any recently missed payments immediately — recent "
                "delinquencies weigh heavily on the score"
            )

        if not suggestions:
            suggestions.append(
                "Continue maintaining current financial habits and reapply "
                "after providing more data history"
            )

        return suggestions

    # ------------------------------------------------------------------
    # Internal suggestion helpers
    # ------------------------------------------------------------------

    def _get_decline_suggestions(
        self,
        default_prob: float,
        expense_ratio: float,
        data_coverage: float,
        missed_payments: int,
    ) -> list[str]:
        """Suggestions for declined applicants."""
        suggestions: list[str] = []
        if missed_payments > 0:
            suggestions.append(
                "Address all outstanding missed payments and maintain a clean "
                "record for 3–6 months before reapplying"
            )
        if expense_ratio > 0.8:
            suggestions.append(
                "Reduce monthly expenses relative to income — aim for an "
                "expense-to-income ratio below 70%"
            )
        if data_coverage < 0.5:
            suggestions.append(
                "Provide more data sources (rent receipts, utility bills, "
                "mobile records) to improve scoring accuracy"
            )
        suggestions.append(
            "Consider starting with a smaller loan amount to build a credit "
            "history within the FairScore system"
        )
        return suggestions

    def _get_review_suggestions(
        self,
        default_prob: float,
        expense_ratio: float,
        data_coverage: float,
        missed_payments: int,
    ) -> list[str]:
        """Suggestions for manual-review applicants."""
        suggestions: list[str] = []
        if data_coverage < 0.6:
            suggestions.append(
                "Provide additional data categories to increase coverage and "
                "improve prediction confidence"
            )
        if missed_payments > 0:
            suggestions.append(
                "Clear any recent missed payments to strengthen your profile"
            )
        if expense_ratio > 0.7:
            suggestions.append(
                "Reducing discretionary spending could lower your expense-to-"
                "income ratio and improve your score"
            )
        suggestions.append(
            "A co-applicant or additional collateral may help during manual review"
        )
        return suggestions

    def _get_reduced_suggestions(
        self,
        data_coverage: float,
        loan_to_income: float,
        expense_ratio: float,
    ) -> list[str]:
        """Suggestions for reduced-limit approvals."""
        suggestions: list[str] = []
        if data_coverage < 0.7:
            suggestions.append(
                "Link more data sources to increase your data coverage score "
                "and qualify for a higher limit"
            )
        if loan_to_income > 3.5:
            suggestions.append(
                "Consider a smaller loan amount relative to your income to "
                "qualify for the full requested amount"
            )
        if expense_ratio > 0.6:
            suggestions.append(
                "Lowering your expense ratio could unlock a higher credit limit"
            )
        suggestions.append(
            "Maintain timely payments on this loan to build history and "
            "qualify for limit increases"
        )
        return suggestions

    # ------------------------------------------------------------------
    # Threshold management
    # ------------------------------------------------------------------

    def update_thresholds(self, new_thresholds: dict) -> None:
        """Update decision thresholds.

        Args:
            new_thresholds: Dictionary of threshold names and new values.
                Only keys already in the threshold dict are accepted.
        """
        for key, value in new_thresholds.items():
            if key in self.thresholds:
                self.thresholds[key] = value
            else:
                raise KeyError(
                    f"Unknown threshold '{key}'. Valid keys: "
                    f"{list(self.thresholds.keys())}"
                )
