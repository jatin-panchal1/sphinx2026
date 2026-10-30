"""
Fairness metrics and bias audit engine for FairScore.

Computes group-level approval rates, false-positive / false-negative rates,
disparate impact ratios, and generates actionable bias alerts across
demographic slices (gender, age, urban/rural, region, occupation, income type).
"""

import numpy as np
import pandas as pd
from typing import Optional


class FairnessAuditor:
    """Fairness metrics and bias audit for alternative credit scoring."""

    DEMOGRAPHIC_COLUMNS: list[str] = [
        "gender_category",
        "age_group",
        "urban_rural",
        "region_category",
        "occupation_type",
        "income_type",
    ]

    def __init__(self, approval_threshold: float = 0.25):
        """Initialise the auditor.

        Args:
            approval_threshold: Applicants with ``default_probability`` below
                this value are considered *approved*.
        """
        self.approval_threshold = approval_threshold

    # ------------------------------------------------------------------
    # Per-group metrics
    # ------------------------------------------------------------------

    def compute_group_metrics(
        self,
        applicants_df: pd.DataFrame,
        predictions_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        group_column: str,
    ) -> pd.DataFrame:
        """Compute evaluation metrics for each value of *group_column*.

        Args:
            applicants_df: Must contain ``applicant_id`` and *group_column*.
            predictions_df: Columns ``applicant_id``, ``default_probability``,
                ``score``, ``risk_grade``.
            labels_df: Columns ``applicant_id``, ``default_label``.
            group_column: Name of the demographic column to slice by.

        Returns:
            DataFrame indexed by group value with columns: ``total_applicants``,
            ``approval_rate``, ``average_score``, ``average_default_prob``,
            ``false_negative_rate``, ``false_positive_rate``,
            ``predicted_default_rate``, ``bias_alert_flag``.
        """
        if group_column not in applicants_df.columns:
            return pd.DataFrame()

        # Merge everything on applicant_id
        merged = (
            applicants_df[["applicant_id", group_column]]
            .merge(predictions_df[["applicant_id", "default_probability", "score"]], on="applicant_id", how="inner")
            .merge(labels_df[["applicant_id", "default_label"]], on="applicant_id", how="inner")
        )

        merged["approved"] = (merged["default_probability"] < self.approval_threshold).astype(int)

        rows: list[dict] = []
        for group_val, grp in merged.groupby(group_column):
            n = len(grp)
            approval_rate = grp["approved"].mean()
            avg_score = grp["score"].mean()
            avg_prob = grp["default_probability"].mean()

            # False negatives: label=0 (reliable) but declined (approved=0)
            reliable = grp[grp["default_label"] == 0]
            fnr = 1.0 - reliable["approved"].mean() if len(reliable) > 0 else 0.0

            # False positives: label=1 (defaulter) but approved (approved=1)
            risky = grp[grp["default_label"] == 1]
            fpr = risky["approved"].mean() if len(risky) > 0 else 0.0

            # Predicted default rate among approved
            approved_set = grp[grp["approved"] == 1]
            pred_def_rate = approved_set["default_label"].mean() if len(approved_set) > 0 else 0.0

            rows.append(
                {
                    "group_value": group_val,
                    "total_applicants": n,
                    "approval_rate": float(approval_rate),
                    "average_score": float(avg_score),
                    "average_default_prob": float(avg_prob),
                    "false_negative_rate": float(fnr),
                    "false_positive_rate": float(fpr),
                    "predicted_default_rate": float(pred_def_rate),
                    "bias_alert_flag": False,  # will be set below
                }
            )

        result = pd.DataFrame(rows)

        if len(result) > 1:
            max_ar = result["approval_rate"].max()
            min_ar = result["approval_rate"].min()
            ar_gap = max_ar - min_ar
            result["bias_alert_flag"] = ar_gap > 0.10

        return result

    def compute_all_group_metrics(
        self,
        applicants_df: pd.DataFrame,
        predictions_df: pd.DataFrame,
        labels_df: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        """Compute metrics for every demographic column present in the data.

        Returns:
            Dictionary mapping ``group_column`` name to its metrics DataFrame.
        """
        results: dict[str, pd.DataFrame] = {}
        for col in self.DEMOGRAPHIC_COLUMNS:
            if col in applicants_df.columns:
                metrics = self.compute_group_metrics(
                    applicants_df, predictions_df, labels_df, col
                )
                if not metrics.empty:
                    results[col] = metrics
        return results

    # ------------------------------------------------------------------
    # Disparate impact
    # ------------------------------------------------------------------

    def compute_disparate_impact(self, group_metrics: pd.DataFrame) -> dict:
        """Compute the disparate impact ratio for a set of group metrics.

        The ratio is ``min_group_approval_rate / max_group_approval_rate``.
        An alert is raised when the ratio falls below 0.8 (the four-fifths
        rule).

        Returns:
            Dictionary with ``ratio``, ``advantaged_group``,
            ``disadvantaged_group``, and ``alert`` (bool).
        """
        if group_metrics.empty or len(group_metrics) < 2:
            return {
                "ratio": 1.0,
                "advantaged_group": "N/A",
                "disadvantaged_group": "N/A",
                "alert": False,
            }

        max_idx = group_metrics["approval_rate"].idxmax()
        min_idx = group_metrics["approval_rate"].idxmin()
        max_rate = group_metrics.loc[max_idx, "approval_rate"]
        min_rate = group_metrics.loc[min_idx, "approval_rate"]

        ratio = min_rate / max_rate if max_rate > 0 else 0.0

        return {
            "ratio": float(ratio),
            "advantaged_group": str(group_metrics.loc[max_idx, "group_value"]),
            "disadvantaged_group": str(group_metrics.loc[min_idx, "group_value"]),
            "alert": ratio < 0.8,
        }

    # ------------------------------------------------------------------
    # Bias alerts
    # ------------------------------------------------------------------

    def generate_bias_alerts(self, all_group_metrics: dict[str, pd.DataFrame]) -> list[dict]:
        """Generate actionable alert messages for detected fairness issues.

        Checks for:
            - Approval rate gap > 10 percentage points.
            - Disparate impact ratio < 0.8.
            - False-negative rate gap > 5 percentage points.
            - Average score gap > 50 points.

        Returns:
            List of alert dicts with ``group_column``, ``alert_type``,
            ``message``, ``severity``, and ``details``.
        """
        alerts: list[dict] = []

        for col, metrics in all_group_metrics.items():
            if metrics.empty or len(metrics) < 2:
                continue

            col_label = col.replace("_", " ").title()

            # --- Approval rate gap ---
            ar_max = metrics["approval_rate"].max()
            ar_min = metrics["approval_rate"].min()
            ar_gap = ar_max - ar_min

            if ar_gap > 0.10:
                best = metrics.loc[metrics["approval_rate"].idxmax(), "group_value"]
                worst = metrics.loc[metrics["approval_rate"].idxmin(), "group_value"]
                severity = "high" if ar_gap > 0.20 else "medium"
                alerts.append(
                    {
                        "group_column": col,
                        "alert_type": "approval_rate_gap",
                        "message": (
                            f"Approval rate gap of {ar_gap:.1%} detected across {col_label}. "
                            f"'{best}' has the highest rate ({ar_max:.1%}) while "
                            f"'{worst}' has the lowest ({ar_min:.1%}). "
                            f"Review feature distributions and consider bias mitigation."
                        ),
                        "severity": severity,
                        "details": {
                            "gap": float(ar_gap),
                            "best_group": str(best),
                            "worst_group": str(worst),
                        },
                    }
                )

            # --- Disparate impact ---
            di = self.compute_disparate_impact(metrics)
            if di["alert"]:
                severity = "high" if di["ratio"] < 0.6 else "medium"
                alerts.append(
                    {
                        "group_column": col,
                        "alert_type": "disparate_impact",
                        "message": (
                            f"Disparate impact ratio of {di['ratio']:.2f} detected for {col_label} "
                            f"(below 0.80 threshold). "
                            f"'{di['disadvantaged_group']}' is disadvantaged relative to "
                            f"'{di['advantaged_group']}'. "
                            f"This may indicate systemic bias in scoring."
                        ),
                        "severity": severity,
                        "details": di,
                    }
                )

            # --- False-negative rate gap ---
            fnr_max = metrics["false_negative_rate"].max()
            fnr_min = metrics["false_negative_rate"].min()
            fnr_gap = fnr_max - fnr_min

            if fnr_gap > 0.05:
                worst_fnr_group = metrics.loc[
                    metrics["false_negative_rate"].idxmax(), "group_value"
                ]
                severity = "high" if fnr_gap > 0.15 else "medium"
                alerts.append(
                    {
                        "group_column": col,
                        "alert_type": "false_negative_rate_gap",
                        "message": (
                            f"False-negative rate gap of {fnr_gap:.1%} across {col_label}. "
                            f"Reliable applicants in '{worst_fnr_group}' are more likely to be "
                            f"wrongly declined. Consider recalibrating thresholds."
                        ),
                        "severity": severity,
                        "details": {
                            "gap": float(fnr_gap),
                            "worst_group": str(worst_fnr_group),
                        },
                    }
                )

            # --- Score distribution gap ---
            score_max = metrics["average_score"].max()
            score_min = metrics["average_score"].min()
            score_gap = score_max - score_min

            if score_gap > 50:
                best_score_group = metrics.loc[
                    metrics["average_score"].idxmax(), "group_value"
                ]
                worst_score_group = metrics.loc[
                    metrics["average_score"].idxmin(), "group_value"
                ]
                severity = "high" if score_gap > 100 else "medium"
                alerts.append(
                    {
                        "group_column": col,
                        "alert_type": "score_distribution_gap",
                        "message": (
                            f"Average score gap of {score_gap:.0f} points across {col_label}. "
                            f"'{best_score_group}' averages {score_max:.0f} vs "
                            f"'{worst_score_group}' at {score_min:.0f}. "
                            f"Investigate feature availability and data coverage across groups."
                        ),
                        "severity": severity,
                        "details": {
                            "gap": float(score_gap),
                            "best_group": str(best_score_group),
                            "worst_group": str(worst_score_group),
                        },
                    }
                )

        return alerts

    # ------------------------------------------------------------------
    # Feature analysis by group
    # ------------------------------------------------------------------

    def get_feature_impact_by_group(
        self,
        features_df: pd.DataFrame,
        applicants_df: pd.DataFrame,
        group_column: str,
    ) -> pd.DataFrame:
        """Compare average feature values across demographic groups.

        Args:
            features_df: Feature matrix (must include ``applicant_id``).
            applicants_df: Must contain ``applicant_id`` and *group_column*.
            group_column: Demographic column to slice by.

        Returns:
            DataFrame with groups as columns and features as rows, showing
            mean feature values per group.
        """
        if group_column not in applicants_df.columns:
            return pd.DataFrame()

        merged = features_df.merge(
            applicants_df[["applicant_id", group_column]], on="applicant_id", how="inner"
        )

        # Identify numeric columns only (exclude applicant_id and group col)
        numeric_cols = [
            c for c in merged.columns if c not in ("applicant_id", group_column)
            and pd.api.types.is_numeric_dtype(merged[c])
        ]

        grouped = merged.groupby(group_column)[numeric_cols].mean()
        return grouped.T  # features as rows, groups as columns

    # ------------------------------------------------------------------
    # Comprehensive summary
    # ------------------------------------------------------------------

    def get_fairness_summary(
        self,
        applicants_df: pd.DataFrame,
        predictions_df: pd.DataFrame,
        labels_df: pd.DataFrame,
    ) -> dict:
        """Produce a comprehensive fairness summary.

        Returns:
            Dictionary with keys:
                - ``group_metrics``: per-demographic metrics.
                - ``bias_alerts``: list of alert dicts.
                - ``disparate_impact``: per-demographic DI ratios.
                - ``summary_stats``: high-level statistics.
        """
        group_metrics = self.compute_all_group_metrics(
            applicants_df, predictions_df, labels_df
        )
        bias_alerts = self.generate_bias_alerts(group_metrics)

        disparate_impact: dict[str, dict] = {}
        for col, metrics in group_metrics.items():
            disparate_impact[col] = self.compute_disparate_impact(metrics)

        # Summary stats
        total_alerts = len(bias_alerts)
        high_severity = sum(1 for a in bias_alerts if a["severity"] == "high")
        di_violations = sum(1 for d in disparate_impact.values() if d.get("alert"))

        summary_stats = {
            "total_demographics_audited": len(group_metrics),
            "total_alerts": total_alerts,
            "high_severity_alerts": high_severity,
            "disparate_impact_violations": di_violations,
            "overall_status": (
                "PASS" if total_alerts == 0
                else "REVIEW" if high_severity == 0
                else "FAIL"
            ),
        }

        # Serialise group_metrics DataFrames to dicts for JSON compatibility
        group_metrics_serialised: dict[str, list[dict]] = {}
        for col, df in group_metrics.items():
            group_metrics_serialised[col] = df.to_dict(orient="records")

        return {
            "group_metrics": group_metrics_serialised,
            "bias_alerts": bias_alerts,
            "disparate_impact": disparate_impact,
            "summary_stats": summary_stats,
        }
