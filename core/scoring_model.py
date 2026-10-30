"""
ML scoring model for alternative credit scoring.

Uses scikit-learn classifiers (Gradient Boosting, Random Forest, Logistic Regression)
to predict default probability from alternative data features, then maps predictions
to a 300–900 credit score with risk grades and confidence levels.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import StandardScaler
import joblib
import os
from typing import Optional


class CreditScoringModel:
    """Alternative credit scoring model using scikit-learn.

    Trains on engineered features from alternative data sources (rent, utility,
    mobile, cash-flow, business) and produces a credit score on a 300–900 scale
    together with a risk grade, confidence indicator, and default probability.
    """

    FEATURE_COLUMNS = [
        "rent_on_time_ratio",
        "utility_on_time_ratio",
        "overall_payment_reliability",
        "missed_payment_ratio",
        "average_days_late_norm",
        "income_stability",
        "zero_income_ratio",
        "positive_balance_ratio",
        "savings_regularity",
        "expense_to_income_ratio",
        "rent_to_income_ratio",
        "bill_to_income_ratio",
        "business_consistency",
        "sales_trend_score",
        "repeat_customer_score",
        "seasonality_risk",
        "income_frequency_score",
        "mobile_tenure_score",
        "recharge_regularity",
        "disconnection_flag",
        "high_expense_flag",
        "zero_savings_flag",
        "recent_missed_flag",
        "data_coverage_score",
        "total_data_points",
    ]

    def __init__(self, model_type: str = "gradient_boosting"):
        """Initialise the scoring model.

        Args:
            model_type: One of ``'logistic_regression'``, ``'random_forest'``,
                or ``'gradient_boosting'``.
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.model_version = "1.0"
        self.evaluation_metrics: dict = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_model(self):
        """Create a fresh sklearn estimator based on ``self.model_type``."""
        if self.model_type == "logistic_regression":
            return LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            )
        elif self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                min_samples_leaf=10,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
        elif self.model_type == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                min_samples_leaf=10,
                subsample=0.8,
                random_state=42,
            )
        else:
            raise ValueError(
                f"Unknown model_type '{self.model_type}'. "
                "Choose from 'logistic_regression', 'random_forest', 'gradient_boosting'."
            )

    def _validate_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Ensure the DataFrame contains exactly the expected feature columns.

        Missing columns are filled with 0.0; extra columns are dropped.
        """
        df = features_df.copy()
        for col in self.FEATURE_COLUMNS:
            if col not in df.columns:
                df[col] = 0.0
        return df[self.FEATURE_COLUMNS]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        features_df: pd.DataFrame,
        labels: pd.Series,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> dict:
        """Train the model and store evaluation metrics.

        Args:
            features_df: DataFrame whose columns are a superset of
                :pyattr:`FEATURE_COLUMNS`.
            labels: Binary series (0 = no default, 1 = default).
            test_size: Fraction held out for evaluation.
            random_state: Random seed for reproducibility.

        Returns:
            Dictionary of evaluation metrics computed on the held-out set.
        """
        X = self._validate_features(features_df)
        y = labels.values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Fit scaler on training data only
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Build and fit model
        self.model = self._build_model()
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True

        # Evaluate on test set
        y_pred = self.model.predict(X_test_scaled)
        y_proba = self.model.predict_proba(X_test_scaled)[:, 1]

        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        self.evaluation_metrics = {
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
            "auc_roc": float(roc_auc_score(y_test, y_proba)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "test_size": len(y_test),
            "train_size": len(y_train),
            "default_rate_train": float(y_train.mean()),
            "default_rate_test": float(y_test.mean()),
        }

        return self.evaluation_metrics

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, features_df: pd.DataFrame) -> dict:
        """Predict default probability, score, risk grade, and confidence.

        Args:
            features_df: DataFrame with one row per applicant.

        Returns:
            Dictionary with arrays:
                - ``default_probability``: float array
                - ``score``: int array (300–900)
                - ``risk_grade``: string array
                - ``confidence``: string array
        """
        if not self.is_trained:
            raise RuntimeError("Model is not trained. Call train() or load_model() first.")

        X = self._validate_features(features_df)
        X_scaled = self.scaler.transform(X)
        proba = self.model.predict_proba(X_scaled)[:, 1]

        # Retrieve data_coverage per row (used for confidence)
        data_coverage = X["data_coverage_score"].values

        scores = np.array([self.get_score_from_probability(p) for p in proba])
        risk_grades = np.array([self.get_risk_grade(p) for p in proba])
        confidences = np.array(
            [self.get_confidence(p, dc) for p, dc in zip(proba, data_coverage)]
        )

        return {
            "default_probability": proba,
            "score": scores,
            "risk_grade": risk_grades,
            "confidence": confidences,
        }

    def predict_single(self, features: dict) -> dict:
        """Predict for a single applicant.

        Args:
            features: Dictionary mapping feature name → value.

        Returns:
            Same structure as :pymeth:`predict` but with scalar values instead
            of arrays.
        """
        df = pd.DataFrame([features])
        result = self.predict(df)
        return {
            "default_probability": float(result["default_probability"][0]),
            "score": int(result["score"][0]),
            "risk_grade": str(result["risk_grade"][0]),
            "confidence": str(result["confidence"][0]),
        }

    # ------------------------------------------------------------------
    # Score / grade helpers
    # ------------------------------------------------------------------

    def get_score_from_probability(self, default_prob: float) -> int:
        """Convert default probability to a 300–900 credit score.

        Lower ``default_prob`` → higher score.
        """
        score = 300 + int((1 - default_prob) * 600)
        return max(300, min(900, score))

    def get_risk_grade(self, default_prob: float) -> str:
        """Map default probability to a risk grade label.

        - ``< 0.10`` → Low
        - ``0.10 – 0.20`` → Medium-Low
        - ``0.20 – 0.35`` → Medium
        - ``0.35 – 0.50`` → Medium-High
        - ``> 0.50`` → High
        """
        if default_prob < 0.10:
            return "Low"
        elif default_prob < 0.20:
            return "Medium-Low"
        elif default_prob < 0.35:
            return "Medium"
        elif default_prob < 0.50:
            return "Medium-High"
        else:
            return "High"

    def get_confidence(self, default_prob: float, data_coverage: float) -> str:
        """Determine prediction confidence based on probability extremity and
        data coverage.

        - **High**: ``prob < 0.15`` or ``prob > 0.60``, and ``coverage > 0.7``
        - **Medium**: ``0.15–0.40`` or ``0.50–0.60``, and ``coverage > 0.5``
        - **Low**: everything else
        """
        extreme = default_prob < 0.15 or default_prob > 0.60
        moderate = (0.15 <= default_prob <= 0.40) or (0.50 <= default_prob <= 0.60)

        if extreme and data_coverage > 0.7:
            return "High"
        elif moderate and data_coverage > 0.5:
            return "Medium"
        else:
            return "Low"

    # ------------------------------------------------------------------
    # Reporting & persistence
    # ------------------------------------------------------------------

    def get_evaluation_report(self) -> dict:
        """Return stored evaluation metrics from the most recent training run."""
        if not self.evaluation_metrics:
            return {"error": "No evaluation metrics available. Train the model first."}
        return self.evaluation_metrics

    def save_model(self, path: str) -> None:
        """Persist the trained model and scaler to disk using *joblib*.

        Creates two files: ``<path>/model.joblib`` and ``<path>/scaler.joblib``.
        """
        if not self.is_trained:
            raise RuntimeError("Cannot save an untrained model.")

        os.makedirs(path, exist_ok=True)
        joblib.dump(self.model, os.path.join(path, "model.joblib"))
        joblib.dump(self.scaler, os.path.join(path, "scaler.joblib"))
        joblib.dump(
            {
                "model_type": self.model_type,
                "model_version": self.model_version,
                "evaluation_metrics": self.evaluation_metrics,
                "feature_columns": self.FEATURE_COLUMNS,
            },
            os.path.join(path, "metadata.joblib"),
        )

    def load_model(self, path: str) -> None:
        """Load a previously saved model, scaler, and metadata from *path*."""
        self.model = joblib.load(os.path.join(path, "model.joblib"))
        self.scaler = joblib.load(os.path.join(path, "scaler.joblib"))

        metadata_path = os.path.join(path, "metadata.joblib")
        if os.path.exists(metadata_path):
            metadata = joblib.load(metadata_path)
            self.model_type = metadata.get("model_type", self.model_type)
            self.model_version = metadata.get("model_version", self.model_version)
            self.evaluation_metrics = metadata.get("evaluation_metrics", {})

        self.is_trained = True
