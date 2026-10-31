"""
Model loader for the FairScore standalone backend.

Loads the best available trained model (highest ROC-AUC) from the models/ directory.
Falls back to training a new model from synthetic data if no model is found.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

log = logging.getLogger("fairscore.model_loader")

MODELS_DIR  = PROJECT_ROOT / "models"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"


class ModelStore:
    """Singleton store for the loaded model pipeline and metadata."""

    pipeline  = None
    metadata: dict = {}
    feature_cols: list[str] = []
    is_loaded: bool = False
    fairness_data: dict = {}
    all_model_summaries: list[dict] = []

    @classmethod
    def load_best(cls) -> bool:
        """Load the highest-ROC-AUC model from METRICS_DIR. Returns True on success."""
        metrics_files = list(METRICS_DIR.glob("*_metrics.json"))
        if not metrics_files:
            log.info("No trained models found – training a fresh model from synthetic data.")
            cls._train_and_load()
            return cls.is_loaded

        # Pick model with highest ROC-AUC
        best: Optional[dict] = None
        for mf in metrics_files:
            try:
                with open(mf) as fh:
                    m = json.load(fh)
                # Normalise metric access
                auc = (
                    m.get("test_metrics", {}).get("roc_auc")
                    or m.get("evaluation", {}).get("roc_auc")
                    or 0.0
                )
                m["_auc"] = auc
                if best is None or auc > best.get("_auc", 0):
                    best = m
            except Exception as exc:
                log.warning("Could not load metadata %s: %s", mf, exc)

        if best is None:
            cls._train_and_load()
            return cls.is_loaded

        model_path = Path(best.get("model_path", ""))
        if not model_path.exists():
            log.warning("Model file %s not found – retraining.", model_path)
            cls._train_and_load()
            return cls.is_loaded

        cls.pipeline     = joblib.load(model_path)
        cls.metadata     = best
        cls.feature_cols = best.get("feature_cols", [])
        cls.is_loaded    = True
        log.info("Loaded model: %s (ROC-AUC=%.4f)", best.get("dataset_name"), best["_auc"])

        # Load fairness data if available
        fairness_files = list(METRICS_DIR.glob("*_fairness.json"))
        if fairness_files:
            try:
                with open(fairness_files[0]) as fh:
                    cls.fairness_data = json.load(fh)
            except Exception:
                pass

        # Collect all model summaries
        cls.all_model_summaries = cls._collect_summaries(metrics_files)
        return True

    @classmethod
    def _collect_summaries(cls, metrics_files: list[Path]) -> list[dict]:
        summaries = []
        for mf in metrics_files:
            try:
                with open(mf) as fh:
                    m = json.load(fh)
                tm = m.get("test_metrics", {})
                summaries.append({
                    "name":       m.get("dataset_name", mf.stem),
                    "path":       m.get("model_path", ""),
                    "roc_auc":    tm.get("roc_auc", 0.0),
                    "f1":         tm.get("f1", 0.0),
                    "accuracy":   tm.get("accuracy", 0.0),
                    "n_rows":     m.get("n_rows", 0),
                    "model_type": m.get("model_type", "unknown"),
                    "timestamp":  m.get("timestamp", ""),
                })
            except Exception:
                pass
        return summaries

    @classmethod
    def _train_and_load(cls) -> None:
        """Train a fresh synthetic model and load it."""
        from scripts.train_model import train_all
        results = train_all(use_xgboost=False, n_synthetic=3_000)
        if results:
            # Reload
            metrics_files = list(METRICS_DIR.glob("*_metrics.json"))
            cls.load_best()


# ===========================================================================
# Prediction helpers
# ===========================================================================

def _prob_to_score(prob: float, min_score: int = 300, max_score: int = 900) -> int:
    """Map default probability [0,1] to a credit score [300,900] (inverse)."""
    score = max_score - (max_score - min_score) * prob
    return int(round(score))


def _score_to_grade(score: int) -> str:
    if score >= 800:
        return "A"
    elif score >= 700:
        return "B"
    elif score >= 600:
        return "C"
    elif score >= 500:
        return "D"
    else:
        return "E"


def _grade_to_recommendation(grade: str, prob: float) -> str:
    if grade in ("A", "B"):
        return "Approve"
    elif grade == "C":
        return "Approve with Conditions"
    elif grade == "D":
        return "Manual Review"
    else:
        return "Decline"


def predict(input_data: dict) -> dict:
    """Run prediction. Returns probability, score, grade, recommendation, factors."""
    if not ModelStore.is_loaded:
        raise RuntimeError("Model not loaded.")

    pipeline     = ModelStore.pipeline
    feature_cols = ModelStore.feature_cols

    # Build input row with fallback median imputation
    row = {}
    for col in feature_cols:
        val = input_data.get(col)
        if val is None:
            # Use median from training metadata if available
            row[col] = np.nan
        else:
            row[col] = val

    X = pd.DataFrame([row], columns=feature_cols)
    prob = float(pipeline.predict_proba(X)[0, 1])
    score = _prob_to_score(prob)
    grade = _score_to_grade(score)
    recommendation = _grade_to_recommendation(grade, prob)

    # Explanation factors
    factors = _compute_factors(pipeline, X, feature_cols)

    return {
        "default_probability": round(float(prob), 4),
        "credit_score":        int(score),
        "risk_grade":          str(grade),
        "recommendation":      str(recommendation),
        "top_positive_factors": factors["positive"],
        "top_negative_factors": factors["negative"],
    }


def _compute_factors(pipeline, X: pd.DataFrame, feature_cols: list[str]) -> dict:
    """Compute positive and negative SHAP/importance factors for a single row."""
    try:
        import shap
        preprocessor = pipeline.named_steps["preprocessor"]
        clf          = pipeline.named_steps["classifier"]
        X_transformed = preprocessor.transform(X)

        try:
            feature_names = list(preprocessor.get_feature_names_out())
        except Exception:
            feature_names = feature_cols

        try:
            explainer   = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X_transformed)
        except Exception:
            return {"positive": [], "negative": []}

        if isinstance(shap_values, list) and len(shap_values) == 2:
            vals = shap_values[1][0]
        else:
            vals = np.array(shap_values).flatten()

        pairs = list(zip(feature_names, vals))
        pairs_sorted = sorted(pairs, key=lambda x: x[1])

        positive = [
            {"feature": str(f), "impact": round(float(-v), 4), "direction": "positive"}
            for f, v in pairs_sorted[:5] if v < 0
        ]
        negative = [
            {"feature": str(f), "impact": round(float(v), 4), "direction": "negative"}
            for f, v in reversed(pairs_sorted[-5:]) if v > 0
        ]
        return {"positive": positive, "negative": negative}

    except Exception as exc:
        log.warning("Factor computation failed: %s", exc)
        # Fallback: use feature importances
        try:
            clf = pipeline.named_steps["classifier"]
            importances = getattr(clf, "feature_importances_", [])
            pairs = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
            positive = [{"feature": str(f), "impact": round(float(v), 4), "direction": "positive"} for f, v in pairs[:3]]
            return {"positive": positive, "negative": []}
        except Exception:
            return {"positive": [], "negative": []}

