"""
Tests for model training, saving, loading, and prediction.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_model import (
    build_pipeline,
    compute_metrics,
    generate_synthetic_dataset,
    save_model,
    train_dataset,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture(scope="module")
def synthetic_record():
    return generate_synthetic_dataset(n_samples=500)


@pytest.fixture(scope="module")
def trained_artifact(synthetic_record, tmp_path_factory):
    """Train a model and return (pipeline, metadata)."""
    from sklearn.model_selection import train_test_split
    import joblib
    import json

    record = synthetic_record
    df     = record["df"]
    target = record["target_col"]
    feats  = record["feature_cols"]

    X = df[feats]
    y = df[target]

    pipeline = build_pipeline(df, feats, use_xgboost=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipeline.fit(X_train, y_train)

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= 0.5).astype(int)
    metrics = compute_metrics(y_test.values, y_pred, y_proba)

    # Save into a temp models dir
    tmp_models = tmp_path_factory.mktemp("models")
    tmp_metrics = tmp_path_factory.mktemp("metrics")

    import os
    # Patch MODELS_DIR and METRICS_DIR for test
    import scripts.train_model as tm
    orig_models  = tm.MODELS_DIR
    orig_metrics = tm.METRICS_DIR
    tm.MODELS_DIR  = tmp_models
    tm.METRICS_DIR = tmp_metrics

    metadata = {
        "dataset_name": "test_synthetic",
        "file_path": "synthetic",
        "target_col": target,
        "feature_cols": feats,
        "n_rows": len(df),
        "n_cols": len(feats),
        "default_rate": float(y.mean()),
        "model_version": "1.0.0",
        "model_type": "RandomForestClassifier",
        "timestamp": "2026-01-01T00:00:00Z",
        "test_metrics": metrics,
        "cv_metrics": {},
    }
    model_path = save_model(pipeline, metadata, "test_synthetic")
    metadata["model_path"] = str(model_path)

    tm.MODELS_DIR  = orig_models
    tm.METRICS_DIR = orig_metrics

    return pipeline, metadata, model_path


# ===========================================================================
# Tests: Synthetic Data Generation
# ===========================================================================

class TestSyntheticDataGeneration:
    def test_generates_correct_shape(self, synthetic_record):
        r = synthetic_record
        assert r["n_rows"] == 500
        assert r["n_cols"] >= 10

    def test_target_is_binary(self, synthetic_record):
        y = synthetic_record["df"][synthetic_record["target_col"]]
        assert set(y.unique()) <= {0, 1}

    def test_has_alternative_features(self, synthetic_record):
        feats = set(synthetic_record["feature_cols"])
        expected = {"rent_payment_score", "utility_payment_score", "cashflow_consistency"}
        assert expected <= feats

    def test_default_rate_reasonable(self, synthetic_record):
        rate = synthetic_record["default_rate"]
        assert 0.05 < rate < 0.60, f"Default rate {rate:.2%} is out of expected range"

    def test_no_protected_attributes_in_features(self, synthetic_record):
        feats = set(f.lower() for f in synthetic_record["feature_cols"])
        protected = {"gender", "urban_rural", "age_group", "occupation"}
        overlap = feats & protected
        assert not overlap, f"Protected attributes in features: {overlap}"


# ===========================================================================
# Tests: Pipeline Building
# ===========================================================================

class TestPipelineBuilding:
    def test_pipeline_has_preprocessor_and_classifier(self, synthetic_record):
        df    = synthetic_record["df"]
        feats = synthetic_record["feature_cols"]
        pipe  = build_pipeline(df, feats, use_xgboost=False)
        assert "preprocessor" in pipe.named_steps
        assert "classifier" in pipe.named_steps

    def test_pipeline_with_xgboost_fallback(self, synthetic_record):
        """XGBoost is installed; pipeline should use it."""
        df    = synthetic_record["df"]
        feats = synthetic_record["feature_cols"]
        pipe  = build_pipeline(df, feats, use_xgboost=True)
        assert "classifier" in pipe.named_steps


# ===========================================================================
# Tests: Model Training
# ===========================================================================

class TestModelTraining:
    def test_train_dataset_returns_metadata(self, synthetic_record, tmp_path):
        """train_dataset should complete and return a metadata dict."""
        import scripts.train_model as tm
        orig_models  = tm.MODELS_DIR
        orig_metrics = tm.METRICS_DIR
        tm.MODELS_DIR  = tmp_path / "models"
        tm.METRICS_DIR = tmp_path / "metrics"
        tm.MODELS_DIR.mkdir()
        tm.METRICS_DIR.mkdir()

        result = train_dataset(synthetic_record, use_xgboost=False)
        tm.MODELS_DIR  = orig_models
        tm.METRICS_DIR = orig_metrics

        assert result is not None
        assert "test_metrics" in result
        assert "model_path" in result

    def test_roc_auc_above_chance(self, synthetic_record, tmp_path):
        import scripts.train_model as tm
        orig_models  = tm.MODELS_DIR
        orig_metrics = tm.METRICS_DIR
        tm.MODELS_DIR  = tmp_path / "models"
        tm.METRICS_DIR = tmp_path / "metrics"
        tm.MODELS_DIR.mkdir()
        tm.METRICS_DIR.mkdir()

        result = train_dataset(synthetic_record, use_xgboost=False)
        tm.MODELS_DIR  = orig_models
        tm.METRICS_DIR = orig_metrics

        assert result["test_metrics"]["roc_auc"] > 0.55, "Model should beat random chance"


# ===========================================================================
# Tests: Model Saving and Loading
# ===========================================================================

class TestModelSaveLoad:
    def test_model_file_exists(self, trained_artifact):
        _, _, model_path = trained_artifact
        assert model_path.exists()

    def test_saved_model_loads(self, trained_artifact):
        import joblib
        _, _, model_path = trained_artifact
        pipeline = joblib.load(model_path)
        assert pipeline is not None

    def test_loaded_model_predicts(self, trained_artifact, synthetic_record):
        import joblib
        _, metadata, model_path = trained_artifact
        pipeline     = joblib.load(model_path)
        feature_cols = metadata["feature_cols"]
        df           = synthetic_record["df"]
        X_sample     = df[feature_cols].head(5)
        proba = pipeline.predict_proba(X_sample)
        assert proba.shape == (5, 2)

    def test_probabilities_in_range(self, trained_artifact, synthetic_record):
        import joblib
        _, metadata, model_path = trained_artifact
        pipeline     = joblib.load(model_path)
        feature_cols = metadata["feature_cols"]
        df           = synthetic_record["df"]
        X_sample     = df[feature_cols].head(20)
        proba = pipeline.predict_proba(X_sample)[:, 1]
        assert np.all(proba >= 0) and np.all(proba <= 1)


# ===========================================================================
# Tests: Compute Metrics
# ===========================================================================

class TestComputeMetrics:
    def test_metrics_keys(self):
        y_true  = np.array([0, 0, 1, 1, 0, 1])
        y_pred  = np.array([0, 1, 1, 1, 0, 0])
        y_proba = np.array([0.1, 0.6, 0.8, 0.9, 0.2, 0.4])
        m = compute_metrics(y_true, y_pred, y_proba)
        for key in ["roc_auc", "f1", "accuracy", "precision", "recall", "confusion_matrix"]:
            assert key in m

    def test_approval_rates_in_metrics(self):
        y_true  = np.array([0, 0, 1, 1, 0, 1])
        y_pred  = np.array([0, 1, 1, 1, 0, 0])
        y_proba = np.array([0.1, 0.6, 0.8, 0.9, 0.2, 0.4])
        m = compute_metrics(y_true, y_pred, y_proba)
        assert "approval_rate_at_0.5" in m

    def test_roc_auc_between_0_and_1(self):
        y_true  = np.array([0, 0, 1, 1])
        y_pred  = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.8, 0.9])
        m = compute_metrics(y_true, y_pred, y_proba)
        assert 0.0 <= m["roc_auc"] <= 1.0

