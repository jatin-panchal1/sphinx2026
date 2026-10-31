"""
Model training script for FairScore Alternative Credit Scoring.

Loads datasets via data_loader, builds sklearn pipelines (RandomForest + optional XGBoost),
trains, evaluates, saves model artefacts, and writes JSON metadata.
"""

import io
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Fix Windows console UTF-8 encoding for tqdm / logging symbols
if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data_loader import load_all_datasets

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("fairscore.train_model")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
METRICS_DIR = OUTPUTS_DIR / "metrics"
LOGS_DIR    = OUTPUTS_DIR / "logs"
MODEL_VERSION = "1.0.0"
TEST_SIZE = 0.20
RANDOM_STATE = 42

PROTECTED_ATTRIBUTES = {"gender", "sex", "race", "ethnicity", "region", "urban_rural", "age_group"}


# ===========================================================================
# Synthetic Data Generation
# ===========================================================================

def generate_synthetic_dataset(n_samples: int = 5_000) -> dict:
    """Generate a synthetic alternative credit-scoring dataset.

    Uses alternative financial signals:
    - rent payment behaviour
    - utility payment behaviour
    - mobile recharge behaviour
    - cash-flow consistency
    - savings behaviour
    - verified cash payments
    - loan repayment history (alternative lenders)
    """
    log.info("Generating synthetic alternative credit dataset (%d samples)…", n_samples)
    rng = np.random.default_rng(RANDOM_STATE)

    n = n_samples

    # ---- Demographic (fairness audit only – NOT used as model features) ----
    occupation = rng.choice(
        ["Gig Worker", "Small Shopkeeper", "Informal Worker", "Student", "Freelancer",
         "Daily Wage", "Home-Based Business", "Seasonal Worker"],
        size=n,
        p=[0.18, 0.15, 0.20, 0.12, 0.10, 0.10, 0.08, 0.07],
    )
    urban_rural = rng.choice(["Urban", "Semi-Urban", "Rural"], size=n, p=[0.45, 0.30, 0.25])
    age_group   = rng.choice(["18-25", "26-35", "36-45", "46-60"], size=n, p=[0.20, 0.35, 0.28, 0.17])

    # ---- Alternative credit signals ----------------------------------------
    monthly_income    = rng.normal(25_000, 10_000, n).clip(3_000, 120_000)
    income_volatility = rng.beta(2, 5, n)          # 0=stable, 1=volatile

    rent_payment_score      = rng.beta(5, 2, n)    # 0-1, higher = reliable
    utility_payment_score   = rng.beta(5, 2, n)
    mobile_recharge_regularity = rng.beta(4, 2, n)
    cashflow_consistency    = rng.beta(4, 2, n)
    savings_rate            = rng.beta(2, 5, n).clip(0, 0.4)
    verified_cash_payments  = rng.integers(0, 25, n)

    months_in_current_job   = rng.integers(1, 120, n)
    num_dependants          = rng.integers(0, 6, n)
    loan_amount_requested   = rng.normal(50_000, 25_000, n).clip(5_000, 500_000)

    monthly_expenses        = monthly_income * rng.uniform(0.4, 0.9, n)
    expense_to_income_ratio = (monthly_expenses / monthly_income).clip(0.1, 1.0)

    prior_defaults          = rng.integers(0, 3, n, endpoint=True)
    prior_defaults          = np.where(rng.random(n) < 0.75, 0, prior_defaults)  # 75% have zero

    loan_to_income_ratio    = (loan_amount_requested / (monthly_income * 12)).clip(0.1, 10)

    # ---- Derive credit score signal (strong enough for realistic AUC ~0.80) -
    repayment_score = (
        0.30 * rent_payment_score
        + 0.25 * utility_payment_score
        + 0.15 * mobile_recharge_regularity
        + 0.20 * cashflow_consistency
        + 0.08 * savings_rate * 5
        + 0.05 * (verified_cash_payments / 25)
        + 0.05 * (months_in_current_job / 120)
        - 0.20 * income_volatility
        - 0.12 * expense_to_income_ratio
        - 0.25 * (prior_defaults / 2)
        - 0.08 * (loan_to_income_ratio / 10)
        + rng.normal(0, 0.03, n)   # Reduced noise for cleaner signal
    ).clip(0, 1)

    # ---- Binary target (default within 90 days) ----------------------------
    # Steeper slope -> sharper decision boundary -> higher discriminability
    default_prob_logistic = 1 / (1 + np.exp(8 * (repayment_score - 0.45)))
    default_prob_logistic = default_prob_logistic.clip(0.02, 0.98)
    is_default = rng.binomial(1, default_prob_logistic).astype(int)

    df = pd.DataFrame({
        # Model features (alternative signals only)
        "monthly_income":           monthly_income.round(2),
        "income_volatility":        income_volatility.round(4),
        "rent_payment_score":       rent_payment_score.round(4),
        "utility_payment_score":    utility_payment_score.round(4),
        "mobile_recharge_regularity": mobile_recharge_regularity.round(4),
        "cashflow_consistency":     cashflow_consistency.round(4),
        "savings_rate":             savings_rate.round(4),
        "verified_cash_payments":   verified_cash_payments,
        "months_in_current_job":    months_in_current_job,
        "num_dependants":           num_dependants,
        "expense_to_income_ratio":  expense_to_income_ratio.round(4),
        "prior_defaults":           prior_defaults,
        "loan_amount_requested":    loan_amount_requested.round(2),
        "loan_to_income_ratio":     loan_to_income_ratio.round(4),

        # Protected attributes (fairness audit only – excluded from model features)
        "occupation":   occupation,
        "urban_rural":  urban_rural,
        "age_group":    age_group,

        # Target
        "is_default": is_default,
    })

    feature_cols = [
        "monthly_income", "income_volatility", "rent_payment_score",
        "utility_payment_score", "mobile_recharge_regularity", "cashflow_consistency",
        "savings_rate", "verified_cash_payments", "months_in_current_job",
        "num_dependants", "expense_to_income_ratio", "prior_defaults",
        "loan_amount_requested", "loan_to_income_ratio",
    ]

    default_rate = float(is_default.mean())
    log.info("Synthetic dataset generated | rows=%d default_rate=%.2f%%", n, 100 * default_rate)

    return {
        "path": "synthetic",
        "name": "synthetic_alt_credit",
        "df": df,
        "target_col": "is_default",
        "feature_cols": feature_cols,
        "n_rows": n,
        "n_cols": len(feature_cols),
        "default_rate": default_rate,
    }


# ===========================================================================
# Pipeline Construction
# ===========================================================================

def _get_column_types(df: pd.DataFrame, feature_cols: list[str]) -> tuple[list, list]:
    """Split feature columns into numeric and categorical."""
    num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
    return num_cols, cat_cols


def build_pipeline(df: pd.DataFrame, feature_cols: list[str], use_xgboost: bool = False) -> Pipeline:
    """Build a full sklearn preprocessing + model pipeline."""
    num_cols, cat_cols = _get_column_types(df, feature_cols)

    num_transformer = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler()),
    ])

    cat_transformer = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    transformers = []
    if num_cols:
        transformers.append(("num", num_transformer, num_cols))
    if cat_cols:
        transformers.append(("cat", cat_transformer, cat_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    if use_xgboost:
        try:
            from xgboost import XGBClassifier
            clf = XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=1,
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                verbosity=0,
            )
            model_name = "XGBClassifier"
            log.info("Using XGBoost classifier.")
        except ImportError:
            log.warning("XGBoost not available – falling back to RandomForest.")
            use_xgboost = False

    if not use_xgboost:
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model_name = "RandomForestClassifier"

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   clf),
    ])

    pipeline.model_name = model_name
    return pipeline


# ===========================================================================
# Evaluation Metrics
# ===========================================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    """Return a comprehensive metrics dictionary."""
    cm = confusion_matrix(y_true, y_pred)
    metrics = {
        "roc_auc":           float(roc_auc_score(y_true, y_proba)),
        "avg_precision":     float(average_precision_score(y_true, y_proba)),
        "accuracy":          float(accuracy_score(y_true, y_pred)),
        "precision":         float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":            float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":                float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix":  cm.tolist(),
        "predicted_default_rate": float(y_pred.mean()),
        "actual_default_rate":    float(y_true.mean()),
    }
    # Approval rates at different thresholds
    for thr in [0.35, 0.50, 0.65]:
        approved = (y_proba < thr).mean()
        metrics[f"approval_rate_at_{thr}"] = float(approved)

    return metrics


def cross_validate_model(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    """Run stratified 5-fold cross validation and return mean/std metrics."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    auc_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    f1_scores  = cross_val_score(pipeline, X, y, cv=cv, scoring="f1", n_jobs=-1)
    return {
        "cv_roc_auc_mean": float(auc_scores.mean()),
        "cv_roc_auc_std":  float(auc_scores.std()),
        "cv_f1_mean":      float(f1_scores.mean()),
        "cv_f1_std":       float(f1_scores.std()),
    }


# ===========================================================================
# Model Saving
# ===========================================================================

def save_model(pipeline: Pipeline, metadata: dict, dataset_name: str) -> Path:
    """Save pipeline with joblib and write JSON metadata."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / f"{dataset_name}_model.joblib"
    meta_path  = METRICS_DIR / f"{dataset_name}_metrics.json"

    joblib.dump(pipeline, model_path, compress=3)
    with open(meta_path, "w") as fh:
        json.dump(metadata, fh, indent=2, default=str)

    log.info("Model saved: %s", model_path)
    log.info("Metrics saved: %s", meta_path)
    return model_path


# ===========================================================================
# Training Loop
# ===========================================================================

def train_dataset(record: dict, use_xgboost: bool = False) -> Optional[dict]:
    """Train, evaluate, cross-validate, and save a model for one dataset."""
    name       = record["name"]
    df         = record["df"]
    target_col = record["target_col"]
    # Exclude protected attributes from features
    feature_cols = [
        c for c in record["feature_cols"]
        if c.lower() not in PROTECTED_ATTRIBUTES
    ]

    if not feature_cols:
        log.warning("[%s] No valid feature columns after removing protected attributes.", name)
        return None

    X = df[feature_cols]
    y = df[target_col]

    log.info("[%s] Splitting train/test (%.0f/%.0f)…", name, (1 - TEST_SIZE) * 100, TEST_SIZE * 100)
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
        )
    except ValueError as exc:
        log.warning("[%s] Train/test split failed: %s", name, exc)
        return None

    log.info("[%s] Building pipeline…", name)
    pipeline = build_pipeline(X_train, feature_cols, use_xgboost=use_xgboost)

    log.info("[%s] Training %s…", name, getattr(pipeline, "model_name", "model"))
    pipeline.fit(X_train, y_train)

    log.info("[%s] Evaluating…", name)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= 0.5).astype(int)
    test_metrics = compute_metrics(y_test.values, y_pred, y_proba)

    log.info("[%s] Cross-validating…", name)
    cv_metrics = cross_validate_model(pipeline, X, y)

    metadata = {
        "dataset_name":   name,
        "file_path":      record["path"],
        "target_col":     target_col,
        "feature_cols":   feature_cols,
        "n_rows":         record["n_rows"],
        "n_cols":         record["n_cols"],
        "default_rate":   record["default_rate"],
        "model_version":  MODEL_VERSION,
        "model_type":     getattr(pipeline, "model_name", "RandomForestClassifier"),
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "test_metrics":   test_metrics,
        "cv_metrics":     cv_metrics,
    }

    clean_name = name.replace(" ", "_").lower()
    metadata["dataset_name"] = clean_name

    model_path = save_model(pipeline, metadata, clean_name)
    metadata["model_path"] = str(model_path)
    # Re-save metadata now that model_path is known
    meta_path = METRICS_DIR / f"{clean_name}_metrics.json"
    with open(meta_path, "w") as fh:
        json.dump(metadata, fh, indent=2, default=str)

    log.info(
        "[%s] ✓ ROC-AUC=%.4f  F1=%.4f  Accuracy=%.4f  Default-rate=%.2f%%",
        name,
        test_metrics["roc_auc"],
        test_metrics["f1"],
        test_metrics["accuracy"],
        100 * record["default_rate"],
    )

    return metadata


def train_all(use_xgboost: bool = False, n_synthetic: int = 5_000) -> list[dict]:
    """Main entry point: load real datasets (if any), fallback to synthetic, train all."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("FairScore Model Training Pipeline")
    log.info("=" * 60)

    datasets = load_all_datasets(
        data_dir=str(PROJECT_ROOT / "data"),
        extracted_dir=str(PROJECT_ROOT / "data" / "extracted"),
    )

    # Always include the alternative credit dataset tailored for credit-invisible applicants
    synth_record = generate_synthetic_dataset(n_samples=n_synthetic)
    if not any(d["name"] == synth_record["name"] for d in datasets):
        datasets.append(synth_record)

    results: list[dict] = []
    for record in tqdm(datasets, desc="Training models", unit="dataset"):
        result = train_dataset(record, use_xgboost=use_xgboost)
        if result:
            results.append(result)

    # Save summary log
    summary_path = LOGS_DIR / "training_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    log.info("Training summary written to %s", summary_path)

    log.info("=" * 60)
    log.info("Training complete. %d model(s) saved.", len(results))
    for r in results:
        log.info(
            "  %s | ROC-AUC=%.4f | F1=%.4f | CV-AUC=%.4f±%.4f",
            r["dataset_name"],
            r["test_metrics"]["roc_auc"],
            r["test_metrics"]["f1"],
            r["cv_metrics"]["cv_roc_auc_mean"],
            r["cv_metrics"]["cv_roc_auc_std"],
        )
    log.info("=" * 60)

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="FairScore model trainer")
    parser.add_argument("--xgboost", action="store_true", help="Use XGBoost instead of RandomForest")
    parser.add_argument("--synthetic-rows", type=int, default=5_000, help="Synthetic dataset size")
    args = parser.parse_args()

    results = train_all(use_xgboost=args.xgboost, n_synthetic=args.synthetic_rows)
    print(f"\nTraining complete. {len(results)} model(s) saved to models/")
