"""
Model evaluation module for FairScore Alternative Credit Scoring.

Loads a trained model and produces a comprehensive evaluation report including:
- ROC-AUC, PR-AUC, accuracy, precision, recall, F1
- Confusion matrix
- Approval rates at multiple thresholds
- SHAP global feature importance
- Calibration diagnostics
- Written JSON + markdown reports
"""

import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("fairscore.evaluate_model")

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
METRICS_DIR = OUTPUTS_DIR / "metrics"
REPORTS_DIR = OUTPUTS_DIR / "reports"
MODELS_DIR  = PROJECT_ROOT / "models"
RANDOM_STATE = 42
TEST_SIZE = 0.20
PROTECTED_ATTRIBUTES = {"gender", "sex", "race", "ethnicity", "region", "urban_rural", "age_group"}


# ===========================================================================
# SHAP Global Importance
# ===========================================================================

def compute_shap_importance(pipeline, X_test: pd.DataFrame, max_rows: int = 80) -> list[dict]:
    """Compute global SHAP feature importance. Returns sorted list of {feature, importance}."""
    try:
        import shap

        preprocessor = pipeline.named_steps["preprocessor"]
        clf          = pipeline.named_steps["classifier"]

        X_transformed = preprocessor.transform(X_test.head(max_rows))

        # Get feature names after ColumnTransformer
        try:
            feature_names = [str(f) for f in preprocessor.get_feature_names_out()]
        except Exception:
            feature_names = [f"f{i}" for i in range(X_transformed.shape[1])]

        try:
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X_transformed)
        except Exception:
            explainer = shap.KernelExplainer(
                clf.predict_proba,
                shap.sample(X_transformed, min(30, len(X_transformed))),
            )
            shap_values = explainer.shap_values(X_transformed, nsamples=50)

        # For binary classification shap_values may be list of 2
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_vals = shap_values[1]
        else:
            shap_vals = np.array(shap_values)

        mean_abs = np.abs(shap_vals).mean(axis=0)
        if mean_abs.ndim > 1:
            mean_abs = mean_abs.mean(axis=0)

        importance_pairs = sorted(
            zip(feature_names, [float(v) for v in mean_abs]),
            key=lambda x: x[1],
            reverse=True,
        )
        return [{"feature": str(f), "importance": float(v)} for f, v in importance_pairs]

    except Exception as exc:
        log.warning("SHAP computation failed: %s – using fallback feature importance.", exc)
        return []


def compute_rf_importance(pipeline, feature_cols: list[str]) -> list[dict]:
    """Fallback: use RandomForest / XGBoost built-in feature importance."""
    try:
        clf = pipeline.named_steps["classifier"]
        preprocessor = pipeline.named_steps["preprocessor"]
        try:
            feature_names = list(preprocessor.get_feature_names_out())
        except Exception:
            feature_names = feature_cols

        importances = getattr(clf, "feature_importances_", None)
        if importances is None:
            return []

        pairs = sorted(
            zip(feature_names, importances.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        return [{"feature": f, "importance": float(v)} for f, v in pairs]
    except Exception as exc:
        log.warning("Feature importance fallback failed: %s", exc)
        return []


# ===========================================================================
# Full Evaluation
# ===========================================================================

def evaluate_model(metadata_path: Path) -> dict:
    """Evaluate a trained model using the metadata + reloading the original dataset."""
    with open(metadata_path) as fh:
        metadata = json.load(fh)

    model_path   = Path(metadata["model_path"])
    feature_cols = metadata["feature_cols"]
    target_col   = metadata["target_col"]
    dataset_name = metadata["dataset_name"]

    log.info("Evaluating model: %s", dataset_name)

    pipeline = joblib.load(model_path)

    # Recreate test split
    from scripts.train_model import generate_synthetic_dataset
    if metadata["file_path"] == "synthetic":
        record = generate_synthetic_dataset(n_samples=metadata["n_rows"])
    else:
        from scripts.data_loader import load_file, build_dataset_record
        fp = Path(metadata["file_path"])
        df_raw = load_file(fp)
        if df_raw is None:
            log.error("Cannot reload dataset %s – skipping evaluation.", fp.name)
            return {}
        record = build_dataset_record(fp, df_raw)
        if record is None:
            return {}

    df = record["df"]
    # Keep only feature_cols that are actually in df (guard against drift)
    available_features = [c for c in feature_cols if c in df.columns]
    X = df[available_features]
    y = df[target_col]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= 0.5).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    metrics = {
        "roc_auc":           float(roc_auc_score(y_test, y_proba)),
        "avg_precision":     float(average_precision_score(y_test, y_proba)),
        "accuracy":          float(accuracy_score(y_test, y_pred)),
        "precision":         float(precision_score(y_test, y_pred, zero_division=0)),
        "recall":            float(recall_score(y_test, y_pred, zero_division=0)),
        "f1":                float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix":  cm.tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "predicted_default_rate": float(y_pred.mean()),
        "actual_default_rate":    float(y_test.mean()),
    }

    for thr in [0.35, 0.50, 0.65]:
        metrics[f"approval_rate_at_{thr}"] = float((y_proba < thr).mean())

    # Calibration
    try:
        frac_pos, mean_pred = calibration_curve(y_test, y_proba, n_bins=10, strategy="uniform")
        metrics["calibration"] = {
            "mean_predicted": mean_pred.tolist(),
            "fraction_positive": frac_pos.tolist(),
        }
    except Exception:
        pass

    # Feature importance
    shap_importance = compute_shap_importance(pipeline, X_test)
    rf_importance   = compute_rf_importance(pipeline, available_features)
    feature_importance = shap_importance if shap_importance else rf_importance
    metrics["feature_importance"] = feature_importance[:20]

    # Cross-validation from metadata (already computed during training)
    metrics["cv_metrics"] = metadata.get("cv_metrics", {})

    result = {**metadata, "evaluation": metrics}

    # Save updated metrics
    out_path = METRICS_DIR / f"{dataset_name}_evaluation.json"
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2, default=str)
    log.info("Evaluation saved to %s", out_path)

    log.info(
        "[%s] ROC-AUC=%.4f | PR-AUC=%.4f | F1=%.4f | Accuracy=%.4f",
        dataset_name,
        metrics["roc_auc"],
        metrics["avg_precision"],
        metrics["f1"],
        metrics["accuracy"],
    )
    return result


def evaluate_all_models() -> list[dict]:
    """Evaluate every model whose metrics file exists in METRICS_DIR."""
    metrics_files = list(METRICS_DIR.glob("*_metrics.json"))
    if not metrics_files:
        log.warning("No model metrics files found in %s – run train_model.py first.", METRICS_DIR)
        return []

    results = []
    for mf in metrics_files:
        r = evaluate_model(mf)
        if r:
            results.append(r)
    return results


# ===========================================================================
# Markdown Report
# ===========================================================================

def write_evaluation_report(results: list[dict], report_path: Path) -> None:
    """Write a markdown evaluation report."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# FairScore Model Evaluation Report",
        "",
        f"Generated: {__import__('datetime').datetime.now().isoformat()}",
        "",
        "---",
        "",
    ]

    for r in results:
        ev = r.get("evaluation", r.get("test_metrics", {}))
        name = r.get("dataset_name", "unknown")
        lines += [
            f"## Dataset: `{name}`",
            "",
            f"- **File:** `{r.get('file_path', 'N/A')}`",
            f"- **Rows:** {r.get('n_rows', 'N/A'):,}",
            f"- **Features:** {r.get('n_cols', 'N/A')}",
            f"- **Target:** `{r.get('target_col', 'N/A')}`",
            f"- **Model:** {r.get('model_type', 'N/A')}",
            f"- **Actual Default Rate:** {ev.get('actual_default_rate', 0):.2%}",
            "",
            "### Test-Set Performance",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| ROC-AUC | {ev.get('roc_auc', 0):.4f} |",
            f"| PR-AUC | {ev.get('avg_precision', 0):.4f} |",
            f"| Accuracy | {ev.get('accuracy', 0):.4f} |",
            f"| Precision | {ev.get('precision', 0):.4f} |",
            f"| Recall | {ev.get('recall', 0):.4f} |",
            f"| F1 Score | {ev.get('f1', 0):.4f} |",
            "",
            "### Approval Rates",
            "",
            f"| Threshold | Approval Rate |",
            f"|-----------|---------------|",
        ]
        for thr in [0.35, 0.50, 0.65]:
            ar = ev.get(f"approval_rate_at_{thr}", 0)
            lines.append(f"| {thr} | {ar:.2%} |")

        cv = ev.get("cv_metrics", r.get("cv_metrics", {}))
        if cv:
            lines += [
                "",
                "### Cross-Validation (5-Fold Stratified)",
                "",
                f"| Metric | Mean | Std |",
                f"|--------|------|-----|",
                f"| ROC-AUC | {cv.get('cv_roc_auc_mean', 0):.4f} | {cv.get('cv_roc_auc_std', 0):.4f} |",
                f"| F1 | {cv.get('cv_f1_mean', 0):.4f} | {cv.get('cv_f1_std', 0):.4f} |",
            ]

        fi = ev.get("feature_importance", [])
        if fi:
            lines += [
                "",
                "### Top Feature Importances",
                "",
                "| Feature | Importance |",
                "|---------|------------|",
            ]
            for item in fi[:10]:
                lines.append(f"| {item['feature']} | {item['importance']:.4f} |")

        lines += ["", "---", ""]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Evaluation report written to %s", report_path)


if __name__ == "__main__":
    results = evaluate_all_models()
    if results:
        write_evaluation_report(results, REPORTS_DIR / "model_evaluation.md")
        print(f"\nEvaluation complete. {len(results)} model(s) evaluated.")
    else:
        print("No models to evaluate. Run train_model.py first.")

