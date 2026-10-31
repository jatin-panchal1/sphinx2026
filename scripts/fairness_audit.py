"""
Fairness audit module for FairScore Alternative Credit Scoring.

Computes approval-rate parity, disparate impact, false rejection rates,
and false approval rates across protected demographic groups.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("fairscore.fairness_audit")

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
METRICS_DIR = OUTPUTS_DIR / "metrics"
REPORTS_DIR = OUTPUTS_DIR / "reports"

PROTECTED_ATTRIBUTES = [
    "occupation", "urban_rural", "age_group", "gender", "region",
    "sex", "famges", "alter", "gastarb", "marriage",
]
DISPARATE_IMPACT_THRESHOLD = 0.80
APPROVAL_RATE_GAP_THRESHOLD = 0.10


# ===========================================================================
# Core Fairness Metrics
# ===========================================================================

def compute_group_metrics(
    df: pd.DataFrame,
    group_col: str,
    y_true_col: str,
    y_pred_col: str,
    y_proba_col: str,
) -> pd.DataFrame:
    """Compute per-group fairness metrics."""
    rows = []
    groups = df[group_col].dropna().unique()

    for g in sorted(groups):
        mask = df[group_col] == g
        sub  = df[mask]
        if len(sub) < 10:
            continue

        y_true  = sub[y_true_col].values
        y_pred  = sub[y_pred_col].values
        y_proba = sub[y_proba_col].values

        n_total   = len(sub)
        n_approved = int((y_pred == 0).sum())   # approved = predicted non-default
        n_denied   = int((y_pred == 1).sum())

        # True positives / negatives / etc.
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())

        # False rejection rate = FP / (FP + TN) – good applicants wrongly denied
        frr = fp / max(fp + tn, 1)
        # False approval rate = FN / (FN + TP) – bad applicants wrongly approved
        far = fn / max(fn + tp, 1)

        approval_rate = n_approved / n_total
        avg_proba     = float(y_proba.mean())
        actual_default_rate = float(y_true.mean())

        rows.append({
            "group_col": group_col,
            "group": str(g),
            "n_total": n_total,
            "n_approved": n_approved,
            "n_denied": n_denied,
            "approval_rate": approval_rate,
            "avg_default_probability": avg_proba,
            "actual_default_rate": actual_default_rate,
            "false_rejection_rate": frr,
            "false_approval_rate": far,
        })

    return pd.DataFrame(rows)


def disparate_impact_ratio(group_df: pd.DataFrame) -> float:
    """Compute disparate impact: min approval rate / max approval rate."""
    if group_df.empty or group_df["approval_rate"].max() == 0:
        return 1.0
    return float(group_df["approval_rate"].min() / group_df["approval_rate"].max())


def approval_rate_gap(group_df: pd.DataFrame) -> float:
    """Max minus min approval rate across groups."""
    if group_df.empty:
        return 0.0
    return float(group_df["approval_rate"].max() - group_df["approval_rate"].min())


# ===========================================================================
# Audit Entry Point
# ===========================================================================

def run_fairness_audit(
    df: pd.DataFrame,
    y_proba: np.ndarray,
    y_true: np.ndarray,
    decision_threshold: float = 0.50,
) -> dict:
    """
    Run a fairness audit over available protected attributes in *df*.

    Args:
        df: DataFrame containing protected attribute columns.
        y_proba: Predicted default probabilities (floats 0–1).
        y_true: True binary labels (0 = good, 1 = default).
        decision_threshold: Probability threshold for approval/denial.

    Returns:
        A dictionary with per-attribute group metrics and audit flags.
    """
    df = df.copy()
    df["_y_true"]  = y_true
    df["_y_pred"]  = (y_proba >= decision_threshold).astype(int)
    df["_y_proba"] = y_proba

    protected_lower = {a.lower(): a for a in PROTECTED_ATTRIBUTES}
    available_attrs = [c for c in df.columns if c.lower() in protected_lower]
    if not available_attrs:
        log.warning("No protected attributes found in dataset for fairness audit.")
        return {"attributes": {}, "alerts": []}

    audit: dict = {"attributes": {}, "alerts": []}

    for attr in available_attrs:
        log.info("Auditing attribute: %s", attr)
        group_df = compute_group_metrics(df, attr, "_y_true", "_y_pred", "_y_proba")
        if group_df.empty:
            continue

        di = disparate_impact_ratio(group_df)
        gap = approval_rate_gap(group_df)

        # Alerts
        if di < DISPARATE_IMPACT_THRESHOLD:
            alert = {
                "severity": "HIGH",
                "attribute": attr,
                "metric": "disparate_impact_ratio",
                "value": round(di, 4),
                "threshold": DISPARATE_IMPACT_THRESHOLD,
                "message": (
                    f"Disparate impact ratio {di:.3f} < {DISPARATE_IMPACT_THRESHOLD} "
                    f"for attribute '{attr}'. Possible systemic bias."
                ),
            }
            audit["alerts"].append(alert)
            log.warning("FAIRNESS ALERT: %s", alert["message"])

        if gap > APPROVAL_RATE_GAP_THRESHOLD:
            best_group  = group_df.loc[group_df["approval_rate"].idxmax(), "group"]
            worst_group = group_df.loc[group_df["approval_rate"].idxmin(), "group"]
            alert = {
                "severity": "MEDIUM" if gap < 0.20 else "HIGH",
                "attribute": attr,
                "metric": "approval_rate_gap",
                "value": round(gap, 4),
                "threshold": APPROVAL_RATE_GAP_THRESHOLD,
                "message": (
                    f"Approval rate gap of {gap:.1%} between '{best_group}' "
                    f"({group_df['approval_rate'].max():.1%}) and "
                    f"'{worst_group}' ({group_df['approval_rate'].min():.1%}) "
                    f"for attribute '{attr}'."
                ),
            }
            audit["alerts"].append(alert)
            log.warning("FAIRNESS ALERT: %s", alert["message"])

        audit["attributes"][attr] = {
            "groups": group_df.to_dict(orient="records"),
            "disparate_impact_ratio": di,
            "approval_rate_gap": gap,
        }

    return audit


# ===========================================================================
# Standalone runner
# ===========================================================================

def audit_trained_model(metadata_path: Path) -> Optional[dict]:
    """Load a trained model's metadata and run the fairness audit on synthetic data."""
    import joblib
    from sklearn.model_selection import train_test_split

    with open(metadata_path) as fh:
        metadata = json.load(fh)

    model_path   = Path(metadata["model_path"])
    feature_cols = metadata["feature_cols"]
    target_col   = metadata["target_col"]
    dataset_name = metadata["dataset_name"]

    pipeline = joblib.load(model_path)

    # Load or regenerate dataset
    from scripts.train_model import generate_synthetic_dataset
    if metadata["file_path"] == "synthetic":
        record = generate_synthetic_dataset(n_samples=metadata["n_rows"])
    else:
        from scripts.data_loader import load_file, build_dataset_record
        fp = Path(metadata["file_path"])
        df_raw = load_file(fp)
        if df_raw is None:
            return None
        record = build_dataset_record(fp, df_raw)
        if record is None:
            return None

    df = record["df"]
    available_features = [c for c in feature_cols if c in df.columns]
    X = df[available_features]
    y = df[target_col]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y,
    )

    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # Reconstruct full df_test with protected attributes
    df_test = df.loc[y_test.index] if hasattr(y_test, "index") else df.iloc[-len(y_test):]
    df_test = df_test.copy()

    audit = run_fairness_audit(df_test, y_proba, y_test.values)
    audit["dataset_name"] = dataset_name

    # Save
    out_path = METRICS_DIR / f"{dataset_name}_fairness.json"
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(audit, fh, indent=2, default=str)
    log.info("Fairness audit saved to %s", out_path)

    if audit["alerts"]:
        log.warning("%d fairness alert(s) for %s:", len(audit["alerts"]), dataset_name)
        for a in audit["alerts"]:
            log.warning("  [%s] %s", a["severity"], a["message"])
    else:
        log.info("[%s] No significant fairness concerns detected.", dataset_name)

    return audit


def audit_all_models() -> list[dict]:
    """Run fairness audit on all trained models."""
    metrics_files = list(METRICS_DIR.glob("*_metrics.json"))
    if not metrics_files:
        log.warning("No model metrics files found. Run train_model.py first.")
        return []

    results = []
    for mf in metrics_files:
        r = audit_trained_model(mf)
        if r:
            results.append(r)
    return results


if __name__ == "__main__":
    results = audit_all_models()
    if results:
        for r in results:
            print(f"\nFairness Audit: {r.get('dataset_name', 'unknown')}")
            print(f"  Alerts: {len(r.get('alerts', []))}")
            for a in r.get("alerts", []):
                print(f"  [{a['severity']}] {a['message']}")
    else:
        print("No models to audit. Run train_model.py first.")

