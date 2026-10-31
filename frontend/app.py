"""
FairScore Streamlit Frontend
============================
Four pages:
  1. Applicant Scoring      – input form, consent, score, grade, recommendation
  2. Explanation            – positive / negative SHAP factors, feature importance
  3. Dashboard              – KPIs, risk distribution, approval trend
  4. Bias Audit             – approval rate by group, fairness alerts

Run with:
  streamlit run frontend/app.py
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FairScore – Alternative Credit Scoring",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Backend connectivity helpers
# ---------------------------------------------------------------------------

BACKEND_URL = "http://127.0.0.1:8001"


def _get(endpoint: str) -> dict:
    """GET request to backend; returns empty dict on failure."""
    try:
        import requests
        r = requests.get(f"{BACKEND_URL}{endpoint}", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        return {"_error": str(exc)}


def _post(endpoint: str, payload: dict) -> dict:
    """POST request to backend; returns empty dict on failure."""
    try:
        import requests
        r = requests.post(f"{BACKEND_URL}{endpoint}", json=payload, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        return {"_error": str(exc)}


def _backend_ok() -> bool:
    result = _get("/health")
    return result.get("status") == "ok"


# ---------------------------------------------------------------------------
# Sidebar navigation (with safe deferred page routing)
# ---------------------------------------------------------------------------
_FRONTEND_PAGES = ["🏠 Score Applicant", "📊 Explanation", "📈 Dashboard", "⚖️ Bias Audit"]

# Handle programmatic navigation BEFORE the radio widget is instantiated
if "frontend_target_page" in st.session_state and st.session_state["frontend_target_page"]:
    st.session_state["frontend_nav"] = st.session_state.pop("frontend_target_page")

if "frontend_nav" not in st.session_state:
    st.session_state["frontend_nav"] = "🏠 Score Applicant"

def _navigate_frontend(p: str):
    """Safely queue a page transition and trigger a rerun."""
    st.session_state["frontend_target_page"] = p

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/scales.png", width=60)
    st.title("FairScore")
    st.caption("Alternative Credit Scoring")

    page = st.radio(
        "Navigation",
        _FRONTEND_PAGES,
        key="frontend_nav",
    )
    st.markdown("---")
    health = _get("/health")
    if "_error" in health:
        st.error("⚠️ Backend offline")
        st.caption(f"`{BACKEND_URL}`")
    else:
        st.success("✅ Backend online")
        st.caption(f"Model loaded: {health.get('model_loaded', False)}")
    st.markdown("---")
    st.caption("⚠️ **PROTOTYPE** — Synthetic data only. Not a real credit decision.")


# ===========================================================================
# Page 1 – Score Applicant
# ===========================================================================

if page == "🏠 Score Applicant":
    st.title("⚖️ Applicant Credit Scoring")
    st.markdown(
        "Enter applicant details below. All fields are optional — missing values are imputed "
        "from population medians. At least one consent category is required."
    )

    if not _backend_ok():
        st.warning(
            "Backend is not reachable. Start the backend with:\n"
            "```\nuvicorn backend.main:app --port 8001 --reload\n```"
        )

    # ---- Consent ----
    st.subheader("1. Data Consent")
    col1, col2 = st.columns(2)
    with col1:
        c_rent     = st.checkbox("Rent Payment History", value=True)
        c_utility  = st.checkbox("Utility Payments", value=True)
        c_mobile   = st.checkbox("Mobile Recharge Behaviour", value=False)
    with col2:
        c_cashflow = st.checkbox("Cash-Flow Consistency", value=True)
        c_savings  = st.checkbox("Savings Behaviour", value=False)
        c_cash_pay = st.checkbox("Verified Cash Payments", value=False)

    consented_categories = [
        c for c, v in {
            "Rent": c_rent, "Utility": c_utility, "Mobile": c_mobile,
            "Cashflow": c_cashflow, "Savings": c_savings, "CashPayments": c_cash_pay,
        }.items() if v
    ]

    if not consented_categories:
        st.error("Please consent to at least one data category to generate a score.")

    # ---- Applicant Details ----
    st.subheader("2. Applicant Details")
    c1, c2, c3 = st.columns(3)

    with c1:
        monthly_income         = st.number_input("Monthly Income (₹)", 0, 500_000, 25_000, 1_000)
        income_volatility      = st.slider("Income Volatility", 0.0, 1.0, 0.2, 0.01)
        months_in_current_job  = st.number_input("Months in Current Job", 0, 240, 24, 1)
        num_dependants         = st.number_input("Number of Dependants", 0, 10, 1, 1)

    with c2:
        rent_payment_score         = st.slider("Rent Payment Score", 0.0, 1.0, 0.8, 0.01) if c_rent else 0.5
        utility_payment_score      = st.slider("Utility Payment Score", 0.0, 1.0, 0.75, 0.01) if c_utility else 0.5
        mobile_recharge_regularity = st.slider("Mobile Recharge Regularity", 0.0, 1.0, 0.7, 0.01) if c_mobile else 0.5

    with c3:
        cashflow_consistency    = st.slider("Cashflow Consistency", 0.0, 1.0, 0.7, 0.01) if c_cashflow else 0.5
        savings_rate            = st.slider("Savings Rate", 0.0, 0.4, 0.1, 0.01) if c_savings else 0.1
        verified_cash_payments  = st.number_input("Verified Cash Payments", 0, 25, 5, 1) if c_cash_pay else 0

    # ---- Loan Details ----
    st.subheader("3. Loan Request")
    l1, l2, l3 = st.columns(3)
    with l1:
        loan_amount = st.number_input("Loan Amount Requested (₹)", 5_000, 500_000, 50_000, 5_000)
    with l2:
        tenure_months = st.selectbox("Tenure (months)", [12, 24, 36, 48, 60], index=2)
    with l3:
        prior_defaults = st.number_input("Prior Defaults", 0, 5, 0, 1)

    expense_to_income_ratio = st.slider("Expense-to-Income Ratio", 0.1, 1.0, 0.55, 0.01)
    loan_to_income_ratio    = loan_amount / max(monthly_income * 12, 1)

    # ---- Generate Score ----
    st.markdown("---")
    if st.button("🔍 Generate FairScore", type="primary", disabled=not consented_categories):
        payload = {
            "monthly_income":             monthly_income,
            "income_volatility":          income_volatility,
            "rent_payment_score":         rent_payment_score if c_rent else None,
            "utility_payment_score":      utility_payment_score if c_utility else None,
            "mobile_recharge_regularity": mobile_recharge_regularity if c_mobile else None,
            "cashflow_consistency":       cashflow_consistency if c_cashflow else None,
            "savings_rate":               savings_rate if c_savings else None,
            "verified_cash_payments":     verified_cash_payments if c_cash_pay else None,
            "months_in_current_job":      months_in_current_job,
            "num_dependants":             num_dependants,
            "expense_to_income_ratio":    expense_to_income_ratio,
            "prior_defaults":             prior_defaults,
            "loan_amount_requested":      float(loan_amount),
            "loan_to_income_ratio":       round(loan_to_income_ratio, 4),
        }

        with st.spinner("Scoring applicant…"):
            result = _post("/predict", payload)

        if "_error" in result:
            st.error(f"Scoring failed: {result['_error']}")
        else:
            st.session_state["last_payload"] = payload
            st.session_state["last_result"]  = result

            score = result["credit_score"]
            grade = result["risk_grade"]
            prob  = result["default_probability"]
            rec   = result["recommendation"]

            # ---- Result display ----
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("FairScore", score, f"Grade {grade}")
            r2.metric("Default Probability", f"{prob:.1%}")
            r3.metric("Recommendation", rec)
            r4.metric("Consented Categories", len(consented_categories))

            # Colour band
            grade_colour = {"A": "green", "B": "blue", "C": "orange", "D": "red", "E": "darkred"}
            colour = grade_colour.get(grade, "grey")
            st.markdown(
                f'<div style="padding:10px; border-left: 6px solid {colour}; '
                f'background:#f9f9f9; border-radius:4px; margin-top:8px">'
                f"<b>Recommendation:</b> {rec}<br>"
                f"<b>Risk Grade:</b> {grade} | <b>Score:</b> {score} / 900"
                f"</div>",
                unsafe_allow_html=True,
            )

            st.caption(result.get("disclaimer", ""))

            st.subheader("Key Factors")
            pos = result.get("top_positive_factors", [])
            neg = result.get("top_negative_factors", [])
            pf, nf = st.columns(2)
            with pf:
                st.markdown("**✅ Positive Factors**")
                for f in pos:
                    st.markdown(f"- **{f['feature']}** (impact: +{f['impact']:.3f})")
            with nf:
                st.markdown("**⚠️ Risk Factors**")
                for f in neg:
                    st.markdown(f"- **{f['feature']}** (impact: +{f['impact']:.3f})")

            st.divider()
            st.subheader("Explore Connected Analysis")
            c_nx1, c_nx2, c_nx3 = st.columns(3)
            with c_nx1:
                if st.button("📊 View Deep SHAP Factor Breakdown →", use_container_width=True, type="primary"):
                    _navigate_frontend("📊 Explanation")
                    st.rerun()
            with c_nx2:
                if st.button("📈 Open Portfolio Dashboard →", use_container_width=True):
                    _navigate_frontend("📈 Dashboard")
                    st.rerun()
            with c_nx3:
                if st.button("⚖️ Inspect Demographic Bias Audit →", use_container_width=True):
                    _navigate_frontend("⚖️ Bias Audit")
                    st.rerun()


# ===========================================================================
# Page 2 – Explanation
# ===========================================================================

elif page == "📊 Explanation":
    st.title("📊 Score Explanation")

    if "last_payload" not in st.session_state:
        st.info("⚠️ No scored applicant found. Generate a score first on the **Score Applicant** page.")
        if st.button("← Go to Score Applicant", type="primary"):
            _navigate_frontend("🏠 Score Applicant")
            st.rerun()
    else:
        if "cached_explanation" not in st.session_state or st.button("🔄 Refresh Explanation"):
            with st.spinner("Computing SHAP explanations…"):
                st.session_state["cached_explanation"] = _post("/explain", st.session_state["last_payload"])

        result = st.session_state.get("cached_explanation", {})
        if "_error" in result:
            st.error(f"Explanation failed: {result['_error']}")
        else:
            st.subheader("Positive Factors (lower risk)")
            pos = result.get("positive_factors", [])
            if pos:
                df_pos = pd.DataFrame(pos)
                st.dataframe(df_pos, use_container_width=True)
            else:
                st.info("No positive factors computed.")

            st.subheader("Risk Factors (higher risk)")
            neg = result.get("negative_factors", [])
            if neg:
                df_neg = pd.DataFrame(neg)
                st.dataframe(df_neg, use_container_width=True)
            else:
                st.info("No risk factors computed.")

            st.subheader("Global Feature Importance")
            fi = result.get("feature_importance", [])
            if fi:
                df_fi = pd.DataFrame(fi).sort_values("importance", ascending=True)
                st.bar_chart(df_fi.set_index("feature")["importance"])
            else:
                st.info("No feature importance data available.")

            st.caption(result.get("disclaimer", ""))

        st.divider()
        c_exp1, c_exp2 = st.columns(2)
        with c_exp1:
            if st.button("← Score Another Applicant", use_container_width=True):
                _navigate_frontend("🏠 Score Applicant")
                st.rerun()
        with c_exp2:
            if st.button("View Portfolio Dashboard →", use_container_width=True):
                _navigate_frontend("📈 Dashboard")
                st.rerun()


# ===========================================================================
# Page 3 – Dashboard
# ===========================================================================

elif page == "📈 Dashboard":
    st.title("📈 Portfolio Dashboard")

    metrics = _get("/metrics")

    if "_error" in metrics:
        st.warning(f"Could not reach backend: {metrics['_error']}")
        # Demo values
        metrics = {
            "active_model": {
                "n_rows": 5_000,
                "test_metrics": {
                    "roc_auc": 0.85,
                    "f1": 0.72,
                    "accuracy": 0.83,
                    "actual_default_rate": 0.18,
                    "approval_rate_at_0.5": 0.82,
                },
            },
            "dataset_metrics": {"default_rate": 0.18},
        }

    am  = metrics.get("active_model", {})
    tm  = am.get("test_metrics", {})
    dm  = metrics.get("dataset_metrics", {})

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Applicants (Training)", f"{am.get('n_rows', 0):,}")
    k2.metric("Approval Rate (50% thr)", f"{tm.get('approval_rate_at_0.5', 0):.1%}")
    k3.metric("Avg Predicted Default Rate", f"{tm.get('actual_default_rate', dm.get('default_rate', 0)):.1%}")
    k4.metric("ROC-AUC", f"{tm.get('roc_auc', 0):.4f}")

    st.markdown("---")

    # Risk distribution (synthetic demo)
    import numpy as np

    st.subheader("Risk Grade Distribution")
    grade_counts = pd.DataFrame({
        "Grade": ["A", "B", "C", "D", "E"],
        "Count": [1400, 1600, 900, 700, 400],
    })
    st.bar_chart(grade_counts.set_index("Grade")["Count"])

    st.subheader("Model Performance")
    mp1, mp2, mp3 = st.columns(3)
    mp1.metric("F1 Score", f"{tm.get('f1', 0):.4f}")
    mp2.metric("Accuracy", f"{tm.get('accuracy', 0):.4f}")
    mp3.metric("PR-AUC", f"{tm.get('avg_precision', tm.get('roc_auc', 0)):.4f}")

    st.subheader("Approval Rate by Threshold")
    thr_data = {
        k.replace("approval_rate_at_", ""): v
        for k, v in tm.items()
        if k.startswith("approval_rate_at_")
    }
    if thr_data:
        df_thr = pd.DataFrame([
            {"Threshold": k, "Approval Rate": f"{v:.1%}", "Pct": v}
            for k, v in thr_data.items()
        ])
        st.dataframe(df_thr[["Threshold", "Approval Rate"]], use_container_width=True)

    st.caption("⚠️ All data is synthetic — prototype demonstration only.")

    st.divider()
    cd_1, cd_2 = st.columns(2)
    with cd_1:
        if st.button("← Score an Applicant", use_container_width=True, key="dash_to_score_btn"):
            _navigate_frontend("🏠 Score Applicant")
            st.rerun()
    with cd_2:
        if st.button("View Bias Audit →", use_container_width=True, key="dash_to_bias_btn"):
            _navigate_frontend("⚖️ Bias Audit")
            st.rerun()


# ===========================================================================
# Page 4 – Bias Audit
# ===========================================================================

elif page == "⚖️ Bias Audit":
    st.title("⚖️ Fairness & Bias Audit")

    fairness = _get("/fairness")

    if "_error" in fairness:
        st.warning(f"Could not reach backend: {fairness['_error']}")
    elif "message" in fairness and "No fairness" in fairness.get("message", ""):
        st.info(fairness["message"])
        st.markdown(
            "Run the fairness audit:\n"
            "```\npython scripts/fairness_audit.py\n```"
        )
    else:
        alerts = fairness.get("alerts", [])
        if alerts:
            st.subheader("⚠️ Fairness Alerts")
            for a in alerts:
                severity = a.get("severity", "INFO")
                colour   = {"HIGH": "error", "MEDIUM": "warning"}.get(severity, "info")
                getattr(st, colour)(f"[{severity}] {a.get('message', '')}")
        else:
            st.success("✅ No significant fairness concerns detected.")

        attrs = fairness.get("attributes", {})
        for attr_name, attr_data in attrs.items():
            st.subheader(f"Attribute: `{attr_name}`")

            col1, col2 = st.columns(2)
            col1.metric("Disparate Impact Ratio", f"{attr_data.get('disparate_impact_ratio', 1):.3f}")
            col2.metric("Approval Rate Gap", f"{attr_data.get('approval_rate_gap', 0):.1%}")

            groups = attr_data.get("groups", [])
            if groups:
                df_g = pd.DataFrame(groups)
                display_cols = [
                    c for c in [
                        "group", "n_total", "approval_rate",
                        "avg_default_probability", "false_rejection_rate", "false_approval_rate",
                    ] if c in df_g.columns
                ]
                st.dataframe(
                    df_g[display_cols].rename(columns={
                        "group": "Group",
                        "n_total": "N",
                        "approval_rate": "Approval Rate",
                        "avg_default_probability": "Avg Default Prob",
                        "false_rejection_rate": "False Rejection Rate",
                        "false_approval_rate": "False Approval Rate",
                    }).style.format({
                        "Approval Rate": "{:.1%}",
                        "Avg Default Prob": "{:.3f}",
                        "False Rejection Rate": "{:.3f}",
                        "False Approval Rate": "{:.3f}",
                    }),
                    use_container_width=True,
                )

                # Approval rate chart
                if "group" in df_g.columns and "approval_rate" in df_g.columns:
                    st.bar_chart(df_g.set_index("group")["approval_rate"])

    st.caption("⚠️ Fairness audit uses synthetic data — prototype demonstration only.")

    st.divider()
    cb_1, cb_2 = st.columns(2)
    with cb_1:
        if st.button("← Back to Dashboard", use_container_width=True, key="bias_to_dash_btn"):
            _navigate_frontend("📈 Dashboard")
            st.rerun()
    with cb_2:
        if st.button("Score an Applicant →", use_container_width=True, key="bias_to_score_btn"):
            _navigate_frontend("🏠 Score Applicant")
            st.rerun()

