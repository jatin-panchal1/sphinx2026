"""
FairScore — Streamlit Frontend
==============================
Multi-page Streamlit app for the FairScore alternative credit scoring platform.
Communicates with the FastAPI backend at http://localhost:8000.
"""

import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="FairScore",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    /* Colour tokens */
    :root {
        --primary: #245B45;
        --success: #245B45;
        --warning: #FFC107;
        --danger:  #DC3545;
    }
    .score-green  { color: #245B45; font-size: 3.5rem; font-weight: 800; }
    .score-yellow { color: #FFC107; font-size: 3.5rem; font-weight: 800; }
    .score-red    { color: #DC3545; font-size: 3.5rem; font-weight: 800; }

    .badge { display: inline-block; padding: 0.35em 0.75em; border-radius: 0.5em;
             font-weight: 700; font-size: 1rem; color: #fff; }
    .badge-green  { background: #245B45; }
    .badge-yellow { background: #FFC107; color: #333; }
    .badge-orange { background: #FD7E14; }
    .badge-red    { background: #DC3545; }

    .factor-card { border: 1px solid #e0e0e0; border-radius: 8px;
                   padding: 12px 16px; margin-bottom: 8px; }
    .factor-positive { border-left: 4px solid #245B45; }
    .factor-negative { border-left: 4px solid #DC3545; }

    .metric-card { background: #f8f9fa; border-radius: 10px;
                   padding: 1.25rem; text-align: center; }
    .metric-value { font-size: 2rem; font-weight: 800; color: #245B45; }
    .metric-label { font-size: 0.85rem; color: #6c757d; }

    .how-step { text-align: center; padding: 1rem; }
    .how-step h3 { color: #245B45; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------

_DEFAULTS: Dict[str, Any] = {
    "current_page": "🏠 Home",
    "nav_radio": "🏠 Home",
    "current_applicant_id": None,
    "applicant_token": None,
    "lender_token": None,
    "consent_submitted": False,
    "score_result": None,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------


def _get_lender_token() -> Optional[str]:
    """Retrieve or automatically authenticate a lender session token."""
    token = st.session_state.get("lender_token")
    if not token:
        try:
            resp = requests.post(
                f"{API_BASE}/api/auth/login",
                json={"username": "lender_admin", "password": "FairScoreUnderwriter2026!"},
                timeout=5,
            )
            if resp.status_code == 200:
                token = resp.json().get("access_token")
                st.session_state["lender_token"] = token
        except Exception:
            pass
    return token


def _get_auth_headers(path: str) -> Dict[str, str]:
    """Construct appropriate Authorization header for endpoint role."""
    headers = {"Accept": "application/json"}
    if path in ["/api/health", "/api/personas"]:
        return headers

    is_lender = any(path.startswith(p) for p in ["/api/applicants", "/api/dashboard", "/api/fairness", "/api/thresholds", "/api/model/metrics"])
    if is_lender:
        token = _get_lender_token()
    else:
        # Prefer applicant's scoped token, fallback to lender session
        token = st.session_state.get("applicant_token") or _get_lender_token()

    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _api_get(path: str, params: Optional[Dict] = None) -> Optional[Dict]:
    """GET request to the FastAPI backend with auth. Returns JSON or None on failure."""
    try:
        headers = _get_auth_headers(path)
        resp = requests.get(f"{API_BASE}{path}", params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        st.error("⚠️ FairScore API is not running. Please start the API server first.")
        return None
    except requests.HTTPError as exc:
        st.error(f"API error: {exc.response.status_code} — {exc.response.text}")
        return None
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")
        return None


def _api_post(path: str, json_body: Optional[Dict] = None) -> Optional[Dict]:
    """POST request to the FastAPI backend with auth. Returns JSON or None on failure."""
    try:
        headers = _get_auth_headers(path)
        resp = requests.post(f"{API_BASE}{path}", json=json_body, headers=headers, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        st.error("⚠️ FairScore API is not running. Please start the API server first.")
        return None
    except requests.HTTPError as exc:
        st.error(f"API error: {exc.response.status_code} — {exc.response.text}")
        return None
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")
        return None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

_NAV_PAGES = [
    "🏠 Home",
    "📝 Applicant Entry",
    "✅ Consent",
    "📊 Score Result",
    "📋 Lender Dashboard",
    "⚖️ Bias Audit Panel",
]

# Handle programmatic navigation BEFORE the radio widget is instantiated
if "target_page" in st.session_state and st.session_state["target_page"]:
    st.session_state["nav_radio"] = st.session_state.pop("target_page")

if "nav_radio" not in st.session_state:
    st.session_state["nav_radio"] = "🏠 Home"

def _navigate(page_name: str) -> None:
    """Safely queue a page transition and trigger a rerun."""
    st.session_state["target_page"] = page_name

with st.sidebar:
    st.markdown("# 🎯 FairScore")
    st.caption("Alternative Credit Scoring for the Credit-Invisible")
    st.divider()

    page = st.radio(
        "Navigate",
        _NAV_PAGES,
        key="nav_radio",
    )
    st.session_state["current_page"] = page

    st.divider()
    if st.session_state.get("current_applicant_id"):
        st.info(f"**Active Applicant:**\n\n`{st.session_state['current_applicant_id']}`")
        if st.button("Clear Active Applicant", use_container_width=True):
            st.session_state["current_applicant_id"] = None
            st.session_state["applicant_token"] = None
            st.session_state["consent_submitted"] = False
            st.session_state["score_result"] = None
            _navigate("📝 Applicant Entry")
            st.rerun()
    else:
        st.caption("No active applicant selected")


# ===================================================================
# PAGE 1 — Home
# ===================================================================


def page_home() -> None:
    """Landing page with product overview and key stats."""
    st.markdown("# 🎯 FairScore")
    st.markdown(
        "### Alternative Credit Scoring for the Credit-Invisible\n\n"
        "FairScore empowers lenders to evaluate applicants who lack traditional "
        "credit history by leveraging alternative financial signals — rent payments, "
        "utility bills, mobile-money activity, and self-declared income. "
        "Our AI-powered engine delivers transparent, bias-audited credit scores "
        "so that every individual gets a fair chance at financial inclusion."
    )

    st.markdown("")

    # Three action columns
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📊  Score an Applicant", use_container_width=True):
            _navigate("📝 Applicant Entry")
            st.rerun()
    with col2:
        if st.button("📋  Lender Dashboard", use_container_width=True):
            _navigate("📋 Lender Dashboard")
            st.rerun()
    with col3:
        if st.button("⚖️  Bias Audit Panel", use_container_width=True):
            _navigate("⚖️ Bias Audit Panel")
            st.rerun()

    st.divider()

    # Key stats
    with st.spinner("Loading system stats…"):
        data = _api_get("/api/dashboard")

    if data:
        m1, m2, m3, m4 = st.columns(4)
        appr_val = data.get("approval_rate", 0)
        appr_disp = appr_val * 100 if appr_val <= 1.0 else appr_val
        m1.metric("Total Applicants", data.get("total_applicants", 0))
        m2.metric("Approval Rate", f"{appr_disp:.1f}%")
        m3.metric("Avg Score", f"{data.get('average_score', 0):.0f}")
        m4.metric("Bias Alerts", data.get("bias_alert_count", data.get("bias_alerts", 0)))

    st.divider()

    # How it works
    st.markdown("### How It Works")
    h1, h2, h3, h4 = st.columns(4)
    with h1:
        st.markdown(
            '<div class="how-step"><h3>1️⃣</h3><b>Profile</b><br>'
            "Enter applicant demographics &amp; loan details</div>",
            unsafe_allow_html=True,
        )
    with h2:
        st.markdown(
            '<div class="how-step"><h3>2️⃣</h3><b>Consent</b><br>'
            "Select which alternative data to share</div>",
            unsafe_allow_html=True,
        )
    with h3:
        st.markdown(
            '<div class="how-step"><h3>3️⃣</h3><b>Score</b><br>'
            "AI generates a transparent credit score</div>",
            unsafe_allow_html=True,
        )
    with h4:
        st.markdown(
            '<div class="how-step"><h3>4️⃣</h3><b>Explain</b><br>'
            "Review SHAP-based explanations &amp; bias checks</div>",
            unsafe_allow_html=True,
        )


# ===================================================================
# PAGE 2 — Applicant Entry
# ===================================================================


def page_applicant_entry() -> None:
    """Applicant intake form."""
    st.markdown("# 📝 Applicant Entry")
    st.caption("Enter applicant details or pick a quick persona to auto-fill the form.")

    # ------ persona auto-fill ------
    personas_data = _api_get("/api/personas")
    persona_names: List[str] = ["— Select a Persona —"]
    persona_map: Dict[str, Dict] = {}
    if personas_data:
        for p in personas_data:
            label = p.get("name", p.get("persona_name", str(p)))
            persona_names.append(label)
            persona_map[label] = p

    selected_persona = st.selectbox("Quick Persona", persona_names)
    prefill: Dict[str, Any] = {}
    if selected_persona != "— Select a Persona —":
        prefill = persona_map.get(selected_persona, {})

    st.divider()

    AGE_GROUPS = ["18-25", "26-35", "36-45", "46-55", "56+"]
    GENDERS = ["Male", "Female", "Non-Binary"]
    REGIONS = ["North", "South", "East", "West", "Central"]
    URBAN_RURAL = ["Urban", "Semi-Urban", "Rural"]
    OCCUPATIONS = [
        "Gig Worker",
        "Small Shopkeeper",
        "Student",
        "Informal Worker",
        "Freelancer",
        "Daily Wage Worker",
    ]
    INCOME_TYPES = ["Weekly", "Monthly", "Daily", "Irregular"]
    TENURES = [3, 6, 9, 12, 18, 24]

    def _idx(options: list, key: str, default: int = 0) -> int:
        val = prefill.get(key)
        if val is not None and val in options:
            return options.index(val)
        return default

    with st.form("applicant_form"):
        c1, c2 = st.columns(2)
        with c1:
            age_group = st.selectbox("Age Group", AGE_GROUPS, index=_idx(AGE_GROUPS, "age_group"))
            gender = st.selectbox("Gender", GENDERS, index=_idx(GENDERS, "gender"))
            region = st.selectbox("Region", REGIONS, index=_idx(REGIONS, "region"))
            urban_rural = st.selectbox("Urban / Rural", URBAN_RURAL, index=_idx(URBAN_RURAL, "urban_rural"))
        with c2:
            occupation = st.selectbox("Occupation", OCCUPATIONS, index=_idx(OCCUPATIONS, "occupation"))
            income_type = st.selectbox("Income Type", INCOME_TYPES, index=_idx(INCOME_TYPES, "income_type"))
            loan_amount = st.slider(
                "Loan Amount (₹)",
                min_value=5000,
                max_value=200000,
                value=int(prefill.get("loan_amount", 50000)),
                step=5000,
            )
            loan_tenure = st.selectbox(
                "Loan Tenure (months)",
                TENURES,
                index=_idx(TENURES, "loan_tenure", default=3),
            )

        submitted = st.form_submit_button("Submit Applicant", use_container_width=True)

    if submitted:
        payload = {
            "age_group": age_group,
            "gender_category": gender,
            "region_category": region,
            "urban_rural": urban_rural,
            "occupation_type": occupation,
            "income_type": income_type,
            "loan_amount_requested": float(loan_amount),
            "loan_tenure_months": int(loan_tenure),
        }
        with st.spinner("Creating applicant profile…"):
            result = _api_post("/api/applicant", json_body=payload)
        if result:
            applicant_id = result.get("applicant_id", result.get("id"))
            auth_token = result.get("auth_token")
            st.session_state["current_applicant_id"] = applicant_id
            st.session_state["applicant_token"] = auth_token
            st.session_state["consent_submitted"] = False
            st.session_state["score_result"] = None
            st.success(f"✅ Applicant created: `{applicant_id}`")
            _navigate("✅ Consent")
            st.rerun()


# ===================================================================
# PAGE 3 — Consent
# ===================================================================

_CONSENT_CATEGORIES = [
    {
        "key": "rent_data",
        "label": "🏠 Rent Data",
        "purpose": "Verify consistent rent payments as a proxy for repayment behaviour.",
    },
    {
        "key": "utility_data",
        "label": "💡 Utility Data",
        "purpose": "Assess regular utility bill payments to gauge financial discipline.",
    },
    {
        "key": "mobile_data",
        "label": "📱 Mobile Data",
        "purpose": "Analyse mobile-money & recharge patterns for spending consistency.",
    },
    {
        "key": "cashflow_data",
        "label": "💰 Cash-flow Data",
        "purpose": "Review bank / wallet cash-flow to evaluate income stability.",
    },
    {
        "key": "business_data",
        "label": "🏪 Business Data",
        "purpose": "Capture small-business revenue signals (shopkeepers only).",
    },
    {
        "key": "self_declared_income",
        "label": "📝 Self-declared Income",
        "purpose": "Allow applicants to declare income sources for additional context.",
    },
]


def page_consent() -> None:
    """Data consent collection page."""
    applicant_id = st.session_state.get("current_applicant_id")
    if not applicant_id:
        st.warning("⚠️ No active applicant selected. Create a new profile or select an existing applicant below.")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            if st.button("➕ Go to Applicant Entry", use_container_width=True, type="primary"):
                _navigate("📝 Applicant Entry")
                st.rerun()
        with col_c2:
            apps_resp = _api_get("/api/applicants", params={"limit": 20})
            if apps_resp and isinstance(apps_resp, dict) and "applicants" in apps_resp:
                app_options = [a["applicant_id"] for a in apps_resp["applicants"]]
                sel_app = st.selectbox("Or choose from existing applicants:", app_options)
                if st.button("Load Selected Applicant", use_container_width=True):
                    st.session_state["current_applicant_id"] = sel_app
                    st.session_state["consent_submitted"] = False
                    st.session_state["score_result"] = None
                    st.rerun()
        return

    st.markdown("# ✅ Data Consent Request")
    st.info(f"**Applicant ID:** `{applicant_id}`")
    st.markdown(
        "FairScore uses alternative financial data to assess your creditworthiness. "
        "Please review and consent to the data categories below."
    )

    st.divider()

    consents: Dict[str, bool] = {}
    for cat in _CONSENT_CATEGORIES:
        consents[cat["key"]] = st.checkbox(
            f"**{cat['label']}** — {cat['purpose']}",
            key=f"consent_{cat['key']}",
        )

    selected_count = sum(consents.values())
    if selected_count == 0:
        st.warning("⚠️ Please select at least 1 alternative data category to permit credit scoring.")
    elif selected_count == 1:
        st.info("ℹ️ 1 category selected. Adding more data categories improves score precision.")

    st.divider()

    col_submit, col_score = st.columns(2)

    with col_submit:
        if st.button("Submit Consent", use_container_width=True, type="primary"):
            payload = {
                "applicant_id": applicant_id,
                **consents,
            }
            with st.spinner("Saving consent…"):
                result = _api_post("/api/consent", json_body=payload)
            if result:
                st.session_state["consent_submitted"] = True
                st.success("✅ Consent recorded successfully.")
                st.markdown("**Consent Receipt:**")
                consented = [c["label"] for c in _CONSENT_CATEGORIES if consents.get(c["key"])]
                for item in consented:
                    st.markdown(f"- {item}")
                st.rerun()

    with col_score:
        can_score = st.session_state.get("consent_submitted", False) and selected_count >= 1
        if st.button("Generate Score →", disabled=not can_score, use_container_width=True):
            with st.spinner("Generating credit score…"):
                score_result = _api_post("/api/score", json_body={"applicant_id": applicant_id})
            if score_result:
                st.session_state["score_result"] = score_result
                _navigate("📊 Score Result")
                st.rerun()

    if st.session_state.get("consent_submitted"):
        st.success("Consent has been submitted. You may now generate the score.")


# ===================================================================
# PAGE 4 — Score Result
# ===================================================================


def _score_color_class(score: float) -> str:
    if score > 700:
        return "score-green"
    elif score >= 500:
        return "score-yellow"
    return "score-red"


def _recommendation_badge(rec: str) -> str:
    rec_lower = rec.lower()
    if "approve" in rec_lower and "reduced" not in rec_lower and "manual" not in rec_lower:
        return "badge badge-green"
    elif "reduced" in rec_lower:
        return "badge badge-yellow"
    elif "manual" in rec_lower or "review" in rec_lower:
        return "badge badge-orange"
    return "badge badge-red"


def _gauge_chart(score: float) -> go.Figure:
    """Create a plotly gauge chart for the credit score."""
    if score > 700:
        bar_color = "#245B45"
    elif score >= 500:
        bar_color = "#FFC107"
    else:
        bar_color = "#DC3545"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": "Credit Score", "font": {"size": 20}},
            number={"font": {"size": 48, "color": bar_color}},
            gauge={
                "axis": {"range": [300, 900], "tickwidth": 2},
                "bar": {"color": bar_color, "thickness": 0.3},
                "bgcolor": "white",
                "steps": [
                    {"range": [300, 500], "color": "rgba(220,53,69,0.15)"},
                    {"range": [500, 700], "color": "rgba(255,193,7,0.15)"},
                    {"range": [700, 900], "color": "rgba(40,167,69,0.15)"},
                ],
                "threshold": {
                    "line": {"color": bar_color, "width": 4},
                    "thickness": 0.8,
                    "value": score,
                },
            },
        )
    )
    fig.update_layout(height=300, margin=dict(t=40, b=20, l=30, r=30))
    return fig


def page_score_result() -> None:
    """Display scoring result with SHAP explanations."""
    result = st.session_state.get("score_result")
    applicant_id = st.session_state.get("current_applicant_id")

    if not result:
        st.warning("⚠️ No score available for current view.")
        c_sc1, c_sc2 = st.columns(2)
        with c_sc1:
            if st.button("📝 Create & Score New Applicant", use_container_width=True, type="primary"):
                _navigate("📝 Applicant Entry")
                st.rerun()
        with c_sc2:
            if st.button("📋 Open Lender Dashboard Queue", use_container_width=True):
                _navigate("📋 Lender Dashboard")
                st.rerun()
        return

    st.markdown("# 📊 Score Result")

    # ---- Applicant summary ----
    applicant_info = result.get("applicant", {})
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"**Applicant ID:** `{applicant_id}`")
    c2.markdown(f"**Occupation:** {applicant_info.get('occupation', 'N/A')}")
    c3.markdown(f"**Loan Amount:** ₹{applicant_info.get('loan_amount', 'N/A'):,}" if isinstance(applicant_info.get('loan_amount'), (int, float)) else f"**Loan Amount:** {applicant_info.get('loan_amount', 'N/A')}")

    st.divider()

    # ---- Score display ----
    score = result.get("score", 0)
    risk_grade = result.get("risk_grade", "N/A")
    confidence = result.get("confidence", 0)

    left, right = st.columns([1, 1])
    with left:
        fig = _gauge_chart(score)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        css_class = _score_color_class(score)
        st.markdown(f'<div class="{css_class}">{score:.0f}</div>', unsafe_allow_html=True)
        st.markdown(f"**Risk Grade:** `{risk_grade}`")
        st.progress(min(confidence, 1.0), text=f"Confidence: {confidence * 100:.0f}%")

    st.divider()

    # ---- Recommendation ----
    raw_rec = result.get("recommendation", "N/A")
    if isinstance(raw_rec, dict):
        rec_decision = raw_rec.get("decision", "N/A")
        rec_reason = raw_rec.get("reason", "")
        rec_limit = raw_rec.get("suggested_limit")
        improvements = raw_rec.get("improvements", [])
    else:
        rec_decision = str(raw_rec)
        rec_reason = result.get("recommendation_reason", "")
        rec_limit = result.get("approved_limit")
        improvements = result.get("improvements", [])

    badge_cls = _recommendation_badge(rec_decision)
    st.markdown(f'<span class="{badge_cls}">{rec_decision}</span>', unsafe_allow_html=True)
    if rec_reason:
        st.markdown(f"_{rec_reason}_")
    if rec_limit is not None:
        st.markdown(f"**Suggested Limit:** ₹{rec_limit:,.0f}")
    if improvements:
        st.markdown("**Improvement Suggestions:**")
        for imp in improvements:
            st.markdown(f"- {imp}")

    st.divider()

    # ---- SHAP explanation ----
    st.markdown("### Explanation — Key Factors")

    explanation = result.get("explanation", {})
    positive_factors: List[Dict] = explanation.get("positive_factors", [])[:5]
    negative_factors: List[Dict] = explanation.get("negative_factors", [])[:5]

    pos_col, neg_col = st.columns(2)

    with pos_col:
        st.markdown("#### ✅ Positive Factors")
        if not positive_factors:
            st.caption("None reported.")
        for f in positive_factors:
            impact = abs(f.get("impact", 0))
            st.markdown(
                f'<div class="factor-card factor-positive">'
                f'<b>{f.get("feature", "")}</b><br>'
                f'{f.get("reason", "")}'
                f"</div>",
                unsafe_allow_html=True,
            )
            st.progress(min(impact, 1.0))

    with neg_col:
        st.markdown("#### ❌ Negative Factors")
        if not negative_factors:
            st.caption("None reported.")
        for f in negative_factors:
            impact = abs(f.get("impact", 0))
            st.markdown(
                f'<div class="factor-card factor-negative">'
                f'<b>{f.get("feature", "")}</b><br>'
                f'{f.get("reason", "")}'
                f"</div>",
                unsafe_allow_html=True,
            )
            st.progress(min(impact, 1.0))

    # ---- Summary text ----
    summary_text = result.get("summary", "")
    if summary_text:
        st.divider()
        st.markdown("### 📝 Plain-Language Summary")
        st.markdown(summary_text)

    # ---- Bottom actions ----
    st.divider()
    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("Score Another", use_container_width=True):
            st.session_state["current_applicant_id"] = None
            st.session_state["consent_submitted"] = False
            st.session_state["score_result"] = None
            _navigate("📝 Applicant Entry")
            st.rerun()
    with b2:
        if st.button("View Dashboard", use_container_width=True):
            _navigate("📋 Lender Dashboard")
            st.rerun()
    with b3:
        if st.button("View Bias Audit", use_container_width=True):
            _navigate("⚖️ Bias Audit Panel")
            st.rerun()


# ===================================================================
# PAGE 5 — Lender Dashboard
# ===================================================================


def page_dashboard() -> None:
    """Lender dashboard with metrics, charts, and applicant table."""
    st.markdown("# 📋 Lender Dashboard")

    top_d1, top_d2, top_d3 = st.columns(3)
    with top_d1:
        if st.button("➕ Score New Applicant", use_container_width=True, key="dash_top_score_btn"):
            _navigate("📝 Applicant Entry")
            st.rerun()
    with top_d2:
        if st.button("✅ Manage Consent", use_container_width=True, key="dash_top_consent_btn"):
            _navigate("✅ Consent")
            st.rerun()
    with top_d3:
        if st.button("⚖️ Inspect Bias Audit Panel", use_container_width=True, key="dash_top_bias_btn"):
            _navigate("⚖️ Bias Audit Panel")
            st.rerun()

    st.divider()

    with st.spinner("Loading dashboard data…"):
        dashboard = _api_get("/api/dashboard")

    if not dashboard:
        return

    # ---- Top row metrics ----
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    ar = dashboard.get("approval_rate", 0)
    ar_disp = ar * 100 if ar <= 1.0 else ar
    dr = dashboard.get("predicted_default_rate", 0)
    dr_disp = dr * 100 if dr <= 1.0 else dr
    m1.metric("Total Applicants", dashboard.get("total_applicants", 0))
    m2.metric("Approval Rate", f"{ar_disp:.1f}%")
    m3.metric("Avg Score", f"{dashboard.get('average_score', 0):.0f}")
    m4.metric("Default Rate", f"{dr_disp:.1f}%")
    m5.metric("Manual Reviews", dashboard.get("manual_review_count", 0))
    m6.metric("Bias Alerts", dashboard.get("bias_alert_count", dashboard.get("bias_alerts", 0)))

    st.divider()

    # ---- Charts row 1 ----
    chart_l, chart_r = st.columns(2)

    with chart_l:
        st.markdown("#### Approval Rate by Risk Grade")
        approval_by_grade = dashboard.get("approval_by_risk_grade", {})
        if approval_by_grade:
            grades = list(approval_by_grade.keys())
            rates = list(approval_by_grade.values())
            fig = px.bar(
                x=grades,
                y=rates,
                labels={"x": "Risk Grade", "y": "Approval Rate (%)"},
                color=rates,
                color_continuous_scale=["#DC3545", "#FFC107", "#245B45"],
            )
            fig.update_layout(showlegend=False, height=350, margin=dict(t=20, b=40))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("No data available.")

    with chart_r:
        st.markdown("#### Risk Distribution")
        risk_dist = dashboard.get("risk_distribution", {})
        if risk_dist:
            fig = px.pie(
                names=list(risk_dist.keys()),
                values=list(risk_dist.values()),
                hole=0.45,
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig.update_layout(height=350, margin=dict(t=20, b=40))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("No data available.")

    st.divider()

    # ---- Charts row 2 ----
    chart_l2, chart_r2 = st.columns(2)

    with chart_l2:
        st.markdown("#### Score Distribution")
        score_distribution = dashboard.get("score_distribution", [])
        if score_distribution:
            fig = px.histogram(
                x=score_distribution,
                nbins=20,
                labels={"x": "Credit Score", "y": "Count"},
                color_discrete_sequence=["#2E86AB"],
            )
            fig.update_layout(height=350, margin=dict(t=20, b=40))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("No data available.")

    with chart_r2:
        st.markdown("#### Consent Completion Rate")
        raw_consent = dashboard.get("consent_completion_rate", 0)
        consent_rate = round(raw_consent * 100, 1) if raw_consent <= 1.0 else raw_consent
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=consent_rate,
                number={"suffix": "%"},
                title={"text": "Consent Completion"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#2E86AB"},
                    "steps": [
                        {"range": [0, 50], "color": "rgba(220,53,69,0.12)"},
                        {"range": [50, 80], "color": "rgba(255,193,7,0.12)"},
                        {"range": [80, 100], "color": "rgba(40,167,69,0.12)"},
                    ],
                },
            )
        )
        fig.update_layout(height=350, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ---- Applicant table ----
    st.markdown("#### Applicant Table")
    with st.spinner("Loading applicants…"):
        applicants_resp = _api_get("/api/applicants")

    if applicants_resp:
        applicants_list = applicants_resp if isinstance(applicants_resp, list) else applicants_resp.get("applicants", [])
        if applicants_list:
            df = pd.DataFrame(applicants_list)
            display_cols = [c for c in ["id", "applicant_id", "occupation", "score", "risk_grade", "recommendation"] if c in df.columns]
            if not display_cols:
                display_cols = df.columns.tolist()

            # Filter by risk grade
            all_grades = ["All"] + sorted(df["risk_grade"].dropna().unique().tolist()) if "risk_grade" in df.columns else ["All"]
            grade_filter = st.selectbox("Filter by Risk Grade", all_grades)
            if grade_filter != "All" and "risk_grade" in df.columns:
                df = df[df["risk_grade"] == grade_filter]

            st.dataframe(df[display_cols], use_container_width=True)

            # Expander detail
            for _, row in df.iterrows():
                row_id = row.get("applicant_id", row.get("id", ""))
                with st.expander(f"Details — {row_id}"):
                    for col_name in df.columns:
                        st.markdown(f"**{col_name}:** {row[col_name]}")
        else:
            st.caption("No applicants yet.")
    else:
        st.caption("Could not load applicants.")

    st.divider()

    # ---- Model performance ----
    st.markdown("#### Model Performance")
    with st.spinner("Loading model metrics…"):
        metrics = _api_get("/api/model/metrics")

    if metrics:
        pm1, pm2, pm3, pm4 = st.columns(4)
        pm1.metric("AUC-ROC", f"{metrics.get('auc_roc', 0):.3f}")
        pm2.metric("Precision", f"{metrics.get('precision', 0):.3f}")
        pm3.metric("Recall", f"{metrics.get('recall', 0):.3f}")
        pm4.metric("F1 Score", f"{metrics.get('f1', 0):.3f}")

        # Confusion matrix
        cm = metrics.get("confusion_matrix")
        if cm:
            cm_array = cm if isinstance(cm, list) else [[cm.get("tn", 0), cm.get("fp", 0)], [cm.get("fn", 0), cm.get("tp", 0)]]
            fig = px.imshow(
                cm_array,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=["Negative", "Positive"],
                y=["Negative", "Positive"],
                text_auto=True,
                color_continuous_scale="Blues",
            )
            fig.update_layout(height=350, margin=dict(t=20, b=40))
            st.plotly_chart(fig, use_container_width=True)


# ===================================================================
# PAGE 6 — Bias Audit Panel
# ===================================================================


def page_bias_audit() -> None:
    """Fairness / bias audit panel."""
    st.markdown("# ⚖️ Bias Audit Panel")

    with st.spinner("Loading fairness data…"):
        fairness = _api_get("/api/fairness")

    if not fairness:
        return

    # ---- Group selector ----
    available_dims = list(fairness.get("group_metrics", {}).keys())
    default_dims = ["urban_rural", "gender_category", "age_group", "region_category", "occupation_type", "income_type"]
    dimensions = available_dims if available_dims else default_dims

    dim_labels = {
        "gender_category": "Gender",
        "gender": "Gender",
        "age_group": "Age Group",
        "urban_rural": "Urban / Rural",
        "region_category": "Region",
        "region": "Region",
        "occupation_type": "Occupation",
        "occupation": "Occupation",
        "income_type": "Income Type",
    }
    selected_dim = st.selectbox(
        "Select Demographic Dimension",
        dimensions,
        format_func=lambda d: dim_labels.get(d, d.replace("_", " ").title()),
    )

    st.divider()

    # ---- Bias alerts ----
    alerts = fairness.get("bias_alerts", fairness.get("alerts", []))
    dim_alerts = [a for a in alerts if a.get("group_column") == selected_dim or a.get("dimension") == selected_dim] if alerts else []
    if not dim_alerts:
        dim_alerts = alerts  # show all if no per-dimension filtering available

    if dim_alerts:
        st.markdown("### 🚨 Bias Alerts")
        for alert in dim_alerts:
            severity = alert.get("severity", "medium").lower()
            color = "#DC3545" if severity == "high" else "#FFC107"
            col_name = dim_labels.get(alert.get("group_column", ""), alert.get("group_column", "General"))
            st.markdown(
                f'<div style="border-left: 4px solid {color}; padding: 12px 16px; '
                f'margin-bottom: 10px; border-radius: 6px; background: {color}15;">'
                f'<b>{alert.get("message", "Alert")}</b><br>'
                f'<b>Dimension:</b> {col_name} &nbsp; | &nbsp; '
                f'<b>Severity:</b> <span style="color:{color}; font-weight:700;">{severity.upper()}</span></div>',
                unsafe_allow_html=True,
            )
    else:
        st.success("No bias alerts for this dimension. ✅")

    st.divider()

    # ---- Retrieve group-level data for selected dimension ----
    raw_group_data = fairness.get("group_metrics", {}).get(selected_dim, [])
    groups, approval_rates, avg_scores, false_rejection, false_approval = [], [], [], [], []

    if isinstance(raw_group_data, list) and raw_group_data:
        groups = [str(item.get("group_value", f"Group {i}")) for i, item in enumerate(raw_group_data)]
        approval_rates = [(item.get("approval_rate", 0) * 100 if item.get("approval_rate", 0) <= 1.0 else item.get("approval_rate", 0)) for item in raw_group_data]
        avg_scores = [item.get("average_score", 0) for item in raw_group_data]
        false_rejection = [(item.get("false_negative_rate", item.get("false_rejection_rate", 0)) * 100 if item.get("false_negative_rate", item.get("false_rejection_rate", 0)) <= 1.0 else item.get("false_negative_rate", item.get("false_rejection_rate", 0))) for item in raw_group_data]
        false_approval = [(item.get("false_positive_rate", item.get("false_approval_rate", 0)) * 100 if item.get("false_positive_rate", item.get("false_approval_rate", 0)) <= 1.0 else item.get("false_positive_rate", item.get("false_approval_rate", 0))) for item in raw_group_data]
    elif isinstance(raw_group_data, dict) and raw_group_data:
        groups = list(raw_group_data.keys())
        approval_rates = [(raw_group_data[g].get("approval_rate", 0) * 100 if raw_group_data[g].get("approval_rate", 0) <= 1.0 else raw_group_data[g].get("approval_rate", 0)) for g in groups]
        avg_scores = [raw_group_data[g].get("average_score", 0) for g in groups]
        false_rejection = [(raw_group_data[g].get("false_negative_rate", raw_group_data[g].get("false_rejection_rate", 0)) * 100 if raw_group_data[g].get("false_negative_rate", raw_group_data[g].get("false_rejection_rate", 0)) <= 1.0 else raw_group_data[g].get("false_negative_rate", raw_group_data[g].get("false_rejection_rate", 0))) for g in groups]
        false_approval = [(raw_group_data[g].get("false_positive_rate", raw_group_data[g].get("false_approval_rate", 0)) * 100 if raw_group_data[g].get("false_positive_rate", raw_group_data[g].get("false_approval_rate", 0)) <= 1.0 else raw_group_data[g].get("false_positive_rate", raw_group_data[g].get("false_approval_rate", 0))) for g in groups]

    if groups:

        ch1, ch2 = st.columns(2)

        with ch1:
            st.markdown("#### Approval Rate by Group")
            fig = px.bar(
                x=groups,
                y=approval_rates,
                labels={"x": dim_labels.get(selected_dim, selected_dim), "y": "Approval Rate (%)"},
                color=approval_rates,
                color_continuous_scale=["#DC3545", "#FFC107", "#245B45"],
            )
            fig.update_layout(showlegend=False, height=350, margin=dict(t=20, b=40))
            st.plotly_chart(fig, use_container_width=True)

        with ch2:
            st.markdown("#### Average Score by Group")
            fig = px.bar(
                x=groups,
                y=avg_scores,
                labels={"x": dim_labels.get(selected_dim, selected_dim), "y": "Average Score"},
                color=avg_scores,
                color_continuous_scale=["#DC3545", "#FFC107", "#245B45"],
            )
            fig.update_layout(showlegend=False, height=350, margin=dict(t=20, b=40))
            st.plotly_chart(fig, use_container_width=True)

        ch3, ch4 = st.columns(2)

        with ch3:
            st.markdown("#### False Rejection Rate by Group")
            fig = px.bar(
                x=groups,
                y=false_rejection,
                labels={"x": dim_labels.get(selected_dim, selected_dim), "y": "False Rejection Rate (%)"},
                color_discrete_sequence=["#DC3545"],
            )
            fig.update_layout(height=350, margin=dict(t=20, b=40))
            st.plotly_chart(fig, use_container_width=True)

        with ch4:
            st.markdown("#### False Approval Rate by Group")
            fig = px.bar(
                x=groups,
                y=false_approval,
                labels={"x": dim_labels.get(selected_dim, selected_dim), "y": "False Approval Rate (%)"},
                color_discrete_sequence=["#FD7E14"],
            )
            fig.update_layout(height=350, margin=dict(t=20, b=40))
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("No per-group metrics available for this dimension.")

    st.divider()

    # ---- Disparate impact ----
    di_data = fairness.get("disparate_impact", {}).get(selected_dim, fairness.get("disparate_impact", {}))
    if di_data and isinstance(di_data, dict):
        st.markdown("### Disparate Impact Indicator")
        ratio = di_data.get("ratio", di_data.get("disparate_impact_ratio"))
        if ratio is not None:
            color = "#245B45" if ratio >= 0.8 else "#DC3545"
            label = "PASS ✅" if ratio >= 0.8 else "FAIL ❌"
            st.markdown(
                f'<div style="font-size:2rem; font-weight:800; color:{color};">'
                f"{ratio:.2f} — {label}</div>",
                unsafe_allow_html=True,
            )
            adv = di_data.get("advantaged_group", "N/A")
            disadv = di_data.get("disadvantaged_group", "N/A")
            st.markdown(f"**Advantaged Group:** {adv} &nbsp; | &nbsp; **Disadvantaged Group:** {disadv}")
        else:
            st.caption("Disparate impact ratio not available.")
    else:
        st.caption("Disparate impact data not available.")

    st.divider()

    # ---- Feature impact comparison ----
    feature_impact = fairness.get("feature_impact", {}).get(selected_dim)
    if feature_impact and isinstance(feature_impact, dict):
        st.markdown("### Feature Impact Comparison")
        fi_df = pd.DataFrame(feature_impact)
        fig = px.imshow(
            fi_df.values,
            labels=dict(x="Group", y="Feature", color="Avg Value"),
            x=fi_df.columns.tolist(),
            y=fi_df.index.tolist(),
            color_continuous_scale="RdYlGn",
            text_auto=".2f",
        )
        fig.update_layout(height=400, margin=dict(t=20, b=40))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    bb1, bb2, bb3 = st.columns(3)
    with bb1:
        if st.button("📋 Back to Lender Dashboard", use_container_width=True, key="bias_back_dash"):
            _navigate("📋 Lender Dashboard")
            st.rerun()
    with bb2:
        if st.button("📝 Score an Applicant", use_container_width=True, key="bias_to_score"):
            _navigate("📝 Applicant Entry")
            st.rerun()
    with bb3:
        if st.button("🏠 Return to Home", use_container_width=True, key="bias_to_home"):
            _navigate("🏠 Home")
            st.rerun()


# ===================================================================
# Router
# ===================================================================

_PAGES = {
    "🏠 Home": page_home,
    "📝 Applicant Entry": page_applicant_entry,
    "✅ Consent": page_consent,
    "📊 Score Result": page_score_result,
    "📋 Lender Dashboard": page_dashboard,
    "⚖️ Bias Audit Panel": page_bias_audit,
}

current = st.session_state["current_page"]
_PAGES.get(current, page_home)()
