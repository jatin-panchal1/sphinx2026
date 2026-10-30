"""Direct Python test runner for FairScore API & Core Modules.

Executes startup, initializes models and datasets, and verifies all endpoints.
"""

import asyncio
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from api.main import (
    app,
    startup,
    create_applicant,
    submit_consent,
    generate_score,
    get_explanation,
    get_fairness_summary,
    get_dashboard_summary,
    get_model_metrics,
    list_applicants,
    list_personas,
    update_thresholds,
    get_applicant_detail,
    health_check,
)
from api.schemas import ApplicantCreate, ConsentSubmit, ScoreRequest, ThresholdUpdate


async def run_all_tests():
    print("=" * 60)
    print("🚀 Initializing FairScore Startup (Training model & SHAP)...")
    print("=" * 60)
    await startup()
    print("✅ Startup succeeded!\n")

    # 1. Health check
    print("1. Testing Health Check...")
    h = await health_check()
    assert h["status"] in ("healthy", "ok")
    print("   ✅ Health check passed:", h)

    # 2. Personas
    print("\n2. Testing Personas List...")
    personas = await list_personas()
    assert len(personas) >= 6
    print(f"   ✅ Retrieved {len(personas)} personas: {[p['name'] for p in personas[:3]]}...")

    # 3. Model metrics
    print("\n3. Testing Model Metrics...")
    metrics = await get_model_metrics()
    print(f"   ✅ AUC-ROC: {metrics.auc_roc:.4f}, Precision: {metrics.precision:.4f}, Recall: {metrics.recall:.4f}, F1: {metrics.f1_score:.4f}")
    assert metrics.auc_roc > 0.5

    # 4. Dashboard summary
    print("\n4. Testing Dashboard Summary...")
    dash = await get_dashboard_summary()
    print(f"   ✅ Total applicants: {dash.total_applicants}, Approval rate: {dash.approval_rate:.1%}, Avg score: {dash.average_score:.0f}")
    assert dash.total_applicants >= 1000

    # 5. Fairness summary & bias alerts
    print("\n5. Testing Fairness Summary & Bias Alerts...")
    fairness = await get_fairness_summary()
    assert "urban_rural" in fairness.group_metrics
    alerts = fairness.bias_alerts
    print(f"   ✅ Detected {len(alerts)} bias alerts across demographics.")
    for a in alerts[:2]:
        print(f"      - [{a.get('severity', 'info').upper()}] {a.get('group_column')}: {a.get('message')[:75]}...")
    di = fairness.disparate_impact
    if "urban_rural" in di:
        print(f"   ✅ Urban/Rural Disparate Impact Ratio: {di['urban_rural'].get('ratio'):.2f}")

    # 6. Applicant Creation
    print("\n6. Testing Applicant Creation (Ravi, Gig Worker)...")
    applicant_req = ApplicantCreate(
        age_group="26-35",
        gender_category="Male",
        region_category="North",
        urban_rural="Urban",
        occupation_type="Gig Worker",
        income_type="Weekly",
        loan_amount_requested=25000.0,
        loan_tenure_months=12,
    )
    app_res = await create_applicant(applicant_req)
    app_id = app_res.applicant_id
    print(f"   ✅ Created Applicant: {app_id}")

    # 7. Consent Submission
    print("\n7. Testing Consent Management...")
    consent_req = ConsentSubmit(
        applicant_id=app_id,
        rent_data=True,
        utility_data=True,
        mobile_data=True,
        cashflow_data=True,
        business_data=False,
        self_declared_income=True,
    )
    c_res = await submit_consent(consent_req)
    assert c_res.sufficient_data is True
    print(f"   ✅ Consent submitted with {len(c_res.consented_categories)} categories: {c_res.consented_categories}")

    # 8. Alternative Credit Scoring
    print("\n8. Testing Alternative Credit Scoring Engine...")
    score_res = await generate_score(ScoreRequest(applicant_id=app_id))
    print(f"   ✅ FairScore Generated: {score_res.score} / 900")
    print(f"   ✅ Risk Grade: {score_res.risk_grade}, Confidence: {score_res.confidence}")
    print(f"   ✅ Default Probability: {score_res.default_probability:.2%}")
    print(f"   ✅ Recommendation: {score_res.recommendation}")
    print(f"   ✅ Reason: {score_res.recommendation_reason}")
    print(f"   ✅ Plain Language Summary: {score_res.summary_text}")

    # 9. SHAP Explainability
    print("\n9. Testing SHAP Explainability Engine...")
    expl_res = await get_explanation(app_id)
    print(f"   ✅ SHAP Positive Factors (lowering risk): {len(expl_res.positive_factors)}")
    for f in expl_res.positive_factors[:3]:
        print(f"      + {f['label']}: impact={f['impact']:.4f} | {f['reason']}")
    print(f"   ✅ SHAP Negative Factors (increasing risk): {len(expl_res.negative_factors)}")
    for f in expl_res.negative_factors[:3]:
        print(f"      - {f['label']}: impact={f['impact']:.4f} | {f['reason']}")

    # 10. List Applicants
    print("\n10. Testing Applicants Listing & Filtering...")
    listed = await list_applicants(limit=5)
    print(f"   ✅ Retrieved {len(listed['applicants'])} applicants from list. Total available: {listed['total']}")
    sample = listed['applicants'][0]
    print(f"      Sample: ID={sample['applicant_id']} | Occ={sample['occupation']} | Score={sample['score']} | Grade={sample['risk_grade']} | Rec={sample['recommendation']}")

    # 11. Thresholds update
    print("\n11. Testing Dynamic Threshold Update...")
    t_res = await update_thresholds(ThresholdUpdate(approve_max_default_prob=0.18))
    print(f"   ✅ Threshold updated: {t_res}")

    print("\n" + "=" * 60)
    print("🎉 ALL 11 TEST MODULES PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_all_tests())
