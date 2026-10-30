"""Demo script automation demonstrating the 6-scene hackathon pitch from PRD Section 25.

Scene 1: Traditional Rejection (Credit-invisible gig worker rejected by traditional bureau)
Scene 2: Alternative Data Entry & Consent (Consenting to rent, utility, mobile, cashflow)
Scene 3: Score Generation (FairScore 300-900, probability, risk grade, recommendation)
Scene 4: Explainability (SHAP positive/negative drivers with plain language summary)
Scene 5: Bias Audit (Group comparisons, rural missing data penalty, bias alerts)
Scene 6: Responsible Lending Decision (Approval with transparent terms & human-in-the-loop)
"""

import asyncio
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from api.main import (
    startup,
    create_applicant,
    submit_consent,
    generate_score,
    get_explanation,
    get_fairness_summary,
)
from api.schemas import ApplicantCreate, ConsentSubmit, ScoreRequest


async def run_demo_script():
    print("\n" + "=" * 70)
    print("🎬 FAIRSCORE HACKATHON LIVE DEMO WALKTHROUGH")
    print("=" * 70)
    await startup()

    # --- SCENE 1: Traditional Rejection ---
    print("\n📍 SCENE 1: TRADITIONAL REJECTION")
    print("----------------------------------------------------------------------")
    print("Applicant: Ravi, 28 years old, delivery agent in North Delhi.")
    print("Monthly gig income: ₹28,000 (steady weekly payouts).")
    print("Traditional Credit Bureau Check: NO HIT / CIBIL SCORE = -1 (Credit Invisible).")
    print("❌ Traditional Bank Decision: REJECTED due to lack of formal credit history.")

    # --- SCENE 2: Alternative Data Entry & Consent ---
    print("\n📍 SCENE 2: CONSENT-BASED ALTERNATIVE DATA ONBOARDING")
    print("----------------------------------------------------------------------")
    applicant_req = ApplicantCreate(
        age_group="26-35",
        gender_category="Male",
        region_category="North",
        urban_rural="Urban",
        occupation_type="Gig Worker",
        income_type="Weekly",
        loan_amount_requested=30000.0,
        loan_tenure_months=12,
    )
    app_res = await create_applicant(applicant_req)
    app_id = app_res.applicant_id
    print(f"Profile created in FairScore: {app_id}")
    print("Explicit granular consent collected for:")
    print("  [✔] Rent payment history (UPI receipts to landlord)")
    print("  [✔] Electricity & mobile utility bill regularity")
    print("  [✔] Mobile recharge patterns (uninterrupted connectivity)")
    print("  [✔] Cash-flow summary (4 weekly payouts/month, positive daily balance)")

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
    print(f"Consent Audit Receipt: {c_res.message} (Timestamp: {c_res.consent_timestamp})")

    # --- SCENE 3: Score Generation ---
    print("\n📍 SCENE 3: ALTERNATIVE CREDIT SCORE GENERATION")
    print("----------------------------------------------------------------------")
    score_res = await generate_score(ScoreRequest(applicant_id=app_id))
    print(f"🏆 FairScore: {score_res.score} / 900")
    print(f"Risk Grade: {score_res.risk_grade} | Confidence: {score_res.confidence}")
    print(f"Predicted Default Probability: {score_res.default_probability:.2%}")
    print(f"Decision Recommendation: 👉 {score_res.recommendation.upper()} 👈")
    print(f"Decision Reason: {score_res.recommendation_reason}")

    # --- SCENE 4: SHAP Explainability ---
    print("\n📍 SCENE 4: EXPLAINABILITY & BORROWER TRUST")
    print("----------------------------------------------------------------------")
    expl = await get_explanation(app_id)
    print("Plain-Language Summary for Underwriter & Borrower:")
    print(f"\"{score_res.summary_text}\"")
    print("\nTop Positive Factors Driving Score:")
    for i, factor in enumerate(expl.positive_factors[:3], 1):
        print(f"  {i}. {factor['label']}: {factor['reason']} (SHAP impact: {factor['impact']:.4f})")
    print("\nRisk Flags / Negative Factors:")
    for i, factor in enumerate(expl.negative_factors[:2], 1):
        print(f"  {i}. {factor['label']}: {factor['reason']} (SHAP impact: {factor['impact']:.4f})")

    # --- SCENE 5: Bias Audit Panel ---
    print("\n📍 SCENE 5: FAIRNESS & BIAS AUDIT PANEL")
    print("----------------------------------------------------------------------")
    fairness = await get_fairness_summary()
    print("Portfolio-wide Bias Monitoring across Demographic Slices:")
    for alert in fairness.bias_alerts[:3]:
        print(f"  ⚠ [{alert['severity'].upper()}] {alert['group_column'].replace('_', ' ').title()}:")
        print(f"     {alert['message']}")
    
    if "urban_rural" in fairness.group_metrics:
        ur_metrics = fairness.group_metrics["urban_rural"]
        print("\nUrban vs Rural Metric Comparison:")
        for row in ur_metrics:
            val = row.get("group_value")
            ar = row.get("approval_rate", 0) * 100
            score = row.get("average_score", 0)
            fnr = row.get("false_negative_rate", 0) * 100
            print(f"  - {val:12s} | Approval: {ar:.1f}% | Avg Score: {score:.0f} | False Rejection: {fnr:.1f}%")

    # --- SCENE 6: Responsible Outcome ---
    print("\n📍 SCENE 6: RESPONSIBLE OUTCOME & BUSINESS IMPACT")
    print("----------------------------------------------------------------------")
    print("✔ FairScore transformed a high-risk Bureau No-Hit into a verified, low-risk borrower.")
    print("✔ Underwriter has human oversight with mathematical SHAP backing.")
    print("✔ Lenders expand addressable market by 30%+ without increasing default rate.")
    print("✔ Compliance team monitors real-time disparate impact and mitigates rural bias.")
    print("=" * 70)
    print("🎯 DEMO SCRIPT COMPLETED SUCCESSFULLY!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(run_demo_script())
