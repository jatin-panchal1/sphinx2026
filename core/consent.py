"""Consent Management Module for FairScore.

Handles explicit, granular consent collection, storage, audit receipt generation,
and verification before alternative data is used in scoring.
Supports single-category or multi-category granular opt-in without forcing
a mandatory two-category requirement.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


class ConsentManager:
    """Manages applicant consent for data usage."""

    DATA_CATEGORIES: List[str] = [
        "rent_data",
        "utility_data",
        "mobile_data",
        "cashflow_data",
        "business_data",
        "self_declared_income",
    ]

    PURPOSE_STATEMENTS: Dict[str, str] = {
        "rent_data": "Rent payment history will be used to assess payment reliability and consistency.",
        "utility_data": "Utility bill payment records will be used to evaluate bill payment discipline.",
        "mobile_data": "Mobile recharge patterns will be used to assess digital payment behaviour.",
        "cashflow_data": "Cash-flow summary will be used to evaluate income stability and savings behaviour.",
        "business_data": "Business transaction records will be used to assess business continuity and revenue stability.",
        "self_declared_income": "Self-declared income information will be used to evaluate repayment capacity.",
    }

    def __init__(self) -> None:
        # applicant_id -> dict of consent details
        self.consent_records: Dict[str, Dict[str, Any]] = {}

    def create_consent_request(self, applicant_id: str, occupation_type: str) -> Dict[str, Any]:
        """Create a consent request showing relevant data categories.

        For non-shopkeepers, business_data is marked optional or excluded.
        """
        is_shopkeeper = "shopkeeper" in occupation_type.lower()
        categories = []
        for cat in self.DATA_CATEGORIES:
            if cat == "business_data" and not is_shopkeeper:
                continue
            categories.append({
                "category": cat,
                "label": cat.replace("_data", "").replace("_", " ").title(),
                "purpose": self.PURPOSE_STATEMENTS.get(cat, ""),
                "required": False,
            })

        return {
            "applicant_id": applicant_id,
            "occupation_type": occupation_type,
            "categories": categories,
            "minimum_required_categories": 1,
            "notice": "FairScore will only access and score data categories that you explicitly approve. You may select any combination or choose none.",
        }

    def submit_consent(self, applicant_id: str, consents: Dict[str, bool]) -> Dict[str, Any]:
        """Store or update consent for an applicant.

        Args:
            applicant_id: Unique applicant identifier
            consents: Mapping of category_name -> boolean approval

        Returns:
            Consent receipt dictionary
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        
        # Normalize and filter keys
        valid_consents = {
            cat: bool(consents.get(cat, False))
            for cat in self.DATA_CATEGORIES
        }
        consented_list = [k for k, v in valid_consents.items() if v]
        
        # Valid consent exists if at least 1 category is consented
        has_valid_consent = len(consented_list) >= 1
        sufficient = has_valid_consent

        record = {
            "applicant_id": applicant_id,
            "consents": valid_consents,
            "consented_categories": consented_list,
            "categories": consented_list,  # alias
            "sufficient_data": sufficient,
            "timestamp": now_iso,
            "consent_version": "v2.0",
            "status": "active" if consented_list else "revoked",
            **valid_consents,  # flat boolean keys for direct attribute access
        }
        self.consent_records[applicant_id] = record

        return {
            "applicant_id": applicant_id,
            "status": record["status"],
            "sufficient_data": sufficient,
            "consented_count": len(consented_list),
            "consented_categories": consented_list,
            "timestamp": now_iso,
            "receipt_id": f"REC-{applicant_id}-{int(datetime.now().timestamp())}",
            "message": (
                f"Consent recorded successfully for {len(consented_list)} categories."
                if has_valid_consent
                else "Zero categories consented. No alternative data will be accessed."
            ),
        }

    def check_consent(self, applicant_id: str, category: str) -> bool:
        """Check if applicant consented to a specific data category."""
        record = self.consent_records.get(applicant_id)
        if not record:
            return False
        return bool(record.get("consents", {}).get(category, False))

    def get_consented_categories(self, applicant_id: str) -> List[str]:
        """Get list of active consented categories for an applicant."""
        record = self.consent_records.get(applicant_id)
        if not record:
            return []
        return [k for k, v in record.get("consents", {}).items() if v]

    def get_consent_status(self, applicant_id: str) -> Dict[str, Any]:
        """Return the current consent status dict for an applicant."""
        if applicant_id in self.consent_records:
            return self.consent_records[applicant_id]
        return {
            "applicant_id": applicant_id,
            "status": "not_submitted",
            "categories": [],
            "sufficient_data": False,
            "consents": {},
        }

    def get_all_consents(self) -> Dict[str, Dict[str, Any]]:
        """Return all consent records."""
        return self.consent_records

    def get_consent_receipt(self, applicant_id: str) -> Optional[Dict[str, Any]]:
        """Return formatted consent receipt or None if not found."""
        record = self.consent_records.get(applicant_id)
        if not record:
            return None
        return {
            "receipt_id": f"REC-{applicant_id}",
            "applicant_id": applicant_id,
            "status": record.get("status"),
            "consents": record.get("consents"),
            "timestamp": record.get("timestamp"),
            "version": record.get("consent_version"),
            "data_coverage": self.get_data_coverage(applicant_id),
            "withdrawal_policy": "Consent may be withdrawn at any time prior to loan contract execution.",
        }

    def has_sufficient_data(self, applicant_id: str) -> Tuple[bool, str]:
        """Check if enough data categories are consented (any valid category accepted)."""
        categories = self.get_consented_categories(applicant_id)
        count = len(categories)
        if count >= 1:
            return True, f"Consent active for {count} categories: {', '.join(categories)}."
        else:
            return False, "No data categories consented. Alternative credit scoring requires at least 1 consented data category."

    def withdraw_consent(self, applicant_id: str, category: Optional[str] = None) -> Dict[str, Any]:
        """Withdraw consent for a category, or withdraw all if category is None."""
        if applicant_id not in self.consent_records:
            return {"status": "error", "message": f"No consent record found for applicant {applicant_id}."}

        record = self.consent_records[applicant_id]
        if category:
            if category in record["consents"]:
                record["consents"][category] = False
                record[category] = False
                record["consented_categories"] = [k for k, v in record["consents"].items() if v]
                record["categories"] = record["consented_categories"]
                record["sufficient_data"] = len(record["consented_categories"]) >= 1
                if not record["consented_categories"]:
                    record["status"] = "revoked"
        else:
            for cat in self.DATA_CATEGORIES:
                record["consents"][cat] = False
                record[cat] = False
            record["consented_categories"] = []
            record["categories"] = []
            record["sufficient_data"] = False
            record["status"] = "revoked"

        record["updated_at"] = datetime.now(timezone.utc).isoformat()
        return {
            "applicant_id": applicant_id,
            "status": record["status"],
            "remaining_categories": record["consented_categories"],
            "sufficient_data": record["sufficient_data"],
            "message": f"Consent updated. Category '{category}' withdrawn." if category else "All consent withdrawn.",
        }

    def get_data_coverage(self, applicant_id: str) -> float:
        """Return fraction of categories with active consent (0.0 to 1.0)."""
        categories = self.get_consented_categories(applicant_id)
        return round(len(categories) / float(len(self.DATA_CATEGORIES)), 4)
