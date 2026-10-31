"""
Security and Authentication Module for FairScore.

Provides:
- PBKDF2-HMAC-SHA256 password hashing and salt verification (standard library).
- Cryptographically signed URLSafeTimedSerializer tokens (using itsdangerous).
- Role-based access control (RBAC) dependencies for FastAPI:
  - 'applicant': Can only read/write their own records.
  - 'lender': Authorized underwriter/risk officer access to queues, dossiers, and governance.
- In-memory secure user store with default test accounts.
"""

import hashlib
import hmac
import secrets
import os
from typing import Optional, Dict, Any
from fastapi import Header, HTTPException, Depends, status
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadTimeSignature, BadSignature

# Secret key for token signing (override via environment variable in production)
FAIRSCORE_SECRET_KEY = os.environ.get("FAIRSCORE_SECRET_KEY", "fairscore-secure-prototype-key-3f7b2c9e1a8d4f0")
TOKEN_MAX_AGE_SECONDS = 86400  # 24 hours

_serializer = URLSafeTimedSerializer(FAIRSCORE_SECRET_KEY, salt="fairscore-auth-salt")

# -----------------------------------------------------------------------------
# Password Hashing Utilities (PBKDF2-HMAC-SHA256)
# -----------------------------------------------------------------------------

def hash_password(password: str) -> str:
    """Hash a plaintext password using PBKDF2-HMAC-SHA256 with a cryptographically secure salt."""
    salt = secrets.token_bytes(16)
    iterations = 100_000
    derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    # Format: iterations$salt_hex$hash_hex
    return f"{iterations}${salt.hex()}${derived.hex()}"


def verify_password(password: str, hashed: str) -> bool:
    """Verify a plaintext password against a stored PBKDF2 hash."""
    try:
        parts = hashed.split("$")
        if len(parts) != 3:
            return False
        iterations = int(parts[0])
        salt = bytes.fromhex(parts[1])
        stored_hash = bytes.fromhex(parts[2])
        derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
        return hmac.compare_digest(derived, stored_hash)
    except Exception:
        return False


# -----------------------------------------------------------------------------
# Token Management
# -----------------------------------------------------------------------------

def create_access_token(user_id: str, role: str, applicant_id: Optional[str] = None) -> str:
    """Create a signed, time-limited authentication token."""
    payload = {
        "user_id": user_id,
        "role": role,
        "applicant_id": applicant_id,
    }
    return _serializer.dumps(payload)


def decode_access_token(token: str) -> Dict[str, Any]:
    """Validate and decode an authentication token."""
    try:
        data = _serializer.loads(token, max_age=TOKEN_MAX_AGE_SECONDS)
        return data
    except SignatureExpired:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session has expired. Please sign in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except (BadTimeSignature, BadSignature):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token signature.",
            headers={"WWW-Authenticate": "Bearer"},
        )


# -----------------------------------------------------------------------------
# User Store & Default Accounts
# -----------------------------------------------------------------------------

# In-memory user database: username -> {user_id, username, password_hash, role, applicant_id}
_USER_STORE: Dict[str, Dict[str, Any]] = {}

def register_user(username: str, password: str, role: str, applicant_id: Optional[str] = None) -> Dict[str, Any]:
    """Register a new user in the auth store."""
    normalized = username.strip().lower()
    if normalized in _USER_STORE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Username '{username}' is already registered."
        )
    user_record = {
        "user_id": f"usr-{secrets.token_hex(6)}",
        "username": normalized,
        "password_hash": hash_password(password),
        "role": role,
        "applicant_id": applicant_id,
    }
    _USER_STORE[normalized] = user_record
    return user_record


def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """Verify user credentials and return user record if valid."""
    normalized = username.strip().lower()
    user = _USER_STORE.get(normalized)
    if not user:
        return None
    if not verify_password(password, user["password_hash"]):
        return None
    return user


def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve user record by user_id."""
    for u in _USER_STORE.values():
        if u["user_id"] == user_id:
            return u
    return None


# Pre-seed default lender and demo applicant accounts
def seed_default_accounts():
    if "lender_admin" not in _USER_STORE:
        register_user("lender_admin", "FairScoreUnderwriter2026!", "lender")
    if "lender_analyst" not in _USER_STORE:
        register_user("lender_analyst", "FairScoreAnalyst2026!", "lender")
    if "demo_applicant" not in _USER_STORE:
        register_user("demo_applicant", "ApplicantPass2026!", "applicant", applicant_id="APP-0001")


seed_default_accounts()


# -----------------------------------------------------------------------------
# FastAPI Security Dependencies
# -----------------------------------------------------------------------------

def get_current_user(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    """Extract and authenticate user from Authorization: Bearer <token> header."""
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header. Please sign in.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    parts = authorization.split(" ")
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization header format. Expected 'Bearer <token>'.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = parts[1]
    payload = decode_access_token(token)
    return payload


def require_lender(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """Enforce that the authenticated user possesses the 'lender' role."""
    role = current_user.get("role")
    if role != "lender":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access restricted: Underwriter/Lender role required."
        )
    return current_user


def require_applicant_ownership(applicant_id: str, current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """Enforce object-level authorization (BOLA prevention).
    
    Rules:
    - If user is 'lender', access is granted for underwriting review.
    - If user is 'applicant', their assigned applicant_id MUST match the requested applicant_id.
    - Otherwise, 403 Forbidden is returned.
    """
    role = current_user.get("role")
    if role == "lender":
        return current_user

    user_app_id = current_user.get("applicant_id")
    if role == "applicant" and user_app_id and user_app_id == applicant_id:
        return current_user

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail=f"Forbidden: You do not have permission to access records for applicant {applicant_id}."
    )
