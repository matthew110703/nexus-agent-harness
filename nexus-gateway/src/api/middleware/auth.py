"""Bearer token authentication and loopback bind enforcement."""

from __future__ import annotations

import hmac
import secrets
from typing import Optional

from fastapi import Header, HTTPException, status

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


def assert_loopback_host(host: str) -> str:
    normalized = host.strip().lower()
    if normalized not in LOOPBACK_HOSTS:
        raise ValueError(f"Nexus gateway must bind to loopback only (127.0.0.1 / ::1), got {host!r}")
    return normalized


def generate_bearer_token() -> str:
    return secrets.token_urlsafe(32)


def verify_bearer_token(expected_token: str, authorization: Optional[str]) -> bool:
    if not expected_token or not authorization:
        return False
    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return False
    supplied = parts[1].strip()
    if not supplied:
        return False
    return hmac.compare_digest(expected_token.encode("utf-8"), supplied.encode("utf-8"))


def make_auth_dependency(expected_token: str):
    async def require_bearer(authorization: Optional[str] = Header(default=None)) -> None:
        if not verify_bearer_token(expected_token, authorization):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"error": {"message": "Invalid or missing bearer token", "type": "authentication_error"}},
                headers={"WWW-Authenticate": "Bearer"},
            )

    return require_bearer
