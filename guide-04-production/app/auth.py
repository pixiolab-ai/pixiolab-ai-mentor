import os

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

ISSUER = os.environ["OIDC_ISSUER"]        # exactly as your IdP writes it
AUDIENCE = os.environ["OIDC_AUDIENCE"]    # the identifier of this API
GROUPS_CLAIM = os.getenv("OIDC_GROUPS_CLAIM", "groups")
jwks = jwt.PyJWKClient(os.environ["OIDC_JWKS_URL"])  # caches the keys
bearer = HTTPBearer(auto_error=False)     # we choose the error codes


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(401, detail, headers={"WWW-Authenticate": "Bearer"})


def current_user(
        creds: HTTPAuthorizationCredentials | None = Depends(bearer)) -> dict:
    if creds is None:
        raise _unauthorized("Missing bearer token")
    token = creds.credentials
    try:
        key = jwks.get_signing_key_from_jwt(token)
        claims = jwt.decode(
            token, key.key, algorithms=["RS256"],  # never trust the header
            audience=AUDIENCE, issuer=ISSUER,
            options={"require": ["exp", "iss", "aud", "sub"]})
    except jwt.PyJWKClientConnectionError:
        raise HTTPException(503, "Identity provider unreachable")
    except jwt.PyJWTError:
        raise _unauthorized("Invalid or expired token")
    return {"id": claims["sub"], "groups": groups_from(claims)}


def groups_from(claims: dict) -> list[str]:
    """Group IDs from a verified token. Missing or malformed means none."""
    if GROUPS_CLAIM in claims.get("_claim_names", {}) or claims.get(
            "hasgroups"):
        # Microsoft Entra ID "overage": too many groups for the token. Look
        # them up server-side (Microsoft Graph) or fail closed, as here.
        raise HTTPException(403, "Groups not in token; directory lookup "
                                 "is not configured")
    groups = claims.get(GROUPS_CLAIM, [])
    if not isinstance(groups, list):
        return []
    return [g for g in groups if isinstance(g, str)]
