"""Feature 06, Step 2: groups come only from a verified token.

The fake identity provider (tests/fixtures.py) signs real RS256 tokens, so
PyJWT's real signature, issuer, audience and expiry checks run here.
"""
import time

import jwt
import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

import app.auth as auth


def verify(token: str | None) -> dict:
    creds = (None if token is None else
             HTTPAuthorizationCredentials(scheme="Bearer", credentials=token))
    return auth.current_user(creds)


def status(token: str | None) -> int:
    try:
        verify(token)
        return 200
    except HTTPException as exc:
        return exc.status_code


def test_a_valid_token_gives_its_subject_and_string_groups(idp):
    token = idp.token(sub="u1", groups=["hr", 5, "all-staff"])
    assert verify(token) == {"id": "u1", "groups": ["hr", "all-staff"]}


@pytest.mark.parametrize("make_token", [
    lambda idp: None,                                     # no token
    lambda idp: "not-a-jwt",                              # garbage
    lambda idp: idp.token(iss="https://evil.example/"),   # wrong issuer
    lambda idp: idp.token(aud="api://another-service"),   # minted for others
    lambda idp: idp.token(exp=int(time.time()) - 10),     # expired
    lambda idp: idp.token(exp=None),                      # no expiry at all
], ids=["missing", "garbage", "issuer", "audience", "expired", "no-exp"])
def test_bad_tokens_get_401(idp, make_token):
    assert status(make_token(idp)) == 401


def test_algorithm_confusion_is_rejected(idp):
    """An HS256 token signed with a guessable secret must not pass: the
    allowed algorithms are fixed in code, never read from the header."""
    claims = {"iss": idp.issuer, "aud": idp.audience, "sub": "x",
              "exp": int(time.time()) + 60}
    forged = jwt.encode(claims, "a-guessable-shared-secret-of-32-bytes",
                        algorithm="HS256", headers={"kid": idp.KID})
    assert status(forged) == 401


def test_missing_or_malformed_groups_mean_none(idp):
    assert verify(idp.token(groups=None))["groups"] == []
    assert verify(idp.token(groups="hr"))["groups"] == []


def test_entra_groups_overage_fails_closed(idp):
    token = idp.token(groups=None, _claim_names={"groups": "src1"})
    assert status(token) == 403


def test_unreachable_identity_provider_is_503_not_401(idp, monkeypatch):
    def down():
        raise jwt.PyJWKClientConnectionError("connection refused")
    monkeypatch.setattr(auth.jwks, "fetch_data", down)
    # A key ID the client hasn't cached forces a fetch from the provider.
    assert status(idp.token(kid="rotated-key")) == 503
