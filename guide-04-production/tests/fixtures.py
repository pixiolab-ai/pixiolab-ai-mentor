"""Fakes shared by the Feature 05 and 06 tests (loaded by ../conftest.py).

- idp: a fake OpenID Connect provider. It signs tokens with a local RSA key
  and serves the public key through the same JWKS client app.auth uses, so
  the real verification code runs.
- fake_redis: an in-memory Redis for the queue, the limiter and the API.
- api: a FastAPI TestClient whose enqueue step records job IDs instead of
  sending them to Celery.
"""
import json
import time
from functools import cache

import pytest


@cache
def _signing_key():
    from cryptography.hazmat.primitives.asymmetric import rsa
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


class FakeIdP:
    KID = "test-key"

    def __init__(self, issuer: str, audience: str):
        self.issuer, self.audience = issuer, audience

    def public_jwk(self) -> dict:
        from jwt.algorithms import RSAAlgorithm
        jwk = json.loads(RSAAlgorithm.to_jwk(_signing_key().public_key()))
        return {**jwk, "kid": self.KID, "use": "sig", "alg": "RS256"}

    def token(self, kid: str = KID, **overrides) -> str:
        """A signed access token. Pass a claim as None to leave it out."""
        import jwt
        claims = {"iss": self.issuer, "aud": self.audience, "sub": "alice",
                  "exp": int(time.time()) + 300,
                  "groups": ["sales", "all-staff"]}
        claims.update(overrides)
        claims = {k: v for k, v in claims.items() if v is not None}
        return jwt.encode(claims, _signing_key(), algorithm="RS256",
                          headers={"kid": kid})

    def header(self, **overrides) -> dict:
        return {"Authorization": f"Bearer {self.token(**overrides)}"}


@pytest.fixture
def idp(monkeypatch):
    import app.auth as auth
    provider = FakeIdP(auth.ISSUER, auth.AUDIENCE)
    monkeypatch.setattr(auth.jwks, "fetch_data",
                        lambda: {"keys": [provider.public_jwk()]})
    return provider


@pytest.fixture
def fake_redis(monkeypatch):
    import fakeredis

    import app.queue.limiter as limiter
    server = fakeredis.FakeRedis()
    monkeypatch.setattr(limiter, "r", server)
    return server


@pytest.fixture
def api(monkeypatch, fake_redis, idp):
    from fastapi.testclient import TestClient

    import app.api as api_mod
    monkeypatch.setattr(api_mod, "r", fake_redis)
    queued = []
    monkeypatch.setattr(api_mod.run_job, "delay", queued.append)
    client = TestClient(api_mod.api)
    client.queued = queued
    return client
