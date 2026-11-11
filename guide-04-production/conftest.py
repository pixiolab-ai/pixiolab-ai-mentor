"""Shared test setup for guide-04-production. pytest loads this file first.

1. Settings: placeholder keys and identity-provider values, so modules that
   read them at import time load. Tests never reach a real provider: the
   base URLs point at a closed local port, so a call that a test forgot to
   mock fails at once instead of spending money.
2. Postgres: tests that need a database are skipped when DATABASE_URL is not
   set. When it is set, the schema is created on first use. Point it at a
   throwaway database: the tests truncate every table they touch.
3. Paths: the app reads prompts/ and evals/ relative to this folder, as in
   the guide, so the tests run from here.
"""
import os
from pathlib import Path

import pytest

pytest_plugins = ["tests.fixtures"]  # fake identity provider, Redis, API client

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

os.environ["ANTHROPIC_API_KEY"] = "test-key-not-real"
os.environ["OPENAI_API_KEY"] = "test-key-not-real"
os.environ["ANTHROPIC_BASE_URL"] = "http://127.0.0.1:9"
os.environ["OPENAI_BASE_URL"] = "http://127.0.0.1:9"
os.environ.setdefault("OIDC_ISSUER", "https://idp.example.com/")
os.environ.setdefault("OIDC_AUDIENCE", "api://pixiolab-rag")
os.environ.setdefault("OIDC_JWKS_URL", "https://idp.example.com/jwks.json")
os.environ.setdefault("SLACK_BOT_TOKEN", "xoxb-test-not-real")

DATABASE_URL = os.getenv("DATABASE_URL")
TABLES = "llm_usage, alerts_sent, job_runs, jobs, chunks, signups"


def needs_db(item) -> bool:
    return "db" in item.keywords or "seeded_db" in item.fixturenames


def pytest_collection_modifyitems(config, items):
    no_db = pytest.mark.skip(
        reason="needs Postgres: set DATABASE_URL to a throwaway database")
    # The regression gate compares two eval runs. Before the first
    # `python -m evals.run` there is nothing to compare, so skip it; once
    # results exist, a missing baseline fails loudly, as the CI gate should.
    have_results = (ROOT / "evals" / "results.json").exists()
    no_results = pytest.mark.skip(
        reason="no eval results yet: run `python -m evals.run` first")
    for item in items:
        if needs_db(item) and not DATABASE_URL:
            item.add_marker(no_db)
        if item.path.name == "test_regression.py" and not have_results:
            item.add_marker(no_results)


def _run_sql_file(conn, path: Path) -> None:
    sql = path.read_text()
    role = conn.execute(
        "SELECT 1 FROM pg_roles WHERE rolname = 'app_rag'").fetchone()
    if role:  # roles belong to the whole server, not this database
        sql = "\n".join(line for line in sql.splitlines()
                        if not line.startswith("CREATE ROLE app_rag"))
    conn.execute(sql)


@pytest.fixture(scope="session", autouse=True)
def database_schema(request):
    """Create the tables in an empty test database, once per run."""
    if not DATABASE_URL or not any(needs_db(i) for i in request.session.items):
        return
    import psycopg
    with psycopg.connect(DATABASE_URL, autocommit=True) as conn:
        if conn.execute("SELECT to_regclass('llm_usage')").fetchone()[0] is None:
            _run_sql_file(conn, ROOT / "db" / "schema.sql")
        if conn.execute("SELECT to_regclass('signups')").fetchone()[0] is None:
            _run_sql_file(conn, ROOT / "db" / "signups.sql")


@pytest.fixture
def clean_db():
    """Empty every table before a test that reads or counts rows."""
    import psycopg
    with psycopg.connect(DATABASE_URL) as conn:
        conn.execute(f"TRUNCATE {TABLES}")
