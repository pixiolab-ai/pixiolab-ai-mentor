# Guide 04 · Production AI Features: the companion app

One small Python service that grows the six features of *Production AI Features* (Guide 04 of PixioLab AI Mentor): reliable model calls, cost tracking, prompt regression tests, scheduled jobs, a request queue and permission-aware retrieval. Every file the guide prints is here, complete, in the state it reaches by the end of the guide.

## What you'll learn

- How to wrap every model call in one layer that retries only what can recover, never runs past a time budget, and falls back to a second provider.
- How to log the tokens and cost of every call, chart them, and alert once a day before the budget runs out.
- How to gate prompt and model changes in CI with a golden dataset, code checks and a judge model.
- How to make scheduled jobs run once, retry, page someone out of band, and notice when they never ran.
- How to put a queue, a shared rate limiter and load shedding between users and the model.
- How to make retrieval search only what the signed-in user may read, and prove it with tests.

| Guide 04 chapter | Where it lives |
|---|---|
| Chapter 00 · Start here: the shared foundation | `requirements.txt`, `docker-compose.yml`, `app/config.py`, `app/llm/text.py` |
| Feature 01 · Retries with a fallback model | `app/llm/client.py`, `app/llm/metrics.py`, `ops/alerts.yml` |
| Feature 02 · Token usage and cost dashboard | `app/cost/`, `app/notify.py`, `dashboard.py`, `db/schema.sql` (`llm_usage`, `alerts_sent`) |
| Feature 03 · Regression tests for prompt and model changes | `prompts/`, `app/summarize.py`, `evals/`, `tests/test_regression.py`, `.github/workflows/evals.yml` |
| Feature 04 · Scheduled workflows | `app/jobs/`, `app/paging.py`, `db/schema.sql` (`job_runs`) |
| Feature 05 · Request queue with concurrency limits | `app/api.py`, `app/queue/`, `app/jobs/celery_app.py`, `db/schema.sql` (`jobs`) |
| Feature 06 · Permission checks before retrieval | `app/auth.py`, `app/rag/`, `db/schema.sql` (`chunks`, row-level security), `tests/conftest.py`, `tests/test_permissions.py` |
| Chapter 07 · Put it all together | `Procfile`, and this README's request walkthrough |

The build order in the guide is the dependency order here: every feature calls a model through Feature 01, Feature 02 costs everything built after it, Feature 03 protects the prompts, Feature 04 introduces Celery, Feature 05 reuses it, and Feature 06 runs on top of all of them.

## Architecture

```mermaid
flowchart LR
    client([Client]) -->|"POST /ask + bearer token"| api

    subgraph web["API process"]
        api["app/api.py<br/>ask(), enqueue(), get_job()"]
        auth["app/auth.py<br/>current_user()"]
        api --> auth
    end

    auth -.->|"signing keys (JWKS)"| idp[("Identity provider")]
    api -->|"INSERT jobs row"| pg[("Postgres + pgvector<br/>jobs, chunks, llm_usage,<br/>job_runs, alerts_sent")]
    api -->|"job ID"| redis[("Redis<br/>llm and celery queues,<br/>rate-limit counters")]

    subgraph llmw["llm workers"]
        worker["app/queue/worker.py<br/>run_job()"]
        limiter["app/queue/limiter.py<br/>wait_needed()"]
        ask["app/rag/answer.py<br/>handle_ask()"]
        search["app/rag/search.py<br/>search()"]
        tracked["app/cost/tracking.py<br/>tracked_generate()"]
        gen["app/llm/client.py<br/>generate()"]
        worker --> limiter
        worker --> ask
        ask --> search
        ask --> tracked
        tracked --> gen
    end

    redis --> worker
    limiter <-->|"shared counters"| redis
    worker -->|"claim, set status"| pg
    search -->|"filtered vector query<br/>as app_rag (RLS)"| pg
    tracked -->|"one llm_usage row"| pg
    gen -->|"primary"| claude[["Anthropic API"]]
    gen -.->|"fallback"| second[["Second provider"]]

    subgraph side["Beside the request path"]
        beat["Celery Beat<br/>app/jobs/celery_app.py"]
        jobs["jobs workers<br/>app/jobs/tasks.py, watchdog.py"]
        dash["dashboard.py (Streamlit)"]
        ci["CI: evals/run.py and<br/>tests/test_regression.py"]
    end
    beat -->|"scheduled tasks"| redis
    redis --> jobs
    jobs --> pg
    jobs -->|"report, alert, page"| out[["Slack, email, heartbeat monitor"]]
    dash -->|"reads llm_usage"| pg
    ci -->|"scores prompts/ via generate()"| gen
```

Three rules shape the picture. The API never calls a model: it records a job and returns. Postgres holds every piece of state (jobs, usage, runs, alerts, chunks), and Redis only carries job IDs and rate-limit counters. And every model call, from a user's question or a scheduled report or a CI eval, goes through the same `generate()`.

## How a request flows

One `POST /ask` from a user in the sales group, end to end. This is Chapter 07's "Follow one request", with the file and function for each step.

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant A as API (app/api.py)
    participant P as Postgres
    participant R as Redis
    participant W as llm worker
    participant M as Model providers
    C->>A: POST /ask with question and bearer token
    A->>A: current_user() verifies the token, reads its groups
    A->>R: LLEN llm, refuse with 503 past MAX_QUEUE_DEPTH
    A->>P: INSERT jobs row (owner, question, token groups)
    A->>R: run_job.delay(job_id), after the commit
    A-->>C: 202 with job_id and status_url
    R->>W: deliver llm.run_job(job_id)
    W->>P: claim the row (status queued to running)
    W->>R: reserve requests and input tokens for this minute
    W->>P: search() for chunks the groups may read
    W->>M: generate() with retries, a 45 s budget, a fallback
    M-->>W: answer and token usage
    W->>P: INSERT llm_usage (user, feature, model, tokens, cost)
    W->>P: UPDATE jobs SET status done, result
    C->>A: GET /jobs/{job_id}
    A->>P: SELECT where id and owner_id match
    A-->>C: status done, answer and sources
```

1. **The token is verified** (Feature 06). `current_user()` in `app/auth.py` reads the bearer token, fetches the signing key through the identity provider's JWKS endpoint (`jwt.PyJWKClient` caches keys), and calls `jwt.decode` with the algorithm fixed to RS256 in code, plus the expected audience and issuer, requiring `exp`, `iss`, `aud` and `sub`. `groups_from()` reads the groups claim: missing or malformed means no groups, and Microsoft Entra ID's "overage" pointer fails closed with 403. A bad token is 401; an unreachable identity provider is 503.
2. **The job is accepted or refused** (Feature 05). `ask()` in `app/api.py` passes the question and the *token's* groups to `enqueue()`; anything the client puts in the body is ignored. `enqueue()` sheds load first: past `MAX_QUEUE_DEPTH` jobs waiting in the `llm` list it answers 503 with `Retry-After: 30`. Otherwise it inserts a `jobs` row, commits, and only then calls `run_job.delay(job_id)`, so the worker can never receive an ID whose row doesn't exist yet. The client gets 202 and a status URL.
3. **A worker claims it** (Feature 05). `celery_app.py` routes every `llm.*` task to the `llm` queue, so model work has its own workers. `run_job()` in `app/queue/worker.py` claims the row with one `UPDATE ... RETURNING` that succeeds only for a queued job or a running one whose 11-minute lease has expired. A redelivered message for a finished job claims nothing and returns; one for a job still running elsewhere checks back in 60 seconds.
4. **It reserves room under the rate limit** (Feature 05). `wait_needed()` in `app/queue/limiter.py` reserves one request and an estimate of the input tokens in per-minute Redis counters that every worker on every machine shares (`INCRBY` and `EXPIRE` in one `MULTI/EXEC`). If either budget is full, it gives the reservation back and the job is requeued for the next window, without counting as a retry.
5. **Retrieval sees only allowed chunks** (Feature 06). The `ask` handler, `handle_ask()` in `app/rag/answer.py`, calls `search()` in `app/rag/search.py`. It embeds the question (`app/rag/embed.py`) and runs one SQL query whose `WHERE allowed_groups && groups` clause filters inside the vector search, with pgvector 0.8's iterative scan so a selective filter still returns k rows. It connects as the read-only `app_rag` role and sets `app.groups`, so the row-level security policy in `db/schema.sql` filters again even if a future query forgets the `WHERE`. The handler logs document IDs, never text, and HTML-escapes each chunk inside its `<doc>` tag so a document can't close the tag and inject instructions.
6. **The model is called safely** (Feature 01). `tracked_generate()` in `app/cost/tracking.py` calls `generate()` in `app/llm/client.py`, which sets one absolute deadline 45 seconds out. The primary gets that deadline minus a 15-second reserve. Inside it, `Retrying` (tenacity) retries only errors `is_retryable()` accepts: connection errors, timeouts, 408, 409, 429 and every 5xx, unless the server sent `x-should-retry: false`. It waits exactly `retry-after` when the server sends one, otherwise about 1, 2, 4 seconds with jitter, and it stops when four attempts are spent or the next wait would leave less than 5 seconds. Each attempt (`_claude_once()`) gets a timeout that shrinks to fit the time left, with the connect phase capped at 3 seconds. The answer is read with `claude_text()` from `app/llm/text.py`, which joins text blocks by type and raises `NoText` on a refusal, a `max_tokens` cut-off or an empty answer. A 4xx or `NoText` is raised to the caller at once: no retry, no fallback. Only a provider failure moves on to `_fallback_once()`, the same request in the second provider's API shape. If both are down, `LLMUnavailable` carries the server's suggested wait.
7. **The cost is recorded** (Feature 02). `record()` writes one `llm_usage` row with the user, the feature (`"ask"`), the model that actually answered, the four token counts and `cost_usd()` from `app/cost/pricing.py`, computed with `Decimal` from the per-million prices. Refusals get a row too, because `NoText` carries the usage. An unknown model's cost is `NULL` instead of an exception, and a database error is logged with the full row rather than failing a call you've already paid for.
8. **The worker records the outcome** (Feature 05). On success `set_job()` stores the answer and its sources on the job row. On `LLMUnavailable` the job goes back to `queued` and `self.retry()` re-runs it after the server's suggested wait (or 30 s, 60 s, 120 s ...), capped at 10 minutes and 5 retries. Any other exception marks the job failed instead of looping on a bug.
9. **The answer is delivered** (Feature 05). The client polls `GET /jobs/{id}`. `get_job()` selects by ID *and* owner, so someone else's job looks exactly like one that doesn't exist: 404.

### Beside the request path

- **CI regression gate (Feature 03).** A pull request that touches `prompts/`, `app/`, `evals/` or `requirements.txt` runs `.github/workflows/evals.yml`: `python -m evals.run` summarizes every case in `evals/golden.yaml` three times with `app/summarize.py` (prompt and settings from `prompts/ticket_summary.yaml`), scores each trial with code checks and then a judge (`evals/scoring.py`), and takes the majority. `tests/test_regression.py` compares `evals/results.json` with the `evals/baseline.json` committed on main and fails on any critical case, more than one net regression, or a judge average down by more than 0.25. `python -m evals.run --fallback` scores the fallback model the same way. The workflow has no `paths:` filter on purpose: it always reports, and decides inside whether to spend money.
- **Scheduled jobs (Feature 04).** One Celery Beat process puts three tasks on the default queue in the business timezone: the daily signups report at 08:00, Feature 02's budget check every 15 minutes, and the watchdog every 30 minutes. `daily_signups()` in `app/jobs/tasks.py` claims the day's run in one statement (`claim_run()` in `app/jobs/runs.py`), counts in SQL (`fetch_signup_metrics()`), lets the model write the words (`write_summary()`, through `tracked_generate()`), posts to Slack, records the outcome and pings an outside heartbeat. A failure marks the run failed so Celery's retry (60 s, 120 s, 240 s) can claim it again; after the last retry the `task_failure` handler pages by email (`app/paging.py`). The watchdog pages once per job per day for a run that never happened, and the missing heartbeat catches a dead Beat.
- **The dashboard (Feature 02).** `dashboard.py` reads the same `llm_usage` table every call writes to, so the spend chart, the budget alert and the per-user numbers always agree.

## File-by-file tour

| File | Responsibility |
|---|---|
| `app/config.py` | Every setting, read from the environment: database and Redis URLs, model IDs, the business timezone, the daily budget, `TRACK_USAGE`. |
| `app/llm/text.py` | `claude_text()` and `NoText`: read an answer by block type, raise a never-retried error for refusals, cut-offs and empty answers. |
| `app/llm/client.py` | `generate()`, the one entry point for model calls: `LLMRequest`/`LLMResult`, error classification, the total time budget, retries with tenacity, the OpenAI-SDK fallback adapter, `LLMUnavailable`. |
| `app/llm/metrics.py` | Prometheus counters for requests, retries (by error type), fallbacks and failures. |
| `app/cost/tracking.py` | `tracked_generate()`: `generate()` plus one `llm_usage` row per call, refusals included. |
| `app/cost/pricing.py` | Illustrative per-million-token prices and `cost_usd()` in `Decimal`; unknown models cost `None`. |
| `app/cost/budget.py` | `check_budget()`: alert once per business day at 80% of the daily budget, safe under concurrent checks. |
| `app/notify.py` | `slack_post()`, shared by the budget alert and the daily report. |
| `app/summarize.py` | The feature under test in Feature 03: `summarize_ticket()` with a JSON schema and a Pydantic model. |
| `app/jobs/celery_app.py` | The Celery app: queues, routing, late acks, prefetch, time limit, visibility timeout, and the Beat schedule. |
| `app/jobs/runs.py` | `claim_run()` and `set_run_status()`: the atomic, lease-based run claim for scheduled jobs. |
| `app/jobs/reports.py` | `fetch_signup_metrics()` (math in SQL, cut at business midnight) and `write_summary()` (words from the model). |
| `app/jobs/tasks.py` | The daily report task, the scheduled budget check, failure paging and the heartbeat ping. |
| `app/jobs/watchdog.py` | `check_jobs()`: page once a day for a scheduled job that missed its deadline. |
| `app/paging.py` | `page()`: an out-of-band alert by email that shares nothing with Slack. |
| `app/queue/worker.py` | The `llm.run_job` task: handler registry, the job claim, rate-limit requeue, retry on `LLMUnavailable`, job status. |
| `app/queue/limiter.py` | `take()` and `wait_needed()`: per-minute request and input-token budgets shared by every worker through Redis. |
| `app/api.py` | FastAPI: `POST /jobs`, `POST /ask`, `GET /jobs/{id}`, request bounds, load shedding, `/metrics` and the queue-depth gauge. |
| `app/auth.py` | `current_user()`: OIDC token verification and `groups_from()`. |
| `app/rag/embed.py` | `embed()` through an embeddings API, and `split_into_chunks()`. |
| `app/rag/ingest.py` | `ingest()` with the source's groups (refuses documents with none) and `set_permissions()` for sharing changes. |
| `app/rag/search.py` | `search()`: the permission filter inside the vector query, iterative scans, row-level security as a second lock. |
| `app/rag/answer.py` | `handle_ask()`, the queue handler for `/ask`: retrieve, log IDs only, escape, answer. |
| `db/schema.sql` | Every table, added feature by feature: `llm_usage`, `alerts_sent`, `job_runs`, `jobs`, `chunks`, the `app_rag` role and its policy. |
| `db/signups.sql` | A local stand-in for your product's `signups` table, which the daily report reads. |
| `prompts/ticket_summary.yaml` | The versioned prompt and every setting that affects output, model ID included. |
| `evals/golden.yaml` | The golden dataset: behaviors to check, key facts, critical cases. |
| `evals/scoring.py` | `judge()` (a different model, structured output) and `score_case()` (code checks first, then the judge). |
| `evals/run.py` | Runs every case three times, majority vote, writes `evals/results.json` (or `results-fallback.json`). |
| `tests/test_regression.py` | The regression gate: critical cases, net regressions, judge drop. |
| `.github/workflows/evals.yml` | The Feature 03 CI job. GitHub runs workflows only from a repository's root, so copy this folder into its own repository (or move the file) to use it. |
| `dashboard.py` | The Streamlit spend dashboard: by day, feature, model and user. |
| `ops/alerts.yml` | The Prometheus rule that fires when over 5% of calls use the fallback for 5 minutes. |
| `Procfile` | The five processes you run: API, llm workers, jobs workers, exactly one Beat, the dashboard. |
| `docker-compose.yml` | Postgres 17 with pgvector, and Redis, for local development. |
| `scripts/block_types.py` | Chapter 00 checkpoint: see thinking blocks before text in a real response. |
| `scripts/usage.py` | Feature 02 checkpoint: the four token counts a response reports. |
| `scripts/metrics_server.py` | Feature 01 checkpoint: serve the counters at `localhost:8001/metrics`. |
| `conftest.py` | Test setup: placeholder settings, closed-port base URLs, database skipping and schema bootstrap. |
| `tests/conftest.py` | The guide's canary documents, `fake_embed()` and the `seeded_db` fixture. |
| `tests/fixtures.py` | Shared fakes: an identity provider that signs real RS256 tokens, an in-memory Redis, an API test client. |
| `tests/test_llm_client.py` | Chapter 00 and Feature 01: block types, classification, retry-after, the time budget, fallback, request bodies. |
| `tests/test_cost.py` | Feature 02: exact cost, one row per call, refusals logged, database outages, the once-a-day alert. |
| `tests/test_evals.py` | Feature 03: summarizer settings, scoring, the judge request, majority vote, the gate end to end. |
| `tests/test_jobs.py` | Feature 04: run claims under races, business-day cut-offs, skipped second triggers, paging, the watchdog. |
| `tests/test_queue.py` | Feature 05: one job from POST to done, ownership, bounds, load shedding, retries, rate-limit requeue, leases. |
| `tests/test_auth.py` | Feature 06: valid and invalid tokens, algorithm confusion, groups, overage, an unreachable provider. |
| `tests/test_permissions.py` | Feature 06, as printed: the canary never reaches a user, a result or a prompt it shouldn't. |
| `tests/test_rag.py` | Feature 06: ingest rules, `/ask` uses the token's groups, escaping, and row-level security without the `WHERE`. |
| `requirements.txt`, `requirements-dev.txt` | Runtime packages as printed in Chapter 00 (see "Differences from the PDF"), plus `fakeredis` for tests. |

## Key design decisions and trade-offs

| Decision | Why | Where |
|---|---|---|
| SDK retries off (`max_retries=0`) | Both SDKs retry twice by default; under your own retries one request becomes up to twelve attempts. One loop, one place to reason about. | `app/llm/client.py` |
| Classify by HTTP status, ask two questions | Exception classes for 503, 504 and 529 have moved between SDK versions. "Is the provider the problem?" decides the fallback; "is another attempt worth it?" decides the retry. | `provider_failed()`, `is_retryable()` |
| Refusals and 4xx never retried or hidden | The same request fails the same way. A fallback would hide your bug, and a refusal is a product decision for each feature, not an outage. | `generate()`, `NoText` |
| One total time budget, with a fallback reserve | Per-attempt timeouts and tenacity's `stop_after_delay` don't bound the request: attempt two can start at second 31. One absolute deadline shrinks every timeout, and a `retry-after` longer than the time left hands over to the fallback instead of sleeping through the user's wait. | `TOTAL_BUDGET_S`, `out_of_budget()` |
| Fallback to a different provider | Two models from one provider tend to share outages. Feature 03's `--fallback` run tells you whether the fallback is good enough to serve. | `_fallback_once()` |
| Read output by block type | Current models can return thinking blocks first, and a refusal can have no text; `content[0].text` fails intermittently. | `claude_text()` |
| Cost computed once, at write time, in `Decimal` | Old rows keep the price they were billed at, any period can be recomputed from the stored tokens, and floats drift over millions of small sums. Prices are illustrative: copy yours from the provider's pricing page. | `app/cost/pricing.py` |
| Bookkeeping never fails a paid call | An unknown price is `NULL`, and a database outage is logged with the full row so it can be replayed. | `record()` |
| One business timezone | The daily report, the daily budget and "spend today" must agree on when today starts; UTC midnight is late afternoon in California. | `BUSINESS_TZ` |
| Alert state in the database, not memory | "Once per day" must survive restarts and two overlapping checks: the `alerts_sent` primary key lets the first insert win. The Slack post runs inside the transaction, so a failed post rolls back and the next check tries again. | `check_budget()`, `watchdog.py` |
| A judge from a different model | Models rate their own outputs generously and share their own blind spots. Calibrate the judge against your labels with TPR/TNR or Cohen's kappa (Guide 03 · Calibrating the judge). A missing score is `None`, never 0. | `evals/scoring.py` |
| Tolerances counted in cases, repeats by majority | With 40 cases one flip is 2.5 points, so a percentage threshold fails on noise. These are noise tolerances, not a significance test: for a model upgrade, use a paired test on the same cases (Guide 03 · Regression testing). | `tests/test_regression.py`, `evals/run.py` |
| No `paths:` filter on the required check | A skipped required workflow stays "Pending" and blocks the merge. The job always runs and decides inside whether to spend money. | `.github/workflows/evals.yml` |
| Claim first, in one statement (idempotent runs) | Check-then-act lets two triggers a second apart both send the report. `INSERT ... ON CONFLICT DO UPDATE ... WHERE` serializes racing workers, and a lease longer than the time limit lets a dead worker's run be reclaimed. Exactly-once delivery across two systems isn't possible; the claim narrows the window to one rare failure. | `app/jobs/runs.py`, the `CLAIM` in `worker.py` |
| Code does the math, the model writes the words | SQL counts exactly; a model asked for a percentage change can be confidently wrong. | `app/jobs/reports.py` |
| Three failure detectors, alerts out of band | A crash retries then pages; a run that never happened is caught by the watchdog; a dead Beat is caught from outside by the missing heartbeat. Email shares nothing with Slack, so it arrives when Slack is what broke. Run exactly one Beat. | `tasks.py`, `watchdog.py`, `paging.py` |
| State in Postgres, IDs in Redis | Ownership checks and real 404s need a table, and Celery's result backend reports any unknown ID as "pending". The row is committed before its ID is queued. | `app/api.py`, `jobs` table |
| At-least-once delivery, made safe | `task_acks_late` plus `task_reject_on_worker_lost` requeue a job whose worker died, `worker_prefetch_multiplier=1` stops hoarding, and the visibility timeout (15 min) stays above the time limit and every retry countdown. The worker's claim makes a second delivery harmless. | `app/jobs/celery_app.py` |
| A rate limiter shared through Redis | Celery's `rate_limit` is per worker instance: three machines at 40/m send 120 a minute. A fixed window lets bursts straddle the boundary, so limits sit at about 80% of the account's. | `app/queue/limiter.py` |
| Bounded requests and load shedding | The client controls prompt size and `max_tokens`, and so your cost; past the maximum depth, a 503 with `Retry-After` beats letting Redis fill up. | `GenerateRequest`, `enqueue()` |
| Groups only from a verified token | Anyone can send `"groups": ["hr"]`. The algorithm list is fixed in code (no `alg` confusion), issuer and audience are checked, and an unreachable identity provider is 503, not 401. | `app/auth.py` |
| Filter inside the query, then lock it with RLS | Filtering the top five afterwards returns nothing when all five are restricted, and restricted text is already in memory. Row-level security catches a future query that forgets the filter, but table owners bypass it, so searches connect as `app_rag`. | `app/rag/search.py`, `db/schema.sql` |
| Default deny everywhere | No groups means no results, a document with no permissions is never indexed (and the `CHECK` constraint makes it unstorable), and an unset `app.groups` matches no rows. | `ingest()`, `search()`, the policy |
| Prove it with a canary | A made-up string in one HR document, and a fake model that echoes its prompt: if the canary isn't in the answer, it was never in the prompt. | `tests/test_permissions.py` |

## Run it

**Prerequisites:** Python 3.11 or later, Docker with Compose, an Anthropic API key. For the full set of features: an OpenAI API key (fallback provider and embeddings), a Slack bot token with `chat:write` (Features 02 and 04), an SMTP account (Feature 04), and an OpenID Connect identity provider that puts group IDs in access tokens (Feature 06).

```bash
cd guide-04-production
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt

cp .env.example .env          # fill in your keys; placeholders only in the example
set -a; . ./.env; set +a      # export the settings into this shell

docker compose up -d          # Postgres 17 + pgvector, and Redis
```

Postgres runs `db/schema.sql` only when its volume is first created. After you change the schema, recreate it with `docker compose down -v && docker compose up -d`. To search as the read-only role, give it a password and point `RAG_DATABASE_URL` at it:

```bash
docker compose exec postgres psql -U app -c "ALTER ROLE app_rag PASSWORD '<choose-one>'"
```

Then try each feature (each command makes real model calls):

```bash
python -m scripts.block_types       # Chapter 00: block types, then claude_text()
python -m scripts.metrics_server    # Feature 01: counters at http://localhost:8001/metrics
python -m scripts.usage             # Feature 02: token counts from one response
streamlit run dashboard.py          # Feature 02: the spend dashboard

python -m evals.run                 # Feature 03: 3 cases x 3 repeats x 2 calls
cp evals/results.json evals/baseline.json
python -m pytest tests/test_regression.py -s
python -m evals.run --fallback      # score the fallback model too
```

Features 04 to 06 run as the processes in the `Procfile`, one terminal each (or a process manager such as honcho or foreman):

```bash
celery -A app.jobs.celery_app worker -Q llm -c 8 -n llm@%h      # model jobs
celery -A app.jobs.celery_app worker -Q celery -c 2 -n jobs@%h  # scheduled jobs
celery -A app.jobs.celery_app beat                              # exactly one, always
uvicorn app.api:api --port 8000                                 # the API
```

`app/auth.py` reads `OIDC_ISSUER`, `OIDC_AUDIENCE` and `OIDC_JWKS_URL` when it is imported, so the API needs your identity provider's values. (The guide's Feature 05 uses a temporary development-only stub before Feature 06 replaces it; this repository holds the final version.) To give `/ask` something to search, ingest documents with the groups copied from their source system:

```python
from app.rag.ingest import ingest
ingest("handbook", open("handbook.txt").read(), ["all-staff"])
```

## Test it

```bash
python -m pytest                    # no database: the Postgres tests are skipped
```

With a throwaway database the whole suite runs. The tests truncate every table they touch, so never point them at data you want to keep:

```bash
docker compose exec postgres createdb -U app app_test
DATABASE_URL=postgresql://app:app@localhost/app_test python -m pytest
```

The tests create the schema in an empty database on first use. CI does the same against a `pgvector/pgvector:pg17` service container (see `.github/workflows/tests.yml` at the repository root).

**How the model APIs are mocked.** No test calls a real provider, and `conftest.py` sets placeholder keys and points both SDKs at a closed local port, so a call a test forgot to mock fails at once instead of spending money.

| Technique | Used for |
|---|---|
| Scripted fake SDK clients that return or raise what each test needs | Classification, retry-after, the time budget (with a fake clock), fallback, refusals |
| The real SDKs on an in-memory HTTP transport (`httpx2.MockTransport`) | Checking the exact request bodies each provider would receive, including `output_config` and `response_format` |
| A fake `tracked_generate()` or `generate()` that returns a fixed `LLMResult` | Cost rows, queue handlers, the summarizer, the RAG answer path |
| An "echo" model that answers with its whole prompt | The canary test: proof that restricted text never reached the prompt |
| A fake identity provider that signs real RS256 tokens (`tests/fixtures.py`) | Running PyJWT's real signature, issuer, audience and expiry checks |
| `fakeredis` and deterministic bag-of-words embeddings | The queue, the limiter and the API without Redis; vector search without an embeddings API |
| Celery tasks run in-process with `.run()` and `.apply()` | Following a job or a scheduled run without starting workers |

**Against the real APIs.** The scripts and commands in "Run it" make real calls, and so do the guide's checkpoints, for example `LLM_MODEL=not-a-model python -m scripts.block_types` (fails at once with a 404, no retries) or `ANTHROPIC_BASE_URL=http://10.255.255.1` with no fallback set (`generate()` raises `LLMUnavailable` well within 45 seconds).

## Exercises

1. **Add a circuit breaker.** After, say, five provider failures in a minute, call `generate(..., skip_primary=True)` for 60 seconds instead of waiting on a provider you know is down. Test it with the fakes in `tests/test_llm_client.py`. This layer grows into Guide 05 · P05 Switchyard.
2. **Enforce budgets, don't just report them.** Check a user's month-to-date spend from `llm_usage` in `enqueue()` and refuse clearly once they reach their quota. Per-team budgets and routing are also part of Guide 05 · P05 Switchyard; caching savings are the point of Guide 05 · P03 Warmstart.
3. **Make the eval gate cost-aware.** Record cost and latency per case in `evals/run.py`, flag a change that is 2% better but three times dearer, and run candidate model IDs nightly as a Beat job. A full CI eval gate is Guide 05 · P02 Tripwire, and a calibrated judge is Guide 05 · P09 Plumbline.
4. **Push instead of poll, fairly.** Stream job status with server-sent events or a webhook, add a separate queue for paid users and a cap on jobs in flight per user.
5. **Evaluate retrieval per access level.** Measure recall@k for each group (filtering can quietly starve one group's results), and add a nightly permission-sync task that calls `set_permissions()`. Permission-aware hybrid search with citations is Guide 05 · P04 Sourcebook.

## Differences from the PDF

- The PDF prints some files in two states as the features build up. Feature 04 shows a first `app/jobs/celery_app.py`, and Feature 05 a temporary `app/auth.py` stub. This repository holds each file's final state: `celery_app.py` with Feature 05's settings and Feature 06's `app.rag.answer` include, and `app/auth.py` with Feature 06's real token checks.
- `requirements.txt` pins `numpy~=2.4` where Chapter 00 prints `numpy~=2.5`: NumPy 2.5 requires Python 3.12, and the guide supports 3.11. `~=2.4` still installs 2.5 or later on Python 3.12+.
- `docker-compose.yml` also mounts `db/signups.sql`, a stand-in for the product table that Feature 04 reads.
- `.env.example`, `conftest.py`, `tests/fixtures.py`, the extra test files and `scripts/` are not printed in the guide. They exist so the code runs and is tested end to end.
