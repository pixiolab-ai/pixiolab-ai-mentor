CREATE TABLE llm_usage (
  id                 BIGSERIAL PRIMARY KEY,
  created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
  user_id            TEXT NOT NULL,
  feature            TEXT NOT NULL,
  model              TEXT NOT NULL,
  fallback           BOOLEAN NOT NULL DEFAULT false,
  input_tokens       INTEGER NOT NULL,          -- uncached input
  output_tokens      INTEGER NOT NULL,
  cache_read_tokens  INTEGER NOT NULL DEFAULT 0,
  cache_write_tokens INTEGER NOT NULL DEFAULT 0,
  cost_usd           NUMERIC(12, 6)             -- NULL: no price set
);
CREATE INDEX ON llm_usage (created_at);
CREATE INDEX ON llm_usage (user_id, created_at);
CREATE INDEX ON llm_usage (feature, created_at);

CREATE TABLE alerts_sent (       -- at most one alert per kind per day
  kind    TEXT NOT NULL,
  day     DATE NOT NULL,
  sent_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (kind, day)
);

CREATE TABLE job_runs (
  job        TEXT NOT NULL,
  run_date   DATE NOT NULL,              -- the business day it covers
  status     TEXT NOT NULL,              -- running | sent | failed
  attempts   INTEGER NOT NULL DEFAULT 1,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (job, run_date)
);

CREATE TABLE jobs (
  id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  owner_id   TEXT NOT NULL,              -- from the verified token
  kind       TEXT NOT NULL,              -- generate | ask
  payload    JSONB NOT NULL,
  status     TEXT NOT NULL DEFAULT 'queued',  -- queued|running|done|failed
  result     JSONB,
  error      TEXT,
  attempts   INTEGER NOT NULL DEFAULT 0,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX ON jobs (owner_id, created_at);

CREATE EXTENSION IF NOT EXISTS vector;   -- pgvector 0.8 or later

CREATE TABLE chunks (
  id             BIGSERIAL PRIMARY KEY,
  doc_id         TEXT NOT NULL,
  chunk_no       INTEGER NOT NULL,
  content        TEXT NOT NULL,
  embedding      vector(1024) NOT NULL,  -- your embedding size
  allowed_groups TEXT[] NOT NULL
                 CHECK (cardinality(allowed_groups) > 0),  -- default deny
  UNIQUE (doc_id, chunk_no)
);
CREATE INDEX ON chunks USING GIN (allowed_groups);
CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops);

-- The API reads chunks as app_rag, a role that does not own the table,
-- so this policy applies to every query it runs, buggy ones included.
CREATE ROLE app_rag LOGIN;  -- set its password from your secret store
GRANT SELECT ON chunks TO app_rag;
ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;
CREATE POLICY read_by_group ON chunks FOR SELECT TO app_rag
  USING (allowed_groups && string_to_array(
           current_setting('app.groups', true), ','));
-- Unset app.groups gives NULL, and NULL matches no rows: default deny.
