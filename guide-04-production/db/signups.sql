-- A stand-in for your product database's signups table, which Feature 04's
-- daily report reads. Local development and tests only: in production the
-- report queries the real table.
CREATE TABLE IF NOT EXISTS signups (
  id         BIGSERIAL PRIMARY KEY,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS signups_created_at_idx ON signups (created_at);
