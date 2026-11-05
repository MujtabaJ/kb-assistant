-- PostgreSQL 16 + pgvector schema for the knowledge base assistant.
-- Embedding size: 1536 for OpenAI text-embedding-3-small; change it if you use another model.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS organizations (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS api_keys (
    key_hash    TEXT PRIMARY KEY,              -- store a SHA-256 hash, never the raw key
    org_id      TEXT NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    label       TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    revoked_at  TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS chunks (
    id          BIGSERIAL PRIMARY KEY,
    org_id      TEXT NOT NULL,                 -- every query filters on this (tenant isolation)
    chunk_key   TEXT NOT NULL,
    source      TEXT NOT NULL,
    title       TEXT NOT NULL,
    section     TEXT NOT NULL,
    position    INT  NOT NULL,
    content     TEXT NOT NULL,
    metadata    JSONB NOT NULL DEFAULT '{}',
    embedding   vector(1536) NOT NULL,
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', section || ' ' || content)) STORED,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (org_id, chunk_key)
);

CREATE INDEX IF NOT EXISTS chunks_org_source_idx ON chunks (org_id, source);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx        ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx  ON chunks USING hnsw (embedding vector_cosine_ops);

-- One row per question: powers analytics, cost tracking and quality review.
CREATE TABLE IF NOT EXISTS runs (
    id              BIGSERIAL PRIMARY KEY,
    org_id          TEXT NOT NULL,
    question        TEXT NOT NULL,
    answer          TEXT NOT NULL,
    refused         BOOLEAN NOT NULL,
    grounded        BOOLEAN NOT NULL,
    top_score       REAL,
    cited_sources   TEXT[],
    injection_flags INT NOT NULL DEFAULT 0,
    model           TEXT,
    input_tokens    INT,
    output_tokens   INT,
    cost_usd        NUMERIC(10, 6),
    latency_ms      INT,
    feedback        SMALLINT,                  -- +1 / -1 from the end user
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS runs_org_time_idx ON runs (org_id, created_at DESC);
