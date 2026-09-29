-- Health Research Intelligence schema. {dim} is replaced with the embedding dimension.
CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------- corpus
CREATE TABLE IF NOT EXISTS papers (
    paper_id            text PRIMARY KEY,
    source              text NOT NULL DEFAULT 'arxiv',   -- arxiv | pmc
    title               text NOT NULL,
    abstract            text NOT NULL,
    authors             text,
    categories          text[] NOT NULL,
    primary_category    text,
    year                int NOT NULL,
    first_version_date  timestamptz,
    doi                 text,
    journal_ref         text,
    license             text,
    health_reason       text NOT NULL,          -- 'category:<cat>' or 'keyword:<term>' — why it is in the health subset
    tsv                 tsvector GENERATED ALWAYS AS (
                           setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
                           setweight(to_tsvector('english', coalesce(abstract, '')), 'B')
                        ) STORED,
    embedding           vector({dim})
);
CREATE INDEX IF NOT EXISTS papers_tsv_idx   ON papers USING gin (tsv);
CREATE INDEX IF NOT EXISTS papers_year_idx  ON papers (year);
CREATE INDEX IF NOT EXISTS papers_cats_idx  ON papers USING gin (categories);
-- HNSW index is created after bulk load (much faster) — see ingestion.load.build_indexes

-- Denominators for trend normalisation: ALL papers per source, year and primary category
-- (every arXiv paper, every open-access PMC article), not just the ones we ingested.
CREATE TABLE IF NOT EXISTS corpus_year_stats (
    source           text NOT NULL DEFAULT 'arxiv',
    year             int  NOT NULL,
    primary_category text NOT NULL,
    n_papers         int  NOT NULL,
    PRIMARY KEY (source, year, primary_category)
);

-- Cleaned full text, fetched lazily for shortlisted papers only
CREATE TABLE IF NOT EXISTS paper_fulltext (
    paper_id     text PRIMARY KEY REFERENCES papers(paper_id) ON DELETE CASCADE,
    status       text NOT NULL,                 -- ok | missing | stub | template_suspect
    clean_text   text,
    sections     jsonb,                          -- [{"heading": ..., "text": ...}]
    resolution   text,
    n_chars      int,
    fetched_at   timestamptz DEFAULT now()
);

-- Structured extraction per paper, cached across runs (keyed by schema version)
CREATE TABLE IF NOT EXISTS extractions (
    paper_id        text NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
    schema_version  text NOT NULL,
    source          text NOT NULL,              -- abstract | fulltext
    data            jsonb NOT NULL,
    model           text,
    created_at      timestamptz DEFAULT now(),
    PRIMARY KEY (paper_id, schema_version)
);

-- Question-specific fields defined by a run's protocol (e.g. forecast horizon, validation strategy).
-- Kept per run, separate from the cached base extraction, because every question defines its own fields.
CREATE TABLE IF NOT EXISTS protocol_extractions (
    run_id      uuid NOT NULL,
    paper_id    text NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
    source      text NOT NULL,                  -- abstract | fulltext
    data        jsonb NOT NULL,
    created_at  timestamptz DEFAULT now(),
    PRIMARY KEY (run_id, paper_id)
);

-- ---------------------------------------------------------------- runs (shared agent workspace)
CREATE TABLE IF NOT EXISTS runs (
    run_id       uuid PRIMARY KEY,
    question     text NOT NULL,
    status       text NOT NULL DEFAULT 'queued', -- queued | running | done | failed
    provider     text,
    model        text,
    report_md    text,
    error        text,
    input_tokens  bigint DEFAULT 0,
    output_tokens bigint DEFAULT 0,
    llm_calls     int DEFAULT 0,
    created_at   timestamptz DEFAULT now(),
    finished_at  timestamptz
);

CREATE TABLE IF NOT EXISTS run_papers (
    run_id    uuid NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    paper_id  text NOT NULL REFERENCES papers(paper_id),
    added_by  text NOT NULL,
    reason    text,
    score     real,
    PRIMARY KEY (run_id, paper_id)
);

-- Agent outputs ("notes") that other agents can read
CREATE TABLE IF NOT EXISTS run_notes (
    run_id   uuid NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    agent    text NOT NULL,
    content  jsonb NOT NULL,
    ts       timestamptz DEFAULT now(),
    PRIMARY KEY (run_id, agent)
);

-- Every agent turn / tool call, for the UI timeline and debugging
CREATE TABLE IF NOT EXISTS run_events (
    id       bigserial PRIMARY KEY,
    run_id   uuid NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    agent    text NOT NULL,
    kind     text NOT NULL,                      -- start | tool_call | tool_result | message | finish | error
    payload  jsonb,
    ts       timestamptz DEFAULT now()
);
CREATE INDEX IF NOT EXISTS run_events_run_idx ON run_events (run_id, id);

-- Claims proposed by analysis agents; verified deterministically by the Evidence layer
CREATE TABLE IF NOT EXISTS claims (
    id          bigserial PRIMARY KEY,
    run_id      uuid NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    agent       text NOT NULL,
    text        text NOT NULL,
    claim_type  text NOT NULL,                   -- prevalence | trend
    predicate   jsonb NOT NULL,
    status      text NOT NULL DEFAULT 'pending', -- pending | supported | unsupported | rejected
    result      jsonb,
    review_note text,
    created_at  timestamptz DEFAULT now()
);
CREATE INDEX IF NOT EXISTS claims_run_idx ON claims (run_id);

-- ---------------------------------------------------------------- migrations (safe to re-run)
-- Anything that touches a new column goes here, AFTER the ALTER that adds it: on an existing database
-- the CREATE TABLE IF NOT EXISTS statements above are skipped, so those columns do not exist yet.
ALTER TABLE papers ADD COLUMN IF NOT EXISTS source text NOT NULL DEFAULT 'arxiv';
CREATE INDEX IF NOT EXISTS papers_source_idx ON papers (source, year);
ALTER TABLE corpus_year_stats ADD COLUMN IF NOT EXISTS source text NOT NULL DEFAULT 'arxiv';
DO $$
BEGIN
    -- older databases keyed corpus_year_stats on (year, primary_category); widen it to include source
    IF EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'corpus_year_stats_pkey'
                 AND array_length(conkey, 1) = 2) THEN
        ALTER TABLE corpus_year_stats DROP CONSTRAINT corpus_year_stats_pkey;
        ALTER TABLE corpus_year_stats ADD PRIMARY KEY (source, year, primary_category);
    END IF;
END $$;

-- Full texts are searchable too (methods sections name models and data that abstracts leave out).
-- Generated from clean_text, so rows fetched before this column existed are indexed on migration.
ALTER TABLE paper_fulltext ADD COLUMN IF NOT EXISTS tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('english', left(coalesce(clean_text, ''), 400000))) STORED;
CREATE INDEX IF NOT EXISTS paper_fulltext_tsv_idx ON paper_fulltext USING gin (tsv);

-- Focused re-checks of papers the extraction may have missed: one quoted yes/no answer per paper and
-- concept. Confirmed ('yes' with a verified quote) values are merged into the paper's extracted field.
CREATE TABLE IF NOT EXISTS rechecks (
    run_id   uuid NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    paper_id text NOT NULL,
    field    text NOT NULL,
    concept  text NOT NULL,              -- normalised, sorted terms joined by '|'
    verdict  text NOT NULL,              -- yes | no | unclear
    value    text,
    quote    text,
    ts       timestamptz DEFAULT now(),
    PRIMARY KEY (run_id, paper_id, field, concept)
);

-- Follow-up questions about a finished run: the conversation, one row per message.
CREATE TABLE IF NOT EXISTS followups (
    id       bigserial PRIMARY KEY,
    run_id   uuid NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    role     text NOT NULL,                 -- user | assistant
    content  text NOT NULL DEFAULT '',
    status   text NOT NULL DEFAULT 'done',  -- pending | done | failed (assistant rows)
    meta     jsonb,                         -- focus item, new claim ids, audit
    ts       timestamptz DEFAULT now()
);
CREATE INDEX IF NOT EXISTS followups_run_idx ON followups (run_id, id);
