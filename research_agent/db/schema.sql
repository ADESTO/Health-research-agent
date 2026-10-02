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

-- Research opportunities: a gap, an untried combination, a candidate design or a researcher's finding, kept
-- as a row with what supports it, what weakens it and where it came from, so it outlives the report that
-- first mentioned it (tools/opportunities.py).
CREATE TABLE IF NOT EXISTS research_opportunities (
    id          bigserial PRIMARY KEY,
    run_id      uuid NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    item_id     text NOT NULL,                  -- G2, N1, D1, X1, R1, F3: its id within the run
    kind        text NOT NULL,                  -- gap | combination | design | finding | direction
    label       text NOT NULL,
    question    text,                           -- the research question it implies
    state       text NOT NULL DEFAULT 'open',   -- open | addressed | dismissed | superseded
    state_note  text,
    field       text,
    value       text,
    confidence  jsonb,                          -- the grade code gave it, with its reasons
    evidence    jsonb NOT NULL DEFAULT '{}',    -- supporting/weakening claims, papers, prior studies,
                                                -- unresolved questions, alternative explanations, methods
    provenance  jsonb NOT NULL DEFAULT '{}',    -- which agent and step computed it, from what
    created_at  timestamptz DEFAULT now(),
    UNIQUE (run_id, item_id)
);
CREATE INDEX IF NOT EXISTS research_opportunities_idx ON research_opportunities (run_id, kind, state);

-- Every verified claim also carries the STATE its measurement supports (tools/epistemics.py), with the
-- numbers behind it: a count of zero is "not reported in this corpus", never "absent from the field".
ALTER TABLE claims ADD COLUMN IF NOT EXISTS state text;
ALTER TABLE claims ADD COLUMN IF NOT EXISTS state_facts jsonb;
CREATE INDEX IF NOT EXISTS claims_state_idx ON claims (run_id, state);

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

-- Drafts written from a finished run: research proposals and review manuscripts
CREATE TABLE IF NOT EXISTS drafts (
    id         bigserial PRIMARY KEY,
    run_id     uuid NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    kind       text NOT NULL,                 -- proposal | review
    direction  text NOT NULL DEFAULT '',      -- the gap or direction it argues for
    about      text[],                        -- map items it builds on (G2, N1, D1...)
    status     text NOT NULL DEFAULT 'pending', -- pending | running | done | failed
    title      text,
    content_md text,
    meta       jsonb,                         -- audit, cited papers and claims, new claims
    created_at timestamptz DEFAULT now(),
    finished_at timestamptz
);
CREATE INDEX IF NOT EXISTS drafts_run_idx ON drafts (run_id, id);
-- the extraction schema version a run was read with (older runs keep their records after a schema change)
ALTER TABLE runs ADD COLUMN IF NOT EXISTS extraction_version text;

-- The scope a run's numbers are counted over (tools/cohort.py): include/exclude conditions on extracted
-- fields, applied in extraction._rows so every claim, test, map and export shares one denominator.
ALTER TABLE runs ADD COLUMN IF NOT EXISTS cohort jsonb;
ALTER TABLE drafts ADD COLUMN IF NOT EXISTS citation_style text NOT NULL DEFAULT 'author-year';

-- Screening log for systematic-review reporting (PRISMA): every paper discovery saw, and what happened to it.
CREATE TABLE IF NOT EXISTS screening (
    run_id   uuid NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    paper_id text NOT NULL,
    stage    text NOT NULL,          -- identified | included | excluded | duplicate | over_limit
    reason   text,
    found_by text,                   -- the search, similarity or citation step that surfaced it
    ts       timestamptz DEFAULT now(),
    PRIMARY KEY (run_id, paper_id)
);

-- Citation graph from OpenAlex (cached; refreshed after CITATION_MAX_AGE_DAYS)
CREATE TABLE IF NOT EXISTS openalex_works (
    paper_id          text PRIMARY KEY,
    openalex_id       text,               -- NULL: looked up, not found
    cited_by_count    int,
    referenced_works  jsonb,              -- OpenAlex ids this paper cites
    fetched_at        timestamptz DEFAULT now()
);
CREATE INDEX IF NOT EXISTS openalex_works_oa_idx ON openalex_works (openalex_id);

-- Disease burden estimates by country and year (WHO GHO or an imported CSV)
CREATE TABLE IF NOT EXISTS burden (
    iso3        text NOT NULL,
    year        int NOT NULL,
    cases       double precision,
    deaths      double precision,
    source      text NOT NULL DEFAULT '',
    PRIMARY KEY (iso3, year, source)
);

-- Durable job queue (runs, maps, follow-ups). Workers claim jobs with FOR UPDATE SKIP LOCKED.
CREATE TABLE IF NOT EXISTS jobs (
    id            bigserial PRIMARY KEY,
    kind          text NOT NULL,                 -- run | map | followup
    payload       jsonb NOT NULL,
    status        text NOT NULL DEFAULT 'queued', -- queued | running | done | failed
    attempts      int NOT NULL DEFAULT 0,
    max_attempts  int NOT NULL DEFAULT 2,
    run_at        timestamptz NOT NULL DEFAULT now(),
    locked_by     text,
    heartbeat_at  timestamptz,
    error         text,
    created_at    timestamptz DEFAULT now(),
    finished_at   timestamptz
);
CREATE INDEX IF NOT EXISTS jobs_queue_idx ON jobs (status, run_at, id);

-- Open-ended researcher: a charter, an agenda of questions, a notebook, tested patterns and graded findings
CREATE TABLE IF NOT EXISTS researchers (
    id            bigserial PRIMARY KEY,
    run_id        uuid NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,   -- the studies it works on
    charter       jsonb NOT NULL,                  -- goal, scope, out_of_bounds, success
    status        text NOT NULL DEFAULT 'active',  -- active | paused | stopped | finished
    status_note   text,
    split         jsonb NOT NULL,                  -- how studies are divided into discovery and held-out halves
    max_cycles    int NOT NULL DEFAULT 20,
    cycles_done   int NOT NULL DEFAULT 0,
    daily_tokens  bigint NOT NULL DEFAULT 2000000,
    total_tokens  bigint NOT NULL DEFAULT 10000000,
    tokens_used   bigint NOT NULL DEFAULT 0,
    provider      text,
    created_at    timestamptz DEFAULT now(),
    updated_at    timestamptz DEFAULT now()
);
CREATE TABLE IF NOT EXISTS research_agenda (
    id            bigserial PRIMARY KEY,
    researcher_id bigint NOT NULL REFERENCES researchers(id) ON DELETE CASCADE,
    parent_id     bigint REFERENCES research_agenda(id) ON DELETE SET NULL,
    question      text NOT NULL,
    why           text,
    status        text NOT NULL DEFAULT 'open',    -- open | answered | parked | pruned | needs_approval
    status_note   text,
    priority      int NOT NULL DEFAULT 50,
    cycles_used   int NOT NULL DEFAULT 0,
    stalled       int NOT NULL DEFAULT 0,          -- consecutive cycles without progress
    tokens_used   bigint NOT NULL DEFAULT 0,
    scope_score   real,
    created_at    timestamptz DEFAULT now(),
    updated_at    timestamptz DEFAULT now()
);
CREATE INDEX IF NOT EXISTS research_agenda_idx ON research_agenda (researcher_id, status);
CREATE TABLE IF NOT EXISTS research_notebook (
    id            bigserial PRIMARY KEY,
    researcher_id bigint NOT NULL REFERENCES researchers(id) ON DELETE CASCADE,
    agenda_id     bigint,
    cycle         int,
    kind          text NOT NULL,                   -- cycle | note | test | finding | checkpoint | park | usage | stop
    content       jsonb NOT NULL,
    ts            timestamptz DEFAULT now()
);
CREATE INDEX IF NOT EXISTS research_notebook_idx ON research_notebook (researcher_id, id);
CREATE TABLE IF NOT EXISTS research_tests (
    id            bigserial PRIMARY KEY,
    researcher_id bigint NOT NULL REFERENCES researchers(id) ON DELETE CASCADE,
    agenda_id     bigint,
    signature     text NOT NULL,
    spec          jsonb NOT NULL,
    description   text,
    result        jsonb NOT NULL,                  -- on the discovery half only
    ts            timestamptz DEFAULT now(),
    UNIQUE (researcher_id, signature)
);
CREATE TABLE IF NOT EXISTS research_findings (
    id            bigserial PRIMARY KEY,
    researcher_id bigint NOT NULL REFERENCES researchers(id) ON DELETE CASCADE,
    agenda_id     bigint,
    test_id       bigint REFERENCES research_tests(id) ON DELETE SET NULL,
    statement     text NOT NULL,
    grade         text NOT NULL,                   -- strong | moderate | provisional | rejected
    status        text NOT NULL,                   -- confirmed | provisional | rejected
    reasons       jsonb,
    evidence      jsonb,                           -- discovery, subgroups, traps, rivals, held-out, examples
    created_at    timestamptz DEFAULT now()
);
