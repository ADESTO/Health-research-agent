"""Rules that define the *health research* subset of arXiv.

arXiv has no single "health" category, so we select in two passes and record which rule admitted
each paper (`health_reason`) — that provenance matters when the Gap agent later reasons about
what is or isn't in the corpus.

1. Category rule: papers whose categories include a core biomedical category.
2. Keyword rule: papers in computational categories (cs, eess, stat, q-bio, physics.soc-ph, econ)
   with real health focus: a health term in the TITLE, or at least MIN_DISTINCT_TERMS different
   health terms in the abstract. A single passing mention ("applications include healthcare")
   is not enough. Tuned after the first scope run on the real corpus (Sept 2026), where one-mention
   papers such as influence maximisation or model-checking tools were getting in.

The keyword list is deliberately conservative (precision over recall). Tune it by running
`python -m research_agent.cli scope` and inspecting samples.
"""
from __future__ import annotations

# Categories that are health/biomedical by definition
# (q-bio.QM was here at first, but it also holds a lot of non-health biophysics and ecology; it now
#  goes through the keyword rule like the other q-bio categories.)
CORE_CATEGORIES = ["physics.med-ph", "q-bio.TO"]

# Keyword rule strength: distinct health terms needed when none appears in the title
MIN_DISTINCT_TERMS = 2

# Category prefixes where we only keep a paper if it mentions a health term
KEYWORD_SCOPE_PREFIXES = ["cs.", "eess.", "stat.", "q-bio.", "physics.soc-ph", "econ."]

HEALTH_TERMS = [
    # care settings & people
    "clinical", "clinician", "clinicians", "patient", "patients", "hospital", "hospitals",
    "healthcare", "health care", "public health", "global health", "primary care",
    "intensive care", "icu", "emergency department",
    # disciplines
    "medical", "medicine", "biomedical", "epidemiology", "epidemiological", "radiology",
    "pathology", "histopathology", "oncology", "cardiology", "dermatology", "ophthalmology",
    "psychiatry", "psychiatric", "pharmacology", "pharmacovigilance", "surgery", "surgical",
    # conditions
    "disease", "diseases", "cancer", "tumor", "tumour", "malaria", "tuberculosis", "hiv",
    "covid-19", "covid", "sars-cov-2", "pandemic", "epidemic", "diabetes", "diabetic",
    "alzheimer", "dementia", "parkinson", "stroke", "cardiovascular", "sepsis", "pneumonia",
    "infection", "infectious", "mental health", "depression", "schizophrenia", "autism",
    "obesity", "hypertension", "neonatal", "maternal", "dengue", "cholera", "ebola",
    # data & tasks
    "electronic health record", "electronic health records", "ehr", "ehrs", "mimic-iii",
    "mimic-iv", "diagnosis", "diagnostic", "prognosis", "prognostic", "mortality prediction",
    "drug discovery", "drug-drug interaction", "vaccine", "vaccination", "chest x-ray",
    "mammography", "ecg", "electrocardiogram", "retinal", "fundus", "lesion", "lesions",
]


def _re2_escape(term: str) -> str:
    out = []
    for ch in term:
        out.append("\\" + ch if ch in r".^$*+?()[]{}|\-" else ch)
    return "".join(out).replace(" ", r"\s+")


def health_term_regex() -> str:
    """Case-insensitive, word-bounded RE2 pattern (DuckDB) with one capture group."""
    alternation = "|".join(_re2_escape(t) for t in sorted(HEALTH_TERMS, key=len, reverse=True))
    return rf"(?i)\b({alternation})\b"


def health_subset_sql(source: str, min_year: int, limit: int | None = None) -> str:
    """DuckDB SQL selecting the health subset from the `metadata` parquet config."""
    core = ", ".join(f"'{c}'" for c in CORE_CATEGORIES)
    scope = " OR ".join(f"starts_with(c, '{p}')" for p in KEYWORD_SCOPE_PREFIXES)
    rx = health_term_regex().replace("'", "''")
    lim = f"LIMIT {int(limit)}" if limit else ""
    return f"""
    WITH base AS (
        SELECT paper_id, title, abstract, authors, categories, primary_category,
               first_version_date, doi, journal_ref, license,
               string_split(trim(categories), ' ') AS cats,
               year(first_version_date) AS year
        FROM read_parquet('{source}')
        WHERE first_version_date IS NOT NULL
          AND year(first_version_date) >= {int(min_year)}
          AND abstract IS NOT NULL AND length(abstract) > 100
          AND coalesce(comments, '') NOT ILIKE '%withdrawn%'
    ),
    tagged AS (
        SELECT *,
            list_filter(cats, c -> list_contains([{core}], c)) AS core_hits,
            list_bool_or(list_transform(cats, c -> ({scope}))) AS in_scope,
            lower(regexp_extract(title, '{rx}', 1)) AS title_kw,
            list_distinct(list_transform(regexp_extract_all(title || ' ' || abstract, '{rx}', 1),
                                         t -> regexp_replace(lower(t), '\s+', ' ', 'g'))) AS terms
        FROM base
    )
    SELECT paper_id, title, abstract, authors, cats AS categories, primary_category, year,
           first_version_date, doi, journal_ref, license,
           CASE WHEN len(core_hits) > 0 THEN 'category:' || core_hits[1]
                ELSE 'keyword:' || coalesce(nullif(title_kw, ''), terms[1]) || ' (' || len(terms) || CASE WHEN len(terms) = 1 THEN ' term' ELSE ' terms' END
                     || CASE WHEN title_kw <> '' THEN ', in title' ELSE '' END || ')' END AS health_reason
    FROM tagged
    WHERE len(core_hits) > 0
       OR (in_scope AND (title_kw <> '' OR len(terms) >= {MIN_DISTINCT_TERMS}))
    {lim}
    """


def year_stats_sql(source: str, min_year: int) -> str:
    """All-arXiv paper counts per year and primary category (trend denominators)."""
    return f"""
    SELECT year(first_version_date) AS year, primary_category, count(*) AS n_papers
    FROM read_parquet('{source}')
    WHERE first_version_date IS NOT NULL AND year(first_version_date) >= {int(min_year)}
    GROUP BY ALL
    """
