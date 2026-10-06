# Health Research Intelligence Agent

A multi-agent system that researches questions over the health-research subset of arXiv
(`secemp9/arxiv-complete`) and topic slices of PubMed Central's open-access set, and writes an
evidence-checked report.

```
                                 USER QUESTION
                                       │
                         ┌─────────────▼─────────────┐
                         │  ORCHESTRATOR (LLM)       │  plans, delegates, re-delegates,
                         │  tools = the other agents │  runs independent agents in parallel
                         └─────────────┬─────────────┘
      ┌──────────────┬──────────────┬──┴───────────┬──────────────┬─────────────┬──────────────┐
      ▼              ▼              ▼              ▼              ▼             ▼              ▼
  DISCOVERY     LITERATURE       METHODS        TRENDS          GAPS        EVIDENCE      SYNTHESIS
  hybrid search  structured      value counts,  normalised      grounded     verifies      writes report
  → shortlist    extraction      cross-tabs     topic trends    gaps         claims by     from verified
                 (abstract +     → claims       → claims        → claims     CODE, rejects claims only
                 full text)                                                  overreach
      └──────────────┴──────────────┴──────┬───────┴──────────────┴─────────────┴──────────────┘
                                           │
                     ┌─────────────────────▼─────────────────────┐
                     │  COHORT: the scope the run counts over,   │  stored once, applied wherever
                     │  applied to every denominator             │  papers are counted
                     └─────────────────────┬─────────────────────┘
                                           ▼
            ┌───────────────────────────────────────────────────────────────┐
            │  RESEARCH INTELLIGENCE (code, no model calls)                 │
            │  claim STATE: supported · partially_supported · contradicted  │
            │               uncertain · not_reported · not_searched_enough  │
            │  contradictions between papers, and what separates the sides  │
            │  evidence strength · gap confidence · untried combinations    │
            │  RESEARCH OPPORTUNITIES: each with what supports it, what     │
            │  weakens it, what is unresolved, and where it came from       │
            │  PRECEDENT: has anyone already done it, searched across the   │
            │  whole corpus and graded direct · partial · adjacent · none   │
            └──────────────────────────────┬────────────────────────────────┘
                                           │
                     ┌─────────────────────▼───────────────────────────┐
                     │  RESOLUTION, before a word is written: claims   │
                     │  whose state rests on reading depth go back to  │
                     │  the papers, are read in full and re-measured   │
                     └─────────────────────┬───────────────────────────┘
                                           ▼
     SHARED RUN STATE (Postgres): shortlist · extractions · claims + states · opportunities · notes · events
                                           │
      ┌───────────────┬───────────────┬────┴──────────┬───────────────┬───────────────┐
      ▼               ▼               ▼               ▼               ▼               ▼
  FOLLOW-UPS      MAP VIEWS       DRAFTS          RESEARCHER      RE-READ         EXPORTS
  count again,    graph · mind    proposal or     its own         papers read     md · docx · bib
  read in full,   map · grid,     review, with    questions,      again in        · ris · csv · xlsx
  same checks     from the same   real papers     tested and      full, fields    · protocol · PRISMA
  as the report   records         cited           graded by code  re-coded        · burden · PACKAGE

      corpus: arXiv + PubMed Central (open access) · tsvector + pgvector (HNSW) · year denominators
              per corpus, so a trend is normalised against the corpus it was measured in
```

## What makes it more than RAG

- **Real agents with separate jobs.** Each agent has its own prompt, a small tool set, and an
  output contract (`finish` schema). The orchestrator is itself an agent whose tools are the other
  agents; it decides the order, gives each one specific instructions, can send an agent back
  (e.g. "broaden the search"), and runs independent agents (Methods ∥ Trends) in parallel threads.
- **LLMs read, code counts.** The only LLM reading of papers is structured extraction. Method
  counts, dataset prevalence, geography, trends and evidence checks are SQL/Python.
- **Claims are verified, not asserted.** Analysis agents propose claims with machine-checkable
  predicates (`"n of N papers use data from East Africa, share ≥ 0.3"`, `"topic share rose ≥1.5×
  between 2018–20 and 2023–25"`). The Evidence agent runs them, can fix a predicate once (logged),
  and rejects overreach. Synthesis may only state supported claims; the evidence table and
  reference list are generated by code, and invented citations are stripped.
- **Honest about the corpus.** `health_reason` records why each paper is in the subset; trends are
  normalised per 10k arXiv papers; the partial final year is excluded; "not reported in the abstract"
  is kept distinct from "absent".

## Quick start (Ubuntu)

```bash
cp .env.example .env              # add ANTHROPIC_API_KEY (and/or GROQ_API_KEY)
docker compose up -d db           # Postgres 17 + pgvector on localhost:5432
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m research_agent.cli init-db
python -m research_agent.cli scope            # measure the health subset first (no writes)
python -m research_agent.cli ingest --limit 3000   # quick dev load; drop --limit for everything
python -m research_agent.cli ask "What machine learning methods are used to forecast malaria, and what gaps exist for East Africa?" --out report.md
```

`--provider groq` or `--provider deepseek` switches the whole team to that provider; `--mode pipeline` runs the same agents in a fixed
order without the LLM planner (a useful baseline).

Research Opportunity Map: `python -m research_agent.cli map "your question" --provider deepseek --out map.md`.
A Protocol agent first defines the question-specific fields (e.g. forecast horizon, validation split,
probabilistic output) that every shortlisted paper is then extracted against, with evidence quotes checked
by code. Code computes what is established, emerging and missing, grades evidence strength and gap
confidence with listed reasons, and finds underexplored combinations. A Gap Reasoning agent explains each
gap through hypotheses that code tests against subgroup counts (supported / not supported / untestable),
and a Research Design agent proposes candidate studies tied to map items, with supporting and challenging
papers. References to unknown papers, gaps or untested hypotheses are removed and listed.

PubMed Central (second corpus, topic slices of the open-access subset):
`python -m research_agent.cli pmc-ingest "malaria forecasting" --from-year 2010`. Papers are tagged
`source='pmc'`, only licences allowing reuse (CC0/CC BY/CC BY-SA) are kept, and PMC gets its own
per-year denominators so trends normalise against the right corpus. Full text is fetched lazily from
NCBI, as arXiv full text is from Hugging Face. Set `NCBI_API_KEY` in `.env` for 10 requests/s.

How the agents avoid missing things:

- **Question-specific fields.** Every run (ask and map) starts with a protocol step that defines fields for
  the question, for example "uses entomological data" or "model family", on top of the general extraction
  form. Set `ASK_PROTOCOL=0` to skip it for ask runs.
- **Coverage probes.** Discovery compares, for each method or data type the question turns on, how many topic
  papers in the corpus mention it with how many were shortlisted, and adds relevant ones it missed.
- **Full-text search.** Fetched full texts are indexed, so mention checks and passage search see methods
  sections, not only abstracts.
- **Focused re-checks.** When papers mention a method or data type that their extracted record does not
  show, each gets one quoted yes/no question (`recheck_field`). Confirmed uses, with a verified quote, are
  added to the counts; background mentions are not. Set `RECHECK=0` to turn it off.
- **Number check.** After the report is written, every "n of N" that code cannot trace is measured, corrected
  or removed by an agent under code checks (`NUMBER_CHECK=0` turns it off).

Follow-up questions about a finished run (terminal or web page):

```bash
python -m research_agent.cli chat <run_id>                      # interactive; 'exit' to leave
python -m research_agent.cli chat <run_id> -q "Which papers are behind C12?" --about C12
```

The Follow-up agent answers from the run's own papers, quotes, claims and map. It can show the papers behind a
claim, count again for a subgroup or a wider definition, or read one paper. New counts become claims marked
as follow-ups; answers go through the same citation and number checks as reports; the report itself never
changes. Ask it to read a claim's abstract-only papers in full ("check C12 against the full texts") and
it downloads those papers, reads them, updates their records and counts again, reporting the count before
and after and which papers changed. The conversation is saved with the run. In the web page, a panel under the report does the same,
and clicking any id in the report (C12, G2, N1, H4, D1) starts a question about it.
API: `POST /runs/{id}/followups {"question": ..., "about": "C12"}`, then poll `GET /runs/{id}/followups`.

Prevalence meta-analysis: `python -m research_agent.cli meta <run_id> "rheumatoid arthritis" --out results/`.
Each of the run's counted papers is read for that condition alone: cases, sample size, setting, case
definition and the ten Hoy et al. (2012) risk-of-bias items. Every number must appear in a quote and every
quote is checked against the paper; a count that disagrees with its own percentage is listed, not pooled.
Code then pools one overall estimate per study and condition (logit proportions, DerSimonian-Laird random
effects, Hartung-Knapp-Sidik-Jonkman interval, I2, prediction interval), with subgroups by setting, case
definition and country and a sensitivity analysis without high-risk studies. It writes a Markdown section, a
CSV of every estimate with its quotes, and a forest plot; drafts made afterwards include the section, and the
data export gets a sheet per condition. API: `POST /runs/{id}/meta {"condition": ...}`, then
`GET /runs/{id}/meta?condition=...` and `GET /runs/{id}/meta/forest.png?condition=...`. It is a model-assisted
analysis of the loaded corpus, not a registered systematic review, and says so.

Your own documents: under "Your documents" in the run form, add a file (PDF, Word .docx, text, Markdown or HTML,
up to 20 MB) or a link, mark it as a paper or an unpublished report, and tick it for a run. A ticked document is
treated like any study: screened against the protocol in a systematic run (and never sampled out), put on the
shortlist in an agent-chosen run, always read in full, every value backed by a quote checked by code, counted in
claims and PRISMA (as "added by the user"), and cited as [UP12] with a reference that says it was supplied by
the user. A link to a PMC or arXiv paper already in the corpus attaches that paper instead of a copy. Documents
are private: every corpus search and count excludes them, agents cannot add them to a run, and only the owner
can list, download, attach or delete them. Deleting one removes it from every run that used it. A scanned PDF
without a text layer cannot be read; export it with OCR first. API: `POST /documents`, `GET /documents`,
`DELETE /documents/{id}`, `GET /documents/{id}/file`, and `documents` / `include_papers` on `POST /runs`.

Guideline library: a reference that reports compare against, never evidence that is counted.

    python -m research_agent.cli guideline-add --pmc PMC9876543 --issuer EULAR --condition "rheumatoid arthritis"
    python -m research_agent.cli guideline-add --file kenya_ra.pdf --issuer "Kenya Ministry of Health" \
        --condition "rheumatoid arthritis" --region Kenya --year 2021 --title "Rheumatology guidelines"
    python -m research_agent.cli guidelines                 # the library; `guidelines <id>` lists its recommendations
    python -m research_agent.cli guideline-compare <run_id> --condition "rheumatoid arthritis"

Each recommendation is recorded word for word, with its strength and evidence grade as written, and checked
against the guideline's text (tables included: EULAR and ACR put their recommendations in one). One not found
in the text is dropped, and a grade not found is left out. `guideline-compare` sets each recommendation against
the run's papers that state which treatments they used or studied (drug classes count their members), and drafts
made afterwards include the table. The Ask tab can search the library. Guidelines never enter a count, a claim
state or a pooled figure. API: `GET /guidelines`, `GET /guidelines/{id}`, `POST /runs/{id}/guidelines/compare`.

Guidelines are used automatically when a run is about a condition the library covers (every word of a
guideline's condition appears in the question or protocol). The report writer, the map's design writer, the
draft writer and the Ask tab get the relevant recommendations with markers such as [GL1.3] and write "According
to the 2022 EULAR recommendations, ... [GL1.3]". Code checks every marker against the library, flags a quote
that is not in the recommendation, renders the marker as [EULAR 2022, rec. 3], appends the comparison table and
lists every recommendation cited, word for word, under "Guidelines referred to".

Filling one field on purpose: `python -m research_agent.cli fieldpass <run_id> q_validation_split`. A
paper's record is written in one reading that must fill fifteen general fields and the run's
question-specific ones at once, so a detail stated once in a methods section loses and the field comes back
"not stated". Counting it then measures the reading rather than the literature: a malaria run found 33 of 45
studies with no validation design recorded, read four of them in full, and every one described a held-out
design in its text. This pass takes the papers that left one field blank, shows each full text, and asks
about that field alone, recording a value only with a passage checked against the text. A paper that
genuinely does not say stays blank, only that one field is written, and the report and its claims are left
as they are. The Gap and Follow-up agents can call it themselves (`fill_field_from_full_text`) and the Gap
agent is told to run it before counting a field that is mostly "not stated". The researcher cannot: the
choice of which papers to re-read is the user's, so it cannot pick the ones that suit its pattern.

The other way a field fails is worse, because it looks full. A category that merges two things the question
has to tell apart cannot be un-merged by reading harder: a run asking how often forecasts are benchmarked
against a *seasonality-aware* baseline found its field offered only `naive_or_seasonal_naive_baseline`, which
counts plain persistence and seasonal naive as one thing, and the question died there. A pass can define the
finer field and code every paper into it from full text, leaving the old field untouched beside it so nothing
already counted moves:

```bash
python -m research_agent.cli fieldpass <run_id> baseline_reference \
  --values "no_reference,other_models_only,naive_persistence,seasonal_naive_or_historical_expectance" \
  --definition "The reference forecast this study compares its accuracy against."
```

The protocol agent is now told that a category may join two names with "or" only when they are the same
practice under different names, and that the distinctions the question makes in its own words are the
distinctions its categories must make.

Re-reading a finished run: `python -m research_agent.cli reread <run_id> [--limit N] [--force]`. A field is
often empty only because the paper was read from its abstract, which carries no sentence to quote for it;
this reads the run's papers in full again and re-codes both the general and the question-specific fields,
then prints which fields moved. `--force` reads again even papers already read in full, for after the
reading instructions change. The report and its claims are left as they are.

### What is known, what is only unreported, and what is worth doing

Two things are computed for every run, with no model calls, so a report, an export and the map all say the
same thing and a reader can check why.

**Every verified claim carries a state**, not only a pass or fail against the bound its agent asserted:

| state | what it means |
|---|---|
| `supported` | measured, the bound holds, on an evidence base big enough to mean something |
| `partially_supported` | the bound holds, but on a thin base, a small subgroup or mostly abstracts |
| `contradicted` | measured on a base worth believing, and the bound fails |
| `uncertain` | fewer papers than `MIN_EVIDENCE_BASE` (10): decides nothing either way |
| `not_reported` | nothing in the corpus reports it, which is an observation about the corpus, not an absence in the field |
| `not_searched_enough` | papers mention it in their text without it reaching the extracted field, or the zero rests on abstracts |

The last two matter most. "0 of 32 papers report allele X" passes a rarity bound and reads in a report as
"the allele is absent from the region", when all it says is that no paper in this shortlist recorded it. The
state travels with the numbers behind it (papers, read in full, abstract only), the Gap and Methods agents
are told to respect it, and the report prints the table.

**Gaps, untried combinations, candidate designs and the researcher's confirmed findings become rows** in
`research_opportunities` rather than paragraphs inside one run's notes. Each carries the claims that support
it, the claims that weaken it (anything `contradicted`, `uncertain` or `not_searched_enough`), its papers and
nearest prior studies, what is still unresolved, what a study would need, and which agent and step computed
it. `GET /runs/{id}/opportunities`, `POST /runs/{id}/opportunities/{item_id} {"state": "dismissed"}`, and
`export <run_id> opportunities` for the CSV.

**Each opportunity is checked for precedent** against the whole corpus, not the shortlist it came from. A gap
computed from sixty papers says nothing about the other hundreds of thousands, so each one is searched for and
every nearby paper is graded by what was actually matched:

| verdict | what it means |
|---|---|
| `direct` | a paper already does this: read it before going further |
| `partial` | part of it is done, but not the part the opportunity turns on |
| `adjacent` | nothing does this, but related work sits next to it |
| `none` | nothing comes close, which may mean the question is new or badly worded |

No model is involved: the grade is a function of the terms the opportunity is written on and the candidate's
own text, and every verdict carries those terms so a reader can disagree with it. Two rules keep it honest. A
value that only says which way a field points (`code_or_data_available = yes`) contributes nothing, because
searching for the word "yes" would return every paper as precedent; the field's own words are used instead.
And an opportunity written as prose rather than on a field and a value is capped at `adjacent`, however good
the hits look, since a sentence cannot establish that a paper did the same thing.

```bash
python -m research_agent.cli precedent <run_id> [--item G2] [--candidates 40]
```

### The scope a run counts over

A question usually carries a scope: African studies, East African isolates, paediatric trials. A researcher's
charter could only ever say so in prose, where nothing enforced it: `scope` scores how close a proposed
QUESTION is to the goal and `out_of_bounds` matches phrases in a question's text, but neither touches which
papers a test counts. So a study from the wrong continent sat in every denominator until an agent happened to
read it, and the scope had to be re-written into each test predicate by hand, every time.

A **cohort** is that scope, stored on the run and applied in the one place paper records are built, so every
count downstream shares it: claims, the researcher's statistical tests, the map, the report, the exports.

```bash
python -m research_agent.cli cohort <run_id> --include "geography=kenya,uganda,tanzania,east africa"
python -m research_agent.cli cohort <run_id> --exclude "geography=india,china" --note "Africa only."
python -m research_agent.cli cohort <run_id>            # show what is stored, and what it drops
python -m research_agent.cli cohort <run_id> --clear
```

Papers that say nothing about a cohort field are **kept** by default and counted separately. Excluding
silence would drop papers for how they were read rather than for what they are, which is the read-depth trap
again, so the count kept that way is reported everywhere the cohort is: a cohort resting on thirty unstated
papers is telling you to run a field pass, not to trust the filter. `--unstated exclude` overrides that. A
cohort is written on extracted fields, so it can only apply after extraction: reading is never filtered, and
excluding a paper removes it from the counting, not from the run.

### Thin evidence is sent back to the papers before the report is written

Naming a weak claim is not the same as doing something about it. Before synthesis writes a word, every claim
whose state rests on **reading depth** has its abstract-only papers read in full (the papers the count missed
first, since that is where under-counting hides) and is measured again with the same predicate. A number can
only move because a paper turned out to say more than its abstract did.

A claim that is thin because the literature is thin is left alone and said to be: too few papers to decide is
not something reading can mend. A claim can also get worse, and `partially_supported` becoming `contradicted`
is the system working, so it is recorded exactly like an improvement. What comes out is a resolution record
in the report: the state before, the state after, the counts, the papers read, and which claims are still
thin afterwards, which is the honest answer to "how much of this rests on abstracts?".

The pass is bounded twice, by claims and by papers over the whole pass, so a run's cost stays predictable:
`RESOLVE_MAX_CLAIMS` (4), `RESOLVE_PAPERS_PER_CLAIM` (8), `RESOLVE_MAX_PAPERS` (defaults to `MAX_FULLTEXT`).
`RESOLVE_EVIDENCE=0` turns it off.

### Purposive or systematic: how the papers are found

By default an agent searches, reads results and **chooses** up to `MAX_SHORTLIST` papers. That is quick and
right for exploring a question, but every number it produces is a share of what the agent picked, and its
gaps can be overturned by papers it never added.

A **systematic search** replaces the choice with a procedure anyone can repeat:

1. **Identify.** The protocol writes a `search_query` for the question itself (concepts joined with AND,
   synonyms with OR, checked with `corpus_count` to land roughly between 30 and 600 papers) before any result
   is read. Every paper it matches is logged as identified.
2. **Screen.** Each paper's title and abstract is judged against the protocol's inclusion and exclusion
   criteria, twenty to a call, and every decision is logged with its reason. A paper the abstract does not
   settle is kept for the full read, which is the safe error at this stage.
3. **Include.** Every eligible paper is analysed. Past `SYSTEMATIC_MAX_ANALYSED` (150), a **random sample**
   is analysed, drawn with a seed taken from the run and recorded, so the shares it produces estimate the
   eligible set without bias; the top of a relevance ranking would not. The same applies when more papers are
   identified than `SYSTEMATIC_MAX_SCREENED` (600) can be screened.
4. **Check recall.** The question's nearest papers by meaning that the query did not match are listed in the
   report as what the query may have missed. They are not added, since that would make the set depend on
   judgement again; widen the query and run again instead.

No agent adds papers afterwards, the screening log and PRISMA counts describe a real search, and the report
opens with how the papers were found. The ceiling is the corpus: this is a systematic search of what is loaded
(open-access PMC topic slices and arXiv), and the report says "of the corpus", never "of the literature".

```bash
python -m research_agent.cli map "..." --systematic
python -m research_agent.cli ask "..." --systematic
```

On the web page it is the **Papers** choice under the run type: *Chosen by an agent* or *Systematic search*.

### A gap is checked against full texts, and against the papers nobody read

A gap is computed from the analysed papers, and its **corpus check** widens that to every paper on the topic.
Titles and abstracts alone are a weak test, because what gaps are about (a validation design, an interaction
term, a safety endpoint) is exactly what abstracts leave out. So the check now has two tiers:

- **Titles and abstracts across the whole topic**, as before, now also counting and naming the papers that
  mention the practice but were **not analysed** in this run. Each of those could overturn the gap.
- **Full texts of the topic's papers.** Up to `FULLTEXT_CHECK_MAX` (300) papers that best match the topic are
  fetched (arXiv from the local files, PMC from NCBI twenty to a request) and kept, so later runs on the same
  topic reuse them, then searched with the same query. Every match comes back with the passage around it,
  because a mention is not a finding: "we did not model the interaction" matches a search for an interaction
  and says the opposite. Absence is only ever stated over texts that were actually searched. `0` turns this
  tier off.

Both feed the gap's confidence grade (absent from 30 or more searched full texts raises it; full-text mentions
or five or more unread mentions lower it), the report prints the passages, and the gap reasoning agent sees
them and is told to read them before calling anything absent.

**Absence is stated over what was counted.** A sentence such as "No paper in the corpus reports a
rainfall-temperature interaction (0/91)" counts the 91 analysed papers and claims the corpus. The claims
layer already refused that wording; gap reasoning did not go through it. Now any absence sentence that names
the corpus, the literature or the field is narrowed to the analysed papers, and what the corpus check found
outside them is added: "No analysed paper reports... 10 further topic papers mention it in their title or
abstract and were not analysed."

### Untried combinations, and the null they are judged against

A pair of components that are each common and never appear together is a candidate research opportunity.
Whether it is really one depends on the null. Measuring expected co-occurrence as `n_a × n_b / N` assumes a
paper picks each component independently, and that is wrong whenever a corpus is two literatures that never
meet. A shortlist of 11 animal experiments and 9 human trials makes any animal-only component and any
human-only component look like a striking absence, with the smallest p-values in the section, because the
partition is clean. One such map proposed "women with an animal experiment" as its leading opportunity.

So the expectation is computed **inside** each level of each single-choice field and summed. The animal group
holds no women's trials and the human group no animal experiments, so the expectation is zero and the pair
falls below the threshold. If any single field accounts for two components never meeting, the pair goes,
exactly as one rival explanation is enough to drop a researcher's finding. A genuinely untried combination
survives, because the group where both components live still expects to see it, and the item records which
field that was.

Two things fall out of the same idea. Established items that are one fact under two names are folded
together: a population of `animal model` and a design of `animal or in vitro experiment` are the same
papers, and reported separately they make a run look like it found two things. Folding needs both
near-identical paper sets and a shared word between the values, because in a twenty-paper run unrelated
fields coincide by chance and folding those would hide real findings; the survivor carries the other wording
under `also_recorded_as`. And the watch list drops items whose rise could not be less convincing: one map
printed seventeen, every one at p = 1.00, which is the field inventory with a p-value stapled on.

### Do the papers back each other up?

Every association a paper reports, a driver moving an outcome with a quote code has checked, is set beside
every other paper's statement about the same driver and outcome:

```bash
python -m research_agent.cli corroborate <run_id> [--driver "testosterone therapy"] [--all]
python -m research_agent.cli export <run_id> -f corroboration
```

**Wordings meet.** Phrases naming one thing are grouped: identical once measurement noise is removed
("serum testosterone levels" is "testosterone"), an acronym of the other ("TRT"), one the other with a word
added, or very close in meaning by embedding (`CORROBORATION_SIMILARITY`, 0.88). The shorter phrase needs two
words of its own before it can absorb a longer one, so "testosterone", an endogenous level, never swallows
"testosterone therapy", a treatment. For outcomes, the measure is noise too: "malaria cases", "malaria
incidence" and "malaria prevalence" are one outcome, while "malaria mortality" stays apart, since death is not
a measure of how much disease there is. A qualified exposure is never merged with the plain one by any route:
"heavy rainfall" lowering malaria does not contradict "rainfall" raising it, so the two stay separate even
though the malaria driver list would join them. Every merge is listed. This replaced the malaria-only driver list in
`contradictions` too, which is what lets it work on any topic.

**Relations come from the extracted directions, not a model.** Same direction supports, opposite contradicts,
an effect against no effect disputes whether there is one, nonlinear or mixed qualifies, and the same
direction with one result not significant agrees in direction only.

**Support is counted in independent sources.** Papers sharing a named dataset, two or more authors, or a
citation are one source: two papers from one registry agreeing is one finding, not a replication. Support
from the same kind of study (human, animal, in vitro) is counted apart from support across kinds. Each
finding gets one verdict:

| verdict | what it means |
|---|---|
| `corroborated` | independent papers report the same direction, and none the opposite |
| `corroborated_across_systems_only` | the only independent agreement comes from a different kind of study |
| `repeated_by_related_papers_only` | papers agree, but share a dataset, authors or a citation: one source |
| `contested` | independent papers report it both ways; what separates the sides is listed |
| `contradicted` | one paper each way, no replication on either side |
| `qualified` | other papers report it as nonlinear or mixed |
| `not_addressed_elsewhere` | one paper reports it: a fact about this corpus, not evidence against it |

Agreement is not truth, since papers can share an assay or a publication bias, so a finding's corroboration
sits beside a claim's state and never raises it. The report prints the table, the follow-up agent has a
`corroborate` tool for "has anyone replicated this?", and the evidence package and workbook carry it. On the
web page it has its own **Agreement** tab: each finding with its verdict, filterable by verdict, opening to the
quote from every paper on each side. The run page also has a **Counting** panel to set the run's scope, and the
Export menu carries the evidence package, the agreement table and the opportunities. The
open-ended researcher does not get the tool: it works on a discovery half with the held-out half hidden, and
corroboration reads every paper.

### One field, one axis

A question-specific field that answers two questions at once loses one of them. The testosterone protocol's
`study_design_for_causal_inference` listed `animal_or_in_vitro_experiment` beside
`randomised_placebo_controlled`, so a rat study randomised to testosterone or vehicle had to be filed as one
or the other: it became "animal", its allocation was never recorded, and the novelty map then reported the
two never meeting as an untried combination. The protocol agent is now told to keep what was studied apart
from how it was allocated, and code enforces it: when a design enum mixes the two, the study-system values
move to their own field, `q_study_system` (human, animal, in vitro or ex vivo, in silico), and the design
field keeps allocation only, with its definition saying an animal experiment can be randomised. A value
naming two systems gives both a home. An existing run can get the split with a field pass:

```bash
python -m research_agent.cli fieldpass <run_id> q_study_system --values "human,animal,in_vitro_or_ex_vivo" \
    --definition "What the study was done in: human participants, animals, or cells or tissue outside a body."
```

### What papers found, and how the literature fits together

- **Reported results and associations.** Every paper's performance numbers (RMSE, AUC, accuracy...) and the
  effects it reports for drivers (rainfall, temperature, bed nets...) are extracted, each with a quote that
  contains the number or names the driver. Each number also carries its **unit** as written, and is converted
  to one canonical unit per metric before anything is pooled: 350 ng/dL and 12.1 nmol/L are the same
  testosterone level, and a median taken across both describes nothing. Values are pooled only within one
  metric and one unit; a unit the table cannot read keeps its quote and stays out of every median, and the
  count of those is reported. Total and free testosterone are separate metrics, because they differ by a
  factor of about fifty. A value outside what its metric can be (an R² of 2) is a reading error and is
  dropped rather than carried. Metrics only one paper reports are left out of the report's table, since a
  median of a single number summarises nothing, and counted instead; every value stays in the data export. `method_comparison` compares method families only inside the same
  paper (same data, same metric), with a sign test; `contradictions` finds drivers reported in opposite
  directions and what separates the two sides. Reports and maps append both as computed tables.
- **Citation graph.** Papers are matched in OpenAlex by DOI; citations among the analysed papers, the most
  cited studies, and snowballing (papers they cite, papers citing them) during discovery. Set
  `OPENALEX_API_KEY` (free) for a 10x larger daily allowance; `CITATIONS=0` turns it off.
- **Research versus burden.** `python -m research_agent.cli burden-fetch` downloads WHO estimates of malaria
  cases and deaths by country (or `burden-import file.csv` for other sources). Reports then compare each
  country's share of the analysed papers with its share of cases.
- **Systematic-review record.** Discovery logs every paper it sees; reports include PRISMA 2020 counts and a
  flow diagram; the protocol document lists question, sources, eligibility, data items and searches.

### Exports

```bash
python -m research_agent.cli export <run_id> -f docx      # md, docx, html, bib, ris, csv, xlsx, protocol,
                                                          # screening, burden, burden_chart, opportunities
python -m research_agent.cli export <run_id> -f package   # all of it, as one zip
```
The web page has the same list under the report (Export).

`package` is the **evidence package**: everything someone needs to check the run without asking for anything
else, in one archive. The report as written; every claim with the predicate code tested, the count, its state
and why that state; the opportunities with what supports and weakens each one and whether anyone has already
done it; one row per paper with the verified quote behind every field; the screening log and PRISMA flow; the
references as BibTeX and RIS. Its `MANIFEST.md` says what each file is, states the cohort the denominators
were counted over, and ends with what the package **cannot** tell you: that a count of zero is about this
corpus and not the literature, that a field's rate across abstract-read papers is partly a fact about reading
depth, and that PMC shares over time track what was loaded. A part that cannot be built is named as missing
rather than quietly left out.

### Map of a run: graph view and mind map

The web page opens a finished run in tabs: Report, Map, Ask (follow-ups) and Draft. The Map tab draws the run three
ways, built from the database with no model calls:

- **Graph** (like Obsidian's graph view) answers "how is this literature connected?": papers, the concepts they
  share, claims, gaps (drawn as rings), designs and disputed drivers. Concepts sit in one region per section
  (methods, data, settings, who or what was studied, interventions and outcomes, mechanisms and targets, design),
  each on a faint wash of its section's colour; papers settle between what they use. Labels never overlap: the
  most important are placed first and zooming in reveals more. Thin links join concepts used together more
  often than chance; dashed pink links join two common concepts that no paper here combines, with the number
  expected by chance. Double-click a node (or "Show only its neighbourhood") for a local view, one or two steps
  out, with a trail back to the whole graph. The year slider shows the literature up to a year; Play steps
  through the years.
- **Mind map** answers "what does it contain, what supports it, where does it disagree, where are the gaps?".
  Sections as above plus Evidence and Research gaps. Concepts show "3/12 papers"; each opens into its papers,
  labelled so they can be told apart ("Uganda · 2021 · LSTM · full text", the title on hover). Under each paper,
  "also in" lists the other branches it sits in; click one to go there. Double-click a branch to focus on it,
  with a trail back. "Show" steps from the overview to concepts, papers or everything. On a phone it becomes an
  indented outline.
- **Grid** sets any two fields against each other (methods against data types, drugs against organisms...):
  each cell counts the papers that use both, shaded by count. Pink cells are pairs nobody combined although
  both are common. Click a cell for its papers.

The detail panel beside both views can compare papers side by side (differences stand out), show the quoted
evidence for a concept, ask a follow-up about an item, or start a proposal draft on a gap. The data is at
`GET /runs/<run_id>/graph`.

### Drafts: research proposals and review manuscripts

A finished run (report or map) can be turned into a first draft:

- **Research proposal**: argues for a gap or direction (pick one of the run's gaps, untried combinations or
  designs, or describe your own) with background, the evidence gap, aims, proposed methods, expected outcomes,
  risks, ethics, a workplan table and a budget placeholder.
- **Review manuscript**: the run written up as a literature review: structured abstract, introduction,
  methods (from the review record), results with the PRISMA flow as Figure 1 and the computed tables, discussion
  and conclusions. It states that screening and extraction were automated.

Drafts cite the studies themselves, not only counts: each statement about the literature names the studies it
rests on and what they did or found (setting, method, data, reported results, from their verified records), a
section that cites too few studies is rewritten once, and a number attributed to a study must appear in that
study's own text or it is marked [unverified]. Citations are rendered for readers outside the tool, author-year
("Okello et al., 2021") or numbered (Vancouver), with a full reference list; Appendix A lists every count with
the studies it covers. Choose the style with `--style` or on the web page.

Checks on what is credited to a study, in reports, follow-up answers and drafts: places (countries, their
adjectives, known regions and cities) and method families named in a clause must appear in the study that clause
cites, and decimals or percentages must appear in the cited study's own text; anything else is marked
[unverified]. Counts written in words ("two of 39", "12 of the 49") are checked like digits. A count that comes out
rare triggers a re-check of the studies' own text even when the claim did not say "rare", and counting test
designs in the metric-names field (or covariates in the datasets field) is refused. Trends over PMC papers carry a
caveat (the PMC part of the corpus is topic slices) and are not contrasted with arXiv. When fewer than a quarter
of the studies were read in full, reports and drafts say so at the top.

Each run records the extraction schema version it was read with, so opening an older run after a schema change
(follow-ups, the map, drafts, exports) uses its own records instead of looking under the new version.

A planner agent first counts what the argument needs (new claims, verified by code), then each section is
written in its own call and the whole draft goes through the report's checks: untraced numbers are measured,
corrected or marked [unverified], and citations to papers outside the run are removed. In a proposal, aims and
methods are marked as the applicant's plan, and decisions only the applicant can make are left as
`[to be confirmed: ...]`. The run's report is never changed.

```bash
python -m research_agent.cli draft <run_id> --directions                 # the run's gaps and directions
python -m research_agent.cli draft <run_id> -t proposal --about G2        # or -d "your direction"
python -m research_agent.cli draft <run_id> -t review -f docx -o review.docx
python -m research_agent.cli draft --export <draft_id> -f bib             # md, docx, html, bib, ris
```
On the web page, the Draft panel sits under the follow-up questions. A draft costs about as much as writing
a report: one planning pass and one call per section, sharing a cached evidence pack.

### Open-ended researcher

A researcher works on a finished run or map on its own initiative, within a charter you set (goal, scope, what is
out of bounds). It keeps an agenda of questions and a notebook, works one question per cycle in the background,
and proposes patterns as specs code can count (a difference between groups, what goes with what, what is rare,
which method wins within the same papers, a change over time). It chooses the questions; code decides what counts.

Judgment, enforced by code:
- it sees only a **discovery half** of the studies; a finding must replicate on the **hidden half**, at a threshold
  that tightens with every finding it proposes (it learns the verdict, never the hidden counts);
- **subgroup tests** run automatically (corpus, read depth, period, leading places);
- a **trap library** catches known artifacts: field misfit, absence in abstracts, PMC topic slices, small cells,
  overlapping groups, groups from different places, text that contradicts a rarity count;
- a separate **critic** proposes rival explanations as testable specs; confounders are tested with a
  Mantel-Haenszel adjusted analysis;
- the **grade is computed** (strong, moderate, provisional, rejected) from those results. Findings are labelled as
  the researcher's own hypotheses and never enter a report.

Drift, prevented by code: the charter is re-read every cycle; new questions must hang under existing ones;
duplicates are refused; questions far from the charter (embedding similarity below `RESEARCH_SCOPE_MIN`) or
touching what is out of bounds wait for your approval; re-running a tested pattern returns the notebook's result;
a question that stalls or uses its cycles is parked; a supervisor reviews the agenda every few cycles; token
budgets per day and in total, a cycle limit, and pause / stop at any time.

```bash
python -m research_agent.cli research start <run_id> -g "Find under-explored directions in how malaria forecasts are validated" \
    --scope "forecast design, data, validation" --out-of-bounds "clinical treatment" --max-cycles 20
python -m research_agent.cli research status <researcher_id>
python -m research_agent.cli research approve <researcher_id> <question_id>     # or --decline
python -m research_agent.cli research pause|resume|stop <researcher_id>
python -m research_agent.cli research budget <researcher_id> --add-cycles 5 [--add-tokens 5000000]
```
A researcher that stops with "cycle limit reached" or "total budget used" has an agenda it was still working
on. `research budget` raises the limit and sets it going again from where it stopped, keeping its notebook,
its tested patterns and its held-out half. One that finished because its questions were answered is left
alone and told so, since more room would change nothing. The Research tab offers the same as a button when a
researcher has run out of room.
Its cycles run on the job worker (the web server runs one; or `python -m research_agent.cli worker`). On the web
page, the Research tab of a run starts one and shows its agenda, notebook and findings with their evidence trail.
It does not run code of its own yet (no sandbox): it works with counts, tests and reading.

### Model routing

Each step can use its own model: `MODEL_ROUTES=cheap=deepseek:deepseek-flash; strong=deepseek:<stronger model>`.
"cheap" covers extraction, re-checks and number checks; "strong" covers the report, gap reasoning, designs,
follow-ups and drafts; a step name (e.g. `synthesis=anthropic:<model>`) overrides its tier.

### Job queue

Runs, maps, follow-ups and drafts started from the web page are jobs in Postgres. The web server works them itself
(`EMBEDDED_WORKERS=1`). For more capacity, or to keep long runs going while the server restarts, run workers
separately and set `EMBEDDED_WORKERS=0`:

```bash
python -m research_agent.cli worker -c 2
```
A worker that dies mid-run is replaced and the run resumes from its last finished step.

Web UI and API: `uvicorn research_agent.api.main:app --reload`, then open http://127.0.0.1:8000 to ask a
question in the browser. Pick a mode: Pipeline (fixed order), Orchestrated (a lead agent plans), or
Opportunity map. The JSON API is the same: `POST /runs {"question": ..., "mode": "pipeline"|"orchestrated"|"map"}`
and poll `GET /runs/{id}` (agent timeline, token usage), `GET /runs/{id}/report`, `GET /runs/{id}/claims`.

### Tests (offline, no API keys, real Postgres)

```bash
pip install -r requirements-dev.txt
docker compose exec db createdb -U research research_test
TEST_DATABASE_URL=postgresql://research:research@localhost:5432/research_test pytest -q
```

The suite builds parquet fixtures with the dataset's real schema, ingests them, and runs full
orchestrated and pipeline runs with a scripted LLM that reacts to real tool results.

## Data notes

- `metadata` (1.6 GB) is scanned by DuckDB — straight from Hugging Face or from a local download.
  Downloading it once is faster if you'll re-ingest.
- `paper_text` (70 GB, raw LaTeX) is **never bulk-loaded**. Full text is fetched only for papers
  the Literature agent chooses to read, cleaned, split into sections and cached. Stubs,
  `\includepdf` wrappers and wrongly-selected publisher templates are flagged (`status`).
- Avoid `datasets` streaming for partial reads (the dataset card documents hangs/aborts); this code
  uses DuckDB throughout.
- The health filter is in `research_agent/ingestion/health_filter.py`. Run `scope` and read the
  random samples before a full ingest; tune the term list if precision looks off.
- More PMC topics load with `pmc-ingest` and any PubMed Central search; MeSH terms keep a slice precise:

      python -m research_agent.cli pmc-ingest '"drug resistance, microbial"[MeSH Terms] OR "antimicrobial resistance"' --from-year 2015 --limit 15000
      python -m research_agent.cli pmc-ingest '"mental disorders"[MeSH Terms] OR "mental health"[MeSH Terms]' --from-year 2015 --limit 15000
      python -m research_agent.cli pmc-ingest '"hormones"[MeSH Terms] OR "endocrine system diseases"[MeSH Terms]' --from-year 2015 --limit 10000
      python -m research_agent.cli pmc-ingest '"pharmacology"[MeSH Terms] OR "pharmacokinetics"[MeSH Terms] OR "drug interactions"[MeSH Terms] OR "mechanism of action"' --from-year 2015 --limit 15000

  Check a slice's size first with `--limit 1` (it prints how many articles match). `--limit 0` means no limit.
- The extraction form (schema `health-v4`) covers clinical, laboratory, pharmacology and epidemiology papers
  as well as AI and modelling ones: besides methods, data, datasets and places it records study designs,
  populations, organisms, interventions (drugs, therapies, exposures), mechanisms (of action or resistance),
  molecular targets and outcomes, each needing a quote from the paper like the other fields. Reported results
  include effect estimates (odds, hazard and risk ratios) and pharmacological measurements (MIC, IC50, Cmax,
  plasma AUC, half-life); those are listed but never ranked as better or worse. Runs read under `health-v3`
  keep their records and their form; new runs read their papers again under v4. The malaria burden comparison
  only appears in runs about malaria; other runs get a table of where their studies come from.

## Cost

Every run records its token usage (`runs.input_tokens/output_tokens`), and `python -m research_agent.cli
usage <run_id>` breaks it down per step with the share served from the prompt cache. Extractions are cached
across runs, so repeated questions on overlapping literature get cheaper. `MAX_SHORTLIST` and `MAX_FULLTEXT`
are the main cost levers; `RESOLVE_MAX_PAPERS` bounds the pre-report resolution pass.

**Prompt caching.** A step's system prompt and tool schema are identical on every call it makes, so both are
marked as cache breakpoints: extraction sends one large, identical prefix per paper, and an agent loop resends
a growing one every turn. The end of the conversation is marked **only** when a later call will read it back,
which means in an agent loop and never in a one-shot read. A cache write costs more than an ordinary input
token, so marking a single read's messages would charge a premium on the paper's own text — the bulk of the
run's spend — for an entry nothing ever reads. Context trimming happens in one large step and the result is
kept, because re-trimming a little each turn would change the cached prefix and throw it away. A cached prefix
a provider reports without being asked (DeepSeek, some Groq models) is recorded too, so cost is not
overstated. A step making many calls whose hit rate stays near zero is the signal that its prefix is not
actually stable.

## Layout

An LLM reads a paper in exactly one place, `tools/reading.py`, and writes a structured record with a quote
behind every value. Everything after that is SQL and Python over those records, so counts, trends, gaps,
map items and the researcher's statistics are computed, not generated.

```
papers ─ingestion/─→ Postgres+pgvector ─search─→ shortlist ─reading.py─→ extractions (quoted)
                                                                              │
                        ┌─────────────────────────────────────────────────────┤
                        ↓                        ↓                ↓           ↓
                 agents/ (a report)      opportunity/ (a map)  research/   tools/graph.py
                        │                        │          (own findings)  (map views)
                        └──→ claims, checked by code against the extractions ←┘
                                   counted over the run's cohort, and read deeper
                                   when their state rests on reading depth
```

```
research_agent/
  agents/        base loop, 7 specialists, orchestrator, report finalisation,
                 follow-up questions, number check, proposal and review drafts
  opportunity/   research opportunity map: protocol (per-run extra fields), agents, scoring, render
  research/      open-ended researcher: patterns (statistics), gauntlet (the checks a
                 finding must survive), researcher (agenda, tools, drift controls)
  tools/         search+shortlist, extraction+analysis, reading, re-checks, trends,
                 claims/evidence, reported results, citations, graph and mind map, burden, review record,
                 epistemics (what each claim's number is worth), opportunities (what is worth doing),
                 fieldpass (one field, read for on purpose, in the papers that left it blank),
                 cohort (the scope every count shares), precedent (has anyone done it already),
                 resolve (thin evidence read deeper before the report leans on it),
                 corroboration (which papers back each other up, in independent sources)
  ingestion/     health filter, DuckDB→pgvector loader, PMC topic slices, full-text fetch + cleaning
  llm/           provider seam: anthropic | groq | deepseek (same internal message format)
  db/            schema.sql: papers, extractions, runs, claims, drafts, researchers, jobs…
  api/           FastAPI + the single-page web app (api/static/index.html)
  jobs.py        queue: runs, maps, drafts and researcher cycles, worked by the API or `cli worker`
  runstate.py    one run's context: its papers, notes, token budget, extraction schema version
  cli.py
tests/           fixtures, scripted LLM, end-to-end tests (no API keys, real Postgres)
```

A run's work is kept in Postgres rather than in memory, so a run can be resumed, re-read (`reread`),
asked follow-up questions, drawn as a map, drafted from, and handed to the researcher, all after it ends.
