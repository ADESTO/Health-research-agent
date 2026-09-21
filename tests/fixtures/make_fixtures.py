"""Build small parquet files that mimic the secemp9/arxiv-complete `metadata` and `paper_text` configs.

Same column names and types as the real dataset, synthetic content. Lets the whole pipeline run
offline (CI, or before you've downloaded anything)."""
from __future__ import annotations

import datetime as dt
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

TOPICS = [
    # (title template, abstract template, categories, years)
    ("Forecasting malaria incidence in {place} with {method} and climate covariates",
     "We forecast monthly malaria cases in {place} using {method} driven by rainfall and temperature "
     "from satellite products. Surveillance data from district health facilities were used. "
     "The model outperformed a seasonal baseline with lower RMSE. Validation was internal only.",
     "cs.LG stat.AP", range(2016, 2027)),
    ("{method} for pneumonia detection on chest X-ray images",
     "We train a {method} on the CheXpert and ChestX-ray14 datasets from the USA to detect pneumonia on "
     "chest X-ray images. The model reaches an AUROC of 0.91. External validation was performed on a "
     "second hospital dataset.",
     "eess.IV cs.CV", range(2017, 2027)),
    ("Early sepsis prediction from electronic health records using {method}",
     "Using the MIMIC-IV intensive care database of patients in the USA we predict sepsis onset with a "
     "{method}. Clinical variables from electronic health records are used. AUROC 0.85.",
     "cs.LG", range(2018, 2027)),
    ("A {method} for breast cancer histopathology classification",
     "We classify breast cancer histopathology whole-slide images with a {method}. Experiments on the "
     "BreakHis and TCGA datasets show improved accuracy. Limitation: single-centre data.",
     "eess.IV cs.CV q-bio.QM", range(2019, 2027)),
]
METHODS_BY_ERA = {2016: ["random forest", "LSTM"], 2019: ["convolutional neural network", "LSTM"],
                  2021: ["vision transformer", "transformer"], 2023: ["foundation model", "large language model"]}
PLACES = ["Kenya", "Uganda", "Brazil", "India", "Malawi", "Tanzania"]
NON_HEALTH = [
    ("Graph neural networks for traffic flow prediction", "We predict traffic flow on road networks.", "cs.LG"),
    ("Robust quadruped locomotion via reinforcement learning", "A robot learns to walk on rough terrain.", "cs.RO"),
    ("Next-to-leading order corrections to diphoton production", "We compute QCD corrections at the LHC.", "hep-ph"),
    ("Efficient transformers for long document summarisation", "We summarise long news documents.", "cs.CL"),
]


def _method_for(year: int, rng: random.Random) -> str:
    era = max(k for k in METHODS_BY_ERA if k <= year) if year >= 2016 else 2016
    return rng.choice(METHODS_BY_ERA[era])


def build(out_dir: Path, n_per_topic_year: int = 2, seed: int = 7) -> tuple[Path, Path]:
    rng = random.Random(seed)
    meta, text = [], []
    counter = 0

    def pid(year: int) -> str:
        nonlocal counter
        counter += 1
        return f"{str(year)[2:]}{rng.randint(1, 12):02d}.{counter:05d}"

    for title_t, abs_t, cats, years in TOPICS:
        for year in years:
            for _ in range(n_per_topic_year):
                method, place = _method_for(year, rng), rng.choice(PLACES)
                p = pid(year)
                title = title_t.format(method=method, place=place)
                abstract = abs_t.format(method=method, place=place)
                meta.append(_row(p, title, abstract, cats, year, rng))
                body = (f"\\documentclass{{article}}\\begin{{document}}\\title{{{title}}}\\maketitle"
                        f"\\begin{{abstract}}{abstract}\\end{{abstract}}"
                        f"\\section{{Introduction}} {title} is important for public health. % a comment\n"
                        f"\\section{{Data}} Data were collected in {place} between 2010 and 2020 from 42 sites."
                        f"\\section{{Methods}} We use a {method} trained with Adam. $y = f(x)$."
                        f"\\begin{{figure}}\\caption{{Overview of the {method} pipeline}}\\end{{figure}}"
                        f"\\section{{Results}} Performance improved over baselines. " + "We report calibration, "
                        "sensitivity analyses and subgroup results across all sites and seasons. " * 6 +
                        f"\\section{{Limitations}} The study uses retrospective data from a single country."
                        f"\\end{{document}}")
                text.append({"paper_id": p, "text": body, "main_file": "main.tex", "resolution": "single",
                             "n_tex_files": 1, "n_files_used": 1, "n_unused_files": 0,
                             "text_encoding": "utf-8", "text_sha256": "x", "title": title,
                             "abstract": abstract, "primary_category": cats.split()[0], "license": None})
    for year in range(2016, 2027):
        for title, abstract, cats in NON_HEALTH:
            meta.append(_row(pid(year), title, abstract + " " * 5 + "We report strong results on benchmarks "
                             "and discuss implications for practitioners.", cats, year, rng))
    # a stub row, like the real withdrawal stubs
    stub_id = meta[0]["paper_id"]
    text[0] = dict(text[0], text="%auto-ignore")
    _ = stub_id

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metadata").mkdir(exist_ok=True)
    (out_dir / "paper_text").mkdir(exist_ok=True)
    mpath = out_dir / "metadata" / "train-00000-of-00001.parquet"
    tpath = out_dir / "paper_text" / "train-00000-of-00001.parquet"
    pq.write_table(pa.Table.from_pylist(meta, schema=META_SCHEMA), mpath)
    pq.write_table(pa.Table.from_pylist(text), tpath)
    return mpath, tpath


META_SCHEMA = pa.schema([
    ("paper_id", pa.string()), ("title", pa.string()), ("authors", pa.string()), ("abstract", pa.string()),
    ("categories", pa.string()), ("primary_category", pa.string()), ("submitter", pa.string()),
    ("license", pa.string()), ("doi", pa.string()), ("journal_ref", pa.string()), ("comments", pa.string()),
    ("report_no", pa.string()), ("msc_class", pa.string()), ("acm_class", pa.string()), ("proxy", pa.string()),
    ("n_versions", pa.int32()), ("first_version_date", pa.timestamp("ms")),
    ("latest_version_date", pa.timestamp("ms")), ("oai_datestamp", pa.string()),
    ("oai_sets", pa.list_(pa.string())), ("arxiv_abs_url", pa.string()),
])


def _row(p, title, abstract, cats, year, rng):
    month = rng.randint(1, 8) if year == 2026 else rng.randint(1, 12)
    d = dt.datetime(year, month, rng.randint(1, 28), 12, 0)
    return {"paper_id": p, "title": title, "authors": "A. Author, B. Author", "abstract": abstract,
            "categories": cats, "primary_category": cats.split()[0], "submitter": "A", "license": None,
            "doi": None, "journal_ref": None, "comments": "10 pages", "report_no": None, "msc_class": None,
            "acm_class": None, "proxy": None, "n_versions": 1, "first_version_date": d,
            "latest_version_date": d, "oai_datestamp": d.date().isoformat(), "oai_sets": ["cs"],
            "arxiv_abs_url": f"https://arxiv.org/abs/{p}"}


if __name__ == "__main__":
    print(build(Path(__file__).parent / "data"))
