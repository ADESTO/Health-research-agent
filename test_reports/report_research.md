## Summary

- **The field is dominated by classical statistics, but machine learning is rising fast.** Statistical/time-series regression appears in 48 of 100 shortlisted papers (48%) and Bayesian hierarchical/geostatistical methods in 35 of 100 (35%), while ML appears in 25 of 100 (25%) and mechanistic/hybrid models in 13 of 100 (13%); families overlap, so shares sum above 100% [C437][C439][C438][C440]. Whole-corpus malaria + ML language rose about 6.7-fold between 2010-2012 and 2023-2025 (0.107 → 0.717 per 10k) [C424][C452], while classical-statistics language fell to x0.65 [C425].
- **Climate and remote-sensing covariates are the established data backbone.** 80 of 100 shortlisted papers (80%) use climate or remote-sensing covariates [C434], and survey data (DHS/MIS-type) appears in 40 of 100 (40%) [C461]. Yet only 22 of 100 (22%) use satellite imagery or remote sensing as a modality [C460], and only 29 of 100 (29%) name any standard climate, population, satellite or survey product at all [C451].
- **Validation is the single biggest barrier to district-level use.** Only 3 of 73 papers stating a validation level (4.1%) report external validation [C441]. The extraction schema records only internal/external, so the *type* of hold-out (random vs temporal vs spatial, leave-one-district-out, leave-one-year-out) is **unmeasured** — a schema limitation, not a finding.
- **Intervention covariates are essentially absent.** None of the 100 shortlisted papers list ITN/bed-net, IRS/spraying or other intervention coverage as a data modality (0 of 100) [C449], and only 5 of 100 (5%) name an intervention dataset such as the Malaria Atlas Project [C443, unsupported as stated — see note below]. Whole-corpus intervention language fell to x0.51 [C428][C454].
- **Uncertainty is rarely quantified.** 23 of 100 papers (23%) report any interval/probabilistic metric [C442, unsupported as stated — see note below]; the rest report point metrics such as RMSE or MAE [C446].
- **Reproducibility is low and falling.** 8 of 62 full-text papers (12.9%) report code or data availability [C436]; only 15 of 100 (15%) state anything about availability, positive or negative [C448]. Whole-corpus reproducibility language fell to x0.41 [C430][C456].
- **Forecasting is a common framing but its horizon is unmeasured.** 57 of 100 papers (57%) carry a forecasting/early-warning/outbreak-prediction task [C458], but the schema has no field for forecast horizon or prediction admin level, so the share delivering genuine 1-6 month lead times **cannot be quantified**.
- **The arXiv-vs-PMC and East-vs-West comparisons are weak.** The shortlist is 16 arXiv and 84 PMC papers; arXiv malaria content is sparse, so any arXiv-specific trend is indicative only. The apparent East-vs-West difference in ML use is **not statistically supportable** (small n, many abstract-only reads) and is reported as suggestive only.

## Scope and method

This review synthesises a shortlist of **100 papers** (run facts: 100 shortlisted, 100 extracted, 62 read in full text, 38 abstract-only; shortlist years 2012-2026). The shortlist splits **16 arXiv preprints and 84 PMC articles**. The whole corpus searched contains 166,462 papers (78,639 arXiv, 87,823 PMC) spanning 2010-2027.

Two denominators recur and must be kept distinct:

- **Shortlist denominator = 100.** Most verified prevalence claims are computed over the full 100-paper shortlist by the verification engine.
- **Extracted-methods subset = 58.** The methods agent worked over 58 papers with extracted method fields; its "X of 58" figures are extraction counts and are **not reproducible by the verification engine**, which uses denominator 100. Where the two disagree, this report quotes the engine's verified count and flags the extraction count as unverified.
- **Full-text denominator = 62** (run facts). One reproducibility claim is verified specifically over full-text papers [C436].

Because the engine computes every prevalence over 100, claims phrased "X of 58" are denominator-unfaithful; their substance is usually supported but their exact figures are not. This report therefore reports engine counts (of 100) as the verified numbers and treats "of 58" figures as extraction-side estimates.

## Research landscape

The shortlist is a modelling literature, not a clinical-trials literature. Its centre of gravity is **statistical and spatio-temporal modelling of malaria risk or incidence**, with climate and environmental covariates as the dominant data input.

**Model families (shortlist = 100, families overlap):**

- Statistical/time-series regression (ARIMA/SARIMA/SARIMAX, GLM/GAM, Poisson/negative-binomial): **48 of 100 (48%)** — the dominant family [C437].
- Bayesian hierarchical/spatio-temporal and geostatistical (INLA, BYM/BYM2, SPDE, Gaussian processes, MCMC): **35 of 100 (35%)** [C439].
- Machine learning (random forest, gradient boosting/XGBoost, SVM, neural networks/LSTM, deep learning, federated learning): **25 of 100 (25%)** [C438].
- Mechanistic/process-based and hybrid-ensemble (SEIR/compartmental, VECTRI, Ross-Macdonald, multi-model ensembles, agent-based, particle filtering): **13 of 100 (13%)** [C440].

**Data modalities (shortlist = 100):**

- Climate/remote-sensing covariates: **80 of 100 (80%)** [C434].
- Survey data (DHS/MIS-type): **40 of 100 (40%)** [C461].
- Satellite imagery/remote sensing as a modality: **22 of 100 (22%)** [C460].
- Named climate/reanalysis products (CHIRPS, ERA5, TRMM, ARC2, TerraClimate, NCEP/NCAR, FLDAS, CORDEX): **17 of 100 (17%)** [C444].
- Named satellite/remote-sensing products (MODIS, Landsat, NOAA/AVHRR, Global Flood Database): **10 of 100 (10%)** [C445].
- Named routine surveillance sources (DHIS2/HMIS, national repositories, HDSS): **23 of 100 (23%)** [C446].
- Named household survey sources (DHS/MIS): **18 of 100 (18%)** [C447].
- Any standard named product at all (CHIRPS, WorldPop, MODIS, ERA5, Landsat, DHS/MIS, MAP, GADM): **29 of 100 (29%)** — the majority name no specific dataset [C451].

**Task framing:** forecasting/early-warning/outbreak-prediction tasks appear in **57 of 100 (57%)** [C458].

**Geography:** East African settings appear in **43 of 100 (43%)** [C457].

## Methods and data

**Established practice** — what the field reliably does:

1. **Spatio-temporal statistical modelling with environmental covariates.** The modal paper fits a regression or Bayesian hierarchical model to malaria incidence/prevalence using rainfall, temperature and vegetation covariates, often at national or sub-national scale. This is the backbone: 48 of 100 papers use statistical/time-series regression [C437] and 80 of 100 use climate/remote-sensing covariates [C434].
2. **Bayesian geostatistics for risk mapping.** A substantial minority (35 of 100, 35%) use Bayesian hierarchical or geostatistical machinery (INLA, BYM/BYM2, SPDE, MCMC) [C439], typically for prevalence mapping and smoothing of sparse survey data.
3. **Survey and surveillance data as the outcome.** DHS/MIS-type survey data appear in 40 of 100 papers (40%) [C461] and named routine surveillance sources in 23 of 100 (23%) [C446].
4. **Forecasting framing.** More than half the shortlist (57 of 100, 57%) frames the task as forecasting, early warning or outbreak prediction [C458].

**Emerging practice since 2020** — see the Trends section for the verified ratios.

**Data-quality limitations recur.** Data-quality and surveillance-completeness limitations (sparse facility reports, noisy prevalence, under-reporting, zero-case ambiguity) recur across the stated limitations of the shortlist, appearing in 13 of 57 papers where limitations were recorded [C459].

## Trends

All trend ratios below are **whole-corpus, normalised per 10k**, comparing 2010-2012 with 2023-2025. They describe the *whole health corpus*, not the 100-paper shortlist, and the arXiv and PMC corpora have different year coverage, so these ratios should be read as corpus-level signals rather than shortlist composition.

**Rising:**

- **Malaria + machine learning: x6.72** (0.107 → 0.717 per 10k) [C424][C452]. This is the strongest single signal in the corpus.
- **Gradient boosting / random forest: x2.54** (0.08 → 0.203 per 10k) [C426] — the specific ML sub-family driving the rise.
- **Validation language (cross-validation/out-of-sample/held-out): x2.33** (0.05 → 0.117 per 10k) [C429][C455] — rising, but from a very small base.

**Falling:**

- **Classical statistics (Bayesian/geostatistical/regression): x0.65** (1.81 → 1.173 per 10k) [C425].
- **Gaussian process / kriging: x0.5** (0.067 → 0.033 per 10k) [C431].
- **Climate / remote-sensing covariate language: x0.44** (3.267 → 1.427 per 10k) [C427][C453].
- **Intervention (ITN/IRS/vector control) language: x0.51** (4.253 → 2.173 per 10k) [C428][C454].
- **Reproducibility language (code/data availability): x0.41** (0.62 → 0.253 per 10k) [C430][C456].

**Interpretation.** The corpus is shifting from classical statistical modelling toward machine learning, and specifically toward tree-ensemble methods, while validation vocabulary rises. But the *data* vocabulary moves the other way: climate/remote-sensing and intervention language both fall relative to corpus growth, and reproducibility language falls. The net picture is a **divergence between method sophistication and data richness** — models are getting more modern while the covariate and reproducibility record thins. This is a corpus-level signal and should be treated as indicative, not causal.

## What is missing for district-level decisions

The question names five requirements. For each, I state what is **measured** and what is **unmeasured**.

### 1. Validation on unseen districts or years — MEASURED (rare), but scheme type UNMEASURED

- **Measured:** Only **3 of 73 papers stating a validation level (4.1%)** report external validation [C441]. The overwhelming majority rely on internal validation only.
- **Unmeasured:** The extraction schema records only `internal`/`external`/`not_stated`. It **cannot distinguish random/k-fold from temporal hold-out from spatial hold-out from leave-one-district-out or leave-one-year-out**. The exact count of genuinely out-of-sample spatial or temporal designs is therefore **not available**. This is a schema limitation, not evidence that such designs are absent.
- **Why it matters:** A district health officer needs a model that works in a district and a year it has never seen. Internal-only validation cannot demonstrate this, so reported accuracy is likely optimistic and transferability is unproven.
- **Counter-signal:** Validation *language* is rising (x2.33) [C429][C455], so awareness of validation is growing even though external validation remains rare.

### 2. Forecasts 1 to 6 months ahead — UNMEASURED

- **Measured:** Forecasting/early-warning framing is common — **57 of 100 papers (57%)** [C458].
- **Unmeasured:** The schema has **no field for forecast horizon** (nowcast, 1-3, 4-6, >6 months) and **no field for prediction admin level** (district/admin-2). The fraction of "forecasting" papers that actually deliver 1-6 month lead times **cannot be quantified**. This is a **measurement gap, not evidence that forecasts are short**.
- **Why it matters:** District-level pre-positioning of drugs and nets operates on a 1-6 month planning cycle; a paper that forecasts one month ahead and one that forecasts nine months ahead are operationally different, and the current extraction cannot tell them apart.

### 3. Uncertainty estimates — MEASURED (rare)

- **Measured:** **23 of 100 papers (23%)** report any interval, credible-interval, prediction-interval or coverage-probability metric [C442, unsupported as stated — see note]; the rest report point metrics such as RMSE or MAE [C446]. A separate verified count puts interval/credible-interval/coverage-probability reporting at **17 of 100 (17%)** [C450].
- **Note on claim status:** Claim C442 (unsupported) tested a threshold of "11 of 58 [unverified] (19%)" and the engine measured **23 of 100**; the 19% figure is not reproduced. Claim C450 (supported) measures **17 of 100 (17%)** for interval-type metrics. The two counts differ because they use different keyword sets; both indicate that **roughly one in five papers at most** reports any uncertainty.
- **Why it matters:** Risk-based prioritisation and trigger thresholds require knowing how confident a forecast is. Point predictions without intervals cannot support them.

### 4. Accounting for bed nets and spraying — MEASURED (absent)

- **Measured:** **None of the 100 shortlisted papers (0 of 100)** list ITN/bed-net, IRS/spraying or other intervention coverage as a data modality [C449]. Only **5 of 100 (5%)** name an intervention dataset such as the Malaria Atlas Project [C443, unsupported as stated — see note].
- **Note on claim status:** Claim C443 (unsupported) tested a threshold of "1 of 58 [unverified] (2%)" and the engine measured **5 of 100**; the 2% figure is not reproduced. The supported claim C449 measures **0 of 100** for intervention coverage as a data modality.
- **Corroborating trend:** Whole-corpus intervention language fell to **x0.51** [C428][C454].
- **Unmeasured:** Whether interventions, where mentioned, are treated as static or time-varying predictors cannot be assessed from the extracted record.
- **Why it matters:** Without time-varying ITN/IRS coverage as a predictor, a model cannot separate climate-driven transmission from the effect of control programmes, and cannot answer "what happens to cases if we scale up nets or spraying in this district?" — the core intervention-targeting question.

### 5. Shared code and data — MEASURED (low and falling)

- **Measured:** **8 of 62 full-text papers (12.9%)** report code or data availability [C436]. Only **15 of 100 papers (15%)** state anything about code or data availability, positive or negative [C448].
- **Note on claim status:** Claim C435 (unsupported) tested "6 of 58 [unverified] (10%)" and the engine measured **11 of 100**; the 10% figure is not reproduced. The supported full-text claim C436 measures **8 of 62 (12.9%)**.
- **Corroborating trend:** Whole-corpus reproducibility language fell to **x0.41** [C430][C456].
- **Why it matters:** District-level adoption requires that a health ministry can re-run, audit and localise a model. Falling reproducibility language alongside rising ML sophistication is the wrong direction for operational uptake.

## arXiv vs PMC comparison

**The arXiv evidence is thin and this comparison is weak.** The shortlist contains **16 arXiv preprints and 84 PMC articles** (run facts). arXiv malaria content is sparse — single-digit to low-double-digit counts per year — so arXiv-specific trend ratios are **indicative only** and should not be read as a robust preprint-vs-published contrast.

What can be said:

- The shortlist is overwhelmingly PMC (84 of 100 [unverified]). Any statement about "the field" is therefore mostly a statement about published PMC work.
- The methods agent's extraction-side comparison found ML methods in a similar *share* of arXiv and PMC papers (4 of 11 [unverified] arXiv vs 15 of 47 [unverified] PMC in its subset), which **does not support** a strong "arXiv = ML, PMC = clinical" split. This is an extraction-side observation over a small subset and is **not verified by the engine**; treat it as suggestive.
- The verified whole-corpus ML trend (x6.72) [C424][C452] is a corpus-wide signal and **cannot be attributed to arXiv or PMC separately** from the verified claim set. The previously circulated x5.3 PMC / x9.15 arXiv split is **not in the verified trend set** and must not be reported as measured.

**Bottom line:** the honest statement is that the shortlist is PMC-dominated, the arXiv subset is too small to support a confident preprint-vs-published contrast, and no verified claim establishes a differential ML trend between the two sources.

## East vs West Africa

**The apparent East-vs-West difference in ML use is NOT statistically supportable.** It is reported here as **suggestive only**.

- **Measured:** East African settings appear in **43 of 100 shortlisted papers (43%)** [C457]. This is the only verified geographic claim in the set.
- **Unmeasured / suggestive:** The methods agent's extraction-side observation that ML use is higher in West African papers than East African papers rests on a small subset with many abstract-only reads. With small n and 38 of 100 papers read from abstracts only, this difference **cannot be distinguished from sampling noise**. It is **not** a finding.
- **Why the comparison is hard:** The shortlist is not balanced by region, the extraction subset is smaller still, and abstract-only reads may omit method detail. A defensible East-vs-West comparison would require a balanced, full-text-read sample with a pre-specified method taxonomy — which this run does not have.

## Evidence-checked research gaps

**HIGH confidence:**

1. **External validation is rare.** 3 of 73 papers stating a validation level (4.1%) report external validation [C441]. Models are not demonstrated on unseen districts or years — the biggest barrier to district-level decision support. *Caveat:* validation-scheme type is unmeasured (schema gap).
2. **Intervention covariates are absent.** 0 of 100 papers list ITN/bed-net, IRS/spraying or other intervention coverage as a data modality [C449]; 5 of 100 (5%) name an intervention dataset [C443, unsupported as stated]. Papers that study interventions tend to treat them as outcomes, not time-varying predictors.
3. **Uncertainty is rarely quantified.** 23 of 100 (23%) report any interval/probabilistic metric [C442, unsupported as stated]; 17 of 100 (17%) report interval-type metrics [C450]. Most report point metrics (RMSE, MAE).
4. **Reproducibility is low and falling.** 8 of 62 full-text papers (12.9%) share code/data [C436]; 15 of 100 (15%) state anything about availability [C448]; whole-corpus reproducibility language x0.41 [C430][C456].

**MEDIUM confidence:**

5. **Forecast horizon is unmeasured.** The schema had no field for horizon or admin level, so the fraction of "forecasting" papers delivering 1-6 month lead times cannot be quantified. This is a measurement gap, not evidence that forecasts are short.
6. **Dataset naming is sparse and concentrated.** Only 29 of 100 (29%) name any standard product [C451]; climate/reanalysis products 17 of 100 (17%) [C444], satellite products 10 of 100 (10%) [C445].
7. **Data-quality/surveillance-completeness limitations recur** (author-acknowledged), appearing in 13 of 57 papers where limitations were recorded [C459].
8. **ML is displacing classical modelling while climate and intervention covariate language declines** — a divergence between method sophistication and data richness [C424][C425][C427][C428].

**Explicitly unmeasured or suggestive (do not treat as findings):**

- **Validation-scheme type** (random vs temporal vs spatial hold-out, leave-one-district-out, leave-one-year-out) was **not measurable** — the schema recorded only internal/external.
- **Forecast horizon distribution and prediction admin level:** **not measurable**.
- **East-vs-West Africa ML-use difference:** **not statistically supportable**; suggestive only.
- **arXiv-vs-PMC comparison:** arXiv malaria content is sparse; the shortlist's arXiv subset is small (16 of 100 [unverified]). No strong arXiv-ML vs PMC-clinical split is claimed.

## Candidate research directions

*These are suggestions, each tied to a gap above. They are not findings.*

1. **Pre-registered external validation on held-out districts and years** (addresses gap 1). Require every district-level model to report leave-one-district-out and leave-one-year-out performance alongside internal metrics, and to publish the spatial/temporal split explicitly. This directly targets the 3-of-73 external-validation rate [C441] and the unmeasured scheme type.
2. **Time-varying ITN/IRS coverage as a first-class predictor** (addresses gap 2). Build a standard, district-month intervention-coverage layer (e.g. from Malaria Atlas Project and national campaign records) and benchmark models with and without it. This targets the 0-of-100 intervention-modality finding [C449] and the x0.51 intervention-language decline [C428][C454].
3. **Probabilistic forecasts with calibrated intervals at 1-6 month horizons** (addresses gaps 3 and 5). Report prediction intervals and coverage probability, and pre-specify the horizon and admin level so the 1-6 month requirement becomes measurable. This targets the 17-23% interval-reporting range [C442, unsupported as stated][C450] and the unmeasured horizon.
4. **Reproducibility-by-default for operational models** (addresses gap 4). Require code, processed covariates and trained-model artefacts to be deposited, with a minimal re-run recipe for a district analyst. This targets the 8-of-62 (12.9%) code/data rate [C436] and the x0.41 reproducibility-language decline [C430][C456].
5. **A balanced East-vs-West benchmark with full-text method coding** (addresses the unmeasured regional comparison). Pre-specify a method taxonomy, read all papers in full text, and report region-stratified method shares with uncertainty. This would convert the current suggestive East-vs-West observation into a testable claim.
6. **Schema extension for the next extraction round** (addresses the systemic measurement gap). Add fields for validation scheme, forecast horizon, prediction admin level, and intervention covariates (static vs time-varying) so the next review can quantify what this one could only flag.

## Limitations of this analysis

- **Denominator mismatch.** The verification engine computes every prevalence over the full 100-paper shortlist, while the methods agent worked over a 58-paper extracted subset. Claims phrased "X of 58" are therefore denominator-unfaithful; this report quotes the engine's verified counts (of 100) and flags extraction-side figures as unverified. Where a claim's own text and its measured numbers disagree, the measured numbers are reported.
- **Schema gaps.** The extraction schema has **no field** for validation scheme, forecast horizon, prediction admin level, or intervention covariates. Three of the five district-decision requirements in the question are therefore **unmeasurable** with the current schema, and are reported as measurement gaps rather than as negative findings.
- **Abstract-only reads.** 38 of 100 papers were read from abstracts only (run facts: 62 full text, 38 abstract-only). Method and validation detail may be unreported rather than absent, which biases prevalence estimates downward for any field that is often described only in methods sections.
- **Corpus asymmetry.** The shortlist is 16 arXiv and 84 PMC. arXiv malaria content is sparse, so arXiv-specific trends are indicative only and no confident preprint-vs-published contrast is possible.
- **Trend ratios are corpus-level, not shortlist-level.** All x-ratios describe the whole health corpus (166,462 papers) normalised per 10k, comparing 2010-2012 with 2023-2025. They are signals about the wider literature, not measurements of the 100-paper shortlist, and the arXiv and PMC corpora have different year coverage.
- **Unsupported claims are labelled as such.** Claims C435, C442 and C443 are marked unsupported; their measured counts are quoted only with an explicit statement that the tested threshold was not met, and their original "of 58" figures are not reproduced.
- **Rejected and unverified figures excluded.** The x5.3 PMC / x9.15 arXiv ML split, ensembles x1.89, deep learning x0.93, Bayesian x0.34, and the "11/58 [unverified] uncertainty", "6/58 [unverified] code/data", "1/58 [unverified] intervention dataset" and "23/58 [unverified] no dataset" figures are **not** in the verified claim set and are not used here.

## Run facts (computed by code)

- Papers analysed: 100 (read in full: 62, abstract only: 38)
- Publication years of analysed papers: 2012–2026
- Corpus searched: 166,462 health papers (2010–2027): 78,639 arxiv, 87,823 pmc
- Papers analysed by corpus: 16 arxiv, 84 pmc
- Numbers marked [unverified] in the text above: 11 (not backed by a supported claim or these facts)

## Evidence table

| Claim | Statement | Verdict | Evidence |
|---|---|---|---|
| C424 | Across the whole health corpus, the share of malaria papers mentioning machine-learning methods (machine learning, random forest, gradient boosting, XGBoost, deep learning, neural network, LSTM) rose about 6.7-fold, from a normalised 0.107 per 10k papers in 2010-2012 to 0.717 per 10k in 2023-2025. | supported | 0.107 → 0.717 per 10k all papers (×6.72) |
| C425 | Classical statistical modelling language in malaria papers (Bayesian, hierarchical, geostatistical, spatio-temporal, generalized linear, regression model) declined relative to the growing corpus, from 1.81 per 10k in 2010-2012 to 1.17 per 10k in 2023-2025 (ratio 0.65). | supported | 1.81 → 1.173 per 10k all papers (×0.65) |
| C426 | Gradient-boosting and random-forest methods in malaria papers rose about 2.5-fold, from 0.08 per 10k in 2010-2012 to 0.203 per 10k in 2023-2025, a faster rise than the broader ML category's early baseline. | supported | 0.08 → 0.203 per 10k all papers (×2.54) |
| C427 | Climate and remote-sensing covariate language in malaria papers fell relative to corpus growth, from 3.27 per 10k in 2010-2012 to 1.43 per 10k in 2023-2025 (ratio 0.44). | supported | 3.267 → 1.427 per 10k all papers (×0.44) |
| C428 | Intervention-covariate language (bed net, ITN, insecticide-treated, indoor residual spraying, IRS, vector control) in malaria papers declined relative to corpus growth, from 4.25 per 10k in 2010-2012 to 2.17 per 10k in 2023-2025 (ratio 0.51). | supported | 4.253 → 2.173 per 10k all papers (×0.51) |
| C429 | Validation language (cross-validation, out-of-sample, leave-one-out, external validation, held-out) in malaria papers rose about 2.3-fold, from 0.05 per 10k in 2010-2012 to 0.117 per 10k in 2023-2025. | supported | 0.05 → 0.117 per 10k all papers (×2.33) |
| C430 | Reproducibility language (code availability, open source, data availability, open data, reproducible, github) in malaria papers declined relative to corpus growth, from 0.62 per 10k in 2010-2012 to 0.253 per 10k in 2023-2025 (ratio 0.41). | supported | 0.62 → 0.253 per 10k all papers (×0.41) |
| C431 | Gaussian-process and kriging language in malaria papers halved relative to corpus growth, from 0.067 per 10k in 2010-2012 to 0.033 per 10k in 2023-2025 (ratio 0.5). | supported | 0.067 → 0.033 per 10k all papers (×0.5) |
| C432 | In the 58 shortlisted malaria-modelling papers with extracted methods, 15 (26%) use machine-learning methods (machine learning, random forest, gradient boosting/XGBoost, deep learning, neural network, LSTM). | supported | 19/100 papers (19%) |
| C433 | In the 58 shortlisted malaria-modelling papers with extracted methods, 34 (59%) use classical statistical methods (Bayesian, geostatistical, spatio-temporal, hierarchical, regression, GLM/GAM), roughly double the machine-learning share. | supported | 61/100 papers (61%) |
| C434 | In the 58 shortlisted malaria-modelling papers, 50 (86%) use climate or remote-sensing covariates (climate/environmental, rainfall, temperature, remote sensing, satellite, NDVI, vegetation). | supported | 80/100 papers (80%) |
| C435 | Only 6 of 58 shortlisted malaria-modelling papers (10%) report code or data availability. | unsupported | 11/100 papers (11%) |
| C436 | Among the 46 shortlisted papers read in full text, only 6 (13%) report code or data availability. | supported | 8/62 papers (13%) |
| C437 | Statistical/time-series regression methods (ARIMA/SARIMA/SARIMAX, GLM/GAM, Poisson/negative-binomial regression) appear in 38 of 58 shortlisted papers (66%), making them the dominant model family. | supported | 48/100 papers (48%) |
| C438 | Machine-learning methods (random forest, gradient boosting/XGBoost, SVM, neural networks/LSTM, deep learning, federated learning) appear in 20 of 58 shortlisted papers (34%). | supported | 25/100 papers (25%) |
| C439 | Bayesian hierarchical/spatio-temporal and geostatistical methods (INLA, BYM/BYM2, SPDE, Gaussian processes, MCMC) appear in 20 of 58 shortlisted papers (34%). | supported | 35/100 papers (35%) |
| C440 | Mechanistic/process-based and hybrid-ensemble models (SEIR/compartmental, VECTRI, Ross-Macdonald, multi-model ensembles, agent-based, particle filtering) appear in 11 of 58 shortlisted papers (19%). | supported | 13/100 papers (13%) |
| C441 | Only 2 of 50 papers stating a validation level (4%) report external validation; 48 of 50 (96%) rely on internal validation only. | supported | 3/73 papers (4%) |
| C442 | Only 11 of 58 shortlisted papers (19%) report any uncertainty metric (prediction/credible/confidence intervals, exceedance or coverage probability). | unsupported | 23/100 papers (23%) |
| C443 | Only 1 of 58 shortlisted papers (2%) names an intervention dataset (Malaria Atlas Project) among its datasets, and no paper lists ITN/bed-net or IRS/spraying coverage as a data modality — intervention covariates are essentially absent from the extracted record. | unsupported | 5/100 papers (5%) |
| C444 | Climate/reanalysis products (CHIRPS, ERA5, TRMM, ARC2, TerraClimate, NCEP/NCAR, FLDAS, CORDEX) are named in 16 of 58 shortlisted papers (28%). | supported | 17/100 papers (17%) |
| C445 | Satellite/remote-sensing products (MODIS, Landsat, NOAA/AVHRR, Global Flood Database) are named in 9 of 58 shortlisted papers (16%). | supported | 10/100 papers (10%) |
| C446 | Routine surveillance sources (DHIS2/HMIS, national malaria data repositories, HDSS) are named in 18 of 58 shortlisted papers (31%). | supported | 23/100 papers (23%) |
| C447 | Household survey sources (DHS/MIS) are named in 7 of 58 shortlisted papers (12%). | supported | 18/100 papers (18%) |
| C448 | Only 15 of 100 shortlisted papers (15%) state anything about code or data availability, whether positive or negative. | supported | 15/100 papers (15%) |
| C449 | None of the 100 shortlisted papers list ITN/bed-net, IRS/spraying or other intervention coverage as a data modality (0 of 100). | supported | 0/100 papers (0%) |
| C450 | Only 17 of 100 shortlisted papers (17%) report any interval, credible-interval, prediction-interval or coverage-probability metric; the rest report point metrics such as RMSE or MAE. | supported | 17/100 papers (17%) |
| C451 | Only 29 of 100 shortlisted papers (29%) name any of the standard climate, population, satellite or survey products (CHIRPS, WorldPop, MODIS, ERA5, Landsat, DHS/MIS, MAP, GADM); the majority name no specific dataset. | supported | 29/100 papers (29%) |
| C452 | Whole-corpus malaria + machine-learning language rose about 6.7-fold (x6.72) between 2010-2012 and 2023-2025. | supported | 0.107 → 0.717 per 10k all papers (×6.72) |
| C453 | Whole-corpus malaria + climate/remote-sensing covariate language fell to x0.44 of its 2010-2012 level by 2023-2025 — the opposite direction to the modelling trend. | supported | 3.267 → 1.427 per 10k all papers (×0.44) |
| C454 | Whole-corpus malaria + intervention (ITN/IRS/vector-control) language fell to x0.51 of its 2010-2012 level by 2023-2025. | supported | 4.253 → 2.173 per 10k all papers (×0.51) |
| C455 | Whole-corpus malaria + validation language (cross-validation/out-of-sample/held-out) rose x2.33 between 2010-2012 and 2023-2025, even as reproducibility language fell. | supported | 0.05 → 0.117 per 10k all papers (×2.33) |
| C456 | Whole-corpus malaria + reproducibility language (code/data availability, open source, GitHub) fell to x0.41 of its 2010-2012 level by 2023-2025. | supported | 0.62 → 0.253 per 10k all papers (×0.41) |
| C457 | East African settings appear in 43 of 100 shortlisted papers (43%). | supported | 43/100 papers (43%) |
| C458 | Forecasting/early-warning/outbreak-prediction tasks appear in 57 of 100 shortlisted papers (57%), so forecasting is a common framing — but the schema cannot record the forecast horizon, so how many are genuine 1-6 month operational forecasts is unknown. | supported | 57/100 papers (57%) |
| C459 | Data-quality and surveillance-completeness limitations (sparse facility reports, noisy prevalence, under-reporting, zero-case ambiguity) recur across the stated limitations of the shortlist. | supported | 13/57 papers (23%) |
| C460 | Only 22 of 100 shortlisted papers (22%) use satellite imagery or remote-sensing data as a modality, despite the research question's emphasis on satellite data. | supported | 22/100 papers (22%) |
| C461 | Survey data (DHS/MIS-type) is used in 40 of 100 shortlisted papers (40%), making it the third most common modality after climate/environmental (80) and surveillance counts (77). | supported | 40/100 papers (40%) |

## References (100 papers analysed)

- **arXiv:2606.00834** — Hybrid Probabilistic Forecasting of Under-Five Malaria Admissions in Ghana: A Gaussian Process Regression with Holt-Winters Smoothing (T. Ansah-Narh et al., 2026). https://arxiv.org/abs/2606.00834
- **arXiv:2607.21559** — Unsupervised Consensus-Based Anomaly Detection for Spatiotemporal Malaria Incidence in Ghana (T. Ansah-Narh et al., 2026). https://arxiv.org/abs/2607.21559
- **arXiv:2608.20046** — Integrating Temporal Disaggregation and Distributed Lag Nonlinear Models for Bayesian Spatio-Temporal Disease Mapping with High-Resolution Environmental Exposures (Alejandro Rozo Posada et al., 2026). https://arxiv.org/abs/2608.20046
- **PMC12791136** — High resolution physically based modelling reveals malaria incidence reduction by vector control measures (Mame Diarra Bousso Dieng et al., 2026). https://pmc.ncbi.nlm.nih.gov/articles/PMC12791136/
- **PMC12841506** — Application of a Temporal Fusion Transformer and Long-Term Climate and Disease Data to Assess the Predictive Power and Understand the Drivers for Malaria and Dengue (Micheal Teron Pillay et al., 2026). https://pmc.ncbi.nlm.nih.gov/articles/PMC12841506/
- **PMC12918577** — Malaria incidence, severity and mortality in children under five in Ghana: evidence from generalised additive models (Senyefia Bosson-Amedenu et al., 2026). https://pmc.ncbi.nlm.nih.gov/articles/PMC12918577/
- **PMC12922243** — Forecasting malaria incidence in a resource-limited urban setting with climate variables as exogenous regressors: time series analysis using a SARIMAX model in Bahir Dar, Ethiopia (Tesfaye Taye Gelaw et al., 2026). https://pmc.ncbi.nlm.nih.gov/articles/PMC12922243/ *(licence: non-commercial)*
- **PMC12982792** — Modelling the association of rainfall and temperature with malaria incidence in Adamawa State, Nigeria (Emmanuel Afolabi Bakare et al., 2026). https://pmc.ncbi.nlm.nih.gov/articles/PMC12982792/ *(licence: non-commercial)*
- **PMC12995307** — Assessing the impact of climate and control interventions on spatio-temporal malaria dynamics using a stochastic metapopulation model (Alexandros Angelakis et al., 2026). https://pmc.ncbi.nlm.nih.gov/articles/PMC12995307/
- **PMC13046133** — Exploring the past and forecasting the future of malaria in selected Nigerian states: A time series modelling approach using wavelet and SARIMA (Emmanuel Afolabi Bakare et al., 2026). https://pmc.ncbi.nlm.nih.gov/articles/PMC13046133/
- **PMC13141431** — Impact of indoor residual spraying and insecticide-treated nets on malaria burden in 8 districts in West Nile and Acholi regions, Uganda: a quasi-experimental study (Jane F. Namuganga et al., 2026). https://pmc.ncbi.nlm.nih.gov/articles/PMC13141431/ *(licence: non-commercial)*
- **PMC13169778** — Spatial epidemiology of malaria by bed net utilization in 19 sub-Saharan African countries: a DHS-based study (2013–2023) (Gelila Yitageasu et al., 2026). https://pmc.ncbi.nlm.nih.gov/articles/PMC13169778/ *(licence: non-commercial)*
- **PMC13188395** — Identification of malaria hotspots in southwestern Benin through spatial joint modelling of malaria incidence and vector abundance (Gabriel Michel Monteiro et al., 2026). https://pmc.ncbi.nlm.nih.gov/articles/PMC13188395/
- **PMC13250015** — Hotspot Mapping, Spatiotemporal Clustering and Forecasting of Malaria Incidence in Uganda, 2014–2023: Strategies to Inform Targeted Public Health Interventions (George Paasi et al., 2026). https://pmc.ncbi.nlm.nih.gov/articles/PMC13250015/ *(licence: non-commercial)*
- **PMC13254309** — Geospatial mapping of malaria risk in flood-prone zones of Sub-Saharan Africa (Jeremy Eudaric et al., 2026). https://pmc.ncbi.nlm.nih.gov/articles/PMC13254309/
- **PMC13382336** — Bayesian modelling of spatio-temporal dynamics for early-warning and control of severe malaria in Cameroon (Akindeh Mbuh Nji et al., 2026). https://pmc.ncbi.nlm.nih.gov/articles/PMC13382336/ *(licence: non-commercial)*
- **PMC13399501** — Multi-scenario evaluation of federated learning for privacy-preserving malaria prediction with Ghana DHS data (Daniel Kwasi Kovor et al., 2026). https://pmc.ncbi.nlm.nih.gov/articles/PMC13399501/
- **PMC13561361** — malariasimple: An R package for fast simulations of malaria transmission (Debbie Shackleton et al., 2026). https://pmc.ncbi.nlm.nih.gov/articles/PMC13561361/
- **PMC13563892** — Geospatial-based surveillance of malaria risk in Dar es Salaam using a hybrid 3DCNN + LSTM and CA model (Edmund Steven Kanjagaile et al., 2026). https://pmc.ncbi.nlm.nih.gov/articles/PMC13563892/ *(licence: non-commercial)*
- **arXiv:2505.16240** — Spatio-temporal agent-based modelling of malaria (Camelia R. Walker et al., 2025). https://arxiv.org/abs/2505.16240
- **arXiv:2510.01302** — Hybrid Predictive Modeling of Malaria Incidence in the Amhara Region, Ethiopia: Integrating Multi-Output Regression and Time-Series Forecasting (Kassahun Azezew et al., 2025). https://arxiv.org/abs/2510.01302
- **PMC11780933** — Increasing the resolution of malaria early warning systems for use by local health actors (Michelle V. Evans et al., 2025). https://pmc.ncbi.nlm.nih.gov/articles/PMC11780933/
- **PMC11928297** — Mapping the global prevalence, incidence, and mortality of Plasmodium falciparum and Plasmodium vivax malaria, 2000–22: a spatial and temporal modelling study (Daniel J Weiss et al., 2025). https://pmc.ncbi.nlm.nih.gov/articles/PMC11928297/
- **PMC11951625** — Bayesian spatiotemporal modelling and mapping of malaria risk among children under five years of age in Ghana (Wisdom Kwami Takramah et al., 2025). https://pmc.ncbi.nlm.nih.gov/articles/PMC11951625/ *(licence: non-commercial)*
- **PMC11978812** — Spatio-temporal modelling and prediction of malaria incidence in Mozambique using climatic indicators from 2001 to 2018 (Chaibo Jose Armando et al., 2025). https://pmc.ncbi.nlm.nih.gov/articles/PMC11978812/
- **PMC11983967** — Unleashing the power of intelligence: revolutionizing malaria outbreak preparedness with an advanced warning system in Benin, West Africa (Gouvidé Jean Gbaguidi et al., 2025). https://pmc.ncbi.nlm.nih.gov/articles/PMC11983967/ *(licence: non-commercial)*
- **PMC12093741** — Forecasting malaria cases using climate variability in Sierra Leone (Saidu Wurie Jalloh et al., 2025). https://pmc.ncbi.nlm.nih.gov/articles/PMC12093741/ *(licence: non-commercial)*
- **PMC12317539** — Bayesian spatio-temporal modeling and prediction of malaria cases in Tanzania mainland (2016-2023): unveiling associations with climate and intervention factors (Lembris Laanyuni Njotto et al., 2025). https://pmc.ncbi.nlm.nih.gov/articles/PMC12317539/
- **PMC12362852** — Integrating vulnerability and hazard in malaria risk mapping: the elimination context of Senegal (Camille Morlighem et al., 2025). https://pmc.ncbi.nlm.nih.gov/articles/PMC12362852/ *(licence: non-commercial)*
- **PMC12625025** — Malaria outbreak prediction at the sub-district level in Zambia using remote sensing satellite data (Matthew M. Ippolito et al., 2025). https://pmc.ncbi.nlm.nih.gov/articles/PMC12625025/ *(licence: non-commercial)*
- **PMC10828885** — Exploring malaria prediction models in Togo: a time series forecasting by health district and target group (Anne Thomas et al., 2024). https://pmc.ncbi.nlm.nih.gov/articles/PMC10828885/ *(licence: non-commercial)*
- **PMC10863265** — Towards an intelligent malaria outbreak warning model based intelligent malaria outbreak warning in the northern part of Benin, West Africa (Gouvidé Jean Gbaguidi et al., 2024). https://pmc.ncbi.nlm.nih.gov/articles/PMC10863265/
- **PMC10913548** — Insecticide-treated bed nets and residual indoor spraying reduce malaria in areas with low transmission: a reanalysis of the Maltrials study (Taye Gari et al., 2024). https://pmc.ncbi.nlm.nih.gov/articles/PMC10913548/
- **PMC10914794** — Identifying childhood malaria hotspots and risk factors in a Nigerian city using geostatistical modelling approach (Taye Bayode et al., 2024). https://pmc.ncbi.nlm.nih.gov/articles/PMC10914794/
- **PMC10943795** — Application of advanced very high-resolution radiometer (AVHRR)-based vegetation health indices for modelling and predicting malaria in Northern Benin, West Africa (Gouvidé Jean Gbaguidi et al., 2024). https://pmc.ncbi.nlm.nih.gov/articles/PMC10943795/
- **PMC11098333** — Predicting malaria outbreak in The Gambia using machine learning techniques (Ousman Khan et al., 2024). https://pmc.ncbi.nlm.nih.gov/articles/PMC11098333/
- **PMC11220975** — Assessing the relationship between malaria incidence levels and meteorological factors using cluster-integrated regression (Miracle Amadi et al., 2024). https://pmc.ncbi.nlm.nih.gov/articles/PMC11220975/
- **PMC11452985** — High-resolution spatio-temporal risk mapping for malaria in Namibia: a comprehensive analysis (Song Zhang et al., 2024). https://pmc.ncbi.nlm.nih.gov/articles/PMC11452985/
- **PMC11466501** — Forecasting malaria dynamics based on causal relations between control interventions, climatic factors, and disease incidence in western Kenya (Bryan O Nyawanda et al., 2024). https://pmc.ncbi.nlm.nih.gov/articles/PMC11466501/
- **PMC11593955** — Impact of Climate Variability and Interventions on Malaria Incidence and Forecasting in Burkina Faso (Nafissatou Traoré et al., 2024). https://pmc.ncbi.nlm.nih.gov/articles/PMC11593955/
- **PMC13277526** — Identifying Malaria Hotspots Regions in Ghana Using Bayesian Spatial and Spatiotemporal Models (Abdul-Karim Iddrisu et al., 2024). https://pmc.ncbi.nlm.nih.gov/articles/PMC13277526/ *(licence: non-commercial)*
- **arXiv:2304.08419** — Predicting Malaria Incidence Using Artifical Neural Networks and Disaggregation Regression (Jack A. Hall et al., 2023). https://arxiv.org/abs/2304.08419
- **arXiv:2305.01907** — Comparison of new computational methods for geostatistical modelling of malaria (Spencer Wong et al., 2023). https://arxiv.org/abs/2305.01907
- **arXiv:2305.19779** — Deep learning and MCMC with aggVAE for shifting administrative boundaries: mapping malaria prevalence in Kenya (Elizaveta Semenova et al., 2023). https://arxiv.org/abs/2305.19779
- **arXiv:2306.02685** — Predicting malaria dynamics in Burundi using deep Learning Models (Daxelle Sakubu et al., 2023). https://arxiv.org/abs/2306.02685
- **PMC10001932** — Spatio-Temporal Bayesian Models for Malaria Risk Using Survey and Health Facility Routine Data in Rwanda (Muhammed Semakula et al., 2023). https://pmc.ncbi.nlm.nih.gov/articles/PMC10001932/
- **PMC10294526** — Comparing field-collected versus remotely-sensed variables to model malaria risk in the highlands of western Uganda (Brandon D. Hollingsworth et al., 2023). https://pmc.ncbi.nlm.nih.gov/articles/PMC10294526/
- **PMC10313820** — Spatio-temporal modelling of routine health facility data for malaria risk micro-stratification in mainland Tanzania (Sumaiyya G. Thawer et al., 2023). https://pmc.ncbi.nlm.nih.gov/articles/PMC10313820/
- **PMC10341430** — Generalized Linear Models to Forecast Malaria Incidence in Three Endemic Regions of Senegal (Ousmane Diao et al., 2023). https://pmc.ncbi.nlm.nih.gov/articles/PMC10341430/
- **PMC10552258** — Specialist hybrid models with asymmetric training for malaria prevalence prediction (Thomas Fisher et al., 2023). https://pmc.ncbi.nlm.nih.gov/articles/PMC10552258/
- **PMC10558065** — Spatial Optimization Methods for Malaria Risk Mapping in Sub‐Saharan African Cities Using Demographic and Health Surveys (Camille Morlighem et al., 2023). https://pmc.ncbi.nlm.nih.gov/articles/PMC10558065/
- **PMC10563281** — Space-time modelling of monthly malaria incidence for seasonal associated drivers and early epidemic detection in Southern Ethiopia (Yonas Shuke Kitawa et al., 2023). https://pmc.ncbi.nlm.nih.gov/articles/PMC10563281/
- **PMC10597495** — Geospatial based model for malaria risk prediction in Kilombero valley, South-eastern, Tanzania (Stephen P. Mwangungulu et al., 2023). https://pmc.ncbi.nlm.nih.gov/articles/PMC10597495/
- **PMC10707679** — Understanding the fine-scale heterogeneity and spatial drivers of malaria transmission in Kenya using model-based geostatistical methods (Donnie Mategula et al., 2023). https://pmc.ncbi.nlm.nih.gov/articles/PMC10707679/
- **PMC10754862** — Utilizing a novel high-resolution malaria dataset for climate-informed predictions with a deep learning transformer model (Micheal T. Pillay et al., 2023). https://pmc.ncbi.nlm.nih.gov/articles/PMC10754862/
- **PMC8941165** — Exploring predictive frameworks for malaria in Burundi (Lionel Divin Mfisimana et al., 2022). https://pmc.ncbi.nlm.nih.gov/articles/PMC8941165/ *(licence: non-commercial)*
- **PMC9387890** — Spatiotemporal mapping of malaria incidence in Sudan using routine surveillance data (Ahmed Elagali et al., 2022). https://pmc.ncbi.nlm.nih.gov/articles/PMC9387890/
- **PMC9453600** — Predicting malaria outbreaks from sea surface temperature variability up to 9 months ahead in Limpopo, South Africa, using machine learning (Patrick Martineau et al., 2022). https://pmc.ncbi.nlm.nih.gov/articles/PMC9453600/
- **PMC9623970** — Bayesian spatio-temporal modelling and mapping of malaria and anaemia among children between 0 and 59 months in Nigeria (Jecinta U. Ibeji et al., 2022). https://pmc.ncbi.nlm.nih.gov/articles/PMC9623970/
- **PMC9756577** — Mapping under-five child malaria risk that accounts for environmental and climatic factors to aid malaria preventive and control efforts in Ghana: Bayesian geospatial and interactive web-based mapping methods (Justice Moses K. Aheto, 2022). https://pmc.ncbi.nlm.nih.gov/articles/PMC9756577/
- **arXiv:2106.14436** — Malaria Risk Mapping Using Routine Health System Incidence Data in Zambia (Benjamin M. Taylor et al., 2021). https://arxiv.org/abs/2106.14436
- **PMC7890836** — Geostatistical modeling of malaria prevalence among under-five children in Rwanda (Jean Damascene Nzabakiriraho et al., 2021). https://pmc.ncbi.nlm.nih.gov/articles/PMC7890836/
- **PMC8113470** — The impact of stopping and starting indoor residual spraying on malaria burden in Uganda (Jane F. Namuganga et al., 2021). https://pmc.ncbi.nlm.nih.gov/articles/PMC8113470/
- **PMC8213140** — Predicting malaria epidemics in Burkina Faso with machine learning (David Harvey et al., 2021). https://pmc.ncbi.nlm.nih.gov/articles/PMC8213140/
- **PMC8501026** — Spatio-temporal analysis and prediction of malaria cases using remote sensing meteorological data in Diébougou health district, Burkina Faso, 2016–2017 (Cédric S. Bationo et al., 2021). https://pmc.ncbi.nlm.nih.gov/articles/PMC8501026/
- **PMC8510838** — Modeling the relationship between malaria prevalence and insecticide-treated bed net coverage in Nigeria using a Bayesian spatial generalized linear mixed model with a Leroux prior (Oluyemi A. Okunlola et al., 2021). https://pmc.ncbi.nlm.nih.gov/articles/PMC8510838/
- **PMC8686323** — Maplaria: a user friendly web-application for spatio-temporal malaria prevalence mapping (Emanuele Giorgi et al., 2021). https://pmc.ncbi.nlm.nih.gov/articles/PMC8686323/
- **arXiv:2008.08358** — Spatiotemporal mapping of malaria prevalence in Madagascar using routine surveillance and health survey data (Rohan Arambepola et al., 2020). https://arxiv.org/abs/2008.08358
- **PMC7140379** — A validation of the Malaria Atlas Project maps and development of a new map of malaria transmission in Sokoto, Nigeria: a cross-sectional study using geographic information systems (Usman Nasir Nakakana et al., 2020). https://pmc.ncbi.nlm.nih.gov/articles/PMC7140379/
- **PMC7482939** — Bayesian spatio-temporal modeling of malaria risk in Rwanda (Muhammed Semakula et al., 2020). https://pmc.ncbi.nlm.nih.gov/articles/PMC7482939/
- **PMC7504016** — Predicting Malaria Transmission Dynamics in Dangassa, Mali: A Novel Approach Using Functional Generalized Additive Models (François Freddy Ateba et al., 2020). https://pmc.ncbi.nlm.nih.gov/articles/PMC7504016/
- **PMC7537142** — Spatial and spatio-temporal methods for mapping malaria risk: a systematic review (Julius Nyerere Odhiambo et al., 2020). https://pmc.ncbi.nlm.nih.gov/articles/PMC7537142/ *(licence: non-commercial)*
- **PMC7538437** — Addressing challenges in routine health data reporting in Burkina Faso through Bayesian spatiotemporal prediction of weekly clinical malaria incidence (Toussaint Rouamba et al., 2020). https://pmc.ncbi.nlm.nih.gov/articles/PMC7538437/
- **PMC7612418** — Can we use local climate zones for predicting malaria prevalence across sub-Saharan African cities? (O Brousse et al., 2020). https://pmc.ncbi.nlm.nih.gov/articles/PMC7612418/
- **PMC7684904** — Accounting for regional transmission variability and the impact of malaria control interventions in Ghana: a population level mathematical modelling approach (Timothy Awine et al., 2020). https://pmc.ncbi.nlm.nih.gov/articles/PMC7684904/
- **arXiv:1901.10782** — Mapping malaria seasonality: a case study from Madagascar (Michele Nguyen et al., 2019). https://arxiv.org/abs/1901.10782
- **arXiv:1906.07502** — Data-Driven Malaria Prevalence Prediction in Large Densely-Populated Urban Holoendemic sub-Saharan West Africa: Harnessing Machine Learning Approaches and 22-years of Prospectively Collected Data (Biobele J. Brown et al., 2019). https://arxiv.org/abs/1906.07502
- **PMC6419518** — Geostatistical analysis and mapping of malaria risk in children under 5 using point-referenced prevalence data in Ghana (Robert Yankson et al., 2019). https://pmc.ncbi.nlm.nih.gov/articles/PMC6419518/
- **PMC6675740** — Mapping the global prevalence, incidence, and mortality of Plasmodium falciparum, 2000–17: a spatial and temporal modelling study (Daniel J Weiss et al., 2019). https://pmc.ncbi.nlm.nih.gov/articles/PMC6675740/
- **PMC6842545** — Modelled and observed mean and seasonal relationships between climate, population density and malaria indicators in Cameroon (Amelie D. Mbouna et al., 2019). https://pmc.ncbi.nlm.nih.gov/articles/PMC6842545/
- **PMC6884483** — Malaria predictions based on seasonal climate forecasts in South Africa: A time series distributed lag nonlinear model (Yoonhee Kim et al., 2019). https://pmc.ncbi.nlm.nih.gov/articles/PMC6884483/
- **PMC6915811** — Malaria Data by District: An open-source web application for increasing access to malaria information (Sean Tomlinson et al., 2019). https://pmc.ncbi.nlm.nih.gov/articles/PMC6915811/
- **PMC7038892** — Dynamical Malaria Forecasts Are Skillful at Regional and Local Scales in Uganda up to 4 Months Ahead (Adrian M. Tompkins et al., 2019). https://pmc.ncbi.nlm.nih.gov/articles/PMC7038892/ *(licence: non-commercial)*
- **PMC6006329** — Spatio-temporal modelling of weekly malaria incidence in children under 5 for early epidemic detection in Mozambique (Kathryn L. Colborn et al., 2018). https://pmc.ncbi.nlm.nih.gov/articles/PMC6006329/
- **PMC6157844** — Uncertainty in malaria simulations in the highlands of Kenya: Relative contributions of model parameter setting, driving climate and initial condition errors (Adrian M. Tompkins et al., 2018). https://pmc.ncbi.nlm.nih.gov/articles/PMC6157844/
- **PMC6174038** — Modeling Spatio-temporal Malaria Risk Using Remote Sensing and Environmental Factors (Muhammad Haris MAZHER et al., 2018). https://pmc.ncbi.nlm.nih.gov/articles/PMC6174038/
- **PMC6200228** — Nets versus spraying: A spatial modelling approach reveals indoor residual spraying targets Anopheles mosquito habitats better than mosquito nets in Tanzania (Emily Sohanna Acheson et al., 2018). https://pmc.ncbi.nlm.nih.gov/articles/PMC6200228/
- **PMC5307250** — A Weather-Based Prediction Model of Malaria Prevalence in Amenfi West District, Ghana (Esther Love Darkoh et al., 2017). https://pmc.ncbi.nlm.nih.gov/articles/PMC5307250/
- **PMC5453969** — Using remote sensing environmental data to forecast malaria incidence at a rural district hospital in Western Kenya (Maquins Odhiambo Sewe et al., 2017). https://pmc.ncbi.nlm.nih.gov/articles/PMC5453969/
- **arXiv:1603.04773** — Malaria elimination campaigns in the Lake Kariba region of Zambia: a spatial dynamical model (Milen Nikolov et al., 2016). https://arxiv.org/abs/1603.04773
- **arXiv:1608.06222** — Modeling the Influence of Local Environmental Factors on Malaria Transmission in Benin and Its Implications for Cohort Study (Gilles Cottrell (UPD5 Pharmacie) et al., 2016). https://arxiv.org/abs/1608.06222
- **PMC4740093** — Forecasting paediatric malaria admissions on the Kenya Coast using rainfall (Stella Wanjugu Karuri et al., 2016). https://pmc.ncbi.nlm.nih.gov/articles/PMC4740093/
- **PMC4942778** — Advances in mapping malaria for elimination: fine resolution modelling of Plasmodium falciparum incidence (Victor A. Alegana et al., 2016). https://pmc.ncbi.nlm.nih.gov/articles/PMC4942778/
- **PMC4404580** — Malaria risk in Nigeria: Bayesian geostatistical modelling of 2010 malaria indicator survey data (Abbas B Adigun et al., 2015). https://pmc.ncbi.nlm.nih.gov/articles/PMC4404580/
- **PMC4470343** — Forecasting malaria in a highly endemic country using environmental and clinical predictors (Kate Zinszer et al., 2015). https://pmc.ncbi.nlm.nih.gov/articles/PMC4470343/
- **arXiv:1407.7612** — Mapping physiological suitability limits of malaria in Africa under climate change (Sadie J. Ryan et al., 2014). https://arxiv.org/abs/1407.7612
- **PMC4090176** — Testing a multi-malaria-model ensemble against 30 years of data in the Kenyan highlands (Daniel Ruiz et al., 2014). https://pmc.ncbi.nlm.nih.gov/articles/PMC4090176/
- **PMC4158077** — Development and validation of climate and ecosystem-based early malaria epidemic prediction models in East Africa (Andrew K Githeko et al., 2014). https://pmc.ncbi.nlm.nih.gov/articles/PMC4158077/
- **PMC3359352** — Spatially Explicit Burden Estimates of Malaria in Tanzania: Bayesian Geostatistical Modeling of the Malaria Indicator Survey Data (Laura Gosoniu et al., 2012). https://pmc.ncbi.nlm.nih.gov/articles/PMC3359352/
- **PMC3533056** — A scoping review of malaria forecasting: past work and future directions (Kate Zinszer et al., 2012). https://pmc.ncbi.nlm.nih.gov/articles/PMC3533056/ *(licence: non-commercial)*