# Research Opportunity Map

**Question:** How are machine learning and statistical models used to predict malaria risk or incidence in sub-Saharan Africa from climate, satellite, surveillance and intervention data; which practices are established, which are emerging since 2020, and what is missing for these models to guide district-level decisions (validation on unseen districts or years, forecasts 1 to 6 months ahead, uncertainty estimates, accounting for bed nets and spraying, shared code and data)? Compare arXiv preprints with published PMC studies, and East Africa with West Africa.

Based on **86 papers** (6 arxiv, 80 pmc), **65 read in full**, published 2010–2026.

**Setting:** Population and place: malaria-endemic human populations in sub-Saharan Africa (SSA), with explicit comparison of East Africa (Kenya, Uganda, Tanzania, Ethiopia, Rwanda, Mozambique, Madagascar) versus West Africa (Nigeria, Benin, Senegal, Mali, Burkina Faso, Ghana, The Gambia, Côte d'Ivoire). Unit of analysis: sub-national administrative units (district, county, region, health-facility catchment) or aggregated case counts/incidence rates reported at that level. Time frame: studies using surveillance, climate, satellite or intervention data from any historical period, with emphasis on work published 2015-2026 and a specific focus on what has emerged since 2020. Models of interest: statistical (GLM, GAM, ARIMA/SARIMA, spatio-temporal Bayesian, distributed lag) and machine-learning (random forest, gradient boosting, deep learning, hybrid) approaches that predict malaria risk, incidence …

## How to read this map

Every section is computed by code from evidence-checked extractions of the papers above. Grades list the reasons that produced them. Explanations of gaps are hypotheses that were each tested against the papers, and their verdicts come from the counts, not from the model. Paper ids after an item are examples, not the full list.

## What is established

- **E1 · data modalities: surveillance counts**: 74 of 86 papers (86%, 95% CI 77%–92%). Evidence **high**: 74 papers; confirmed in 55 full-text reads; found in both arXiv and PMC; 73 distinct place names; 47 papers in the last 5 years. [arXiv:2306.02685], [arXiv:2606.00834], [PMC10313820], [PMC10201255] …
- **E2 · data modalities: climate/environmental**: 71 of 86 papers (83%, 95% CI 73%–89%). Evidence **high**: 71 papers; confirmed in 56 full-text reads; found in both arXiv and PMC; 81 distinct place names; 46 papers in the last 5 years. [arXiv:2306.02685], [PMC10313820], [PMC10201255], [PMC11227462] …
- **E3 · spatial unit: district or finer**: 54 of 86 papers (63%, 95% CI 52%–72%). Evidence **high**: 54 papers; confirmed in 42 full-text reads; found in both arXiv and PMC; 69 distinct place names; 32 papers in the last 5 years. [arXiv:2606.00834], [PMC10201255], [PMC10707679], [PMC12676887] …
- **E4 · target variable: predicting actionable case burden at district level**: 49 of 86 papers (57%, 95% CI 46%–67%) (counts any of: incidence rate, case count). Evidence **high**: 49 papers; confirmed in 41 full-text reads; found in both arXiv and PMC; 57 distinct place names; 32 papers in the last 5 years. [arXiv:2606.00834], [PMC10201255], [PMC11997905], [PMC6884483] …
- **E5 · intervention covariates: none**: 36 of 86 papers (42%, 95% CI 32%–52%). Evidence **high**: 36 papers; confirmed in 30 full-text reads; found in both arXiv and PMC; 50 distinct place names; 20 papers in the last 5 years. [PMC12676887], [PMC6884483], [PMC8213140], [arXiv:2510.01302] …
- **E6 · target variable: incidence rate**: 30 of 86 papers (35%, 95% CI 26%–45%). Evidence **high**: 30 papers; confirmed in 23 full-text reads; found in both arXiv and PMC; 42 distinct place names; 20 papers in the last 5 years. [PMC10201255], [PMC11997905], [PMC6884483], [arXiv:2510.01302] …
- **E7 · model family: statistical timeseries**: 28 of 86 papers (33%, 95% CI 24%–43%). Evidence **high**: 28 papers; confirmed in 23 full-text reads; 36 distinct place names; 17 papers in the last 5 years. [PMC10201255], [PMC11227462], [PMC6884483], [PMC10341430] …
- **E8 · forecast horizon: operational 1-6 month lead time for district decisions**: 27 of 86 papers (31%, 95% CI 23%–42%) (counts any of: short 1 to 3 months, medium 4 to 6 months, multiple horizons reported). Evidence **high**: 27 papers; confirmed in 23 full-text reads; 40 distinct place names; 14 papers in the last 5 years. [PMC11997905], [PMC6884483], [PMC8213140], [PMC11983967] …
- **E9 · data modalities: survey**: 26 of 86 papers (30%, 95% CI 22%–41%). Evidence **high**: 26 papers; confirmed in 20 full-text reads; found in both arXiv and PMC; 39 distinct place names; 20 papers in the last 5 years. [arXiv:2306.02685], [PMC10201255], [PMC11227462], [PMC5380319] …

## What is emerging

- **R1 · spatial unit: region or province**: 12 papers; 3% of papers before 2022 vs 19% since (×5.68, one-sided p = 0.04). Evidence **high**: 12 papers; confirmed in 11 full-text reads; 16 distinct place names; 11 papers in the last 5 years; the rise is unlikely to be chance (one-sided Fisher p=0.040). [PMC11227462], [PMC5380319], [PMC11997905], [PMC10341430] …
- **R2 · data modalities: geospatial**: 12 papers; 3% of papers before 2022 vs 19% since (×5.68, one-sided p = 0.04). Evidence **high**: 12 papers; confirmed in 11 full-text reads; found in both arXiv and PMC; 24 distinct place names; 11 papers in the last 5 years; the rise is unlikely to be chance (one-sided Fisher p=0.040). [PMC11227462], [PMC10707679], [PMC13382336], [PMC11780933] …
- **R3 · intervention covariates: itn only**: 9 papers; 0% of papers before 2022 vs 16% since (new, one-sided p = 0.02). Evidence **high**: 9 papers; confirmed in 9 full-text reads; 19 distinct place names; 9 papers in the last 5 years; the rise is unlikely to be chance (one-sided Fisher p=0.020). [PMC11227462], [PMC10707679], [PMC11466501], [PMC8941165] …
- **R4 · intervention covariates: accounting for vector-control interventions in the model**: 23 papers; 17% of papers before 2022 vs 32% since (×1.84, one-sided p = 0.12). Evidence **high**: 23 papers; confirmed in 20 full-text reads; 35 distinct place names; 18 papers in the last 5 years; the rise is suggestive (p=0.12). [PMC10201255], [PMC11227462], [PMC5380319], [PMC10707679] …
- **R5 · data modalities: ehr structured**: 17 papers; 10% of papers before 2022 vs 25% since (×2.39, one-sided p = 0.10). Evidence **high**: 17 papers; confirmed in 17 full-text reads; 14 distinct place names; 14 papers in the last 5 years; the rise is suggestive (p=0.10). [PMC10313820], [PMC10201255], [PMC11098333], [PMC12676887] …
- **R6 · forecast horizon: long over 6 months**: 9 papers; 3% of papers before 2022 vs 14% since (×4.12, one-sided p = 0.12). Evidence **high**: 9 papers; confirmed in 8 full-text reads; found in both arXiv and PMC; 10 distinct place names; 8 papers in the last 5 years; the rise is suggestive (p=0.12). [arXiv:2606.00834], [PMC12922243], [PMC13563892], [PMC4740093] …
- **R7 · code or data shared: publicly shared code or data for reuse**: 6 papers; 0% of papers before 2022 vs 10% since (new, one-sided p = 0.08). Evidence **moderate**: only 6 papers; confirmed in 5 full-text reads; found in both arXiv and PMC; 12 distinct place names; 6 papers in the last 5 years; the rise is suggestive (p=0.08). [PMC12676887], [PMC10341430], [PMC10552258], [PMC10558065] …
- **R8 · model family: hybrid**: 11 papers; 7% of papers before 2022 vs 16% since (×2.29, one-sided p = 0.21). Evidence **moderate**: 11 papers; confirmed in 11 full-text reads; found in both arXiv and PMC; 13 distinct place names; 9 papers in the last 5 years; the rise could be chance at this sample size (p=0.21). [arXiv:2606.00834], [PMC11997905], [PMC8213140], [arXiv:2510.01302] …
- **R9 · validation split: cross validation only**: 5 papers; 0% of papers before 2022 vs 9% since (new, one-sided p = 0.12). Evidence **moderate**: only 5 papers; found in both arXiv and PMC; 6 distinct place names; 5 papers in the last 5 years; the rise is suggestive (p=0.12). [PMC11978812], [PMC10914794], [PMC12362852], [PMC10824718] …
- **R10 · code or data shared: on request**: 5 papers; 0% of papers before 2022 vs 9% since (new, one-sided p = 0.12). Evidence **moderate**: only 5 papers; confirmed in 5 full-text reads; found in both arXiv and PMC; 7 distinct place names; 5 papers in the last 5 years; the rise is suggestive (p=0.12). [arXiv:2606.00834], [PMC11227462], [PMC9453600], [arXiv:2606.00783] …
- **R11 · code or data shared**: 5 papers; 0% of papers before 2022 vs 9% since (new, one-sided p = 0.12). Evidence **moderate**: only 5 papers; found in both arXiv and PMC; 10 distinct place names; 5 papers in the last 5 years; the rise is suggestive (p=0.12). [PMC10341430], [PMC10552258], [PMC10558065], [PMC10021332] …
- **R12 · methods: support vector machine**: 5 papers; 0% of papers before 2022 vs 9% since (new, one-sided p = 0.12). Evidence **moderate**: only 5 papers; confirmed in 5 full-text reads; 6 distinct place names; 5 papers in the last 5 years; the rise is suggestive (p=0.12). [PMC11098333], [PMC10863265], [PMC11983967], [PMC10943795] …
- **R13 · data modalities: satellite imagery**: 13 papers; 10% of papers before 2022 vs 18% since (×1.7, one-sided p = 0.29). Evidence **moderate**: 13 papers; confirmed in 12 full-text reads; found in both arXiv and PMC; 20 distinct place names; 10 papers in the last 5 years; the rise could be chance at this sample size (p=0.29). [PMC10201255], [arXiv:2510.01302], [PMC4470343], [PMC13563892] …
- **R14 · data modalities: remote sensing**: 13 papers; 10% of papers before 2022 vs 18% since (×1.7, one-sided p = 0.29). Evidence **moderate**: 13 papers; confirmed in 12 full-text reads; 23 distinct place names; 10 papers in the last 5 years; the rise could be chance at this sample size (p=0.29). [PMC10313820], [PMC11227462], [PMC13563892], [PMC10828885] …
- **R15 · intervention covariates: both itn and irs**: 10 papers; 7% of papers before 2022 vs 14% since (×2.03, one-sided p = 0.28). Evidence **moderate**: 10 papers; confirmed in 8 full-text reads; 16 distinct place names; 8 papers in the last 5 years; the rise could be chance at this sample size (p=0.28). [PMC10201255], [PMC5380319], [PMC10341430], [PMC11593955] …
- **R16 · code or data shared: data only**: 4 papers; 0% of papers before 2022 vs 7% since (new, one-sided p = 0.19). Evidence **low**: only 4 papers; 11 distinct place names; 4 papers in the last 5 years; the rise is suggestive (p=0.19). [PMC12676887], [PMC10341430], [PMC10552258], [PMC10558065]
- **R17 · methods: bayesian spatio-temporal model**: 4 papers; 0% of papers before 2022 vs 7% since (new, one-sided p = 0.19). Evidence **low**: only 4 papers; 4 distinct place names; 4 papers in the last 5 years; the rise is suggestive (p=0.19). [PMC10313820], [PMC11978812], [PMC13382336], [PMC12317539]
- **R18 · methods: logistic regression**: 4 papers; 0% of papers before 2022 vs 7% since (new, one-sided p = 0.19). Evidence **low**: only 4 papers; 6 distinct place names; 4 papers in the last 5 years; the rise is suggestive (p=0.19). [PMC11098333], [PMC12676887], [PMC13399501], [PMC10914794]
- **R19 · methods: random forest**: 9 papers; 7% of papers before 2022 vs 12% since (×1.78, one-sided p = 0.36). Evidence **moderate**: 9 papers; confirmed in 9 full-text reads; 12 distinct place names; 7 papers in the last 5 years; the rise could be chance at this sample size (p=0.36). [PMC11098333], [PMC12676887], [PMC7522256], [PMC8213140] …
- **R20 · methods: negative binomial regression**: 6 papers; 3% of papers before 2022 vs 9% since (×2.59, one-sided p = 0.34). Evidence **low**: only 6 papers; confirmed in 6 full-text reads; 12 distinct place names; 5 papers in the last 5 years; the rise could be chance at this sample size (p=0.34). [PMC10863265], [PMC10341430], [PMC10828885], [PMC12625025] …
- **R21 · model family: tree ensemble**: 3 papers; 0% of papers before 2022 vs 5% since (new, one-sided p = 0.29). Evidence **low**: only 3 papers; 10 distinct place names; 3 papers in the last 5 years; the rise could be chance at this sample size (p=0.29). [PMC12676887], [PMC9453600], [PMC10558065]
- **R22 · baseline comparison: benchmarking against a simple baseline**: 5 papers; 3% of papers before 2022 vs 7% since (×2.06, one-sided p = 0.45). Evidence **moderate**: only 5 papers; confirmed in 5 full-text reads; found in both arXiv and PMC; 7 distinct place names; 4 papers in the last 5 years; the rise could be chance at this sample size (p=0.45). [arXiv:2606.00834], [PMC11997905], [PMC13563892], [PMC11780933] …
- **R23 · methods: generalized linear model**: 5 papers; 3% of papers before 2022 vs 7% since (×2.06, one-sided p = 0.45). Evidence **low**: only 5 papers; 8 distinct place names; 4 papers in the last 5 years; the rise could be chance at this sample size (p=0.45). [PMC10707679], [PMC10341430], [PMC8941165], [PMC7504016] …
- **R24 · target variable: outbreak binary**: 4 papers; 3% of papers before 2022 vs 5% since (×1.56, one-sided p = 0.59). Evidence **low**: only 4 papers; found in both arXiv and PMC; 11 distinct place names; 3 papers in the last 5 years; the rise could be chance at this sample size (p=0.59). [PMC12676887], [PMC9453600], [arXiv:2411.06436], [PMC4158077]
- **R25 · validation split: random split**: 4 papers; 3% of papers before 2022 vs 5% since (×1.56, one-sided p = 0.59). Evidence **low**: only 4 papers; 8 distinct place names; 3 papers in the last 5 years; the rise could be chance at this sample size (p=0.59). [PMC12676887], [PMC10863265], [PMC10943795], [PMC3359352]
- **R26 · baseline comparison: seasonal naive or persistence**: 4 papers; 3% of papers before 2022 vs 5% since (×1.56, one-sided p = 0.59). Evidence **low**: only 4 papers; 6 distinct place names; 3 papers in the last 5 years; the rise could be chance at this sample size (p=0.59). [PMC11997905], [PMC13563892], [PMC11780933], [PMC3081772]

## What is missing

### G1 · reporting delay handling: yes modelled

- **Observed:** 0 of 86 papers (0%, 95% CI 0%–4%); 0 of 65 papers read in full.
- **Gap confidence:** **moderate**: absent from all 65 papers read in full; the field is reported by only 15% of papers; absence may be non-reporting; absent in both arXiv and PMC papers; rare across the corpus too: 1 of 336 topic papers mention it; true rate in this literature plausibly up to 4% (95% upper bound)
- **Corpus check:** 1 of 336 topic papers in the whole corpus match `malaria (forecast OR prediction) (nowcasting OR "reporting delay" OR "backfill" OR "delay distribution")`.
- **Why (reasoning):** Reporting-delay handling is genuinely absent, but the absence is partly a non-reporting artifact: only 13 of 86 papers state any value for the field (H9), and of those 13 none model it — all are 'no' or 'acknowledged_only' (H4). The practice is also absent among the 17 nowcast_or_none papers (H17), i.e. even studies that predict the current period do not correct for the fact that recent surveillance counts are incomplete. The corpus check (1 of 336 topic papers) confirms this is a field-wide blind spot, not a quirk of the shortlist.
  - H4 (re-measures the gap itself, not an explanation; 0 of 13 papers (0%); bounds pass): Reporting-delay handling is absent even where the field is reported: of the 13 papers that state a value for q_reporting_delay_handling, none model it (all are 'no' or 'acknowledged_only').
  - H9 (artifact, **supported**, 13 of 86 papers (15%); bounds pass): Cause: reporting-delay handling is absent because the field is rarely reported at all — only 13 of 86 papers state any value, so the gap is partly a non-reporting artifact rather than a proven absence of practice.
  - H17 (cause, **supported**, 0 of 17 papers (0%); bounds pass): Cause: reporting-delay handling is absent because the studies are retrospective model-fitting exercises, not operational nowcasting systems — of the 17 nowcast_or_none papers, none model reporting delay.
- **Could it be an artifact?** medium risk. The field is reported by only 15% of papers, so 'absent' partly means 'not reported'; however, among the 13 that do report, none model delay, and the corpus-wide rate is ~0.3%.
- **Near miss** [arXiv:2606.00834]: Explicitly acknowledges that routine surveillance is affected by changes in reporting practices and data completeness; missing: Acknowledges the problem but does not model or correct for reporting delay
- **Near miss** [PMC7038892]: Flags reporting-delay as a limitation; missing: No delay distribution or backfill adjustment in the model
- **Would change if:** A paper that fits a delay distribution or backfills recent counts before forecasting would show the practice exists but is under-reported; none found.
- **Untested speculation:** Operational nowcasting systems in ministries of health may handle delay internally without publishing it, so the literature gap may understate real practice.

### G2 · validation split: testing on places the model never saw

- **Observed:** 3 of 86 papers (4%, 95% CI 1%–10%); 2 of 65 papers read in full. (counts any of: spatial holdout, spatiotemporal holdout)
- **Gap confidence:** **moderate**: in only 2 of 65 papers read in full; rare across the corpus too: 0 of 336 topic papers mention it; true rate in this literature plausibly up to 10% (95% upper bound)
- **Corpus check:** 0 of 336 topic papers in the whole corpus match `(malaria (forecast OR prediction) ("spatial cross-validation" OR "leave-one-district-out" OR "leave-one-region-out" OR "spatial holdout")) OR (malaria (forecast OR prediction) ("spatiotemporal cross-validation" OR "leave-one-location-out" OR "blocked cross-validation"))`.
- **Why (reasoning):** Spatial holdout is rare for structural reasons, not because of abstract-only reading: only 2 of 65 full-text papers use it (H1), and it is absent from all 6 arXiv papers (H8). The dominant statistical time-series family (ARIMA/SARIMAX/GLM) is single-series and almost never does spatial holdout — 1 of 28 (H7) — while even spatiotemporal Bayesian models, which are built for multiple locations, do so in only 2 of 20 (10%). The gap is field-wide rather than regional: equally rare in West Africa (1 of 35, H14) as in East Africa (2 of 43).
  - H1 (artifact, **supported**, 2 of 65 papers (3%); bounds pass): The spatial-holdout gap is not an artifact of abstract-only reading: among the 65 papers read in full, only 2 (3.1%) use a spatial or spatiotemporal holdout, so the absence persists at full-text depth.
  - H7 (cause, **supported**, 1 of 28 papers (4%); bounds pass): Cause: the dominant statistical time-series family (ARIMA/SARIMAX/GLM) is structurally single-series and rarely does spatial holdout — only 1 of 28 statistical_timeseries papers uses a spatial or spatiotemporal holdout.
  - H8 (artifact, **supported**, 0 of 6 papers (0%); bounds pass): Artifact/corpus: the spatial-holdout gap is not a PMC-only artifact — 0 of 6 arXiv papers use a spatial or spatiotemporal holdout, so the practice is absent in both corpora.
  - H14 (artifact, **supported**, 1 of 35 papers (3%); bounds pass): Artifact/region: the spatial-holdout gap is not an East-Africa artifact — it is equally rare in West Africa (1 of 35 papers, 3%) as in East Africa (2 of 43, 5%), so it is a field-wide practice gap.
  - H2 (cause, **supported**, 3 of 54 papers (6%); bounds pass): Spatial holdout is rare because most studies are single-location/single-district designs where there is no second district to hold out: among papers coded district_or_finer, only 3 of 54 use a spatial or spatiotemporal holdout.
- **Could it be an artifact?** low risk. Absence persists at full-text depth (2/65), in both corpora, and in both regions; the field is stated for 38% of papers.
- **Near miss** [PMC11780933]: Trains a spatio-temporal hierarchical Bayesian (INLA) model on 195 communities and validates via cross-validation across space and time; missing: Cross-validation rather than a true leave-one-district-out holdout; still one of only two spatial-holdout papers
- **Near miss** [PMC10201255]: Uses spatial block cross-validation and reports the model performs worse at distant, environmentally distinct locations; missing: Spatial blocks are within the same surveillance network, not a fully unseen district
- **Near miss** [PMC6420752]: Extrapolates a risk model to a neighbouring district using independent surveillance data; missing: Abstract-only; single neighbouring district, not a systematic spatial holdout
- **Would change if:** A cluster of ML papers using leave-one-district-out or blocked spatial CV would show the practice is emerging; currently only 2 papers do it.
- **Untested speculation:** Spatial holdout is hard when a study has only one district's surveillance data, so the gap may partly reflect study design (single-site) rather than methodological choice.

### G3 · validation beyond the development data (external, prospective or trial)

- **Observed:** 3 of 86 papers (4%, 95% CI 1%–10%); 2 of 65 papers read in full.
- **Gap confidence:** **moderate**: in only 2 of 65 papers read in full; the field is reported by 78% of papers, so absence is informative; true rate in this literature plausibly up to 10% (95% upper bound)
- **Why (reasoning):** External validation is genuinely rare and not a reporting artifact: validation_level is populated for 78% of papers, yet only 2 of 65 full-text papers report external/prospective/trial validation (H3). It is not concentrated in district-level studies (3 of 54, H10 — so it is spread thin everywhere), and even the most rigorous internal designs do not translate into external testing: of 21 papers using a temporal holdout, only 2 also report external validation (H16). This links to G2 — the same papers that avoid spatial holdout also avoid external settings.
  - H3 (artifact, **supported**, 2 of 65 papers (3%); bounds pass): The external-validation gap is not a reporting artifact: validation_level is populated for 78% of papers, yet only 2 of 65 full-text papers report external/prospective/trial validation.
  - H10 (cause, **not supported**, 3 of 54 papers (6%); bounds fail): Cause: external validation is rare because studies are single-site designs with no second setting to validate in — of the 54 district_or_finer papers, only 2 report external/prospective/trial validation.
  - H16 (cause, **supported**, 2 of 21 papers (10%); bounds pass): Cause: external validation is rare even among the most rigorous internal designs — of the 21 papers using a temporal holdout, only 2 (10%) also report external/prospective/trial validation, so rigour in splitting does not translate into external testing.
- **Could it be an artifact?** low risk. The field is well-populated (78% stated) and the absence holds at full-text depth; it is not a non-reporting artifact.
- **Near miss** [PMC5453969]: Withholds the year 2013 from model building to test predictive generalizability at a district hospital in Western Kenya; missing: The GAM could not generalize to the external 2013 data — an honest negative result, but still only temporal, not a second district
- **Near miss** [PMC12625025]: Uses 2017-2024 as an out-of-sample holdout for a sub-district outbreak model in Zambia; missing: Holdout is future years in the same catchment, not an unseen district
- **Near miss** [PMC6420752]: Tests extrapolation to a neighbouring district with independent surveillance data; missing: Abstract-only and a single neighbouring district
- **Would change if:** Papers reporting prospective deployment or validation in a held-out district/region would show external validation is emerging; only 2-3 partial cases exist.
- **Untested speculation:** External validation requires a second funded field site or a prospective deployment, which most academic modelling projects lack.

### Reasoning on N1

- **Why (reasoning):** The operational 1-6 month horizon gap is driven by data type, not region: survey-based studies (DHS/MIS prevalence) are cross-sectional and almost never produce operational forecasts — only 1 of 26 survey papers reports a short/medium/multiple horizon (H12). The gap is not region-specific: East Africa reports a 1-6 month horizon in 11 of 43 papers (26%) and West Africa in 11 of 35 (31%) (H15), so both regions are similarly under-served. The horizon gap therefore tracks the surveillance-vs-survey design split rather than geography.
  - H12 (cause, **supported**, 1 of 26 papers (4%); bounds pass): Cause: survey-based studies (DHS/MIS prevalence) rarely produce operational 1-6 month forecasts because surveys are cross-sectional — of the 26 survey papers, only 1 reports a short/medium/multiple operational horizon.
  - H15 (artifact, **supported**, 11 of 35 papers (31%); bounds pass): Artifact/region: the operational-horizon gap is not region-specific — East Africa reports a 1-6 month horizon in 11 of 43 papers (26%) and West Africa in 11 of 35 (31%), so both regions are similarly under-served.
- **Could it be an artifact?** medium risk. q_forecast_horizon is not stated for 33 of 86 papers, so some operational horizons may be unreported; but the survey-vs-surveillance split is a clear structural driver.
- **Near miss** [PMC12625025]: Predicts seasonal malaria cases up to four months ahead at sub-district level in Zambia; missing: Single catchment; horizon is seasonal rather than a fixed 1-6 month operational lead
- **Near miss** [PMC11780933]: Predicts malaria cases up to three months in advance at village level and integrates into an automated workflow; missing: One of the few true operational-horizon papers; still not multi-district
- **Would change if:** More surveillance-based district studies reporting fixed 1-6 month lead times would close the gap; currently concentrated in a handful of papers.
- **Untested speculation:** Operational horizons may be under-reported because academic papers frame results as model skill rather than decision lead time.

### Reasoning on N2

- **Why (reasoning):** Survey-based studies rarely use temporal holdout because their data are not a time series: only 1 of 26 survey papers uses a temporal holdout (H13). This is the same structural driver as N1 — cross-sectional DHS/MIS prevalence data cannot support temporal validation, so these papers default to random or spatial splits. The gap is a data-type consequence rather than a methodological oversight.
  - H13 (cause, **supported**, 1 of 26 papers (4%); bounds pass): Cause: survey-based studies rarely use temporal holdout because their data are not a time series — of the 26 survey papers, only 1 uses a temporal holdout.
- **Could it be an artifact?** medium risk. q_validation_split is unstated for 53 of 86 papers, so some temporal holdouts may be unreported; but the survey/time-series mismatch is a clear structural cause.
- **Near miss** [PMC11227462]: GIS-based spatiotemporal mapping of malaria prevalence from Nigeria DHS/MIS surveys; missing: Survey-based, so no temporal holdout; maps prevalence rather than forecasting
- **Would change if:** Repeated cross-sectional surveys linked into a panel would allow temporal validation of survey-based models; not seen here.
- **Untested speculation:** As DHS/MIS rounds accumulate, survey-based temporal validation may become feasible and the gap may narrow.

### Reasoning on N5

- **Why (reasoning):** The uncertainty gap is not caused by the spatiotemporal Bayesian family: of 20 spatiotemporal_bayesian papers, only 1 reports a point estimate only (H11), so this family generally does supply posterior intervals. The point-estimate-only papers (23 of 86) come from other families, so the uncertainty gap is spread across model types rather than concentrated in the Bayesian spatiotemporal work that is best equipped to quantify it.
  - H11 (link, **supported**, 1 of 20 papers (5%); bounds pass): Link: spatiotemporal Bayesian models, which could produce full posterior uncertainty, are not the ones giving intervals — of the 20 spatiotemporal_bayesian papers, only 1 reports a point estimate only, so this family is not the source of the uncertainty gap.
- **Could it be an artifact?** medium risk. q_probabilistic_forecast is unstated for 37 of 86 papers, so some intervals may be unreported; but the Bayesian family clearly does report them.
- **Near miss** [PMC11780933]: INLA-based spatio-temporal model producing full posterior uncertainty; missing: Uncertainty is reported but not propagated to the operational ACT-ordering decision
- **Would change if:** Evidence that point-estimate-only papers cluster in a specific family (e.g. ML ensembles) would localise the gap; current data show it is spread.
- **Untested speculation:** ML papers may omit intervals because tree/ensemble methods do not natively produce them, unlike Bayesian models.

### Patterns across gaps

- Structural driver: the surveillance-vs-survey data split explains several gaps at once — survey papers lack operational horizons (H12) and temporal holdout (H13), while surveillance time-series papers lack spatial holdout (H7). The gaps are largely a consequence of study design, not oversight.
- Model family matters less than expected: statistical time-series papers rarely do spatial holdout (H7) and spatiotemporal Bayesian papers rarely do it either (2/20), so no family is driving the validation gap — it is field-wide.
- The gaps are not artifacts of corpus or region: spatial holdout is absent in both arXiv (H8) and PMC, and equally rare in East (2/43) and West Africa (1/35, H14); operational horizons are similarly distributed (H15).
- Non-reporting is a real but partial confound: reporting-delay handling (H9) and code/data sharing are stated by only ~15% of papers, so their gaps are partly 'not reported'; by contrast validation_level is stated by 78% (H3), so the external-validation gap is a genuine absence.
- Rigour does not compound: papers with the strongest internal design (temporal holdout) still rarely add external validation (H16), so improving one practice does not automatically improve the other.

## Evidence strength at a glance

| Item | Papers | Full-text | Corpora | Grade |
|---|---|---|---|---|
| E1 data modalities: surveillance counts | 74/86 | 55 | arxiv, pmc | high |
| E2 data modalities: climate/environmental | 71/86 | 56 | arxiv, pmc | high |
| E3 spatial unit: district or finer | 54/86 | 42 | arxiv, pmc | high |
| E4 target variable: predicting actionable case burden at district level | 49/86 | 41 | arxiv, pmc | high |
| E5 intervention covariates: none | 36/86 | 30 | arxiv, pmc | high |
| E6 target variable: incidence rate | 30/86 | 23 | arxiv, pmc | high |
| E7 model family: statistical timeseries | 28/86 | 23 | pmc | high |
| E8 forecast horizon: operational 1-6 month lead time for district decisions | 27/86 | 23 | pmc | high |
| E9 data modalities: survey | 26/86 | 20 | arxiv, pmc | high |
| R1 spatial unit: region or province | 12/86 | 11 | pmc | high |
| R2 data modalities: geospatial | 12/86 | 11 | arxiv, pmc | high |
| R3 intervention covariates: itn only | 9/86 | 9 | pmc | high |
| R4 intervention covariates: accounting for vector-control interventions in the model | 23/86 | 20 | pmc | high |
| R5 data modalities: ehr structured | 17/86 | 17 | pmc | high |
| R6 forecast horizon: long over 6 months | 9/86 | 8 | arxiv, pmc | high |
| R7 code or data shared: publicly shared code or data for reuse | 6/86 | 5 | arxiv, pmc | moderate |
| R8 model family: hybrid | 11/86 | 11 | arxiv, pmc | moderate |
| R9 validation split: cross validation only | 5/86 | 4 | arxiv, pmc | moderate |
| R10 code or data shared: on request | 5/86 | 5 | arxiv, pmc | moderate |
| R11 code or data shared | 5/86 | 4 | arxiv, pmc | moderate |
| R12 methods: support vector machine | 5/86 | 5 | pmc | moderate |
| R13 data modalities: satellite imagery | 13/86 | 12 | arxiv, pmc | moderate |
| R14 data modalities: remote sensing | 13/86 | 12 | pmc | moderate |
| R15 intervention covariates: both itn and irs | 10/86 | 8 | pmc | moderate |
| R16 code or data shared: data only | 4/86 | 4 | pmc | low |
| R17 methods: bayesian spatio-temporal model | 4/86 | 4 | pmc | low |
| R18 methods: logistic regression | 4/86 | 4 | pmc | low |
| R19 methods: random forest | 9/86 | 9 | pmc | moderate |
| R20 methods: negative binomial regression | 6/86 | 6 | pmc | low |
| R21 model family: tree ensemble | 3/86 | 3 | pmc | low |
| R22 baseline comparison: benchmarking against a simple baseline | 5/86 | 5 | arxiv, pmc | moderate |
| R23 methods: generalized linear model | 5/86 | 4 | pmc | low |
| R24 target variable: outbreak binary | 4/86 | 2 | arxiv, pmc | low |
| R25 validation split: random split | 4/86 | 4 | pmc | low |
| R26 baseline comparison: seasonal naive or persistence | 4/86 | 4 | pmc | low |
| G1 reporting delay handling: yes modelled (gap) | 0/86 | 0/65 | none | moderate confidence |
| G2 validation split: testing on places the model never saw (gap) | 3/86 | 2/65 | pmc | moderate confidence |
| G3 validation beyond the development data (external, prospective or trial) (gap) | 3/86 | 2/65 | pmc | moderate confidence |

## Potential novelty

Pairs of components from different dimensions (for example a method and a data source) that are each common in these papers but rarely combined. *Expected* is how often they would co-occur if chosen independently; the p-value is the hypergeometric chance of seeing this few or fewer. Many pairs are compared, so treat single p-values near 0.05 as leads, not findings.

- **N1 · forecast horizon: operational 1-6 month lead time for district decisions + data modalities: survey**: together in 1 paper(s), 8.2 expected (27 and 26 papers use each; chance of so few by independence p < 0.001). Together in [PMC11466501].
- **N2 · validation split: temporal holdout + data modalities: survey**: together in 1 paper(s), 6.3 expected (21 and 26 papers use each; chance of so few by independence p = 0.002). Together in [PMC12995307].
- **N3 · target variable: case count + data modalities: survey**: together in 1 paper(s), 5.7 expected (19 and 26 papers use each; chance of so few by independence p = 0.005). Together in [PMC8941165].
- **N4 · methods: logistic regression + data modalities: surveillance counts**: together in 1 paper(s), 3.4 expected (4 and 74 papers use each; chance of so few by independence p = 0.008). Together in [PMC11098333].
- **N5 · probabilistic forecast: point estimate only + model family: spatiotemporal bayesian**: together in 1 paper(s), 5.3 expected (23 and 20 papers use each; chance of so few by independence p = 0.008). Together in [PMC11780933].
- **N6 · methods: logistic regression + data modalities: climate/environmental**: together in 1 paper(s), 3.3 expected (4 and 71 papers use each; chance of so few by independence p = 0.02). Together in [PMC11098333].
- **N7 · target variable: incidence rate + methods: random forest**: together in 0 paper(s), 3.1 expected (30 and 9 papers use each; chance of so few by independence p = 0.02). 
- **N8 · methods: arima + data modalities: survey**: together in 0 paper(s), 3.0 expected (10 and 26 papers use each; chance of so few by independence p = 0.02). 

## Candidate research designs

### D1 · Leave-one-district-out benchmark of climate+surveillance malaria forecasters across East and West Africa

*How much forecast skill is lost when a district-level malaria model is evaluated on districts it has never seen, and does that loss differ between East and West Africa and between arXiv and PMC model families?*

| Slot | Choice |
|---|---|
| Target | Monthly malaria incidence rate (cases per 1,000 person-months) at district level, forecast 1-3 months ahead; secondary target case counts. |
| Predictors | CHIRPS rainfall (lagged 1-3 months), MODIS/ERA5 land surface temperature and NDVI, seasonality (month, Fourier terms), recent surveillance counts (autoregressive lags), population denominators (WorldPop), elevation (SRTM) |
| Data sources | DHIS2/HMIS district surveillance (Uganda UMSP, Tanzania HMIS, Burkina Faso DHIS2, Benin, Zambia), CHIRPS, MODIS MOD11A2/MOD13, ERA5 / ERA5-Land, WorldPop |
| Horizon | 1-3 months ahead (with 4-6 month secondary horizon) |
| Spatial unit | District / county (admin-2) |
| Validation strategy | Leave-one-district-out (spatial holdout) as the primary split, benchmarked against the field-standard random and temporal holdouts on the same data; report the skill gap between splits. This directly fixes G2, where only 2 of 65 full-text papers use a spatial or spatiotemporal holdout (H1, H2). |
| Baseline | Seasonal-naive / persistence (last year same month), the only baseline used by the 4 papers that report one (PMC11780933, PMC11997905, PMC13563892, PMC3081772). |
| Addresses | G2, G3, E1 |

**Rests on:** H1 (supported): The spatial-holdout gap is not an artifact of abstract-only reading: among the 65 papers read in full, only 2 (3.1%) use a spatial or spatiotemporal holdout, so the absence persists at full-text depth.; H2 (supported): Spatial holdout is rare because most studies are single-location/single-district designs where there is no second district to hold out: among papers coded district_or_finer, only 3 of 54 use a spatial or spatiotemporal holdout.

**Builds on:** [PMC11780933], [PMC10201255], [PMC12625025], [PMC5453969], [PMC11593955], [PMC10824718]

**Evidence supporting this direction:**

- [PMC11780933] Increasing the resolution of malaria early warning systems for use by local health actors (2025): The only full-text paper combining a spatiotemporal holdout with a 1-3 month horizon and a seasonal-naive baseline in Madagascar; provides a working template for the split and the baseline. From the paper: "A spatio-temporal hierarchical generalized linear regression model was trained on monthly malaria case data from 195 communities"
- [PMC10201255] Mapping malaria incidence using routine health facility surveillance data in Uganda (2023): Maps malaria incidence from routine facility surveillance in Uganda using a spatial holdout and both ITN and IRS covariates, showing district-level spatial validation is feasible with DHIS2-type data. From the paper: "we specified a spatio-temporal GAM with GMRF smooths to account for spatial autocorrelation"
- [PMC12625025] Malaria outbreak prediction at the sub-district level in Zambia using remote sensing satellite data (2025): Sub-district outbreak prediction in Zambia with CHIRPS+MODIS and a 4-6 month horizon, demonstrating the climate/satellite covariate stack the design reuses. From the paper: "A negative binomial regression model was then trained on 2010–2016 data to predict total cases in a malaria season"
- [PMC5453969] Using remote sensing environmental data to forecast malaria incidence at a rural district hospital in Western Kenya (2017): Western Kenya district-hospital forecast with TRMM/MODIS and external validation, an East African anchor for the cross-region comparison. From the paper: "We developed two different general additive models, one using a boosting algorithm to optimize model fit and the other without boosting."

**Evidence challenging this direction:**

- [PMC13399501] Multi-scenario evaluation of federated learning for privacy-preserving malaria prediction with Ghana DHS data (2026): Authors state their regional clients are 'statistical entities from one unified data pool, not data-generating institutions' and that partitioning was simulated, warning that cross-district generalization claims can be inflated by shared data-generating processes. From the paper: "comparing FedAvg [5] and FedProx [6] under three data distribution scenarios using Ghana DHS/MIS datasets"
- [PMC5324298] Integrating malaria surveillance with climate data for outbreak detection and forecasting: the EPIDEMIA system (2017): EPIDEMIA authors flag 'availability of timely, high-quality data is a key limiting factor' and 'need to be adaptable to changes in the input data' - spatial holdout across districts with heterogeneous reporting quality may fail for data reasons, not model reasons. From the paper: "EASTWeb, an open-source client-based application that automatically connects to earth observation data archives and acquires, processes, and summarizes selected remote sensing datasets"
- [PMC10914794] Identifying childhood malaria hotspots and risk factors in a Nigerian city using geostatistical modelling approach (2024): Documents 'paucity of malaria data' and 'barriers on data availability' in a Nigerian city, so West African districts may not have enough complete series to hold out. From the paper: "model-based geostatistical modeling (MBG) technique to predict U5 malaria burden at a 100 × 100 m grid"

**Risks:** Few districts have long, complete monthly series, so leave-one-district-out may leave too few test units.; Spatial holdout conflates model failure with genuine ecological differences between districts.; Cross-region pooling requires harmonizing case definitions and reporting systems across countries.

### D2 · Reporting-delay-corrected nowcasting and 1-6 month forecasting for district early warning

*Does explicitly modelling surveillance reporting delay (backfill) change the accuracy and calibration of district-level malaria nowcasts and 1-6 month forecasts, and does it close the gap between retrospective skill and operational skill?*

| Slot | Choice |
|---|---|
| Target | Nowcast and 1-6 month-ahead district malaria case counts, with prediction intervals; secondary target: probability that a district exceeds its seasonal threshold. |
| Predictors | Reporting-delay distribution (time from case onset to DHIS2 entry), recent partially-reported counts, CHIRPS rainfall and ERA5 temperature lags, seasonality, ITN/IRS coverage |
| Data sources | DHIS2/HMIS with report timestamps (Uganda, Tanzania, Burkina Faso, Mozambique SIS-MA), CHIRPS, ERA5, Malaria Indicator Survey / DHS ITN coverage |
| Horizon | Nowcast plus 1-6 months ahead (multiple horizons reported) |
| Spatial unit | District / health-facility catchment |
| Validation strategy | Temporal holdout on future years the model never saw, plus a simulated real-time evaluation where only data available at forecast issue date (with realistic reporting lag) is used. This targets G1, where 0 of 86 papers model reporting delay (H4) and 0 of 17 nowcast papers do (H17). |
| Baseline | Uncorrected model on raw counts (the field default) and seasonal-naive. |
| Addresses | G1, G3 |

**Rests on:** H4 (supported, restates the gap): Reporting-delay handling is absent even where the field is reported: of the 13 papers that state a value for q_reporting_delay_handling, none model it (all are 'no' or 'acknowledged_only').; H9 (supported): Cause: reporting-delay handling is absent because the field is rarely reported at all — only 13 of 86 papers state any value, so the gap is partly a non-reporting artifact rather than a proven absence of practice.; H17 (supported): Cause: reporting-delay handling is absent because the studies are retrospective model-fitting exercises, not operational nowcasting systems — of the 17 nowcast_or_none papers, none model reporting delay.

**Builds on:** [PMC11997905], [PMC5324298], [PMC11780933], [PMC11593955]

**Evidence supporting this direction:**

- [PMC11997905] Infectious disease forecasting to support public health: use of readily available methods to predict malaria and diarrhoeal diseases in Mozambique (2025): Mozambique study uses SIS-MA surveillance with multiple horizons, temporal holdout and WIS, providing the probabilistic scoring framework and an operational data source. From the paper: "we employed three models that represent these three model types: Exponential Smoothing"
- [PMC5324298] Integrating malaria surveillance with climate data for outbreak detection and forecasting: the EPIDEMIA system (2017): EPIDEMIA is the closest thing to an operational near-real-time system (Amhara, Ethiopia) and explicitly names timely high-quality data as the limiting factor - the exact problem delay correction addresses. From the paper: "EASTWeb, an open-source client-based application that automatically connects to earth observation data archives and acquires, processes, and summarizes selected remote sensing datasets"
- [PMC11593955] Impact of Climate Variability and Interventions on Malaria Incidence and Forecasting in Burkina Faso (2024): Burkina Faso DHIS2 study with 1-3 month horizon and 95% BCI shows the surveillance+climate+intervention stack and interval reporting are already in use. From the paper: "Bayesian generalized autoregressive moving average negative binomial models"

**Evidence challenging this direction:**

- [PMC5324298] Integrating malaria surveillance with climate data for outbreak detection and forecasting: the EPIDEMIA system (2017): Authors note 'ongoing challenges in evaluating and improving forecasting models' and the need to 'better incorporate early detection and early warning results into decision making', suggesting delay correction alone may not translate into operational gains. From the paper: "EASTWeb, an open-source client-based application that automatically connects to earth observation data archives and acquires, processes, and summarizes selected remote sensing datasets"
- [PMC13399501] Multi-scenario evaluation of federated learning for privacy-preserving malaria prediction with Ghana DHS data (2026): Warns that retrospective data with simulated partitioning overstates real deployment performance; a delay-corrected retrospective evaluation may still not reflect operational reality. From the paper: "comparing FedAvg [5] and FedProx [6] under three data distribution scenarios using Ghana DHS/MIS datasets"
- [PMC10914794] Identifying childhood malaria hotspots and risk factors in a Nigerian city using geostatistical modelling approach (2024): Reports 'paucity of malaria data' and data-availability barriers, so report-timestamp data needed to estimate delay distributions may be missing in many districts. From the paper: "model-based geostatistical modeling (MBG) technique to predict U5 malaria burden at a 100 × 100 m grid"

**Risks:** Report timestamps are often not retained in DHIS2 extracts, so delay distributions may have to be assumed rather than estimated.; Delay correction adds parameters that may not be identifiable with short series.; Operational evaluation requires data snapshots that most published datasets do not preserve.

### D3 · Intervention-aware district forecasting: does adding ITN and IRS coverage improve skill and change district prioritization?

*Do ITN and IRS coverage covariates improve district-level malaria forecast skill and alter which districts are flagged as high-risk, compared with climate-only models?*

| Slot | Choice |
|---|---|
| Target | Monthly district malaria incidence rate (cases per 1,000 person-months), 1-6 months ahead, with prediction intervals. |
| Predictors | ITN ownership/use coverage (survey-interpolated), IRS coverage / spray rounds, CHIRPS rainfall, ERA5/MODIS temperature and NDVI, seasonality, recent incidence lags |
| Data sources | DHIS2/HMIS district surveillance, Malaria Indicator Survey / DHS ITN indicators, PMI/National Malaria Programme IRS records, CHIRPS, ERA5, MODIS |
| Horizon | 1-6 months ahead |
| Spatial unit | District (admin-2) |
| Validation strategy | Spatiotemporal holdout: hold out both future years and a set of districts, so intervention effects are not learned from the same districts being predicted. This addresses G2 and the intervention gap, where 36 of 59 papers stating a value use no intervention covariates and only 10 use both ITN and IRS. |
| Baseline | Climate-only model (no intervention covariates) and seasonal-naive. |
| Addresses | G2, G3 |

**Rests on:** H1 (supported): The spatial-holdout gap is not an artifact of abstract-only reading: among the 65 papers read in full, only 2 (3.1%) use a spatial or spatiotemporal holdout, so the absence persists at full-text depth.; H2 (supported): Spatial holdout is rare because most studies are single-location/single-district designs where there is no second district to hold out: among papers coded district_or_finer, only 3 of 54 use a spatial or spatiotemporal holdout.

**Builds on:** [PMC10201255], [PMC10824718], [PMC11593955], [PMC11466501]

**Evidence supporting this direction:**

- [PMC10201255] Mapping malaria incidence using routine health facility surveillance data in Uganda (2023): Uganda incidence mapping already integrates both ITN and IRS covariates with a spatial holdout, proving the covariate stack and split are jointly feasible. From the paper: "we specified a spatio-temporal GAM with GMRF smooths to account for spatial autocorrelation"
- [PMC10824718] Predicting malaria risk considering vector control interventions under climate change scenarios (2024): Uganda study predicts malaria risk under climate change while explicitly considering vector control interventions (both ITN and IRS), showing intervention-adjusted forecasting is tractable. From the paper: "a generalized linear model based on a negative binomial distribution was used to estimate the association between malaria incident cases and environmental factors"
- [PMC11466501] Forecasting malaria dynamics based on causal relations between control interventions, climatic factors, and disease incidence in western Kenya (2024): Western Kenya study uses causal relations between control interventions, climate and incidence with ITN covariates, supporting the intervention-aware framing. From the paper: "we used convergent cross-mapping (CCM) to identify suitable lags for climatic drivers"
- [PMC11593955] Impact of Climate Variability and Interventions on Malaria Incidence and Forecasting in Burkina Faso (2024): Burkina Faso study combines both ITN and IRS with climate and reports posterior inclusion probabilities, giving a template for testing whether intervention terms matter. From the paper: "Bayesian generalized autoregressive moving average negative binomial models"

**Evidence challenging this direction:**

- [PMC13491694] Modelling the impact of climate variability on malaria morbidity in the Tamale Metropolitan Area: a time series analysis (2026): Authors state 'intervention coverage and health system data not integrated' and that 'climate represents only one component of malaria transmission dynamics', indicating intervention data are often unavailable or too coarse to help. From the paper: "The Autoregressive Distributed Lag (ARDL) bounds testing approach [20] was therefore selected"
- [PMC13399501] Multi-scenario evaluation of federated learning for privacy-preserving malaria prediction with Ghana DHS data (2026): Shows that under regional prevalence heterogeneity performance degrades (Greater Accra's 2.9% prevalence yielded ~50% false negative rate), so intervention-adjusted models may still misclassify low-burden districts. From the paper: "comparing FedAvg [5] and FedProx [6] under three data distribution scenarios using Ghana DHS/MIS datasets"
- [PMC10914794] Identifying childhood malaria hotspots and risk factors in a Nigerian city using geostatistical modelling approach (2024): Notes ITN use and window protection reduce risk but data availability is a barrier, so intervention covariates may be missing exactly where they matter. From the paper: "model-based geostatistical modeling (MBG) technique to predict U5 malaria burden at a 100 × 100 m grid"

**Risks:** ITN/IRS coverage is measured at coarse survey intervals and may not vary enough at monthly district scale.; Intervention coverage is endogenous to incidence, risking reverse causation.; Spatiotemporal holdout reduces training data substantially.

### D4 · Probabilistic district forecasts with shared code: a reproducible benchmark of calibration and uncertainty across corpora

*Are district-level malaria forecasts from arXiv and PMC studies well calibrated, and can a shared, reproducible pipeline with open code and data deliver reliable prediction intervals 1-6 months ahead?*

| Slot | Choice |
|---|---|
| Target | Predictive distribution of monthly district malaria incidence (cases per 1,000 person-months) 1-6 months ahead; evaluated by interval coverage and weighted interval score (WIS). |
| Predictors | CHIRPS rainfall lags, ERA5/MODIS temperature and NDVI, seasonality, surveillance lags, ITN/IRS coverage |
| Data sources | DHIS2/HMIS district surveillance, CHIRPS, ERA5, MODIS, WorldPop, Malaria Indicator Survey |
| Horizon | 1-6 months ahead (multiple horizons) |
| Spatial unit | District (admin-2) |
| Validation strategy | Temporal holdout on unseen future years plus leave-one-district-out, with calibration assessed by empirical interval coverage and WIS; all code and derived data released. This targets the code/data gap (only 13 of 86 papers state any sharing, 73 silent) and the uncertainty gap (37 of 86 papers do not state whether forecasts are probabilistic). |
| Baseline | Point-estimate-only model and seasonal-naive with empirical intervals. |
| Addresses | G2, G3 |

**Rests on:** H1 (supported): The spatial-holdout gap is not an artifact of abstract-only reading: among the 65 papers read in full, only 2 (3.1%) use a spatial or spatiotemporal holdout, so the absence persists at full-text depth.; H3 (supported): The external-validation gap is not a reporting artifact: validation_level is populated for 78% of papers, yet only 2 of 65 full-text papers report external/prospective/trial validation.

**Builds on:** [PMC11997905], [PMC12625025], [PMC7038892], [arXiv:2606.00834], [PMC11780933]

**Evidence supporting this direction:**

- [PMC11997905] Infectious disease forecasting to support public health: use of readily available methods to predict malaria and diarrhoeal diseases in Mozambique (2025): Uses WIS and coefficient of variation for probabilistic malaria forecasting in Mozambique, providing the exact scoring metric and a multiple-horizon design. From the paper: "we employed three models that represent these three model types: Exponential Smoothing"
- [PMC7038892] Dynamical Malaria Forecasts Are Skillful at Regional and Local Scales in Uganda up to 4 Months Ahead (2019): Uganda dynamical forecasts are skillful up to 4 months ahead with intervals and relative economic value, showing probabilistic multi-month district forecasts are achievable. From the paper: "the uncalibrated dynamical malaria model VECTRI"
- [PMC12625025] Malaria outbreak prediction at the sub-district level in Zambia using remote sensing satellite data (2025): Zambia sub-district model reports 95% prediction intervals with external validation, an example of interval reporting at district scale. From the paper: "A negative binomial regression model was then trained on 2010–2016 data to predict total cases in a malaria season"
- [arXiv:2606.00834] Hybrid Probabilistic Forecasting of Under-Five Malaria Admissions in Ghana: A Gaussian Process Regression with Holt-Winters Smoothing (2026): arXiv Ghana study produces probabilistic forecasts with a simple-regression baseline and states data are available on request, illustrating the arXiv-side practice the benchmark would test. From the paper: "This study proposes a hybrid framework integrating Gaussian Process Regression (GPR) with Holt-Winters exponential smoothing"

**Evidence challenging this direction:**

- [arXiv:2606.00783] Bayesian Inference of Nonlinear Malaria Dynamics in Ghana via an Ensemble Markov Chain Monte Carlo Sampler (2026): Authors state 'data are not publicly accessible' and cite a 'short annual time series', directly undermining a shared-code-and-data benchmark on this corpus. From the paper: "estimated via an affine-invariant ensemble Markov Chain Monte Carlo sampler"
- [PMC11983967] Unleashing the power of intelligence: revolutionizing malaria outbreak preparedness with an advanced warning system in Benin, West Africa (2025): Benin warning system reports point estimates only with no baseline comparison and no shared code/data, exemplifying the practice the design must overcome. From the paper: "an intelligent model for forecasting malaria outbreaks was developed using support vector machine (SVM) algorithm"
- [PMC13399501] Multi-scenario evaluation of federated learning for privacy-preserving malaria prediction with Ghana DHS data (2026): Highlights that retrospective evaluation with simulated partitioning overstates performance, so calibration measured retrospectively may not hold prospectively. From the paper: "comparing FedAvg [5] and FedProx [6] under three data distribution scenarios using Ghana DHS/MIS datasets"

**Risks:** Data-sharing restrictions (national health ministries, DHS terms) may prevent full open release.; Calibration on short series is unstable, especially for extreme quantiles.; arXiv and PMC papers use incompatible targets (incidence rate vs case count vs risk index), complicating pooling.

## Protocol

- **forecast_horizon** (enum: nowcast_or_none, short_1_to_3_months, medium_4_to_6_months, long_over_6_months, multiple_horizons_reported, not_stated). The longest lead time over which the model produces a prediction ahead of the target period, as stated in the methods or results. Use the maximum horizon evaluated; if the model is only fitted contemporaneously or retrospectively without a lead time, code as nowcast_or_none. Desirable: short_1_to_3_months, medium_4_to_6_months, multiple_horizons_reported.
- **spatial_unit** (enum: district_or_finer, region_or_province, national, multi_scale, not_stated). The finest spatial resolution at which predictions are made and evaluated. Code district_or_finer if predictions are for districts, counties, health-facility catchments or smaller; region_or_province if first-level administrative regions; national if only country totals; multi_scale if several resolutions are modelled. Desirable: district_or_finer.
- **target_variable** (enum: incidence_rate, case_count, outbreak_binary, prevalence, risk_index_or_suitability, not_stated). What quantity the model predicts. Code incidence_rate for cases per population per time; case_count for absolute counts; outbreak_binary for a binary outbreak/exceedance indicator; prevalence for parasite prevalence; risk_index_or_suitability for an environmental suitability or risk score without case units. Desirable: incidence_rate, case_count.
- **probabilistic_forecast** (enum: yes_interval, point_estimate_only, qualitative_uncertainty, not_stated). Whether the model outputs a distribution or interval for the prediction rather than only a point estimate. Code yes_interval if prediction/credible intervals or full posterior predictive distributions are reported and evaluated; point_estimate_only if only a single predicted value per target; qualitative_uncertainty if uncertainty is discussed but not quantified. Desirable: yes_interval.
- **validation_split** (enum: random_split, temporal_holdout, spatial_holdout, spatiotemporal_holdout, cross_validation_only, not_stated). How data were partitioned for model evaluation. Code random_split if observations were randomly assigned to train/test; temporal_holdout if later time periods were held out; spatial_holdout if entire districts/regions were held out; spatiotemporal_holdout if both unseen places and future times were held out; cross_validation_only if only k-fold CV without a held-out future or place; not_stated if the split is not described. Desirable: temporal_holdout, spatial_holdout, spatiotemporal_holdout.
- **intervention_covariates** (enum: none, itn_only, irs_only, both_itn_and_irs, other_interventions, not_stated). Whether malaria control interventions are included as model inputs or explicitly adjusted for. Code itn_only, irs_only, both_itn_and_irs, other_interventions (e.g. treatment coverage, larviciding, housing) or none, based on the variable list in the methods. Desirable: itn_only, irs_only, both_itn_and_irs, other_interventions.
- **baseline_comparison** (enum: none, seasonal_naive_or_persistence, simple_regression, both, not_stated). Whether the model's performance is compared against a simple reference. Code seasonal_naive_or_persistence if compared with a climatological/seasonal-naive or persistence baseline; simple_regression if compared with a plain GLM/linear or ARIMA baseline; both if both types; none if no baseline is reported. Desirable: seasonal_naive_or_persistence, simple_regression, both.
- **code_or_data_shared** (enum: code_and_data, code_only, data_only, on_request, none, not_stated). Whether the study makes its analysis code and/or the modelling dataset publicly available. Code code_and_data if both are shared via a repository or supplement; code_only or data_only if one is; on_request if the paper states availability upon request; none if neither is offered. Desirable: code_and_data, code_only, data_only.
- **model_family** (enum: statistical_timeseries, spatiotemporal_bayesian, tree_ensemble, deep_learning, hybrid, not_stated). The dominant modelling approach used for the reported prediction. Code statistical_timeseries for ARIMA/SARIMA/GLM/GAM/distributed-lag; spatiotemporal_bayesian for hierarchical/Bayesian spatio-temporal or INLA models; tree_ensemble for random forest/boosting/XGBoost; deep_learning for neural networks, LSTM, transformers; hybrid for explicit combinations of statistical and ML components. Desirable: hybrid.
- **reporting_delay_handling** (enum: yes_modelled, acknowledged_only, no, not_stated). Whether the study accounts for the lag between case occurrence and reporting/case confirmation in surveillance data. Code yes_modelled if delays are explicitly modelled or corrected (e.g. nowcasting, delay distributions); acknowledged_only if mentioned as a limitation without correction; no if not addressed. Desirable: yes_modelled.

**Inclusion:** Study develops, fits or evaluates a statistical or machine-learning model that predicts malaria risk, incidence, case counts or outbreak occurrence.; Study setting is one or more countries in sub-Saharan Africa, or a clearly delimited sub-national region within it.; Predictors include at least one of: climate/meteorological variables, satellite/remote-sensing variables, routine surveillance data, or malaria interventions (ITNs, IRS, treatment).; Prediction target is at an aggregated spatial unit (district, county, region, health-facility catchment) or a population-level time series, not an individual diagnosis.; Reports at least one quantitative performance or validation result (e.g. RMSE, MAE, AUC, R2, sensitivity/specificity, coverage of intervals).; Peer-reviewed article (PMC) or arXiv preprint, in English, with a retrievable abstract and methods description.

**Exclusion:** Setting outside sub-Saharan Africa (e.g. India, Colombia, Southeast Asia) with no SSA component.; Individual-level clinical prediction (diagnosis, severe-disease prognosis, drug or insecticide resistance prediction) rather than population malaria risk/incidence.; Entomological, vector-bionomics or parasite-genetics study with no malaria-risk prediction model fitted to epidemiological data.; Pure methodological or simulation paper with no empirical malaria data application.; Review, protocol, commentary or editorial without a new model or new validation result (retained only as background, not as a mapped study).; No quantitative performance metric reported, or model described only conceptually without fitting.

**How often each question-specific field is reported:** forecast_horizon 62%, spatial_unit 80%, target_variable 80%, probabilistic_forecast 57%, validation_split 38%, intervention_covariates 69%, baseline_comparison 13%, code_or_data_shared 15%, model_family 74%, reporting_delay_handling 15%

## Method

- Established: at least 30% of papers and 5 papers.
- Emerging: at least 3 papers and a share since 2022 at least 1.5x its earlier share.
- Missing: a desirable value in at most 2 papers or 5% of papers.
- Evidence strength and gap confidence are point scores; the reasons listed are the points awarded.
- Shares carry 95% Wilson intervals. Emerging rises use a one-sided Fisher exact test (recent vs earlier papers); novelty uses the hypergeometric probability of so little overlap.
- Values the protocol groups as one practice (for example spatial and spatiotemporal holdout) form one gap, so near-duplicates are not reported twice.
- Keyword searches honour quotes, OR, AND, NOT, -exclusions and parentheses.
- Extracted values without a verified quote from the paper were discarded before counting.

## References (50 cited of 86 analysed)

- **arXiv:2306.02685**: Predicting malaria dynamics in Burundi using deep Learning Models (Daxelle Sakubu et al., 2023). https://arxiv.org/abs/2306.02685
- **arXiv:2606.00834**: Hybrid Probabilistic Forecasting of Under-Five Malaria Admissions in Ghana: A Gaussian Process Regression with Holt-Winters Smoothing (T. Ansah-Narh et al., 2026). https://arxiv.org/abs/2606.00834
- **PMC10313820**: Spatio-temporal modelling of routine health facility data for malaria risk micro-stratification in mainland Tanzania (Sumaiyya G. Thawer et al., 2023). https://pmc.ncbi.nlm.nih.gov/articles/PMC10313820/
- **PMC10201255**: Mapping malaria incidence using routine health facility surveillance data in Uganda (Adrienne Epstein et al., 2023). https://pmc.ncbi.nlm.nih.gov/articles/PMC10201255/
- **PMC11227462**: GIS-based spatiotemporal mapping of malaria prevalence and exploration of environmental inequalities (Ropo Ebenezer Ogunsakin et al., 2024). https://pmc.ncbi.nlm.nih.gov/articles/PMC11227462/
- **PMC10707679**: Understanding the fine-scale heterogeneity and spatial drivers of malaria transmission in Kenya using model-based geostatistical methods (Donnie Mategula et al., 2023). https://pmc.ncbi.nlm.nih.gov/articles/PMC10707679/
- **PMC12676887**: From risk factors to predictive modelling: applying machine learning to childhood malaria surveillance in resource-limited settings (Joseph Opeolu Ashaolu et al., 2025). https://pmc.ncbi.nlm.nih.gov/articles/PMC12676887/
- **PMC11997905**: Infectious disease forecasting to support public health: use of readily available methods to predict malaria and diarrhoeal diseases in Mozambique (Rami Yaari et al., 2025). https://pmc.ncbi.nlm.nih.gov/articles/PMC11997905/
- **PMC6884483**: Malaria predictions based on seasonal climate forecasts in South Africa: A time series distributed lag nonlinear model (Yoonhee Kim et al., 2019). https://pmc.ncbi.nlm.nih.gov/articles/PMC6884483/
- **PMC8213140**: Predicting malaria epidemics in Burkina Faso with machine learning (David Harvey et al., 2021). https://pmc.ncbi.nlm.nih.gov/articles/PMC8213140/
- **arXiv:2510.01302**: Hybrid Predictive Modeling of Malaria Incidence in the Amhara Region, Ethiopia: Integrating Multi-Output Regression and Time-Series Forecasting (Kassahun Azezew et al., 2025). https://arxiv.org/abs/2510.01302
- **PMC10341430**: Generalized Linear Models to Forecast Malaria Incidence in Three Endemic Regions of Senegal (Ousmane Diao et al., 2023). https://pmc.ncbi.nlm.nih.gov/articles/PMC10341430/
- **PMC11983967**: Unleashing the power of intelligence: revolutionizing malaria outbreak preparedness with an advanced warning system in Benin, West Africa (Gouvidé Jean Gbaguidi et al., 2025). https://pmc.ncbi.nlm.nih.gov/articles/PMC11983967/
- **PMC5380319**: Geostatistical modelling of malaria indicator survey data to assess the effects of interventions on the geographical distribution of malaria prevalence in children less than 5 years in Uganda (Julius Ssempiira et al., 2017). https://pmc.ncbi.nlm.nih.gov/articles/PMC5380319/
- **PMC13382336**: Bayesian modelling of spatio-temporal dynamics for early-warning and control of severe malaria in Cameroon (Akindeh Mbuh Nji et al., 2026). https://pmc.ncbi.nlm.nih.gov/articles/PMC13382336/
- **PMC11780933**: Increasing the resolution of malaria early warning systems for use by local health actors (Michelle V. Evans et al., 2025). https://pmc.ncbi.nlm.nih.gov/articles/PMC11780933/
- **PMC11466501**: Forecasting malaria dynamics based on causal relations between control interventions, climatic factors, and disease incidence in western Kenya (Bryan O Nyawanda et al., 2024). https://pmc.ncbi.nlm.nih.gov/articles/PMC11466501/
- **PMC8941165**: Exploring predictive frameworks for malaria in Burundi (Lionel Divin Mfisimana et al., 2022). https://pmc.ncbi.nlm.nih.gov/articles/PMC8941165/
- **PMC11098333**: Predicting malaria outbreak in The Gambia using machine learning techniques (Ousman Khan et al., 2024). https://pmc.ncbi.nlm.nih.gov/articles/PMC11098333/
- **PMC12922243**: Forecasting malaria incidence in a resource-limited urban setting with climate variables as exogenous regressors: time series analysis using a SARIMAX model in Bahir Dar, Ethiopia (Tesfaye Taye Gelaw et al., 2026). https://pmc.ncbi.nlm.nih.gov/articles/PMC12922243/
- **PMC13563892**: Geospatial-based surveillance of malaria risk in Dar es Salaam using a hybrid 3DCNN + LSTM and CA model (Edmund Steven Kanjagaile et al., 2026). https://pmc.ncbi.nlm.nih.gov/articles/PMC13563892/
- **PMC4740093**: Forecasting paediatric malaria admissions on the Kenya Coast using rainfall (Stella Wanjugu Karuri et al., 2016). https://pmc.ncbi.nlm.nih.gov/articles/PMC4740093/
- **PMC10552258**: Specialist hybrid models with asymmetric training for malaria prevalence prediction (Thomas Fisher et al., 2023). https://pmc.ncbi.nlm.nih.gov/articles/PMC10552258/
- **PMC10558065**: Spatial Optimization Methods for Malaria Risk Mapping in Sub‐Saharan African Cities Using Demographic and Health Surveys (Camille Morlighem et al., 2023). https://pmc.ncbi.nlm.nih.gov/articles/PMC10558065/
- **PMC11978812**: Spatio-temporal modelling and prediction of malaria incidence in Mozambique using climatic indicators from 2001 to 2018 (Chaibo Jose Armando et al., 2025). https://pmc.ncbi.nlm.nih.gov/articles/PMC11978812/
- **PMC10914794**: Identifying childhood malaria hotspots and risk factors in a Nigerian city using geostatistical modelling approach (Taye Bayode et al., 2024). https://pmc.ncbi.nlm.nih.gov/articles/PMC10914794/
- **PMC12362852**: Integrating vulnerability and hazard in malaria risk mapping: the elimination context of Senegal (Camille Morlighem et al., 2025). https://pmc.ncbi.nlm.nih.gov/articles/PMC12362852/
- **PMC10824718**: Predicting malaria risk considering vector control interventions under climate change scenarios (Margaux L. Sadoine et al., 2024). https://pmc.ncbi.nlm.nih.gov/articles/PMC10824718/
- **PMC9453600**: Predicting malaria outbreaks from sea surface temperature variability up to 9 months ahead in Limpopo, South Africa, using machine learning (Patrick Martineau et al., 2022). https://pmc.ncbi.nlm.nih.gov/articles/PMC9453600/
- **arXiv:2606.00783**: Bayesian Inference of Nonlinear Malaria Dynamics in Ghana via an Ensemble Markov Chain Monte Carlo Sampler (T. Ansah-Narh et al., 2026). https://arxiv.org/abs/2606.00783
- **PMC10021332**: Assessing the effectiveness of malaria interventions at the regional level in Ghana using a mathematical modelling application (Timothy Awine et al., 2022). https://pmc.ncbi.nlm.nih.gov/articles/PMC10021332/
- **PMC10863265**: Towards an intelligent malaria outbreak warning model based intelligent malaria outbreak warning in the northern part of Benin, West Africa (Gouvidé Jean Gbaguidi et al., 2024). https://pmc.ncbi.nlm.nih.gov/articles/PMC10863265/
- **PMC10943795**: Application of advanced very high-resolution radiometer (AVHRR)-based vegetation health indices for modelling and predicting malaria in Northern Benin, West Africa (Gouvidé Jean Gbaguidi et al., 2024). https://pmc.ncbi.nlm.nih.gov/articles/PMC10943795/
- **PMC4470343**: Forecasting malaria in a highly endemic country using environmental and clinical predictors (Kate Zinszer et al., 2015). https://pmc.ncbi.nlm.nih.gov/articles/PMC4470343/
- **PMC10828885**: Exploring malaria prediction models in Togo: a time series forecasting by health district and target group (Anne Thomas et al., 2024). https://pmc.ncbi.nlm.nih.gov/articles/PMC10828885/
- **PMC11593955**: Impact of Climate Variability and Interventions on Malaria Incidence and Forecasting in Burkina Faso (Nafissatou Traoré et al., 2024). https://pmc.ncbi.nlm.nih.gov/articles/PMC11593955/
- **PMC12317539**: Bayesian spatio-temporal modeling and prediction of malaria cases in Tanzania mainland (2016-2023): unveiling associations with climate and intervention factors (Lembris Laanyuni Njotto et al., 2025). https://pmc.ncbi.nlm.nih.gov/articles/PMC12317539/
- **PMC13399501**: Multi-scenario evaluation of federated learning for privacy-preserving malaria prediction with Ghana DHS data (Daniel Kwasi Kovor et al., 2026). https://pmc.ncbi.nlm.nih.gov/articles/PMC13399501/
- **PMC7522256**: Data-driven malaria prevalence prediction in large densely populated urban holoendemic sub-Saharan West Africa (Biobele J. Brown et al., 2020). https://pmc.ncbi.nlm.nih.gov/articles/PMC7522256/
- **PMC12625025**: Malaria outbreak prediction at the sub-district level in Zambia using remote sensing satellite data (Matthew M. Ippolito et al., 2025). https://pmc.ncbi.nlm.nih.gov/articles/PMC12625025/
- **PMC7504016**: Predicting Malaria Transmission Dynamics in Dangassa, Mali: A Novel Approach Using Functional Generalized Additive Models (François Freddy Ateba et al., 2020). https://pmc.ncbi.nlm.nih.gov/articles/PMC7504016/
- **arXiv:2411.06436**: Predictors of disease outbreaks at continentalscale in the African region: Insights and predictions with geospatial artificial intelligence using earth observations and routine disease surveillance data (Scott Pezanowski et al., 2024). https://arxiv.org/abs/2411.06436
- **PMC4158077**: Development and validation of climate and ecosystem-based early malaria epidemic prediction models in East Africa (Andrew K Githeko et al., 2014). https://pmc.ncbi.nlm.nih.gov/articles/PMC4158077/
- **PMC3359352**: Spatially Explicit Burden Estimates of Malaria in Tanzania: Bayesian Geostatistical Modeling of the Malaria Indicator Survey Data (Laura Gosoniu et al., 2012). https://pmc.ncbi.nlm.nih.gov/articles/PMC3359352/
- **PMC3081772**: Epidemic malaria and warmer temperatures in recent decades in an East African highland (David Alonso et al., 2011). https://pmc.ncbi.nlm.nih.gov/articles/PMC3081772/
- **PMC6420752**: Characterizing local-scale heterogeneity of malaria risk: a case study in Bunkpurugu-Yunyoo district in northern Ghana (Punam Amratia et al., 2019). https://pmc.ncbi.nlm.nih.gov/articles/PMC6420752/
- **PMC5453969**: Using remote sensing environmental data to forecast malaria incidence at a rural district hospital in Western Kenya (Maquins Odhiambo Sewe et al., 2017). https://pmc.ncbi.nlm.nih.gov/articles/PMC5453969/
- **PMC5324298**: Integrating malaria surveillance with climate data for outbreak detection and forecasting: the EPIDEMIA system (Christopher L. Merkord et al., 2017). https://pmc.ncbi.nlm.nih.gov/articles/PMC5324298/
- **PMC13491694**: Modelling the impact of climate variability on malaria morbidity in the Tamale Metropolitan Area: a time series analysis (Abdul-Ganiu Zakaria et al., 2026). https://pmc.ncbi.nlm.nih.gov/articles/PMC13491694/
- **PMC7038892**: Dynamical Malaria Forecasts Are Skillful at Regional and Local Scales in Uganda up to 4 Months Ahead (Adrian M. Tompkins et al., 2019). https://pmc.ncbi.nlm.nih.gov/articles/PMC7038892/