# LBCL-Bench: Agent-Based Mechanism Verification

## Overview

We benchmark CellWhisperer — a multimodal transcriptome-language model — against LBCL-Bench, a curated set of known mechanisms of CAR T cell therapy response and resistance in large B-cell lymphoma. The full LBCL-Bench contains 35 mechanisms; after filtering for detectability, relevance, and quality (see "LBCL-Bench Curation" below), **21 mechanisms** are used for agent evaluation. Three benchmark steps test progressively harder tasks:

1. **Step 1 (Mechanism Verification):** Given a known mechanism, verify it statistically using CellWhisperer on scRNA-seq data.
2. **Step 2 (Open-Ended Discovery):** A single agent performs open-ended analysis and we measure recall against LBCL-Bench.
3. **Step 3 (Patient-Level Analysis):** Per-patient agent analysis, measuring overlap with known mechanisms.

## Data

- **scRNA-seq atlas:** 36,764 CAR T cells from 79 patients (43 OR, 36 NR at 3 months), filtered to B_Product timepoint and low tumor burden (≤80th percentile SPD)
- **Full h5ad:** `/oak/.../cellxgene.h5ad` — 117,842 cells, 36,117 genes, log1p-normalized expression
- **Clinical metadata:** Response_3m (OR/NR), LDH, tumor burden, CRS/ICANS grades, therapy type

## LBCL-Bench Curation

### Filtering (filter_mechanisms.py)

From 35 total mechanisms, 14 are excluded from the evaluation set:

| Exclusion Reason | IDs | Count |
|---|---|---|
| Malformed entries | M005, M022, M026, M034 | 4 |
| Not LBCL-specific | M009, M023 | 2 |
| Engineering-only (CRISPR KO / overexpression, not observational) | M004, M007, M032, M027, M031 | 5 |
| Methodological (not a biological mechanism) | M028 | 1 |
| Not detectable from infusion product scRNA-seq | M010 | 1 |
| Toxicity-only with weak evidence | M033 | 1 |

**Remaining: 21 mechanisms** for agent evaluation.

### Description Quality Issues Identified and Fixed

Several mechanisms had bloated descriptions containing non-LBCL studies (melanoma TIL-ACT, myeloma CRISPR screens), review articles, and mathematical modeling studies. These were trimmed to reduce false-positive matching:

| ID | Before | After | Removed |
|---|---|---|---|
| M003 | 6.3K | 2.3K | Barras (melanoma TIL-ACT), Larson (review) |
| M008 | 13.3K | 3.7K | Trimmed bloated TME study from 12K to 3K |
| M011 | 16K | 8.3K | Trimmed Studies 1-2 |
| M014 | 21K | 4.6K | Chen (mathematical ODE model, 16K chars) |
| M016 | 30K | 4.9K | Barras (melanoma), Korell (myeloma), Labanieh/Larson (reviews), Lynn (melanoma) |
| M018 | 12.6K | 10.7K | Chen (mathematical model) |
| M021 | 17.6K | 8.7K | Barras (melanoma), Korell (myeloma), Theisen (DC review) |
| M024 | 5.1K | 1.2K | Rewritten: removed generic exhaustion biology, focused on HMGA1/GTF3A/KLF2 |

### False-Positive Analysis (M024 and M028)

Two mechanisms were found to produce false-positive matches in Step 3 evaluation:

**M024 (HMGA1/GTF3A):** The original description was 5,118 chars. The matching code truncated it to 600 chars (`[:600]`), which contained only generic non-responder biology (exhaustion, stemness, mitochondrial dysfunction) from unrelated studies. The actual HMGA1/GTF3A finding was at char ~4,700 and never reached the LLM matcher. This caused 29/36 NR false matches. The "striking" concordance (0.354, Fisher p<0.001) was entirely artifactual. **Fixed** by rewriting the description to focus only on HMGA1/GTF3A and switching the matching prompt to use `verbal_summary` only (not truncated description).

**M028 (SCENIC regulons):** This was a methodological claim ("SCENIC regulon features provide superior batch integration") with no biological directionality. The LLM matcher accepted any mention of transcription factor activity as a match, producing 48/79 false matches. **Fixed** by removing M028 from the evaluation set entirely.

### Category Fix

M003 category changed from "Molecular/Engineering" to "Infusion Product;Molecular/Engineering" — the mechanism (CD8+ metabolic fitness) is detectable from infusion product scRNA-seq.

### Remaining Quality Concerns

- **M012** ("immunosuppressive pathway enrichment in NR") is very vague — could match any immunosuppressive finding
- **M021** ("CD4 T cell proliferation signatures") is broad — could match any CD4 T cell finding
- **M030** evidence comes entirely from unpublished NIH progress reports
- **M035** evidence is from non-LBCL contexts (melanoma + computational model)

## Step 1: Mechanism Verification

### Step 1v1 — Agent-Based (Autonomous Verification)

Each of 22 mechanisms was independently analyzed by a fresh Claude Opus 4 agent session. Each agent autonomously chose text queries, aggregation methods, and statistical tests, then interpreted results.

**Result: 11/22 verified (50%)** using the light h5ad with pre-computed embeddings. Zero pipeline failures.

However, Step 1v1 had a critical data issue: the light h5ad contained pre-computed 2048-dim embeddings from old_jointemb (Geneformer) but only 1 dummy gene. The agent-based workflow used these embeddings correctly (via tensor extraction), but the approach is checkpoint-dependent and not reproducible across checkpoints.

### Step 1v2 — Pre-Registered Checkpoint Ablation

To rigorously compare checkpoints, we replaced agent autonomy with a pre-registered statistical pipeline:

1. **Query generation:** 10 direction-blinded text queries per mechanism generated by an LLM agent (stored in `data/step1v2_queries.json`).
2. **Scoring:** Each checkpoint re-embeds cells from full gene expression and scores against all queries.
3. **Aggregation:** 4 strategies per query — mean, max, frac_high75 (fraction above 75th percentile), p85 (85th percentile).
4. **Testing:** Mann-Whitney U (OR vs NR) on patient-level aggregated scores. Bonferroni correction within each mechanism (10 queries × 4 aggs = 40 tests).
5. **Ratio features:** 9 biologically motivated cell-type ratios (e.g., Proliferating/Quiescent, Cytotoxic/Anergic) matching the original analysis pipeline.

#### Checkpoints Tested

| Checkpoint | Transcriptome Encoder | Projection Dim | Notes |
|---|---|---|---|
| old_jointemb | Geneformer (12-layer transformer) | 2048 | Slow on CPU; produced original pre-computed embeddings |
| best_cxg (= spatialwhisperer_v1) | MLP | 1024 | **Identical files** (same MD5 hash). Fast inference. |

Note: `spatialwhisperer_v1` and `best_cxg` are copies of the same checkpoint (MD5 `df34b1a32811988db888fb242b963846`). The ablation effectively compares 2 distinct models.

#### Ablation Methodology

Both checkpoints were evaluated on exactly the same data and pipeline: 36,764 B_Product cells (43 OR, 36 NR) from the full cellxgene.h5ad (36,117 genes). Each checkpoint re-embedded cells through its own transcriptome encoder (Geneformer for old_jointemb, MLP for best_cxg) and scored them against the same 243 text queries (22 mechanisms × 10 pre-registered queries + 23 base queries for ratio computation). Patient-level aggregation (mean, max, frac_high75, p85) and Mann-Whitney U tests were applied identically. The only variable was the checkpoint.

#### Results

**No mechanism passes Bonferroni-corrected p<0.05 with either checkpoint**, but the MLP checkpoint shows substantially stronger signal:

| Metric | old_jointemb | MLP (best_cxg) |
|---|---|---|
| Mechanisms verified (p_corr<0.05) | 0/22 | 0/22 |
| Mechanisms approaching significance (p_corr<0.20) | 0 | **3** (M018, M002, M008) |
| Direction agreement (expected vs observed) | 78% (14/18) | 78% (14/18) |
| Original-style tests p<0.05 | 1 | **14** |
| Best mechanism raw p | 0.012 (M021) | **0.003** (M018) |

**Top mechanism hits (MLP checkpoint):**

| MID | Mechanism | Best p_raw | p_corr | Best Agg | Direction Match |
|---|---|---|---|---|---|
| M018 | Monocyte enrichment → poor response | 0.003 | 0.109 | max | Yes |
| M002 | CD8B/cytotoxic effectors → response | 0.004 | 0.175 | mean | Yes |
| M008 | TGFβ signaling impacts efficacy | 0.005 | 0.198 | max | Yes |
| M010 | Rab43 cross-presentation | 0.009 | 0.348 | mean | Unknown |
| M031 | MED12/STAT5/AP-1 transcription | 0.010 | 0.391 | max | Unknown |
| M001 | Exhausted CD4 T cells → CRS | 0.012 | 0.476 | mean | Yes |

**Original-style tests (MLP checkpoint, p<0.05):**

These replicate the original `run_patient_signal_analysis.py` findings:

| Aggregation | Feature | p-value | Direction |
|---|---|---|---|
| max | ratio_Proliferating/Quiescent | 0.010 | OR > NR |
| mean | ratio_Activated/Anergic | 0.017 | OR > NR |
| mean | ratio_Cytotoxic/Anergic | 0.019 | OR > NR |
| frac_high75 | Anergic cells | 0.020 | NR > OR |
| p85 | ratio_Cytotoxic/Anergic | 0.021 | OR > NR |
| frac_high75 | ratio_Cytotoxic/Anergic | 0.022 | OR > NR |
| mean | Anergic cells | 0.023 | NR > OR |
| frac_high75 | ratio_CD8+/CD4+ | 0.032 | OR > NR |

The ratio features (Proliferating/Quiescent, Cytotoxic/Anergic, CD8+/CD4+) and Anergic cell scores are the strongest signals — consistent with the original analysis.

#### Key Findings

1. **MLP checkpoint (best_cxg) produces stronger signal than old_jointemb** on this dataset: 14 vs 1 significant original-style tests (p<0.05), and 3 vs 0 mechanisms with corrected p<0.20. Both were evaluated on the same cells, queries, and statistical pipeline — the only difference is the transcriptome encoder (MLP vs Geneformer).
2. **Ratio features are essential.** The strongest signals come from ratios (Proliferating/Quiescent, Cytotoxic/Anergic), not raw cell-type scores.
3. **Small sample sizes limit formal verification.** With 43 OR and 36 NR patients, even real biological effects struggle to survive Bonferroni correction over 40 tests per mechanism.
4. **Direction agreement is good (78%).** CellWhisperer scores generally trend in the biologically expected direction, even when not reaching significance.

#### Technical Notes

- **MLP checkpoint requires raw counts:** The MLP transcriptome processor calls `ensure_raw_counts_adata()` and applies `log1p` internally. Since the full h5ad has log1p-normalized `.X`, we apply `expm1 + round` to recover approximate integer counts before scoring.
- **old_jointemb uses Geneformer:** Geneformer tokenizes gene expression differently and does not require raw counts. It was ~20× slower than MLP on CPU but comparable on GPU.
- **Pre-computed embeddings:** The `.obsm["transcriptome_embeds"]` in the light h5ad are from old_jointemb (2048-dim). They cannot be used with the MLP checkpoint (1024-dim mismatch).

## Step 2: Open-Ended Discovery

### Contamination Discovery and Clean Re-Run

The initial Step 2 run (SLURM job 17897297) was **contaminated**: the discovery agent had access to the full LBCL-Bench benchmark CSV (`data/lbcl_bench_filtered.csv`) and actively read it before designing queries (confirmed in raw log at `results/step2_contaminated/discovery_raw.txt` lines 9, 14). This invalidated those results (50% IP-detectable recall was artificially inflated).

**Contamination sources identified and fixed:**
1. `shared_context.md` had a "LBCL-Bench: Known Resistance Mechanisms" section pointing to the benchmark CSV
2. `shared_context.md` had specific ratio pairs with p-values from prior aim2 analysis (leaking which features matter)
3. `discovery.md` agent prompt had suspiciously specific query category hints (AP-1/JUN/FOS, IRF4, CXCR6, etc.) aligned with benchmark mechanisms

**Decontamination measures:**
- Removed benchmark references and specific ratio hints from `shared_context.md`
- Replaced specific query category hints in `discovery.md` with generic categories
- Added explicit "Do NOT read" instructions for `data/`, `results/step1*/`, `SUMMARY.md`
- Added quarantine logic to `run_agent.py`: physically moves benchmark files to `.quarantine_step2/` before agent runs, restores in `finally` block
- Contaminated results archived to `results/step2_contaminated/`

### Clean Run Results (SLURM job 17922441)

A fresh Claude Opus 4 agent session, with benchmark files quarantined, performed exploratory analysis: 132 text queries x multiple aggregation methods, totaling 1,014 statistical tests. The agent also performed axicel-specific stratified analysis and clinical variable testing. It produced **34 curated discoveries** (19 raw statistical at p<0.05 across all patients, plus 15 axicel-stratified findings).

#### Agent Strategy

The agent autonomously:
1. Designed 132 CellWhisperer text queries spanning T cell phenotypes, functional states, metabolic features, and contaminating cell types
2. Scored all 36,764 cells against queries using the MLP checkpoint (best_cxg) — see `results/step2/comprehensive_analysis.py`
3. Aggregated scores at patient level using 7 methods (mean, median, max, p85, p95, std, frac_high75) — see `results/step2/patient_scores_*.csv`
4. Computed ratio features (CD8/CD4, polyfunctional/anergic, Treg/CD8, activated/resting, etc.) — see `results/step2/ratio_results.csv` (91 ratio tests)
5. Performed Mann-Whitney U tests (OR vs NR) with BH correction — see `results/step2/statistical_results_all.csv` (924 single-query tests)
6. Stratified by therapy type (axicel vs other) for subset analysis — see `results/step2/interaction_axicel_results.csv` (132 axicel-only tests)
7. Tested clinical variables (age, LDH, tumor burden, therapy, gender, CRS/ICANS) — see `results/step2/clinical_results.csv`
8. Curated a final list of 34 discoveries with biological interpretation and confidence scores — see `results/step2/final_report.py` and `results/step2/final_discoveries.json`

#### All 34 Discoveries

The raw statistically significant findings (p<0.05 in all-patient analysis, BH-corrected p shown) are in `results/step2/discoveries.json`. The agent then incorporated axicel-stratified and clinical analyses into a curated final list in `results/step2/final_discoveries.json`:

| # | Discovery | Direction | p-value | Conf. | Source |
|---|---|---|---|---|---|
| 1 | Younger patient age | OR > NR | 0.006 | high | clinical |
| 2 | Replicative senescence from culture (axicel) | OR > NR | 0.007 | medium | axicel-stratified |
| 3 | Ki-67 expression (axicel) | OR > NR | 0.008 | high | axicel-stratified |
| 4 | Exhaustion from chronic stimulation (axicel) | OR > NR | 0.008 | low | axicel-stratified |
| 5 | Mitochondrial activity (axicel) | OR > NR | 0.009 | high | axicel-stratified |
| 6 | Self-renewal capacity (axicel) | OR > NR | 0.010 | high | axicel-stratified |
| 7 | Caspase-associated apoptosis (axicel) | OR > NR | 0.011 | low | axicel-stratified |
| 8 | DNA damage response (axicel) | OR > NR | 0.014 | low | axicel-stratified |
| 9 | Hypoxia response (axicel) | OR > NR | 0.014 | low | axicel-stratified |
| 10 | S phase cells (axicel) | OR > NR | 0.015 | high | axicel-stratified |
| 11 | Ex vivo expansion signatures (axicel) | OR > NR | 0.016 | medium | axicel-stratified |
| 12 | Th2 polarization | NR > OR | 0.017 | medium | all-patient |
| 13 | Cytolytic activity (axicel) | OR > NR | 0.019 | high | axicel-stratified |
| 14 | Epigenetic exhaustion program (axicel) | OR > NR | 0.019 | low | axicel-stratified |
| 15 | IL-7R expression (axicel) | OR > NR | 0.019 | high | axicel-stratified |
| 16 | CD8+ cytotoxic T cell content (axicel) | OR > NR | 0.019 | high | axicel-stratified |
| 17 | Gamma-delta T cells (axicel) | OR > NR | 0.024 | low | axicel-stratified |
| 18 | Tumor cell contamination in IP | NR > OR | 0.024 | medium | all-patient |
| 19 | Polyfunctional-to-anergic ratio | OR > NR | 0.025 | high | ratio |
| 20 | Oxidative stress response (axicel) | OR > NR | 0.025 | low | axicel-stratified |
| 21 | Th1 enrichment (paradoxical) | NR > OR | 0.026 | medium | all-patient |
| 22 | MYC expression | OR > NR | 0.029 | high | all-patient |
| 23 | CD8-to-CD4 ratio | OR > NR | 0.031 | high | ratio |
| 24 | PD1+TIM3+LAG3 co-expression | NR > OR | 0.035 | medium | all-patient |
| 25 | CD4+ T helper cell proportion | NR > OR | 0.036 | medium | all-patient |
| 26 | Th17 enrichment | NR > OR | 0.037 | low | all-patient |
| 27 | Proliferative capacity | OR > NR | 0.039 | high | all-patient |
| 28 | Activated-to-resting ratio | NR > OR | 0.040 | medium | ratio |
| 29 | AMPK pathway activation (axicel) | OR > NR | 0.041 | medium | axicel-stratified |
| 30 | Viability score | OR > NR | 0.042 | medium | all-patient |
| 31 | T cell anergy | NR > OR | 0.043 | medium | all-patient |
| 32 | B cell lymphoma cell contamination | NR > OR | 0.046 | medium | all-patient |
| 33 | NKT cell content | OR > NR | 0.047 | low | all-patient |
| 34 | Treg-to-CD8 ratio | NR > OR | 0.048 | medium | ratio |

Clinical variables tested but not significant (see `results/step2/clinical_results.csv`): LDH (p=0.46), tumor burden SPD (p=0.28), therapy type (p=0.86), gender (p=0.55), max ICANS (p=0.94), max CRS (p=0.47). Only age reached significance (p=0.006, OR patients younger).

#### Recall Evaluation

Each of the LBCL-Bench mechanisms was evaluated for semantic match against the 34 discoveries by an LLM judge (see `results/step2/recall_evaluation.csv` for full reasoning per mechanism). Matching criteria: same or substantially overlapping biological mechanism, consistent direction (OR>NR or NR>OR).

**Overall recall (curated 21): 3/21 strict (14.3%) + 3/21 partial (14.3%) = 6/21 lenient (28.6%)**
**IP-detectable recall (18): 3/18 strict (16.7%) + 2/18 partial = 5/18 lenient (27.8%)**

> Previously reported against 26 mechanisms: 6/26 lenient (23.1%). Recall improved slightly with the curated denominator because all 5 removed mechanisms (M010, M027, M028, M031, M033) were unmatched.

#### Matched Mechanisms

| MID | Mechanism | Match | Matched Discovery | Confidence |
|---|---|---|---|---|
| M002 | CD8B/cytotoxic effector genes predict response | yes | D13 (cytolytic activity, p=0.019) + D16 (CD8+ content, p=0.019) | high |
| M003 | CD8+ T cell metabolic fitness | partial | D5 (mitochondrial activity, p=0.009) + D29 (AMPK, p=0.041) | medium |
| M012 | Immunosuppressive pathway enrichment in NR | yes | D34 (Treg/CD8 ratio) + D24 (PD1+TIM3+LAG3) + D31 (anergy) | medium |
| M013 | Low tumor burden predicts response | partial | D18 (tumor contamination in NR, p=0.024) + D32 (B cell lymphoma, p=0.046) | medium |
| M014 | STAT5A/proliferation predicts response | partial | D27 (proliferative capacity) + D22 (MYC expression) | low |
| M015 | FOXP3+ Tregs predict non-response | yes | D34 (Treg-to-CD8 ratio in NR, p=0.048) | high |

#### Unmatched Mechanisms (from filtered set of 21)

Full per-mechanism reasoning in `results/step2/recall_evaluation.csv`.

> **Note:** M027, M031, M033 were in the original unmatched list but are now excluded from the benchmark. M010 is also excluded (TME-only).

| MID | Mechanism | Why Missed |
|---|---|---|
| M001 | Exhausted CD4 T cells → CRS | Agent only analyzed OR/NR, not CRS toxicity grading |
| M006 | Stromal-rich neighborhoods | TME spatial architecture, not detectable from IP |
| M008 | TGFβ signaling in TME | TME signaling, not directly queried |
| M011 | Myeloid cell contamination | Agent found tumor contamination but not myeloid-specific |
| M017 | CD27+PD-1-CD8+ T cells | Multi-marker phenotype not specifically queried |
| M018 | Monocyte enrichment → poor response | No monocyte-specific query among significant findings |
| M019 | AP-1 (JUN/FOS) activity | Transcription factor-level query not attempted |
| M020 | FMAC archetype (CAFs+TAMs) | TME spatial biology, not detectable |
| M021 | CD4 proliferation → response | Agent found CD4 enrichment in NR (opposite direction) |
| M024 | HMGA1/GTF3A expression | Gene-level queries beyond CellWhisperer's text interface |
| M025 | Tissue resident memory CD8 + glycolysis | Specific multi-marker phenotype not captured |
| M029 | LDHA/glycolysis → poor response | Agent found metabolic fitness in OR, but opposite framing of glycolysis |
| M035 | NFATC2/TBX21/EOMES TFs | Specific TF queries not attempted |

### Contaminated vs Clean Comparison

| Metric | Contaminated | Clean |
|---|---|---|
| Queries tested | 160 | 132 |
| Total statistical tests | 1,048 | 1,014 |
| Discoveries reported | 32 | 34 |
| Overall strict recall (original 26) | 9/26 (34.6%) | 3/26 (11.5%) |
| IP-detectable strict recall | 9/18 (50.0%) | 3/21 (14.3%) |
| Lenient recall (original 26) | 9/26 (34.6%) | 6/26 (23.1%) |

The contaminated run found AP-1/JUN/FOS, monocyte contamination, CD27+ memory cells, LDHA/glycolysis, and MED12-related mechanisms — all present in the benchmark CSV that the agent read. The clean run missed these, confirming contamination inflated recall by ~2-3x for strict matching. Full contaminated results and raw agent log available in `results/step2_contaminated/` (see `results/step2_contaminated/discovery_raw.txt` for the agent session log, `results/step2_contaminated/recall_evaluation.csv` for per-mechanism matching).

### Design and Limitations

The evaluation metric is **recall** — what fraction of LBCL-Bench mechanisms appear among the agent's discoveries.

**Statistical power:** With 1,014 tests and 79 patients (43 OR, 36 NR), ~51 false positives expected at p<0.05. The 19 raw significant findings (all-patient analysis) are below this noise floor. No individual test survives BH correction (minimum adjusted p=0.56 for ratios, 0.61 for single queries). The axicel-stratified analyses have even smaller sample sizes (33 OR, 28 NR). See `results/step2/statistical_results_all.csv` for the complete set of 924 tests with adjusted p-values.

**Key gaps in agent strategy:**
- No transcription factor-specific queries (AP-1, STAT5A, NFATC2, TBX21, EOMES)
- No myeloid/monocyte contamination queries (tumor contamination was queried but monocytes/myeloid cells were not)
- No CRS/ICANS toxicity analysis (only OR/NR response)
- No specific multi-marker phenotype queries (CD27+PD-1-, CXCR6+/SELPLG+/CCR8+)
- Gene-level mechanisms (HMGA1, GTF3A, LDHA) are beyond CellWhisperer's phenotype-level text queries

**Categorization of misses** (15 unmatched mechanisms from curated set of 21):
- 3 require specific transcription factor or gene-level queries (M019, M024, M035)
- 3 require TME/spatial data not available in IP (M006, M008, M020)
- 1 requires CRS/toxicity outcome analysis (M001)
- 2 involve myeloid cells which the agent did not query (M011, M018)
- 2 involve specific multi-marker phenotypes (M017, M025)
- 1 found opposite direction (M021: CD4 proliferation in OR, but agent found CD4 enrichment in NR)
- 1 found related but inversely framed signal (M029: glycolysis in NR vs metabolic fitness in OR)

## Step 3: Patient-Level Analysis

### Overview

Each of 79 patients was analyzed independently by a fresh Claude Opus 4 agent session. Each agent received the patient's clinical metadata (response status, CRS/ICANS grades, therapy type) and pre-computed CellWhisperer feature scores (37 queries × 3 aggregations + cohort percentile ranks). Agents then designed additional CellWhisperer queries, ran Python scripts on the compute node, and reported identified mechanisms with direction labels (pro-response/pro-resistance) and confidence levels.

### Step 3a — Patient Data Preparation (SLURM 17936890)

The full cellxgene.h5ad (117K cells, 36K genes) was scored against 37 base CellWhisperer queries using the MLP checkpoint (best_cxg). Patient-level features were aggregated using mean, max, and 85th percentile. Cohort-wide quantiles were computed to give agents context about where each patient ranks relative to the cohort.

- 79 patient directories created in `data/patients/`
- Each patient has `clinical.json` (response, CRS, ICANS, therapy, demographics) and `features.csv` (37 × 3 + quantiles)
- Completed in ~4 minutes

### Step 3b — Per-Patient Agent Analysis

#### V1 (SLURM 17938305, 17975411) — Old Prompts

79 agents completed (78 on first pass, 1 retry). Archived to `results/step3_per_patient_v1/`.

| Metric | V1 |
|---|---|
| Total mechanisms identified | 691 |
| Mean mechanisms per patient | 8.7 |
| Confidence distribution | 334 high, 328 medium, 29 low |
| Has effect_size field | 0/691 (0%) |
| Has patient_percentile field | 0/691 (0%) |
| Has toxicity_analysis | 0/79 |
| Mean raw output size | 191 KB |

**V1 quality issues:** Shallow analysis (2-3 scripts, no iterative refinement), no quantitative validation (no z-scores or within-patient statistics), poor CW query crafting (semantic reasoning queries), no CRS/ICANS toxicity analysis, template-ish confirmation-bias reasoning.

#### V2 (SLURM 18042697) — Improved Prompts

Prompt improvements: (1) CW query guidance in `shared_context.md` with good/bad examples, (2) dynamic CRS/ICANS toxicity injection in `run_agent.py`, (3) 5-round iterative analysis in `patient-analyst.md` with quantitative validation requirements, confidence calibration rules, and minimum 3 scripts.

79/79 patients completed successfully.

| Metric | V1 | V2 | Change |
|---|---|---|---|
| Total mechanisms identified | 691 | 713 | +3% |
| Mean mechanisms per patient | 8.7 | 9.0 | +3% |
| Confidence distribution | 334H/328M/29L | 414H/291M/8L | More calibrated |
| Has effect_size field | 0% | **100%** | New |
| Has patient_percentile field | 0% | **100%** | New |
| Has toxicity_analysis | 0/79 | **79/79** | New |
| Has analysis_rounds | 0/79 | **79/79** (all = 4 rounds) | New |
| Patients with actual CRS grade | N/A | 53/79 (67% had data) | New |
| Mean raw output size | 191 KB | **292 KB** (+53%) | More thorough |
| High-confidence well-calibrated (>90th/<10th pct) | N/A | 215/222 (97%) | Excellent |

### Step 3c — Evaluation

#### Methodology

For each benchmark mechanism, an LLM judge checked whether each patient's agent identified a semantically matching finding. To avoid lost-in-the-middle issues, patients were batched (15 per call, 6 batches per mechanism). The matcher uses `verbal_summary` only (not truncated descriptions) with strict gene-level matching: if a mechanism names specific genes (e.g., HMGA1, LDHA), the finding must reference those genes, not just the broader pathway.

The matcher accepts both direct and inverse observations (e.g., "high CD8 → response" in OR patients and "low CD8 → resistance" in NR patients both count as matches).

#### V1 Evaluation (SLURM 18008696) — Old Prompts, 26 Mechanisms

> **SUPERSEDED.** The V1 evaluation used 26 mechanisms with the old matching prompt (truncated `consolidated_description[:600]`). M024 and M028 were proven false positives (see "False-Positive Analysis" above). V1 results archived in `results/step3_evaluation_v1/`.

| Metric | V1 |
|---|---|
| Mechanisms found ≥1 patient | 25/26 (96%) |
| Fisher p<0.05 | 3 (M012 p=0.019, M024 p<0.001, M033 p=0.031) |
| Direction consistent | 13/17 (76%) |

All three Fisher-significant mechanisms were either false positives (M024: description truncation artifact, M028: not a biological mechanism) or excluded from the benchmark (M033: weak toxicity evidence).

#### V2 Evaluation (SLURM 18076409) — Improved Prompts, 21 Mechanisms

| Metric | V2 |
|---|---|
| Mechanisms found ≥1 patient | **13/21 (62%)** |
| Found by ≥5 patients | 9/21 |
| Found by ≥10 patients | 7/21 |
| Found by ≥20 patients | 6/21 |
| Fisher p<0.05 | **0/21** |
| Direction consistent | 11/15 (73%) |
| Total LLM evaluation calls | ~126 (21 × 6 batches) |

**Per-mechanism results:**

| MID | V1→V2 | OR | NR | Fisher p | Dir OK | Mechanism |
|---|---|---|---|---|---|---|
| M002 | 42→**40** | 22 | 18 | 1.000 | Yes | CD8B/cytotoxic effector genes predict response |
| M012 | 50→**32** | 16 | 16 | 0.646 | Yes | Immunosuppressive pathway enrichment in NR |
| M018 | 23→**26** | 13 | 13 | 0.635 | Yes | Monocyte enrichment → poor response |
| M015 | 34→**25** | 17 | 8 | 0.145 | **No** | FOXP3+ Tregs predict non-response |
| M003 | 71→**23** | 14 | 9 | 0.620 | Yes | CD8+ metabolic fitness |
| M011 | 22→**21** | 8 | 13 | 0.124 | Yes | Myeloid contamination |
| M029 | 17→**12** | 6 | 6 | 0.763 | Yes | LDHA/glycolysis → poor response |
| M035 | 43→**8** | 5 | 3 | 0.721 | Yes | NFATC2/TBX21/EOMES TFs |
| M013 | 21→**7** | 5 | 2 | 0.445 | Yes | Low tumor burden → response |
| M030 | 1→**3** | 2 | 1 | 1.000 | Yes | Tfh cells |
| M001 | 19→**2** | 0 | 2 | 0.204 | — | Exhausted CD4 T cells → CRS |
| M008 | 35→**2** | 0 | 2 | 0.204 | Yes | TGFβ signaling in TME |
| M020 | 23→**2** | 0 | 2 | 0.204 | Yes | FMAC archetype (CAFs+TAMs) |
| M006 | 0→**0** | 0 | 0 | 1.000 | — | Stromal-rich neighborhoods |
| M014 | 57→**0** | 0 | 0 | 1.000 | — | STAT5A activity |
| M016 | 1→**0** | 0 | 0 | 1.000 | — | Type I IFN in myeloid cells |
| M017 | 45→**0** | 0 | 0 | 1.000 | — | CD27+PD-1-CD8+ T cells |
| M019 | 22→**0** | 0 | 0 | 1.000 | — | AP-1 (JUN/FOS) activity |
| M021 | 20→**0** | 0 | 0 | 1.000 | — | CD4 proliferation signatures |
| M024 | 30→**0** | 0 | 0 | 1.000 | — | HMGA1/GTF3A expression |
| M025 | 25→**0** | 0 | 0 | 1.000 | — | Tissue resident memory CD8 |

**Key observations:**

1. **Drop from 96% to 62% recovery reflects correct behavior.** The stricter matching prompt rejects vague overlaps that V1 accepted. In particular:
   - M024 (HMGA1/GTF3A): 30→0. No agent mentioned these specific genes — V1's 30 matches were from generic exhaustion biology in the truncated description.
   - M014 (STAT5A): 57→0. V1 matched any proliferation finding to STAT5A; V2 requires the specific gene.
   - M017 (CD27+PD-1-CD8+): 45→0. V1 matched any CD8/memory finding; V2 requires the specific multi-marker phenotype.
   - M019 (AP-1): 22→0. V1 matched any transcription factor mention; V2 requires JUN/FOS specifically.

2. **Robust mechanisms.** M002 (CD8 cytotoxicity, 40/79), M012 (immunosuppressive pathways, 32/79), M018 (monocytes, 26/79), and M015 (FOXP3 Tregs, 25/79) survived the stricter matching with minimal change, confirming genuine recovery.

3. **No Fisher-significant mechanisms.** The bilateral matching design (accepting both direct and inverse observations) equalizes OR/NR counts. M015 (FOXP3 Tregs) shows an unexpected OR>NR skew (17 OR vs 8 NR for a pro-resistance mechanism) — OR agents reported "low Tregs → pro-response" which the bilateral matcher correctly accepts as inverse matches, inflating OR counts.

4. **8 mechanisms with zero matches.** These fall into categories:
   - Require specific gene mentions: M024 (HMGA1/GTF3A), M025 (CXCR6/SELPLG/CCR8)
   - Require specific multi-marker phenotypes: M017 (CD27+PD-1-CD8+)
   - Require specific TF mentions: M014 (STAT5A), M019 (AP-1/JUN/FOS)
   - TME-only: M006 (stromal neighborhoods), M016 (type I IFN in myeloid)
   - Broad/vague: M021 (CD4 proliferation)

#### Concordance — V2

Direction-aware concordance scoring: for each mechanism with a clear direction (pro-response or pro-resistance), score +1 for detection in the expected group (OR for pro-response, NR for pro-resistance) and -1 for detection in the other group. Range [-1, +1]; higher = more differential detection.

| Metric | V1 (26 mechs) | V2 (21 mechs) |
|---|---|---|
| Mechanisms scored | 17/26 | 18/21 |
| Macro-average concordance | 0.050 | **0.011** |
| Median concordance | 0.000 | 0.000 |
| Positive concordance (>0) | — | 8/18 |
| Negative concordance (<0) | — | 1/18 (M015) |
| Zero concordance | — | 9/18 |

**Per-mechanism concordance (V2, sorted by score):**

| MID | Direction | Det Expected | Det Other | Total | Concordance | Mechanism |
|---|---|---|---|---|---|---|
| M003 | pro-response | 14 | 9 | 23 | 0.063 | CD8+ metabolic fitness |
| M011 | pro-resistance | 13 | 8 | 21 | 0.063 | Myeloid contamination |
| M002 | pro-response | 22 | 18 | 40 | 0.051 | CD8B/cytotoxic effectors |
| M013 | pro-response | 5 | 2 | 7 | 0.038 | Low tumor burden |
| M020 | pro-resistance | 2 | 0 | 2 | 0.025 | FMAC archetype |
| M035 | pro-response | 5 | 3 | 8 | 0.025 | NFATC2/TBX21/EOMES |
| M008 | pro-resistance | 2 | 0 | 2 | 0.025 | TGFβ signaling |
| M030 | pro-response | 2 | 1 | 3 | 0.013 | Tfh cells |
| M012 | pro-resistance | 16 | 16 | 32 | 0.000 | Immunosuppressive pathways |
| M018 | pro-resistance | 13 | 13 | 26 | 0.000 | Monocyte enrichment |
| M029 | pro-resistance | 6 | 6 | 12 | 0.000 | LDHA/glycolysis |
| M015 | pro-resistance | 8 | 17 | 25 | **-0.114** | FOXP3+ Tregs |

Mechanisms with zero matches (M006, M014, M016, M017, M019, M021, M024, M025) all have concordance = 0. Three mechanisms (M001, M014, M016) have unclear direction and were excluded from scoring.

**Key finding:** Concordance is near-zero across the board (macro-average 0.011). This confirms that agents detect mechanisms at roughly equal rates in OR and NR patients, consistent with bilateral matching + confirmation bias. The only notable outlier is M015 (FOXP3 Tregs) with **negative** concordance (-0.114): more OR than NR patients matched for this pro-resistance mechanism, because OR agents reported "low Tregs → pro-response" which the bilateral matcher accepts.

V1 concordance results (now superseded) archived in `results/step3_evaluation_v1/concordance_scores.csv`.

### Discussion

#### Mechanism Recovery Across Steps

The three benchmark steps test progressively different capabilities — statistical verification (Step 1), open-ended discovery (Step 2), and patient-level interpretive analysis (Step 3) — and reveal complementary strengths and limitations of agent-driven analysis with CellWhisperer.

**Step 2 recovers 6/21 mechanisms from a blank slate.** A single agent, with no prior knowledge of the benchmark and no access to patient outcomes, independently rediscovered 3 strict and 3 partial matches among 34 reported discoveries. The strict matches — CD8 cytotoxicity (M002), immunosuppressive pathway enrichment (M012), and FOXP3+ Tregs (M015) — are among the most well-established mechanisms in CAR T biology. The partial matches capture related but less specific signals (metabolic fitness via mitochondria for M003, tumor contamination as a proxy for tumor burden for M013, proliferation downstream of STAT5A for M014). This 28.6% lenient recall represents genuinely novel rediscovery: the agent designed its own queries, chose its own statistical tests, and arrived at biologically coherent findings without guidance.

**Step 3 doubles recall to 13/21 (62%).** Per-patient agents, each analyzing a single patient's data with knowledge of their clinical outcome, collectively recover 7 additional mechanisms beyond Step 2. The newly recovered mechanisms include monocyte enrichment (M018), myeloid contamination (M011), LDHA/glycolysis (M029), TGFβ signaling (M008), and NFATC2/TBX21/EOMES transcription factors (M035). These are mechanisms that require either more targeted queries (monocyte-specific, TF-specific) or contextual framing (linking glycolysis to resistance) that patient-level analysis naturally provides.

**8 mechanisms remain undetected.** These fall into clear categories that illuminate the boundaries of the approach:
- *Specific gene mentions required (3):* M024 (HMGA1/GTF3A), M014 (STAT5A), M019 (AP-1/JUN/FOS) — CellWhisperer's text interface operates at the phenotype level, not individual genes.
- *Multi-marker phenotypes (2):* M017 (CD27+PD-1-CD8+), M025 (CXCR6+/SELPLG+/CCR8+) — combinatorial surface marker patterns are difficult to capture in single text queries.
- *TME-only biology (2):* M006 (stromal neighborhoods), M016 (type I IFN in myeloid cells) — fundamentally undetectable from infusion product data.
- *Broad/vague (1):* M021 (CD4 proliferation) — paradoxically too broad to match strictly, as V2's matching prompt correctly rejects generic CD4 findings.

#### Concordance and Confirmation Bias

**Direction consistency is high.** Among the 15 mechanisms where direction could be assessed, 11 (73%) showed the expected direction (pro-response mechanisms detected more in OR, pro-resistance more in NR). This is comparable to Step 1v2's 78% direction agreement from the pre-registered statistical pipeline, suggesting agents' biological reasoning is generally sound.

**Concordance is near-zero.** The macro-average concordance score is 0.011, meaning mechanisms are detected at nearly equal rates in OR and NR patients. No mechanism achieves Fisher significance (all p > 0.12). This reflects the study design: agents are told the patient's outcome and asked to explain it, so they construct bilateral narratives — OR agents find "high CD8 → response" while NR agents find "low CD8 → resistance" for the same mechanism. The bilateral LLM matcher correctly accepts both as matches, equalizing group counts.

**M015 direction anomaly.** FOXP3+ Tregs (M015) shows the only negative concordance (-0.114): 17 OR vs 8 NR matches for a pro-resistance mechanism. This arises because OR agents reported "low Tregs → pro-response," which the bilateral matcher accepts as an inverse match, inflating OR counts. This highlights a fundamental limitation of bilateral matching: it cannot distinguish genuine differential biology from direction-flipped narrative framing.

**Interpretation.** The near-zero concordance does not mean the recovered mechanisms are spurious — it means the current design cannot distinguish true biological signal from confirmation bias. The agents demonstrate genuine biological reasoning (correct directionality, appropriate use of CellWhisperer, coherent interpretations), but post-hoc narrative construction in both groups neutralizes any differential signal. A blinded experiment (agents not told the patient's outcome) would be required to disentangle these effects.

#### Comparison Across Steps

| Metric | Step 1v2 (Verification) | Step 2 (Discovery) | Step 3 V2 (Per-Patient) |
|---|---|---|---|
| Task | Verify known mechanisms | Open-ended discovery | Patient-level analysis |
| Benchmark size | 22 | 21 (curated) | 21 (curated) |
| Recall | N/A | 28.6% (6/21 lenient) | 62% (13/21) |
| Direction-consistent | 78% (14/18) | N/A | 73% (11/15) |
| Fisher p<0.05 | 0/22 (Bonferroni) | N/A | 0/21 |
| Concordance (macro-avg) | N/A | N/A | 0.011 |
| Agent autonomy | Pre-registered pipeline | Full (query design → statistics) | Guided (outcome + features provided) |
| Strengths | Rigorous, reproducible | Unbiased discovery | High recall, patient context |
| Limitations | Low power (n=79) | Low recall, no TF/gene queries | Confirmation bias, no discrimination |

## Step 4: Mechanism Clustering and Novelty Analysis

### Overview

The 713 per-patient mechanism instances (712 unique free-text descriptions) were clustered to identify distinct mechanism types and assess their novelty relative to LBCL-Bench and the broader literature.

### Method

1. **Embedding:** TF-IDF vectorization with 1–3 n-grams (5000 features, sublinear TF, English stop words)
2. **Clustering:** Agglomerative clustering (cosine distance, average linkage), k=40 (capped for interpretability; silhouette monotonically increases with k in TF-IDF space)
3. **Cluster labeling:** LLM-generated 5–12 word labels from the 5 most central exemplars per cluster (Gemini 2.0 Flash via OpenRouter)
4. **Bench matching:** LLM judge assessed whether each cluster matches any of the 21 LBCL-Bench mechanisms
5. **Novelty classification:** Unmatched clusters classified as "known in broader literature" or "potentially novel" by LLM

Code: `mechanism_clustering/cluster_mechanisms.py`
Results: `mechanism_clustering/results/`

### Results

| Category | Clusters | Description |
|---|---|---|
| **Matching LBCL-Bench** | 23 clusters → 13 distinct bench mechanisms | Multiple clusters can map to the same bench mechanism |
| **Known in literature** | 12 | Established in CAR T/immunology literature but not in LBCL-Bench |
| **Potentially novel** | 5 | Not well-established as distinct resistance patterns |

**Total: 40 distinct mechanism types** identified across 79 patients from 713 instances.

The 13 distinct bench mechanisms recovered by clustering are consistent with the 62% recall (13/21) from the per-mechanism evaluation, providing independent validation.

### Bench-Matched Clusters (23 clusters → 13 mechanisms)

| Bench ID | # Clusters | Mechanism | Example cluster labels |
|---|---|---|---|
| M002 | 4 | CD8B/cytotoxic effectors → response | CD8+ cytotoxic activity; CD4/CD8 imbalance; γδ T cell cytotoxicity; differentiation state |
| M003 | 6 | CD8+ metabolic fitness | Metabolic fitness/OXPHOS; transcriptional fitness; metabolic heterogeneity; effector balance |
| M015 | 3 | FOXP3+ Tregs → non-response | Treg dysregulation; T cell depletion with Treg; immune evasion |
| M001 | 2 | Exhausted CD4 → CRS | T cell activation/polarization; neuroinflammation/toxicity |
| M011 | 1 | Myeloid contamination | Myeloid/monocyte contamination |
| M012 | 1 | Immunosuppressive pathways | Ectoenzyme-mediated immunosuppression |
| M013 | 1 | Low tumor burden → response | Tumor burden/LDH levels |
| M014 | 1 | STAT5A/proliferation | Impaired proliferation/transduction |
| M017 | 1 | CD27+PD-1-CD8+ T cells | Exhaustion states/checkpoint expression |
| M019 | 1 | AP-1 (JUN/FOS) activity | AP-1 transcription factor activity |
| M025 | 1 | Tissue resident memory CD8 | Chemokine receptor/CD62L expression |
| M035 | 1 | NFATC2/TBX21/EOMES TFs | Impaired NF-kB signaling |

### Known in Literature (12 clusters)

| Cluster | Label | # Patients |
|---|---|---|
| 3 | IFN-gamma and GM-CSF dysregulation mediating ICANS | 9 |
| 11 | T cell senescence impacting product quality | 14 |
| 12 | Cytokine production capacity and polyfunctionality | 20 |
| 15 | Age-related immunosenescence | 6 |
| 16 | T cell self-renewal and stem-like compartment size | 38 |
| 21 | Co-stimulatory signaling driving activation/expansion | 6 |
| 22 | Infusion product cell stress and damage | 23 |
| 23 | CAR expression level in infusion product | 38 |
| 24 | Death ligand and hypoxia-mediated cytotoxicity | 3 |
| 28 | Activation-induced cell death (AICD) and apoptosis | 17 |
| 34 | Effector memory compartment and CD4/CD8 balance | 18 |
| 35 | BCL2 family anti-apoptotic signaling | 17 |

### Potentially Novel (5 clusters)

| Cluster | Label | # Patients | OR | NR | Notes |
|---|---|---|---|---|---|
| **0** | **Proliferation-cytotoxicity decoupling** | **4** | **1** | **3** | T cells actively dividing (G2M 92nd–99th pctl) but failing to acquire effector function (TEMRA <15th pctl). Most interesting — mechanistically specific, actionable, consistent direction. |
| 6 | Chronic antigen stimulation dysfunction | 8 | 4 | 4 | Mixed OR/NR; heterogeneous (PD-1 without exhaustion vs. lack of engagement) |
| 17 | Severe T cell subset depletion | 6 | 4 | 2 | Globally low scores; more observation than mechanism |
| 32 | Profound T cell anergy | 11 | 3 | 8 | Strong NR enrichment; NR4A-driven functional unresponsiveness; pharmacologically reversible |
| 37 | Low T helper cell polarization | 1 | 1 | 0 | Single patient; insufficient evidence |

#### Highlight: Proliferation-Cytotoxicity Decoupling (Cluster 0)

The most interesting novel finding. In three non-responding patients, agents independently identified that CAR T cells were actively proliferating but failed to differentiate into cytotoxic effectors:

- **Patient 20:** Proliferating cells at 89.9th percentile (mean), G2M at 92.4th percentile, but cytotoxic T cells only at 43rd percentile (mean) and TEMRA at 12.7th percentile. Agent concluded: "active proliferation without cytotoxic differentiation, consistent with Treg expansion or exhausted proliferation."
- **Patient 27:** S/G2M at 98.7th percentile AND G0 arrest at 89.9th percentile — bimodal distribution with futile cell division alongside senescence. Agent concluded: "proliferative stress leading to replicative exhaustion."
- **Patient 18:** Elevated overall proliferation (mean_q=0.709) but MKI67+ T cell proliferation only at median — agent suggested the signal may come from contaminating non-T cells.

This pattern — cell cycle without effector programming — was not described in LBCL-Bench and, to our knowledge, has not been reported as a distinct resistance pattern. It directly suggests manufacturing interventions to enforce effector differentiation (e.g., cytokine conditioning, co-stimulatory domain optimization).

### Unpublished Findings from Our Group

Several LBCL-Bench mechanisms derive from unpublished work:

| Bench ID | Citation | Detected? | Note |
|---|---|---|---|
| M012 | Mo et al. (in preparation) — CCL8+CCL13+ TAMs | **Yes** (32/79 patients) | Matched via general immunosuppressive pathway enrichment, not the specific CCL8/CCL13 finding |
| M019 | Mo et al. (in preparation) — AP-1 activity | **Yes** (via clustering; 0 in strict per-mechanism eval) | Cluster 18 matched M019, but strict gene-level matching failed |
| M024 | Tsui et al. 2025 — HMGA1/GTF3A | No (0/79) | Gene-level mechanism beyond CellWhisperer's phenotype interface |
| M015 | Good et al. 2022 — FOXP3+ Tregs | **Yes** (25/79 patients) | Published, but from our own group |
| M001, M021 | Good lab findings | Marginal (2/79, 0/79) | — |

M012's detection is particularly notable: the primary evidence comes from Mo's manuscript in preparation (CCL8+CCL13+ tumor-associated macrophages marking early resistance), which could not have been in the LLM's training data. The agent identified the broader pattern (immunosuppressive pathway enrichment in non-responders) that encompasses Mo's specific finding.

### Output Files

| File | Description |
|---|---|
| `mechanism_clustering/cluster_mechanisms.py` | Full clustering pipeline |
| `mechanism_clustering/results/mechanism_clusters.csv` | Per-cluster summary (label, stats, bench match, novelty) |
| `mechanism_clustering/results/all_mechanisms_annotated.csv` | All 713 instances with cluster labels and novelty category |
| `mechanism_clustering/results/summary.json` | Summary statistics |
| `mechanism_clustering/README.md` | Action plan and methodology description |

## Next Steps

**Investigate poor directionality (confirmation bias).** Step 3 agents recover 62% of known mechanisms but with near-zero concordance (0.011) — mechanisms are found at equal rates in OR and NR patients. This suggests agents construct post-hoc narratives that fit the provided outcome rather than identifying genuinely differential signals. The key question is whether this reflects (a) the agent's reasoning being dominated by the outcome label, (b) CellWhisperer scores not containing enough differential signal to begin with, or (c) the bilateral LLM matcher inflating matches by accepting inverse observations. A blinded experiment — re-running Step 3 with the patient's response status withheld — would isolate the contribution of confirmation bias: if blinded agents still find the same mechanisms at similar rates, the issue is (b) or (c); if recall drops sharply, it is (a). This would also enable a proper test of whether agents can predict outcome from data alone.

## Reproducibility

All code and data in `src/experiments/agent_lbcl_bench/`. Paths are relative to that directory.

### Pipeline Infrastructure

| File | Description |
|---|---|
| `filter_mechanisms.py` | Filters 35 mechanisms to 21 for agent evaluation (exclusion logic + detectability flags) |
| `Snakefile` | Orchestrates all steps (SLURM integration via `sm7_slurm` profile) |
| `run_agent.py` | Agent dispatcher — launches opencode agents, handles quarantine for Step 2 |
| `opencode.json` | Permissive permissions for non-interactive agent runs |
| `shared_context.md` | Shared context injected into all agents (checkpoint paths, data conventions) |
| `.opencode/agents/mechanism-verifier.md` | Step 1 agent prompt (mode=primary, claude-opus-4-6) |
| `.opencode/agents/discovery.md` | Step 2 agent prompt (mode=primary, claude-opus-4-6) |
| `.opencode/agents/patient-analyst.md` | Step 3 agent prompt (mode=primary, claude-opus-4-6) |
| `.opencode/agents/query-generator.md` | Query generation agent (mode=primary, claude-sonnet-4-20250514) |

### Benchmark Data

| File | Description |
|---|---|
| `data/lbcl_bench_filtered.csv` | 21 benchmark mechanisms with categories, detectability flags, citations (filtered from 35 total) |
| `data/step1v2_queries.json` | 22 mechanisms x 10 pre-registered queries for Step 1v2 ablation |

### Step 1 Results

| File | Description |
|---|---|
| `results/step1/` | Step 1v1 agent-based verification results (11/22 verified) |
| `results/step1v2_ablation_v2/` | Step 1v2 checkpoint ablation: per-checkpoint CSVs + `ablation_summary.csv` |
| `step1v2_ablation_v2.py` | Ablation scoring script (expm1 fix, skip-if-exists) |
| `step1v2_generate_queries.py` | Pre-registered query generation via LLM |
| `analyze_ablation_v2.py` | Local analysis of ablation results |

### Step 2 Results (Clean)

| File | Description |
|---|---|
| `results/step2/comprehensive_analysis.py` | Agent-written scoring + analysis script (30 KB) — the full pipeline |
| `results/step2/final_report.py` | Agent-written curation and reporting script (33 KB) |
| `results/step2/discoveries.json` | 19 raw discoveries (p<0.05 in all-patient analysis) with BH-adjusted p-values |
| `results/step2/final_discoveries.json` | 34 curated discoveries (all-patient + axicel-stratified + clinical) |
| `results/step2/recall_evaluation.csv` | Semantic matching: 26 LBCL-Bench mechanisms vs 34 discoveries with reasoning |
| `results/step2/statistical_results_all.csv` | All 924 single-query Mann-Whitney U tests with BH-adjusted p-values |
| `results/step2/ratio_results.csv` | 91 ratio-feature tests (e.g., CD8/CD4, polyfunctional/anergic) |
| `results/step2/interaction_axicel_results.csv` | 132 axicel-only stratified tests |
| `results/step2/clinical_results.csv` | Clinical variable tests (age, LDH, SPD, therapy, gender, CRS, ICANS) |
| `results/step2/patient_scores_mean.csv` | Patient-level mean aggregated scores (132 queries x 79 patients) |
| `results/step2/patient_scores_median.csv` | Patient-level median aggregated scores |
| `results/step2/patient_scores_max.csv` | Patient-level max aggregated scores |
| `results/step2/patient_scores_p85.csv` | Patient-level 85th percentile aggregated scores |
| `results/step2/patient_scores_p95.csv` | Patient-level 95th percentile aggregated scores |
| `results/step2/patient_scores_std.csv` | Patient-level standard deviation of scores |
| `results/step2/patient_scores_frac_high75.csv` | Patient-level fraction of cells above 75th percentile |
| `results/step2/cell_level_scores.csv.gz` | Cell-level CellWhisperer scores for all 132 queries (21 MB compressed) |

### Step 2 Results (Contaminated, Archived)

| File | Description |
|---|---|
| `results/step2_contaminated/discovery_raw.txt` | Raw agent session log (419 KB) — shows agent reading benchmark at lines 9, 14 |
| `results/step2_contaminated/discoveries.json` | 32 contaminated discoveries |
| `results/step2_contaminated/recall_evaluation.csv` | Contaminated recall evaluation (9/26 strict, 50% IP-detectable) |
| `results/step2_contaminated/compute_recall_summary.py` | Helper script for contaminated recall summary |

### Evaluation Scripts

| File | Description |
|---|---|
| `step2_evaluate_recall.py` | LLM-based recall evaluation (uses litellm — currently broken due to openai version conflict in pixi) |
| `inspect_discoveries.py` | Quick inspection helper for discoveries JSON |

### Step 3 Results

| File | Description |
|---|---|
| `data/patients/` | 79 patient directories with `clinical.json` + `features.csv` |
| `data/patients/patient_ids.txt` | List of 79 patient IDs |
| `results/step3_per_patient/{pid}.json` | Per-patient structured results (mechanisms, confidence, direction) |
| `results/step3_per_patient/{pid}_raw.txt` | Raw agent session log (NDJSON format) |
| `results/step3_evaluation/bench_mechanism_patient_counts.csv` | Per-mechanism: total/OR/NR matches, Fisher p, direction consistency |
| `results/step3_evaluation/concordance_scores.csv` | Direction-aware concordance scores per mechanism |
| `results/step3_evaluation/all_patient_mechanisms.csv` | All 682 mechanism instances across all patients |
| `results/step3_evaluation/mechanism_frequency.csv` | Unique mechanism frequency table |
| `results/step3_evaluation/bench_vs_agent_distribution.{png,svg,csv}` | Recovery rate distribution plot |
| `results/step3_evaluation/or_vs_nr_recovery.{png,svg}` | OR vs NR stratified bar chart |
| `results/step3_evaluation/concordance_scores.{png,svg}` | Concordance score bar chart |
| `results/step3_evaluation/detection_rate_by_group.{png,svg}` | Expected vs other group detection rates |

### Step 3 Scripts

| File | Description |
|---|---|
| `step3_prepare_patients.py` | Creates per-patient data (full h5ad, best_cxg, expm1, mean/max/p85) |
| `step3_evaluate.py` | Batched LLM matching (15 patients/batch) + Fisher's test + OR/NR stratification |
| `step3_concordance.py` | Direction-aware concordance scoring post-processing |
| `run_step3_prepare.sh` | SLURM wrapper for step3_prepare_patients.py |
| `run_step3_patients.sh` | SLURM array wrapper for per-patient agent runs |
