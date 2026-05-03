# Your Task: Patient-Level Mechanism Analysis (Multi-Modal)

You are given data for a single patient from the CAR T cell therapy cohort. Your job is to perform an in-depth, patient-centric analysis to identify mechanistic explanations for this patient's clinical outcome. This patient may have one or both data modalities: infusion product (CellWhisperer scores) and/or spatial TME (CosMx proportions + proximities).

## CRITICAL: Information Boundaries

**Do NOT read any of the following files or directories:**
- `data/` (contains benchmark ground truth)
- `results/step1*/`, `results/step2*/` (contains prior analysis results)
- `SUMMARY.md` (contains benchmark evaluation results)
- Any files in parent directories (`../`)

**You may ONLY read:**
- The patient data directory specified in your prompt (`clinical.json`, `data_sources.json`, `infusion_features.csv`, `spatial_features.csv`)
- `shared_context.md` (already injected in system prompt)
- Scripts you write yourself

## Analysis Protocol

Follow these four phases strictly and in order. Do NOT skip ahead. Each phase has a specific objective; do not mix objectives across phases.

### Phase 1: Quantitative Profiling

**Objective:** Observe and report. Do NOT interpret, hypothesize, or explain.

1. Read `data_sources.json` to determine available modalities.
2. Read `clinical.json` and all available feature files (`infusion_features.csv`, `spatial_features.csv`).
3. For each data source, produce a structured profile:
   - **Infusion product** (if available): List all features with extreme quantiles (<0.10 or >0.90). For each, report the feature name, score_mean, quantile_mean, and z-score relative to cohort statistics in the CSV. Also compute and report ratios between opposing cell states (e.g., Proliferating/Quiescent, Cytotoxic/Anergic, Effector/Exhausted, CD8/CD4) using score_mean values. Additionally, you may examine key marker genes directly from the h5ad (see "Direct gene expression analysis" in shared_context.md) and report per-patient expression quantiles for genes relevant to the patient's profile.
   - **Spatial TME** (if available): List all cell type proportions and proximity scores with extreme quantiles (<0.10 or >0.90). Report the value, quantile, and which cell types are unusually enriched, depleted, or co-localized.
   - **Clinical**: Report age, gender, therapy, LDH, tumor burden, CRS/ICANS grades. Flag any unusual values.
4. Output: A structured table of observations. No narrative. No causal claims. No "this suggests" or "this indicates."

### Phase 2: Hypothesis Generation

**Objective:** Propose candidate mechanisms grounded in Phase 1 observations and published CAR T biology.

1. Review your Phase 1 profile. For each extreme observation (or combination of observations), propose a mechanistic hypothesis. Each hypothesis must:
   - State a specific biological mechanism (e.g., "T cell exhaustion driven by chronic antigen stimulation")
   - Cite the Phase 1 observation(s) that motivate it (feature names, values, quantiles)
   - State the expected direction: does this mechanism favor response (pro-response) or resistance (pro-resistance)?

2. Generate 5-10 candidate hypotheses. Prioritize hypotheses supported by multiple concordant observations over those resting on a single feature.

3. **Go beyond pre-computed features.** The pre-computed `infusion_features.csv` covers ~37 cell-state queries. If your hypothesis involves a cell state or phenotype not covered (e.g., a specific transcription factor program, a metabolic state, a rare subset), note it as requiring live CellWhisperer scoring in Phase 3.

4. Do NOT filter or reject hypotheses yet. That is Phase 3's job.

### Phase 3: Falsification

**Objective:** Test each hypothesis against the data. Actively search for counter-evidence. Use live CellWhisperer scoring to probe signals not covered by pre-computed features.

For EACH hypothesis from Phase 2, do the following:

1. **State a testable prediction.** What additional data pattern would you expect to see if this mechanism is real? What pattern would contradict it?

2. **Search for counter-evidence.** Examine the patient's data for observations that CONTRADICT the hypothesis:
   - If the hypothesis claims exhaustion drives failure: are there features showing high effector function, high cytotoxicity, or low inhibitory receptor expression?
   - If the hypothesis claims TME suppression: are there spatial features showing immune infiltration, low Treg proximity, or low myeloid content?
   - If the hypothesis invokes a ratio (e.g., CD8/CD4): check whether both numerator and denominator are individually extreme, or just one.

3. **Probe with gene expression and/or CellWhisperer.** Use direct gene expression for specific marker checks (e.g., verifying GZMB/PRF1 levels for a cytotoxicity hypothesis) and CellWhisperer for higher-level cell-state queries not covered by pre-computed features. See "Analyzing Infusion Product Data" in `shared_context.md` for both approaches. Follow this workflow:
   - Write a Python script that loads the full cohort h5ad (all 79 patients), analyzes ALL patients, and aggregates per patient.
   - Compute this patient's **quantile rank** relative to the full cohort distribution.
   - Compare OR vs NR distributions (Mann-Whitney U test).
   - **You MUST analyze the entire cohort, not just this patient.** A score is meaningless without cohort context.
   - Combine multiple queries/genes in a single script to amortize loading cost.

4. **Check quantitative thresholds.** A mechanism is only credible if the supporting features are genuinely extreme:
   - **High confidence**: quantile <0.10 or >0.90 AND |z-score| > 1.5
   - **Medium confidence**: quantile <0.25 or >0.75
   - Below that: insufficient evidence.

5. **Verdict.** For each hypothesis, assign one of:
   - **Survived**: Prediction confirmed, no counter-evidence found, supporting features meet quantitative thresholds.
   - **Weakened**: Some supporting evidence but counter-evidence exists or thresholds not met. State specifically what weakens it.
   - **Rejected**: Counter-evidence outweighs supporting evidence, or key features are not extreme. State the falsifying observation.

### Phase 4: Synthesis

**Objective:** Report only surviving and weakened mechanisms. Construct a coherent mechanistic narrative.

1. Drop all rejected hypotheses. Do not mention them in the output.
2. For surviving and weakened mechanisms, integrate across modalities:
   - **If both modalities available**: Does the infusion product quality align with or contradict the TME context? Mechanisms supported by both modalities are stronger.
   - Tag each mechanism with `data_source`: "infusion", "spatial", or "both".
3. Write a 2-3 paragraph narrative grounded in the surviving mechanisms. Each claim must reference specific features and quantiles.

## Output Requirements

Save analysis scripts to the patient's data directory.
**Save your final JSON output to `final_results.json` in the patient's data directory.** This file is used by the evaluation pipeline — always use this exact filename.

Your FINAL message must contain a JSON block (fenced with ```json) with:

```json
{
  "patient_id": "PAT01",
  "response": "OR",
  "clinical_summary": "Brief clinical profile",
  "phase1_profile": {
    "n_extreme_infusion": 5,
    "n_extreme_spatial": 3,
    "n_extreme_clinical": 1,
    "key_ratios": {"Proliferating_Quiescent": 2.1, "CD8_CD4": 0.8}
  },
  "mechanisms_identified": [
    {
      "mechanism": "Short description",
      "data_source": "infusion|spatial|both",
      "evidence": "Phase 1 observations supporting this — include specific scores, z-scores, and percentiles",
      "counter_evidence": "What counter-evidence was searched for and not found (or found but insufficient)",
      "falsification_verdict": "survived|weakened",
      "confidence": "high|medium",
      "direction": "pro-response|pro-resistance|neutral",
      "effect_size": "z-score or Cohen's d relative to cohort",
      "patient_percentile": "quantile rank in cohort"
    }
  ],
  "rejected_hypotheses": [
    {
      "hypothesis": "What was proposed",
      "falsifying_observation": "What contradicted it"
    }
  ],
  "toxicity_analysis": {
    "crs_grade": 2,
    "icans_grade": 0,
    "crs_mechanisms": ["mechanisms explaining CRS grade"],
    "icans_mechanisms": ["mechanisms explaining ICANS grade"]
  },
  "narrative": "Integrated mechanistic explanation (2-3 paragraphs)",
  "unusual_features": ["feature1 at 95th percentile (z=2.1)", "feature2 at 5th percentile (z=-1.8)"],
  "analysis_phases": 4,
  "live_cellwhisperer_queries": ["Novel queries scored via live CellWhisperer (if any)"],
  "suggested_follow_up": ["Additional analyses that could strengthen these conclusions"]
}
```

## Important Guidelines

- You may write and execute Python scripts. Run with: `cd /sailhome/moritzs/cellwhisperer_public && pixi run python /path/to/script.py`
- Use `flush=True` on all print statements.
- You have access to cellwhisperer, anndata, scanpy, pandas, numpy, scipy, torch. Use the pre-computed CSV/JSON files for quick analysis, and live CellWhisperer scoring (see shared_context.md) for novel queries beyond the pre-computed feature set.
- Focus on what makes THIS patient unique compared to the cohort.
- **CellWhisperer is for infusion product ONLY** — spatial features are pre-computed as proportions and proximities. Do NOT try to score spatial data with CellWhisperer.
- **Tag every mechanism with `data_source`**: "infusion", "spatial", or "both".
- **Do NOT report "high confidence" mechanisms unless quantile < 0.1 or > 0.9 AND |z-score| > 1.5.**
- Ground your reasoning in published CAR T cell biology.
- **Phase discipline is critical.** If you catch yourself interpreting during Phase 1 or skipping falsification in Phase 3, stop and redo the phase correctly.
