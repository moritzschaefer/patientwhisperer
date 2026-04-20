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

## Step 0: Check Available Modalities

**First**, read `data_sources.json` to know what data is available:
```json
{"has_infusion": true/false, "has_spatial": true/false, "n_spatial_cells": N}
```

Adapt your analysis based on available modalities.

## Round 1: Profile & Hypothesize

1. **Review all available data**: Read clinical variables and ALL available feature files.

2. **Contextualize within cohort**: Features include quantile rankings relative to the cohort. Identify unusual features (quantiles < 0.1 or > 0.9).

3. **Generate initial hypotheses** based on the patient's profile:
   - **If infusion data available**: Which cell populations are over/under-represented? What functional states dominate?
   - **If spatial data available**: Which cell types are enriched/depleted in the TME? Are there unusual proximity patterns suggesting cell-cell interactions?
   - Are there unusual clinical features?

## Round 2: Infusion Product Deep Dive (if infusion data available)

4. **Analyze the pre-computed CellWhisperer scores** in `infusion_features.csv` to test your hypotheses. Each row is a cell-type/state query already scored across this patient's infusion product cells.

   Focus on:
   - Features with extreme quantiles (<0.1 or >0.9) indicating unusual enrichment/depletion
   - Ratios and contrasts between opposing cell states (e.g., effector vs. exhausted)
   - Consistency across aggregation levels (mean vs. max vs. p85)
   - Computing z-scores relative to cohort statistics provided in the CSV

## Round 3: Spatial Hypothesis Testing (if spatial data available)

5. **Analyze TME composition**:
   - Which cell types are enriched or depleted compared to the cohort?
   - Which proximity scores are extreme — suggesting unusual spatial relationships?
   - Are there known immunosuppressive patterns (e.g., high Treg + high proximity to CD8 T cells)?
   - Are there patterns of immune exclusion vs. infiltration?

## Round 4: Cross-Modal Integration (if both modalities available)

6. **Link infusion product quality to TME context**:
   - Does the TME explain why a good/poor infusion product translated (or didn't) to response?
   - Are there concordant signals across modalities?
   - Are there discordant signals that suggest one modality dominates the outcome?

## Round 5: Quantitative Validation

7. **Quantify your claims with statistics.** For each mechanism:
   - Compute where this patient sits relative to the cohort (z-score or percentile)
   - Compare to OR and NR group means
   - **High confidence**: <10th or >90th percentile AND z > ±1.5
   - **Medium confidence**: >75th or <25th percentile
   - Below that, the finding is not meaningful

## Round 6: Synthesis

8. **Synthesize findings** into a coherent mechanistic narrative. Ground reasoning in published CAR T cell biology.

## Output Requirements

Save analysis scripts to the patient's results directory.

Your FINAL message must contain a JSON block (fenced with ```json) with:

```json
{
  "patient_id": "PAT01",
  "response": "OR",
  "clinical_summary": "Brief clinical profile",
  "mechanisms_identified": [
    {
      "mechanism": "Short description",
      "data_source": "infusion|spatial|both",
      "evidence": "What data supports this — include specific scores, z-scores, and percentiles",
      "confidence": "high|medium|low",
      "direction": "pro-response|pro-resistance|neutral",
      "effect_size": "z-score or Cohen's d relative to cohort",
      "patient_percentile": "quantile rank in cohort"
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
  "analysis_rounds": 6,
  "suggested_follow_up": ["Additional analyses that could strengthen these conclusions"]
}
```

## Important Guidelines

- You may write and execute small Python scripts (stdlib only) to compute z-scores, parse CSVs, etc. Run with `python3 <script.py>`.
- Use `flush=True` on all print statements.
- **Do NOT attempt to import cellwhisperer, anndata, or other scientific packages.** Use only the pre-computed CSV/JSON files.
- Focus on what makes THIS patient unique compared to the cohort.
- **CellWhisperer is for infusion product ONLY** — spatial features are pre-computed as proportions and proximities. Do NOT try to score spatial data with CellWhisperer.
- **Tag every mechanism with `data_source`**: "infusion", "spatial", or "both".
- **Do NOT report "high confidence" mechanisms unless quantile < 0.1 or > 0.9 AND |z-score| > 1.5.**
- Ground your reasoning in published CAR T cell biology.
