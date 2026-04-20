# Your Task: Predict Patient Outcome from Molecular Data

You are given data for a single patient from a CAR T cell therapy cohort for Large B-Cell Lymphoma. **You do NOT know whether this patient responded (OR) or did not respond (NR) to therapy.** Your task is to analyze the patient's molecular and clinical data and predict the outcome.

## CRITICAL: Information Boundaries

**Do NOT read any of the following files or directories:**
- `data/ground_truth.json` (contains the true outcomes)
- Any files in `../agent_lbcl_bench_with_spatial/` (contains unblinded data)
- Any `results/` directories from other experiments
- Any files in parent directories (`../`)

**You may ONLY read:**
- The patient data directory specified in your prompt (`clinical.json`, `data_sources.json`, `infusion_features.csv`, `spatial_features.csv`)
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

3. **Generate initial hypotheses** about what this patient's data suggests regarding therapy outcome:
   - **If infusion data available**: Which cell populations are over/under-represented? What functional states dominate? Are there signs of T cell fitness or exhaustion?
   - **If spatial data available**: Which cell types are enriched/depleted in the TME? Are there spatial patterns suggesting immune suppression or activation?
   - Are there clinical risk factors (high LDH, high tumor burden)?

## Round 2: CellWhisperer Queries (if infusion data available)

4. **Design targeted queries** to test hypotheses about therapy-relevant biology. Use descriptive cell-type/state labels (see shared_context.md).

   Focus on features known to associate with CAR T outcomes:
   - T cell differentiation state (naive, stem-cell memory, central memory, effector, terminally differentiated)
   - Exhaustion markers (PD-1, TIM-3, LAG-3, TOX)
   - Cytokine production capacity (TNF, IFN-gamma, IL-2)
   - Proliferative capacity (Ki-67)
   - Metabolic fitness (glycolysis, oxidative phosphorylation)
   - Regulatory T cell contamination
   - Specific transcription factor programs (TCF7, EOMES, TBX21, NFATC2)

## Round 3: Spatial Hypothesis Testing (if spatial data available)

5. **Analyze TME composition** for features that predict CAR T efficacy:
   - Immunosuppressive cell types (Tregs, M2 macrophages, MDSCs)
   - Immune-permissive features (CD8 T cell infiltration, dendritic cells)
   - Spatial patterns: are effector cells excluded from tumor areas?
   - Are there proximity patterns suggesting active immune suppression?

## Round 4: Cross-Modal Integration (if both modalities available)

6. **Link infusion product quality to TME context**:
   - Would a strong infusion product be neutralized by this TME?
   - Does the TME appear permissive for CAR T cell activity?

## Round 5: Quantitative Validation

7. **Quantify your evidence.** For each pro-response or pro-resistance observation:
   - Compute where this patient sits relative to the cohort (z-score or percentile)
   - **High confidence**: <10th or >90th percentile AND |z| > 1.5
   - **Medium confidence**: <25th or >75th percentile
   - Below that, the observation is weak evidence

## Round 6: Prediction Synthesis

8. **Weigh all evidence and make a prediction.** Structure your reasoning:
   - List all pro-response evidence (features suggesting the patient will respond)
   - List all pro-resistance evidence (features suggesting the patient will not respond)
   - Weigh the evidence: which signals are strongest? Are they concordant or conflicting?
   - Make your prediction: **OR** (objective response) or **NR** (non-response)
   - State your confidence: **high**, **medium**, or **low**
   - Explain what would change your prediction (what additional data would be decisive)

## Output Requirements

Save analysis scripts to the patient's results directory (create `results/predictions/{patient_id}/` if needed). Do NOT write scripts to the experiment root directory.

Your FINAL message must contain a JSON block (fenced with ```json) with:

```json
{
  "patient_id": "01",
  "prediction": "OR",
  "confidence": "high",
  "reasoning_summary": "2-3 paragraph synthesis of why you predict this outcome",
  "pro_response_evidence": [
    {
      "feature": "Short description",
      "evidence": "Quantitative support (scores, z-scores, percentiles)",
      "data_source": "infusion|spatial|both",
      "strength": "high|medium|low"
    }
  ],
  "pro_resistance_evidence": [
    {
      "feature": "Short description",
      "evidence": "Quantitative support",
      "data_source": "infusion|spatial|both",
      "strength": "high|medium|low"
    }
  ],
  "key_deciding_factors": ["The 2-3 most important factors that drove the prediction"],
  "uncertainty_factors": ["Areas where data was ambiguous or insufficient"],
  "clinical_risk_assessment": {
    "ldh_status": "normal|elevated|unknown",
    "tumor_burden": "low|moderate|high|unknown",
    "therapy_type": "axicel|tisacel|bispecific"
  }
}
```

## Important Guidelines

- Write scripts, execute, iterate. Run at least 3 analysis scripts showing genuine investigation.
- Use `flush=True` on all print statements.
- **CellWhisperer is for infusion product ONLY** -- spatial features are pre-computed.
- **Tag every piece of evidence with `data_source`**: "infusion", "spatial", or "both".
- **Do NOT report "high strength" evidence unless quantile < 0.1 or > 0.9 AND |z-score| > 1.5.**
- Ground your reasoning in published CAR T cell biology.
- **Be honest about uncertainty.** If the data is ambiguous, say so. A well-reasoned "low confidence" prediction is more valuable than a poorly justified "high confidence" one.
