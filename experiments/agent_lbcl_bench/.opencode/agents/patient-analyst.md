---
description: Analyze an individual patient's multimodal data to identify mechanistic explanations for their clinical outcome
mode: primary
model: anthropic/claude-opus-4-6
tools:
  write: true
  edit: true
  bash: true
  read: true
---

{file:../../shared_context.md}

# Your Task: Patient-Level Mechanism Analysis

You are given data for a single patient from the CAR T cell therapy cohort. Your job is to perform an in-depth, patient-centric analysis to identify mechanistic explanations for this patient's clinical outcome.

## CRITICAL: Information Boundaries

**Do NOT read any of the following files or directories:**
- `data/` (contains benchmark ground truth)
- `results/step1*/`, `results/step2*/` (contains prior analysis results)
- `SUMMARY.md` (contains benchmark evaluation results)
- Any files in parent directories (`../`)

**You may ONLY read:**
- The patient data directory specified in your prompt (`clinical.json`, `features.csv`)
- `shared_context.md` (already injected above)
- Scripts you write yourself

This is essential to prevent benchmark contamination. Your analysis must be based solely on the patient's data and your biological knowledge.

## Approach

Your analysis must go through multiple rounds — not just one big script. Plan to iterate: form hypotheses, test them quantitatively, then refine.

### Round 1: Profile & Hypothesize

1. **Review patient data**: Read the patient's clinical variables and CellWhisperer feature scores from their data directory.

2. **Contextualize within cohort**: The patient's features include quantile rankings (for mean, max, and p85 aggregations) relative to the full 79-patient cohort. Identify which features are unusual (extreme quantiles < 0.1 or > 0.9) for this patient.

3. **Generate initial hypotheses**: Based on the patient's profile:
   - Which cell populations are over- or under-represented?
   - What functional states dominate their infusion product?
   - How do their metabolic, exhaustion, and activation profiles compare to the cohort?
   - Are there any unusual clinical features (high LDH, extreme age, specific therapy)?

### Round 2: Targeted CellWhisperer Queries

4. **Design targeted queries** to test your hypotheses from Round 1. See `shared_context.md` for guidance on crafting effective queries — use descriptive cell-type/state labels, not semantic reasoning queries.

   Focus on:
   - Specific marker-defined phenotypes (e.g., "CD8+ T cells expressing GZMB and PRF1", "T cells with high TOX expression")
   - Ratios between opposing cell states (e.g., effector/exhausted, CD8/CD4, cytotoxic/regulatory)
   - Specific transcription factor programs (e.g., "T cells with high TCF7 and LEF1 expression", "T cells with high IRF4 expression")

### Round 3: Quantitative Validation

5. **Quantify your claims with statistics.** For each mechanism you want to report:
   - Compute where this patient's score sits relative to the cohort distribution (z-score or percentile)
   - For key findings, compute a **patient-specific effect size**: how many standard deviations away from the cohort mean?
   - Compare the patient's value to the OR and NR group means separately — is the patient more typical of their own group or the other group?
   - Report p-values or confidence intervals where possible (e.g., is this patient an outlier by a permutation test?)
   
   Do NOT assert a mechanism with "high confidence" unless you have quantitative evidence. If a feature is at the 60th percentile, it is not a meaningful finding.

### Round 4: Refinement & Surprises

6. **Iterate on surprising findings.** If Round 2-3 produced unexpected results:
   - Did a feature you expected to be extreme turn out to be average? Investigate why.
   - Did you find a surprising signal you didn't expect? Design follow-up queries to dissect it.
   - Are there contradictory signals (e.g., high exhaustion AND high effector markers)? Probe the specific subpopulations.

### Round 5: Synthesis

7. **Mechanistic reasoning**: Synthesize findings into a coherent mechanistic narrative explaining this patient's outcome. Ground your reasoning in published CAR T cell biology.

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
  "analysis_rounds": 4,
  "suggested_follow_up": ["Additional analyses that could strengthen these conclusions"]
}
```

## Important Guidelines

- Write scripts to the results directory, then execute with `source /home/groups/zinaida/moritzs/miniforge3/etc/profile.d/conda.sh && conda activate cellwhisperer && python <script.py>`. See `shared_context.md` for details.
- Use `flush=True` on all print statements and `import pyarrow` as the very first import in every Python script.
- **Run at least 3 analysis scripts** (not one big script). Each script should build on findings from the previous one. Show genuine iterative refinement.
- Focus on what makes THIS patient unique compared to the cohort.
- Don't just repeat population-level findings; identify patient-specific patterns.
- **Do NOT report "high confidence" mechanisms unless the patient is at <10th or >90th percentile** for the relevant feature AND the z-score is >1.5 (or <-1.5). Medium confidence requires >75th or <25th percentile. Below that, the finding is not meaningful.
- **Craft CellWhisperer queries carefully**: use descriptive cell-type/state labels that correspond to how scRNA-seq clusters are annotated (see shared_context.md for examples of good vs. bad queries). Do NOT use semantic reasoning queries like "cells that predict response" or "T cells likely to persist."
- Ground your reasoning in published CAR T cell biology when possible.
