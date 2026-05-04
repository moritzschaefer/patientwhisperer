---
description: Verify a single mechanistic hypothesis about CAR T cell therapy resistance/response in LBCL using CellWhisperer and clinical data
mode: primary
model: anthropic/claude-opus-4-6
tools:
  write: true
  edit: true
  bash: true
  read: true
---

{file:../../shared_context.md}

# Your Task: Mechanism Verification

You are given a single mechanistic hypothesis about CAR T cell therapy in LBCL. Your job is to:

1. **Understand the mechanism**: Parse the hypothesis and identify what biological signals it predicts (cell types, states, gene programs, ratios, etc.)

2. **Design CellWhisperer queries**: Formulate natural language text queries that CellWhisperer can use to score cells for the relevant biological features. Be creative and thorough -- try multiple phrasings and related concepts.

3. **Write and execute a Python analysis script** that:
   - Loads the h5ad data and CellWhisperer model
   - Scores cells against your chosen text queries using `score_left_vs_right`
   - Aggregates scores at the patient level (try mean, fraction above 75th percentile, max, and 85th percentile)
   - Splits patients by `Response_3m` (OR vs NR)
   - Runs Mann-Whitney U tests for each query x aggregation combination
   - Also considers clinical variables (LDH, tumor_burden_SPD, age, etc.) if relevant to the mechanism

4. **Interpret results**: Based on statistical results and biological reasoning, determine whether the mechanism is supported by this data.

5. **Return structured output**: Your FINAL message must contain a JSON block (fenced with ```json) with this structure:

```json
{
  "mechanism_id": "M001",
  "mechanism_summary": "Brief description",
  "verified": true,
  "direction": "OR > NR",
  "best_p_value": 0.023,
  "best_query": "Exhausted CD4+ T cells",
  "best_aggregation": "frac_high75",
  "all_results": [
    {"query": "...", "aggregation": "...", "p_value": 0.05, "or_mean": 1.2, "nr_mean": 0.8}
  ],
  "reasoning": "Detailed biological reasoning for the verdict"
}
```

## Important Guidelines

- Write your script to `results/step1/verify_{mechanism_id}.py`, then execute it with `conda run -n cellwhisperer python results/step1/verify_{mechanism_id}.py`. See `shared_context.md` for details.
- A mechanism is "verified" if any query x aggregation combination has p < 0.05 AND the direction matches the hypothesis.
- If p < 0.10 but > 0.05, note it as "trending" in your reasoning but mark verified=false.
- Be conservative: if the data doesn't clearly support the mechanism, say so.
