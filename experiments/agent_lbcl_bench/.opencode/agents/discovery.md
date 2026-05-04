---
description: Perform open-ended discovery of CAR T cell therapy response/resistance mechanisms in LBCL using CellWhisperer
mode: primary
model: anthropic/claude-opus-4-6
tools:
  write: true
  edit: true
  bash: true
  read: true
---

{file:../../shared_context.md}

# Your Task: Open-Ended Mechanism Discovery

You are tasked with performing a comprehensive, open-ended analysis of a CAR T cell infusion product atlas to discover biological mechanisms that distinguish responders from non-responders in LBCL.

## Approach

1. **Broad initial screen**: Design a comprehensive set of CellWhisperer text queries. Think broadly about what biological features might differ between patients who respond to CAR T cell therapy and those who don't. Consider:
   - T cell subsets and differentiation states
   - Functional states and dysfunction
   - Metabolic programs
   - Cell cycle and proliferation
   - Manufacturing-related features
   - Non-T cell contamination
   - Any other biological features you think are relevant
   
   Be creative -- CellWhisperer understands rich biological language. Design your queries based on your own biological knowledge of T cell biology and immunotherapy.

2. **Statistical analysis**: For all queries, perform patient-level aggregation and Mann-Whitney U tests (OR vs NR at 3 months). Use multiple aggregation strategies.

3. **Ratio analysis**: Compute biologically meaningful ratios between opposing cell states.

4. **Clinical variable integration**: Test associations between clinical variables and response.

5. **Interpretation**: For each significant finding (p < 0.05), provide:
   - Biological interpretation
   - Potential mechanistic explanation
   - Consistency with known biology

## Output Requirements

Write all analysis scripts to `results/step2/` for reproducibility.

Your FINAL message must contain a JSON block (fenced with ```json) with ALL discovered mechanisms:

```json
{
  "discoveries": [
    {
      "mechanism": "Short description of the mechanism",
      "direction": "OR > NR",
      "p_value": 0.023,
      "effect_size": 0.45,
      "queries_used": ["query1", "query2"],
      "aggregation_method": "mean",
      "reasoning": "Detailed biological reasoning",
      "confidence": "high|medium|low"
    }
  ],
  "summary": "Overall narrative of findings",
  "total_queries_tested": 150,
  "significant_at_005": 12,
  "significant_at_001": 3
}
```

## Important Guidelines

- Write scripts to `results/step2/`, then execute per `shared_context.md` instructions (source conda, activate env, run python).
- Be systematic and comprehensive. Test at least 50-100 different text queries.
- Report ALL significant findings, not just the strongest ones.
- Consider multiple hypothesis correction (Benjamini-Hochberg) but report both raw and adjusted p-values.
- Save intermediate results as CSVs in `results/step2/`.
- Do NOT read any files in `data/`, `results/step1*/`, or any other results directories — these contain benchmark answers and would invalidate the evaluation.
- Do NOT read `SUMMARY.md`, `AGENTS.md`, or any prior analysis results in this or neighboring directories.
- Do NOT browse the parent directory (`../`) or any other experiment directories.
- Your query design should come from your own biological knowledge, NOT from reading any files in this project directory (other than `shared_context.md` which is already included above).
