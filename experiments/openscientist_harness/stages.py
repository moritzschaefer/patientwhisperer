"""
Per-patient analysis stages and prompt construction for the OpenScientist harness.

Each stage guides the agent through a distinct phase of the discovery loop:
data profiling, hypothesis generation with literature grounding, and falsification.
"""

DEFAULT_STAGES = ["data_profiling", "hypothesize_literature", "falsification"]

_STAGE_LABELS = {
    "data_profiling": "Data Profiling",
    "hypothesize_literature": "Hypothesis Generation & Literature Search",
    "falsification": "Falsification & Evidence Testing",
}


def build_stage_prompt(
    stage_name: str,
    stage_index: int,
    total_stages: int,
    ks,
    research_question: str,
    data_files: list,
) -> str:
    """Return the full prompt string for the given analysis stage.

    Parameters
    ----------
    stage_name : str
        One of ``DEFAULT_STAGES``.
    stage_index : int
        Zero-based index of this stage.
    total_stages : int
        Total number of stages in the pipeline.
    ks : KnowledgeState
        Knowledge state object (has ``.get_summary()`` and ``.data`` dict).
    research_question : str
        The per-patient research question.
    data_files : list
        List of data file paths/names available in the job directory.

    Returns
    -------
    str

    Raises
    ------
    ValueError
        If *stage_name* is not recognised.
    """
    if stage_name not in _STAGE_LABELS:
        raise ValueError(
            f"Unknown stage name {stage_name!r}. "
            f"Valid stages: {list(_STAGE_LABELS)}"
        )

    label = _STAGE_LABELS[stage_name]
    header = f"# Stage {stage_index + 1}/{total_stages}: {label}\n\n"
    footer = (
        "\n\n**REQUIRED:** Call `set_status` at the start. "
        "Call `save_iteration_summary` as your FINAL action."
    )

    body = _BUILDERS[stage_name](ks, research_question, data_files)
    return header + body + footer


# ------------------------------------------------------------------
# Stage-specific prompt builders
# ------------------------------------------------------------------

def _build_data_profiling(ks, research_question: str, data_files: list) -> str:
    files_block = "\n".join(f"- `{f}`" for f in data_files)
    return f"""\
## Research Question
{research_question}

## Available Data Files
{files_block}

## Instructions

1. Read ALL patient CSV/JSON files in the job directory.
2. For each CellWhisperer feature, note: **score_mean**, **quantile_mean**, **score_p85**, **quantile_p85**.
3. Identify features with extreme quantiles (quantile_mean < 0.10 or quantile_mean > 0.90).
4. Compute the following key ratios using **score_mean** values:
   - CD8/CD4
   - Effector/Exhausted
   - Proliferating/Quiescent
   - Cytotoxic/Anergic
   - Memory/Naive
   - Oxidative/Glycolytic
5. Record the structured profile as a finding via `update_knowledge_state` with:
   - **title**: "Quantitative Profile"
   - **evidence**: all extreme features (with quantile values) and computed ratios
6. If spatial data exists, also profile cell-type proportions and proximity scores.
7. Call `save_iteration_summary` as your final action.

**CRITICAL**: Do NOT interpret findings. Do NOT generate hypotheses. Only observe and record."""


def _build_hypothesize_literature(ks, research_question: str, data_files: list) -> str:
    summary = ks.get_summary()
    return f"""\
## Research Question
{research_question}

## Current Knowledge State
{summary}

## Instructions

1. Review the quantitative profile from stage 1 (above).
2. Search PubMed for CAR T resistance mechanisms relevant to the observed extremes.
   Ground each candidate mechanism in published literature.
3. Propose 5-10 candidate mechanisms using `add_hypothesis` for each.
   - Each hypothesis **must** cite specific features from the profile (with quantile values).
   - Each hypothesis **must** state the expected direction (pro-response or pro-resistance).
4. Do NOT test hypotheses yet -- only propose them.
5. Call `save_iteration_summary` as your final action."""


def _build_falsification(ks, research_question: str, data_files: list) -> str:
    summary = ks.get_summary()
    return f"""\
## Research Question
{research_question}

## Current Knowledge State
{summary}

## Instructions

1. Review all pending hypotheses from stage 2 (listed above).
2. For EACH hypothesis:
   a. State a testable prediction.
   b. Search for counter-evidence in the patient's data.
   c. Assign a confidence level based on the supporting quantile evidence:
      - **high**: quantile < 0.10 or > 0.90 AND |z-score| > 1.5
      - **medium**: quantile < 0.15 or > 0.85 (or multiple concordant features)
      - **low**: quantile < 0.25 or > 0.75
      - **reject**: quantile between 0.25 and 0.75 (no signal above noise), OR clear counter-evidence contradicts the mechanism
   d. Update the hypothesis via `update_hypothesis`:
      - **status**: "supported" (high, medium, or low confidence) or "rejected"
      - **result_summary**: explanation of the evidence
      - **direction**: pro-response or pro-resistance
      - **confidence**: high, medium, or low
      - **data_source**: infusion, spatial, or both
      - **p_value** and **effect_size**: if available
3. For each **supported** hypothesis (any confidence level), record as a finding via `update_knowledge_state` with:
   - **title**: concise mechanism name
   - **evidence**: quantitative evidence with specific values; include metadata at the end:
     `[direction=pro-response; confidence=medium; data_source=infusion]`
   - **direction**, **confidence**, **data_source**: pass these as tool parameters too
4. Call `save_iteration_summary` as your final action."""


_BUILDERS = {
    "data_profiling": _build_data_profiling,
    "hypothesize_literature": _build_hypothesize_literature,
    "falsification": _build_falsification,
}
