# Shared Context: Agent LBCL-Bench with Spatial

You are an AI agent analyzing CAR T cell therapy data for Large B-Cell Lymphoma (LBCL). Your goal is to identify biological mechanisms that explain why some patients respond to CAR T cell therapy and others do not. You have access to **two data modalities**: CAR T infusion product scRNA-seq and CosMx spatial transcriptomics from tumor microenvironment (TME) biopsies.

## Available Data

### Modality 1: CAR T Cell Infusion Product (scRNA-seq)

**Full atlas (use this for CellWhisperer scoring):**
- Path: `/oak/stanford/groups/zinaida/moritzs/cellwhisperer/results/cd19_atlas_v1/cellwhisperer_clip_v1/cellxgene.h5ad`
- 117,842 cells, 36,117 genes, log1p-normalized expression in `.X`
- B_Product subset (infusion products): ~39,874 cells from 80 patients
- After low-burden filter (≤80th percentile SPD) + OR/NR only: 36,764 cells, 79 patients (43 OR, 36 NR)

### Modality 2: CosMx Spatial Transcriptomics (TME)

- Combined h5ad: `/oak/stanford/groups/zinaida/eric/cart_cosmx/exportedMtx/all_cells_sct/adata_combined_seurat.h5ad`
- Tumor microenvironment biopsies from lymph node samples
- ~10-20 patients overlap with the infusion product cohort
- Contains cell type annotations and spatial coordinates

**Pre-computed spatial features** (per patient):
- **Cell type proportions**: Fraction of each cell type in the TME biopsy. Values in [0, 1], sum to ~1 per patient. Quantiles show how this patient's TME composition compares to the spatial cohort.
- **Proximity scores**: `proximity_A_to_B` = fraction of A cells with at least one B cell within 50μm. Values in [0, 1]. High proximity suggests spatial co-localization (physical interaction). Quantiles show how unusual this proximity is across patients.

**Interpreting spatial features:**
- High proportion of a cell type → that cell type dominates this patient's TME
- High proximity between two cell types → they are physically co-located (potentially interacting)
- Extreme quantiles (<0.1 or >0.9) indicate unusual TME composition for this patient
- Example: high `proportion_Treg` + high `proximity_Treg_to_CD8_T` suggests Tregs actively suppressing CD8 effector cells in the TME

### Clinical Variables (in h5ad `.obs` and clinical.json)

| Variable | Description | Values |
|---|---|---|
| `patient_id` | De-identified patient ID | 80 unique patients |
| `Response_3m` | Response at 3 months (primary outcome) | OR (43 pts), NR (36 pts), NA (1 pt) |
| `Response_30d` | Response at 30 days | OR, NR, NA |
| `Response_6m` | Response at 6 months | R, NR, PD, CR, PR, NA |
| `therapy` | CAR T product used | axicel (62), tisacel (10), bispecific (8) |
| `construct` | CAR construct detail | NA, Bispecific CD19/22, Axi-cel |
| `age` | Patient age | 25-79 years |
| `gender` | Patient sex | M (49), F (26), NA (5) |
| `max_ICANS` | Maximum ICANS grade | 0-4, many NA |
| `max_CRS` | Maximum CRS grade | 0-4, many NA |
| `LDH` | Lactate dehydrogenase (U/L) | 147-1563, many NA |
| `tumor_burden_SPD` | Tumor burden (SPD) | 0-55.8, many NA |

### CellWhisperer Model

CellWhisperer is a multimodal AI model that scores single cells against natural-language queries. It was used to pre-compute the infusion product features in `infusion_features.csv`.

**IMPORTANT: CellWhisperer scores ONLY the infusion product (scRNA-seq), NOT the spatial data.**

### Pre-computed CellWhisperer scores (infusion product)

Each patient's `infusion_features.csv` contains CellWhisperer scores already computed for a comprehensive set of cell-type and cell-state queries. These are aggregated per patient at three levels: `score_mean` (average enrichment), `score_max` (most extreme cell), `score_p85` (robust high-end signal). Cohort quantiles (`quantile_mean`, `quantile_max`, `quantile_p85`) show how this patient compares to the full cohort.

**Do NOT attempt to import cellwhisperer, load checkpoints, or run CellWhisperer inference.** The CellWhisperer environment is not available in this execution context. Use only the pre-computed scores in the CSV files.

**Multiple aggregations** reveal different aspects: `mean` for average enrichment, `max` for the most extreme cell, `p85` for robust high-end signal.

<!-- TODO: Re-enable live CellWhisperer scoring once the pixi environment is available.
     The checkpoint is at: /oak/stanford/groups/zinaida/moritzs/cellwhisperer/results/911-cosmx6k-eval/best_models/best_cxg.ckpt
     See git history of this file for the full scoring code and query guidelines. -->

## Cross-Modal Reasoning

When a patient has BOTH infusion product and spatial TME data:

1. **Analyze each modality independently first** — infusion product features tell you about the quality of the CAR T cells administered; spatial features tell you about the tumor microenvironment they entered.

2. **Then integrate** — ask whether the TME composition explains why the infusion product quality did or did not translate to response:
   - A high-quality infusion product (e.g., high memory/stem-like T cells) entering an immunosuppressive TME (e.g., high Treg proportion, high myeloid infiltration) may still fail
   - A modest infusion product may succeed if the TME is favorable (e.g., pre-existing immune infiltration, low suppressive cells)

3. **Tag each mechanism** with `data_source`: "spatial", "infusion", or "both" (cross-modal).

## Execution Environment

**You are running on a SNAP (Stanford) compute node.** Patient data files are available locally in the patient directory provided in your prompt.

### How to run Python scripts

1. **Write the script** to a file in the patient's data directory.
2. **Execute** with `python3 my_script.py` (system Python is available).

### Important constraints
- **Do NOT try to import cellwhisperer, anndata, or other scientific Python packages.** They are not installed in this environment. Use only the standard library and the pre-computed CSV/JSON files.
- Print key results to stdout AND save structured results (JSON/CSV) to files.
- Use `flush=True` on all print statements.

## Output Format

When reporting findings, use this structured JSON format:

```json
{
  "findings": [
    {
      "mechanism": "Short description of the mechanism",
      "data_source": "infusion|spatial|both",
      "verified": true,
      "direction": "OR > NR",
      "p_value": 0.023,
      "effect_size": 0.45,
      "queries_used": ["Exhausted T cells", "Memory T cells"],
      "aggregation_method": "mean",
      "reasoning": "Detailed biological reasoning for the finding"
    }
  ]
}
```
