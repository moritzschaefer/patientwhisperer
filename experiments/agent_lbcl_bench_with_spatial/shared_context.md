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

## Cohort-level Feature Discrimination Reference

The following statistics summarize how features differ between overall responders (OR, n=43) and non-responders (NR, n=36) across the full cohort. Use these to ground your per-patient observations: a feature that is extreme in this patient AND discriminates at the cohort level is stronger evidence for an outcome-related mechanism than one that is merely extreme for this patient.

**How to use:**
- When a feature is extreme for a patient, check whether it discriminates OR vs NR at the cohort level.
- Prioritize mechanisms built on features where the patient's deviation ALIGNS with the cohort-level direction.
- Features that do NOT discriminate cohort-wide (p > 0.10) may still reflect patient-specific biology but are weaker evidence for outcome-linked mechanisms.

### Clinical Variables

| Variable | OR mean | NR mean | Direction | p-value |
|---|---|---|---|---|
| age | 55.6 | 63.7 | NR > OR | 0.006 |
| LDH | 261.4 | 368.9 | NR > OR | 0.463 |
| tumor_burden_SPD | 17.9 | 23.9 | NR > OR | 0.283 |

### Ratio Features (p < 0.10)

Ratios capture biologically meaningful cell-state balances and are the strongest discriminators.

| Ratio | Agg | OR mean | NR mean | Direction | p-value | d |
|---|---|---|---|---|---|---|
| polyfunctional_vs_anergic | median | 1.677 | 1.555 | OR > NR | 0.024 | -0.296 |
| polyfunctional_vs_anergic | mean | 1.652 | 1.541 | OR > NR | 0.028 | -0.289 |
| cd8_vs_cd4 | median | 1.068 | 0.913 | OR > NR | 0.031 | -0.284 |
| cd8_vs_cd4 | mean | 1.063 | 0.937 | OR > NR | 0.039 | -0.271 |
| activated_vs_resting | mean | 0.830 | 0.855 | NR > OR | 0.040 | 0.270 |
| treg_vs_cd8 | mean | 1.181 | 1.370 | NR > OR | 0.048 | 0.260 |
| treg_vs_cd8 | median | 1.225 | 1.466 | NR > OR | 0.048 | 0.260 |
| cytotoxic_vs_dysfunctional | mean | 0.848 | 0.817 | OR > NR | 0.058 | -0.249 |
| cd8_vs_cd4 | p85 | 1.110 | 1.012 | OR > NR | 0.061 | -0.247 |
| cytotoxic_vs_dysfunctional | median | 0.837 | 0.802 | OR > NR | 0.062 | -0.245 |
| activated_vs_resting | median | 0.829 | 0.850 | NR > OR | 0.074 | 0.235 |
| polyfunctional_vs_anergic | p85 | 1.448 | 1.378 | OR > NR | 0.081 | -0.230 |
| mito_vs_hypoxia | mean | 1.123 | 1.096 | OR > NR | 0.091 | -0.222 |
| activated_vs_resting | p85 | 0.865 | 0.883 | NR > OR | 0.091 | 0.222 |

**Key biological signals:**
- **CD8/CD4 ratio**: Responders have higher CD8 relative to CD4. Low CD8/CD4 tracks with non-response.
- **Polyfunctional/Anergic**: Responders have more polyfunctional vs anergic cells. Low ratio signals T cell dysfunction.
- **Treg/CD8**: Non-responders have higher Treg relative to CD8. High ratio suggests regulatory suppression of cytotoxic cells.
- **Activated/Resting**: Non-responders show higher activated/resting ratio, possibly reflecting activation without effective cytotoxicity (activation-exhaustion coupling).
- **Cytotoxic/Dysfunctional**: Responders have higher cytotoxic vs dysfunctional ratio. Low ratio indicates impaired killing capacity.
- **Mitochondrial/Hypoxia**: Responders show higher mitochondrial activity relative to hypoxia response, suggesting better metabolic fitness.

### Individual CellWhisperer Features (top discriminators, p < 0.05)

| Feature | Agg | OR mean | NR mean | Direction | p-value |
|---|---|---|---|---|---|
| th2 | median | 5.373 | 5.759 | NR > OR | 0.017 |
| tumor_cells | p85 | 3.779 | 4.153 | NR > OR | 0.024 |
| th1 | median | 4.619 | 4.981 | NR > OR | 0.026 |
| myc_high | median | 7.005 | 6.735 | OR > NR | 0.029 |
| myc_high | mean | 6.870 | 6.616 | OR > NR | 0.030 |
| multi_checkpoint | p85 | 7.785 | 7.908 | NR > OR | 0.035 |
| cd4_helper | median | 4.981 | 5.286 | NR > OR | 0.036 |
| th17 | median | 6.293 | 6.597 | NR > OR | 0.037 |
| proliferating | p85 | 9.120 | 8.902 | OR > NR | 0.039 |
| high_viability | median | 5.370 | 5.161 | OR > NR | 0.042 |
| anergic | median | 4.980 | 5.299 | NR > OR | 0.043 |
| lymphoma_cells | median | 1.032 | 1.240 | NR > OR | 0.046 |
| nkt | median | 3.153 | 2.727 | OR > NR | 0.047 |

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
