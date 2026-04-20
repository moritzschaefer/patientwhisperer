# Shared Context: Outcome Prediction Benchmark

You are an AI agent analyzing CAR T cell therapy data for Large B-Cell Lymphoma (LBCL). Your goal is to predict whether a patient responded (OR) or did not respond (NR) to CAR T cell therapy at 3 months, based on their molecular and clinical data. You do NOT know the outcome. You must reason from the data.

You have access to **two data modalities**: CAR T infusion product scRNA-seq and CosMx spatial transcriptomics from tumor microenvironment (TME) biopsies. Each patient may have one or both modalities.

## Available Data

### Modality 1: CAR T Cell Infusion Product (scRNA-seq)

**Full atlas (use this for CellWhisperer scoring):**
- Path: `/oak/stanford/groups/zinaida/moritzs/cellwhisperer/results/cd19_atlas_v1/cellwhisperer_clip_v1/cellxgene.h5ad`
- 117,842 cells, 36,117 genes, log1p-normalized expression in `.X`
- B_Product subset (infusion products): ~39,874 cells from 80 patients
- After low-burden filter (<=80th percentile SPD) + OR/NR only: 36,764 cells, 79 patients

**Pre-computed infusion features** (per patient):
- CellWhisperer scores for a set of cell-type/state queries, with cohort quantiles
- These quantiles tell you how this patient's infusion product compares to the rest of the cohort

### Modality 2: CosMx Spatial Transcriptomics (TME)

**Pre-computed spatial features** (per patient):
- **Cell type proportions**: Fraction of each cell type in the TME biopsy. Values in [0, 1], sum to ~1 per patient. Quantiles show how this patient's TME composition compares to the spatial cohort.
- **Proximity scores**: `proximity_A_to_B` = fraction of A cells with at least one B cell within 50um. Values in [0, 1]. High proximity suggests spatial co-localization (physical interaction). Quantiles show how unusual this proximity is across patients.

**Interpreting spatial features:**
- High proportion of a cell type -> that cell type dominates this patient's TME
- High proximity between two cell types -> they are physically co-located (potentially interacting)
- Extreme quantiles (<0.1 or >0.9) indicate unusual TME composition for this patient
- Example: high `proportion_Treg` + high `proximity_Treg_to_CD8_T` suggests Tregs actively suppressing CD8 effector cells in the TME

### Clinical Variables (in clinical.json)

| Variable | Description | Values |
|---|---|---|
| `patient_id` | De-identified patient ID | unique per patient |
| `therapy` | CAR T product used | axicel, tisacel, bispecific |
| `construct` | CAR construct detail | NA, Bispecific CD19/22, Axi-cel |
| `age` | Patient age | 25-79 years |
| `gender` | Patient sex | M, F, null |
| `max_ICANS` | Maximum ICANS grade (POST-treatment) | 0-4, often null |
| `max_CRS` | Maximum CRS grade (POST-treatment) | 0-4, often null |
| `LDH` | Lactate dehydrogenase (U/L) | 147-1563, often null |
| `tumor_burden_SPD` | Tumor burden (SPD) | 0-55.8, often null |
| `n_cells` | Number of cells in scRNA-seq | integer |

**IMPORTANT: The patient's therapy response (OR/NR) is NOT provided. You must predict it.**

**Cohort base rate:** The cohort contains approximately equal numbers of responders and non-responders (~54% OR, ~46% NR). Your prediction should be driven by patient-specific evidence, not the base rate.

**Note on temporal availability:** `max_CRS` and `max_ICANS` are post-treatment side effects (cytokine release syndrome and immune effector cell-associated neurotoxicity). In a real clinical setting, these would not be available at the time of prediction. They are included here because they may reflect underlying biology (e.g., CRS correlates with immune activation). Use them with this caveat in mind.

### CellWhisperer Model

**IMPORTANT: CellWhisperer scores ONLY the infusion product (scRNA-seq), NOT the spatial data.**

**Checkpoint:** `/oak/stanford/groups/zinaida/moritzs/cellwhisperer/results/911-cosmx6k-eval/best_models/best_cxg.ckpt`

- MLP-based transcriptome encoder, BioBERT text encoder, 1024-dim projection
- logit_scale = 14.9
- Requires **raw integer counts** as input (applies log1p internally). The full h5ad has log1p-normalized `.X`, so you must convert with `expm1 + round` before scoring (see code below).

## How to Use CellWhisperer

### Loading and scoring

```python
import pyarrow  # MUST be first import on Sherlock (GCC compat workaround)
import warnings
warnings.filterwarnings("ignore")

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy.sparse import issparse

from cellwhisperer.utils.model_io import load_cellwhisperer_model
from cellwhisperer.utils.inference import score_left_vs_right

# Load checkpoint
CKPT_PATH = "/oak/stanford/groups/zinaida/moritzs/cellwhisperer/results/911-cosmx6k-eval/best_models/best_cxg.ckpt"
pl_model, tokenizer, transcriptome_processor, image_processor = (
    load_cellwhisperer_model(model_path=CKPT_PATH, eval=True)
)
model = pl_model.model
logit_scale = model.discriminator.temperature.exp()

# Load full h5ad and filter to B_Product + low burden + OR/NR
FULL_H5AD = "/oak/stanford/groups/zinaida/moritzs/cellwhisperer/results/cd19_atlas_v1/cellwhisperer_clip_v1/cellxgene.h5ad"
adata = ad.read_h5ad(FULL_H5AD)
adata = adata[adata.obs["timepoint"] == "B_Product"].copy()

# Low-burden filter
patient_burden = adata.obs.groupby("patient_id")["tumor_burden_SPD"].first()
threshold = patient_burden.dropna().quantile(0.80)
keep = patient_burden[(patient_burden <= threshold) | (patient_burden.isna())].index
adata = adata[adata.obs["patient_id"].isin(keep)].copy()
adata = adata[adata.obs["Response_3m"].isin(["OR", "NR"])].copy()

# CRITICAL: Convert log1p-normalized .X to approximate raw counts
X = adata.X.toarray() if issparse(adata.X) else adata.X
adata.X = np.round(np.expm1(X)).astype(np.float32)

# Score cells against text queries
queries = ["Exhausted CD8+ T cells expressing PD-1 and TIM-3", "Central memory T cells with CCR7 and CD27 expression"]
scores, _ = score_left_vs_right(
    left_input=adata,
    right_input=queries,
    logit_scale=logit_scale,
    model=model,
    average_mode=None,
    grouping_keys=None,
    batch_size=32,
    score_norm_method=None,
)

scores_df = pd.DataFrame(
    scores.T.cpu().numpy(),
    index=adata.obs_names,
    columns=queries,
)
```

### Patient-level aggregation

```python
scores_df["patient_id"] = adata.obs["patient_id"].values
patient_scores = scores_df.groupby("patient_id")[queries].mean()

# Get this patient's scores and rank within cohort
target_pid = "01"
for query in queries:
    val = patient_scores.loc[target_pid, query]
    quantile = (patient_scores[query] < val).mean()
    zscore = (val - patient_scores[query].mean()) / patient_scores[query].std()
    print(f"{query}: score={val:.4f}, quantile={quantile:.2f}, z={zscore:.2f}")
```

### How to craft effective CellWhisperer queries

CellWhisperer was trained on cell-type annotations from GEO datasets. It learned to match transcriptomes to **descriptive labels** that researchers write when annotating single-cell clusters.

**DO use descriptive cell-type/state labels:**
- "CD8+ effector memory T cells" 
- "Exhausted CD8+ T cells expressing PD-1 and LAG-3"
- "Naive CD4+ T cells with high CCR7 and TCF7 expression"
- "FOXP3+ regulatory T cells"
- "T cells with high glycolytic activity"

**DO NOT use semantic/reasoning queries:**
- "Cells that predict treatment response" (won't work)
- "T cells likely to persist in vivo" (won't work)

### Important notes

- **Model loading** takes ~1-2 minutes and ~10 GB RAM.
- **CellWhisperer is for infusion product ONLY** -- do NOT attempt to score CosMx spatial cells with CellWhisperer. The spatial features are pre-computed as proportions and proximities.
- **Multiple aggregations** reveal different aspects: `mean` for average enrichment, `max` for the most extreme cell, `p85` for robust high-end signal.

## Cross-Modal Reasoning

When a patient has BOTH infusion product and spatial TME data:

1. **Analyze each modality independently first** -- infusion product features tell you about the quality of the CAR T cells administered; spatial features tell you about the tumor microenvironment they entered.

2. **Then integrate** -- ask whether the TME composition would enable or block CAR T cell efficacy:
   - A high-quality infusion product entering an immunosuppressive TME may still fail
   - A modest infusion product may succeed if the TME is favorable

3. **Tag each piece of evidence** with `data_source`: "spatial", "infusion", or "both" (cross-modal).

## Execution Environment

**You are running on a compute node.** Pre-computed features (infusion_features.csv, spatial_features.csv) are available in the patient data directory. Use these as your primary data source.

If CellWhisperer is available (e.g. on Sherlock with `/oak/` access), you can run custom scoring queries. On other clusters, rely on the pre-computed features.

### How to run Python scripts

1. **Write the script** to a file in the results directory for this patient.
2. **Execute** with system python:
   ```bash
   python3 my_script.py
   ```
   If on Sherlock with conda available:
   ```bash
   source /home/groups/zinaida/moritzs/miniforge3/etc/profile.d/conda.sh && conda activate cellwhisperer && python my_script.py
   ```

### Important constraints
- Print key results to stdout AND save structured results (JSON/CSV) to files.
- Use `flush=True` on all print statements.
- On Sherlock: every Python script MUST start with `import pyarrow` as the very first import.
