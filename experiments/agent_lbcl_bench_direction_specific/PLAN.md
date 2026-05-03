# LBCL-Bench with Spatial: Project Plan

Per-patient mechanistic analysis combining CAR T infusion product scRNA-seq (CellWhisperer-scored) with CosMx tumor microenvironment spatial transcriptomics.

## Goal

Extend baseline `agent_lbcl_bench` Step 3 to include spatial features (cell type proportions + pairwise proximities) alongside infusion product features. Enable cross-modal mechanistic reasoning for ~10–20 patients with both modalities, while still running infusion-only patients for direct comparison to baseline.

## Patient scope

ALL patients with at least one modality. Target: ~99 CAR T atlas patients + ~17 TME patients, with 10–20 expected to overlap once the patient ID crosswalk is resolved.

---

## Data sources

| Source | Path (local) | Path (Sherlock) | Notes |
|---|---|---|---|
| CAR T metadata | `/mnt/onedrive-zina/Good-Lab/T-Cell-Data-Warehouse/Metadata/CART-Atlas/CD19_atlas_patient_metadata.xlsx` | `/home/users/moritzs/patientwhisperer/metadata/CART-Atlas/...` | 99 patients, `patient_id` |
| TME metadata | `/mnt/onedrive-zina/.../TME/cart-cohort-tma-metadata_v2.xlsx` | `/home/users/moritzs/patientwhisperer/metadata/TME/...` | 17 TMA cores, `TMA` |
| TME QC | `/mnt/onedrive-zina/.../TME/deident_valid_cores_sheren_annote.xlsx` | (same prefix) | valid cores |
| CosMx h5ad | — | `/oak/stanford/groups/zinaida/eric/cart_cosmx/exportedMtx/all_cells_sct/adata_combined_seurat.h5ad` | 71 GB, 1.3M cells |
| CosMx metadata.csv | — | `/oak/stanford/groups/zinaida/eric/cart_cosmx/exportedMtx/all_cells_sct/metadata.csv` | Use for inspection (small) |
| CAR T h5ad | — | `/oak/stanford/groups/zinaida/moritzs/cellwhisperer/results/cd19_atlas_v1/cellwhisperer_clip_v1/cellxgene.h5ad` | baseline 79 patients |
| CARTAtlas warehouse | — | `/oak/stanford/groups/zinaida/CAR_T_data_warehouse/CARTAtlas/` | Possible additional patients (RDS) |

---

## Data transfer strategy

OneDrive content (metadata Excel files, etc.) lives on a SharePoint drive that is FUSE-mounted locally on `moair` at `/mnt/onedrive-zina/`. The workflow:

**Local mount → scp/lsyncd → Sherlock**

- The local `/mnt/onedrive-zina/` mount is reliable and fast (NixOS).
- Read files locally, then push to Sherlock with `scp` or via `lsyncd` (`/home/moritz/code/patientwhisperer/lsyncd.conf`).
- Files land under `/home/users/moritzs/patientwhisperer/metadata/<subdir>/`.

---

## Phases

### Phase 0: Infrastructure (DONE)

- `lsyncd.conf` at repo root → syncs to both Sherlock and SNAP (ILC)
- **Sherlock** dirs:
  - `/home/users/moritzs/patientwhisperer/`
  - `~/oak_home/patientwhisperer/results` ← symlinked as `results/`
  - `~/scratch/patientwhisperer/scratch` ← symlinked as `scratch/`
- **SNAP (ILC)** dirs:
  - `/sailhome/moritzs/patientwhisperer/`
  - `/dfs/user/moritzs/patientwhisperer/results` ← symlinked as `results/`
  - `/lfs/local/0/moritzs/patientwhisperer/scratch` ← symlinked as `scratch/`
  - Oak access via `~/bin/mount-oak.sh` (sshfs to sherlock-dtn, requires SSHPASS)
  - UV env at `/lfs/local/0/$USER/uv-envs/patientwhisperer`
- OneDrive access: local mount on `moair`, files pushed via scp/lsyncd.
- All Python scripts use `OAK_ROOT` env var (defaults to `/oak/stanford/groups/zinaida`, overridden on SNAP to `$LFS/oak`).

### Phase 1: QC & Patient Matching (DONE)

`step0_qc_patient_matching.py` (two-phase: `--phase metadata` local, `--phase h5ad` on Sherlock).

**Result:** 99 CAR T patients (`patient_id`: `ac01`, `Pt010`, …), 17 TMA cores (`TMA1`…), CosMx `ANON_pathID` (`SHF-*` / `SHS-*`), pathology CSV `ANON_MRN` (`PAT-*`). **No direct ID overlap** and no crosswalk available.

**Decision:** proceed with both modalities **independently** — treat CAR T and CosMx patients as disjoint cohorts. Every patient gets `data_sources.json` with exactly one of `{has_infusion, has_spatial}` set to `true`. The agent gracefully degrades to single-modality analysis. Cross-modal reasoning is deferred until/if a crosswalk becomes available.

### Phase 2: Data preparation (DONE)

- `step3a_prepare_infusion_features.py`: ✅ Completed on Sherlock (job 21070103). 80 patients → `data/infusion_features/{pid}/`
- `step3a_prepare_spatial_features.py`: ✅ Completed (SNAP job 19887). Filters: SHS/SP prefix + lymph node biopsy_site + exclude TMA1. 19 patients (8 SHS + 11 SP).
  - SNAP wrapper: `run_snap_spatial.sh` (mounts oak via sshfs, sets OAK_ROOT)
- `step3a_merge_patient_data.py`: ⏭ after spatial re-run — union merge → `data/patients/{pid}/{clinical.json,infusion_features.csv,spatial_features.csv,data_sources.json}`

### Phase 3: Per-patient agent analysis (DONE — scripts written)

- `run_agent.py` with `AGENT_FRAMEWORK = "claudecode"` switch (alternative `"opencode"`)
- `shared_context.md` + `patient-analyst-instructions.md` → concatenated to `system_prompt_combined.md` for Claude Code's `--append-system-prompt-file`
- `.opencode/agents/patient-analyst.md` for opencode compat
- 6 rounds: review → CellWhisperer → spatial → cross-modal → validation → synthesis
- Output JSON includes `data_source` per mechanism (`spatial` / `infusion` / `both`)
- `run_step3_patients.sh`: SLURM array job

### Phase 4: Evaluation (DONE — scripts written)

- `step3c_evaluate.py`: per-modality stratified mechanism recovery (`spatial_matches`, `infusion_only_matches`)
- `step3c_concordance.py`: per-modality concordance breakdown
- `data/lbcl_bench_filtered.csv` symlinked to baseline bench

### Phase 5: Snakemake (DONE)

`Snakefile` orchestrates all of the above with a checkpoint on `prepare_infusion`.

---

## Current status

| Item | State |
|---|---|
| Infrastructure (lsyncd, dirs, SNAP setup) | ✅ |
| All plan files created | ✅ |
| Metadata loaded (calamine engine) | ✅ |
| Patient ID crosswalk | ⏭ **skipped** — disjoint cohorts |
| Infusion features (80 patients, Sherlock) | ✅ job 21070103 |
| Spatial features (SHS/SP+LN, excl TMA1, SNAP) | ✅ job 19887 — 19 patients |
| Infusion synced locally | ✅ `data/infusion_features/` (79 patients) |
| Spatial synced locally | ✅ `data/spatial_features/` (19 patients) |
| Merge patient data | ✅ 98 patients (79 infusion-only, 19 spatial-only, 0 overlap) |
| Agent analysis | ⏳ SNAP job 19888 (array 0-97, ~8 min/patient, ~80 min total) |
| Evaluation | ⏸ |
| Agent analysis | ⏸ |
| Evaluation | ⏸ |

## Next actions

1. ~~Re-run `step3a_prepare_spatial_features.py` on SNAP with LN filter~~ → submitted as job 19591 (SHS + LN + excl TMA1).
2. Sync filtered spatial features locally (use `--delete` to clean stale dirs).
3. Run `step3a_merge_patient_data.py` locally → disjoint union; each patient dir marked with the single modality it has.
4. Dispatch per-patient agents via `run_step3_patients.sh` (adapt for SNAP if Sherlock still congested).
5. Evaluate with `step3c_evaluate.py` + `step3c_concordance.py`.

## Files copied to clusters

**Sherlock** `/home/users/moritzs/patientwhisperer/metadata/`:
- `CART-Atlas/CD19_atlas_patient_metadata.xlsx`
- `TME/cart-cohort-tma-metadata_v2.xlsx`
- `TME/deident_valid_cores_sheren_annote.xlsx`
- `TME/deident_annotated_pathology_v2.csv` (biopsy_site for LN filtering)

**SNAP** `/sailhome/moritzs/patientwhisperer/metadata/`:
- Same files as above.

**Local** `/home/moritz/code/patientwhisperer/metadata/`:
- `TME/deident_annotated_pathology_v2.csv`

## Key risks

- **No cross-modal patients** → cannot do true multi-modal reasoning; restricted to per-modality comparisons against the bench
- **Additional patients in CARTAtlas/** (RDS) may need conversion
- **CosMx h5ad load over sshfs is slow** (~10 min); for repeated runs, rsync-stage to `/dfs/user/moritzs/`
