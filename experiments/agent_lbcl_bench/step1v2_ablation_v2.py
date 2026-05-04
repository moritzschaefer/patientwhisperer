"""
Step 1v2 Ablation v2: Re-embed cells from full gene expression for each checkpoint.

Key differences from v1:
- Uses the FULL cellxgene.h5ad (36K genes) instead of the light version (1 dummy gene)
- Each checkpoint's own transcriptome encoder is used to embed cells
- Includes ratio features (matching the original run_patient_signal_analysis.py)
- All 3 checkpoints tested: old_jointemb, spatialwhisperer_v1, best_cxg

For each checkpoint:
1. Load model
2. Score cells vs text queries using score_left_vs_right(left_input=adata, ...)
   This embeds cells through the model's transcriptome encoder
3. Compute patient-level aggregations (mean, frac_high75, max, p85)
4. Compute ratio features for biologically meaningful query pairs
5. Test OR vs NR with Mann-Whitney U

Usage (on Sherlock compute node, needs ~60GB RAM):
    conda run -n cellwhisperer python step1v2_ablation_v2.py
"""
import pyarrow  # MUST be first import on Sherlock

import gc
import json
import os
import sys
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy.stats import mannwhitneyu

warnings.filterwarnings("ignore")

# -- Paths --
FULL_H5AD_PATH = "/oak/stanford/groups/zinaida/moritzs/cellwhisperer/results/cd19_atlas_v1/cellwhisperer_clip_v1/cellxgene.h5ad"
QUERIES_JSON = os.path.join(os.path.dirname(__file__), "data", "step1v2_queries.json")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results", "step1v2_ablation_v2")

CHECKPOINTS = {
    "old_jointemb": "/oak/stanford/groups/zinaida/moritzs/cellwhisperer/results/models/jointemb/old_format/spotwhisperer_cellxgene_census__archs4_geo.ckpt",
    "spatialwhisperer_v1": "/oak/stanford/groups/zinaida/moritzs/cellwhisperer/results/models/spatialwhisperer/v1.ckpt",
    "best_cxg": "/oak/stanford/groups/zinaida/moritzs/cellwhisperer/results/911-cosmx6k-eval/best_models/best_cxg.ckpt",
}

# Ratio pairs: (numerator_query, denominator_query) — matching original analysis
RATIO_PAIRS = [
    ("CD8+ T cells", "CD4+ T cells"),
    ("Memory T cells", "Naive T cells"),
    ("Effector T cells", "Regulatory T cells"),
    ("Activated T cells", "Anergic cells"),
    ("Cytotoxic T cells", "Anergic cells"),
    ("Proliferating cells", "Quiescent cells"),
    ("Glycolytic cells", "Oxidative cells"),
    ("Stem cell-like T cells", "Terminally differentiated cells"),
    ("Antigen-experienced cells", "Antigen-naive cells"),
]

# Queries from original analysis (these are the "base" queries for ratio computation)
ORIGINAL_QUERIES = [
    "CD8+ T cells", "CD4+ T cells", "Memory T cells", "Naive T cells",
    "Effector T cells", "Regulatory T cells", "Senescent T cells",
    "Activated T cells", "Anergic cells", "Cytotoxic T cells",
    "Th1 cells", "Th17 cells",
    "Proliferating cells", "Apoptotic cells", "Quiescent cells",
    "Hypoxic cells", "Damaged cells",
    "Glycolytic cells", "Oxidative cells",
    "Stem cell-like T cells", "Terminally differentiated cells",
    "Antigen-experienced cells", "Antigen-naive cells",
]

AGG_NAMES = ["mean", "frac_high75", "max", "p85"]


def load_and_filter_adata():
    """Load full h5ad, filter to B_Product + low burden (same as subset_cellxgene.py)."""
    print(f"Loading full h5ad from {FULL_H5AD_PATH}", flush=True)
    adata = ad.read_h5ad(FULL_H5AD_PATH)
    print(f"  Full atlas: {adata.n_obs} cells, {adata.n_vars} genes", flush=True)

    # Filter to B_Product
    mask_tp = adata.obs["timepoint"] == "B_Product"
    adata_bp = adata[mask_tp].copy()
    print(f"  After B_Product filter: {adata_bp.n_obs} cells", flush=True)

    # Low burden filter (same as subset_cellxgene.py)
    patient_burden = adata_bp.obs.groupby("patient_id")["tumor_burden_SPD"].first()
    threshold = patient_burden.dropna().quantile(0.80)
    keep_patients = patient_burden[(patient_burden <= threshold) | (patient_burden.isna())].index
    mask_pat = adata_bp.obs["patient_id"].isin(keep_patients)
    adata_bp = adata_bp[mask_pat].copy()
    print(f"  After low-burden filter: {adata_bp.n_obs} cells", flush=True)

    # Filter to OR/NR
    mask_resp = adata_bp.obs["Response_3m"].isin(["OR", "NR"])
    adata_bp = adata_bp[mask_resp].copy()
    print(f"  After OR/NR filter: {adata_bp.n_obs} cells", flush=True)

    patient_response = adata_bp.obs.groupby("patient_id")["Response_3m"].first()
    print(f"  OR: {(patient_response == 'OR').sum()}, NR: {(patient_response == 'NR').sum()}", flush=True)

    # Free full atlas
    del adata
    gc.collect()

    return adata_bp, patient_response


def score_with_checkpoint(ckpt_path, adata, queries):
    """Load checkpoint, score adata vs queries, return scores DataFrame. Frees model."""
    from cellwhisperer.utils.model_io import load_cellwhisperer_model
    from cellwhisperer.utils.inference import score_left_vs_right
    from cellwhisperer.jointemb.mlp_model import MLPTranscriptomeProcessor
    from scipy.sparse import issparse

    print(f"  Loading checkpoint: {os.path.basename(ckpt_path)}", flush=True)
    pl_model, tokenizer, transcriptome_processor, image_processor = (
        load_cellwhisperer_model(model_path=ckpt_path, eval=True)
    )
    model = pl_model.model
    logit_scale = model.discriminator.temperature.exp()
    print(f"  logit_scale={logit_scale.item():.4f}", flush=True)
    print(f"  transcriptome_processor: {type(transcriptome_processor).__name__}", flush=True)

    # MLP processor expects raw counts (it applies log1p internally).
    # Our h5ad has log1p-normalized data. Convert back via expm1 + round.
    adata_input = adata
    if isinstance(transcriptome_processor, MLPTranscriptomeProcessor):
        print("  Converting log1p-normalized .X → approximate raw counts for MLP processor", flush=True)
        adata_input = adata.copy()
        X = adata_input.X
        if issparse(X):
            X = X.toarray()
        adata_input.X = np.round(np.expm1(X)).astype(np.float32)

    # Score: this re-embeds cells through the model's transcriptome encoder
    print(f"  Scoring {adata_input.n_obs} cells vs {len(queries)} queries...", flush=True)
    scores, _ = score_left_vs_right(
        left_input=adata_input,
        right_input=queries,
        logit_scale=logit_scale,
        model=model,
        average_mode=None,
        grouping_keys=None,
        batch_size=32,
        score_norm_method=None,
    )
    del adata_input
    # scores shape: (n_queries, n_cells)
    scores_df = pd.DataFrame(
        scores.T.cpu().numpy(),
        index=adata.obs_names,
        columns=queries,
    )
    print(f"  Score stats: min={scores_df.values.min():.4f}, max={scores_df.values.max():.4f}, "
          f"mean={scores_df.values.mean():.4f}, std={scores_df.values.std():.4f}", flush=True)

    ls_val = logit_scale.item()

    # Free model
    del pl_model, model, tokenizer, transcriptome_processor, image_processor, logit_scale
    gc.collect()
    torch.cuda.empty_cache()

    return scores_df, ls_val


def compute_patient_aggregations(scores_df, patient_ids, queries):
    """Compute all 4 aggregation strategies. Returns dict of agg_name -> DataFrame."""
    scores_with_pid = scores_df[queries].copy()
    scores_with_pid["patient_id"] = patient_ids

    aggs = {}
    aggs["mean"] = scores_with_pid.groupby("patient_id")[queries].mean()
    aggs["max"] = scores_with_pid.groupby("patient_id")[queries].max()
    aggs["p85"] = scores_with_pid.groupby("patient_id")[queries].quantile(0.85)

    thresholds = scores_df[queries].quantile(0.75)
    aggs["frac_high75"] = scores_with_pid.groupby("patient_id")[queries].apply(
        lambda g: (g > thresholds).mean()
    )

    return aggs


def add_ratios(agg_df, agg_name, ratio_pairs, available_queries):
    """Add ratio columns to an aggregation DataFrame. Returns list of new ratio column names."""
    ratio_cols = []
    for a, b in ratio_pairs:
        if a in available_queries and b in available_queries and a in agg_df.columns and b in agg_df.columns:
            col = f"ratio_{a}/{b}"
            agg_df[col] = agg_df[a] / (agg_df[b] + 1e-6)
            ratio_cols.append(col)
    return ratio_cols


def test_or_vs_nr(agg_df, patient_response, features):
    """Test each feature for OR vs NR difference. Returns list of result dicts."""
    or_pids = patient_response[patient_response == "OR"].index
    nr_pids = patient_response[patient_response == "NR"].index
    results = []
    for feat in features:
        if feat not in agg_df.columns:
            continue
        or_vals = agg_df.loc[agg_df.index.isin(or_pids), feat].dropna()
        nr_vals = agg_df.loc[agg_df.index.isin(nr_pids), feat].dropna()
        if len(or_vals) < 3 or len(nr_vals) < 3:
            continue
        stat, p = mannwhitneyu(or_vals, nr_vals, alternative="two-sided")
        or_mean = or_vals.mean()
        nr_mean = nr_vals.mean()
        results.append({
            "feature": feat,
            "p_raw": p,
            "or_mean": or_mean,
            "nr_mean": nr_mean,
            "direction": "OR > NR" if or_mean > nr_mean else "NR > OR",
            "effect_size": or_mean - nr_mean,
            "n_or": len(or_vals),
            "n_nr": len(nr_vals),
        })
    return results


def test_mechanisms(scores_df, patient_ids, patient_response, mechanisms, all_queries):
    """Run the full mechanism verification pipeline for one checkpoint."""
    # Compute patient-level aggregations
    aggs = compute_patient_aggregations(scores_df, patient_ids, all_queries)

    # Add ratio features to each aggregation
    for agg_name, agg_df in aggs.items():
        add_ratios(agg_df, agg_name, RATIO_PAIRS, all_queries)

    # Test each mechanism's queries + test original queries with ratios
    all_test_rows = []
    mechanism_summaries = []

    for mid, mdata in mechanisms.items():
        queries = mdata["queries"]
        expected_dir = mdata["expected_direction"]

        mech_tests = []
        for agg_name, agg_df in aggs.items():
            results = test_or_vs_nr(agg_df, patient_response, queries)
            for r in results:
                r["agg"] = agg_name
                r["mechanism_id"] = mid
                r["is_ratio"] = False
                mech_tests.append(r)

        all_test_rows.extend(mech_tests)

        # Summarize this mechanism
        if mech_tests:
            best = min(mech_tests, key=lambda r: r["p_raw"])
            directions = [r["direction"] for r in mech_tests]
            majority_dir = max(set(directions), key=directions.count)
            dir_agreement = directions.count(majority_dir) / len(directions)

            if expected_dir != "unknown":
                dir_matches = majority_dir == expected_dir
            else:
                dir_matches = None

            # Bonferroni: 10 queries × 4 aggs = 40 tests per mechanism
            bonferroni_n = len(mech_tests)
            best_p_corr = min(best["p_raw"] * bonferroni_n, 1.0)

            verified = (
                best_p_corr < 0.05
                and dir_agreement >= 0.7
                and (dir_matches is True or dir_matches is None)
            )

            mechanism_summaries.append({
                "mechanism_id": mid,
                "verbal_summary": mdata["summary"],
                "expected_direction": expected_dir,
                "verified": verified,
                "direction_matches_expected": dir_matches,
                "best_agg": best["agg"],
                "best_query": best["feature"],
                "best_p_raw": best["p_raw"],
                "best_p_corrected": best_p_corr,
                "best_direction": best["direction"],
                "best_effect_size": best["effect_size"],
                "majority_direction": majority_dir,
                "direction_agreement": dir_agreement,
                "n_tests": len(mech_tests),
            })

    # Also test the ORIGINAL queries + ratios (not mechanism-specific)
    original_test_rows = []
    for agg_name, agg_df in aggs.items():
        # Base queries
        base_results = test_or_vs_nr(agg_df, patient_response, ORIGINAL_QUERIES)
        for r in base_results:
            r["agg"] = agg_name
            r["is_ratio"] = False
            original_test_rows.append(r)

        # Ratio features
        ratio_cols = [c for c in agg_df.columns if c.startswith("ratio_")]
        ratio_results = test_or_vs_nr(agg_df, patient_response, ratio_cols)
        for r in ratio_results:
            r["agg"] = agg_name
            r["is_ratio"] = True
            original_test_rows.append(r)

    return (
        pd.DataFrame(mechanism_summaries),
        pd.DataFrame(all_test_rows),
        pd.DataFrame(original_test_rows),
    )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load pre-registered queries
    with open(QUERIES_JSON) as f:
        mechanisms = json.load(f)
    print(f"Loaded queries for {len(mechanisms)} mechanisms", flush=True)

    # Collect ALL unique queries (mechanism queries + original queries for ratios)
    all_queries = list(ORIGINAL_QUERIES)  # start with original
    seen = set(all_queries)
    for mid, mdata in mechanisms.items():
        for q in mdata["queries"]:
            if q not in seen:
                all_queries.append(q)
                seen.add(q)
    print(f"Total unique queries: {len(all_queries)}", flush=True)

    # Load and filter data (once)
    adata, patient_response = load_and_filter_adata()
    patient_ids = adata.obs["patient_id"].values

    # Results collector
    ablation_summary = []

    for ckpt_name, ckpt_path in CHECKPOINTS.items():
        print(f"\n{'='*60}", flush=True)
        print(f"CHECKPOINT: {ckpt_name}", flush=True)
        print(f"{'='*60}", flush=True)

        if not os.path.exists(ckpt_path):
            print(f"  WARNING: Checkpoint not found at {ckpt_path}. Skipping.", flush=True)
            continue

        # Skip if results already exist
        verify_path = os.path.join(OUTPUT_DIR, f"{ckpt_name}__mechanism_verification.csv")
        if os.path.exists(verify_path):
            print(f"  Results already exist at {verify_path}. Skipping.", flush=True)
            continue

        # Score cells vs all queries (re-embeds cells through this checkpoint's encoder)
        scores_df, logit_scale = score_with_checkpoint(ckpt_path, adata, all_queries)

        # Run mechanism verification + original-style tests
        mech_summary_df, mech_tests_df, orig_tests_df = test_mechanisms(
            scores_df, patient_ids, patient_response, mechanisms, all_queries,
        )

        # Save results
        mech_summary_df.to_csv(os.path.join(OUTPUT_DIR, f"{ckpt_name}__mechanism_verification.csv"), index=False)
        mech_tests_df.to_csv(os.path.join(OUTPUT_DIR, f"{ckpt_name}__mechanism_all_tests.csv"), index=False)
        orig_tests_df.to_csv(os.path.join(OUTPUT_DIR, f"{ckpt_name}__original_style_tests.csv"), index=False)

        n_verified = mech_summary_df["verified"].sum() if len(mech_summary_df) > 0 else 0

        # Print mechanism verification summary
        print(f"\n  MECHANISMS: {n_verified}/{len(mech_summary_df)} verified", flush=True)
        for _, r in mech_summary_df.sort_values("best_p_raw").head(5).iterrows():
            print(f"    {r['mechanism_id']} p_raw={r['best_p_raw']:.4f} p_corr={r['best_p_corrected']:.4f} "
                  f"agg={r['best_agg']} dir={r['best_direction']} match={r['direction_matches_expected']} "
                  f"{r['verbal_summary'][:45]}", flush=True)

        # Print original-style top hits
        if len(orig_tests_df) > 0:
            sig_orig = orig_tests_df[orig_tests_df["p_raw"] < 0.05].sort_values("p_raw")
            print(f"\n  ORIGINAL-STYLE: {len(sig_orig)} tests with p_raw < 0.05:", flush=True)
            for _, r in sig_orig.head(10).iterrows():
                print(f"    agg={r['agg']:12s} p={r['p_raw']:.4f} dir={r['direction']} "
                      f"ratio={r['is_ratio']} {r['feature'][:50]}", flush=True)

        ablation_summary.append({
            "checkpoint": ckpt_name,
            "logit_scale": logit_scale,
            "n_mechanisms_verified": int(n_verified),
            "n_mechanisms_total": len(mech_summary_df),
            "n_orig_sig_005": int((orig_tests_df["p_raw"] < 0.05).sum()) if len(orig_tests_df) > 0 else 0,
            "n_orig_tests": len(orig_tests_df),
            "score_mean": float(scores_df.values.mean()),
            "score_std": float(scores_df.values.std()),
        })

        # Free scores
        del scores_df
        gc.collect()

    # Save ablation summary
    ablation_df = pd.DataFrame(ablation_summary)
    ablation_df.to_csv(os.path.join(OUTPUT_DIR, "ablation_summary.csv"), index=False)

    print(f"\n{'='*60}", flush=True)
    print("ABLATION SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    print(ablation_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
