"""
Step 1v2 Ablation: 3 checkpoints × 2 aggregation strategies.

Tests which CellWhisperer checkpoint's text encoder produces meaningful
cosine similarities with the pre-computed cell embeddings in obsm, and
whether non-mean aggregations (frac_high75, max, p85) are needed.

All 6 conditions use the SAME pre-computed obsm["transcriptome_embeds"]
and the SAME 220 pre-registered queries from data/step1v2_queries.json.

Checkpoints loaded sequentially to avoid OOM.

Usage (on Sherlock compute node):
    conda run -n cellwhisperer python step1v2_ablation.py
"""
import pyarrow  # MUST be first import on Sherlock

import gc
import json
import os
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy.stats import mannwhitneyu

warnings.filterwarnings("ignore")

# -- Paths --
H5AD_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "car_t_ignite_prelim_analysis",
    "cellxgene_B_Product_lowburden.light.h5ad",
)
QUERIES_JSON = os.path.join(os.path.dirname(__file__), "data", "step1v2_queries.json")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results", "step1v2_ablation")

CHECKPOINTS = {
    "old_jointemb": "/oak/stanford/groups/zinaida/moritzs/cellwhisperer/results/models/jointemb/old_format/spotwhisperer_cellxgene_census__archs4_geo.ckpt",
    "spatialwhisperer_v1": "/oak/stanford/groups/zinaida/moritzs/cellwhisperer/results/models/spatialwhisperer/v1.ckpt",
    "best_cxg": "/oak/stanford/groups/zinaida/moritzs/cellwhisperer/results/911-cosmx6k-eval/best_models/best_cxg.ckpt",
}

AGG_STRATEGIES = {
    "mean_only": {
        "aggs": ["mean"],
        "bonferroni_n": 10,  # 10 queries per mechanism
    },
    "all_4_aggs": {
        "aggs": ["mean", "frac_high75", "max", "p85"],
        "bonferroni_n": 40,  # 10 queries × 4 aggregations
    },
}


def load_cell_data():
    """Load h5ad, extract pre-computed embeddings as tensor, return tensor + metadata."""
    print(f"Loading h5ad from {H5AD_PATH}")
    adata = ad.read_h5ad(H5AD_PATH)
    print(f"  {adata.n_obs} cells, {adata.n_vars} genes")

    # Filter to OR/NR
    mask = adata.obs["Response_3m"].isin(["OR", "NR"])
    adata = adata[mask].copy()
    print(f"  After OR/NR filter: {adata.n_obs} cells")

    # Extract pre-computed embeddings
    cell_embeds = torch.tensor(adata.obsm["transcriptome_embeds"], dtype=torch.float32)
    print(f"  Embedding shape: {cell_embeds.shape}")

    # Check and enforce L2 normalization
    norms = torch.norm(cell_embeds, dim=1)
    print(f"  Embedding norms: min={norms.min():.6f}, max={norms.max():.6f}, mean={norms.mean():.6f}")
    if not ((norms - 1).abs() < 1e-3).all():
        print("  WARNING: Embeddings not L2-normalized. Normalizing now.")
        cell_embeds = cell_embeds / norms.unsqueeze(1)
    else:
        print("  Embeddings already L2-normalized.")

    # Metadata
    patient_ids = adata.obs["patient_id"].values
    obs_names = adata.obs_names.tolist()
    patient_response = adata.obs.groupby("patient_id")["Response_3m"].first()

    return cell_embeds, patient_ids, obs_names, patient_response


def embed_texts_with_checkpoint(ckpt_path, all_queries):
    """Load checkpoint, embed text queries, return text_embeds tensor and logit_scale. Frees model."""
    from cellwhisperer.utils.model_io import load_cellwhisperer_model

    print(f"  Loading checkpoint: {os.path.basename(ckpt_path)}")
    pl_model, tokenizer, transcriptome_processor, image_processor = (
        load_cellwhisperer_model(model_path=ckpt_path, eval=True)
    )
    model = pl_model.model
    logit_scale = model.discriminator.temperature.exp().item()
    print(f"  logit_scale={logit_scale:.4f}")

    # Embed all text queries
    print(f"  Embedding {len(all_queries)} text queries...")
    text_embeds = model.embed_texts(all_queries, chunk_size=64)
    print(f"  Text embeddings shape: {text_embeds.shape}")

    # Verify normalization
    text_norms = torch.norm(text_embeds, dim=1)
    print(f"  Text embedding norms: min={text_norms.min():.6f}, max={text_norms.max():.6f}")

    # Detach and move to CPU
    text_embeds = text_embeds.detach().cpu()

    # Free model memory
    del pl_model, model, tokenizer, transcriptome_processor, image_processor
    gc.collect()
    torch.cuda.empty_cache()

    return text_embeds, logit_scale


def compute_cell_query_scores(cell_embeds, text_embeds, logit_scale):
    """Compute per-cell × per-query similarity scores. Returns (n_cells, n_queries) numpy array."""
    from cellwhisperer.utils.inference import score_left_vs_right

    scores, _ = score_left_vs_right(
        left_input=cell_embeds,       # (n_cells, embed_dim) tensor
        right_input=text_embeds,      # (n_queries, embed_dim) tensor
        logit_scale=logit_scale,
        model=None,                   # Not needed when both inputs are tensors
        average_mode=None,            # Per-cell scores
        grouping_keys=None,
        batch_size=128,
        score_norm_method=None,
    )
    # scores shape: (n_queries, n_cells)
    return scores.T.cpu().numpy()  # → (n_cells, n_queries)


def aggregate_patient_scores(scores_df, queries, patient_ids, agg_name):
    """Aggregate cell-level scores to patient level using the specified strategy."""
    scores_with_pid = scores_df[queries].copy()
    scores_with_pid["patient_id"] = patient_ids

    if agg_name == "mean":
        return scores_with_pid.groupby("patient_id")[queries].mean()
    elif agg_name == "max":
        return scores_with_pid.groupby("patient_id")[queries].max()
    elif agg_name == "p85":
        return scores_with_pid.groupby("patient_id")[queries].quantile(0.85)
    elif agg_name == "frac_high75":
        thresholds = scores_df[queries].quantile(0.75)
        return scores_with_pid.groupby("patient_id")[queries].apply(
            lambda g: (g > thresholds).mean()
        )
    else:
        raise ValueError(f"Unknown aggregation: {agg_name}")


def test_mechanism_multi_agg(scores_df, patient_ids, patient_response, queries, agg_names, bonferroni_n):
    """Test all queries × aggregations for one mechanism. Returns per-test results + summary."""
    or_pids = patient_response[patient_response == "OR"].index
    nr_pids = patient_response[patient_response == "NR"].index

    test_results = []
    for agg_name in agg_names:
        patient_scores = aggregate_patient_scores(scores_df, queries, patient_ids, agg_name)

        for query in queries:
            or_vals = patient_scores.loc[patient_scores.index.isin(or_pids), query].dropna()
            nr_vals = patient_scores.loc[patient_scores.index.isin(nr_pids), query].dropna()

            if len(or_vals) < 3 or len(nr_vals) < 3:
                continue

            stat, p_raw = mannwhitneyu(or_vals, nr_vals, alternative="two-sided")
            p_corrected = min(p_raw * bonferroni_n, 1.0)

            or_mean = or_vals.mean()
            nr_mean = nr_vals.mean()
            direction = "OR > NR" if or_mean > nr_mean else "NR > OR"
            effect_size = or_mean - nr_mean

            test_results.append({
                "agg": agg_name,
                "query": query,
                "p_raw": p_raw,
                "p_corrected": p_corrected,
                "direction": direction,
                "or_mean": or_mean,
                "nr_mean": nr_mean,
                "effect_size": effect_size,
                "n_or": len(or_vals),
                "n_nr": len(nr_vals),
            })

    if not test_results:
        return test_results, {"verified": False, "best_p_corrected": 1.0}

    # Direction consistency across ALL tests (all aggs × all queries)
    directions = [r["direction"] for r in test_results]
    majority_direction = max(set(directions), key=directions.count)
    direction_agreement = directions.count(majority_direction) / len(directions)

    # Best test (lowest corrected p-value)
    best = min(test_results, key=lambda r: r["p_corrected"])

    summary = {
        "best_agg": best["agg"],
        "best_query": best["query"],
        "best_p_raw": best["p_raw"],
        "best_p_corrected": best["p_corrected"],
        "best_direction": best["direction"],
        "best_effect_size": best["effect_size"],
        "majority_direction": majority_direction,
        "direction_agreement": direction_agreement,
        "n_significant_corrected": sum(1 for r in test_results if r["p_corrected"] < 0.05),
        "n_significant_raw": sum(1 for r in test_results if r["p_raw"] < 0.05),
        "n_tests": len(test_results),
    }

    return test_results, summary


def run_condition(ckpt_name, agg_strategy_name, scores_array, all_queries,
                  mechanisms, patient_ids, patient_response, obs_names):
    """Run one condition of the ablation (1 checkpoint × 1 agg strategy)."""
    agg_config = AGG_STRATEGIES[agg_strategy_name]
    agg_names = agg_config["aggs"]
    bonferroni_n = agg_config["bonferroni_n"]

    # Build scores DataFrame
    scores_df = pd.DataFrame(scores_array, index=obs_names, columns=all_queries)

    all_test_rows = []
    summary_rows = []

    for mid, mdata in mechanisms.items():
        queries = mdata["queries"]
        expected_dir = mdata["expected_direction"]

        test_results, summary = test_mechanism_multi_agg(
            scores_df, patient_ids, patient_response, queries, agg_names, bonferroni_n,
        )

        # Check direction match
        if expected_dir != "unknown" and test_results:
            direction_matches = summary["majority_direction"] == expected_dir
        else:
            direction_matches = None

        # Verified = significant + direction consistent + direction matches
        verified = (
            summary.get("best_p_corrected", 1.0) < 0.05
            and summary.get("direction_agreement", 0) >= 0.7
            and (direction_matches is True or direction_matches is None)
        )

        summary_row = {
            "mechanism_id": mid,
            "verbal_summary": mdata["summary"],
            "expected_direction": expected_dir,
            "verified": verified,
            "direction_matches_expected": direction_matches,
            **summary,
        }
        summary_rows.append(summary_row)

        for tr in test_results:
            all_test_rows.append({"mechanism_id": mid, **tr})

    return pd.DataFrame(summary_rows), pd.DataFrame(all_test_rows)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load queries
    with open(QUERIES_JSON) as f:
        mechanisms = json.load(f)
    print(f"Loaded queries for {len(mechanisms)} mechanisms")

    # Collect all unique queries, preserving order
    all_queries = []
    seen = set()
    for mid, mdata in mechanisms.items():
        for q in mdata["queries"]:
            if q not in seen:
                all_queries.append(q)
                seen.add(q)
    print(f"Total unique queries: {len(all_queries)}")

    # Load cell data (once, shared across all conditions)
    cell_embeds, patient_ids, obs_names, patient_response = load_cell_data()
    print(f"OR: {(patient_response == 'OR').sum()}, NR: {(patient_response == 'NR').sum()}")

    # Results collector
    ablation_summary = []

    # Loop over checkpoints (loaded sequentially)
    for ckpt_name, ckpt_path in CHECKPOINTS.items():
        print(f"\n{'='*60}")
        print(f"CHECKPOINT: {ckpt_name}")
        print(f"{'='*60}")

        if not os.path.exists(ckpt_path):
            print(f"  WARNING: Checkpoint not found at {ckpt_path}. Skipping.")
            continue

        # Embed texts with this checkpoint's text encoder
        text_embeds, logit_scale = embed_texts_with_checkpoint(ckpt_path, all_queries)

        # Compute per-cell × per-query scores using pre-computed cell embeds + this text encoder
        print(f"  Computing scores ({cell_embeds.shape[0]} cells × {len(all_queries)} queries)...")
        scores_array = compute_cell_query_scores(cell_embeds, text_embeds, logit_scale)
        print(f"  Scores shape: {scores_array.shape}")
        print(f"  Score stats: min={scores_array.min():.4f}, max={scores_array.max():.4f}, "
              f"mean={scores_array.mean():.4f}, std={scores_array.std():.4f}")

        # Free text embeds
        del text_embeds
        gc.collect()

        # Run both aggregation strategies
        for agg_name, agg_config in AGG_STRATEGIES.items():
            print(f"\n  --- Aggregation: {agg_name} (Bonferroni ÷{agg_config['bonferroni_n']}) ---")

            summary_df, tests_df = run_condition(
                ckpt_name, agg_name, scores_array, all_queries,
                mechanisms, patient_ids, patient_response, obs_names,
            )

            # Save per-condition results
            condition_tag = f"{ckpt_name}__{agg_name}"
            summary_df.to_csv(os.path.join(OUTPUT_DIR, f"{condition_tag}_verification.csv"), index=False)
            tests_df.to_csv(os.path.join(OUTPUT_DIR, f"{condition_tag}_all_tests.csv"), index=False)

            n_verified = summary_df["verified"].sum()
            n_total = len(summary_df)
            n_sig_raw = (tests_df["p_raw"] < 0.05).sum() if len(tests_df) > 0 else 0
            n_sig_corr = (tests_df["p_corrected"] < 0.05).sum() if len(tests_df) > 0 else 0

            print(f"  RESULT: {n_verified}/{n_total} mechanisms verified")
            print(f"  Sig tests (raw): {n_sig_raw}/{len(tests_df)}, (corrected): {n_sig_corr}/{len(tests_df)}")

            # Effect size diagnostic
            effect_sizes = tests_df["effect_size"].abs() if len(tests_df) > 0 else pd.Series(dtype=float)
            if len(effect_sizes) > 0:
                print(f"  Effect sizes: min={effect_sizes.min():.6f}, max={effect_sizes.max():.6f}, "
                      f"median={effect_sizes.median():.6f}")

            ablation_summary.append({
                "checkpoint": ckpt_name,
                "agg_strategy": agg_name,
                "bonferroni_n": agg_config["bonferroni_n"],
                "logit_scale": logit_scale,
                "n_mechanisms_verified": int(n_verified),
                "n_mechanisms_total": n_total,
                "verification_rate": n_verified / n_total if n_total > 0 else 0,
                "n_tests_sig_raw": int(n_sig_raw),
                "n_tests_sig_corrected": int(n_sig_corr),
                "n_tests_total": len(tests_df),
                "median_abs_effect_size": float(effect_sizes.median()) if len(effect_sizes) > 0 else 0,
                "max_abs_effect_size": float(effect_sizes.max()) if len(effect_sizes) > 0 else 0,
                "score_mean": float(scores_array.mean()),
                "score_std": float(scores_array.std()),
            })

        # Free scores array before loading next checkpoint
        del scores_array
        gc.collect()

    # Save ablation summary
    ablation_df = pd.DataFrame(ablation_summary)
    ablation_df.to_csv(os.path.join(OUTPUT_DIR, "ablation_summary.csv"), index=False)

    # Print final comparison
    print(f"\n{'='*60}")
    print("ABLATION SUMMARY")
    print(f"{'='*60}")
    print(ablation_df[["checkpoint", "agg_strategy", "n_mechanisms_verified",
                        "verification_rate", "median_abs_effect_size", "logit_scale"]].to_string(index=False))

    # Highlight best condition
    if len(ablation_df) > 0:
        best = ablation_df.loc[ablation_df["n_mechanisms_verified"].idxmax()]
        print(f"\nBest condition: {best['checkpoint']} + {best['agg_strategy']} "
              f"→ {int(best['n_mechanisms_verified'])}/{int(best['n_mechanisms_total'])} verified")


if __name__ == "__main__":
    main()
