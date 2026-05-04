"""
Step 1v2: Deterministic mechanism verification with pre-registered queries.

For each mechanism, runs 10 pre-registered queries through CellWhisperer,
aggregates per-patient (mean), tests OR vs NR with two-sided Mann-Whitney U,
applies Bonferroni correction (÷10), and checks direction consistency.

No agent involvement — purely deterministic statistical testing.

Input: data/step1v2_queries.json (from step1v2_generate_queries.py)
Output: results/step1v2/step1v2_verification.csv
        results/step1v2/step1v2_all_tests.csv (full query-level results)
"""
import pyarrow  # MUST be first import on Sherlock

import json
import os
import warnings

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

warnings.filterwarnings("ignore")

# -- Paths (Sherlock) --
H5AD_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "car_t_ignite_prelim_analysis",
    "cellxgene_B_Product_lowburden.light.h5ad",
)
CKPT_PATH = "/oak/stanford/groups/zinaida/moritzs/cellwhisperer/results/models/spatialwhisperer/v1.ckpt"
QUERIES_JSON = os.path.join(os.path.dirname(__file__), "data", "step1v2_queries.json")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results", "step1v2")

BONFERRONI_N = 10  # Number of queries per mechanism


def load_model():
    from cellwhisperer.utils.model_io import load_cellwhisperer_model
    print("Loading CellWhisperer model...")
    pl_model, tokenizer, transcriptome_processor, image_processor = (
        load_cellwhisperer_model(model_path=CKPT_PATH, eval=True)
    )
    model = pl_model.model
    logit_scale = model.discriminator.temperature.exp()
    print(f"Model loaded. logit_scale={logit_scale.item():.4f}")
    return model, logit_scale


def score_queries(adata, queries, model, logit_scale):
    """Score all queries against all cells. Returns DataFrame (n_cells x n_queries)."""
    from cellwhisperer.utils.inference import score_left_vs_right

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
    return pd.DataFrame(
        scores.T.cpu().numpy(),
        index=adata.obs_names,
        columns=queries,
    )


def test_mechanism(scores_df, patient_response, queries, n_bonferroni=BONFERRONI_N):
    """Test all queries for one mechanism. Returns per-query results + summary."""
    or_pids = patient_response[patient_response == "OR"].index
    nr_pids = patient_response[patient_response == "NR"].index

    # Patient-level mean aggregation
    patient_scores = scores_df.groupby("patient_id")[queries].mean()

    query_results = []
    for query in queries:
        or_vals = patient_scores.loc[patient_scores.index.isin(or_pids), query].dropna()
        nr_vals = patient_scores.loc[patient_scores.index.isin(nr_pids), query].dropna()

        stat, p_raw = mannwhitneyu(or_vals, nr_vals, alternative="two-sided")
        p_corrected = min(p_raw * n_bonferroni, 1.0)

        or_mean = or_vals.mean()
        nr_mean = nr_vals.mean()
        direction = "OR > NR" if or_mean > nr_mean else "NR > OR"
        effect_size = or_mean - nr_mean

        query_results.append({
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

    # Direction consistency: what fraction of queries agree on direction?
    directions = [r["direction"] for r in query_results]
    majority_direction = max(set(directions), key=directions.count)
    direction_agreement = directions.count(majority_direction) / len(directions)

    # Best query (lowest corrected p-value)
    best = min(query_results, key=lambda r: r["p_corrected"])

    return query_results, {
        "best_query": best["query"],
        "best_p_raw": best["p_raw"],
        "best_p_corrected": best["p_corrected"],
        "best_direction": best["direction"],
        "best_effect_size": best["effect_size"],
        "majority_direction": majority_direction,
        "direction_agreement": direction_agreement,
        "n_significant_corrected": sum(1 for r in query_results if r["p_corrected"] < 0.05),
        "n_significant_raw": sum(1 for r in query_results if r["p_raw"] < 0.05),
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load pre-registered queries
    with open(QUERIES_JSON) as f:
        mechanisms = json.load(f)
    print(f"Loaded queries for {len(mechanisms)} mechanisms")

    # Collect ALL queries across all mechanisms for a single CellWhisperer pass
    all_queries = []
    query_to_mechanism = {}
    for mid, mdata in mechanisms.items():
        for q in mdata["queries"]:
            if q not in query_to_mechanism:  # Deduplicate
                all_queries.append(q)
            query_to_mechanism.setdefault(q, []).append(mid)
    print(f"Total unique queries: {len(all_queries)}")

    # Load data
    print(f"Loading h5ad from {H5AD_PATH}")
    adata = ad.read_h5ad(H5AD_PATH)
    print(f"  {adata.n_obs} cells, {adata.n_vars} genes")

    # Filter to OR/NR
    mask = adata.obs["Response_3m"].isin(["OR", "NR"])
    adata = adata[mask].copy()
    print(f"  After OR/NR filter: {adata.n_obs} cells")

    # Load model and score ALL queries in one pass
    model, logit_scale = load_model()
    print(f"\nScoring {len(all_queries)} queries against {adata.n_obs} cells...")
    scores_df = score_queries(adata, all_queries, model, logit_scale)
    scores_df["patient_id"] = adata.obs["patient_id"].values

    # Patient-level response mapping
    patient_response = (
        adata.obs.groupby("patient_id")["Response_3m"].first()
    )
    print(f"  OR: {(patient_response == 'OR').sum()}, NR: {(patient_response == 'NR').sum()}")

    # Test each mechanism
    all_test_rows = []
    summary_rows = []

    for mid, mdata in mechanisms.items():
        queries = mdata["queries"]
        expected_dir = mdata["expected_direction"]

        query_results, summary = test_mechanism(
            scores_df, patient_response, queries
        )

        # Check if observed direction matches expected
        if expected_dir != "unknown":
            direction_matches = summary["majority_direction"] == expected_dir
        else:
            direction_matches = None  # Can't assess

        # Verified = significant after correction AND direction consistent AND direction matches expected
        verified = (
            summary["best_p_corrected"] < 0.05
            and summary["direction_agreement"] >= 0.7  # ≥7/10 queries agree
            and (direction_matches is True or direction_matches is None)
        )

        summary_row = {
            "mechanism_id": mid,
            "verbal_summary": mdata["summary"],
            "expected_direction": expected_dir,
            "verified": verified,
            **summary,
            "direction_matches_expected": direction_matches,
        }
        summary_rows.append(summary_row)

        # Store per-query results
        for qr in query_results:
            all_test_rows.append({"mechanism_id": mid, **qr})

        status = "VERIFIED" if verified else "not verified"
        print(f"\n{mid} [{status}]: {mdata['summary'][:60]}...")
        print(f"  Best: p_corr={summary['best_p_corrected']:.4f}, "
              f"dir={summary['best_direction']}, "
              f"agreement={summary['direction_agreement']:.0%}")
        print(f"  Expected: {expected_dir}, matches: {direction_matches}")
        print(f"  Sig queries (raw): {summary['n_significant_raw']}/10, "
              f"(corrected): {summary['n_significant_corrected']}/10")

    # Save results
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "step1v2_verification.csv"), index=False)

    tests_df = pd.DataFrame(all_test_rows)
    tests_df.to_csv(os.path.join(OUTPUT_DIR, "step1v2_all_tests.csv"), index=False)

    # Print summary
    n_verified = summary_df["verified"].sum()
    n_total = len(summary_df)
    print(f"\n{'='*60}")
    print(f"SUMMARY: {n_verified}/{n_total} mechanisms verified ({n_verified/n_total:.0%})")
    print(f"{'='*60}")
    for _, row in summary_df.iterrows():
        v = "+" if row["verified"] else "-"
        print(f"  [{v}] {row['mechanism_id']}: p_corr={row['best_p_corrected']:.4f}, "
              f"dir_agree={row['direction_agreement']:.0%}, "
              f"match={row['direction_matches_expected']}")


if __name__ == "__main__":
    main()
