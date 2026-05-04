"""
Cluster and deduplicate agent-generated mechanisms from Level 3 per-patient analysis.

Steps:
1. Load mechanism descriptions + LBCL-Bench
2. Embed using TF-IDF (no extra deps)
3. Cluster with Agglomerative Clustering
4. Label clusters via LLM
5. Match clusters to LBCL-Bench via LLM
6. Classify novel clusters
7. Export results
"""

import csv
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

EXPERIMENT_DIR = Path(__file__).parent.parent
DATA_DIR = EXPERIMENT_DIR / "data"
RESULTS_DIR = EXPERIMENT_DIR / "results" / "step3_evaluation"
OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

API_URL = "https://openrouter.ai/api/v1/chat/completions"
API_KEY = os.environ.get("OPENROUTER_API_KEY", "").split("\n")[0].strip()
LLM_MODEL = "google/gemini-2.0-flash-001"  # fast + cheap for labeling


def llm_call(prompt, system="You are a concise scientific assistant.", max_tokens=500):
    """Call OpenRouter API (OpenAI-compatible)."""
    resp = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": 0,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def load_data():
    """Load agent mechanisms and LBCL-Bench."""
    mechs = pd.read_csv(RESULTS_DIR / "all_patient_mechanisms.csv")
    bench = pd.read_csv(DATA_DIR / "lbcl_bench_filtered.csv")
    print(
        f"Loaded {len(mechs)} mechanism instances from {mechs['patient_id'].nunique()} patients"
    )
    print(f"Loaded {len(bench)} LBCL-Bench mechanisms")
    return mechs, bench


def embed_tfidf(texts):
    """Embed texts using TF-IDF with n-gram features."""
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=5000,
        stop_words="english",
        sublinear_tf=True,
    )
    embeddings = vectorizer.fit_transform(texts)
    return embeddings, vectorizer


def find_optimal_clusters(embeddings, min_k=20, max_k=80, step=5):
    """Find optimal number of clusters via silhouette score."""
    dense = embeddings.toarray() if hasattr(embeddings, "toarray") else embeddings
    results = []
    for k in range(min_k, max_k + 1, step):
        clustering = AgglomerativeClustering(
            n_clusters=k, metric="cosine", linkage="average"
        )
        labels = clustering.fit_predict(dense)
        score = silhouette_score(dense, labels, metric="cosine")
        results.append((k, score))
        print(f"  k={k:3d}  silhouette={score:.4f}")
    best_k, best_score = max(results, key=lambda x: x[1])
    print(f"Best: k={best_k} (silhouette={best_score:.4f})")
    return best_k, results


def cluster_mechanisms(embeddings, n_clusters):
    """Cluster with Agglomerative Clustering."""
    dense = embeddings.toarray() if hasattr(embeddings, "toarray") else embeddings
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters, metric="cosine", linkage="average"
    )
    labels = clustering.fit_predict(dense)
    return labels


def get_cluster_exemplars(mechs_df, labels, embeddings, n_exemplars=5):
    """For each cluster, find the most central mechanism descriptions."""
    dense = embeddings.toarray() if hasattr(embeddings, "toarray") else embeddings
    clusters = defaultdict(list)
    for idx, label in enumerate(labels):
        clusters[label].append(idx)

    exemplars = {}
    for cluster_id, indices in clusters.items():
        cluster_vecs = dense[indices]
        centroid = cluster_vecs.mean(axis=0, keepdims=True)
        sims = cosine_similarity(centroid, cluster_vecs)[0]
        top_indices = np.argsort(sims)[-n_exemplars:][::-1]
        exemplars[cluster_id] = [
            mechs_df.iloc[indices[i]]["mechanism"] for i in top_indices
        ]

    return exemplars


def label_clusters_llm(exemplars, cluster_stats):
    """Use LLM to label each cluster."""
    labels = {}
    for cluster_id in sorted(exemplars.keys()):
        examples = exemplars[cluster_id]
        n_patients = cluster_stats[cluster_id]["n_patients"]
        examples_text = "\n".join(f"  - {e}" for e in examples[:5])

        prompt = f"""These are {min(5, len(examples))} representative examples from a cluster of {n_patients} mechanism descriptions identified by AI agents analyzing CAR T cell therapy patient data:

{examples_text}

Give a concise label (5-12 words) that captures the shared biological mechanism. Return ONLY the label, nothing else."""

        label = llm_call(prompt, max_tokens=50)
        labels[cluster_id] = label.strip('"').strip("'").strip(".")
        print(
            f"  Cluster {cluster_id:2d} ({n_patients:2d} patients): {labels[cluster_id]}"
        )
        time.sleep(0.3)  # rate limit

    return labels


def match_clusters_to_bench(cluster_labels, exemplars, bench_df):
    """Use LLM to match each cluster to LBCL-Bench mechanisms."""
    bench_summaries = bench_df["verbal_summary"].tolist()
    bench_ids = bench_df["mechanism_id"].tolist()
    bench_list = "\n".join(
        f"  {mid}: {summ}" for mid, summ in zip(bench_ids, bench_summaries)
    )

    matches = {}
    for cluster_id in sorted(cluster_labels.keys()):
        label = cluster_labels[cluster_id]
        examples = exemplars[cluster_id][:3]
        examples_text = "\n".join(f"  - {e}" for e in examples)

        prompt = f"""Agent-identified mechanism cluster: "{label}"
Example descriptions from this cluster:
{examples_text}

LBCL-Bench mechanisms (established CAR T resistance/response mechanisms):
{bench_list}

Does this cluster match any LBCL-Bench mechanism? Reply in this exact format:
MATCH: <mechanism_id> (or NONE)
CONFIDENCE: high/medium/low
REASONING: <one sentence>"""

        response = llm_call(prompt, max_tokens=150)
        match_id = "NONE"
        confidence = "low"
        reasoning = ""
        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("MATCH:"):
                match_id = line.split(":", 1)[1].strip()
                if match_id.startswith("NONE") or match_id == "None":
                    match_id = "NONE"
                else:
                    match_id = match_id.split()[0].strip()  # take first word (the ID)
            elif line.startswith("CONFIDENCE:"):
                confidence = line.split(":", 1)[1].strip().lower()
            elif line.startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()

        matches[cluster_id] = {
            "bench_match": match_id,
            "match_confidence": confidence,
            "match_reasoning": reasoning,
        }
        status = f"→ {match_id}" if match_id != "NONE" else "→ NOVEL"
        print(f"  Cluster {cluster_id:2d} ({label}): {status} [{confidence}]")
        time.sleep(0.3)

    return matches


def classify_novel_clusters(cluster_labels, exemplars, matches):
    """For clusters not matching LBCL-Bench, classify as known-in-literature vs novel."""
    novelty = {}
    for cluster_id, match_info in matches.items():
        if match_info["bench_match"] != "NONE":
            novelty[cluster_id] = "bench_match"
            continue

        label = cluster_labels[cluster_id]
        examples = exemplars[cluster_id][:3]
        examples_text = "\n".join(f"  - {e}" for e in examples)

        prompt = f"""This mechanism was identified by an AI agent analyzing CAR T cell therapy patient data, but does NOT match any established benchmark mechanism:

Cluster label: "{label}"
Examples:
{examples_text}

Is this mechanism well-established in the CAR T cell therapy or cancer immunology literature? 
Reply in this exact format:
STATUS: KNOWN_IN_LITERATURE or POTENTIALLY_NOVEL
REASONING: <one sentence>"""

        response = llm_call(prompt, max_tokens=100)
        status = "potentially_novel"
        reasoning = ""
        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("STATUS:"):
                raw = line.split(":", 1)[1].strip().upper()
                if "KNOWN" in raw:
                    status = "known_in_literature"
                else:
                    status = "potentially_novel"
            elif line.startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()

        novelty[cluster_id] = status
        print(f"  Cluster {cluster_id:2d} ({label}): {status}")
        time.sleep(0.3)

    return novelty


def main():
    print("=" * 60)
    print("MECHANISM CLUSTERING AND NOVELTY ANALYSIS")
    print("=" * 60)

    # Step 1: Load data
    print("\n[1/7] Loading data...")
    mechs, bench = load_data()

    # Step 2: Embed
    print("\n[2/7] Embedding mechanism descriptions (TF-IDF)...")
    all_texts = mechs["mechanism"].tolist()
    embeddings, vectorizer = embed_tfidf(all_texts)
    print(f"  TF-IDF matrix: {embeddings.shape}")

    # Step 3: Find optimal clusters
    print("\n[3/7] Finding optimal cluster count...")
    best_k, sweep_results = find_optimal_clusters(
        embeddings, min_k=15, max_k=60, step=5
    )
    # Silhouette monotonically increases with k for TF-IDF — cap at 40 for interpretability
    best_k = min(best_k, 40)
    print(f"Using k={best_k} (capped for interpretability)")

    # Step 4: Cluster
    print(f"\n[4/7] Clustering with k={best_k}...")
    labels = cluster_mechanisms(embeddings, best_k)
    mechs["cluster"] = labels

    # Compute cluster statistics
    cluster_stats = {}
    for cluster_id in range(best_k):
        mask = mechs["cluster"] == cluster_id
        cluster_mechs = mechs[mask]
        cluster_stats[cluster_id] = {
            "n_instances": len(cluster_mechs),
            "n_patients": cluster_mechs["patient_id"].nunique(),
            "n_OR": cluster_mechs[cluster_mechs["response"] == "OR"][
                "patient_id"
            ].nunique(),
            "n_NR": cluster_mechs[cluster_mechs["response"] == "NR"][
                "patient_id"
            ].nunique(),
            "directions": dict(Counter(cluster_mechs["direction"])),
            "confidences": dict(Counter(cluster_mechs["confidence"])),
        }

    # Get exemplars
    exemplars = get_cluster_exemplars(mechs, labels, embeddings)

    # Step 5: Label clusters via LLM
    print(f"\n[5/7] Labeling {best_k} clusters via LLM...")
    cluster_labels = label_clusters_llm(exemplars, cluster_stats)

    # Step 6: Match to LBCL-Bench
    print(f"\n[6/7] Matching clusters to LBCL-Bench...")
    bench_matches = match_clusters_to_bench(cluster_labels, exemplars, bench)

    # Step 7: Classify novel clusters
    print(f"\n[7/7] Classifying novel clusters...")
    novelty = classify_novel_clusters(cluster_labels, exemplars, bench_matches)

    # ========================
    # Export results
    # ========================
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    results_rows = []
    for cluster_id in sorted(cluster_labels.keys()):
        row = {
            "cluster_id": cluster_id,
            "label": cluster_labels[cluster_id],
            "n_instances": cluster_stats[cluster_id]["n_instances"],
            "n_patients": cluster_stats[cluster_id]["n_patients"],
            "n_OR": cluster_stats[cluster_id]["n_OR"],
            "n_NR": cluster_stats[cluster_id]["n_NR"],
            "dominant_direction": max(
                cluster_stats[cluster_id]["directions"],
                key=cluster_stats[cluster_id]["directions"].get,
            ),
            "bench_match": bench_matches[cluster_id]["bench_match"],
            "match_confidence": bench_matches[cluster_id]["match_confidence"],
            "match_reasoning": bench_matches[cluster_id]["match_reasoning"],
            "novelty_category": novelty[cluster_id],
            "exemplars": " | ".join(exemplars[cluster_id][:3]),
        }
        results_rows.append(row)

    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(OUTPUT_DIR / "mechanism_clusters.csv", index=False)

    # Annotated mechanisms
    mechs["cluster_label"] = mechs["cluster"].map(cluster_labels)
    mechs["bench_match"] = mechs["cluster"].map(
        lambda c: bench_matches[c]["bench_match"]
    )
    mechs["novelty_category"] = mechs["cluster"].map(novelty)
    mechs.to_csv(OUTPUT_DIR / "all_mechanisms_annotated.csv", index=False)

    # Summary
    n_bench = sum(1 for v in novelty.values() if v == "bench_match")
    n_known = sum(1 for v in novelty.values() if v == "known_in_literature")
    n_novel = sum(1 for v in novelty.values() if v == "potentially_novel")

    summary = {
        "total_mechanism_instances": len(mechs),
        "total_patients": mechs["patient_id"].nunique(),
        "total_clusters": best_k,
        "clusters_matching_bench": n_bench,
        "clusters_known_in_literature": n_known,
        "clusters_potentially_novel": n_novel,
        "silhouette_score": float(
            silhouette_score(embeddings.toarray(), labels, metric="cosine")
        ),
        "sweep_results": [(k, float(s)) for k, s in sweep_results],
    }

    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nTotal distinct mechanism types: {best_k}")
    print(f"  Matching LBCL-Bench:        {n_bench}")
    print(f"  Known in literature:         {n_known}")
    print(f"  Potentially novel:           {n_novel}")
    print(f"\nResults written to {OUTPUT_DIR}")
    print(f"  mechanism_clusters.csv       — per-cluster summary")
    print(f"  all_mechanisms_annotated.csv — per-instance with cluster labels")
    print(f"  summary.json                 — summary statistics")


if __name__ == "__main__":
    main()
