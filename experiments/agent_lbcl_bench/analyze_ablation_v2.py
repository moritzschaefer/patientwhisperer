"""Quick analysis of ablation v2 results."""
import pandas as pd
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "step1v2_ablation_v2")
CHECKPOINTS = ["old_jointemb", "spatialwhisperer_v1", "best_cxg"]

# Check if MLP checkpoints are identical
sv1 = pd.read_csv(os.path.join(RESULTS_DIR, "spatialwhisperer_v1__mechanism_verification.csv"))
bcxg = pd.read_csv(os.path.join(RESULTS_DIR, "best_cxg__mechanism_verification.csv"))
print(f"spatialwhisperer_v1 == best_cxg? {sv1.equals(bcxg)}")
print()

# Ablation summary
summary = pd.read_csv(os.path.join(RESULTS_DIR, "ablation_summary.csv"))
print("=== ABLATION SUMMARY ===")
print(summary.to_string(index=False))
print()

# Compare mechanism verification across all 3
print("=== MECHANISM VERIFICATION (top 10 by raw p, all checkpoints) ===")
for name in CHECKPOINTS:
    df = pd.read_csv(os.path.join(RESULTS_DIR, f"{name}__mechanism_verification.csv"))
    n_ver = df["verified"].sum()
    near_sig = (df["best_p_corrected"] < 0.10).sum()
    print(f"\n--- {name}: {n_ver}/22 verified, {near_sig} with p_corr<0.10 ---")
    for _, r in df.sort_values("best_p_raw").head(10).iterrows():
        print(f"  {r['mechanism_id']} p_raw={r['best_p_raw']:.4f} p_corr={r['best_p_corrected']:.4f} "
              f"agg={r['best_agg']:12s} dir={r['best_direction']:10s} "
              f"match={str(r['direction_matches_expected']):5s} "
              f"{str(r['verbal_summary'])[:50]}")

# Compare original-style tests
print("\n\n=== ORIGINAL-STYLE TESTS (p<0.05) ===")
for name in CHECKPOINTS:
    df = pd.read_csv(os.path.join(RESULTS_DIR, f"{name}__original_style_tests.csv"))
    sig = df[df["p_raw"] < 0.05].sort_values("p_raw")
    print(f"\n--- {name}: {len(sig)} sig (p<0.05) / {len(df)} total ---")
    for _, r in sig.iterrows():
        print(f"  {r['agg']:12s} p={r['p_raw']:.4f} {r['direction']:10s} "
              f"ratio={str(r['is_ratio']):5s} {r['feature'][:55]}")

# Direction agreement comparison
print("\n\n=== DIRECTION AGREEMENT ===")
for name in CHECKPOINTS:
    df = pd.read_csv(os.path.join(RESULTS_DIR, f"{name}__mechanism_verification.csv"))
    match_col = df["direction_matches_expected"].dropna()
    n_match = match_col.sum()
    print(f"{name}: {n_match}/{len(match_col)} mechanisms match expected direction "
          f"({100*n_match/len(match_col):.0f}%)")
