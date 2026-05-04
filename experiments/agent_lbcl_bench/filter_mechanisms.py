"""
Phase 0: Filter LBCL-Bench mechanisms to those detectable with available data.

v2 exclusions (confirmed by Wael):
- Malformed entries (M005, M022, M026, M034)
- NotLBCL mechanisms (M009, M023)
- Engineering-only (M004, M007, M027, M031, M032)
- Methodological / not a biological mechanism (M028)
- Not detectable from infusion product scRNA-seq (M010)

v3 exclusions (Wael expert review, 2026-03-24):
- Invalidated by expert review (M001, M003, M008, M014, M016, M018, M024, M029)
- M033 re-included per Wael's recommendation (PSI also correlates with response)

Adds detectability columns based on category tags.
"""
import pandas as pd

INPUT_CSV = "/home/moritz/Projects/PatientWhisperer/KnownMechanisms/consolidated_mechanisms_cleaned.csv"
OUTPUT_CSV = "data/lbcl_bench_filtered.csv"

EXCLUDE_IDS = {
    # Malformed entries (bad verbal_summary / placeholder rows)
    "M005", "M022", "M026", "M034",
    # NotLBCL (not relevant to LBCL CAR T therapy)
    "M009", "M023",
    # Engineering-only (not detectable from patient observational data)
    "M004", "M007", "M032",
    "M027",  # TGFBR2 deletion in myeloma CRISPR screen (Korell et al.)
    "M031",  # MED12 deficiency — CRISPR KO study (Freitas) + c-Jun OE (Lynn)
    # Methodological (not a biological mechanism)
    "M028",
    # Not detectable from infusion product scRNA-seq (basic DC biology review)
    "M010",
    # Invalidated by expert review (Wael Gamal, 2026-03-24)
    "M001",  # Dean et al. not about exhausted T cells; Deng et al. focuses CD8 not CD4
    "M003",  # Citations not related to LBCL patient response
    "M008",  # Verbal summary overestimates TGFb findings
    "M014",  # Reference not related to STAT5 signaling
    "M016",  # Barras et al. unrelated; verbal summary misleading
    "M018",  # Reference not related to monocytes
    "M024",  # Verbal summary does not describe research findings
    "M029",  # Citations not related to LBCL patient response
    # NOTE: M033 excluded in v2 but re-included per Wael (PSI correlates with response)
    # NOTE: M036 (exhaustion), M037 (antigen escape), M038 (senescence) added in v4 — included
}

df = pd.read_csv(INPUT_CSV)
print(f"Loaded {len(df)} mechanisms")

df = df[~df["mechanism_id"].isin(EXCLUDE_IDS)].copy()
print(f"After exclusions: {len(df)} mechanisms (removed {len(EXCLUDE_IDS)})")

# Add detectability columns based on category tags
df["detectable_with_infusion_product"] = df["category"].str.contains("Infusion Product", na=False)
df["detectable_with_tme"] = df["category"].str.contains("Spatial/TME", na=False)
df["detectable_with_clinical"] = df["category"].str.contains("Clinical/Real-world", na=False)

# Summary
print("\nDetectability summary:")
print(f"  Infusion Product: {df['detectable_with_infusion_product'].sum()}")
print(f"  TME/Spatial:      {df['detectable_with_tme'].sum()}")
print(f"  Clinical:         {df['detectable_with_clinical'].sum()}")
print()

for _, row in df.iterrows():
    flags = []
    if row["detectable_with_infusion_product"]:
        flags.append("IP")
    if row["detectable_with_tme"]:
        flags.append("TME")
    if row["detectable_with_clinical"]:
        flags.append("Clin")
    print(f"  {row['mechanism_id']}: [{','.join(flags)}] {str(row['verbal_summary'])[:70]}")

df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved to {OUTPUT_CSV}")
