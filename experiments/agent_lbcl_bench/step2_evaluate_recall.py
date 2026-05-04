"""
Step 2: Evaluate Discovery Recall against LBCL-Bench

Uses LLM-based semantic matching to determine whether each LBCL-Bench mechanism
was rediscovered by the open-ended discovery agent.

Usage:
    pixi run --no-progress python step2_evaluate_recall.py
"""
import json
import os
import sys

import pandas as pd

BENCH_CSV = "data/lbcl_bench_filtered.csv"
DISCOVERIES_JSON = "results/step2/discoveries.json"
RESULTS_DIR = "results/step2"
RECALL_CSV = os.path.join(RESULTS_DIR, "recall_evaluation.csv")


def llm_match(bench_mechanism, discoveries, model="anthropic/claude-sonnet-4-20250514"):
    """Use an LLM to judge whether any discovery matches a benchmark mechanism."""
    import litellm

    discoveries_text = "\n".join(
        f"- {d.get('mechanism', 'unknown')}: {d.get('reasoning', '')[:200]}"
        for d in discoveries
    )

    prompt = f"""You are evaluating whether a known biological mechanism was rediscovered by an AI agent.

KNOWN MECHANISM (from literature):
ID: {bench_mechanism['mechanism_id']}
Summary: {bench_mechanism['verbal_summary']}
Description: {str(bench_mechanism['consolidated_description'])[:1000]}

AGENT'S DISCOVERIES:
{discoveries_text}

Does any of the agent's discoveries match the known mechanism? A match means the agent identified 
the same or very similar biological concept, even if phrased differently.

Respond with EXACTLY this JSON (no other text):
{{"match": true/false, "matched_discovery": "text of matching discovery or null", "confidence": "high/medium/low", "reasoning": "brief explanation"}}"""

    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    
    content = response.choices[0].message.content.strip()
    # Parse JSON from response
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Try extracting JSON block
        import re
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {"match": False, "reasoning": f"Parse error: {content[:200]}"}


def main():
    bench = pd.read_csv(BENCH_CSV)
    
    if not os.path.exists(DISCOVERIES_JSON):
        print(f"Discoveries file not found: {DISCOVERIES_JSON}")
        print("Run step2_run_discovery.py first.")
        sys.exit(1)

    with open(DISCOVERIES_JSON) as f:
        discovery_data = json.load(f)
    
    discoveries = discovery_data.get("discoveries", [])
    print(f"Loaded {len(bench)} benchmark mechanisms and {len(discoveries)} discoveries")

    results = []
    for _, mech in bench.iterrows():
        mid = mech["mechanism_id"]
        print(f"Evaluating {mid}: {str(mech['verbal_summary'])[:60]}...", end=" ")
        
        match_result = llm_match(mech.to_dict(), discoveries)
        matched = match_result.get("match", False)
        print("MATCH" if matched else "no match")
        
        results.append({
            "mechanism_id": mid,
            "verbal_summary": mech["verbal_summary"],
            "category": mech["category"],
            "detectable_with_infusion_product": mech["detectable_with_infusion_product"],
            "matched": matched,
            "matched_discovery": match_result.get("matched_discovery", ""),
            "confidence": match_result.get("confidence", ""),
            "reasoning": match_result.get("reasoning", ""),
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(RECALL_CSV, index=False)

    # Summary statistics
    total = len(results_df)
    matched = results_df["matched"].sum()
    ip_mask = results_df["detectable_with_infusion_product"]
    ip_total = ip_mask.sum()
    ip_matched = results_df.loc[ip_mask, "matched"].sum()

    print(f"\n=== Recall Summary ===")
    print(f"Overall: {matched}/{total} = {matched/total:.1%}")
    print(f"Infusion Product detectable: {ip_matched}/{ip_total} = {ip_matched/ip_total:.1%}" if ip_total > 0 else "")
    print(f"\nSaved to {RECALL_CSV}")


if __name__ == "__main__":
    main()
