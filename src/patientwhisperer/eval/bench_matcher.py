"""
LLM-based benchmark mechanism matching with deterministic per-pair caching.

Matches agent-discovered mechanisms against LBCL-Bench entries using an LLM judge.
Each (bench_description_hash, agent_mechanism_string) pair is cached permanently,
so identical inputs always produce identical outputs regardless of when evaluation runs.

Usage as library:
    from patientwhisperer.eval.bench_matcher import BenchMatcher
    matcher = BenchMatcher(cache_dir="path/to/cache")
    matches = matcher.match_mechanism(bench_mech, patient_results)

Usage as CLI:
    python -m patientwhisperer.eval.bench_matcher \
        --bench-csv data/lbcl_bench_filtered.csv \
        --patient-dir results/step3_per_patient \
        --output-dir results/step3_evaluation
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
from collections import Counter


def _hash_bench(verbal_summary: str, direction: str) -> str:
    """Stable hash of the benchmark mechanism identity."""
    key = f"{verbal_summary.strip()}|{direction}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _hash_pair(bench_hash: str, agent_mechanism: str) -> str:
    """Stable hash for a (bench, agent_mechanism) pair."""
    key = f"{bench_hash}|{agent_mechanism.strip()}"
    return hashlib.sha256(key.encode()).hexdigest()[:20]


def _detect_llm_cmd():
    """Detect available LLM CLI (claude or opencode via apptainer)."""
    if shutil.which("claude"):
        return ["claude"]
    if shutil.which("apptainer"):
        return ["apptainer", "run", "docker://openeuler/opencode"]
    if shutil.which("opencode"):
        return ["opencode"]
    raise RuntimeError("No LLM CLI found (claude, opencode, or apptainer)")


def _extract_json(stdout: str) -> dict | None:
    """Extract JSON from LLM output (handles NDJSON stream and raw JSON)."""
    # Try NDJSON (opencode/claude stream-json format)
    full_text = ""
    for line in stdout.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
            if event.get("type") == "text":
                full_text += event.get("part", {}).get("text", "")
        except json.JSONDecodeError:
            full_text += line

    search_text = full_text if full_text else stdout

    # Try ```json ... ``` block
    m = re.search(r"```json\s*\n(.*?)\n\s*```", search_text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Try raw JSON object
    m = re.search(r'\{"matches"\s*:\s*\{.*?\}\s*\}', search_text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass

    # Try parsing entire output
    try:
        return json.loads(search_text.strip())
    except json.JSONDecodeError:
        pass

    return None


def _infer_direction(verbal_summary: str) -> str:
    """Infer expected direction from benchmark verbal_summary."""
    vs = verbal_summary.lower()
    pro_response = [
        "predict response", "complete responder", "durable", "marks effective",
        "enhance", "superior", "prolonged", "associates with response",
        "mediates tumor control", "associate with complete",
    ]
    pro_resistance = [
        "non-response", "non-responder", "poor", "negatively",
        "inferior", "exclusion", "contamination", "resistance",
        "predict.*resistance",
    ]
    if any(w in vs for w in pro_response):
        return "pro-response"
    if any(w in vs for w in pro_resistance):
        return "pro-resistance"
    return "unclear"


class BenchMatcher:
    """Deterministic LLM-based benchmark mechanism matcher with per-pair caching."""

    def __init__(self, cache_dir: str, batch_size: int = 15, timeout: int = 180):
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.timeout = timeout
        self.llm_cmd = _detect_llm_cmd()
        os.makedirs(cache_dir, exist_ok=True)

    def _cache_path(self, pair_hash: str) -> str:
        # Two-level directory to avoid huge flat dirs
        return os.path.join(self.cache_dir, pair_hash[:2], f"{pair_hash}.json")

    def _load_cached(self, pair_hash: str) -> dict | None:
        path = self._cache_path(pair_hash)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    def _save_cached(self, pair_hash: str, result: dict):
        path = self._cache_path(pair_hash)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(result, f)

    def _build_prompt(self, bench_mech: dict, patient_mechanisms: list[tuple[str, str, str, str]]) -> str:
        """Build matching prompt.

        patient_mechanisms: list of (patient_id, mechanism_text, direction, data_source)
        """
        direction = bench_mech.get("_direction", "either")
        patient_sections = []
        for pid, mech_text, mech_dir, data_src in patient_mechanisms:
            patient_sections.append(f"[{pid}] [{mech_dir}] [src:{data_src}] {mech_text}")

        patients_text = "\n".join(patient_sections)

        return (
            "You are evaluating whether patient-level analyses rediscovered a known biological mechanism.\n\n"
            "KNOWN MECHANISM:\n"
            f'  "{bench_mech["verbal_summary"]}"\n'
            f"  Expected direction: {direction}\n\n"
            "PATIENT FINDINGS:\n"
            f"{patients_text}\n\n"
            "TASK: For each finding, determine if it matches the SPECIFIC mechanism above. Be STRICT:\n"
            "- The finding must describe the SAME specific biological mechanism\n"
            "- If the mechanism names a SPECIFIC gene, the finding must reference that gene\n"
            "- Direction must be consistent\n"
            "- When in doubt, do NOT match\n\n"
            "Respond with ONLY a JSON object. Include ONLY findings that match:\n"
            '{"matches": {"patient_id": {"finding": "brief matched finding", "data_source": "infusion|spatial|both"}, ...}}\n'
            'If NO findings match: {"matches": {}}'
        )

    def _call_llm(self, prompt: str, max_retries: int = 2) -> dict | None:
        """Call LLM CLI and extract JSON response."""
        if self.llm_cmd[0] == "claude":
            cmd = self.llm_cmd + [
                "-p", prompt,
                "--output-format", "stream-json",
                "--max-turns", "1",
                "--allowedTools", "",
                "--dangerously-skip-permissions",
            ]
        else:
            cmd = self.llm_cmd + ["run", "--format", "json", prompt]

        for attempt in range(max_retries + 1):
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=self.timeout
                )
                parsed = _extract_json(result.stdout)
                if parsed and "matches" in parsed:
                    return parsed["matches"]
            except subprocess.TimeoutExpired:
                pass
            except Exception as e:
                print(f"  LLM error (attempt {attempt + 1}): {e}")
        return None

    def match_mechanism(
        self,
        bench_mech: dict,
        patient_results: dict[str, dict],
    ) -> dict[str, dict]:
        """Match one benchmark mechanism against all patients.

        Returns dict of {patient_id: {"finding": ..., "data_source": ...}}.
        Uses per-pair cache: only uncached (bench, agent_mechanism) pairs are sent to LLM.
        """
        bench_hash = _hash_bench(
            bench_mech["verbal_summary"],
            bench_mech.get("_direction", _infer_direction(bench_mech["verbal_summary"])),
        )
        bench_mech["_direction"] = bench_mech.get(
            "_direction", _infer_direction(bench_mech["verbal_summary"])
        )

        all_matches = {}
        uncached_items = []  # (pid, mech_text, direction, data_source, pair_hash)

        # Check cache for every (bench, agent_mechanism) pair
        for pid, presult in patient_results.items():
            for m in presult.get("mechanisms_identified", []):
                mech_text = m.get("mechanism", "")
                pair_hash = _hash_pair(bench_hash, mech_text)
                cached = self._load_cached(pair_hash)

                if cached is not None:
                    if cached.get("is_match"):
                        all_matches[pid] = cached["match_info"]
                else:
                    uncached_items.append((
                        pid, mech_text,
                        m.get("direction", "unknown"),
                        m.get("data_source", "unknown"),
                        pair_hash,
                    ))

        if not uncached_items:
            return all_matches

        # Batch uncached items and send to LLM
        for i in range(0, len(uncached_items), self.batch_size):
            batch = uncached_items[i:i + self.batch_size]
            patient_mechs = [(pid, text, d, src) for pid, text, d, src, _ in batch]

            prompt = self._build_prompt(bench_mech, patient_mechs)
            matches = self._call_llm(prompt)

            if matches is None:
                continue

            # Cache results for each item in batch
            matched_pids = set(matches.keys())
            for pid, mech_text, _, _, pair_hash in batch:
                if pid in matched_pids:
                    match_info = matches[pid]
                    if isinstance(match_info, str):
                        match_info = {"finding": match_info, "data_source": "unknown"}
                    self._save_cached(pair_hash, {"is_match": True, "match_info": match_info})
                    all_matches[pid] = match_info
                else:
                    self._save_cached(pair_hash, {"is_match": False})

        return all_matches


def load_patient_results(patient_dir: str) -> dict[str, dict]:
    """Load all patient JSON results from a directory."""
    results = {}
    for fname in sorted(os.listdir(patient_dir)):
        if not fname.endswith(".json"):
            continue
        pid = fname.replace(".json", "")
        with open(os.path.join(patient_dir, fname)) as f:
            data = json.load(f)
        if data.get("status") == "success":
            results[pid] = data
    return results


def main():
    import argparse
    import pandas as pd
    from scipy.stats import fisher_exact

    parser = argparse.ArgumentParser(description="Match patient mechanisms against LBCL-Bench")
    parser.add_argument("--bench-csv", required=True, help="Path to LBCL-Bench filtered CSV")
    parser.add_argument("--patient-dir", required=True, help="Path to step3_per_patient results")
    parser.add_argument("--output-dir", required=True, help="Output directory for evaluation CSVs")
    parser.add_argument("--cache-dir", default=None, help="Cache directory (default: output-dir/_match_cache)")
    parser.add_argument("--batch-size", type=int, default=15)
    args = parser.parse_args()

    cache_dir = args.cache_dir or os.path.join(args.output_dir, "_match_cache")
    os.makedirs(args.output_dir, exist_ok=True)

    bench = pd.read_csv(args.bench_csv)
    patient_results = load_patient_results(args.patient_dir)

    or_pids = {pid for pid, d in patient_results.items() if d.get("response") == "OR"}
    nr_pids = {pid for pid, d in patient_results.items() if d.get("response") == "NR"}
    n_or, n_nr = len(or_pids), len(nr_pids)
    print(f"Loaded {len(patient_results)} patients ({n_or} OR, {n_nr} NR)")

    matcher = BenchMatcher(cache_dir=cache_dir, batch_size=args.batch_size)

    rows = []
    for idx, (_, mech) in enumerate(bench.iterrows()):
        mid = mech["mechanism_id"]
        print(f"  [{idx + 1}/{len(bench)}] {mid}: {mech['verbal_summary'][:60]}...", flush=True)

        matches = matcher.match_mechanism(mech.to_dict(), patient_results)
        matched_pids = set(matches.keys())
        matched_or = matched_pids & or_pids
        matched_nr = matched_pids & nr_pids

        table = [
            [len(matched_or), len(matched_nr)],
            [n_or - len(matched_or), n_nr - len(matched_nr)],
        ]
        odds_ratio, fisher_p = fisher_exact(table)

        direction = _infer_direction(mech["verbal_summary"])

        rows.append({
            "mechanism_id": mid,
            "verbal_summary": mech["verbal_summary"],
            "direction": direction,
            "total_matches": len(matched_pids),
            "or_matches": len(matched_or),
            "nr_matches": len(matched_nr),
            "or_rate": len(matched_or) / n_or if n_or else 0,
            "nr_rate": len(matched_nr) / n_nr if n_nr else 0,
            "fisher_odds_ratio": odds_ratio,
            "fisher_p": fisher_p,
            "matched_or_patients": ";".join(sorted(matched_or)),
            "matched_nr_patients": ";".join(sorted(matched_nr)),
            "matched_findings": json.dumps(matches),
        })

        print(f"    -> {len(matched_pids)} matches ({len(matched_or)} OR, {len(matched_nr)} NR), "
              f"Fisher p={fisher_p:.3f}", flush=True)

    results_df = pd.DataFrame(rows)
    out_path = os.path.join(args.output_dir, "bench_mechanism_patient_counts.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

    # Also dump all patient mechanisms
    all_mechs = []
    for pid, presult in patient_results.items():
        for m in presult.get("mechanisms_identified", []):
            all_mechs.append({
                "patient_id": pid,
                "response": presult.get("response", "?"),
                "mechanism": m.get("mechanism", "unknown"),
                "confidence": m.get("confidence", "unknown"),
                "direction": m.get("direction", "unknown"),
                "data_source": m.get("data_source", "unknown"),
            })
    pd.DataFrame(all_mechs).to_csv(
        os.path.join(args.output_dir, "all_patient_mechanisms.csv"), index=False
    )

    # Summary
    found = (results_df["total_matches"] > 0).sum()
    print(f"\nRecall: {found}/{len(results_df)} ({100 * found / len(results_df):.0f}%)")
    sig = (results_df["fisher_p"] < 0.05).sum()
    print(f"Fisher p<0.05: {sig}/{len(results_df)}")


if __name__ == "__main__":
    main()
