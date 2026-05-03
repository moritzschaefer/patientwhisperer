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
import asyncio
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys


def _hash_bench(verbal_summary: str, direction: str) -> str:
    """Stable hash of the benchmark mechanism identity."""
    key = f"{verbal_summary.strip()}|{direction}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _hash_pair(bench_hash: str, agent_mechanism: str) -> str:
    """Stable hash for a (bench, agent_mechanism) pair."""
    key = f"{bench_hash}|{agent_mechanism.strip()}"
    return hashlib.sha256(key.encode()).hexdigest()[:20]


def _extract_json(stdout: str) -> dict | None:
    """Extract JSON from LLM output (handles NDJSON stream, markdown blocks, and raw JSON)."""
    # First try markdown block on raw stdout (before NDJSON processing strips newlines)
    m = re.search(r"```json\s*\n(.*?)\n\s*```", stdout, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

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

    # Try ```json ... ``` block on NDJSON-reassembled text
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
        if not shutil.which("claude"):
            raise RuntimeError("claude CLI not found on PATH")
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

    def _build_cmd(self, prompt: str) -> list[str]:
        return [
            "claude",
            "-p", prompt,
            "--output-format", "text",
            "--max-turns", "1",
            "--tools", "",
            "--allowedTools", "",
            "--model", "sonnet",
            "--dangerously-skip-permissions",
            "--strict-mcp-config",
            "--no-session-persistence",
            "--system-prompt", "You are a mechanism matching judge. Respond with ONLY a valid JSON object, no markdown formatting.",
        ]

    def _call_llm(self, prompt: str, max_retries: int = 2) -> dict | None:
        """Call claude CLI for JSON response (synchronous, single call)."""
        cmd = self._build_cmd(prompt)
        for attempt in range(max_retries + 1):
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=self.timeout,
                    cwd="/tmp",
                )
                if result.returncode != 0:
                    raise RuntimeError(
                        f"claude exited with code {result.returncode}: "
                        f"{result.stderr.strip()[:200] or result.stdout.strip()[:200]}"
                    )
                parsed = _extract_json(result.stdout)
                if parsed and "matches" in parsed:
                    return parsed["matches"]
                print(f"  WARNING: unparseable output (attempt {attempt + 1})",
                      file=sys.stderr, flush=True)
            except subprocess.TimeoutExpired:
                print(f"  WARNING: LLM timed out (attempt {attempt + 1})", file=sys.stderr, flush=True)
            except Exception as e:
                print(f"  LLM error (attempt {attempt + 1}): {e}", file=sys.stderr, flush=True)
                if attempt == max_retries:
                    raise
        return None

    async def _call_llm_async(self, prompt: str, max_retries: int = 2) -> dict | None:
        """Call claude CLI for JSON response (async, for concurrent execution).

        Raises RuntimeError immediately on rate limits (429 or daily cap).
        Cache is safe; rerun picks up where it left off.
        """
        cmd = self._build_cmd(prompt)
        for attempt in range(max_retries + 1):
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                    cwd="/tmp",
                )
                try:
                    stdout_bytes, stderr_bytes = await asyncio.wait_for(
                        proc.communicate(), timeout=self.timeout
                    )
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
                    print(f"  WARNING: LLM timed out (attempt {attempt + 1})",
                          file=sys.stderr, flush=True)
                    continue

                stdout = stdout_bytes.decode()
                stderr = stderr_bytes.decode()
                if proc.returncode != 0:
                    err_msg = stderr.strip()[:200] or stdout.strip()[:200]
                    if "429" in err_msg or "Rate limited" in err_msg or "rate limit" in err_msg.lower() \
                            or "hit your limit" in err_msg.lower():
                        raise RuntimeError(f"Rate limited: {err_msg}")
                    raise RuntimeError(
                        f"claude exited with code {proc.returncode}: {err_msg}"
                    )
                parsed = _extract_json(stdout)
                if parsed and "matches" in parsed:
                    return parsed["matches"]
                print(f"  WARNING: unparseable output (attempt {attempt + 1})",
                      file=sys.stderr, flush=True)
            except asyncio.TimeoutError:
                pass
            except RuntimeError:
                raise
            except Exception as e:
                print(f"  LLM error (attempt {attempt + 1}): {e}", file=sys.stderr, flush=True)
                if attempt == max_retries:
                    raise
        return None

    def _collect_uncached(self, bench_mech: dict, patient_results: dict[str, dict]):
        """Separate cached from uncached (bench, agent_mechanism) pairs."""
        bench_hash = _hash_bench(
            bench_mech["verbal_summary"],
            bench_mech.get("_direction", _infer_direction(bench_mech["verbal_summary"])),
        )
        bench_mech["_direction"] = bench_mech.get(
            "_direction", _infer_direction(bench_mech["verbal_summary"])
        )

        all_matches = {}
        uncached_items = []

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

        return all_matches, uncached_items

    def _process_batch_result(self, batch, matches, all_matches):
        """Cache batch results and update all_matches."""
        if matches is None:
            return
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

    def match_mechanism(
        self,
        bench_mech: dict,
        patient_results: dict[str, dict],
    ) -> dict[str, dict]:
        """Match one benchmark mechanism against all patients (sequential)."""
        all_matches, uncached_items = self._collect_uncached(bench_mech, patient_results)
        if not uncached_items:
            return all_matches

        for i in range(0, len(uncached_items), self.batch_size):
            batch = uncached_items[i:i + self.batch_size]
            patient_mechs = [(pid, text, d, src) for pid, text, d, src, _ in batch]
            prompt = self._build_prompt(bench_mech, patient_mechs)
            matches = self._call_llm(prompt)
            if matches is None:
                mid = bench_mech.get("mechanism_id", "?")
                raise RuntimeError(
                    f"LLM matching failed for {mid} batch {i // self.batch_size + 1}"
                )
            self._process_batch_result(batch, matches, all_matches)

        return all_matches

    async def match_mechanism_async(
        self,
        bench_mech: dict,
        patient_results: dict[str, dict],
        concurrency: int = 8,
    ) -> dict[str, dict]:
        """Match one benchmark mechanism against all patients (concurrent).

        Runs up to `concurrency` claude CLI calls in parallel using asyncio.
        """
        all_matches, uncached_items = self._collect_uncached(bench_mech, patient_results)
        if not uncached_items:
            return all_matches

        batches = []
        for i in range(0, len(uncached_items), self.batch_size):
            batches.append(uncached_items[i:i + self.batch_size])

        sem = asyncio.Semaphore(concurrency)

        async def process_batch(batch):
            async with sem:
                patient_mechs = [(pid, text, d, src) for pid, text, d, src, _ in batch]
                prompt = self._build_prompt(bench_mech, patient_mechs)
                return batch, await self._call_llm_async(prompt)

        tasks = [process_batch(b) for b in batches]
        failed = 0
        for coro in asyncio.as_completed(tasks):
            batch, matches = await coro
            if matches is None:
                failed += len(batch)
            else:
                self._process_batch_result(batch, matches, all_matches)

        if failed:
            mid = bench_mech.get("mechanism_id", "?")
            raise RuntimeError(
                f"LLM matching failed for {mid}: {failed} items returned None. "
                f"Cache is safe but results are incomplete — rerun to retry."
            )

        return all_matches


def load_patient_results(patient_dir: str) -> dict[str, dict]:
    """Load all patient JSON results from a directory.

    Includes any result file that contains mechanisms_identified,
    regardless of status field.
    """
    results = {}
    for fname in sorted(os.listdir(patient_dir)):
        if not fname.endswith(".json"):
            continue
        pid = fname.replace(".json", "")
        with open(os.path.join(patient_dir, fname)) as f:
            data = json.load(f)
        if data.get("mechanisms_identified"):
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
