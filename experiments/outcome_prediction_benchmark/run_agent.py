"""
Agent dispatch wrapper for outcome prediction benchmark.

Dispatches a Claude agent to analyze a blinded patient and predict
therapy response (OR/NR). Captures full reasoning traces.

Usage:
    python run_agent.py patient --patient-dir data/patients/01 \
        --output results/predictions/01.json \
        --raw-output results/predictions/01_raw.txt
"""
import argparse
import json
import os
import re
import subprocess
import sys

SYSTEM_PROMPT_FILE = "system_prompt_combined.md"


def ensure_combined_system_prompt():
    """Concatenate shared_context.md and predictor-instructions.md if stale."""
    parts_files = ["shared_context.md", "predictor-instructions.md"]
    if os.path.exists(SYSTEM_PROMPT_FILE):
        combined_mtime = os.path.getmtime(SYSTEM_PROMPT_FILE)
        if all(os.path.getmtime(p) <= combined_mtime for p in parts_files):
            return
    parts = []
    for path in parts_files:
        with open(path) as f:
            parts.append(f.read())
    with open(SYSTEM_PROMPT_FILE, "w") as f:
        f.write("\n\n---\n\n".join(parts))


MODEL = os.environ.get("PREDICT_MODEL", "claude-opus-4-6")
EFFORT = os.environ.get("PREDICT_EFFORT", "")


def run_claudecode(prompt, timeout=1800, max_turns=50):
    """Run Claude Code CLI and return stdout, stderr, returncode.

    Uses stream-json output to capture full reasoning traces (every tool call,
    intermediate message, and result event).
    """
    ensure_combined_system_prompt()
    cmd = [
        "claude", "-p", prompt,
        "--model", MODEL,
        "--append-system-prompt-file", SYSTEM_PROMPT_FILE,
        "--output-format", "stream-json",
        "--verbose",
        "--max-turns", str(max_turns),
        "--allowedTools", "Bash", "Read", "Write", "Edit",
        "--dangerously-skip-permissions",
    ]
    if EFFORT:
        cmd.extend(["--effort", EFFORT])
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.stdout, result.stderr, result.returncode


def _extract_json_from_text(text):
    """Extract ```json blocks or raw JSON objects from text."""
    pattern = r"```json\s*\n(.*?)\n\s*```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        for match in reversed(matches):
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def extract_json_from_output(output):
    """Extract the agent's prediction JSON from Claude Code stream-json output.

    stream-json emits one JSON object per line (NDJSON). We collect all assistant
    text messages and search the last one for the prediction JSON block.
    """
    assistant_text = ""
    for line in output.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        # Collect assistant message text from various event types
        etype = event.get("type", "")
        if etype == "assistant" and "message" in event:
            # Full assistant message
            msg = event["message"]
            if isinstance(msg, dict):
                for block in msg.get("content", []):
                    if isinstance(block, dict) and block.get("type") == "text":
                        assistant_text += block.get("text", "")
            elif isinstance(msg, str):
                assistant_text += msg
        elif etype == "content_block_delta":
            delta = event.get("delta", {})
            if delta.get("type") == "text_delta":
                assistant_text += delta.get("text", "")
        elif etype == "result":
            # Final result event (like --output-format json)
            text = event.get("result", "")
            if text:
                parsed = _extract_json_from_text(text)
                if parsed:
                    return parsed

    if assistant_text:
        return _extract_json_from_text(assistant_text)
    # Fallback: try the raw output
    return _extract_json_from_text(output)


def cmd_patient(args):
    """Analyze a single patient (blinded)."""
    data_sources_path = os.path.join(args.patient_dir, "data_sources.json")
    with open(data_sources_path) as f:
        data_sources = json.load(f)

    clinical_path = os.path.join(args.patient_dir, "clinical.json")
    if os.path.exists(clinical_path):
        with open(clinical_path) as f:
            clinical = json.load(f)
    else:
        clinical = {}

    pid = clinical.get("patient_id", os.path.basename(args.patient_dir))
    age = clinical.get("age", "unknown")
    gender = clinical.get("gender", "unknown")
    therapy = clinical.get("therapy", "unknown")

    has_infusion = data_sources.get("has_infusion", False)
    has_spatial = data_sources.get("has_spatial", False)
    n_spatial_cells = data_sources.get("n_spatial_cells", 0)

    # Build modality description
    modality_desc = []
    if has_infusion:
        modality_desc.append(
            "CAR T infusion product scRNA-seq (CellWhisperer scores in infusion_features.csv)")
    if has_spatial:
        modality_desc.append(
            f"CosMx spatial transcriptomics from TME biopsy ({n_spatial_cells} cells, "
            "features in spatial_features.csv)")

    modality_section = "\n".join(f"- {m}" for m in modality_desc)

    # Build available files section
    files_section = "- data_sources.json: Available modalities\n"
    if os.path.exists(clinical_path):
        files_section += "- clinical.json: Clinical variables (response is NOT included)\n"
    if has_infusion:
        files_section += "- infusion_features.csv: CellWhisperer scores and cohort quantiles\n"
    if has_spatial:
        files_section += "- spatial_features.csv: Cell type proportions and proximity scores with cohort quantiles\n"

    # Build clinical summary (without response)
    clinical_summary_parts = [f"age={age}", f"gender={gender}", f"therapy={therapy}"]
    ldh = clinical.get("LDH")
    if ldh is not None:
        clinical_summary_parts.append(f"LDH={ldh}")
    spd = clinical.get("tumor_burden_SPD")
    if spd is not None:
        clinical_summary_parts.append(f"tumor_burden_SPD={spd:.1f}")
    max_crs = clinical.get("max_CRS")
    if max_crs is not None:
        clinical_summary_parts.append(f"max_CRS={int(max_crs)}")
    max_icans = clinical.get("max_ICANS")
    if max_icans is not None:
        clinical_summary_parts.append(f"max_ICANS={int(max_icans)}")
    clinical_summary = ", ".join(clinical_summary_parts)

    # Cross-modal section
    cross_modal = ""
    if has_infusion and has_spatial:
        cross_modal = (
            "\n\n## Cross-Modal Integration\n\n"
            "This patient has BOTH infusion product and spatial TME data. After analyzing each "
            "modality independently, integrate findings to inform your prediction."
        )

    prompt = (
        f"Predict the therapy outcome for patient {pid}.\n\n"
        f"Clinical profile: {clinical_summary}\n\n"
        f"## Available Data Modalities\n\n{modality_section}\n\n"
        f"## Patient Data Directory\n\n"
        f"Path: {os.path.abspath(args.patient_dir)}\n"
        f"Files:\n{files_section}\n\n"
        f"Analyze this patient's molecular data and predict whether they responded (OR) "
        f"or did not respond (NR) to CAR T cell therapy at 3 months. "
        f"You do NOT know the outcome. Reason carefully from the data."
        f"{cross_modal}"
    )

    print(f"Predicting outcome for patient {pid} (infusion={has_infusion}, spatial={has_spatial})...",
          flush=True)

    stdout, stderr, returncode = run_claudecode(prompt, timeout=1800)

    # Save raw output (full reasoning trace)
    os.makedirs(os.path.dirname(args.raw_output), exist_ok=True)
    with open(args.raw_output, "w") as f:
        f.write(f"=== STDOUT ===\n{stdout}\n\n=== STDERR ===\n{stderr}\n")

    if returncode != 0:
        print(f"ERROR: Agent returned code {returncode}", file=sys.stderr, flush=True)
        result = {
            "patient_id": pid,
            "prediction": "unknown",
            "confidence": "none",
            "status": f"error_code_{returncode}",
            "data_sources_available": data_sources,
        }
    else:
        parsed = extract_json_from_output(stdout)
        if parsed:
            parsed.setdefault("patient_id", pid)
            parsed.setdefault("status", "success")
            parsed.setdefault("data_sources_available", data_sources)
            result = parsed
            pred = result.get("prediction", "unknown")
            conf = result.get("confidence", "unknown")
            print(f"  Prediction: {pred} (confidence: {conf})", flush=True)
        else:
            print("WARNING: Could not parse JSON from agent output",
                  file=sys.stderr, flush=True)
            result = {
                "patient_id": pid,
                "prediction": "unknown",
                "confidence": "none",
                "status": "parse_error",
                "data_sources_available": data_sources,
            }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved to {args.output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Outcome prediction benchmark agent dispatcher")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_patient = subparsers.add_parser("patient")
    p_patient.add_argument("--patient-dir", required=True)
    p_patient.add_argument("--output", required=True)
    p_patient.add_argument("--raw-output", required=True)

    args = parser.parse_args()
    if args.command == "patient":
        cmd_patient(args)


if __name__ == "__main__":
    main()
