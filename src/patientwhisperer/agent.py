"""
Stable agent dispatch module for PatientWhisperer per-patient analyses.

Provides reusable plumbing for invoking Claude Code CLI, extracting
structured JSON from agent output, loading patient data, and saving results.

Each experiment's run_agent.py should import from this module and customize
only the prompt construction logic.

Usage:
    from patientwhisperer.agent import (
        run_claudecode, extract_json, build_system_prompt,
        load_patient_data, save_result,
    )
"""
import json
import math
import os
import re
import subprocess
import sys


# ---------------------------------------------------------------------------
# System prompt assembly
# ---------------------------------------------------------------------------

def build_system_prompt(parts: list[str], output_path: str) -> str:
    """Concatenate multiple prompt files into a single system prompt.

    Args:
        parts: List of file paths to concatenate (e.g. shared_context.md,
               patient-analyst-instructions.md).
        output_path: Where to write the combined prompt.

    Returns:
        The output_path (for convenience).
    """
    sections = []
    for path in parts:
        with open(path) as f:
            sections.append(f.read())
    with open(output_path, "w") as f:
        f.write("\n\n---\n\n".join(sections))
    return output_path


# ---------------------------------------------------------------------------
# Agent invocation
# ---------------------------------------------------------------------------

def run_claudecode(
    prompt: str,
    system_prompt_file: str,
    *,
    model: str = "claude-opus-4-6",
    max_turns: int = 50,
    timeout: int = 1800,
    allowed_tools: str = "Bash,Read,Write,Edit",
) -> tuple[str, str, int]:
    """Run Claude Code CLI and return (stdout, stderr, returncode).

    Args:
        prompt: The user prompt to send.
        system_prompt_file: Path to the system prompt file
            (passed via --append-system-prompt-file).
        model: Model identifier.
        max_turns: Maximum agent turns.
        timeout: Subprocess timeout in seconds.
        allowed_tools: Comma-separated tool allowlist.

    Returns:
        Tuple of (stdout, stderr, returncode).
    """
    cmd = [
        "claude", "-p", prompt,
        "--model", model,
        "--append-system-prompt-file", system_prompt_file,
        "--output-format", "json",
        "--max-turns", str(max_turns),
        "--allowedTools", allowed_tools,
        "--dangerously-skip-permissions",
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout,
    )
    return result.stdout, result.stderr, result.returncode


def run_opencode(
    agent: str,
    prompt: str,
    *,
    timeout: int = 1800,
) -> tuple[str, str, int]:
    """Run an opencode agent via apptainer and return (stdout, stderr, returncode)."""
    cmd = [
        "apptainer", "run", "docker://openeuler/opencode",
        "run", "--agent", agent, "--format", "json", prompt,
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout,
    )
    return result.stdout, result.stderr, result.returncode


# ---------------------------------------------------------------------------
# JSON extraction from agent output
# ---------------------------------------------------------------------------

def extract_json(output: str, framework: str = "claudecode") -> dict | None:
    """Extract the agent's structured JSON from raw output.

    Handles both Claude Code --output-format json and opencode NDJSON streams.

    Args:
        output: Raw stdout from the agent process.
        framework: "claudecode" or "opencode".

    Returns:
        Parsed dict, or None if extraction fails.
    """
    if framework == "claudecode":
        return _extract_json_claudecode(output)
    return _extract_json_opencode(output)


def _extract_json_claudecode(output: str) -> dict | None:
    """Extract JSON from Claude Code --output-format json envelope."""
    try:
        envelope = json.loads(output)
        text = envelope.get("result", "")
        if not text:
            for key in ("content", "text", "output"):
                if key in envelope:
                    text = envelope[key]
                    break
        if text:
            return _extract_json_from_text(text)
    except json.JSONDecodeError:
        pass
    return _extract_json_from_text(output)


def _extract_json_opencode(output: str) -> dict | None:
    """Extract JSON from opencode NDJSON stream."""
    full_text = ""
    for line in output.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
            if event.get("type") == "text":
                full_text += event.get("part", {}).get("text", "")
        except json.JSONDecodeError:
            full_text += line
    return _extract_json_from_text(full_text if full_text else output)


def _extract_json_from_text(text: str) -> dict | None:
    """Extract ```json blocks or raw JSON objects from text."""
    # Fenced ```json ... ``` blocks (take last match — final synthesis)
    pattern = r"```json\s*\n(.*?)\n\s*```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        for match in reversed(matches):
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

    # Raw JSON objects on single lines
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue

    return None


# ---------------------------------------------------------------------------
# Patient data loading
# ---------------------------------------------------------------------------

def load_patient_data(patient_dir: str) -> dict:
    """Load a patient's data directory into a structured dict.

    Returns:
        {
            "patient_dir": str (absolute path),
            "patient_id": str,
            "clinical": dict (from clinical.json, empty if absent),
            "data_sources": dict (from data_sources.json),
            "has_infusion": bool,
            "has_spatial": bool,
            "n_spatial_cells": int,
            "response": str,
            "age": any,
            "gender": any,
            "therapy": any,
            "max_CRS": any | None,
            "max_ICANS": any | None,
            "has_crs": bool,
            "has_icans": bool,
            "available_files": list[str],
        }
    """
    patient_dir = os.path.abspath(patient_dir)

    # Data sources (required)
    with open(os.path.join(patient_dir, "data_sources.json")) as f:
        data_sources = json.load(f)

    # Clinical (optional — spatial-only patients may lack it)
    clinical_path = os.path.join(patient_dir, "clinical.json")
    if os.path.exists(clinical_path):
        with open(clinical_path) as f:
            clinical = json.load(f)
    else:
        clinical = {}

    pid = clinical.get("patient_id", os.path.basename(patient_dir))
    has_infusion = data_sources.get("has_infusion", False)
    has_spatial = data_sources.get("has_spatial", False)

    max_crs = clinical.get("max_CRS")
    max_icans = clinical.get("max_ICANS")
    has_crs = max_crs is not None and not (isinstance(max_crs, float) and math.isnan(max_crs))
    has_icans = max_icans is not None and not (isinstance(max_icans, float) and math.isnan(max_icans))

    # Enumerate available files
    available_files = sorted(os.listdir(patient_dir))

    return {
        "patient_dir": patient_dir,
        "patient_id": pid,
        "clinical": clinical,
        "data_sources": data_sources,
        "has_infusion": has_infusion,
        "has_spatial": has_spatial,
        "n_spatial_cells": data_sources.get("n_spatial_cells", 0),
        "response": clinical.get("Response_3m", "unknown"),
        "age": clinical.get("age", "unknown"),
        "gender": clinical.get("gender", "unknown"),
        "therapy": clinical.get("therapy", "unknown"),
        "max_CRS": max_crs,
        "max_ICANS": max_icans,
        "has_crs": has_crs,
        "has_icans": has_icans,
        "available_files": available_files,
    }


# ---------------------------------------------------------------------------
# Prompt building helpers
# ---------------------------------------------------------------------------

def build_modality_section(pdata: dict) -> str:
    """Build the 'Available Data Modalities' prompt section."""
    lines = []
    if pdata["has_infusion"]:
        lines.append("- CAR T infusion product scRNA-seq (CellWhisperer scores in infusion_features.csv)")
    if pdata["has_spatial"]:
        lines.append(
            f"- CosMx spatial transcriptomics from TME biopsy "
            f"({pdata['n_spatial_cells']} cells, features in spatial_features.csv)"
        )
    return "\n".join(lines)


def build_files_section(pdata: dict) -> str:
    """Build the 'Files' prompt section."""
    lines = ["- data_sources.json: Available modalities"]
    if pdata["clinical"]:
        lines.append("- clinical.json: Full clinical variables")
    if pdata["has_infusion"]:
        lines.append("- infusion_features.csv: CellWhisperer scores and cohort quantiles")
    if pdata["has_spatial"]:
        lines.append("- spatial_features.csv: Cell type proportions and proximity scores with cohort quantiles")
    return "\n".join(lines)


def build_toxicity_section(pdata: dict) -> str:
    """Build the toxicity analysis prompt section (empty if no CRS/ICANS data)."""
    if not pdata["has_crs"] and not pdata["has_icans"]:
        return ""

    parts = ["\n\n## Toxicity Analysis\n\n"]
    if pdata["has_crs"]:
        grade = int(pdata["max_CRS"]) if isinstance(pdata["max_CRS"], (int, float)) else pdata["max_CRS"]
        parts.append(f"This patient experienced CRS grade {grade}. ")
    if pdata["has_icans"]:
        grade = int(pdata["max_ICANS"]) if isinstance(pdata["max_ICANS"], (int, float)) else pdata["max_ICANS"]
        parts.append(f"This patient experienced ICANS grade {grade}. ")
    parts.append(
        "\nIn addition to explaining the response outcome, also explain these toxicity outcomes. "
        "What features of this patient's infusion product and/or tumor microenvironment might explain "
        "the severity (or mildness) of CRS and/or ICANS?"
    )
    return "".join(parts)


def build_cross_modal_section(pdata: dict) -> str:
    """Build the cross-modal integration prompt section (empty if single modality)."""
    if not (pdata["has_infusion"] and pdata["has_spatial"]):
        return ""
    return (
        "\n\n## Cross-Modal Integration\n\n"
        "This patient has BOTH infusion product and spatial TME data. After analyzing each modality "
        "independently, synthesize findings across modalities:\n"
        "- Do TME composition features (e.g., immune-suppressive cell types) explain why certain "
        "infusion product qualities did or did not translate to response?\n"
        "- Are there concordant signals (e.g., exhausted infusion product AND immunosuppressive TME)?\n"
        "- Tag each mechanism with its data source: 'spatial', 'infusion', or 'both'."
    )


# ---------------------------------------------------------------------------
# Result saving
# ---------------------------------------------------------------------------

def save_result(
    result: dict,
    output_path: str,
    raw_output_path: str,
    stdout: str,
    stderr: str,
) -> None:
    """Save structured result JSON and raw agent output.

    Args:
        result: Parsed result dict to save as JSON.
        output_path: Path for the structured JSON output.
        raw_output_path: Path for the raw stdout/stderr dump.
        stdout: Raw stdout from the agent.
        stderr: Raw stderr from the agent.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    os.makedirs(os.path.dirname(raw_output_path), exist_ok=True)
    with open(raw_output_path, "w") as f:
        f.write(f"=== STDOUT ===\n{stdout}\n\n=== STDERR ===\n{stderr}\n")


def process_agent_output(
    stdout: str,
    stderr: str,
    returncode: int,
    pdata: dict,
    framework: str = "claudecode",
) -> dict:
    """Parse agent output into a result dict, handling errors and parse failures.

    Args:
        stdout, stderr, returncode: From the agent subprocess.
        pdata: Patient data dict (from load_patient_data).
        framework: "claudecode" or "opencode".

    Returns:
        Result dict with at least: patient_id, response, status,
        mechanisms_identified, data_sources_available.
    """
    pid = pdata["patient_id"]
    response = pdata["response"]

    if returncode != 0:
        print(f"ERROR: Agent returned code {returncode}", file=sys.stderr, flush=True)
        return {
            "patient_id": pid,
            "response": response,
            "status": f"error_code_{returncode}",
            "mechanisms_identified": [],
            "data_sources_available": pdata["data_sources"],
        }

    parsed = extract_json(stdout, framework)
    if parsed:
        parsed.setdefault("patient_id", pid)
        parsed.setdefault("response", response)
        parsed.setdefault("status", "success")
        parsed.setdefault("data_sources_available", pdata["data_sources"])
        n = len(parsed.get("mechanisms_identified", []))
        print(f"  Found {n} mechanisms", flush=True)
        return parsed

    print("WARNING: Could not parse JSON from agent output", file=sys.stderr, flush=True)
    return {
        "patient_id": pid,
        "response": response,
        "status": "parse_error",
        "mechanisms_identified": [],
        "data_sources_available": pdata["data_sources"],
    }
