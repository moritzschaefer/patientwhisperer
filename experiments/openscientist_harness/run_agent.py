"""
Agent dispatch for OpenScientist-based per-patient mechanism discovery.

Replaces the single-shot Claude Code invocation with OpenScientist's
multi-iteration discovery loop, KnowledgeState tracking, and skill-guided
hypothesis testing.

TODO/NOTE: This experiment depends on `vendor/openscientist`, which is
NOT tracked in this repo (kept slim). Clone or otherwise obtain it into
`<project_root>/vendor/openscientist/` before running.

Usage:
    python run_agent.py patient \
        --patient-dir data/patients/PAT01 \
        --output results/step3_per_patient/PAT01.json \
        --raw-output results/step3_per_patient/PAT01_raw.txt
"""
import argparse
import asyncio
import json
import os
import shutil
import sys
from pathlib import Path
from uuid import uuid4

# Wire up both OpenScientist and patientwhisperer on sys.path
VENDOR_DIR = Path(__file__).resolve().parent.parent.parent / "vendor" / "openscientist" / "src"
PROJECT_SRC = Path(__file__).resolve().parent.parent.parent / "src"
sys.path.insert(0, str(VENDOR_DIR))
sys.path.insert(0, str(PROJECT_SRC))

import re

from patientwhisperer.agent import load_patient_data
from stages import DEFAULT_STAGES

AGENT_FRAMEWORK = "openscientist"
DB_PATH = os.environ.get("OS_DB_PATH", str(Path(__file__).resolve().parent / "openscientist.db"))
STAGES = json.loads(os.environ.get("OS_STAGES", "null")) or DEFAULT_STAGES


def _ensure_db_url():
    """Set DATABASE_URL if not already set."""
    if "DATABASE_URL" not in os.environ:
        os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{DB_PATH}"


def build_research_question(pdata: dict) -> str:
    """Build the per-patient research question for OpenScientist."""
    pid = pdata["patient_id"]
    response = pdata["response"]
    age = pdata.get("age", "unknown")
    gender = pdata.get("gender", "unknown")
    therapy = pdata.get("therapy", "unknown")

    modalities = []
    if pdata["has_infusion"]:
        modalities.append("CAR T infusion product scRNA-seq (CellWhisperer scores)")
    if pdata["has_spatial"]:
        modalities.append("CosMx spatial transcriptomics (tumor microenvironment)")

    if response != "unknown":
        outcome = "responded (OR)" if response == "OR" else "did not respond (NR)"
        task = f"This patient {outcome} to CAR T therapy at 3 months."
    else:
        task = "This patient's treatment outcome is not available."

    return (
        f"Identify the mechanistic explanations for CAR T therapy outcome "
        f"in patient {pid}.\n\n"
        f"**Patient profile:** age={age}, gender={gender}, therapy={therapy}, "
        f"Response_3m={response}\n\n"
        f"**Available data:** {', '.join(modalities)}\n\n"
        f"{task}\n\n"
        f"Analyze the patient's pre-computed feature data (CSV files in the job "
        f"directory). Identify 3-10 resistance or response mechanisms with "
        f"quantitative evidence. Follow the falsification protocol in your "
        f"workflow skills. Search PubMed to ground your hypotheses in published "
        f"CAR T biology."
    )


async def create_job(pdata: dict, job_dir: Path) -> tuple[str, Path]:
    """Create an OpenScientist Job row in the database."""
    from openscientist.database.models.job import Job
    from openscientist.database.models.job_data_file import JobDataFile
    from openscientist.database.session import AsyncSessionLocal

    job_id = uuid4()
    job_dir_actual = job_dir / str(job_id)
    job_dir_actual.mkdir(parents=True, exist_ok=True)

    # Copy patient data files into the job directory
    patient_dir = Path(pdata["patient_dir"])
    for f in patient_dir.iterdir():
        if f.is_file():
            shutil.copy2(f, job_dir_actual / f.name)

    research_question = build_research_question(pdata)

    async with AsyncSessionLocal(thread_safe=True) as session:
        job = Job(
            id=job_id,
            title=research_question,
            description=f"Per-patient analysis: {pdata['patient_id']}",
            status="pending",
            investigation_mode="autonomous",
            use_hypotheses=True,
            max_iterations=len(STAGES),
            current_iteration=1,
            llm_provider=os.environ.get("CLAUDE_PROVIDER", "anthropic"),
            llm_config={"stages": STAGES},
        )
        session.add(job)

        # Register data files
        for f in (job_dir_actual).iterdir():
            if f.is_file():
                data_file = JobDataFile(
                    id=uuid4(),
                    job_id=job_id,
                    file_path=str(f),
                    filename=f.name,
                    file_type=f.suffix.lstrip(".") or "other",
                    file_size=f.stat().st_size,
                    mime_type="application/json" if f.suffix == ".json" else "text/csv",
                )
                session.add(data_file)

        await session.commit()

    return str(job_id), job_dir_actual


def _parse_metadata_tag(text: str, field: str, default: str) -> str:
    """Extract [field=value] from evidence text, e.g. [direction=pro-resistance]."""
    m = re.search(rf"\[{field}=([^\];]+)", text)
    return m.group(1).strip() if m else default


def _extract_mechanism(finding: dict) -> dict:
    """Convert a KnowledgeState finding to evaluation mechanism dict."""
    evidence = finding.get("evidence", "")
    return {
        "mechanism": finding["title"],
        "data_source": finding.get("data_source") or _parse_metadata_tag(evidence, "data_source", "infusion"),
        "evidence": evidence,
        "counter_evidence": "",
        "falsification_verdict": "survived",
        "confidence": finding.get("confidence") or _parse_metadata_tag(evidence, "confidence", "medium"),
        "direction": finding.get("direction") or _parse_metadata_tag(evidence, "direction", "unknown"),
        "effect_size": "",
        "patient_percentile": "",
    }


def _extract_hypothesis_mechanism(hyp: dict) -> dict:
    """Convert a supported hypothesis to evaluation mechanism dict."""
    result = hyp.get("result") or {}
    return {
        "mechanism": hyp["statement"],
        "data_source": result.get("data_source", "infusion"),
        "evidence": result.get("summary", ""),
        "counter_evidence": "",
        "falsification_verdict": "survived",
        "confidence": result.get("confidence", "medium"),
        "direction": result.get("direction", "unknown"),
        "effect_size": result.get("effect_size", ""),
        "patient_percentile": "",
    }


def _find_job_dir(job_id: str) -> Path | None:
    """Locate the job directory by job_id under the jobs/ tree."""
    jobs_base = Path(__file__).resolve().parent / "jobs"
    candidate = jobs_base / job_id
    if candidate.is_dir():
        return candidate
    return None


def _read_final_report(job_id: str) -> str | None:
    """Read final_report.md from the job directory if it exists."""
    job_dir = _find_job_dir(job_id)
    if job_dir is None:
        return None
    report_path = job_dir / "final_report.md"
    if report_path.is_file():
        return report_path.read_text(encoding="utf-8")
    return None


def ks_to_evaluation_json(job_id: str, pdata: dict) -> dict:
    """Convert OpenScientist KnowledgeState into our evaluation JSON format."""
    from openscientist.knowledge_state import KnowledgeState

    ks = KnowledgeState.load_from_database_sync(job_id)

    # Skip the "Quantitative Profile" finding from stage 1 (it's metadata, not a mechanism)
    mechanisms = []
    for finding in ks.data["findings"]:
        if finding["title"].lower().startswith("quantitative profile"):
            continue
        mechanisms.append(_extract_mechanism(finding))

    # Include supported hypotheses not already covered by findings
    finding_titles = {m["mechanism"].lower() for m in mechanisms}
    for hyp in ks.data["hypotheses"]:
        if hyp["status"] == "supported" and hyp["statement"].lower() not in finding_titles:
            mechanisms.append(_extract_hypothesis_mechanism(hyp))

    rejected = []
    for hyp in ks.data["hypotheses"]:
        if hyp["status"] == "rejected":
            result = hyp.get("result") or {}
            rejected.append({
                "mechanism": hyp["statement"],
                "reason": result.get("conclusion", result.get("summary", "")),
            })

    return {
        "status": "success",
        "patient_id": pdata["patient_id"],
        "response": pdata["response"],
        "clinical_summary": f"Patient {pdata['patient_id']}, {pdata.get('therapy', 'unknown')}",
        "phase1_profile": {
            "n_extreme_infusion": 0,
            "n_extreme_spatial": 0,
            "key_ratios": {},
        },
        "mechanisms_identified": mechanisms,
        "rejected_hypotheses": rejected,
        "toxicity_analysis": {},
        "narrative": ks.data.get("consensus_answer") or _read_final_report(job_id) or "",
        "suggested_follow_up": [],
        "openscientist_meta": {
            "job_id": str(ks.data["config"]["job_id"]),
            "iterations": ks.data["iteration"],
            "stages": STAGES,
            "n_hypotheses": len(ks.data["hypotheses"]),
            "n_findings": len(ks.data["findings"]),
            "n_literature": len(ks.data["literature"]),
        },
    }


async def run_patient_async(pdata: dict, output: str, raw_output: str) -> None:
    """Run OpenScientist discovery loop for a single patient."""
    _ensure_db_url()

    from openscientist.orchestrator.discovery import run_discovery_async

    job_base = Path(__file__).resolve().parent / "jobs"
    job_base.mkdir(exist_ok=True)

    job_id, job_dir = await create_job(pdata, job_base)
    pid = pdata["patient_id"]

    print(
        f"  OpenScientist job {job_id[:8]}... "
        f"(stages={STAGES}, dir={job_dir})",
        flush=True,
    )

    result = await run_discovery_async(job_dir)

    # Save raw OpenScientist output
    Path(raw_output).parent.mkdir(parents=True, exist_ok=True)
    with open(raw_output, "w") as f:
        json.dump(result, f, indent=2, default=str)

    # Convert to evaluation format
    eval_json = ks_to_evaluation_json(job_id, pdata)

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(eval_json, f, indent=2)

    n_mech = len(eval_json["mechanisms_identified"])
    print(f"  Job {job_id[:8]}... completed: {n_mech} mechanisms identified", flush=True)


def cmd_patient(args):
    """Analyze a single patient (called by run_experiment.py)."""
    pdata = load_patient_data(args.patient_dir)
    pid = pdata["patient_id"]

    print(
        f"Analyzing patient {pid} ({pdata['response']}, "
        f"infusion={pdata['has_infusion']}, spatial={pdata['has_spatial']})...",
        flush=True,
    )

    asyncio.run(run_patient_async(pdata, args.output, args.raw_output))


def main():
    parser = argparse.ArgumentParser(
        description="OpenScientist-based per-patient agent dispatcher"
    )
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
