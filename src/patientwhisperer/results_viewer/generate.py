"""Generate an interactive HTML browser for PatientWhisperer analysis traces."""
import argparse
import json
import os
from pathlib import Path

TEMPLATE_PATH = Path(__file__).parent / "template.html"


def load_patient_results(results_dir: str, patients: list[str] | None = None) -> dict:
    """Load patient result JSONs from a results directory.

    Args:
        results_dir: Path to directory containing <pid>.json files.
        patients: Optional list of patient IDs to include. If None, load all.

    Returns:
        Dict mapping patient_id -> parsed result dict.
    """
    results = {}
    for f in sorted(os.listdir(results_dir)):
        if not f.endswith(".json") or f.endswith("_raw.json"):
            continue
        pid = f.removesuffix(".json")
        if patients and pid not in patients:
            continue
        path = os.path.join(results_dir, f)
        if os.path.getsize(path) < 10:
            continue
        with open(path) as fh:
            data = json.load(fh)
        if data.get("status") != "success":
            continue
        results[pid] = data
    return results


def generate_html(results: dict, title: str = "PatientWhisperer Analysis Traces") -> str:
    """Generate HTML from patient results and the template.

    Args:
        results: Dict mapping patient_id -> result dict.
        title: Page title.

    Returns:
        Complete HTML string.
    """
    template = TEMPLATE_PATH.read_text()
    patients_json = json.dumps(results)

    # Sort: OR patients first, then NR, alphabetically within each group
    or_pids = sorted(pid for pid, d in results.items() if d.get("response") == "OR")
    nr_pids = sorted(pid for pid, d in results.items() if d.get("response") != "OR")
    order_json = json.dumps(or_pids + nr_pids)

    html = template.replace("{{PATIENT_DATA}}", patients_json)
    html = html.replace("{{PATIENT_ORDER}}", order_json)
    html = html.replace("{{TITLE}}", title)
    return html


def main():
    parser = argparse.ArgumentParser(
        description="Generate interactive HTML browser for PatientWhisperer traces"
    )
    parser.add_argument("--results-dir", required=True, help="Directory with <pid>.json result files")
    parser.add_argument("--output", default="trace_browser.html", help="Output HTML file path")
    parser.add_argument("--patients", nargs="*", default=None, help="Subset of patient IDs (default: all)")
    parser.add_argument("--title", default="PatientWhisperer Analysis Traces", help="Page title")
    args = parser.parse_args()

    results = load_patient_results(args.results_dir, args.patients)
    if not results:
        print(f"No successful results found in {args.results_dir}")
        return

    print(f"Loaded {len(results)} patients ({sum(1 for d in results.values() if d.get('response')=='OR')} OR, "
          f"{sum(1 for d in results.values() if d.get('response')!='OR')} NR)")

    html = generate_html(results, title=args.title)

    with open(args.output, "w") as f:
        f.write(html)
    print(f"Written to {args.output}")
