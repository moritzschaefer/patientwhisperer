"""Quick KnowledgeState inspection script."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "vendor", "openscientist", "src"))
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///openscientist.db")

from openscientist.knowledge_state import KnowledgeState

job_id = sys.argv[1] if len(sys.argv) > 1 else "ea7da4ce-db61-4cc8-98e3-9588b1bed4d3"
ks = KnowledgeState.load_from_database_sync(job_id)

print(f"Literature: {len(ks.data['literature'])}")
for lit in ks.data["literature"][:5]:
    print(f"  PMID {lit['pmid']}: {lit['title'][:80]}")

print(f"\nFindings: {len(ks.data['findings'])}")
for f in ks.data["findings"]:
    print(f"  {f['id']}: {f['title'][:60]} [dir={f.get('direction','?')}, conf={f.get('confidence','?')}]")

print(f"\nHypotheses: {len(ks.data['hypotheses'])}")
for h in ks.data["hypotheses"]:
    r = h.get("result") or {}
    print(f"  {h['id']} [{h['status']}]: {h['statement'][:60]} dir={r.get('direction','?')}")

print(f"\nConsensus: {repr(ks.data.get('consensus_answer'))[:200]}")
print(f"Iteration: {ks.data['iteration']}")
