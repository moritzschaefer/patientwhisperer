"""Quick inspection of Step 2 discoveries."""
import json

with open("results/step2/discoveries.json") as f:
    data = json.load(f)

print("Keys:", list(data.keys()))
print("N discoveries:", len(data.get("discoveries", [])))
print("Total queries tested:", data.get("total_queries_tested"))
print("Sig at 0.05:", data.get("significant_at_005"))
print("Sig at 0.01:", data.get("significant_at_001"))
print()
print("Summary:", str(data.get("summary", ""))[:500])
print()

for i, d in enumerate(data.get("discoveries", [])):
    conf = d.get("confidence", "?")
    p = d.get("p_value", "?")
    direction = d.get("direction", "?")
    mech = d.get("mechanism", "?")[:90]
    agg = d.get("aggregation_method", "?")
    print(f"  {i+1:2d}. [{conf:6s}] p={p:<8} {direction:10s} agg={agg:12s} {mech}")
