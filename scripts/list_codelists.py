"""List codelists in a saved structure payload. Usage: python scripts/list_codelists.py IL"""
import sys, json, pathlib

flow = sys.argv[1]
payload = json.loads(pathlib.Path(f"data/raw/structure_{flow}.json").read_text(encoding="utf-8"))
data = payload.get("data", payload)
cls = data.get("codelists", [])

def nm(o):
    n = o.get("name") or o.get("names") or ""
    return n.get("en") if isinstance(n, dict) else n

specific = [c for c in cls if c.get("agencyID") != "IMF"]
generic = [c for c in cls if c.get("agencyID") == "IMF"]

print("=" * 90)
print(f"DATAFLOW-SPECIFIC CODELISTS ({len(specific)}) — these are the ones you want")
print("=" * 90)
for c in sorted(specific, key=lambda x: x.get("id", "")):
    print(f"  {str(c.get('agencyID')):<10} {str(c.get('id')):<34} n={len(c.get('codes', [])):<5} {str(nm(c))[:44]}")

print(f"\nGENERIC (agency=IMF), {len(generic)} — showing any with 'unit', 'country' or 'reserve' in the name")
for c in sorted(generic, key=lambda x: x.get("id", "")):
    label = f"{c.get('id')} {nm(c)}".lower()
    if any(k in label for k in ("unit", "country", "reserve", "liquid", "instrument")):
        print(f"  {str(c.get('id')):<34} n={len(c.get('codes', [])):<5} {str(nm(c))[:44]}")