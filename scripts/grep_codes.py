"""Search a codelist for matching codes.
Usage: python scripts/grep_codes.py IL CL_UNIT usd dollar sdr"""
import sys, json, pathlib

flow, cid, terms = sys.argv[1], sys.argv[2], [t.lower() for t in sys.argv[3:]]
payload = json.loads(pathlib.Path(f"data/raw/structure_{flow}.json").read_text(encoding="utf-8"))
data = payload.get("data", payload)
cl = next((c for c in data.get("codelists", []) if c.get("id") == cid), None)
if cl is None:
    raise SystemExit(f"{cid} not found")

def nm(o):
    n = o.get("name") or o.get("names") or ""
    return n.get("en") if isinstance(n, dict) else n

hits = [c for c in cl.get("codes", [])
        if any(t in f"{c.get('id')} {nm(c)}".lower() for t in terms)]
print(f"{cid}: {len(hits)} of {len(cl.get('codes', []))} match {terms}\n")
for c in hits:
    print(f"  {c['id']:<28} {str(nm(c))[:60]}")

    