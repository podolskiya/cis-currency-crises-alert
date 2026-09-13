"""Inspect an IMF dataflow's dimensions and codelists. Usage: python scripts/imf_structure.py IMF.STA ER"""
import sys, json, pathlib, httpx

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.config import ISO3

BASE = "https://api.imf.org/external/sdmx/3.0"
HEADERS = {"Accept": "application/vnd.sdmx.structure+json;version=2.0.0"}
MAX_CODES = 45

agency, flow = sys.argv[1], sys.argv[2]
url = f"{BASE}/structure/dataflow/{agency}/{flow}/~"
r = httpx.get(url, headers=HEADERS, params={"references": "all", "detail": "full"},
              timeout=180, follow_redirects=True)
print(f"GET {url} -> {r.status_code}  {len(r.content):,} bytes\n")
r.raise_for_status()
payload = r.json()

pathlib.Path("data/raw").mkdir(parents=True, exist_ok=True)
pathlib.Path(f"data/raw/structure_{flow}.json").write_text(
    json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

data = payload.get("data", payload)
codelists = {cl.get("id"): cl for cl in data.get("codelists", [])}
print(f"Codelists returned: {len(codelists)}\n")

def name_of(obj):
    n = obj.get("name") or obj.get("names") or ""
    return n.get("en") if isinstance(n, dict) else n

def codelist_id(dim):
    rep = dim.get("localRepresentation") or {}
    urn = rep.get("enumeration") or ""
    if isinstance(urn, dict):
        urn = urn.get("urn", "")
    return urn.split(":")[-1].split("(")[0] if urn else None

ds = data["dataStructures"][0]
comp = ds["dataStructureComponents"]
dims = comp["dimensionList"]["dimensions"]
time_dims = comp["dimensionList"].get("timeDimensions", [])

print("=" * 90)
print(f"DIMENSION ORDER for {agency}:{flow} (this is your SDMX key order)")
print("=" * 90)
for d in sorted(dims, key=lambda x: x.get("position", 0)):
    print(f"  {d.get('position')}. {d['id']:<20} codelist={codelist_id(d)}")
for t in time_dims:
    print(f"  TIME: {t['id']}")
print()

for d in sorted(dims, key=lambda x: x.get("position", 0)):
    cid = codelist_id(d)
    cl = codelists.get(cid)
    if not cl:
        print(f"--- {d['id']}: no codelist returned (free text or uncoded)\n")
        continue
    codes = cl.get("codes", [])
    print(f"--- {d['id']}  ({cid}, {len(codes)} codes)")
    if "COUNTRY" in d["id"].upper() or "REF_AREA" in d["id"].upper():
        hits = [c for c in codes if c.get("id") in ISO3]
        missing = set(ISO3) - {c.get("id") for c in hits}
        for c in hits:
            print(f"      {c['id']:<10} {name_of(c)}")
        print(f"      >>> panel codes found: {len(hits)}/12   MISSING: {sorted(missing) or 'none'}")
    else:
        for c in codes[:MAX_CODES]:
            print(f"      {c['id']:<26} {str(name_of(c))[:55]}")
        if len(codes) > MAX_CODES:
            print(f"      ... {len(codes) - MAX_CODES} more (see data/raw/structure_{flow}.json)")
    print()