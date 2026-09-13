"""Print codelists by ID from a saved structure payload.
Usage: python scripts/inspect_structure.py ER CL_ER_COUNTRY_PUB CL_ER_INDICATOR_PUB CL_ER_TYPE_OF_TRANSFORMATION CL_FREQ"""
import sys, json, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.config import ISO3

flow, wanted = sys.argv[1], sys.argv[2:]
payload = json.loads(pathlib.Path(f"data/raw/structure_{flow}.json").read_text(encoding="utf-8"))
data = payload.get("data", payload)
codelists = {cl.get("id"): cl for cl in data.get("codelists", [])}

def nm(o):
    n = o.get("name") or o.get("names") or ""
    return n.get("en") if isinstance(n, dict) else n

for cid in wanted:
    cl = codelists.get(cid)
    print("=" * 80)
    print(f"{cid}")
    print("=" * 80)
    if not cl:
        print(f"  not present. available: {sorted(codelists)[:15]} ...\n")
        continue
    codes = cl.get("codes", [])
    print(f"  agency={cl.get('agencyID')} v{cl.get('version')} isPartial={cl.get('isPartial')} n={len(codes)}")
    if "COUNTRY" in cid:
        found = [c for c in codes if c.get("id") in ISO3]
        for c in found:
            print(f"    {c['id']:<8} {nm(c)}")
        print(f"    >>> panel: {len(found)}/12  MISSING: {sorted(set(ISO3) - {c['id'] for c in found}) or 'none'}")
    elif len(codes) <= 60:
        for c in codes:
            print(f"    {c['id']:<30} {str(nm(c))[:58]}")
    else:
        for c in codes[:40]:
            print(f"    {c['id']:<30} {str(nm(c))[:58]}")
        print(f"    ... {len(codes)-40} more")
    print()