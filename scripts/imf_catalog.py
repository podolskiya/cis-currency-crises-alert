import json, pathlib, httpx, pandas as pd

BASE = "https://api.imf.org/external/sdmx/3.0"
ACCEPTS = [
    {"Accept": "application/vnd.sdmx.structure+json;version=2.0.0"},
    {"Accept": "application/json"},
]

resp = None
for h in ACCEPTS:
    r = httpx.get(f"{BASE}/structure/dataflow", headers=h, timeout=120, follow_redirects=True)
    print(f"Accept={h['Accept'][:45]:45s} -> {r.status_code}  {len(r.content):,} bytes")
    if r.status_code == 200 and r.content:
        resp = r
        break
if resp is None:
    raise SystemExit("No 200 from the dataflow catalogue. Stop and report this.")

payload = resp.json()
pathlib.Path("data/raw").mkdir(parents=True, exist_ok=True)
pathlib.Path("data/raw/imf_dataflows_raw.json").write_text(
    json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

def find_flows(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "dataflows" and isinstance(v, list):
                return v
            hit = find_flows(v)
            if hit:
                return hit
    elif isinstance(obj, list):
        for item in obj:
            hit = find_flows(item)
            if hit:
                return hit
    return None

flows = find_flows(payload) or []
print(f"\nParsed {len(flows)} dataflows")

def flatten_name(n):
    if isinstance(n, dict):
        return n.get("en") or next(iter(n.values()), "")
    return n or ""

rows = [{
    "id": f.get("id"),
    "agency": f.get("agencyID") or f.get("agency"),
    "version": f.get("version"),
    "name": flatten_name(f.get("name") or f.get("names")),
} for f in flows]

df = pd.DataFrame(rows).sort_values(["agency", "id"]).reset_index(drop=True)
df.to_csv("data/raw/imf_dataflows.csv", index=False, encoding="utf-8")
print(f"Wrote data/raw/imf_dataflows.csv\n")

print("Agencies:", df["agency"].value_counts().to_dict(), "\n")

KEYWORDS = ["exchange", "reserve", "monetar", "financial", "price",
            "consumer", "trade", "debt", "payment", "government finance",
            "bank", "credit", "external", "liquidity"]
mask = df["name"].str.lower().str.contains("|".join(KEYWORDS), na=False)
pd.set_option("display.max_colwidth", 70, "display.width", 200)
print(df[mask].to_string(index=False))