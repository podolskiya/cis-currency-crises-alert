import io, pathlib, sys
import httpx, pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from src.config import IMF_BASE, IL, SDMX_KEY, COUNTRIES

frames = []
for ind, unit in IL["indicators"].items():
    key = f"{SDMX_KEY}.{ind}.{unit}.{IL['freq']}"
    url = f"{IMF_BASE}/data/dataflow/{IL['agency']}/{IL['flow']}/~/{key}"
    r = httpx.get(url, headers={"Accept": "text/csv"}, timeout=300, follow_redirects=True)
    print(f"{ind:<16} unit={unit:<5} -> {r.status_code}  {len(r.content):,} bytes")
    if r.status_code == 204:
        print(f"   no content for {ind}")
        continue
    r.raise_for_status()
    frames.append(pd.read_csv(io.StringIO(r.text)))

if not frames:
    raise SystemExit("Nothing returned. Stop here and report.")

raw = pd.concat(frames, ignore_index=True)
pathlib.Path("data/interim").mkdir(parents=True, exist_ok=True)
raw.to_parquet("data/interim/imf_il_raw.parquet", index=False)

print(f"\nROWS: {len(raw):,}")
missing = set(COUNTRIES) - set(raw["COUNTRY"].unique())
print(f"Countries with no reserves data: {sorted(missing) or 'none'}")

print("\nCOVERAGE (reserves ex gold):")
sub = raw[raw["INDICATOR"] == "RXF11_REVS"]
print(sub.groupby("COUNTRY")["TIME_PERIOD"].agg(["min", "max", "count"]).to_string())

print("\nRESERVE LEVELS, USD MILLIONS, LATEST OBSERVATION:")
latest = (sub.sort_values("TIME_PERIOD").groupby("COUNTRY")
             .agg(date=("TIME_PERIOD", "last"), usd_mn=("OBS_VALUE", "last")))
latest["usd_mn"] = (latest["usd_mn"] / 1e6).round(1)
print(latest.to_string())