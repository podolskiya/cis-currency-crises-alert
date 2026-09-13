import io, pathlib, sys
import httpx, pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from src.config import IMF_BASE, IL, SDMX_KEY, COUNTRIES

frames = []
for ind, unit in IL["indicators"].items():
    key = f"{SDMX_KEY}.{ind}.{unit}.{IL['freq']}"
    url = f"{IMF_BASE}/data/dataflow/{IL['agency']}/{IL['flow']}/~/{key}"
    r = httpx.get(url, headers={"Accept": "text/csv"}, timeout=300, follow_redirects=True)
    print(f"{ind:<20} unit={unit:<5} -> {r.status_code}  {len(r.content):,} bytes")
    if r.status_code == 204:
        print(f"   no content — check the unit code for {ind}")
        continue
    r.raise_for_status()
    frames.append(pd.read_csv(io.StringIO(r.text)))

if not frames:
    raise SystemExit("Nothing returned. Stop here and report.")

raw = pd.concat(frames, ignore_index=True)
pathlib.Path("data/interim").mkdir(parents=True, exist_ok=True)
raw.to_parquet("data/interim/imf_il_raw.parquet", index=False)

print("\nCOLUMNS:", list(raw.columns))
print(f"ROWS: {len(raw):,}")
print("\nUNIT_MULT values:", raw["UNIT_MULT"].unique() if "UNIT_MULT" in raw else "absent")
print("SCALE values:", raw["SCALE"].unique() if "SCALE" in raw else "absent")

cov = (raw.groupby(["COUNTRY", "INDICATOR"])["TIME_PERIOD"]
          .agg(["min", "max", "count"]).reset_index())
print("\nCOVERAGE:")
print(cov.to_string(index=False))

print("\nSAMPLE (KAZ, reserves ex gold, recent):")
s = raw[(raw.COUNTRY == "KAZ") & (raw.INDICATOR == "RXF11_REVS")]
print(s[["TIME_PERIOD", "OBS_VALUE"]].tail(6).to_string(index=False))