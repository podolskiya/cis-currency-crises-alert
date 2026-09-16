import io, pathlib, sys
import httpx, pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from src.config import IMF_BASE, CPI, ISO3, COUNTRIES


def fetch(countries):
    key = (f"{'+'.join(countries)}.{CPI['index_type']}.{CPI['coicop']}"
           f".{CPI['transformation']}.{CPI['freq']}")
    url = f"{IMF_BASE}/data/dataflow/{CPI['agency']}/{CPI['flow']}/~/{key}"
    r = httpx.get(url, headers={"Accept": "text/csv"}, timeout=300, follow_redirects=True)
    print(f"{'+'.join(countries)[:40]:<42} -> {r.status_code}  {len(r.content):,} bytes")
    if r.status_code == 204 or not r.content:
        return None
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))


frames = []
d = fetch(ISO3)
if d is not None:
    frames.append(d)

for anchor, code in CPI["anchors"].items():
    d = fetch([code])
    if d is None or len(d) == 0:
        print(f"  WARNING: no CPI for anchor {anchor} ({code})")
    else:
        frames.append(d)

raw = pd.concat(frames, ignore_index=True)
pathlib.Path("data/interim").mkdir(parents=True, exist_ok=True)
raw.to_parquet("data/interim/imf_cpi_raw.parquet", index=False)

print(f"\nROWS: {len(raw):,}")
print("COLUMNS:", list(raw.columns))

missing = set(ISO3) - set(raw["COUNTRY"].unique())
print(f"\nPanel countries with no CPI: {sorted(missing) or 'none'}")

print("\nCOVERAGE:")
cov = raw.groupby("COUNTRY")["TIME_PERIOD"].agg(["min", "max", "count"])
cov["is_anchor"] = [c not in ISO3 for c in cov.index]
print(cov.to_string())

print("\nSANITY — Turkey CPI index, last 6 observations (should rise steeply):")
t = raw[raw["COUNTRY"] == "TUR"].sort_values("TIME_PERIOD")
print(t[["TIME_PERIOD", "OBS_VALUE"]].tail(6).to_string(index=False))