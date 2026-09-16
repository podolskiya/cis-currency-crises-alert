import io, pathlib, sys
import httpx, pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from src.config import IMF_BASE, ER, SDMX_KEY, COUNTRIES

OUT = pathlib.Path("data/interim")


def fetch(indicator: str, transformation: str) -> pd.DataFrame | None:
    key = f"{SDMX_KEY}.{indicator}.{transformation}.{ER['freq']}"
    url = f"{IMF_BASE}/data/dataflow/{ER['agency']}/{ER['flow']}/~/{key}"
    r = httpx.get(url, headers={"Accept": "text/csv"}, timeout=300, follow_redirects=True)
    print(f"{indicator:<10} {transformation:<8} -> {r.status_code}  {len(r.content):,} bytes")
    if r.status_code == 204:
        print("   204 no content")
        return None
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df["ANCHOR"] = indicator.split("_")[1]
    df["TRANSFORMATION"] = transformation
    return df


frames = []
for ind in (ER["indicator_usd"], ER["indicator_eur"]):
    for t in ER["transformations"]:
        d = fetch(ind, t)
        if d is not None:
            frames.append(d)

raw = pd.concat(frames, ignore_index=True)
OUT.mkdir(parents=True, exist_ok=True)
raw.to_parquet(OUT / "imf_er_raw.parquet", index=False)

print(f"\nROWS: {len(raw):,}")
missing = set(COUNTRIES) - set(raw["COUNTRY"].unique())
print(f"Countries with no data at all: {sorted(missing) or 'none'}")

print("\nCOVERAGE BY COUNTRY AND ANCHOR (EOP only):")
eop = raw[raw["TRANSFORMATION"] == "EOP_RT"]
cov = eop.groupby(["COUNTRY", "ANCHOR"])["TIME_PERIOD"].agg(["min", "max", "count"])
print(cov.to_string())

print("\nNEW COUNTRIES — check these resolved:")
for c in ("SRB", "ALB", "MKD", "BIH", "MNG", "TUR"):
    sub = eop[eop["COUNTRY"] == c]
    print(f"  {c}: {len(sub)} rows, anchors {sorted(sub['ANCHOR'].unique()) or 'NONE'}")