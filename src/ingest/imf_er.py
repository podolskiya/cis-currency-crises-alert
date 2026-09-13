import io, pathlib, sys
import httpx, pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from src.config import IMF_BASE, ER, SDMX_KEY, COUNTRIES

OUT = pathlib.Path("data/interim")


def fetch(transformation: str) -> pd.DataFrame:
    key = f"{SDMX_KEY}.{ER['indicator']}.{transformation}.{ER['freq']}"
    url = f"{IMF_BASE}/data/dataflow/{ER['agency']}/{ER['flow']}/~/{key}"
    r = httpx.get(url, headers={"Accept": "text/csv"}, timeout=300, follow_redirects=True)
    print(f"{transformation}: {r.status_code}  {len(r.content):,} bytes")
    if r.status_code == 204:
        raise SystemExit(f"204 No Content for key {key} — the key matched nothing.")
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df["TRANSFORMATION"] = transformation
    return df


frames = [fetch(t) for t in ER["transformations"]]
raw = pd.concat(frames, ignore_index=True)

OUT.mkdir(parents=True, exist_ok=True)
raw.to_parquet(OUT / "imf_er_raw.parquet", index=False)

print("\nCOLUMNS:", list(raw.columns))
print(f"\nROWS: {len(raw):,}")
print("\nHEAD:")
print(raw.head(8).to_string())

ccol = next((c for c in raw.columns if c.upper() == "COUNTRY"), None)
tcol = next((c for c in raw.columns if "TIME" in c.upper()), None)
if ccol and tcol:
    cov = (raw.groupby([ccol, "TRANSFORMATION"])[tcol]
              .agg(["min", "max", "count"]).reset_index())
    print("\nCOVERAGE BY COUNTRY:")
    print(cov.to_string(index=False))
    missing = set(COUNTRIES) - set(raw[ccol].unique())
    print(f"\nCountries with no data returned: {sorted(missing) or 'none'}")