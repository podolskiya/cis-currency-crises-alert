"""
Phase 1b (inspection) - BIS Locational Banking Statistics bulk download.

This deliberately does NOT filter or aggregate anything yet. The file
covers every BIS reporting country and every counterparty country
worldwide across decades of quarterly data, and BIS often uses its own
reference-area codes rather than plain ISO3 -- so before writing any
filtering logic, we need to actually see the column layout and how
countries are coded in it.

Run:
    pip install requests pandas
    python inspect_bis_lbs.py
"""

import zipfile
from pathlib import Path

import requests
import pandas as pd

URL = "https://data.bis.org/static/bulk/WS_LBS_D_PUB_csv_flat.zip"
ZIP_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "bis" / "WS_LBS_D_PUB_csv_flat.zip"

# Our 11-country panel, for cross-checking once we know the coding scheme
CIS_ISO3 = ["KAZ", "KGZ", "TJK", "TKM", "UZB", "ARM", "AZE", "GEO", "RUS", "BLR", "MDA"]


def download():
    if ZIP_PATH.exists():
        print(f"{ZIP_PATH} already downloaded ({ZIP_PATH.stat().st_size / 1e6:.1f} MB), skipping.")
        return
    print("Downloading BIS locational banking statistics bulk file (may take a while, could be large)...")
    ZIP_PATH.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(URL, timeout=300, stream=True)
    resp.raise_for_status()
    with open(ZIP_PATH, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    print(f"Downloaded {ZIP_PATH.stat().st_size / 1e6:.1f} MB.")


def inspect():
    # Read directly from inside the zip -- do NOT extract. The CSV inside
    # is ~17.9 GB uncompressed; nrows caps how much pandas actually parses
    # from the stream, so this only ever touches a small slice of it.
    with zipfile.ZipFile(ZIP_PATH) as z:
        print("\nFiles inside zip:")
        for n in z.namelist():
            info = z.getinfo(n)
            print(f"  {n}  ({info.file_size / 1e6:.1f} MB uncompressed)")
        csv_name = next(n for n in z.namelist() if n.lower().endswith(".csv"))

        with z.open(csv_name) as f:
            header = pd.read_csv(f, nrows=0)
        print(f"\nColumns ({len(header.columns)}):")
        for c in header.columns:
            print(f"  {c}")

        with z.open(csv_name) as f:
            sample = pd.read_csv(f, nrows=2000)
    print("\nFirst 10 rows of sample:")
    print(sample.head(10).to_string())

    # Find whichever column(s) look like they identify country, and show
    # their unique values so we can confirm how our 11 countries appear
    likely_cols = [c for c in sample.columns if any(k in c.upper() for k in ("COUNTRY", "CP_", "AREA", "L_REP"))]
    print(f"\nCandidate country-identifying columns: {likely_cols}")
    for c in likely_cols:
        vals = sample[c].dropna().unique()
        print(f"\n  Unique sample values in '{c}' ({len(vals)} shown, may not be exhaustive from a 2000-row sample):")
        print(sorted(str(v) for v in vals)[:40])


if __name__ == "__main__":
    download()
    inspect()