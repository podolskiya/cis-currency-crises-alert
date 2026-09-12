import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config.countries import ISO2_LIST

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ZIP_PATH = PROJECT_ROOT / "data" / "raw" / "bis" / "WS_LBS_D_PUB_csv_flat.zip"
OUT_PATH = PROJECT_ROOT / "data" / "raw" / "bis" / "bis_claims_on_cis.csv"
CHUNK_SIZE = 1_000_000

COL_CP_COUNTRY = "L_CP_COUNTRY:Counterparty country"
COL_REP_CTY = "L_REP_CTY:Reporting country"
COL_POSITION = "L_POSITION:Balance sheet position"
COL_MEASURE = "L_MEASURE:Measure"
COL_CP_SECTOR = "L_CP_SECTOR:Counterparty sector"
COL_INSTR = "L_INSTR:Type of instruments"
COL_DENOM = "L_DENOM:Currency denomination"
COL_CURR_TYPE = "L_CURR_TYPE:Currency type of reporting country"
COL_PARENT_CTY = "L_PARENT_CTY:Parent country"
COL_POS_TYPE = "L_POS_TYPE:Position type"
COL_TIME = "TIME_PERIOD:Time period or range"
COL_VALUE = "OBS_VALUE:Observation Value"

USECOLS = [
    COL_CP_COUNTRY, COL_REP_CTY, COL_POSITION, COL_MEASURE, COL_CP_SECTOR,
    COL_INSTR, COL_DENOM, COL_CURR_TYPE, COL_PARENT_CTY, COL_POS_TYPE,
    COL_TIME, COL_VALUE,
]

DTYPES = {col: str for col in USECOLS if col != COL_VALUE}
DTYPES[COL_VALUE] = "float64"

CIS_CODES = set(ISO2_LIST)


def code_of(series: pd.Series) -> pd.Series:
    """Extract the leading 'CODE' from BIS's 'CODE: Label' string values."""
    return series.astype(str).str.split(":", n=1).str[0].str.strip()


def matches_filter(chunk: pd.DataFrame) -> pd.Series:
    return (
        code_of(chunk[COL_CP_COUNTRY]).isin(CIS_CODES)
        & (code_of(chunk[COL_REP_CTY]) == "5A")
        & (code_of(chunk[COL_POSITION]) == "C")
        & (code_of(chunk[COL_MEASURE]) == "S")
        & (code_of(chunk[COL_CP_SECTOR]) == "A")
        & (code_of(chunk[COL_INSTR]) == "A")
        & (code_of(chunk[COL_DENOM]) == "TO1")
        & (code_of(chunk[COL_CURR_TYPE]) == "A")
        & (code_of(chunk[COL_PARENT_CTY]) == "5J")
        & (code_of(chunk[COL_POS_TYPE]) == "N")
    )


def main():
    matches = []
    with zipfile.ZipFile(ZIP_PATH) as z:
        csv_name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        with z.open(csv_name) as f:
            reader = pd.read_csv(f, usecols=USECOLS, dtype=DTYPES, chunksize=CHUNK_SIZE)
            for i, chunk in enumerate(reader, start=1):
                mask = matches_filter(chunk)
                if mask.any():
                    matches.append(chunk.loc[mask])
                found = sum(len(m) for m in matches)
                print(f"  chunk {i} ({i * CHUNK_SIZE:,} rows scanned so far) -- {found} matches so far")

    if not matches:
        print("\nNo matching rows found at all. That points to a wrong filter "
              "code somewhere, not a real data gap -- stopping before writing "
              "anything so we can check the codes rather than save an empty file.")
        return

    result = pd.concat(matches, ignore_index=True)
    result["country_code"] = code_of(result[COL_CP_COUNTRY])

    out = (
        result[["country_code", COL_TIME, COL_VALUE]]
        .rename(columns={COL_TIME: "year_quarter", COL_VALUE: "claims_usd_millions"})
        .sort_values(["country_code", "year_quarter"])
    )
    out.to_csv(OUT_PATH, index=False)

    print(f"\nSaved {len(out)} observations to {OUT_PATH}")
    print("\n=== Coverage summary ===")
    print(out.groupby("country_code")["year_quarter"].agg(n_obs="count", min_period="min", max_period="max"))


if __name__ == "__main__":
    main()