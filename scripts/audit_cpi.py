"""Diagnose the CPI pull: duplicate vintages, reference periods, real coverage."""
import pathlib, sys
import pandas as pd

raw = pd.read_parquet("data/interim/imf_cpi_raw.parquet")

print("ROWS:", f"{len(raw):,}")
for col in ("REFERENCE_PERIOD", "COMMON_REFERENCE_PERIOD", "INDEX_TYPE",
            "TYPE_OF_TRANSFORMATION", "UNIT", "OVERLAP", "IFS_FLAG", "DERIVATION_TYPE"):
    if col in raw.columns:
        v = raw[col].dropna().unique()
        print(f"{col:<26} {len(v)} distinct -> {list(v)[:8]}")

print("\nDUPLICATES per COUNTRY+TIME_PERIOD:")
d = raw.groupby(["COUNTRY", "TIME_PERIOD"]).size()
print(d.value_counts().rename("rows_per_month").to_string())

print("\nKAZ 2015-M01 — every row returned:")
k = raw[(raw.COUNTRY == "KAZ") & (raw.TIME_PERIOD == "2015-M01")]
cols = [c for c in ("REFERENCE_PERIOD", "COMMON_REFERENCE_PERIOD", "OBS_VALUE",
                    "OVERLAP", "IFS_FLAG", "DERIVATION_TYPE") if c in k.columns]
print(k[cols].to_string(index=False))

print("\nTUR — rows at 1955-M01 and 2026-M07:")
t = raw[(raw.COUNTRY == "TUR") & (raw.TIME_PERIOD.isin(["1955-M01", "2026-M07"]))]
print(t[["TIME_PERIOD"] + cols].to_string(index=False))

print("\nREAL COVERAGE if we keep the LONGEST reference period per country:")
if "REFERENCE_PERIOD" in raw.columns:
    best = (raw.groupby(["COUNTRY", "REFERENCE_PERIOD"]).size()
               .rename("n").reset_index()
               .sort_values("n", ascending=False)
               .groupby("COUNTRY").first())
    print(best.to_string())