import pathlib, sys
import numpy as np, pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from src.config import COUNTRIES

PUB_LAG = 3
STRESS_Z = 1.5  

p = pd.read_parquet("data/processed/emp.parquet").sort_values(["COUNTRY", "date"])

parts = []
for c, g in p.groupby("COUNTRY", sort=False):
    g = g.copy()
    g["emp_lag"] = g["emp"].shift(PUB_LAG)
    g["empz_lag"] = g["emp_z"].shift(PUB_LAG)
    g["COUNTRY"] = c
    parts.append(g)
p = pd.concat(parts, ignore_index=True)

p["in_stress"] = (p["empz_lag"] > STRESS_Z).astype(float)
p.loc[p["empz_lag"].isna(), "in_stress"] = np.nan

by_date = p.groupby("date").agg(
    sum_emp=("emp_lag", "sum"), n_emp=("emp_lag", "count"),
    sum_stress=("in_stress", "sum"), n_stress=("in_stress", "count"),
    med_emp=("emp_lag", "median"))
p = p.merge(by_date, left_on="date", right_index=True, how="left")

p["region_emp_loo"] = (p["sum_emp"] - p["emp_lag"].fillna(0)) / (p["n_emp"] - p["emp_lag"].notna())
p["region_stress_share"] = ((p["sum_stress"] - p["in_stress"].fillna(0))
                            / (p["n_stress"] - p["in_stress"].notna()))
p["emp_vs_region_median"] = p["emp_lag"] - p["med_emp"]

rus = (p[p["COUNTRY"] == "RUS"][["date", "emp_lag", "empz_lag"]]
       .rename(columns={"emp_lag": "rus_emp", "empz_lag": "rus_empz"}))
p = p.merge(rus, on="date", how="left")
p.loc[p["COUNTRY"] == "RUS", ["rus_emp", "rus_empz"]] = np.nan

p["anchor_grp"] = p["COUNTRY"].map({c: v["anchor"] for c, v in COUNTRIES.items()})
blk = p.groupby(["date", "anchor_grp"]).agg(
    blk_sum=("emp_lag", "sum"), blk_n=("emp_lag", "count")).reset_index()
p = p.merge(blk, on=["date", "anchor_grp"], how="left")
p["bloc_emp_loo"] = (p["blk_sum"] - p["emp_lag"].fillna(0)) / (p["blk_n"] - p["emp_lag"].notna())

parts = []
for c, g in p.groupby("COUNTRY", sort=False):
    g = g.sort_values("date").copy()
    g["region_stress_6m"] = g["region_stress_share"].rolling(6).mean()
    g["region_emp_6m"] = g["region_emp_loo"].rolling(6).mean()
    g["rus_empz_6m"] = g["rus_empz"].rolling(6).mean()
    g["COUNTRY"] = c
    parts.append(g)
p = pd.concat(parts, ignore_index=True)

CONTAGION = ["region_emp_loo", "region_stress_share", "emp_vs_region_median",
             "rus_empz", "bloc_emp_loo", "region_stress_6m", "region_emp_6m",
             "rus_empz_6m"]

keep = ["COUNTRY", "date"] + CONTAGION
p[keep].to_parquet("data/processed/contagion.parquet", index=False)

print(f"Rows: {len(p):,}   features: {len(CONTAGION)}")
print("\nMISSINGNESS:")
print((100 * p[CONTAGION].isna().mean()).round(1).to_string())

print("\nLEAK CHECK — correlation of each contagion feature with OWN emp_lag")
print("(should be modest; a near-1 value means own value leaked in)")
print(p[CONTAGION + ["emp_lag"]].corr()["emp_lag"].drop("emp_lag").round(3).to_string())

print("\nREGIONAL STRESS SHARE, PEAK MONTHS (share of other countries in stress):")
peak = (p.groupby("date")["region_stress_share"].mean()
          .dropna().sort_values(ascending=False).head(12))
print(peak.round(3).to_string())