import pathlib, sys
import numpy as np, pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from src.config import COUNTRIES, CPI, ISO3

raw = pd.read_parquet("data/interim/imf_cpi_raw.parquet")
cpi = raw[["COUNTRY", "TIME_PERIOD", "OBS_VALUE"]].copy()
cpi["date"] = pd.PeriodIndex(cpi["TIME_PERIOD"].str.replace("-M", "-", regex=False), freq="M")
cpi = cpi[cpi["OBS_VALUE"] > CPI["min_index"]]
cpi = cpi.rename(columns={"OBS_VALUE": "cpi"})[["COUNTRY", "date", "cpi"]]

anchors = {v: k for k, v in CPI["anchors"].items()}   # USA->USD, DEU->EUR
anc = cpi[cpi["COUNTRY"].isin(anchors)].copy()
anc["anchor"] = anc["COUNTRY"].map(anchors)
anc = anc.pivot_table(index="date", columns="anchor", values="cpi")
anc.columns = [f"cpi_anchor_{c}" for c in anc.columns]

dom = cpi[cpi["COUNTRY"].isin(ISO3)]

p = pd.read_parquet("data/processed/panel_monthly.parquet")
p = p.merge(dom, on=["COUNTRY", "date"], how="left").merge(anc, on="date", how="left")
p["cpi_anchor"] = np.where(p["anchor"].eq("EUR"), p["cpi_anchor_EUR"], p["cpi_anchor_USD"])

parts = []
for c, g in p.sort_values("date").groupby("COUNTRY", sort=False):
    g = g.copy()
    lrer = np.log(g["fx_eop"]) + np.log(g["cpi_anchor"]) - np.log(g["cpi"])
    g["log_rer"] = lrer
    g["rer_vs_36m"] = lrer - lrer.rolling(36, min_periods=24).mean()
    g["rer_vs_60m"] = lrer - lrer.rolling(60, min_periods=36).mean()
    g["rer_chg_12m"] = lrer.diff(12)
    infl = np.log(g["cpi"]).diff(12)
    g["infl_12m"] = infl
    g["infl_diff_anchor"] = infl - np.log(g["cpi_anchor"]).diff(12)
    g["infl_accel"] = infl - infl.shift(12)
    g["COUNTRY"] = c
    parts.append(g)
p = pd.concat(parts, ignore_index=True)

REER = ["rer_vs_36m", "rer_vs_60m", "rer_chg_12m",
        "infl_12m", "infl_diff_anchor", "infl_accel"]
p[["COUNTRY", "date"] + REER + ["log_rer", "cpi"]].to_parquet(
    "data/processed/reer.parquet", index=False)

print("CPI COVERAGE MERGED INTO PANEL:")
print(p.groupby("COUNTRY").agg(n=("date", "size"), cpi_obs=("cpi", "count"),
                               rer_obs=("log_rer", "count")).to_string())

print("\nMISSINGNESS (from 1999):")
sub = p[p["date"] >= pd.Period("1999-01", freq="M")]
print((100 * sub[REER].isna().mean()).round(1).to_string())

print("\nREAL OVERVALUATION vs 36m trend — most overvalued months on record")
print("(negative rer_vs_36m = real appreciation = overvaluation)")
ov = sub.nsmallest(15, "rer_vs_36m")[["COUNTRY", "date", "rer_vs_36m"]]
print(ov.assign(pct=(100 * (np.exp(ov.rer_vs_36m) - 1)).round(1)).to_string(index=False))