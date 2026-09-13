"""Tidy the IL pull and merge with the FX panel."""
import pathlib, sys
import numpy as np, pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

raw = pd.read_parquet("data/interim/imf_il_raw.parquet")
df = raw[["COUNTRY", "INDICATOR", "TIME_PERIOD", "OBS_VALUE"]].copy()
df["date"] = pd.PeriodIndex(df["TIME_PERIOD"].str.replace("-M", "-", regex=False), freq="M")

res = (df.pivot_table(index=["COUNTRY", "date"], columns="INDICATOR", values="OBS_VALUE")
         .rename(columns={"RXF11_REVS": "res_exgold", "RXF11FX_REVS": "res_fx"})
         .reset_index())

er = pd.read_parquet("data/processed/er_monthly.parquet")
panel = er.merge(res, on=["COUNTRY", "date"], how="outer").sort_values(["COUNTRY", "date"])

g = panel.groupby("COUNTRY")
panel["dlog_res"] = g["res_exgold"].transform(lambda s: np.log(s.where(s > 0)).diff())
panel["fx_share_res"] = panel["res_fx"] / panel["res_exgold"]

panel.to_parquet("data/processed/panel_monthly.parquet", index=False)

print("ROWS:", f"{len(panel):,}", "| COLUMNS:", list(panel.columns))

print("\nDATA AGE BY COUNTRY (months behind 2026-09):")
now = pd.Period("2026-09", freq="M")
age = (panel.dropna(subset=["fx_eop"]).groupby("COUNTRY")["date"].max().rename("fx_last")
         .to_frame()
         .join(panel.dropna(subset=["res_exgold"]).groupby("COUNTRY")["date"].max().rename("res_last")))
age["fx_lag"] = age["fx_last"].apply(lambda p: (now - p).n)
age["res_lag"] = age["res_last"].apply(lambda p: (now - p).n)
print(age.to_string())

print("\nOVERLAP: months with BOTH fx and reserves, from 1996-01")
sub = panel[panel["date"] >= pd.Period("1996-01", freq="M")]
ov = (sub.assign(both=sub[["fx_eop", "res_exgold"]].notna().all(axis=1))
        .groupby("COUNTRY")["both"].agg(["sum", "size"]))
ov["pct"] = (100 * ov["sum"] / ov["size"]).round(1)
print(ov.to_string())

print("\nLARGEST MONTHLY RESERVE DRAWDOWNS (top 12, from 1996):")
top = sub.nsmallest(12, "dlog_res")[["COUNTRY", "date", "res_exgold", "dlog_res"]]
print(top.assign(pct=(100 * (np.exp(top["dlog_res"]) - 1)).round(1)).to_string(index=False))