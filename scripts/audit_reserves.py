import pathlib, sys
import numpy as np, pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

p = pd.read_parquet("data/processed/panel_monthly.parquet")
p = p.dropna(subset=["res_exgold"]).copy()

print("=" * 86)
print("RESERVE LEVELS (USD millions) AT BENCHMARK DATES")
print("=" * 86)
marks = ["1999-12", "2005-12", "2010-12", "2015-12", "2020-12", "2025-12"]
rows = {}
for c, g in p.groupby("COUNTRY"):
    s = g.set_index("date")["res_exgold"]
    rows[c] = {m: (round(s.get(pd.Period(m, freq="M"), np.nan) / 1e6, 1)) for m in marks}
    latest = s.dropna()
    rows[c]["latest"] = round(latest.iloc[-1] / 1e6, 1) if len(latest) else np.nan
print(pd.DataFrame(rows).T.to_string())

print("\n" + "=" * 86)
print("LEVEL DISCONTINUITIES (|monthly log change| > 0.7)")
print("=" * 86)
b = p[p["dlog_res"].abs() > 0.7][["COUNTRY", "date", "res_exgold", "dlog_res"]].copy()
b["usd_mn"] = (b["res_exgold"] / 1e6).round(1)
b["ratio"] = np.exp(b["dlog_res"]).round(2)
print(b[["COUNTRY", "date", "usd_mn", "ratio"]].sort_values(["COUNTRY", "date"]).to_string(index=False))

print("\n" + "=" * 86)
print("RESERVE VOLATILITY BY COUNTRY (sd of monthly log change, from 1996)")
print("=" * 86)
sub = p[p["date"] >= pd.Period("1996-01", freq="M")]
v = sub.groupby("COUNTRY").agg(sd_res=("dlog_res", "std"), n=("dlog_res", "count"))
v["sd_fx"] = sub.groupby("COUNTRY")["dlog_eop"].std()
v["ratio_res_to_fx"] = (v["sd_res"] / v["sd_fx"]).round(2)
print(v.round(4).to_string())