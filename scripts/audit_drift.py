import pathlib, sys
import numpy as np, pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

h = pd.DataFrame(pd.read_json("web/public/data/history.json"))
h["date"] = pd.PeriodIndex(h["date"], freq="M")
h["year"] = h["date"].dt.year

print("CROSS-SECTIONAL MEAN PROBABILITY BY YEAR")
yr = h.groupby("year").agg(mean_prob=("probability", "mean"),
                           median_prob=("probability", "median"),
                           n=("probability", "size"),
                           crises=("crisis", "sum"))
print(yr.round(3).to_string())

print("\nSHARE OF PANEL ABOVE ITS OWN 90TH PERCENTILE, BY YEAR")
top = h.assign(hot=h["percentile"] > 90).groupby("year")["hot"].mean()
print((100 * top).round(1).to_string())

print("\nTREND CHECK — correlation of probability with time, per country")
out = []
for iso, g in h.groupby("iso3"):
    g = g.sort_values("date")
    t = np.arange(len(g))
    out.append({"iso3": iso, "corr_with_time": round(float(np.corrcoef(t, g["probability"])[0, 1]), 3),
                "first_prob": round(float(g["probability"].iloc[0]), 3),
                "last_prob": round(float(g["probability"].iloc[-1]), 3)})
print(pd.DataFrame(out).sort_values("corr_with_time", ascending=False).to_string(index=False))

print("\nPERCENTILE IF COMPUTED ON LAST 10 YEARS ONLY (vs full history)")
recent = h[h["date"] >= pd.Period("2016-01", freq="M")].copy()
recent["pctile_10y"] = recent.groupby("iso3")["probability"].rank(pct=True) * 100
last = recent.sort_values("date").groupby("iso3").last()
full = h.sort_values("date").groupby("iso3").last()["percentile"]
cmp = pd.DataFrame({"pctile_full": full, "pctile_10y": last["pctile_10y"].round(1)})
cmp["shift"] = (cmp["pctile_10y"] - cmp["pctile_full"]).round(1)
print(cmp.sort_values("pctile_full", ascending=False).to_string())