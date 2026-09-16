import pathlib, sys
import numpy as np, pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from src.config import COUNTRIES

raw = pd.read_parquet("data/interim/imf_er_raw.parquet")

df = raw[["COUNTRY", "ANCHOR", "TYPE_OF_TRANSFORMATION", "TIME_PERIOD", "OBS_VALUE"]].copy()
df["date"] = pd.PeriodIndex(df["TIME_PERIOD"].str.replace("-M", "-", regex=False), freq="M")

anchor_map = {c: v["anchor"] for c, v in COUNTRIES.items()}
df["want"] = df["COUNTRY"].map(anchor_map)
kept = df[df["ANCHOR"] == df["want"]].copy()
print("Anchor selection:")
print(kept.groupby(["COUNTRY", "ANCHOR"]).size().rename("rows").to_string())

wide = (kept.pivot_table(index=["COUNTRY", "date"],
                         columns="TYPE_OF_TRANSFORMATION", values="OBS_VALUE")
            .rename(columns={"EOP_RT": "fx_eop", "PA_RT": "fx_pa"})
            .reset_index())

parts = []
for c, g in wide.groupby("COUNTRY", sort=False):
    grid = pd.period_range(g["date"].min(), g["date"].max(), freq="M")
    g = g.set_index("date").reindex(grid).rename_axis("date").reset_index()
    g["COUNTRY"] = c
    parts.append(g)
panel = pd.concat(parts, ignore_index=True).sort_values(["COUNTRY", "date"])

for src, dst in (("fx_eop", "dlog_eop"), ("fx_pa", "dlog_pa")):
    panel[dst] = panel.groupby("COUNTRY")[src].transform(lambda s: np.log(s).diff())

panel["anchor"] = panel["COUNTRY"].map(anchor_map)
pathlib.Path("data/processed").mkdir(parents=True, exist_ok=True)
panel.to_parquet("data/processed/er_monthly.parquet", index=False)

print("\nCOMPLETENESS:")
rep = (panel.groupby("COUNTRY")
            .agg(anchor=("anchor", "first"), start=("date", "min"), end=("date", "max"),
                 months=("date", "size"), eop=("fx_eop", "count"), pa=("fx_pa", "count")))
rep["eop_miss_pct"] = (100 * (1 - rep.eop / rep.months)).round(1)
print(rep.to_string())

print("\nFX VOLATILITY FROM 1999 (sd of monthly log change, EOP) — regime check:")
sub = panel[panel["date"] >= pd.Period("1999-01", freq="M")]
vol = sub.groupby("COUNTRY")["dlog_eop"].std().sort_values()
reg = pd.DataFrame({"sd": vol.round(4),
                    "regime": [COUNTRIES[c]["regime"] for c in vol.index],
                    "anchor": [COUNTRIES[c]["anchor"] for c in vol.index]})
print(reg.to_string())