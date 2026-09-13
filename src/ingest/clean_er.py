import pathlib, sys
import numpy as np, pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from src.config import COUNTRIES

raw = pd.read_parquet("data/interim/imf_er_raw.parquet")
print("SCALE values present:", raw["SCALE"].unique())

df = raw[["COUNTRY", "TYPE_OF_TRANSFORMATION", "TIME_PERIOD", "OBS_VALUE"]].copy()
df["date"] = pd.PeriodIndex(df["TIME_PERIOD"].str.replace("-M", "-", regex=False), freq="M")
df = df.drop(columns="TIME_PERIOD")

wide = (df.pivot_table(index=["COUNTRY", "date"],
                       columns="TYPE_OF_TRANSFORMATION",
                       values="OBS_VALUE")
          .rename(columns={"EOP_RT": "fx_eop", "PA_RT": "fx_pa"})
          .reset_index())

out = []
for c, g in wide.groupby("COUNTRY"):
    grid = pd.period_range(g["date"].min(), g["date"].max(), freq="M")
    g = g.set_index("date").reindex(grid)
    g["COUNTRY"] = c
    out.append(g.rename_axis("date").reset_index())
panel = pd.concat(out, ignore_index=True).sort_values(["COUNTRY", "date"])

panel["dlog_eop"] = panel.groupby("COUNTRY")["fx_eop"].transform(lambda s: np.log(s).diff())
panel["dlog_pa"] = panel.groupby("COUNTRY")["fx_pa"].transform(lambda s: np.log(s).diff())

pathlib.Path("data/processed").mkdir(parents=True, exist_ok=True)
panel.to_parquet("data/processed/er_monthly.parquet", index=False)

print("\n" + "=" * 78)
print("COMPLETENESS (on each country's own span)")
print("=" * 78)
rep = (panel.groupby("COUNTRY")
            .agg(span_start=("date", "min"), span_end=("date", "max"),
                 months=("date", "size"),
                 eop_obs=("fx_eop", "count"), pa_obs=("fx_pa", "count")))
rep["eop_missing_pct"] = (100 * (1 - rep.eop_obs / rep.months)).round(1)
rep["pa_missing_pct"] = (100 * (1 - rep.pa_obs / rep.months)).round(1)
print(rep.to_string())

print("\n" + "=" * 78)
print("SUSPECTED BREAKS  (|monthly log change| > 1.6, i.e. ~5x)")
print("=" * 78)
brk = panel[panel["dlog_eop"].abs() > 1.6][["COUNTRY", "date", "fx_eop", "dlog_eop"]]
if brk.empty:
    print("  none — series appear chained through redenominations")
else:
    brk = brk.assign(implied_ratio=np.exp(brk["dlog_eop"]).round(1))
    print(brk.to_string(index=False))

print("\n" + "=" * 78)
print("LARGEST MONTHLY DEPRECIATIONS (top 15, EOP)")
print("=" * 78)
top = panel.nlargest(15, "dlog_eop")[["COUNTRY", "date", "dlog_eop"]]
top = top.assign(pct=(100 * (np.exp(top["dlog_eop"]) - 1)).round(1))
print(top.to_string(index=False))