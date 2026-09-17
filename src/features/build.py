import pathlib, sys
import numpy as np, pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from src.config import COUNTRIES

PUB_LAG = 3
PEG_VOL = 0.010

CONTAGION = ["region_emp_loo", "region_stress_share", "rus_empz",
             "bloc_emp_loo", "region_stress_6m", "region_emp_6m", "rus_empz_6m"]

REGIME = ["anchor_eur", "defacto_peg", "fx_vol_24m", "fx_vol_rel_region",
          "months_since_regime_flip", "log_res_usd"]

REER = ["rer_vs_36m", "rer_vs_60m", "rer_chg_12m",
        "infl_12m", "infl_diff_anchor", "infl_accel"]


def feats(g):
    g = g.copy()
    fx = g["fx_eop"].shift(PUB_LAG)
    res = g["res_exgold"].shift(PUB_LAG)
    resfx = g["res_fx"].shift(PUB_LAG)
    emp = g["emp"].shift(PUB_LAG)

    lfx, lres = np.log(fx), np.log(res.where(res > 0))

    for h in (3, 6, 12):
        g[f"fx_chg_{h}m"] = lfx.diff(h)
        g[f"res_chg_{h}m"] = lres.diff(h)
        g[f"emp_ma_{h}m"] = emp.rolling(h).mean()

    g["fx_vol_12m"] = lfx.diff().rolling(12).std()
    g["res_vol_12m"] = lres.diff().rolling(12).std()
    g["res_vs_24m_max"] = lres - lres.rolling(24).max()
    g["res_vs_36m_mean"] = lres - lres.rolling(36).mean()

    share = (resfx / res).clip(0, 1)
    g["fx_share"] = share
    g["fx_share_chg_6m"] = share.diff(6)
    g["fx_vs_36m_trend"] = lfx - lfx.rolling(36).mean()

    g["fx_vol_24m"] = lfx.diff().rolling(24).std()
    g["defacto_peg"] = (g["fx_vol_24m"] < PEG_VOL).astype(float)
    g.loc[g["fx_vol_24m"].isna(), "defacto_peg"] = np.nan
    g["log_res_usd"] = lres

    flip = g["defacto_peg"].diff().abs().fillna(0) > 0
    since = np.zeros(len(g)); ctr = 0
    for i, f in enumerate(flip.to_numpy()):
        ctr = 0 if f else ctr + 1
        since[i] = ctr
    g["months_since_regime_flip"] = since
    return g


p = pd.read_parquet("data/processed/target.parquet").sort_values(["COUNTRY", "date"]).reset_index(drop=True)

parts = []
for country, g in p.groupby("COUNTRY", sort=False):
    out = feats(g)
    out["COUNTRY"] = country
    parts.append(out)
p = pd.concat(parts, ignore_index=True)

p["anchor_eur"] = p["COUNTRY"].map(
    {c: 1.0 if v["anchor"] == "EUR" else 0.0 for c, v in COUNTRIES.items()})
p["fx_vol_rel_region"] = p["fx_vol_24m"] / p.groupby("date")["fx_vol_24m"].transform("median")
p["fx_vol_rel_region"] = p["fx_vol_rel_region"].replace([np.inf, -np.inf], np.nan)

cg = pd.read_parquet("data/processed/contagion.parquet")
p = p.merge(cg[["COUNTRY", "date"] + CONTAGION], on=["COUNTRY", "date"], how="left")

rr = pd.read_parquet("data/processed/reer.parquet")
p = p.merge(rr[["COUNTRY", "date"] + REER], on=["COUNTRY", "date"], how="left")

COUNTRY_FEATURES = [c for c in p.columns if any(
    c.startswith(s) for s in ("fx_chg", "res_chg", "emp_ma", "fx_vol_12",
                              "res_vol", "res_vs", "fx_share_chg", "fx_vs_36"))] + ["fx_share"]
FEATURES = COUNTRY_FEATURES + REGIME + CONTAGION + REER

for c in CONTAGION + REER + ["fx_vol_rel_region", "months_since_regime_flip"]:
    p[c] = p[c].fillna(p.groupby("date")[c].transform("median")).fillna(0.0)

CORE = COUNTRY_FEATURES + ["fx_vol_24m", "defacto_peg", "log_res_usd"]
p["complete"] = p[CORE].notna().all(axis=1)
p["modelable"] = p["trainable"] & p["complete"]
p.to_parquet("data/processed/features.parquet", index=False)

print(f"Country ({len(COUNTRY_FEATURES)}): {', '.join(COUNTRY_FEATURES)}")
print(f"Regime ({len(REGIME)}): {', '.join(REGIME)}")
print(f"Contagion ({len(CONTAGION)}): {', '.join(CONTAGION)}")
print(f"REER ({len(REER)}): {', '.join(REER)}")

m = p[p["modelable"]]
print(f"\nTrainable: {int(p['trainable'].sum()):,} -> modelable: {len(m):,}")
print(f"Positives: {int(m['y'].sum()):,} ({100*m['y'].mean():.1f}%)")

print("\nREER IMPUTATION RATE (share of modelable rows imputed, by country):")
rr_have = rr.dropna(subset=["rer_vs_36m"])[["COUNTRY", "date"]].assign(have=1)
chk = m[["COUNTRY", "date"]].merge(rr_have, on=["COUNTRY", "date"], how="left")
print((1 - chk.groupby("COUNTRY")["have"].mean().fillna(0)).round(3).to_string())

print("\nBY COUNTRY:")
print(m.groupby("COUNTRY").agg(n=("y", "size"), pos=("y", "sum")).to_string())