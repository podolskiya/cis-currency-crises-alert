import pathlib, sys
import numpy as np, pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

PUB_LAG = 3   # months; conservative uniform lag on all IMF inputs

p = pd.read_parquet("data/processed/target.parquet").sort_values(["COUNTRY", "date"]).reset_index(drop=True)

def feats(g):
    g = g.copy()
    # Apply publication lag to every raw input first
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

    # Reserve level vs its own recent history (adequacy proxy without imports)
    g["res_vs_24m_max"] = lres - lres.rolling(24).max()
    g["res_vs_36m_mean"] = lres - lres.rolling(36).mean()

    # Composition: FX share of non-gold reserves, and its drift
    share = (resfx / res).clip(0, 1)
    g["fx_share"] = share
    g["fx_share_chg_6m"] = share.diff(6)

    # Real appreciation proxy: FX vs trailing trend (overvaluation signal)
    g["fx_vs_36m_trend"] = lfx - lfx.rolling(36).mean()
    return g

parts = []
for country, g in p.groupby("COUNTRY", sort=False):
    out = feats(g)
    out["COUNTRY"] = country
    parts.append(out)
p = pd.concat(parts, ignore_index=True)

FEATURES = [c for c in p.columns if any(
    c.startswith(s) for s in ("fx_chg", "res_chg", "emp_ma", "fx_vol",
                              "res_vol", "res_vs", "fx_share_chg", "fx_vs"))] + ["fx_share"]

p["complete"] = p[FEATURES].notna().all(axis=1)
p["modelable"] = p["trainable"] & p["complete"]
p.to_parquet("data/processed/features.parquet", index=False)

print(f"Features ({len(FEATURES)}):")
for f in FEATURES:
    print(f"  {f}")

m = p[p["modelable"]]
print(f"\nTrainable: {int(p['trainable'].sum()):,} -> modelable: {len(m):,}")
print(f"Positives: {int(m['y'].sum()):,} ({100*m['y'].mean():.1f}%)")

print("\nBY COUNTRY:")
print(m.groupby("COUNTRY").agg(n=("y", "size"), pos=("y", "sum"),
                               first=("date", "min")).to_string())

print("\nMISSINGNESS PER FEATURE (on trainable rows):")
miss = (100 * p[p["trainable"]][FEATURES].isna().mean()).round(1).sort_values(ascending=False)
print(miss.to_string())