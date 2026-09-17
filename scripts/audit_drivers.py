import pathlib, sys
import numpy as np, pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.model.baseline import COUNTRY_FEATURES, REGIME, pipe

FEATURES = COUNTRY_FEATURES + REGIME

p = pd.read_parquet("data/processed/features.parquet")
train = p[p["modelable"]].reset_index(drop=True)
model = pipe().fit(train[FEATURES], train["y"].astype(int))
scor = p[p["complete"]].copy().reset_index(drop=True)

imp, sc = model.named_steps["impute"], model.named_steps["scale"]
coef = model.named_steps["clf"].coef_[0]
Z = pd.DataFrame(sc.transform(imp.transform(scor[FEATURES])),
                 columns=FEATURES, index=scor.index)

print("STANDARDISED FEATURE VALUES AT LAST ROW — are any extreme (|z| > 4)?")
for iso in sorted(set(scor["COUNTRY"])):
    g = scor[scor["COUNTRY"] == iso].sort_values("date")
    z = Z.loc[g.index[-1]]
    ext = z[z.abs() > 4]
    if len(ext):
        print(f"  {iso}: " + ", ".join(f"{k}={v:+.1f}" for k, v in ext.items()))

print("\nfx_vol_rel_region — distribution across the whole panel")
print(scor["fx_vol_rel_region"].describe(percentiles=[.5, .9, .99]).round(2).to_string())

print("\nBELARUS — full decomposition at last row, all 22 features")
g = scor[scor["COUNTRY"] == "BLR"].sort_values("date")
i = g.index[-1]
d = pd.DataFrame({"raw": g.iloc[-1][FEATURES], "z": Z.loc[i],
                  "coef": coef, "contrib": Z.loc[i].values * coef})
print(d.sort_values("contrib", key=abs, ascending=False).round(3).to_string())
print(f"\nSum of contributions: {d['contrib'].sum():.3f}"
      f"  intercept: {model.named_steps['clf'].intercept_[0]:.3f}")