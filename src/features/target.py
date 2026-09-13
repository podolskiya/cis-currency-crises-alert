import pathlib, sys
import numpy as np, pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

HORIZON = 12      
POST_EXCL = 12   

p = pd.read_parquet("data/processed/emp.parquet").sort_values(["COUNTRY", "date"]).reset_index(drop=True)

def build(g):
    g = g.copy()
    n = len(g)
    crisis = g["crisis"].to_numpy()
    y = np.zeros(n, dtype=bool)
    post = np.zeros(n, dtype=bool)
    for i in np.flatnonzero(crisis):
        y[max(0, i - HORIZON):i] = True
        post[i:min(n, i + POST_EXCL + 1)] = True
    g["y"] = y
    g["post_crisis"] = post
    return g

parts = []
for country, g in p.groupby("COUNTRY", sort=False):
    out = build(g)
    out["COUNTRY"] = country          # explicit, never lost
    parts.append(out)
p = pd.concat(parts, ignore_index=True)

# Trainable rows: scored, and not inside a crisis/recovery window
p["trainable"] = p["emp_z"].notna() & ~p["post_crisis"]
p.to_parquet("data/processed/target.parquet", index=False)

tr = p[p["trainable"]]
print(f"Total rows:      {len(p):,}")
print(f"Trainable rows:  {len(tr):,}")
print(f"Positives (y=1): {int(tr['y'].sum()):,}  ({100*tr['y'].mean():.1f}%)")
print(f"Negatives:       {int((~tr['y']).sum()):,}")

print("\nBY COUNTRY:")
print(tr.groupby("COUNTRY").agg(n=("y", "size"), pos=("y", "sum"),
                                rate=("y", "mean")).round(3).to_string())

print("\nPOSITIVES BY YEAR:")
yr = tr.assign(year=tr["date"].dt.year).groupby("year")["y"].agg(["size", "sum"])
print(yr[yr["sum"] > 0].to_string())