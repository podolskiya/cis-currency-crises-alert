"""Exchange market pressure index — regime-aware, anchor-aware.

Weighting: inverse standard deviation per country (Kaminsky-Reinhart-Vegh),
so components are comparable across economies that absorb pressure differently.
Moments: expanding and lagged, so every label uses only information a
contemporary analyst would have had.
Currency-board countries use a reserve-only variant, because their FX term is
degenerate by construction and would otherwise dominate the inverse-SD weight.
"""
import pathlib, sys
import numpy as np, pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from src.config import (PANEL_START, COUNTRY_START_OVERRIDE, EXCLUDE_FROM_MODEL,
                        POLICY_REFORM_EVENTS, PREFER_PA, RESERVE_ONLY_EMP, COUNTRIES)

MIN_OBS = 24
THRESHOLD_SD = 2.5
WINDOW_EXCL = 18
MIN_FX_MOVE = 0.02          # corroboration filter for FX-bearing countries
MIN_RES_DROP = -0.05        # corroboration for reserve-only countries

p = pd.read_parquet("data/processed/panel_monthly.parquet")
p = p[~p["COUNTRY"].isin(EXCLUDE_FROM_MODEL)].copy()

floor = pd.Period(PANEL_START, freq="M")
p = p[p["date"] >= floor]
for c, start in COUNTRY_START_OVERRIDE.items():
    p = p[~((p["COUNTRY"] == c) & (p["date"] < pd.Period(start, freq="M")))]

p = p.sort_values(["COUNTRY", "date"]).reset_index(drop=True)
p["fx_ret"] = np.where(p["COUNTRY"].isin(PREFER_PA), p["dlog_pa"], p["dlog_eop"])
p["res_ret"] = p["dlog_res"]
p["reserve_only"] = p["COUNTRY"].isin(RESERVE_ONLY_EMP)

sd = p.groupby("COUNTRY")[["fx_ret", "res_ret"]].std()
inv = 1 / sd
w = inv.div(inv.sum(axis=1), axis=0).rename(columns={"fx_ret": "w_fx", "res_ret": "w_res"})
# Currency boards: all weight on reserves
w.loc[w.index.isin(RESERVE_ONLY_EMP), ["w_fx", "w_res"]] = [0.0, 1.0]
p = p.merge(w, left_on="COUNTRY", right_index=True)

p["emp"] = p["w_fx"] * p["fx_ret"] - p["w_res"] * p["res_ret"]

g = p.groupby("COUNTRY")["emp"]
p["emp_mu"] = g.transform(lambda s: s.expanding(MIN_OBS).mean().shift(1))
p["emp_sd"] = g.transform(lambda s: s.expanding(MIN_OBS).std().shift(1))
p["emp_z"] = (p["emp"] - p["emp_mu"]) / p["emp_sd"]

corroborated = np.where(p["reserve_only"],
                        p["res_ret"] < MIN_RES_DROP,
                        p["fx_ret"] > MIN_FX_MOVE)
p["crisis_raw"] = (p["emp_z"] > THRESHOLD_SD) & corroborated

reform = {(c, pd.Period(d, freq="M")) for c, d in POLICY_REFORM_EVENTS}
p["is_reform"] = [(c, d) in reform for c, d in zip(p["COUNTRY"], p["date"])]
p.loc[p["is_reform"], "crisis_raw"] = False


def dedupe(g):
    flags, last = [], None
    for d, f in zip(g["date"], g["crisis_raw"]):
        if f and (last is None or (d - last).n > WINDOW_EXCL):
            flags.append(True); last = d
        else:
            flags.append(False)
    return pd.Series(flags, index=g.index)


p["crisis"] = p.groupby("COUNTRY", group_keys=False).apply(dedupe)
p.to_parquet("data/processed/emp.parquet", index=False)

print("EMP WEIGHTS")
wt = w.round(3).copy()
wt["regime"] = [COUNTRIES[c]["regime"] for c in wt.index]
print(wt.to_string())

print(f"\nUsable obs: {p['emp_z'].notna().sum():,} of {len(p):,}")
print(f"Crisis episodes: {int(p['crisis'].sum())}")

print("\nBY COUNTRY:")
print(p.groupby("COUNTRY").agg(n=("emp_z", "count"), episodes=("crisis", "sum")).to_string())

print("\nALL DETECTED EPISODES:")
ep = p[p["crisis"]][["COUNTRY", "date", "emp_z", "fx_ret", "res_ret"]]
print(ep.assign(fx_pct=(100 * (np.exp(ep.fx_ret) - 1)).round(1),
                res_pct=(100 * (np.exp(ep.res_ret) - 1)).round(1),
                emp_z=ep.emp_z.round(2))
        .drop(columns=["fx_ret", "res_ret"]).to_string(index=False))