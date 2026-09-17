"""Production scoring: fit the final model on all data, score every country now,
decompose each score, and write the JSON artifacts the dashboard reads.

Model: logistic regression, country fundamentals + regime conditioning.
Chosen over gradient boosting, contagion and REER blocks, each tested on
identical folds and each underperforming. See README for the comparison.
"""
import json, pathlib, sys
from datetime import datetime, timezone

import numpy as np, pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from src.config import COUNTRIES, EXCLUDE_FROM_MODEL
from src.model.baseline import COUNTRY_FEATURES, REGIME, pipe

FEATURES = COUNTRY_FEATURES + REGIME
OUT = pathlib.Path("web/public/data")
RECENT_START = "2016-01"

CV = {
    "mean_auc": 0.654, "sd_auc": 0.036, "min_auc": 0.605,
    "mean_ap": 0.233, "mean_base_rate": 0.139,
    "folds": 4, "fold_span": "2011-03 to 2023-03",
    "last_validated": "2023-03",
}

# Regularisation spreads signal across correlated features, so no single driver
# dominates. Grouped contributions sum to the full score; a top-N list does not.
GROUPS = {
    "Currency pressure": ["fx_chg_3m", "fx_chg_6m", "fx_chg_12m", "fx_vs_36m_trend",
                          "emp_ma_3m", "emp_ma_6m", "emp_ma_12m"],
    "Reserve adequacy": ["res_chg_3m", "res_chg_6m", "res_chg_12m", "res_vs_24m_max",
                         "res_vs_36m_mean", "fx_share", "fx_share_chg_6m", "log_res_usd"],
    "Volatility": ["fx_vol_12m", "fx_vol_24m", "res_vol_12m", "fx_vol_rel_region"],
    "Exchange rate regime": ["anchor_eur", "defacto_peg", "months_since_regime_flip"],
}

READABLE = {
    "fx_chg_3m": "FX change, 3m", "fx_chg_6m": "FX change, 6m",
    "fx_chg_12m": "FX change, 12m", "res_chg_3m": "Reserves change, 3m",
    "res_chg_6m": "Reserves change, 6m", "res_chg_12m": "Reserves change, 12m",
    "emp_ma_3m": "Market pressure, 3m avg", "emp_ma_6m": "Market pressure, 6m avg",
    "emp_ma_12m": "Market pressure, 12m avg", "fx_vol_12m": "FX volatility, 12m",
    "res_vol_12m": "Reserve volatility, 12m",
    "res_vs_24m_max": "Reserves vs 24m peak", "res_vs_36m_mean": "Reserves vs 36m average",
    "fx_share": "FX share of reserves", "fx_share_chg_6m": "FX share change, 6m",
    "fx_vs_36m_trend": "FX vs 36m trend", "anchor_eur": "EUR-anchored",
    "defacto_peg": "De facto peg", "fx_vol_24m": "FX volatility, 24m",
    "fx_vol_rel_region": "FX flexibility vs region",
    "months_since_regime_flip": "Months in current regime",
    "log_res_usd": "Reserve base (log USD)",
}

assert sorted(sum(GROUPS.values(), [])) == sorted(FEATURES), "GROUPS must cover FEATURES"


def contributions(model, X):
    imp = model.named_steps["impute"]
    sc = model.named_steps["scale"]
    coef = model.named_steps["clf"].coef_[0]
    Z = sc.transform(imp.transform(X))
    return pd.DataFrame(Z * coef, index=X.index, columns=X.columns)


def main():
    p = pd.read_parquet("data/processed/features.parquet")
    panel = pd.read_parquet("data/processed/panel_monthly.parquet")

    train = p[p["modelable"]].reset_index(drop=True)
    model = pipe().fit(train[FEATURES], train["y"].astype(int))
    intercept = float(model.named_steps["clf"].intercept_[0])

    scor = p[p["complete"]].copy().reset_index(drop=True)
    scor["prob"] = model.predict_proba(scor[FEATURES])[:, 1]
    contrib = contributions(model, scor[FEATURES])

    scor["pctile"] = scor.groupby("COUNTRY")["prob"].rank(pct=True).mul(100).round(1)
    cutoff = pd.Period(RECENT_START, freq="M")
    rec = scor[scor["date"] >= cutoff].copy()
    rec["pctile_recent"] = rec.groupby("COUNTRY")["prob"].rank(pct=True).mul(100).round(1)
    scor = scor.merge(rec[["COUNTRY", "date", "pctile_recent"]],
                      on=["COUNTRY", "date"], how="left")

    freshness = (panel.dropna(subset=["fx_eop"]).groupby("COUNTRY")["date"].max()
                 .rename("fx_last").to_frame()
                 .join(panel.dropna(subset=["res_exgold"])
                       .groupby("COUNTRY")["date"].max().rename("res_last")))

    now = pd.Period(datetime.now(timezone.utc).strftime("%Y-%m"), freq="M")
    OUT.mkdir(parents=True, exist_ok=True)

    countries, history, tail_issues = [], [], {}
    for iso in sorted(set(scor["COUNTRY"])):
        g = scor[scor["COUNTRY"] == iso].sort_values("date")
        if g.empty:
            continue
        last = g.iloc[-1]
        c = contrib.loc[last.name]

        imputed = [f for f in FEATURES if pd.isna(last[f])]
        if imputed:
            tail_issues[iso] = imputed

        fx_last = freshness.loc[iso, "fx_last"] if iso in freshness.index else None
        res_last = freshness.loc[iso, "res_last"] if iso in freshness.index else None

        peg_tail = g["defacto_peg"].tail(24)
        peg = bool(peg_tail.mean() >= 0.5) if peg_tail.notna().any() else None

        pr_full = float(last["pctile"])
        pr_rec = None if pd.isna(last["pctile_recent"]) else float(last["pctile_recent"])

        groups = [{
            "group": name,
            "contribution": round(float(c[cols].sum()), 4),
            "direction": "raises" if c[cols].sum() > 0 else "lowers",
            "items": [{"label": READABLE.get(k, k),
                       "contribution": round(float(c[k]), 4),
                       "value": (None if pd.isna(last[k]) else round(float(last[k]), 4)),
                       "imputed": bool(pd.isna(last[k]))}
                      for k in sorted(cols, key=lambda x: -abs(c[x]))],
        } for name, cols in GROUPS.items()]
        groups.sort(key=lambda gr: -abs(gr["contribution"]))

        total = float(c.sum())
        countries.append({
            "iso3": iso,
            "name": COUNTRIES[iso]["name"],
            "currency": COUNTRIES[iso]["ccy"],
            "anchor": COUNTRIES[iso]["anchor"],
            "caveat": COUNTRIES[iso]["caveat"],
            "score_date": str(last["date"]),
            "probability": round(float(last["prob"]), 4),
            "percentile": pr_full,
            "percentile_recent": pr_rec,
            "percentile_gap": (None if pr_rec is None else round(pr_rec - pr_full, 1)),
            "percentile_12m_ago": (float(g.iloc[-13]["pctile"]) if len(g) > 13 else None),
            "data_age_months": int((now - last["date"]).n),
            "fx_last": str(fx_last) if fx_last is not None else None,
            "fx_lag_months": int((now - fx_last).n) if fx_last is not None else None,
            "res_last": str(res_last) if res_last is not None else None,
            "res_lag_months": int((now - res_last).n) if res_last is not None else None,
            "defacto_peg": peg,
            "imputed_features": imputed,
            "score_quality": "imputed" if imputed else "complete",
            "log_odds": round(total + intercept, 4),
            "contribution_total": round(total, 4),
            "groups": groups,
        })

        for _, r in g.iterrows():
            history.append({
                "iso3": iso, "date": str(r["date"]),
                "probability": round(float(r["prob"]), 4),
                "percentile": float(r["pctile"]),
                "crisis": bool(r.get("crisis", False)),
            })

    for iso in EXCLUDE_FROM_MODEL:
        countries.append({
            "iso3": iso, "name": COUNTRIES[iso]["name"],
            "currency": COUNTRIES[iso]["ccy"], "anchor": COUNTRIES[iso]["anchor"],
            "caveat": COUNTRIES[iso]["caveat"], "score_date": None,
            "probability": None, "percentile": None, "percentile_recent": None,
            "percentile_gap": None, "percentile_12m_ago": None,
            "data_age_months": None, "fx_last": None, "fx_lag_months": None,
            "res_last": None, "res_lag_months": None, "defacto_peg": None,
            "imputed_features": [], "score_quality": "unavailable",
            "log_odds": None, "contribution_total": None, "groups": [],
            "excluded_reason": "Insufficient IMF data to score",
        })

    h = pd.DataFrame(history)
    h["year"] = pd.PeriodIndex(h["date"], freq="M").year
    regional = [{"year": int(y), "share_above_p90": round(float(v), 3)}
                for y, v in h.assign(hot=h["percentile"] > 90)
                              .groupby("year")["hot"].mean().items()]

    coef = pd.Series(model.named_steps["clf"].coef_[0], index=FEATURES)
    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "as_of_month": str(now),
        "model": "Logistic regression (L2, C=0.1, balanced class weights)",
        "features": len(FEATURES),
        "intercept": round(intercept, 4),
        "horizon_months": 12,
        "publication_lag_months": 3,
        "recent_window_start": RECENT_START,
        "training_rows": int(len(train)),
        "training_positives": int(train["y"].sum()),
        "crisis_episodes": int(p["crisis"].sum()) if "crisis" in p else None,
        "countries_scored": len([c for c in countries if c["probability"] is not None]),
        "validation": CV,
        "regional_stress_by_year": regional,
        "coefficients": [{"feature": k, "label": READABLE.get(k, k),
                          "coefficient": round(float(v), 4)}
                         for k, v in coef.sort_values(key=abs, ascending=False).items()],
        "caveats": [
            "Scores estimate the probability of a currency crisis beginning within "
            "12 months, given data available three months before the score date.",
            "No crisis onset has been detected in the panel since July 2022, so the "
            "most recent validated test window ends March 2023. Readings after that "
            "date are extrapolation.",
            "Probabilities are not calibrated to observed frequencies. Percentile "
            "rank against the country's own history is the more reliable signal.",
            f"Fitted probabilities drift mildly upward over time, so the "
            f"{RECENT_START}-onward percentile is the fairer current comparison.",
            "Regularisation spreads signal across correlated features. Contributions "
            "are shown grouped because no single driver carries a score.",
            "Data freshness varies by country. Check the lag fields before acting on "
            "any single reading.",
            "Research tool, not investment advice.",
        ],
    }

    (OUT / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (OUT / "countries.json").write_text(
        json.dumps(sorted(countries, key=lambda c: (c["percentile_recent"] is None,
                                                    -(c["percentile_recent"] or 0))),
                   indent=2), encoding="utf-8")
    (OUT / "history.json").write_text(json.dumps(history), encoding="utf-8")

    print(f"Trained on {len(train):,} rows, {int(train['y'].sum())} positives")
    print(f"Scored {meta['countries_scored']} countries, {len(history):,} history points")
    print(f"Wrote {OUT}/meta.json, countries.json, history.json\n")

    tbl = pd.DataFrame([{
        "country": c["name"][:22], "date": c["score_date"], "prob": c["probability"],
        "pct_full": c["percentile"], "pct_10y": c["percentile_recent"],
        "gap": c["percentile_gap"], "age": c["data_age_months"], "peg": c["defacto_peg"],
    } for c in countries if c["probability"] is not None])
    print("CURRENT READINGS (ranked by recent-window percentile)")
    print(tbl.sort_values("pct_10y", ascending=False).to_string(index=False))

    print("\nTAIL DIAGNOSTIC — features imputed at last scored row:")
    print("  none" if not tail_issues
          else "\n".join(f"  {k}: {', '.join(v)}" for k, v in sorted(tail_issues.items())))

    print("\nGROUPED ATTRIBUTION CHECK — do groups sum to the total?")
    for c in sorted((c for c in countries if c["groups"]),
                    key=lambda x: -(x["percentile_recent"] or 0))[:4]:
        gs = sum(g["contribution"] for g in c["groups"])
        print(f"\n  {c['name']}  log-odds {c['log_odds']:+.3f} "
              f"(contrib {c['contribution_total']:+.3f}, groups {gs:+.3f})")
        for g in c["groups"]:
            bar = "#" * min(30, int(abs(g["contribution"]) * 20))
            print(f"    {g['direction']:<6} {g['group']:<22} {g['contribution']:+.3f}  {bar}")


if __name__ == "__main__":
    main()