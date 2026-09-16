"""Logistic regression baseline on the blocked folds.

Runs three nested feature sets on identical folds so each block's marginal
contribution is isolated: country fundamentals, plus regime conditioning,
plus cross-sectional contagion.
"""
import pathlib, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from src.model.splits import make_folds, usable_folds

COUNTRY_FEATURES = ["fx_chg_3m", "res_chg_3m", "emp_ma_3m",
                    "fx_chg_6m", "res_chg_6m", "emp_ma_6m",
                    "fx_chg_12m", "res_chg_12m", "emp_ma_12m",
                    "fx_vol_12m", "res_vol_12m",
                    "res_vs_24m_max", "res_vs_36m_mean",
                    "fx_share", "fx_share_chg_6m", "fx_vs_36m_trend"]

CONTAGION = ["region_emp_loo", "region_stress_share", "rus_empz",
             "bloc_emp_loo", "region_stress_6m", "region_emp_6m", "rus_empz_6m"]

REGIME = ["anchor_eur", "defacto_peg", "fx_vol_24m", "fx_vol_rel_region",
          "months_since_regime_flip", "log_res_usd"]

FEATURES = COUNTRY_FEATURES + REGIME + CONTAGION


def pipe():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=0.1, class_weight="balanced",
                                   max_iter=2000, random_state=0)),
    ])


def run(m, folds, features, label, save=None):
    rows, coefs, preds = [], [], []
    for tr, te, md in folds:
        Xtr, ytr = m.loc[tr, features], m.loc[tr, "y"].astype(int)
        Xte, yte = m.loc[te, features], m.loc[te, "y"].astype(int)

        model = pipe().fit(Xtr, ytr)
        pr = model.predict_proba(Xte)[:, 1]

        rows.append({
            "test": f"{md['test_start']}..{md['test_end']}",
            "pos": int(yte.sum()), "base_rate": round(yte.mean(), 3),
            "auc": round(roc_auc_score(yte, pr), 3),
            "ap": round(average_precision_score(yte, pr), 3),
        })
        coefs.append(pd.Series(model.named_steps["clf"].coef_[0], index=features))
        preds.append(m.loc[te, ["COUNTRY", "date", "y"]].assign(
            pred=pr, fold=f"{md['test_start']}..{md['test_end']}", model=label))

    res = pd.DataFrame(rows)
    ok = res[res["auc"] >= 0.6]
    print(f"\n=== {label} ({len(features)} features) ===")
    print(res.to_string(index=False))
    print(f"Mean AUC {res['auc'].mean():.3f} (sd {res['auc'].std():.3f})"
          f" | folds>=0.6 {ok['auc'].mean() if len(ok) else float('nan'):.3f}"
          f" | mean AP {res['ap'].mean():.3f} vs base {res['base_rate'].mean():.3f}")
    if save:
        pd.concat(preds, ignore_index=True).to_parquet(save, index=False)
    return res, pd.concat(coefs, axis=1)


def family(name):
    if name in CONTAGION:
        return "contagion"
    if name in REGIME:
        return "regime"
    return "country"


if __name__ == "__main__":
    m = pd.read_parquet("data/processed/features.parquet")
    m = m[m["modelable"]].reset_index(drop=True)
    folds = usable_folds(make_folds(m))

    res_c, _ = run(m, folds, COUNTRY_FEATURES, "logit_country")
    res_r, cf_r = run(m, folds, COUNTRY_FEATURES + REGIME, "logit_country_regime")
    res_a, cf_a = run(m, folds, FEATURES, "logit_full",
                      save="data/processed/preds_baseline.parquet")

    print("\n=== MARGINAL CONTRIBUTION OF EACH BLOCK (AUC) ===")
    comp = (res_c[["test", "auc"]].rename(columns={"auc": "country"})
            .merge(res_r[["test", "auc"]].rename(columns={"auc": "plus_regime"}), on="test")
            .merge(res_a[["test", "auc"]].rename(columns={"auc": "plus_contagion"}), on="test"))
    comp["d_regime"] = (comp["plus_regime"] - comp["country"]).round(3)
    comp["d_contagion"] = (comp["plus_contagion"] - comp["plus_regime"]).round(3)
    print(comp.to_string(index=False))
    print("\nMeans:", {c: round(comp[c].mean(), 3)
                       for c in ("country", "plus_regime", "plus_contagion")})
    print(f"Mean delta from regime:    {comp['d_regime'].mean():+.3f}")
    print(f"Mean delta from contagion: {comp['d_contagion'].mean():+.3f}")

    best = max([("country", res_c), ("country+regime", res_r), ("full", res_a)],
               key=lambda t: t[1]["auc"].mean())
    print(f"\nBest block by mean AUC: {best[0]} ({best[1]['auc'].mean():.3f})")

    print("\nCOEFFICIENTS — country+regime model (standardised, mean across folds)")
    s = pd.DataFrame({"mean": cf_r.mean(axis=1).round(3),
                      "sd": cf_r.std(axis=1).round(3),
                      "sign_stable": (np.sign(cf_r).nunique(axis=1) == 1)})
    s["family"] = [family(i) for i in s.index]
    print(s.sort_values("mean", key=abs, ascending=False).to_string())

    print("\nCOEFFICIENTS — full model (standardised, mean across folds)")
    sa = pd.DataFrame({"mean": cf_a.mean(axis=1).round(3),
                       "sd": cf_a.std(axis=1).round(3),
                       "sign_stable": (np.sign(cf_a).nunique(axis=1) == 1)})
    sa["family"] = [family(i) for i in sa.index]
    print(sa.sort_values("mean", key=abs, ascending=False).to_string())

    print("\nSIGN-STABLE (full model):")
    print("  " + ", ".join(sa[sa["sign_stable"]].index.tolist()))