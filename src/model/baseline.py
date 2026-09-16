"""Logistic regression on the blocked folds, nested feature blocks."""
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

REGIME = ["anchor_eur", "defacto_peg", "fx_vol_24m", "fx_vol_rel_region",
          "months_since_regime_flip", "log_res_usd"]

CONTAGION = ["region_emp_loo", "region_stress_share", "rus_empz",
             "bloc_emp_loo", "region_stress_6m", "region_emp_6m", "rus_empz_6m"]

REER = ["rer_vs_36m", "rer_vs_60m", "rer_chg_12m",
        "infl_12m", "infl_diff_anchor", "infl_accel"]

FEATURES = COUNTRY_FEATURES + REGIME + REER + CONTAGION


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
        rows.append({"test": f"{md['test_start']}..{md['test_end']}",
                     "pos": int(yte.sum()), "base_rate": round(yte.mean(), 3),
                     "auc": round(roc_auc_score(yte, pr), 3),
                     "ap": round(average_precision_score(yte, pr), 3)})
        coefs.append(pd.Series(model.named_steps["clf"].coef_[0], index=features))
        preds.append(m.loc[te, ["COUNTRY", "date", "y"]].assign(
            pred=pr, fold=f"{md['test_start']}..{md['test_end']}", model=label))

    res = pd.DataFrame(rows)
    print(f"\n=== {label} ({len(features)} features) ===")
    print(res.to_string(index=False))
    print(f"Mean AUC {res['auc'].mean():.3f} (sd {res['auc'].std():.3f})"
          f" | min {res['auc'].min():.3f}"
          f" | mean AP {res['ap'].mean():.3f} vs base {res['base_rate'].mean():.3f}")
    if save:
        pd.concat(preds, ignore_index=True).to_parquet(save, index=False)
    return res, pd.concat(coefs, axis=1)


def family(n):
    if n in CONTAGION: return "contagion"
    if n in REGIME: return "regime"
    if n in REER: return "reer"
    return "country"


def select_in_fold(m, tr, features, min_abs=0.10):
    """Choose features using ONLY training data, via bootstrap sign stability."""
    Xtr, ytr = m.loc[tr, features], m.loc[tr, "y"].astype(int)
    rng = np.random.default_rng(0)
    coefs = []
    for _ in range(25):
        idx = rng.choice(len(Xtr), size=len(Xtr), replace=True)
        Xb, yb = Xtr.iloc[idx], ytr.iloc[idx]
        if yb.sum() < 10:
            continue
        mdl = pipe().fit(Xb, yb)
        coefs.append(pd.Series(mdl.named_steps["clf"].coef_[0], index=features))
    if not coefs:
        return features
    cf = pd.concat(coefs, axis=1)
    stable = np.sign(cf).nunique(axis=1) == 1
    big = cf.mean(axis=1).abs() > min_abs
    sel = cf.index[stable & big].tolist()
    return sel if len(sel) >= 5 else features


def run_selected(m, folds, features, label, save=None):
    """Per-fold selection then fit — no test-set information used in selection."""
    rows, preds, chosen = [], [], []
    for tr, te, md in folds:
        sel = select_in_fold(m, tr, features)
        chosen.append(set(sel))
        Xtr, ytr = m.loc[tr, sel], m.loc[tr, "y"].astype(int)
        Xte, yte = m.loc[te, sel], m.loc[te, "y"].astype(int)
        model = pipe().fit(Xtr, ytr)
        pr = model.predict_proba(Xte)[:, 1]
        rows.append({"test": f"{md['test_start']}..{md['test_end']}",
                     "n_feat": len(sel),
                     "auc": round(roc_auc_score(yte, pr), 3),
                     "ap": round(average_precision_score(yte, pr), 3)})
        preds.append(m.loc[te, ["COUNTRY", "date", "y"]].assign(
            pred=pr, fold=f"{md['test_start']}..{md['test_end']}", model=label))

    res = pd.DataFrame(rows)
    print(f"\n=== {label} (per-fold selection from {len(features)}) ===")
    print(res.to_string(index=False))
    print(f"Mean AUC {res['auc'].mean():.3f} (sd {res['auc'].std():.3f})"
          f" | min {res['auc'].min():.3f} | mean AP {res['ap'].mean():.3f}")
    always = set.intersection(*chosen) if chosen else set()
    ever = set.union(*chosen) if chosen else set()
    print(f"\nChosen in EVERY fold ({len(always)}): {', '.join(sorted(always))}")
    print(f"Chosen in SOME fold ({len(ever)}): {', '.join(sorted(ever - always))}")
    if save:
        pd.concat(preds, ignore_index=True).to_parquet(save, index=False)
    return res


if __name__ == "__main__":
    m = pd.read_parquet("data/processed/features.parquet")
    m = m[m["modelable"]].reset_index(drop=True)
    folds = usable_folds(make_folds(m))

    blocks = {
        "country": COUNTRY_FEATURES,
        "country+regime": COUNTRY_FEATURES + REGIME,
        "country+regime+reer": COUNTRY_FEATURES + REGIME + REER,
        "all": FEATURES,
    }
    out, coefs = {}, {}
    for label, feats in blocks.items():
        save = "data/processed/preds_baseline.parquet" if label == "all" else None
        out[label], coefs[label] = run(m, folds, feats, f"logit_{label}", save=save)

    cf = coefs["all"]
    s = pd.DataFrame({"stable": (np.sign(cf).nunique(axis=1) == 1),
                      "mean": cf.mean(axis=1)})
    keep = s[(s["stable"]) & (s["mean"].abs() > 0.10)].index.tolist()
    print(f"\nSTABLE AND NON-TRIVIAL, full-sample view ({len(keep)}): {', '.join(keep)}")
    print("  (optimistic — selected with all folds visible; see per-fold version below)")

    res_fixed, cf_sel = run(m, folds, keep, "logit_selected_fixed")
    res_honest = run_selected(m, folds, FEATURES, "logit_selected_perfold",
                              save="data/processed/preds_selected.parquet")

    print("\n" + "=" * 74)
    print("FINAL COMPARISON (AUC by fold)")
    print("=" * 74)
    final = out["country"][["test"]].copy()
    for label, r in out.items():
        final[label] = r["auc"].values
    final["selected_fixed"] = res_fixed["auc"].values
    final["selected_perfold"] = res_honest["auc"].values
    print(final.to_string(index=False))

    means = {k: round(v["auc"].mean(), 3) for k, v in out.items()}
    sds = {k: round(v["auc"].std(), 3) for k, v in out.items()}
    means["selected_fixed"] = round(res_fixed["auc"].mean(), 3)
    sds["selected_fixed"] = round(res_fixed["auc"].std(), 3)
    means["selected_perfold"] = round(res_honest["auc"].mean(), 3)
    sds["selected_perfold"] = round(res_honest["auc"].std(), 3)
    print("\nMean:", means)
    print("Sd:  ", sds)

    winner = max(means, key=means.get)
    print(f"\nHighest mean AUC: {winner} ({means[winner]:.3f}, sd {sds[winner]:.3f})")
    print("Note: selected_fixed is optimistic. selected_perfold is the honest number.")

    s2 = pd.DataFrame({"mean": cf_sel.mean(axis=1).round(3),
                       "sd": cf_sel.std(axis=1).round(3),
                       "stable": (np.sign(cf_sel).nunique(axis=1) == 1)})
    s2["family"] = [family(i) for i in s2.index]
    print("\nCOEFFICIENTS — selected (fixed set)")
    print(s2.sort_values("mean", key=abs, ascending=False).to_string())