"""LightGBM on the blocked folds, compared against the logistic baseline."""
import pathlib, sys, warnings
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from src.model.splits import make_folds, usable_folds
from src.model.baseline import FEATURES as FULL_FEATURES

warnings.filterwarnings("ignore", category=UserWarning)

REDUCED = ["fx_chg_3m", "res_chg_3m", "emp_ma_3m", "res_chg_6m", "emp_ma_6m",
           "res_chg_12m", "fx_vol_12m", "res_vs_24m_max", "fx_vs_36m_trend"]

PARAMS = dict(
    objective="binary",
    n_estimators=300,
    learning_rate=0.03,
    num_leaves=7,            
    max_depth=3,
    min_child_samples=40,    
    subsample=0.7,
    subsample_freq=1,
    colsample_bytree=0.7,
    reg_alpha=1.0,
    reg_lambda=5.0,
    class_weight="balanced",
    verbosity=-1,
    random_state=0,
)


def run(m, folds, features, label):
    rows, preds, imps = [], [], []
    for tr, te, md in folds:
        Xtr, ytr = m.loc[tr, features], m.loc[tr, "y"].astype(int)
        Xte, yte = m.loc[te, features], m.loc[te, "y"].astype(int)

        model = lgb.LGBMClassifier(**PARAMS).fit(Xtr, ytr)
        pr = model.predict_proba(Xte)[:, 1]

        rows.append({
            "test": f"{md['test_start']}..{md['test_end']}",
            "pos": int(yte.sum()), "base_rate": round(yte.mean(), 3),
            "auc": round(roc_auc_score(yte, pr), 3),
            "ap": round(average_precision_score(yte, pr), 3),
        })
        imps.append(pd.Series(model.feature_importances_, index=features))
        preds.append(m.loc[te, ["COUNTRY", "date", "y"]].assign(
            pred=pr, fold=f"{md['test_start']}..{md['test_end']}", model=label))

    res = pd.DataFrame(rows)
    ok = res[res["auc"] >= 0.6]
    print(f"\n=== {label} ({len(features)} features) ===")
    print(res.to_string(index=False))
    print(f"Mean AUC: {res['auc'].mean():.3f} (sd {res['auc'].std():.3f})"
          f"   | pre-COVID folds: {ok['auc'].mean() if len(ok) else float('nan'):.3f}")
    return res, pd.concat(preds, ignore_index=True), pd.concat(imps, axis=1)


if __name__ == "__main__":
    m = pd.read_parquet("data/processed/features.parquet")
    m = m[m["modelable"]].reset_index(drop=True)
    folds = usable_folds(make_folds(m))

    res_full, pred_full, imp_full = run(m, folds, FULL_FEATURES, "lgbm_full")
    res_red, pred_red, imp_red = run(m, folds, REDUCED, "lgbm_reduced")

    print("\n=== COMPARISON vs LOGISTIC BASELINE ===")
    base = pd.DataFrame({
        "test": ["2011-03..2014-03", "2014-03..2017-03",
                 "2017-03..2020-03", "2020-03..2023-03"],
        "logistic": [0.768, 0.820, 0.781, 0.448]})
    comp = (base.merge(res_full[["test", "auc"]].rename(columns={"auc": "lgbm_full"}), on="test")
                .merge(res_red[["test", "auc"]].rename(columns={"auc": "lgbm_reduced"}), on="test"))
    print(comp.to_string(index=False))
    print("\nMeans:", {c: round(comp[c].mean(), 3)
                       for c in ("logistic", "lgbm_full", "lgbm_reduced")})
    print("Pre-COVID means (first 3):",
          {c: round(comp[c].head(3).mean(), 3)
           for c in ("logistic", "lgbm_full", "lgbm_reduced")})

    print("\nFEATURE IMPORTANCE (reduced model, mean gain-split count across folds)")
    print(imp_red.mean(axis=1).round(1).sort_values(ascending=False).to_string())

    pd.concat([pred_full, pred_red], ignore_index=True).to_parquet(
        "data/processed/preds_lgbm.parquet", index=False)