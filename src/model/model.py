"""LightGBM on the blocked folds, compared against the logistic baseline."""
import pathlib, sys, warnings
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from src.model.splits import make_folds, usable_folds
from src.model.baseline import COUNTRY_FEATURES, CONTAGION, REGIME

warnings.filterwarnings("ignore")

RUS_ONLY = ["rus_empz", "rus_empz_6m"]
BLOCKS = {
    "country": COUNTRY_FEATURES,
    "country+regime": COUNTRY_FEATURES + REGIME,
    "country+regime+rus": COUNTRY_FEATURES + REGIME + RUS_ONLY,
    "full": COUNTRY_FEATURES + REGIME + CONTAGION,
}

PARAMS = dict(objective="binary", n_estimators=300, learning_rate=0.03,
              num_leaves=7, max_depth=3, min_child_samples=40,
              subsample=0.7, subsample_freq=1, colsample_bytree=0.7,
              reg_alpha=1.0, reg_lambda=5.0, class_weight="balanced",
              verbosity=-1, random_state=0)


def run(m, folds, features, label):
    rows, imps, preds = [], [], []
    for tr, te, md in folds:
        Xtr, ytr = m.loc[tr, features], m.loc[tr, "y"].astype(int)
        Xte, yte = m.loc[te, features], m.loc[te, "y"].astype(int)
        model = lgb.LGBMClassifier(**PARAMS).fit(Xtr, ytr)
        pr = model.predict_proba(Xte)[:, 1]
        rows.append({"test": f"{md['test_start']}..{md['test_end']}",
                     "auc": round(roc_auc_score(yte, pr), 3),
                     "ap": round(average_precision_score(yte, pr), 3)})
        imps.append(pd.Series(model.feature_importances_, index=features))
        preds.append(m.loc[te, ["COUNTRY", "date", "y"]].assign(
            pred=pr, fold=f"{md['test_start']}..{md['test_end']}", model=label))
    res = pd.DataFrame(rows)
    print(f"\n=== lgbm {label} ({len(features)} features) ===")
    print(res.to_string(index=False))
    print(f"Mean AUC {res['auc'].mean():.3f} (sd {res['auc'].std():.3f})"
          f" | mean AP {res['ap'].mean():.3f}")
    return res, pd.concat(imps, axis=1), pd.concat(preds, ignore_index=True)


if __name__ == "__main__":
    m = pd.read_parquet("data/processed/features.parquet")
    m = m[m["modelable"]].reset_index(drop=True)
    folds = usable_folds(make_folds(m))

    out, allpreds = {}, []
    for label, feats in BLOCKS.items():
        res, imp, pr = run(m, folds, feats, label)
        out[label] = res
        allpreds.append(pr)
        if label == "country+regime+rus":
            best_imp = imp

    print("\n=== LGBM BLOCK SUMMARY ===")
    summ = pd.DataFrame({k: {"mean_auc": round(v["auc"].mean(), 3),
                             "sd": round(v["auc"].std(), 3),
                             "mean_ap": round(v["ap"].mean(), 3)}
                         for k, v in out.items()}).T
    print(summ.to_string())

    print("\n=== VS LOGISTIC (mean AUC) ===")
    print("  logit country          0.637 (sd 0.121)")
    print("  logit country+regime   0.653 (sd 0.036)")
    print("  logit full             0.647 (sd 0.045)")
    for k, v in out.items():
        print(f"  lgbm  {k:<21} {v['auc'].mean():.3f} (sd {v['auc'].std():.3f})")

    print("\nFEATURE IMPORTANCE — lgbm country+regime+rus (mean splits)")
    print(best_imp.mean(axis=1).round(1).sort_values(ascending=False).head(15).to_string())

    pd.concat(allpreds, ignore_index=True).to_parquet(
        "data/processed/preds_lgbm.parquet", index=False)