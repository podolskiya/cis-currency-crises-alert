"""Logistic regression baseline on the blocked folds."""
import pathlib, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from src.model.splits import make_folds, usable_folds

FEATURES = ["fx_chg_3m", "res_chg_3m", "emp_ma_3m",
            "fx_chg_6m", "res_chg_6m", "emp_ma_6m",
            "fx_chg_12m", "res_chg_12m", "emp_ma_12m",
            "fx_vol_12m", "res_vol_12m",
            "res_vs_24m_max", "res_vs_36m_mean",
            "fx_share", "fx_share_chg_6m", "fx_vs_36m_trend"]


def pipe():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=0.1, class_weight="balanced",
                                   max_iter=2000, random_state=0)),
    ])


if __name__ == "__main__":
    m = pd.read_parquet("data/processed/features.parquet")
    m = m[m["modelable"]].reset_index(drop=True)
    folds = usable_folds(make_folds(m))

    rows, coefs, preds = [], [], []
    for tr, te, md in folds:
        Xtr, ytr = m.loc[tr, FEATURES], m.loc[tr, "y"].astype(int)
        Xte, yte = m.loc[te, FEATURES], m.loc[te, "y"].astype(int)

        model = pipe().fit(Xtr, ytr)
        pr = model.predict_proba(Xte)[:, 1]

        rows.append({
            "test": f"{md['test_start']}..{md['test_end']}",
            "n_test": len(te), "pos": int(yte.sum()),
            "base_rate": round(yte.mean(), 3),
            "auc": round(roc_auc_score(yte, pr), 3),
            "ap": round(average_precision_score(yte, pr), 3),
        })
        coefs.append(pd.Series(model.named_steps["clf"].coef_[0], index=FEATURES))
        preds.append(m.loc[te, ["COUNTRY", "date", "y"]].assign(
            pred=pr, fold=f"{md['test_start']}..{md['test_end']}"))

    res = pd.DataFrame(rows)
    pd.concat(preds, ignore_index=True).to_parquet(
        "data/processed/preds_baseline.parquet", index=False)

    print("FOLD RESULTS")
    print(res.to_string(index=False))
    print(f"\nMean AUC: {res['auc'].mean():.3f}   sd: {res['auc'].std():.3f}")
    print(f"Mean AP:  {res['ap'].mean():.3f}   mean base rate: {res['base_rate'].mean():.3f}")

    ok = res[res["auc"] >= 0.6]
    print(f"\nFolds above 0.6 AUC: {len(ok)} of {len(res)}")
    if len(ok):
        print(f"  mean AUC on those: {ok['auc'].mean():.3f}"
              f"   mean AP: {ok['ap'].mean():.3f}"
              f"   vs base rate: {ok['base_rate'].mean():.3f}")

    print("\nCOEFFICIENTS (standardised, mean across folds)")
    cf = pd.concat(coefs, axis=1)
    summary = pd.DataFrame({"mean": cf.mean(axis=1).round(3),
                            "sd": cf.std(axis=1).round(3),
                            "sign_stable": (np.sign(cf).nunique(axis=1) == 1)})
    print(summary.sort_values("mean", key=abs, ascending=False).to_string())

    print("\nSIGN-STABLE FEATURES (candidates for the reduced set):")
    print("  " + ", ".join(summary[summary["sign_stable"]].index.tolist()))