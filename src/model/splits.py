"""Expanding-window CV with purge gap. No model code here — splits only."""
import pathlib, sys
import numpy as np, pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

HORIZON = 12           # must match target.py
TEST_MONTHS = 36       # length of each test window
MIN_TRAIN_MONTHS = 60  # first fold needs this much history
MIN_TRAIN_POS = 30     # folds below this are listed but not scored


def make_folds(df, horizon=HORIZON, test_months=TEST_MONTHS, min_train=MIN_TRAIN_MONTHS):
    """Return a list of (train_idx, test_idx, meta) over an expanding time window.

    Training uses every row dated before train_end. A purge gap of `horizon`
    months is then skipped, because a label at month t encodes crisis onsets up
    to t+horizon and would otherwise leak into the test period.
    """
    dates = np.sort(df["date"].unique())
    start = dates[0]
    folds = []
    cursor = start + min_train
    while True:
        train_end = cursor
        test_start = train_end + horizon
        if test_start > dates[-1]:
            break
        test_end = min(test_start + test_months, dates[-1] + 1)

        tr = df.index[df["date"] < train_end]
        te = df.index[(df["date"] >= test_start) & (df["date"] < test_end)]

        if len(tr) > 0 and len(te) > 0:
            pos_test = int(df.loc[te, "y"].sum())
            folds.append((tr, te, {
                "train_end": str(train_end),
                "test_start": str(test_start),
                "test_end": str(test_end),
                "n_train": len(tr),
                "n_test": len(te),
                "pos_train": int(df.loc[tr, "y"].sum()),
                "pos_test": pos_test,
                "scorable": pos_test > 0,
            }))
        cursor = cursor + test_months
    return folds


def usable_folds(folds, min_train_pos=MIN_TRAIN_POS):
    """Folds with enough training positives to fit, and test positives to score."""
    return [f for f in folds if f[2]["pos_train"] >= min_train_pos and f[2]["scorable"]]


if __name__ == "__main__":
    p = pd.read_parquet("data/processed/features.parquet")
    m = p[p["modelable"]].reset_index(drop=True)

    folds = make_folds(m)
    usable = usable_folds(folds)

    print(f"Modelable rows: {len(m):,}  positives: {int(m['y'].sum())}")
    print(f"Date range: {m['date'].min()} to {m['date'].max()}")

    print(f"\nALL FOLDS: {len(folds)}\n")
    print(pd.DataFrame([f[2] for f in folds]).to_string(index=False))

    print(f"\nUsable: {len(usable)} of {len(folds)}"
          f"  (need >= {MIN_TRAIN_POS} train positives and >= 1 test positive)")
    for _, _, md in folds:
        if md["pos_train"] < MIN_TRAIN_POS:
            why = f"only {md['pos_train']} train positives"
        elif not md["scorable"]:
            why = "no crisis onset in the test window"
        else:
            continue
        print(f"  excluded {md['test_start']}..{md['test_end']}: {why}")

    print("\nTEST-FOLD COUNTRY COVERAGE (positives):")
    for tr, te, md in folds:
        cc = m.loc[te].groupby("COUNTRY")["y"].sum()
        hit = ", ".join(f"{c}:{int(v)}" for c, v in cc.items() if v > 0) or "none"
        print(f"  {md['test_start']}..{md['test_end']}  {hit}")