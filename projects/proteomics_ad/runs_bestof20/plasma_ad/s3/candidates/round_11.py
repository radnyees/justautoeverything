"""log2 + per-sample median centring + per-batch per-protein median centring + half-minimum imputation + L2 LR."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def featurize(items, ctx):
    L = np.log2(items.astype(float))
    return L.sub(L.median(axis=1), axis=0)                 # per-sample median centring


def _batches(meta, n):
    if meta is None:
        return np.zeros(n, dtype=object)
    try:
        if isinstance(meta, pd.DataFrame) and "batch" in meta.columns:
            return meta["batch"].astype(str).values
        return pd.DataFrame(meta)["batch"].astype(str).values
    except Exception:
        return np.zeros(n, dtype=object)


def fit_predict(F_train, y_train, F_test, ctx):
    keep = F_train.columns[F_train.notna().mean() >= 0.7]  # quantified in >= 70% of train samples
    Atr = F_train[keep].astype(float).copy()
    Ate = F_test[keep].astype(float).copy()

    # Label-free batch harmonization: remove per-batch protein median (observed values only).
    btr = _batches(ctx.get("meta_train"), len(Atr))
    bte = _batches(ctx.get("meta_test"), len(Ate))
    allA = pd.concat([Atr, Ate], axis=0, ignore_index=True)
    allb = np.concatenate([btr, bte])
    gmed = allA.median()
    for b in np.unique(allb):
        m = allb == b
        if m.sum() < 5:
            continue
        shift = (allA[m].median() - gmed).fillna(0.0)
        allA.loc[m] = allA.loc[m] - shift
    Atr = allA.iloc[:len(Atr)].reset_index(drop=True)
    Ate = allA.iloc[len(Atr):].reset_index(drop=True)

    fill = Atr.min() - 1.0                                  # half-minimum on the log2 scale
    Xtr = Atr.fillna(fill).values
    Xte = Ate.fillna(fill).values
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.05, max_iter=5000)
    clf.fit(sc.transform(Xtr), y_train)
    return clf.predict_proba(sc.transform(Xte))[:, 1]