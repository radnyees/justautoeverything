"""log2 + per-sample median centring + per-site protein median centring + half-min imputation + L2 LR."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _site_center(F, sites):
    """Subtract per-site, per-protein median (label-free harmonization)."""
    F = F.copy()
    sites = np.asarray(sites)
    for s in np.unique(sites):
        m = sites == s
        if m.sum() < 3:
            continue
        med = F.loc[m].median(axis=0, skipna=True).fillna(0.0)
        F.loc[m] = F.loc[m] - med
    return F


def featurize(items, ctx):
    L = np.log2(items.astype(float))
    return L.sub(L.median(axis=1), axis=0)                 # per-sample median centring


def _get_sites(meta, n):
    if meta is None:
        return np.zeros(n)
    try:
        if isinstance(meta, pd.DataFrame) and "site" in meta.columns:
            return meta["site"].astype(str).values
        return pd.DataFrame(meta)["site"].astype(str).values
    except Exception:
        return np.zeros(n)


def fit_predict(F_train, y_train, F_test, ctx):
    s_tr = _get_sites(ctx.get("meta_train"), len(F_train))
    s_te = _get_sites(ctx.get("meta_test"), len(F_test))
    keep = F_train.columns[F_train.notna().mean() >= 0.7]  # quantified in >= 70% of train samples
    A = pd.concat([F_train[keep], F_test[keep]], axis=0, ignore_index=True)
    A = _site_center(A, np.concatenate([s_tr, s_te]))
    Atr, Ate = A.iloc[:len(F_train)], A.iloc[len(F_train):]
    fill = Atr.min() - 1.0                                  # half-minimum on the log2 scale
    Xtr = Atr.fillna(fill).values
    Xte = Ate.fillna(fill).values
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.05, max_iter=5000)
    clf.fit(sc.transform(Xtr), y_train)
    return clf.predict_proba(sc.transform(Xte))[:, 1]