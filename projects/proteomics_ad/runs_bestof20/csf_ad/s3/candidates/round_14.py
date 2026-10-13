"""log2 + per-sample median centring + half-minimum imputation + per-site standardization + L2 logistic regression."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def featurize(items, ctx):
    L = np.log2(items.astype(float))
    return L.sub(L.median(axis=1), axis=0)                 # per-sample median centring


def _sites(meta, n):
    if meta is None:
        return np.zeros(n, dtype=object)
    try:
        if isinstance(meta, pd.DataFrame) and "site" in meta.columns:
            return meta["site"].astype(str).values
        if isinstance(meta, dict) and "site" in meta:
            return np.asarray(meta["site"]).astype(str)
    except Exception:
        pass
    return np.zeros(n, dtype=object)


def _site_standardize(Xtr, Xte, s_tr, s_te, min_n=6):
    g_mu = Xtr.mean(0)
    g_sd = Xtr.std(0) + 1e-6
    Ztr = np.empty_like(Xtr)
    Zte = np.empty_like(Xte)
    stats = {}
    for s in np.unique(s_tr):
        m = s_tr == s
        if m.sum() >= min_n:
            mu = Xtr[m].mean(0)
            sd = Xtr[m].std(0)
            sd = np.where(sd < 1e-3, g_sd, sd)
        else:
            mu, sd = g_mu, g_sd
        stats[s] = (mu, sd)
        Ztr[m] = (Xtr[m] - mu) / sd
    for i, s in enumerate(s_te):
        mu, sd = stats.get(s, (g_mu, g_sd))
        Zte[i] = (Xte[i] - mu) / sd
    return Ztr, Zte


def fit_predict(F_train, y_train, F_test, ctx):
    keep = F_train.columns[F_train.notna().mean() >= 0.7]  # quantified in >= 70% of train samples
    fill = F_train[keep].min() - 1.0                        # half-minimum on the log2 scale
    Xtr = F_train[keep].fillna(fill).values
    Xte = F_test[keep].fillna(fill).values
    s_tr = _sites(ctx.get("meta_train"), len(Xtr))
    s_te = _sites(ctx.get("meta_test"), len(Xte))
    Ztr, Zte = _site_standardize(Xtr, Xte, s_tr, s_te)
    clf = LogisticRegression(C=0.05, max_iter=5000)
    clf.fit(Ztr, y_train)
    return clf.predict_proba(Zte)[:, 1]