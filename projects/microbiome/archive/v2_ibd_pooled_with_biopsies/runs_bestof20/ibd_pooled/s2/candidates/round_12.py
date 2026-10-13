"""Genus-level log relative abundance, per-study standardized (label-free harmonization) + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

PSEUDO = 1e-5


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    return rel.T.groupby(np.array(genus)).sum().T          # samples x genera


def _study(meta, n):
    if meta is None or "study" not in getattr(meta, "columns", []):
        return np.zeros(n, dtype=object)
    return np.asarray(meta["study"].values, dtype=object)


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr = np.log10(F_train[cols].values + PSEUDO)
    Xte = np.log10(F_test.reindex(columns=cols, fill_value=0).values + PSEUDO)
    s_tr = _study(ctx.get("meta_train"), len(Xtr))
    s_te = _study(ctx.get("meta_test"), len(Xte))

    # per-study standardization using all (label-free) samples of that study
    X_all = np.vstack([Xtr, Xte])
    s_all = np.concatenate([s_tr, s_te])
    Z = np.zeros_like(X_all)
    for s in pd.unique(s_all):
        m = s_all == s
        mu = X_all[m].mean(axis=0)
        sd = X_all[m].std(axis=0)
        sd[sd < 1e-8] = 1.0
        Z[m] = (X_all[m] - mu) / sd
    Ztr, Zte = Z[: len(Xtr)], Z[len(Xtr):]

    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Ztr, y_train)
    return rf.predict_proba(Zte)[:, 1]