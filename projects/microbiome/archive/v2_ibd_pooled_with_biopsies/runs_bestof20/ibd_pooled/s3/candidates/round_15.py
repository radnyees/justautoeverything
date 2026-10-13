"""Genus-level CLR + per-study robust standardization (label-free harmonization) + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def featurize(items, ctx):
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in items.columns]
    counts = items.T.groupby(np.array(genus)).sum().T      # samples x genera (raw counts)
    lg = np.log(counts.values.astype(float) + 0.5)
    clr = lg - lg.mean(axis=1, keepdims=True)
    return pd.DataFrame(clr, index=counts.index, columns=counts.columns)


def _study(meta, n):
    if meta is None or "study" not in getattr(meta, "columns", []):
        return np.zeros(n, dtype=object)
    return np.asarray(meta["study"].values, dtype=object)


def _per_study_scale(X, studies):
    Z = np.empty_like(X)
    for s in pd.unique(studies):
        m = studies == s
        Xs = X[m]
        med = np.median(Xs, axis=0)
        q75, q25 = np.percentile(Xs, [75, 25], axis=0)
        scale = (q75 - q25) / 1.349
        sd = Xs.std(axis=0)
        scale = np.where(scale > 1e-8, scale, sd)
        scale = np.where(scale > 1e-8, scale, 1.0)
        Z[m] = (Xs - med) / scale
    return Z


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr = F_train[cols].values
    Xte = F_test.reindex(columns=cols).values
    # missing genera in test: fill with each test sample's minimum CLR value (≈ zero count)
    if np.isnan(Xte).any():
        rowmin = np.nanmin(np.where(np.isnan(Xte), np.inf, Xte), axis=1, keepdims=True)
        Xte = np.where(np.isnan(Xte), rowmin, Xte)

    s_tr = _study(ctx.get("meta_train"), len(Xtr))
    s_te = _study(ctx.get("meta_test"), len(Xte))
    X_all = np.vstack([Xtr, Xte])
    s_all = np.concatenate([s_tr, s_te])
    Z = _per_study_scale(X_all, s_all)                      # label-free per-study harmonization
    Ztr, Zte = Z[: len(Xtr)], Z[len(Xtr):]

    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Ztr, y_train)
    return rf.predict_proba(Zte)[:, 1]