"""Genus-level relative abundance, per-study percentile harmonization (zeros kept at 0) + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def _study_labels(items, ctx):
    meta = ctx.get("meta")
    if meta is None or "study" not in meta:
        return np.zeros(len(items), dtype=object)
    try:
        if meta.index.equals(items.index) or set(items.index) <= set(meta.index):
            return meta.loc[items.index, "study"].astype(str).values
    except Exception:
        pass
    if len(meta) == len(items):
        return meta["study"].astype(str).values
    return np.zeros(len(items), dtype=object)


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    G = rel.T.groupby(np.array(genus)).sum().T              # samples x genera
    studies = _study_labels(items, ctx)
    out = np.zeros(G.shape, dtype=float)
    vals = G.values
    for s in pd.unique(studies):
        m = studies == s
        sub = pd.DataFrame(vals[m])
        r = sub.rank(axis=0, method="average", pct=True).values   # within-study percentile
        r[vals[m] <= 0] = 0.0                                      # absent stays absent
        out[m] = r
    return pd.DataFrame(out, index=G.index, columns=G.columns)


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]