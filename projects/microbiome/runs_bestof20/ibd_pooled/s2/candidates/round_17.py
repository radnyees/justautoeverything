"""Genus-level relative abundance, per-study percentile normalization (batch harmonization) + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def _study_labels(items, ctx):
    meta = ctx.get("meta", None)
    if meta is None or "study" not in getattr(meta, "columns", []):
        return np.zeros(len(items), dtype=object)
    if meta.index.equals(items.index) or set(items.index).issubset(set(meta.index)):
        try:
            return meta["study"].reindex(items.index).fillna("NA").astype(str).values
        except Exception:
            pass
    if len(meta) == len(items):
        return meta["study"].fillna("NA").astype(str).values
    return np.zeros(len(items), dtype=object)


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    G = rel.T.groupby(np.array(genus)).sum().T             # samples x genera
    studies = _study_labels(items, ctx)
    out = pd.DataFrame(0.0, index=G.index, columns=G.columns)
    vals = G.values
    res = np.zeros_like(vals, dtype=float)
    for s in pd.unique(studies):
        m = studies == s
        sub = pd.DataFrame(vals[m])
        n = m.sum()
        # within-study percentile (ties averaged); zeros share the same low percentile
        res[m] = (sub.rank(axis=0, method="average").values - 0.5) / max(n, 1)
        # genera never observed in this study carry no information -> neutral 0
        absent = (vals[m] > 0).sum(axis=0) == 0
        res[np.ix_(m, absent)] = 0.0
    out[:] = res
    return out


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]