"""Genus-level relative abundance, per-study percentile (rank) normalization + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def _study_labels(items, ctx):
    meta = ctx.get("meta") if isinstance(ctx, dict) else None
    if meta is None or "study" not in getattr(meta, "columns", []):
        return np.zeros(len(items), dtype=object)
    try:
        st = meta["study"].reindex(items.index)
        if st.isna().any():
            raise KeyError
        return st.values
    except Exception:
        if len(meta) == len(items):
            return meta["study"].values
        return np.zeros(len(items), dtype=object)


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    G = rel.T.groupby(np.array(genus)).sum().T              # samples x genera
    studies = _study_labels(items, ctx)
    out = pd.DataFrame(0.0, index=G.index, columns=G.columns)
    for s in pd.unique(studies):
        m = studies == s
        sub = G.loc[m]
        # within-study percentile rank; zeros tie at the bottom, absent genera -> 0
        r = sub.rank(axis=0, method="average", pct=True)
        r = r.where(sub > 0, 0.0)
        out.loc[m] = r.values
    return out


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]