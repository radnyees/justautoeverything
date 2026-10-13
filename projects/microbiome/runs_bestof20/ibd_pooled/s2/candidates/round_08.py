"""Genus-level relative abundance, per-study percentile-rank harmonization + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def _study_labels(items, ctx):
    meta = ctx.get("meta") if isinstance(ctx, dict) else None
    if meta is None or "study" not in meta:
        return np.zeros(len(items), dtype=object)
    st = meta["study"]
    if st.index.equals(items.index):
        return st.values
    if set(items.index).issubset(set(st.index)):
        return st.reindex(items.index).values
    return st.values[: len(items)]


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    gen = rel.T.groupby(np.array(genus)).sum().T                # samples x genera
    studies = pd.Series(_study_labels(items, ctx), index=gen.index).astype(str)
    out = gen.copy()
    for s in studies.unique():                                  # within-study percentile ranks
        m = (studies == s).values
        out.loc[m] = gen.loc[m].rank(axis=0, method="average", pct=True).values
    return out.astype(float)


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                      # align test genera to training genera
    Xtr = F_train[cols].values
    Xte = F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]