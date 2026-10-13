"""Genus-level relative abundance, harmonized by within-study percentile ranks + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def _study_labels(items, ctx):
    meta = ctx.get("meta") if isinstance(ctx, dict) else None
    if meta is None or "study" not in getattr(meta, "columns", []):
        return pd.Series("all", index=items.index)
    st = meta["study"]
    if st.index.equals(items.index):
        return st.astype(str)
    if items.index.isin(st.index).all():
        return st.reindex(items.index).astype(str)
    if len(st) == len(items):
        return pd.Series(st.astype(str).values, index=items.index)
    return pd.Series("all", index=items.index)


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    G = rel.T.groupby(np.array(genus)).sum().T              # samples x genera
    study = _study_labels(items, ctx)
    # per-study percentile ranks (label-free harmonization of depth/region/protocol effects)
    R = G.groupby(study.values).rank(pct=True, method="average")
    R = R.reindex(index=G.index).fillna(0.5)
    return R


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0.5).values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]