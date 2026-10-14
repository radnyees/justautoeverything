"""Genus-level relative abundance, percentile-ranked within each study (batch harmonization) + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def _study_labels(items, ctx):
    meta = ctx.get("meta") if isinstance(ctx, dict) else None
    if meta is None or "study" not in getattr(meta, "columns", []):
        return np.zeros(len(items), dtype=object)
    if meta.index.equals(items.index):
        return meta["study"].values
    if set(items.index).issubset(set(meta.index)):
        return meta.loc[items.index, "study"].values
    if len(meta) == len(items):
        return meta["study"].values
    return np.zeros(len(items), dtype=object)


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    G = rel.T.groupby(np.array(genus)).sum().T              # samples x genera

    # Within-study percentile normalization: removes study-specific depth/region/protocol scale shifts
    study = _study_labels(G, ctx)
    out = pd.DataFrame(0.0, index=G.index, columns=G.columns)
    for s in pd.unique(study):
        m = np.asarray(study == s)
        sub = G.iloc[m]
        pct = sub.rank(axis=0, method="average", pct=True)
        absent = (sub > 0).sum(axis=0) == 0                  # genus never seen in this study -> 0
        pct.loc[:, absent.values] = 0.0
        out.iloc[np.where(m)[0]] = pct.values
    return out


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]