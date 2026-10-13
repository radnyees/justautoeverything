"""Genus-level log relative abundance, robust-standardized WITHIN study (batch harmonization), + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    G = rel.T.groupby(np.array(genus)).sum().T              # samples x genera
    L = np.log10(G + 1e-5)

    meta = ctx.get("meta", None)
    if meta is None or "study" not in getattr(meta, "columns", []):
        return L
    study = meta["study"].reindex(L.index) if meta.index.equals(L.index) or L.index.isin(meta.index).all() \
        else pd.Series(meta["study"].values, index=L.index)
    study = study.fillna("_unknown").astype(str)

    out = L.copy()
    for s in study.unique():                                 # label-free per-study harmonization
        m = (study == s).values
        block = L.values[m]
        med = np.median(block, axis=0)
        q75, q25 = np.percentile(block, [75, 25], axis=0)
        scale = (q75 - q25) / 1.349
        sd = block.std(axis=0)
        scale = np.where(scale > 1e-8, scale, sd)
        scale = np.where(scale > 1e-8, scale, 1.0)
        out.values[m] = (block - med) / scale
    return out


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]