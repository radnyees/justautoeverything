"""Genus-level log relative abundance, per-study standardized (batch harmonization) + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    G = rel.T.groupby(np.array(genus)).sum().T              # samples x genera
    L = np.log10(G + 1e-5)                                  # log relative abundance

    meta = ctx.get("meta") if isinstance(ctx, dict) else None
    if meta is not None and "study" in meta:
        study = pd.Series(meta["study"].values, index=meta.index).reindex(L.index)
        if study.isna().any():                              # fall back to positional alignment
            study = pd.Series(meta["study"].values[: len(L)], index=L.index)
        out = L.copy()
        for s, idx in L.groupby(study.values).groups.items():
            block = L.loc[idx]
            mu = block.mean(axis=0)
            sd = block.std(axis=0, ddof=0)
            sd = sd.where(sd > 1e-6, 1.0)
            out.loc[idx] = (block - mu) / sd
        L = out
    return L


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]