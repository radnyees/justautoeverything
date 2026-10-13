"""Genus-level CLR, z-scored within each study (batch harmonization), + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def _study_labels(items, ctx):
    meta = ctx.get("meta", None)
    if meta is None or "study" not in getattr(meta, "columns", []):
        return np.zeros(len(items), dtype=object)
    try:
        s = meta["study"].reindex(items.index)
        if s.isna().any():
            raise KeyError
        return s.astype(str).values
    except Exception:
        if len(meta) == len(items):
            return meta["study"].astype(str).values
        return np.zeros(len(items), dtype=object)


def featurize(items, ctx):
    counts = items.astype(float)
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in counts.columns]
    g = counts.T.groupby(np.array(genus)).sum().T              # samples x genera (counts)
    rel = g.div(g.sum(axis=1).clip(lower=1), axis=0)
    # mild prevalence filter: keep genera seen (>0) in at least 2% of samples
    keep = (g > 0).mean(axis=0) >= 0.02
    g, rel = g.loc[:, keep], rel.loc[:, keep]
    # CLR with a depth-relative pseudocount (half the min nonzero rel. abundance per sample)
    rv = rel.values
    nz = np.where(rv > 0, rv, np.inf).min(axis=1, keepdims=True)
    nz[~np.isfinite(nz)] = 1e-6
    logx = np.log(rv + 0.5 * nz)
    clr = logx - logx.mean(axis=1, keepdims=True)
    # per-study standardization (label-free harmonization)
    studies = _study_labels(items, ctx)
    Z = np.zeros_like(clr)
    for s in pd.unique(studies):
        m = studies == s
        mu = clr[m].mean(axis=0)
        sd = clr[m].std(axis=0)
        sd[sd < 1e-8] = 1.0
        Z[m] = (clr[m] - mu) / sd
    return pd.DataFrame(Z, index=items.index, columns=g.columns)


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns
    Xtr = F_train[cols].values
    Xte = F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=1000, max_features="sqrt", min_samples_leaf=2,
                                random_state=0, n_jobs=-1)
    rf.fit(Xtr, np.asarray(y_train))
    return rf.predict_proba(Xte)[:, 1]