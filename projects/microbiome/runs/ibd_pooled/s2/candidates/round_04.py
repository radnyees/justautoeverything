"""Genus-level relative abundance, cross-study prevalence filter, percentile-normalized within study + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

MIN_PREV = 0.10          # within-study detection rate required
MIN_STUDY_FRAC = 0.5     # fraction of studies in which a genus must reach MIN_PREV


def _study_labels(items, ctx):
    meta = ctx.get("meta") if isinstance(ctx, dict) else None
    if meta is None or "study" not in getattr(meta, "columns", []):
        return None
    if meta.index.equals(items.index) or set(items.index).issubset(set(meta.index)):
        return meta["study"].reindex(items.index).astype(str).values
    if len(meta) == len(items):
        return meta["study"].astype(str).values
    return None


def _prevalence_filter(G, studies):
    detected = G > 0
    if studies is None:
        return G.loc[:, detected.mean(axis=0) >= MIN_PREV]
    uniq = pd.unique(studies)
    prev = np.vstack([detected.loc[studies == s].mean(axis=0).values for s in uniq])
    n_ok = (prev >= MIN_PREV).sum(axis=0)
    need = max(1, int(np.ceil(MIN_STUDY_FRAC * len(uniq))))
    keep = n_ok >= need
    if keep.sum() < 10:                                     # safety fallback
        keep = (prev >= MIN_PREV).any(axis=0)
    return G.loc[:, keep]


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    G = rel.T.groupby(np.array(genus)).sum().T              # samples x genera
    studies = _study_labels(items, ctx)
    G = _prevalence_filter(G, studies)
    if studies is None:
        return G.rank(pct=True)
    out = G.copy()
    for s in pd.unique(studies):
        m = studies == s
        out.loc[m] = G.loc[m].rank(axis=0, pct=True, method="average").values
    return out


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr = F_train[cols].values
    Xte = F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]