"""Genus-level relative abundance + random forest, with study x class balanced sample weights
so that study identity (confounded with case/control ratio) carries no label information.
Genera are prevalence-filtered (label-free): kept if present in >=10% of samples of at least
one study, removing sparse noise genera that dilute the forest's candidate splits."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

MIN_PREV = 0.10


def _study_vector(items, ctx):
    meta = ctx.get("meta", None)
    if meta is None or "study" not in meta:
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
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    G = rel.T.groupby(np.array(genus)).sum().T              # samples x genera

    # label-free prevalence filter: keep genera prevalent in at least one study
    study = _study_vector(items, ctx)
    present = (G.values > 0)
    keep = np.zeros(G.shape[1], dtype=bool)
    for s in np.unique(study):
        ms = study == s
        if ms.sum() == 0:
            continue
        keep |= present[ms].mean(axis=0) >= MIN_PREV
    if keep.sum() < 10:                                     # safety fallback
        keep = present.mean(axis=0) > 0
    return G.loc[:, keep]


def _balanced_weights(y, study):
    """Within each study, cases and controls get equal total weight; each study keeps
    total weight proportional to its size. Removes the study <-> label confound."""
    y = np.asarray(y)
    study = np.asarray(study)
    w = np.ones(len(y), dtype=float)
    for s in np.unique(study):
        ms = study == s
        n_s = ms.sum()
        classes = np.unique(y[ms])
        if len(classes) < 2:
            # single-class study gives no within-study contrast; down-weight it
            w[ms] = 0.5
            continue
        for c in classes:
            m = ms & (y == c)
            w[m] = n_s / (len(classes) * m.sum())
    return w * len(y) / w.sum()


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    meta_tr = ctx.get("meta_train", None)
    if meta_tr is not None and "study" in meta_tr and len(meta_tr) == len(Xtr):
        w = _balanced_weights(y_train, meta_tr["study"].astype(str).values)
    else:
        w = None
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, np.asarray(y_train), sample_weight=w)
    return rf.predict_proba(Xte)[:, 1]