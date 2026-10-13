"""Genus-level relative abundance, per-study percentile-rank harmonization + random forest
trained with per-(study, class) balanced sample weights so study prevalence is not learnable."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def _study_from_meta(meta, index, n):
    if meta is None or "study" not in getattr(meta, "columns", []):
        return np.zeros(n, dtype=object)
    try:
        s = meta["study"].reindex(index)
        if s.isna().any():
            raise KeyError
        return s.astype(str).values
    except Exception:
        if len(meta) == n:
            return meta["study"].astype(str).values
        return np.zeros(n, dtype=object)


def _study_labels(items, ctx):
    meta = ctx.get("meta") if isinstance(ctx, dict) else None
    return _study_from_meta(meta, items.index, len(items))


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    G = rel.T.groupby(np.array(genus)).sum().T              # samples x genera
    studies = _study_labels(items, ctx)
    out = G.copy().astype(float)
    for s in pd.unique(studies):
        m = studies == s
        sub = G.loc[m]
        # within-study percentile ranks (ties averaged); robust to depth/region/protocol shifts
        out.loc[m] = (sub.rank(axis=0, method="average") / max(m.sum(), 1)).values
    return out


def _balanced_weights(studies, y):
    """Weight 1/(n in study x class cell), rescaled to mean 1: each study contributes
    equal case and control mass, removing study-prevalence as a learnable shortcut."""
    y = np.asarray(y)
    df = pd.DataFrame({"s": studies, "y": y})
    cell = df.groupby(["s", "y"])["y"].transform("size").values.astype(float)
    n_cls = df.groupby("s")["y"].transform("nunique").values.astype(float)
    study_n = df.groupby("s")["y"].transform("size").values.astype(float)
    # each study keeps total mass proportional to its size, split equally across its classes
    w = study_n / (n_cls * cell)
    return w * (len(w) / w.sum())


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    meta_tr = ctx.get("meta_train") if isinstance(ctx, dict) else None
    studies_tr = _study_from_meta(meta_tr, F_train.index, len(F_train))
    w = _balanced_weights(studies_tr, y_train)
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, np.asarray(y_train), sample_weight=w)
    return rf.predict_proba(Xte)[:, 1]