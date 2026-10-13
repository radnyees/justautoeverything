"""Lineage-resolved genus-level relative abundance (unclassified genera keep their family identity),
per-study percentile-rank harmonization + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def _study_labels(items, ctx):
    meta = ctx.get("meta") if isinstance(ctx, dict) else None
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


def _genus_lineage(col):
    # full lineage truncated at genus; an empty genus falls back to its deepest named ancestor
    parts = [p.strip() for p in str(col).split(";")][:6]
    while len(parts) < 6:
        parts.append("")
    named = [p for p in parts if len(p) > 3]               # 'x__' alone means unassigned
    if not named:
        return "unassigned"
    key = ";".join(named)
    if len(parts[5]) <= 3:
        key += ";g__unclassified"
    return key


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    keys = np.array([_genus_lineage(c) for c in rel.columns])
    G = rel.T.groupby(keys).sum().T                          # samples x lineage-resolved genera
    studies = _study_labels(items, ctx)
    out = G.copy().astype(float)
    for s in pd.unique(studies):
        m = studies == s
        sub = G.loc[m]
        # within-study percentile ranks (ties averaged); robust to depth/region/protocol shifts
        out.loc[m] = (sub.rank(axis=0, method="average") / max(m.sum(), 1)).values
    return out


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]