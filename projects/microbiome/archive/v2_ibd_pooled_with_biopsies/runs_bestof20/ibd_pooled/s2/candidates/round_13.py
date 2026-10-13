"""Genus-level relative abundance, per-study percentile-rank harmonization + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    return rel.T.groupby(np.array(genus)).sum().T          # samples x genera


def _studies(meta, n):
    if meta is None or "study" not in getattr(meta, "columns", []):
        return np.zeros(n, dtype=object)
    return np.asarray(meta["study"].values, dtype=object)


def _per_study_rank(X, studies):
    """Percentile rank of each feature within each study (label-free harmonization)."""
    out = np.zeros_like(X, dtype=float)
    for s in pd.unique(studies):
        m = studies == s
        out[m] = pd.DataFrame(X[m]).rank(axis=0, method="average", pct=True).values
    return out


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    s_tr = _studies(ctx.get("meta_train"), len(Xtr))
    s_te = _studies(ctx.get("meta_test"), len(Xte))
    X_all = _per_study_rank(np.vstack([Xtr, Xte]), np.concatenate([s_tr, s_te]))
    Xtr, Xte = X_all[:len(Xtr)], X_all[len(Xtr):]
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]