"""Genus-level log relative abundance, z-scored within study (batch harmonization) + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

PSEUDO = 1e-5


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    G = rel.T.groupby(np.array(genus)).sum().T              # samples x genera
    return np.log10(G + PSEUDO)


def _studies(meta, n):
    if meta is not None and "study" in meta and len(meta) == n:
        return meta["study"].astype(str).values
    return np.array(["_all"] * n)


def _per_study_z(Xtr, Xte, s_tr, s_te):
    """Z-score each feature within study; stats from all (label-free) samples of that study."""
    Xtr, Xte = Xtr.copy(), Xte.copy()
    X_all = np.vstack([Xtr, Xte])
    s_all = np.concatenate([s_tr, s_te])
    for s in np.unique(s_all):
        m_all = s_all == s
        mu = X_all[m_all].mean(axis=0)
        sd = X_all[m_all].std(axis=0)
        sd = np.where(sd > 1e-8, sd, 1.0)
        mtr, mte = s_tr == s, s_te == s
        if mtr.any():
            Xtr[mtr] = (Xtr[mtr] - mu) / sd
        if mte.any():
            Xte[mte] = (Xte[mte] - mu) / sd
    return Xtr, Xte


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr = F_train[cols].values.astype(float)
    Xte = F_test.reindex(columns=cols, fill_value=np.log10(PSEUDO)).values.astype(float)
    s_tr = _studies(ctx.get("meta_train"), len(Xtr))
    s_te = _studies(ctx.get("meta_test"), len(Xte))
    Xtr, Xte = _per_study_z(Xtr, Xte, s_tr, s_te)
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, np.asarray(y_train))
    return rf.predict_proba(Xte)[:, 1]