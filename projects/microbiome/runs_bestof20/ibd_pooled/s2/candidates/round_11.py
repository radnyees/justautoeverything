"""Genus-level CLR with per-study standardization (harmonization) + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def featurize(items, ctx):
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in items.columns]
    counts = items.T.groupby(np.array(genus)).sum().T              # samples x genera (raw counts)
    prev = (counts > 0).mean(axis=0)
    counts = counts.loc[:, prev >= 0.05]                            # label-free prevalence filter
    logc = np.log(counts.values.astype(float) + 0.5)
    clr = logc - logc.mean(axis=1, keepdims=True)                   # centered log-ratio
    return pd.DataFrame(clr, index=counts.index, columns=counts.columns)


def _study(meta, n):
    if meta is None or "study" not in getattr(meta, "columns", []):
        return np.zeros(n, dtype=object)
    return np.asarray(meta["study"].values)


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns
    Xtr = F_train[cols].values.astype(float)
    Xte = F_test.reindex(columns=cols, fill_value=0).values.astype(float)
    s_tr = _study(ctx.get("meta_train"), len(Xtr))
    s_te = _study(ctx.get("meta_test"), len(Xte))

    # per-study z-scoring (label-free harmonization) using all samples of that study
    X_all = np.vstack([Xtr, Xte])
    s_all = np.concatenate([s_tr, s_te]).astype(str)
    Z = np.empty_like(X_all)
    for s in np.unique(s_all):
        m = s_all == s
        mu = X_all[m].mean(axis=0)
        sd = X_all[m].std(axis=0)
        sd[sd < 1e-8] = 1.0
        Z[m] = (X_all[m] - mu) / sd
    Ztr, Zte = Z[: len(Xtr)], Z[len(Xtr):]

    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Ztr, y_train)
    return rf.predict_proba(Zte)[:, 1]