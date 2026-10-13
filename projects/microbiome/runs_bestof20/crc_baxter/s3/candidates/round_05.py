"""Genus-level CLR (centered log-ratio, pseudocount 0.5) + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def _genus_counts(items):
    genus = []
    for c in items.columns:
        parts = c.split(";")
        genus.append(parts[5] if len(parts) > 5 else "g__")
    return items.T.groupby(np.array(genus)).sum().T          # samples x genera (raw counts)


def featurize(items, ctx):
    counts = _genus_counts(items).astype(float)
    # Return raw genus counts; CLR is done in fit_predict on the aligned training genus set
    return counts


def _clr(X, pseudo=0.5):
    L = np.log(X + pseudo)
    return L - L.mean(axis=1, keepdims=True)


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns[(F_train.values > 0).any(axis=0)]  # genera observed in training
    Xtr = F_train[cols].values.astype(float)
    Xte = F_test.reindex(columns=cols, fill_value=0).values.astype(float)
    Xtr, Xte = _clr(Xtr), _clr(Xte)
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]