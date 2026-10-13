"""Genus-level relative abundance + training-fold prevalence filtering + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

MIN_PREVALENCE = 0.10   # fraction of training samples in which a genus must be observed


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    return rel.T.groupby(np.array(genus)).sum().T          # samples x genera


def fit_predict(F_train, y_train, F_test, ctx):
    # label-free prevalence filter computed on the training fold only
    prev = (F_train > 0).mean(axis=0)
    cols = prev.index[prev >= MIN_PREVALENCE]
    cols = [c for c in cols if c != "g__"]                  # drop unassigned-genus bucket
    Xtr = F_train[cols].values
    Xte = F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]