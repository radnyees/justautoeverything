"""Multi-level taxonomy (phylum..genus, full lineage prefixes) relative abundance + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

LEVELS = [1, 2, 3, 4, 5]   # phylum, class, order, family, genus (index into lineage)


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    parts = [c.split(";") for c in rel.columns]
    blocks = []
    for lv in LEVELS:
        # key by full lineage prefix so unassigned taxa stay separated by their parent clade
        keys = np.array([";".join((p + [""] * 8)[:lv + 1]) for p in parts])
        blocks.append(rel.T.groupby(keys).sum().T)
    F = pd.concat(blocks, axis=1)
    F = F.loc[:, ~F.columns.duplicated()]
    return F                                                # samples x multi-level taxa


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test taxa to training taxa
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]