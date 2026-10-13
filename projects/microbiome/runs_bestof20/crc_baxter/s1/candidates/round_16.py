"""Multi-level taxonomic aggregation (phylum..genus lineages) of relative abundance + prevalence filter + RF."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

LEVELS = [1, 2, 3, 4, 5]          # phylum, class, order, family, genus (index into lineage)


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    parts = [c.split(";") for c in rel.columns]
    blocks = []
    for lv in LEVELS:
        # full lineage prefix as the name, so unclassified "g__" under different families stay distinct
        names = np.array([";".join(p[:lv + 1]) if len(p) > lv else ";".join(p) for p in parts])
        agg = rel.T.groupby(names).sum().T
        agg.columns = [f"L{lv}|{n}" for n in agg.columns]
        blocks.append(agg)
    F = pd.concat(blocks, axis=1)
    return F.loc[:, ~F.columns.duplicated()]


def fit_predict(F_train, y_train, F_test, ctx):
    prev = (F_train > 0).mean(axis=0)                       # label-free prevalence filter on train fold
    cols = F_train.columns[prev >= 0.05]
    Xtr = F_train[cols].values
    Xte = F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]