"""Genus-level relative abundance + composite CRC oral-pathobiont marker features + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Oral pathobionts consistently enriched in CRC stool across cohorts (Zeller 2014, Baxter 2016,
# Wirbel 2019 meta-analysis). Matched by genus name substring so it works across studies.
CRC_MARKERS = ["Fusobacterium", "Parvimonas", "Peptostreptococcus", "Porphyromonas", "Gemella"]
DETECT = 1e-4  # relative-abundance detection threshold (robust to sequencing-depth differences)


def _genus(c):
    for tok in c.split(";"):
        if tok.startswith("g__"):
            return tok
    return "g__"


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = np.array([_genus(c) for c in rel.columns])
    G = rel.T.groupby(genus).sum().T                        # samples x genera

    marker_cols = [g for g in G.columns if any(m.lower() in g.lower() for m in CRC_MARKERS)]
    M = G[marker_cols] if marker_cols else pd.DataFrame(np.zeros((len(G), 1)), index=G.index)
    msum = M.sum(axis=1)
    G = G.copy()
    G["__marker_sum"] = msum.values
    G["__marker_logsum"] = np.log10(msum.values + 1e-6)
    G["__marker_nprev"] = (M.values > DETECT).sum(axis=1).astype(float)
    return G


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]