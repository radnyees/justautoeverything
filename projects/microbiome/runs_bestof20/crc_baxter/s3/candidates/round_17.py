"""Genus-level relative abundance + oral-pathogen CRC marker consortium score + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Oral-pathobiont genera repeatedly enriched in CRC stool across cohorts
CRC_MARKERS = ["fusobacterium", "parvimonas", "peptostreptococcus", "porphyromonas", "gemella"]
PSEUDO = 1e-5      # pseudocount on relative abundance for the log score
DETECT = 1e-4      # relative-abundance detection threshold (depth-robust presence)


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    G = rel.T.groupby(np.array(genus)).sum().T              # samples x genera

    names = [str(g).lower().replace("g__", "").strip("[]' ") for g in G.columns]
    mk_cols = [G.columns[i] for i, n in enumerate(names)
               if any(n.startswith(m) for m in CRC_MARKERS)]
    M = G[mk_cols] if mk_cols else pd.DataFrame(np.zeros((len(G), 1)), index=G.index)
    # per-marker-genus sums are already in G; collapse duplicates by marker name
    feats = pd.DataFrame(index=G.index)
    logs, pres = [], []
    for m in CRC_MARKERS:
        cols = [G.columns[i] for i, n in enumerate(names) if n.startswith(m)]
        a = G[cols].sum(axis=1) if cols else pd.Series(0.0, index=G.index)
        logs.append(np.log10(a + PSEUDO) - np.log10(PSEUDO))
        pres.append((a > DETECT).astype(float))
    feats["__marker_logscore"] = np.sum(logs, axis=0)
    feats["__marker_npresent"] = np.sum(pres, axis=0)
    del M
    return pd.concat([G, feats], axis=1)


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]