"""Genus-level relative abundance + known CRC oral-pathobiont composite features + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Oral pathobionts consistently enriched in CRC stool across cohorts (taxonomy names, study-agnostic)
CRC_MARKERS = ["fusobacterium", "parvimonas", "peptostreptococcus", "porphyromonas", "gemella"]
# Butyrate producers typically depleted in CRC
PROTECTIVE = ["faecalibacterium", "roseburia", "eubacterium", "lachnospira", "butyrivibrio"]


def _genus(c):
    parts = c.split(";")
    return parts[5] if len(parts) > 5 else "g__"


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = np.array([_genus(c) for c in rel.columns])
    G = rel.T.groupby(genus).sum().T                        # samples x genera

    gl = [g.lower().replace("g__", "").strip("[]") for g in G.columns]
    mk = [c for c, g in zip(G.columns, gl) if any(g.startswith(m) for m in CRC_MARKERS)]
    pr = [c for c, g in zip(G.columns, gl) if any(g.startswith(m) for m in PROTECTIVE)]
    eps = 1e-5
    msum = G[mk].sum(axis=1) if mk else pd.Series(0.0, index=G.index)
    psum = G[pr].sum(axis=1) if pr else pd.Series(0.0, index=G.index)
    extra = pd.DataFrame({
        "__marker_log": np.log10(msum + eps),
        "__marker_count": (G[mk] > 0).sum(axis=1) if mk else 0,
        "__marker_vs_protective": np.log10(msum + eps) - np.log10(psum + eps),
    }, index=G.index)
    return pd.concat([G, extra], axis=1)


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr = F_train[cols].values
    Xte = F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]