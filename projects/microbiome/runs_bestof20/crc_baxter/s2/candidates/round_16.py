"""Genus-level relative abundance + explicit CRC oral-pathogen marker features + random forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Oral-pathogen CRC signature (Baxter 2016, Zeller 2014, Wirbel 2019) -- names comparable across studies
CRC_MARKERS = ["Fusobacterium", "Parvimonas", "Peptostreptococcus", "Porphyromonas", "Gemella"]
# Butyrate producers typically depleted in CRC
PROTECTIVE = ["Faecalibacterium", "Roseburia", "Lachnospira", "Butyricicoccus", "Coprococcus"]


def _genus_name(col):
    parts = col.split(";")
    g = parts[5].strip() if len(parts) > 5 else "g__"
    return g


def _clean(g):
    return g.replace("g__", "").strip().strip("[]").strip()


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = np.array([_genus_name(c) for c in rel.columns])
    G = rel.T.groupby(genus).sum().T                        # samples x genera
    names = {g: _clean(g) for g in G.columns}
    mk = [g for g in G.columns if names[g] in CRC_MARKERS]
    pr = [g for g in G.columns if names[g] in PROTECTIVE]
    eps = 1e-5
    msum = G[mk].sum(axis=1) if mk else pd.Series(0.0, index=G.index)
    psum = G[pr].sum(axis=1) if pr else pd.Series(0.0, index=G.index)
    extra = pd.DataFrame({
        "mk__log_marker_sum": np.log10(msum + eps),
        "mk__n_markers_present": (G[mk] > 1e-4).sum(axis=1) if mk else 0.0,
        "mk__log_marker_vs_protective": np.log10(msum + eps) - np.log10(psum + eps),
    }, index=G.index)
    return pd.concat([G, extra], axis=1)


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]