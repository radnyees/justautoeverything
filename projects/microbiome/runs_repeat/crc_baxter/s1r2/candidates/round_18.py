"""Genus-level relative abundance (log-scaled) + aggregated CRC oral-pathogen marker features
+ always-kept individual marker genera
+ training-fold prevalence filtering + training-fold univariate (Mann-Whitney) genus selection
+ large extremely-randomized-trees ensemble."""
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.ensemble import ExtraTreesClassifier

# Oral-pathogen genera consistently enriched in CRC stool across cohorts
CRC_MARKERS = ["Fusobacterium", "Peptostreptococcus", "Parvimonas", "Porphyromonas", "Gemella"]
MARKER_FEATS = ["marker_logsum", "marker_count"]
MIN_PREVALENCE = 0.10                                       # fraction of training samples with genus > 0
TOP_K = 40                                                  # genera kept after univariate case/control test
N_TREES = 3000                                              # more trees -> less Monte Carlo noise in ranking
LOG_FLOOR = 1e-5                                            # pseudo-abundance below typical detection limit


def _genus_name(c):
    return str(c).strip().replace("g__", "")


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    G = rel.T.groupby(np.array(genus)).sum().T              # samples x genera
    names = pd.Series([_genus_name(g) for g in G.columns], index=G.columns)
    mk_cols = [c for c in G.columns if names[c] in CRC_MARKERS]
    if mk_cols:
        M = G[mk_cols]
        G["marker_logsum"] = np.log10(M.sum(axis=1) + LOG_FLOOR)
        G["marker_count"] = (M > 0).sum(axis=1).astype(float)
    else:
        G["marker_logsum"] = np.log10(LOG_FLOOR)
        G["marker_count"] = 0.0
    return G


def _select_genera(X, y, cols, k):
    """Rank genera by two-sided Mann-Whitney p-value (cases vs controls) on the training fold."""
    y = np.asarray(y)
    if len(cols) <= k or len(np.unique(y)) < 2:
        return cols
    pos, neg = X[y == 1], X[y == 0]
    pvals = np.ones(len(cols))
    for j in range(len(cols)):
        a, b = pos[:, j], neg[:, j]
        if np.ptp(np.concatenate([a, b])) == 0:
            continue
        pvals[j] = mannwhitneyu(a, b, alternative="two-sided").pvalue
    order = np.argsort(pvals, kind="stable")[:k]
    return [cols[i] for i in sorted(order)]


def fit_predict(F_train, y_train, F_test, ctx):
    genus_cols = [c for c in F_train.columns if c not in MARKER_FEATS]
    # Individual marker genera bypass prevalence/selection filters (sparse but prior-known signal)
    marker_genera = [c for c in genus_cols if _genus_name(c) in CRC_MARKERS]
    other = [c for c in genus_cols if c not in marker_genera]
    prev = (F_train[other] > 0).mean(axis=0)
    keep = [c for c in other if prev[c] >= MIN_PREVALENCE]
    keep = _select_genera(np.log10(F_train[keep].values + LOG_FLOOR), y_train, keep, TOP_K)
    keep = keep + marker_genera
    mk = [c for c in MARKER_FEATS if c in F_train.columns]
    # Log-scale genus abundances so random split thresholds spread over the informative range
    Xtr = np.hstack([np.log10(F_train[keep].values + LOG_FLOOR), F_train[mk].values])
    Fte = F_test.reindex(columns=keep + mk, fill_value=0)
    Xte = np.hstack([np.log10(Fte[keep].values + LOG_FLOOR), Fte[mk].values])
    et = ExtraTreesClassifier(n_estimators=N_TREES, random_state=0, n_jobs=-1)
    et.fit(Xtr, y_train)
    return et.predict_proba(Xte)[:, 1]