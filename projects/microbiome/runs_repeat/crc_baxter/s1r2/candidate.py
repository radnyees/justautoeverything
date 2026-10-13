"""Genus-level relative abundance (log-scaled) + aggregated CRC oral-pathogen marker features
+ training-fold prevalence filtering + training-fold univariate (Mann-Whitney) genus selection
+ training-fold co-abundance groups (CAGs) of genera (incl. sparse genera), top CAGs by Mann-Whitney
+ large extremely-randomized-trees ensemble."""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.ensemble import ExtraTreesClassifier

# Oral-pathogen genera consistently enriched in CRC stool across cohorts
CRC_MARKERS = ["Fusobacterium", "Peptostreptococcus", "Parvimonas", "Porphyromonas", "Gemella"]
MARKER_FEATS = ["marker_logsum", "marker_count"]
MIN_PREVALENCE = 0.10                                       # fraction of training samples with genus > 0
CAG_MIN_PREVALENCE = 0.03                                   # sparse genera (oral consortium) may still form CAGs
TOP_K = 40                                                  # genera kept after univariate case/control test
N_CAG = 5                                                   # co-abundance groups kept after the same test
CAG_DIST = 0.7                                              # cut height on 1 - Spearman rho (average linkage)
N_TREES = 3000                                              # more trees -> less Monte Carlo noise in ranking
LOG_FLOOR = 1e-5                                            # pseudo-abundance below typical detection limit


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    G = rel.T.groupby(np.array(genus)).sum().T              # samples x genera
    names = pd.Series([g.strip().replace("g__", "") for g in G.columns], index=G.columns)
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


def _co_abundance_groups(R, cols):
    """Cluster genera by Spearman co-abundance on the training fold; return groups of >= 2 genera."""
    if len(cols) < 3:
        return []
    rho = np.nan_to_num(np.asarray(spearmanr(R).correlation, dtype=float))
    D = np.clip(1.0 - (rho + rho.T) / 2.0, 0.0, 2.0)
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(D, checks=False), method="average")
    lab = fcluster(Z, t=CAG_DIST, criterion="distance")
    groups = [[cols[i] for i in np.where(lab == c)[0]] for c in np.unique(lab)]
    return [g for g in groups if len(g) >= 2]


def _cag_matrix(F, groups):
    return np.column_stack([np.log10(F[g].values.sum(axis=1) + LOG_FLOOR) for g in groups])


def fit_predict(F_train, y_train, F_test, ctx):
    genus_cols = [c for c in F_train.columns if c not in MARKER_FEATS]
    prev = (F_train[genus_cols] > 0).mean(axis=0)
    prevalent = [c for c in genus_cols if prev[c] >= MIN_PREVALENCE]
    cag_pool = [c for c in genus_cols if prev[c] >= CAG_MIN_PREVALENCE]
    keep = _select_genera(np.log10(F_train[prevalent].values + LOG_FLOOR), y_train, prevalent, TOP_K)
    mk = [c for c in MARKER_FEATS if c in F_train.columns]
    Fte = F_test.reindex(columns=genus_cols + mk, fill_value=0)

    # Co-abundance groups learned on the training fold (sparse genera included), applied to test by name
    groups = _co_abundance_groups(F_train[cag_pool].values, cag_pool)
    if groups:
        Ctr = _cag_matrix(F_train, groups)
        sel = _select_genera(Ctr, y_train, list(range(len(groups))), N_CAG)
        Ctr = Ctr[:, sel]
        Cte = _cag_matrix(Fte, [groups[i] for i in sel])
    else:
        Ctr = np.zeros((len(F_train), 0))
        Cte = np.zeros((len(F_test), 0))

    # Log-scale genus abundances so random split thresholds spread over the informative range
    Xtr = np.hstack([np.log10(F_train[keep].values + LOG_FLOOR), F_train[mk].values, Ctr])
    Xte = np.hstack([np.log10(Fte[keep].values + LOG_FLOOR), Fte[mk].values, Cte])
    et = ExtraTreesClassifier(n_estimators=N_TREES, random_state=0, n_jobs=-1)
    et.fit(Xtr, y_train)
    return et.predict_proba(Xte)[:, 1]