"""Genus-level relative abundance + pooled oral-pathogen CRC marker features + alpha diversity
(rarefied expected richness, Shannon) + prevalence filter + a dysbiosis index (scaled Aitchison
distance to the healthy-control CLR centroid, out-of-fold for training rows); rank-averaged
ensemble of a bagged tree ensemble (random forest or extremely randomized trees, with leaf size /
max_features / tree type chosen by out-of-bag AUROC on the training fold) and an elastic-net
logistic regression on CLR-transformed genera (depth-aware zero replacement: zeros -> half a read
at the sample's own depth; inner grouped CV jointly picks C and the L1 share, with pure L2
included as a candidate, and also chooses between per-genus standardization of the CLR block and
a single shared scale that keeps the natural log-ratio variances)."""
import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

# Oral-origin taxa consistently enriched in CRC stool across cohorts (Baxter, Zeller, Wirbel, ...)
CRC_MARKERS = ["g__Fusobacterium", "g__Parvimonas", "g__Peptostreptococcus",
               "g__Porphyromonas", "g__Gemella"]
PRESENCE_THR = 1e-4   # relative-abundance threshold, depth-robust notion of "detected"
MIN_PREVALENCE = 0.10  # keep genera detected in >= 10% of training samples
RAREFY_DEPTH = 1000    # fixed depth for expected (analytic) rarefied richness
ZERO_READS = 0.5       # zeros replaced by this many reads at the sample's own depth (for CLR)
DEPTH_COL = "__libsize"  # library size: used only for zero replacement, never as a feature
TREE_TYPES = {"rf": RandomForestClassifier, "et": ExtraTreesClassifier}  # OOB-selected
RF_LEAF_GRID = [1, 3, 5, 10]          # min_samples_leaf candidates (OOB-selected)
RF_MAXFEAT_GRID = ["sqrt", 0.3]       # max_features candidates (OOB-selected)
LR_CS = np.logspace(-4, 1, 11)        # inverse regularization strengths (inner CV)
LR_L1_RATIOS = [0.0, 0.2, 0.5, 0.8]   # elastic-net L1 share (0 = pure L2), inner CV
CLR_SCALINGS = ["per_genus", "shared"]  # CLR block scaling, chosen by inner CV


def _shannon(P):
    P = np.asarray(P, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        H = -np.where(P > 0, P * np.log(P), 0.0).sum(axis=1)
    return H


def _expected_richness(C, n):
    """E[#OTUs observed] when subsampling n reads without replacement (Hurlbert rarefaction)."""
    C = np.asarray(C, dtype=float)
    out = np.zeros(C.shape[0])
    for i in range(C.shape[0]):
        c = C[i][C[i] > 0]
        N = c.sum()
        if N <= n:
            out[i] = float(len(c))
            continue
        # P(OTU absent) = C(N - Ni, n) / C(N, n)
        a = N - c
        logp = np.full(len(c), -np.inf)
        ok = a >= n
        logp[ok] = (gammaln(a[ok] + 1) - gammaln(a[ok] - n + 1)
                    - gammaln(N + 1) + gammaln(N - n + 1))
        out[i] = float(np.sum(1.0 - np.exp(logp)))
    return out


def featurize(items, ctx):
    counts = items.values.astype(float)
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [c.split(";")[5].strip() if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    G = rel.T.groupby(np.array(genus)).sum().T              # samples x genera
    present = [m for m in CRC_MARKERS if m in G.columns]
    M = G[present] if present else pd.DataFrame(0.0, index=G.index, columns=["none"])
    G["__oral_marker_logsum"] = np.log10(M.sum(axis=1) + 1e-6)
    G["__oral_marker_count"] = (M > PRESENCE_THR).sum(axis=1).astype(float)
    taxa_cols = [c for c in G.columns if not c.startswith("__")]
    G["__shannon_otu"] = _shannon(rel.values)
    G["__shannon_genus"] = _shannon(G[taxa_cols].values)
    G["__rarefied_richness"] = _expected_richness(counts, RAREFY_DEPTH)
    G[DEPTH_COL] = np.clip(counts.sum(axis=1), 1, None)
    return G


def _clr(X, depth):
    """CLR with multiplicative zero replacement at each sample's own detection limit."""
    floor = (ZERO_READS / np.asarray(depth, dtype=float))[:, None]
    Xr = np.where(X > 0, X, floor)
    L = np.log(Xr)
    return L - L.mean(axis=1, keepdims=True)


def _dysbiosis(C_ref, C_query):
    """log RMS distance of query CLR profiles to the reference (healthy) centroid, each genus
    scaled by its healthy spread (floored so near-constant genera cannot dominate)."""
    mu = C_ref.mean(axis=0)
    sd = C_ref.std(axis=0)
    sd = np.maximum(sd, 0.5 * np.median(sd) + 1e-6)
    d = np.sqrt(np.mean(((C_query - mu) / sd) ** 2, axis=1))
    return np.log(d + 1e-6)


def _fit_trees_oob(X, y):
    """Pick tree type (RF / bootstrapped ExtraTrees), min_samples_leaf and max_features by
    out-of-bag AUROC (training fold only)."""
    best, best_auc = ("rf", 1, "sqrt"), -np.inf
    for kind, cls in TREE_TYPES.items():
        for mf in RF_MAXFEAT_GRID:
            for leaf in RF_LEAF_GRID:
                m = cls(n_estimators=500, min_samples_leaf=leaf, max_features=mf,
                        bootstrap=True, oob_score=True, random_state=0, n_jobs=-1)
                m.fit(X, y)
                oob = m.oob_decision_function_[:, 1]
                ok = np.isfinite(oob)
                if ok.sum() < 10 or len(np.unique(y[ok])) < 2:
                    continue
                auc = roc_auc_score(y[ok], oob[ok])
                if auc > best_auc + 1e-9:
                    best, best_auc = (kind, leaf, mf), auc
    kind, leaf, mf = best
    m = TREE_TYPES[kind](n_estimators=1000, min_samples_leaf=leaf, max_features=mf,
                         bootstrap=True, random_state=0, n_jobs=-1)
    m.fit(X, y)
    return m


def _lr_design(Ctr, Etr, Cte, Ete, scaling):
    """Per-genus standardization, or one shared scale for the whole CLR block (keeps the natural
    log-ratio variances, so depth-noise-dominated rare genera are not inflated)."""
    if scaling == "per_genus":
        sc = StandardScaler().fit(np.hstack([Ctr, Etr]))
        return sc.transform(np.hstack([Ctr, Etr])), sc.transform(np.hstack([Cte, Ete]))
    mu = Ctr.mean(axis=0)
    s = np.sqrt(Ctr.var(axis=0).mean()) + 1e-9
    esc = StandardScaler().fit(Etr)
    Ztr = np.hstack([(Ctr - mu) / s, esc.transform(Etr)])
    Zte = np.hstack([(Cte - mu) / s, esc.transform(Ete)])
    return Ztr, Zte


def fit_predict(F_train, y_train, F_test, ctx):
    y_train = np.asarray(y_train)
    eng = [c for c in F_train.columns if c.startswith("__") and c != DEPTH_COL]
    taxa = [c for c in F_train.columns if not c.startswith("__")]
    prev = (F_train[taxa] > PRESENCE_THR).mean(axis=0)      # label-free, training fold only
    keep = [c for c in taxa if prev[c] >= MIN_PREVALENCE]
    marker_keep = [m for m in CRC_MARKERS if m in F_train.columns and m not in keep]
    tcols = keep + marker_keep                              # always retain known markers
    cols = tcols + eng
    Ftr = F_train[cols]
    Fte = F_test.reindex(columns=cols, fill_value=0)
    dtr = F_train[DEPTH_COL].values
    dte = F_test[DEPTH_COL].values

    groups = ctx.get("groups_train")
    groups = np.asarray(groups) if groups is not None else np.arange(len(y_train))
    splits = list(GroupKFold(n_splits=5).split(Ftr.values, y_train, groups))

    # Dysbiosis index: distance to healthy CLR centroid (out-of-fold for training rows)
    Ctr = _clr(Ftr[tcols].values, dtr)
    Cte = _clr(Fte[tcols].values, dte)
    dys_tr = np.zeros(len(y_train))
    for tr_idx, va_idx in splits:
        ref = Ctr[tr_idx][y_train[tr_idx] == 0]
        dys_tr[va_idx] = _dysbiosis(ref, Ctr[va_idx])
    dys_te = _dysbiosis(Ctr[y_train == 0], Cte)

    # Tree ensemble (RF or ExtraTrees, OOB-selected) on relative abundances + engineered features
    Xtr = np.hstack([Ftr.values, dys_tr[:, None]])
    Xte = np.hstack([Fte.values, dys_te[:, None]])
    trees = _fit_trees_oob(Xtr, y_train)
    p_rf = trees.predict_proba(Xte)[:, 1]

    # Elastic-net logistic regression on CLR genera + engineered features
    # (C, L1 share and CLR-block scaling tuned jointly by inner grouped CV)
    Etr = np.hstack([Ftr[eng].values, dys_tr[:, None]])
    Ete = np.hstack([Fte[eng].values, dys_te[:, None]])
    best_lr, best_score, best_Zte = None, -np.inf, None
    for scaling in CLR_SCALINGS:
        Ztr, Zte = _lr_design(Ctr, Etr, Cte, Ete, scaling)
        lr = LogisticRegressionCV(Cs=LR_CS, cv=splits, penalty="elasticnet", solver="saga",
                                  l1_ratios=LR_L1_RATIOS, scoring="roc_auc", max_iter=5000,
                                  random_state=0, n_jobs=-1)
        lr.fit(Ztr, y_train)
        sc_arr = np.asarray(list(lr.scores_.values())[0])
        score = float(np.nanmax(np.nanmean(sc_arr, axis=0)))
        if score > best_score + 1e-9:
            best_lr, best_score, best_Zte = lr, score, Zte
    p_lr = best_lr.predict_proba(best_Zte)[:, 1]

    n = len(p_rf)
    return 0.5 * rankdata(p_rf) / n + 0.5 * rankdata(p_lr) / n