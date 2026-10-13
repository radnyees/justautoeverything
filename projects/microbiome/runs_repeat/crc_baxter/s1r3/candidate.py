"""Genus-level relative abundance + aggregate CRC oral-pathobiont marker features;
rank-averaged ensemble of a random forest (genera + markers) and an L2 logistic regression
on multi-resolution (genus + family) log-abundances plus alpha-diversity summaries.
The blend weight is chosen per training fold from out-of-fold predictions (RF out-of-bag,
LR nested grouped CV)."""
import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

# Oral-origin pathobionts consistently enriched in CRC stool (Baxter 2016, Zeller 2014, Wirbel 2019)
CORE_MARKERS = ["g__Fusobacterium", "g__Porphyromonas", "g__Parvimonas", "g__Peptostreptococcus"]
EXT_MARKERS = CORE_MARKERS + ["g__Gemella", "g__Solobacterium", "g__Prevotella",
                              "g__Campylobacter", "g__Dialister", "g__Filifactor"]
PSEUDO = 5e-5          # ~half the smallest detectable relative abundance at typical MiSeq depth
MIN_PREV = 0.10        # prevalence filter for the linear model only (label-free, per training fold)
FAM_PREFIX = "FAM|"    # family-level aggregates, used by the linear model only
DIV_PREFIX = "DIV_"    # alpha-diversity summaries, used by the linear model only
RARE_DEPTH = 1000      # depth for expected (analytic) rarefied richness
W_GRID = [0.2, 0.35, 0.5, 0.65, 0.8]   # candidate LR weights in the rank blend (both models kept)


def _level(c, i, empty):
    parts = c.split(";")
    return parts[i] if len(parts) > i else empty


def _shannon(P):
    P = np.asarray(P, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(P > 0, P * np.log(P), 0.0)
    return -t.sum(axis=1)


def _rarefied_richness(C, m):
    """Expected number of OTUs observed when subsampling m reads without replacement."""
    C = np.asarray(C, dtype=float)
    N = C.sum(axis=1)
    out = np.empty(len(C))
    for k in range(len(C)):
        n = C[k][C[k] > 0]
        Nk = N[k]
        if Nk <= m:
            out[k] = len(n)
            continue
        rest = Nk - n
        ok = rest >= m
        p_abs = np.zeros_like(n)
        p_abs[ok] = np.exp(gammaln(rest[ok] + 1) - gammaln(rest[ok] - m + 1)
                           - gammaln(Nk + 1) + gammaln(Nk - m + 1))
        out[k] = np.sum(1.0 - p_abs)
    return out


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [_level(c, 5, "g__") for c in rel.columns]
    G = rel.T.groupby(np.array(genus)).sum().T          # samples x genera
    family = [FAM_PREFIX + _level(c, 4, "f__") for c in rel.columns]
    Fm = rel.T.groupby(np.array(family)).sum().T        # samples x families

    def col(name):
        return G[name] if name in G.columns else pd.Series(0.0, index=G.index)

    core = sum(col(g) for g in CORE_MARKERS)
    ext = sum(col(g) for g in EXT_MARKERS if g != "g__Prevotella")
    G = G.copy()
    G["MK_core_sum_log"] = np.log10(core + 1e-6)
    G["MK_ext_sum_log"] = np.log10(ext + 1e-6)
    G["MK_core_npresent"] = sum((col(g) > 0).astype(float) for g in CORE_MARKERS)
    G["MK_ext_npresent"] = sum((col(g) > 0).astype(float) for g in EXT_MARKERS)

    P = rel.values
    D = pd.DataFrame(index=rel.index)
    D[DIV_PREFIX + "shannon_otu"] = _shannon(P)
    D[DIV_PREFIX + "log_invsimpson"] = np.log(1.0 / np.clip((P ** 2).sum(axis=1), 1e-12, None))
    gcols = [c for c in G.columns if not c.startswith("MK_")]
    D[DIV_PREFIX + "shannon_genus"] = _shannon(G[gcols].values)
    D[DIV_PREFIX + "log_rare_rich"] = np.log1p(_rarefied_richness(items.values, RARE_DEPTH))
    return pd.concat([G, Fm, D], axis=1)


def _linear_scores(F_train, y_train, F_test, groups):
    extra = [c for c in F_train.columns if c.startswith("MK_") or c.startswith(DIV_PREFIX)]
    taxa = [c for c in F_train.columns if c not in set(extra)]   # genera + families
    prev = (F_train[taxa] > 0).mean(axis=0)
    keep = [g for g in taxa if prev[g] >= MIN_PREV]
    Xtr = np.hstack([np.log10(F_train[keep].values + PSEUDO), F_train[extra].values])
    Xte = np.hstack([np.log10(F_test.reindex(columns=keep, fill_value=0).values + PSEUDO),
                     F_test.reindex(columns=extra, fill_value=0).values])
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

    n_splits = min(5, len(np.unique(groups)))
    cv = list(GroupKFold(n_splits=n_splits).split(Xtr, y_train, groups))
    lr = LogisticRegressionCV(Cs=np.logspace(-4, 0, 9), cv=cv, penalty="l2",
                              scoring="roc_auc", max_iter=5000, n_jobs=-1)
    lr.fit(Xtr, y_train)
    return lr.predict_proba(Xte)[:, 1]


def _blend(p_rf, p_lr, w):
    n = len(p_rf)
    return (1.0 - w) * rankdata(p_rf) / n + w * rankdata(p_lr) / n


def fit_predict(F_train, y_train, F_test, ctx):
    y_train = np.asarray(y_train)
    groups = ctx.get("groups_train") if isinstance(ctx, dict) else None
    if groups is None:
        groups = np.arange(len(y_train))
    groups = np.asarray(groups)

    cols = [c for c in F_train.columns
            if not c.startswith(FAM_PREFIX) and not c.startswith(DIV_PREFIX)]  # RF: genera + markers
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1, oob_score=True)
    rf.fit(Xtr, y_train)
    p_rf = rf.predict_proba(Xte)[:, 1]

    p_lr = _linear_scores(F_train, y_train, F_test, groups)

    # Choose the blend weight from out-of-fold predictions on the training fold only.
    w_best = 0.5
    try:
        oob_rf = rf.oob_decision_function_[:, 1]
        oob_rf = np.nan_to_num(oob_rf, nan=0.5)
        oof_lr = np.zeros(len(y_train))
        n_splits = min(5, len(np.unique(groups)))
        for tr, va in GroupKFold(n_splits=n_splits).split(Xtr, y_train, groups):
            oof_lr[va] = _linear_scores(F_train.iloc[tr], y_train[tr],
                                        F_train.iloc[va], groups[tr])
        scores = [roc_auc_score(y_train, _blend(oob_rf, oof_lr, w)) for w in W_GRID]
        w_best = W_GRID[int(np.argmax(scores))]
    except Exception:
        w_best = 0.5

    return _blend(p_rf, p_lr, w_best)