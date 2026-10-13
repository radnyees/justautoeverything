"""Genus-level relative abundance + aggregated CRC oral-pathogen marker features
+ training-fold prevalence filtering; ensemble of random forest (raw relative
abundance) and L2 logistic regression (CLR genus abundances), probability-averaged."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Oral-pathogen genera consistently enriched in CRC stool across cohorts
CRC_MARKERS = ["Fusobacterium", "Peptostreptococcus", "Parvimonas", "Porphyromonas", "Gemella"]
MARKER_FEATS = ["marker_logsum", "marker_count"]
MIN_PREVALENCE = 0.10                                       # fraction of training samples with genus > 0
PSEUDO = 1e-5                                               # pseudocount on relative abundance for CLR


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    G = rel.T.groupby(np.array(genus)).sum().T              # samples x genera
    names = pd.Series([g.strip().replace("g__", "") for g in G.columns], index=G.columns)
    mk_cols = [c for c in G.columns if names[c] in CRC_MARKERS]
    if mk_cols:
        M = G[mk_cols]
        G["marker_logsum"] = np.log10(M.sum(axis=1) + 1e-5)
        G["marker_count"] = (M > 0).sum(axis=1).astype(float)
    else:
        G["marker_logsum"] = np.log10(1e-5)
        G["marker_count"] = 0.0
    return G


def _clr(A):
    L = np.log(A + PSEUDO)
    return L - L.mean(axis=1, keepdims=True)


def _inner_splits(X, y, groups):
    if groups is not None and len(np.unique(groups)) >= 5:
        return list(GroupKFold(n_splits=5).split(X, y, groups))
    return list(StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(X, y))


def fit_predict(F_train, y_train, F_test, ctx):
    y = np.asarray(y_train)
    genus_cols = [c for c in F_train.columns if c not in MARKER_FEATS]
    prev = (F_train[genus_cols] > 0).mean(axis=0)
    keep = [c for c in genus_cols if prev[c] >= MIN_PREVALENCE]
    mk = [c for c in MARKER_FEATS if c in F_train.columns]
    cols = keep + mk
    Te = F_test.reindex(columns=cols, fill_value=0)

    # Random forest on relative abundances + marker features
    Xtr, Xte = F_train[cols].values, Te.values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y)
    p_rf = rf.predict_proba(Xte)[:, 1]

    # L2 logistic regression on CLR genus abundances + marker features
    Ltr = np.hstack([_clr(F_train[keep].values), F_train[mk].values])
    Lte = np.hstack([_clr(Te[keep].values), Te[mk].values])
    sc = StandardScaler().fit(Ltr)
    Ltr, Lte = sc.transform(Ltr), sc.transform(Lte)
    groups = ctx.get("groups_train") if isinstance(ctx, dict) else None
    groups = np.asarray(groups) if groups is not None else None
    lr = LogisticRegressionCV(Cs=np.logspace(-4, 0, 9), penalty="l2", scoring="roc_auc",
                              cv=_inner_splits(Ltr, y, groups), max_iter=5000)
    lr.fit(Ltr, y)
    p_lr = lr.predict_proba(Lte)[:, 1]

    return 0.5 * p_rf + 0.5 * p_lr