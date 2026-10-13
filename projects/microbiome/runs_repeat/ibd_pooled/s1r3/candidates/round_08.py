"""Genus-level relative abundance + random forest (study x class balanced weights), rank-averaged
with an L2 logistic regression on per-study-centred genus + family CLR features."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

FAM_PREFIX = "FAM::"


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    parts = [c.split(";") for c in rel.columns]
    genus = [p[5] if len(p) > 5 else "g__" for p in parts]
    family = [FAM_PREFIX + ";".join(p[1:5]) for p in parts]   # full lineage to family level
    G = rel.T.groupby(np.array(genus)).sum().T               # samples x genera
    Fm = rel.T.groupby(np.array(family)).sum().T             # samples x families
    return pd.concat([G, Fm], axis=1)


def _balanced_weights(y, study):
    """Within each study, cases and controls get equal total weight; each study keeps
    total weight proportional to its size. Removes the study <-> label confound."""
    y = np.asarray(y)
    study = np.asarray(study)
    w = np.ones(len(y), dtype=float)
    for s in np.unique(study):
        ms = study == s
        n_s = ms.sum()
        classes = np.unique(y[ms])
        if len(classes) < 2:
            # single-class study gives no within-study contrast; down-weight it
            w[ms] = 0.5
            continue
        for c in classes:
            m = ms & (y == c)
            w[m] = n_s / (len(classes) * m.sum())
    return w * len(y) / w.sum()


def _clr_centred(Rtr, Rte, st_tr, st_te, min_prev=0.05, eps=1e-5):
    """CLR on prevalent taxa, then subtract each study's (label-free) mean profile."""
    keep = (Rtr > 0).mean(axis=0) >= min_prev
    L = np.log(np.vstack([Rtr[:, keep], Rte[:, keep]]) + eps)
    L = L - L.mean(axis=1, keepdims=True)
    st = np.concatenate([st_tr, st_te])
    for s in np.unique(st):
        m = st == s
        L[m] -= L[m].mean(axis=0)
    mu, sd = L[:len(Rtr)].mean(axis=0), L[:len(Rtr)].std(axis=0) + 1e-6
    L = (L - mu) / sd
    return L[:len(Rtr)], L[len(Rtr):]


def _within_auc(y, p, study, w=None):
    """Mean within-study AUROC over studies with both classes."""
    aucs = []
    for s in np.unique(study):
        m = study == s
        if len(np.unique(y[m])) == 2:
            aucs.append(roc_auc_score(y[m], p[m]))
    return np.mean(aucs) if aucs else 0.5


def _fit_lr(Xtr, y, w, study, groups, Xte):
    Cs = [0.003, 0.01, 0.03, 0.1, 0.3]
    if groups is not None and len(np.unique(groups)) >= 3:
        splits = list(GroupKFold(n_splits=3).split(Xtr, y, groups))
    else:
        splits = list(StratifiedKFold(3, shuffle=True, random_state=0).split(Xtr, y))
    best_c, best = Cs[0], -1
    for C in Cs:
        oof = np.zeros(len(y))
        for tr, va in splits:
            lr = LogisticRegression(C=C, max_iter=2000)
            lr.fit(Xtr[tr], y[tr], sample_weight=w[tr])
            oof[va] = lr.decision_function(Xtr[va])
        sc = _within_auc(y, oof, study)
        if sc > best:
            best, best_c = sc, C
    lr = LogisticRegression(C=best_c, max_iter=2000)
    lr.fit(Xtr, y, sample_weight=w)
    return lr.decision_function(Xte)


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test taxa to training taxa
    is_fam = np.array([str(c).startswith(FAM_PREFIX) for c in cols])
    Atr = F_train[cols].values
    Ate = F_test.reindex(columns=cols, fill_value=0).values
    Xtr, Xte = Atr[:, ~is_fam], Ate[:, ~is_fam]             # genus block
    Ftr, Fte = Atr[:, is_fam], Ate[:, is_fam]               # family block
    y = np.asarray(y_train)
    meta_tr, meta_te = ctx.get("meta_train", None), ctx.get("meta_test", None)
    if meta_tr is not None and "study" in meta_tr and len(meta_tr) == len(Xtr):
        st_tr = meta_tr["study"].astype(str).values
        w = _balanced_weights(y, st_tr)
    else:
        st_tr = np.array(["all"] * len(Xtr))
        w = np.ones(len(y))
    if meta_te is not None and "study" in meta_te and len(meta_te) == len(Xte):
        st_te = meta_te["study"].astype(str).values
    else:
        st_te = np.array(["all"] * len(Xte))

    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y, sample_weight=w)
    p_rf = rf.predict_proba(Xte)[:, 1]

    Gtr, Gte = _clr_centred(Xtr, Xte, st_tr, st_te)
    if Ftr.shape[1] > 1:
        Hfr, Hfe = _clr_centred(Ftr, Fte, st_tr, st_te)
        Ztr, Zte = np.hstack([Gtr, Hfr]), np.hstack([Gte, Hfe])
    else:
        Ztr, Zte = Gtr, Gte
    groups = ctx.get("groups_train", None)
    if groups is not None and len(groups) != len(y):
        groups = None
    p_lr = _fit_lr(Ztr, y, w, st_tr, None if groups is None else np.asarray(groups), Zte)

    n = len(Xte)
    return 0.5 * rankdata(p_rf) / n + 0.5 * rankdata(p_lr) / n