"""Genus-level relative abundance + random forest (study x class balanced weights, min leaf size 3),
rank-averaged with an L2 logistic regression on per-study-centred CLR features at genus, family,
order, class and phylum level plus per-study-centred alpha-diversity (Shannon / Renyi-2 entropy)
and a beta-diversity dysbiosis score (median Bray-Curtis distance to same-study training controls,
computed out-of-fold for training samples)."""
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold

FAM_PREFIX = "FAM::"
DIV_PREFIX = "DIV::"
# higher taxonomic levels (lineage depth into the RDP string) used only by the linear model
HIGHER_LEVELS = {"ORD::": 4, "CLS::": 3, "PHY::": 2}
LR_PREFIXES = [FAM_PREFIX] + list(HIGHER_LEVELS)


def _entropies(P):
    P = np.asarray(P, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        lp = np.where(P > 0, np.log(np.where(P > 0, P, 1.0)), 0.0)
    shannon = -(P * lp).sum(axis=1)
    renyi2 = -np.log(np.clip((P ** 2).sum(axis=1), 1e-12, None))
    return shannon, renyi2


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    parts = [c.split(";") for c in rel.columns]
    genus = [p[5] if len(p) > 5 else "g__" for p in parts]
    family = [FAM_PREFIX + ";".join(p[1:5]) for p in parts]   # full lineage to family level
    G = rel.T.groupby(np.array(genus)).sum().T               # samples x genera
    Fm = rel.T.groupby(np.array(family)).sum().T             # samples x families
    higher = []
    for pref, depth in HIGHER_LEVELS.items():                # full lineage to order/class/phylum
        lab = [pref + ";".join(p[1:depth]) for p in parts]
        higher.append(rel.T.groupby(np.array(lab)).sum().T)
    sh_otu, r2_otu = _entropies(rel.values)
    sh_gen, _ = _entropies(G.values)
    D = pd.DataFrame({DIV_PREFIX + "shannon_lineage": sh_otu,
                      DIV_PREFIX + "renyi2_lineage": r2_otu,
                      DIV_PREFIX + "shannon_genus": sh_gen}, index=rel.index)
    return pd.concat([G, Fm] + higher + [D], axis=1)


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
    return _study_centre_scale(L, len(Rtr), st_tr, st_te)


def _study_centre_scale(L, n_tr, st_tr, st_te):
    """Subtract each study's (label-free) mean, then standardize with training stats."""
    L = np.array(L, dtype=float)
    st = np.concatenate([st_tr, st_te])
    for s in np.unique(st):
        m = st == s
        L[m] -= L[m].mean(axis=0)
    mu, sd = L[:n_tr].mean(axis=0), L[:n_tr].std(axis=0) + 1e-6
    L = (L - mu) / sd
    return L[:n_tr], L[n_tr:]


def _renorm(R):
    R = np.asarray(R, dtype=float)
    return R / np.clip(R.sum(axis=1, keepdims=True), 1e-12, None)


def _dysbiosis(Rtr, Rte, y, st_tr, st_te, groups):
    """Median Bray-Curtis distance to training controls of the same study (HMP2-style
    dysbiosis score). Training samples are scored against controls in other folds only."""
    Rtr, Rte = _renorm(Rtr), _renorm(Rte)
    ctrl = y == np.unique(y)[0]

    def score(Q, st_q, ref_idx):
        out = np.zeros(len(Q))
        for s in np.unique(st_q):
            mq = st_q == s
            ref = ref_idx[st_tr[ref_idx] == s]
            if len(ref) < 3:
                ref = ref_idx
            if len(ref) == 0:
                continue
            d = np.nan_to_num(cdist(Q[mq], Rtr[ref], metric="braycurtis"), nan=1.0)
            out[mq] = np.median(d, axis=1)
        return out

    if groups is not None and len(np.unique(groups)) >= 5:
        splits = GroupKFold(n_splits=5).split(Rtr, y, groups)
    else:
        splits = KFold(5, shuffle=True, random_state=0).split(Rtr)
    d_tr = np.zeros(len(Rtr))
    for tr, va in splits:
        d_tr[va] = score(Rtr[va], st_tr[va], tr[ctrl[tr]])
    d_te = score(Rte, st_te, np.where(ctrl)[0])
    return d_tr, d_te


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
    pref_masks = {p: np.array([str(c).startswith(p) for c in cols]) for p in LR_PREFIXES}
    is_div = np.array([str(c).startswith(DIV_PREFIX) for c in cols])
    is_gen = ~is_div
    for m in pref_masks.values():
        is_gen &= ~m
    Atr = F_train[cols].values.astype(float)
    Ate = F_test.reindex(columns=cols, fill_value=0).values.astype(float)
    Xtr, Xte = Atr[:, is_gen], Ate[:, is_gen]               # genus block
    Dtr, Dte = Atr[:, is_div], Ate[:, is_div]               # diversity block
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
    groups = ctx.get("groups_train", None)
    if groups is not None and len(groups) != len(y):
        groups = None
    groups = None if groups is None else np.asarray(groups)

    rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=3,
                                random_state=0, n_jobs=-1)
    rf.fit(Xtr, y, sample_weight=w)
    p_rf = rf.predict_proba(Xte)[:, 1]

    Gtr, Gte = _clr_centred(Xtr, Xte, st_tr, st_te)
    blocks_tr, blocks_te = [Gtr], [Gte]
    for p in LR_PREFIXES:                                   # family, order, class, phylum CLR
        m = pref_masks[p]
        if m.sum() > 1:
            Hr, He = _clr_centred(Atr[:, m], Ate[:, m], st_tr, st_te)
            blocks_tr.append(Hr)
            blocks_te.append(He)
    # beta-diversity dysbiosis score at genus and family level
    dys_tr, dys_te = [], []
    if len(np.unique(y)) == 2:
        for m in [is_gen, pref_masks[FAM_PREFIX]]:
            if m.sum() > 1:
                a, b = _dysbiosis(Atr[:, m], Ate[:, m], y, st_tr, st_te, groups)
                dys_tr.append(a)
                dys_te.append(b)
    if dys_tr:
        Dtr = np.column_stack([Dtr] + dys_tr)
        Dte = np.column_stack([Dte] + dys_te)
    if Dtr.shape[1] > 0:
        Ddr, Dde = _study_centre_scale(np.vstack([Dtr, Dte]), len(Dtr), st_tr, st_te)
        blocks_tr.append(Ddr)
        blocks_te.append(Dde)
    Ztr, Zte = np.hstack(blocks_tr), np.hstack(blocks_te)
    p_lr = _fit_lr(Ztr, y, w, st_tr, groups, Zte)

    n = len(Xte)
    return 0.5 * rankdata(p_rf) / n + 0.5 * rankdata(p_lr) / n