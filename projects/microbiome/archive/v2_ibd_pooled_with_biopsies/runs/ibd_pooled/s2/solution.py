"""Genus-level relative abundance + random forest, with study x class balanced sample weights
so the pooled model cannot use study identity (differing case/control ratios) as a label proxy.
Genera are restricted to those reliably detected in >=2 training studies (label-free filter),
removing study-specific / noise genera that only encode batch.
A CLR-transformed L2 logistic regression (C tuned by grouped inner CV) is rank-averaged with
the RF: compositional log-ratios capture smooth additive dysbiosis gradients trees miss.
RF leaf size / feature fraction are tuned by grouped inner CV scored with within-study AUROC
(the target metric), regularizing trees against study-specific idiosyncrasies.
Depth harmonization: studies differ strongly in sequencing depth, so presence/absence and
per-sample pseudocounts encode batch. A common detection limit L (1 / low-quantile training
depth) zeroes genus abundances below L and a single pseudocount 0.5*L is used for CLR, making
zeros and log-ratios comparable across studies.
Hurdle-style linear model: because zeros now mean "below a common detection limit" in every
study, the logistic model also receives presence/absence indicators per genus, so taxon loss
(a hallmark of IBD dysbiosis) is modelled separately from abundance shifts among carriers."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

MIN_PREV = 0.10      # within-study detection prevalence
MIN_STUDIES = 2      # number of training studies in which the genus must reach MIN_PREV
RF_GRID = [(1, "sqrt"), (3, "sqrt"), (10, "sqrt"), (1, 0.2), (3, 0.2), (10, 0.2)]
RF_INNER_TREES = 250
DEPTH_COL = "__depth__"
DEPTH_Q = 0.10       # detection limit = 1 / this quantile of training read depth
L_BOUNDS = (1e-4, 2e-3)


def featurize(items, ctx):
    depth = items.sum(axis=1).clip(lower=1)
    rel = items.div(depth, axis=0)
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    G = rel.T.groupby(np.array(genus)).sum().T              # samples x genera
    G[DEPTH_COL] = depth.values.astype(float)
    return G


def _train_study(y, ctx):
    meta = ctx.get("meta_train") if isinstance(ctx, dict) else None
    if meta is None or "study" not in getattr(meta, "columns", []) or len(meta) != len(y):
        return np.zeros(len(y), dtype=object)
    return np.asarray(meta["study"].astype(str).values)


def _study_class_weights(y, study):
    y = np.asarray(y)
    w = np.ones(len(y), dtype=float)
    for s in np.unique(study):
        in_s = study == s
        classes = np.unique(y[in_s])
        for c in classes:
            m = in_s & (y == c)
            # each (study, class) cell gets equal total weight within its study
            w[m] = in_s.sum() / (len(classes) * m.sum())
    return w * (len(w) / w.sum())


def _shared_genera(F_train, study):
    X = F_train.values > 0
    studies = np.unique(study)
    need = min(MIN_STUDIES, len(studies))
    count = np.zeros(X.shape[1], dtype=int)
    for s in studies:
        count += (X[study == s].mean(axis=0) >= MIN_PREV).astype(int)
    keep = count >= need
    if keep.sum() < 10:                                     # safety fallback
        keep = X.mean(axis=0) > 0
    return F_train.columns[keep]


def _clr(X, pc):
    X = np.array(X, dtype=float, copy=True)
    L = np.log(X + pc)                                      # common, depth-independent pseudocount
    return L - L.mean(axis=1, keepdims=True)


def _inner_splits(y, groups):
    y = np.asarray(y)
    if groups is not None and len(groups) == len(y) and len(np.unique(groups)) >= 5:
        return list(GroupKFold(n_splits=5).split(np.zeros(len(y)), y, np.asarray(groups)))
    return list(StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
                .split(np.zeros(len(y)), y))


def _within_study_auc(y, p, study):
    aucs, ns = [], []
    for s in np.unique(study):
        m = study == s
        if len(np.unique(y[m])) == 2:
            aucs.append(roc_auc_score(y[m], p[m]))
            ns.append(m.sum())
    if not aucs:
        return roc_auc_score(y, p) if len(np.unique(y)) == 2 else 0.5
    return float(np.average(aucs, weights=ns))


def _tune_rf(Xtr, y, w, study, splits):
    best, best_score = RF_GRID[0], -np.inf
    oof = {cfg: np.zeros(len(y)) for cfg in RF_GRID}
    try:
        for tr, va in splits:
            if len(np.unique(y[tr])) < 2:
                continue
            for cfg in RF_GRID:
                rf = RandomForestClassifier(n_estimators=RF_INNER_TREES, min_samples_leaf=cfg[0],
                                            max_features=cfg[1], random_state=0, n_jobs=-1)
                rf.fit(Xtr[tr], y[tr], sample_weight=w[tr])
                oof[cfg][va] = rf.predict_proba(Xtr[va])[:, 1]
        for cfg in RF_GRID:
            sc = _within_study_auc(y, oof[cfg], study)
            if sc > best_score:
                best, best_score = cfg, sc
    except Exception:
        return RF_GRID[0]
    return best


def _apply_floor(F, L):
    G = F.drop(columns=[DEPTH_COL], errors="ignore")
    V = G.values.astype(float, copy=True)
    V[V < L] = 0.0
    return pd.DataFrame(V, index=G.index, columns=G.columns)


def _hurdle_design(X, pc):
    """CLR abundances plus presence/absence indicators (common detection limit)."""
    return np.hstack([_clr(X, pc), (np.asarray(X) > 0).astype(float)])


def fit_predict(F_train, y_train, F_test, ctx):
    y = np.asarray(y_train)
    study = _train_study(y, ctx)

    if DEPTH_COL in F_train.columns:
        d_q = float(np.quantile(F_train[DEPTH_COL].values, DEPTH_Q))
        L = float(np.clip(1.0 / max(d_q, 1.0), *L_BOUNDS))
    else:
        L = L_BOUNDS[0]
    Ftr = _apply_floor(F_train, L)
    Fte = _apply_floor(F_test, L)

    cols = _shared_genera(Ftr, study)                       # align test genera to kept genera
    Xtr, Xte = Ftr[cols].values, Fte.reindex(columns=cols, fill_value=0).values
    w = _study_class_weights(y, study)
    groups = ctx.get("groups_train") if isinstance(ctx, dict) else None
    splits = _inner_splits(y, groups)

    leaf, mf = _tune_rf(Xtr, y, w, study, splits)
    rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=leaf, max_features=mf,
                                random_state=0, n_jobs=-1)
    rf.fit(Xtr, y, sample_weight=w)
    p_rf = rf.predict_proba(Xte)[:, 1]

    try:
        sc = StandardScaler()
        Ztr = sc.fit_transform(_hurdle_design(Xtr, 0.5 * L))
        Zte = sc.transform(_hurdle_design(Xte, 0.5 * L))
        lr = LogisticRegressionCV(Cs=np.logspace(-4, 0, 9), cv=splits,
                                  penalty="l2", scoring="roc_auc", max_iter=5000,
                                  solver="lbfgs", n_jobs=-1)
        lr.fit(Ztr, y, sample_weight=w)
        p_lr = lr.predict_proba(Zte)[:, 1]
    except Exception:
        return p_rf

    n = len(p_rf)
    return 0.5 * (rankdata(p_rf) + rankdata(p_lr)) / n