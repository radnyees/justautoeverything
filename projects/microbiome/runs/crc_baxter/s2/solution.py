"""Genus-level relative abundance + training-fold prevalence filter; weighted rank-average of RF (rel. abundance),
core-reference log-ratio logistic regression, and shallow CatBoost on core-reference log-ratio genera.
Log-ratios use a per-genus pseudocount (half the genus's minimum nonzero training abundance).
Blend weights are chosen by grouped inner CV on the training fold and shrunk halfway toward equal weights."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier

MIN_PREVALENCE = 0.05   # keep genera detected in >=5% of training samples
CORE_PREVALENCE = 0.8   # reference genera for log-ratios: detected in >=80% of training samples
PSEUDO = 1e-5           # fallback pseudocount for genera never observed in the training fold
WEIGHT_STEP = 10        # simplex grid resolution for blend weights (0.1 steps)
SHRINK = 0.5            # fraction of the inner-CV weights kept; rest is equal weighting


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    return rel.T.groupby(np.array(genus)).sum().T          # samples x genera


def _genus_pseudo(X):
    """Per-genus pseudocount: half the smallest nonzero abundance seen in training (label-free).
    Places zeros just below each genus's own detection floor instead of a fixed, often far-off value."""
    Xm = np.where(X > 0, X, np.inf)
    mn = Xm.min(axis=0)
    return np.where(np.isfinite(mn), 0.5 * mn, PSEUDO)


def _core_lr(X, core_idx, pseudo):
    """Log-ratio of each genus vs. the geometric mean of near-ubiquitous core genera.
    Unlike the full CLR centre, this reference is not dragged around by how many sparse genera are zero."""
    L = np.log(X + pseudo[None, :])
    return L - L[:, core_idx].mean(axis=1, keepdims=True)


def _splits(n, y, groups):
    if groups is not None and len(np.unique(groups)) >= 5:
        return list(GroupKFold(n_splits=5).split(np.zeros(n), y, groups))
    return list(StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(np.zeros(n), y))


def _members(F_train, y, F_test, groups):
    """Fit the three ensemble members on (F_train, y); return their test predictions (3 x n_test)."""
    prev = (F_train > 0).mean(axis=0)                       # label-free prevalence on training fold
    cols = prev.index[prev >= MIN_PREVALENCE]
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values

    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y)
    p_rf = rf.predict_proba(Xte)[:, 1]

    # reference set: core genera (exclude the unclassified "g__" bucket); fall back to all genera
    pv = prev[cols].values
    core_idx = np.where((pv >= CORE_PREVALENCE) & (np.asarray(cols) != "g__"))[0]
    if len(core_idx) < 5:
        core_idx = np.arange(len(cols))

    # compositional linear model on core-reference log-ratios; C tuned by inner CV on the training fold
    pseudo = _genus_pseudo(Xtr)
    Ctr, Cte = _core_lr(Xtr, core_idx, pseudo), _core_lr(Xte, core_idx, pseudo)
    sc = StandardScaler()
    Ztr = sc.fit_transform(Ctr)
    Zte = sc.transform(Cte)
    lr = LogisticRegressionCV(Cs=np.logspace(-4, 0, 9), cv=_splits(len(y), y, groups), penalty="l2",
                              scoring="roc_auc", max_iter=5000, n_jobs=-1)
    lr.fit(Ztr, y)
    p_lr = lr.predict_proba(Zte)[:, 1]

    # shallow, strongly regularized boosting on log-ratio genera: nonlinear/interaction view
    cb = CatBoostClassifier(iterations=600, depth=4, learning_rate=0.03, l2_leaf_reg=5,
                            rsm=0.5, random_seed=0, verbose=0, thread_count=-1,
                            allow_writing_files=False)
    cb.fit(Ctr, y)
    p_cb = cb.predict_proba(Cte)[:, 1]
    return np.vstack([p_rf, p_lr, p_cb])


def _inner_weights(F_train, y, groups):
    """Blend weights from grouped inner CV: mean within-fold AUROC of weighted member ranks."""
    g_all = None if groups is None else np.asarray(groups)
    folds = []
    for tr, te in _splits(len(y), y, g_all):
        g = None if g_all is None else g_all[tr]
        P = _members(F_train.iloc[tr], y[tr], F_train.iloc[te], g)
        folds.append((np.vstack([rankdata(p) for p in P]), y[te]))
    equal = np.ones(3) / 3.0
    best_w, best_s = equal, -np.inf
    for a in range(WEIGHT_STEP + 1):
        for b in range(WEIGHT_STEP + 1 - a):
            w = np.array([a, b, WEIGHT_STEP - a - b], dtype=float) / WEIGHT_STEP
            s = np.mean([roc_auc_score(yt, w @ R) for R, yt in folds if len(np.unique(yt)) == 2])
            s -= 1e-6 * np.abs(w - equal).sum()             # tie-break toward equal weights
            if s > best_s:
                best_w, best_s = w, s
    return SHRINK * best_w + (1.0 - SHRINK) * equal


def fit_predict(F_train, y_train, F_test, ctx):
    y = np.asarray(y_train)
    groups = ctx.get("groups_train")
    w = _inner_weights(F_train, y, groups)
    P = _members(F_train, y, F_test, groups)
    R = np.vstack([rankdata(p) for p in P])
    return (w @ R) / len(P[0])