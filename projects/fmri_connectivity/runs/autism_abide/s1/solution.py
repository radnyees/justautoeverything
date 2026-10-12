"""CC200, two preprocessing strategies (no GSR and GSR): Ledoit-Wolf covariance of z-scored
ROI series -> per-site Riemannian re-centering (label-free harmonization) -> tangent space at
the identity -> per-site feature scale equalization (shrunk toward pooled SD) -> standardized
L2 logistic regression per strategy, with C chosen per strategy by grouped inner CV on
within-site AUROC; the per-strategy decision functions (scaled by their inner-OOF SD) are
blended with simplex weights chosen on the inner OOF within-site AUROC, shrunk toward equal."""
import itertools

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from pyriemann.utils.mean import mean_riemann


C_GRID = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
W_STEP = 0.1
W_SHRINK = 0.5  # fraction of weight pulled back toward the equal blend


def _covs(ts):
    covs = []
    for x in ts:
        x = np.nan_to_num(np.asarray(x, dtype=np.float64))
        sd = x.std(axis=0)
        dead = sd < 1e-8
        x = (x - x.mean(axis=0)) / np.where(dead, 1.0, sd)
        x[:, dead] = 0.0
        c = LedoitWolf(assume_centered=True).fit(x).covariance_
        # Out-of-FOV ROIs: decouple and give unit variance so the matrix stays SPD
        c[dead, :] = 0.0
        c[:, dead] = 0.0
        c[dead, dead] = 1.0
        c = 0.5 * (c + c.T)
        covs.append(c.astype(np.float32))
    return np.asarray(covs, dtype=np.float32)


def featurize(items, ctx):
    strategies = [s for s in ("filt_noglobal", "filt_global") if s in ctx["strategies"]]
    if not strategies:
        strategies = [ctx["strategies"][0]]
    blocks = [_covs(ctx["load_timeseries"](items, "cc200", s)) for s in strategies]
    return np.stack(blocks, axis=1)  # (n_items, n_strategies, p, p)


def _sites(meta, n):
    try:
        s = pd.DataFrame(meta)["site"].astype(str).values
        if len(s) == n:
            return s
    except Exception:
        pass
    return np.array(["all"] * n)


def _powm(c, power):
    w, v = np.linalg.eigh(c)
    w = np.clip(w, 1e-10, None)
    return (v * w ** power) @ v.T


def _log_vec(c, iu, coef):
    w, v = np.linalg.eigh(0.5 * (c + c.T))
    w = np.clip(w, 1e-10, None)
    L = (v * np.log(w)) @ v.T
    return L[iu] * coef


def _tangent_features(C, sites):
    p = C.shape[1]
    C = C + 1e-6 * np.eye(p)
    iu = np.triu_indices(p)
    coef = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    X = np.empty((C.shape[0], len(iu[0])), dtype=np.float64)
    # Per-site re-centering: parallel-transport each site's mean to the identity
    for s in np.unique(sites):
        idx = np.where(sites == s)[0]
        M = mean_riemann(C[idx], maxiter=50) if len(idx) > 1 else C[idx[0]]
        W = _powm(M, -0.5)
        for i in idx:
            X[i] = _log_vec(W @ C[i] @ W, iu, coef)

    # Per-site scale equalization (label-free): z-score each feature within site,
    # with the site variance shrunk toward the pooled variance for small sites.
    pooled_var = X.var(axis=0) + 1e-12
    k = 10.0
    for s in np.unique(sites):
        idx = np.where(sites == s)[0]
        n_s = len(idx)
        mu = X[idx].mean(axis=0)
        var_s = X[idx].var(axis=0) if n_s > 1 else pooled_var
        v = (n_s * var_s + k * pooled_var) / (n_s + k)
        X[idx] = (X[idx] - mu) / np.sqrt(v + 1e-12)
    return X


def _within_site_auc(y, s, sites):
    tot, acc = 0, 0.0
    for st in np.unique(sites):
        m = sites == st
        if len(np.unique(y[m])) < 2:
            continue
        acc += m.sum() * roc_auc_score(y[m], s[m])
        tot += m.sum()
    return acc / tot if tot > 0 else 0.5


def _select_C(X, y, sites, groups):
    """Returns the chosen C and the inner out-of-fold decision values at that C."""
    if groups is None or len(np.unique(groups)) < 5:
        groups = np.arange(len(y))
    oof = {c: np.zeros(len(y)) for c in C_GRID}
    for tr, va in GroupKFold(n_splits=5).split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        for c in C_GRID:
            clf = LogisticRegression(C=c, max_iter=5000)
            clf.fit(X[tr], y[tr])
            oof[c][va] = clf.decision_function(X[va])
    scores = {c: _within_site_auc(y, oof[c], sites) for c in C_GRID}
    c_best = max(C_GRID, key=lambda c: (scores[c], -abs(np.log(c / 1e-3))))
    return c_best, oof[c_best]


def _simplex_grid(n, step):
    m = int(round(1.0 / step))
    for comb in itertools.product(range(m + 1), repeat=n):
        if sum(comb) == m:
            yield np.asarray(comb, dtype=np.float64) / m


def _select_weights(oofs, y, sites):
    n = len(oofs)
    eq = np.full(n, 1.0 / n)
    if n == 1:
        return eq
    O = np.stack(oofs, axis=1)
    best_w, best_key = eq, None
    for w in _simplex_grid(n, W_STEP):
        auc = _within_site_auc(y, O @ w, sites)
        key = (round(auc, 6), -np.abs(w - eq).sum())
        if best_key is None or key > best_key:
            best_w, best_key = w, key
    return W_SHRINK * eq + (1.0 - W_SHRINK) * best_w


def fit_predict(F_train, y_train, F_test, ctx):
    Ftr = np.asarray(F_train, dtype=np.float32)
    Fte = np.asarray(F_test, dtype=np.float32)
    if Ftr.ndim == 3:
        Ftr, Fte = Ftr[:, None], Fte[:, None]
    y = np.asarray(y_train).astype(int)
    ntr = Ftr.shape[0]
    sites = np.concatenate([_sites(ctx.get("meta_train"), ntr),
                            _sites(ctx.get("meta_test"), Fte.shape[0])])
    groups = ctx.get("groups_train")
    groups = np.asarray(groups) if groups is not None and len(groups) == ntr else None

    oofs, tests = [], []
    for b in range(Ftr.shape[1]):
        C = np.concatenate([Ftr[:, b], Fte[:, b]], axis=0).astype(np.float64)
        X = _tangent_features(C, sites)
        del C
        sc = StandardScaler().fit(X[:ntr])
        Xtr, Xte = sc.transform(X[:ntr]), sc.transform(X[ntr:])
        del X
        c_best, oof = _select_C(Xtr, y, sites[:ntr], groups)
        clf = LogisticRegression(C=c_best, max_iter=5000)
        clf.fit(Xtr, y)
        # Put blocks on a common scale using the spread of their inner OOF logits
        scale = oof.std() + 1e-12
        oofs.append(oof / scale)
        tests.append(clf.decision_function(Xte) / scale)

    w = _select_weights(oofs, y, sites[:ntr])
    score = np.stack(tests, axis=1) @ w
    return 1.0 / (1.0 + np.exp(-score))