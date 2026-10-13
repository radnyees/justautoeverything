"""Tangent-space embedding of Ledoit-Wolf correlation matrices on CC200 (no GSR), L2 logistic regression
(features centred and globally scaled to preserve the tangent-space metric) trained with within-site
class-balancing sample weights (de-confounds site base rates from the connectivity model), stacked with
demographic/motion/site covariates (age, sex, log mean FD, site one-hot) in a small second-stage logistic model."""
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from pyriemann.tangentspace import TangentSpace


def _shrunk_corr(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0  # all-zero ROIs (outside FOV) stay zero
    x = x / sd
    c = LedoitWolf(assume_centered=True).fit(x).covariance_
    c = 0.5 * (c + c.T)
    # guarantee positive definiteness (zero ROIs / very short scans)
    w, v = np.linalg.eigh(c)
    w = np.maximum(w, 1e-6 * max(w.max(), 1e-6))
    c = (v * w) @ v.T
    return 0.5 * (c + c.T)


def featurize(items, ctx):
    strategy = "filt_noglobal" if "filt_noglobal" in ctx["strategies"] else ctx["strategies"][0]
    ts = ctx["load_timeseries"](items, "cc200", strategy)
    covs = [_shrunk_corr(x) for x in ts]
    return np.asarray(covs, dtype=np.float32)


def _col(meta, name, n):
    if meta is None or name not in getattr(meta, "columns", []):
        return np.full(n, np.nan)
    s = pd.Series(np.asarray(meta[name]))
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().mean() < 0.5:  # categorical coding (e.g. 'M'/'F')
        st = s.astype(str).str.upper().str[0]
        num = st.map({"M": 1.0, "F": 0.0}).astype(float)
    return num.to_numpy(dtype=np.float64)


def _covariates(meta, n):
    age = _col(meta, "age", n)
    sex = _col(meta, "sex", n)
    fd = _col(meta, "mean_fd", n)
    fd = np.log(np.clip(fd, 1e-3, None))
    return np.column_stack([age, sex, fd])


def _sites(meta, n):
    if meta is None or "site" not in getattr(meta, "columns", []):
        return np.array(["_"] * n, dtype=object)
    return np.asarray(pd.Series(np.asarray(meta["site"])).astype(str), dtype=object)


def _site_onehot(site_tr, site_te, min_count=5):
    vals, counts = np.unique(site_tr, return_counts=True)
    levels = [v for v, c in zip(vals, counts) if c >= min_count]
    if len(levels) < 2:
        return np.zeros((len(site_tr), 0)), np.zeros((len(site_te), 0))
    Otr = np.column_stack([(site_tr == v).astype(np.float64) for v in levels])
    Ote = np.column_stack([(site_te == v).astype(np.float64) for v in levels])  # unseen site -> all zeros
    return Otr, Ote


def _site_balance_weights(y, sites):
    """Weights making each class carry equal total weight within every site (site-label de-confounding)."""
    y = np.asarray(y)
    w = np.ones(len(y), dtype=np.float64)
    for s in np.unique(sites):
        m = sites == s
        classes, counts = np.unique(y[m], return_counts=True)
        for c, k in zip(classes, counts):
            w[m & (y == c)] = m.sum() / (len(classes) * k)
    return w / w.mean()


class _GlobalScaler:
    """Centre features; divide by a single global RMS std (keeps relative tangent-space scales)."""

    def fit(self, X):
        self.mu_ = X.mean(axis=0)
        self.s_ = float(np.sqrt(np.mean(X.var(axis=0))))
        if not np.isfinite(self.s_) or self.s_ < 1e-12:
            self.s_ = 1.0
        return self

    def transform(self, X):
        return (X - self.mu_) / self.s_


def _stage1(Xtr, ytr, wtr):
    sc = _GlobalScaler().fit(Xtr)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(Xtr), ytr, sample_weight=wtr)
    return sc, clf


def fit_predict(F_train, y_train, F_test, ctx):
    y_train = np.asarray(y_train)
    Ctr = np.asarray(F_train, dtype=np.float64)
    Cte = np.asarray(F_test, dtype=np.float64)
    ts = TangentSpace(metric="riemann").fit(Ctr)  # reference = training-fold geometric mean
    Xtr = ts.transform(Ctr)
    Xte = ts.transform(Cte)

    meta_tr, meta_te = ctx.get("meta_train"), ctx.get("meta_test")
    site_tr = _sites(meta_tr, len(y_train))
    site_te = _sites(meta_te, len(Cte))

    # stage 1: connectivity model on full training fold -> test score
    w_all = _site_balance_weights(y_train, site_tr)
    sc, clf = _stage1(Xtr, y_train, w_all)
    s_te = clf.decision_function(sc.transform(Xte))

    # out-of-fold stage-1 scores on the training fold (grouped inner CV)
    groups = ctx.get("groups_train")
    if groups is None:
        groups = np.arange(len(y_train))
    groups = np.asarray(groups)
    n_splits = int(min(5, len(np.unique(groups))))
    s_tr = np.zeros(len(y_train))
    for tr_i, va_i in GroupKFold(n_splits=n_splits).split(Xtr, y_train, groups):
        w_i = _site_balance_weights(y_train[tr_i], site_tr[tr_i])
        sc_i, clf_i = _stage1(Xtr[tr_i], y_train[tr_i], w_i)
        s_tr[va_i] = clf_i.decision_function(sc_i.transform(Xtr[va_i]))

    # stage 2: OOF connectivity score + covariates + site indicators
    Ztr = _covariates(meta_tr, len(y_train))
    Zte = _covariates(meta_te, len(Cte))
    med = np.nanmedian(Ztr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    for j in range(Ztr.shape[1]):
        Ztr[np.isnan(Ztr[:, j]), j] = med[j]
        Zte[np.isnan(Zte[:, j]), j] = med[j]
    keep = Ztr.std(axis=0) > 1e-8
    Ztr, Zte = Ztr[:, keep], Zte[:, keep]

    Otr, Ote = _site_onehot(site_tr, site_te)

    Str = np.column_stack([s_tr, Ztr])
    Ste = np.column_stack([s_te, Zte])
    sc2 = StandardScaler().fit(Str)
    Str = np.column_stack([sc2.transform(Str), Otr])
    Ste = np.column_stack([sc2.transform(Ste), Ote])
    meta_clf = LogisticRegression(C=1.0, max_iter=5000)
    meta_clf.fit(Str, y_train)
    return meta_clf.predict_proba(Ste)[:, 1]