"""Ledoit-Wolf covariance on standardized CC200 time series -> Riemannian tangent-space embedding, L2 logistic
regression; blended with a nodal logistic model on multi-scale relative-ALFF (subject-normalized log ROI amplitude,
CC200 + CC400) plus a regional-homogeneity proxy (CC400 sub-parcel coherence with its parent CC200 parcel).
Regularization strengths and blend weight are chosen by grouped inner CV on within-site AUROC."""
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from pyriemann.tangentspace import TangentSpace

W_AMP = 0.3
C_CONN = 0.001
C_AMP = 0.01
C_CONN_GRID = [0.001, 0.0003, 0.003]   # default first: wins ties
C_AMP_GRID = [0.01, 0.003, 0.03]
W_GRID = [0.3, 0.2, 0.4, 0.5]


def _rel_log_amp(x):
    x = np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0)
    x = x - x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0)
    valid = sd > 1e-8
    # Relative log-amplitude (ALFF proxy): removes subject/scanner global scaling
    lsd = np.zeros_like(sd)
    if valid.any():
        lsd[valid] = np.log(sd[valid])
        lsd[valid] -= np.median(lsd[valid])
    lsd = np.clip(lsd, -3.0, 3.0)
    lsd[~valid] = 0.0
    return x, sd, valid, lsd


def _standardize(x, sd, valid):
    sdn = sd.copy()
    sdn[~valid] = 1.0
    return x / sdn[None, :]


def featurize(items, ctx):
    strategy = "filt_noglobal" if "filt_noglobal" in ctx["strategies"] else ctx["strategies"][0]
    ts = ctx["load_timeseries"](items, "cc200", strategy)
    has400 = "cc400" in ctx["atlases"]
    ts4 = ctx["load_timeseries"](items, "cc400", strategy) if has400 else None
    covs, amps, amps4, Rs, v2s, v4s = [], [], [], [], [], []
    Rsum = None
    for i, x in enumerate(ts):
        x, sd, valid, lsd = _rel_log_amp(x)
        amps.append(lsd)
        x = _standardize(x, sd, valid)
        c = LedoitWolf(assume_centered=True).fit(x).covariance_
        c = 0.5 * (c + c.T)
        c[np.diag_indices_from(c)] += 1e-6
        covs.append(c)
        if has400:
            x4, sd4, valid4, lsd4 = _rel_log_amp(ts4[i])
            amps4.append(lsd4)
            x4 = _standardize(x4, sd4, valid4)
            T = min(len(x), len(x4))
            # Correlation of each CC400 sub-parcel with every CC200 parcel (for local coherence)
            R = (x[:T].T @ x4[:T]) / max(T, 1)
            Rs.append(R.astype(np.float32))
            v2s.append(valid)
            v4s.append(valid4)
            Rsum = R.copy() if Rsum is None else Rsum + R
    del ts, ts4
    covs = np.asarray(covs)
    # Label-free reference point: Riemannian mean of all subjects' covariances
    ts_map = TangentSpace(metric="riemann").fit(covs)
    T_feat = np.asarray(ts_map.transform(covs), dtype=np.float32)

    blocks = [np.asarray(amps, dtype=np.float32)]
    if has400:
        # Finer-scale relative ALFF (CC400) for multi-scale localization of amplitude effects
        blocks.append(np.asarray(amps4, dtype=np.float32))
        # Regional homogeneity proxy: each CC400 sub-parcel's coherence with its (label-free) parent CC200 parcel
        parent = np.argmax(Rsum, axis=0)
        cols = np.arange(Rsum.shape[1])
        L = np.zeros((len(items), Rsum.shape[1]), dtype=np.float32)
        for i, R in enumerate(Rs):
            r = np.clip(R[parent, cols].astype(np.float64), -0.99, 0.99)
            z = np.arctanh(r)
            ok = v2s[i][parent] & v4s[i]
            if ok.any():
                z[ok] -= np.median(z[ok])
            z = np.clip(z, -3.0, 3.0)
            z[~ok] = 0.0
            L[i] = z
        blocks.append(L)
        del Rs
    A = np.hstack(blocks)
    n_amp = np.full((len(items), 1), A.shape[1], dtype=np.float32)
    return np.hstack([n_amp, T_feat, A])


def _zscored_logit(clf, Xtr, Xte):
    dtr = clf.decision_function(Xtr)
    dte = clf.decision_function(Xte)
    s = dtr.std()
    return (dte - dtr.mean()) / (s if s > 1e-12 else 1.0)


def _fit_z(Xtr, ytr, Xte, C):
    sc = StandardScaler().fit(Xtr)
    Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)
    clf = LogisticRegression(C=C, max_iter=5000)
    clf.fit(Ztr, ytr)
    return _zscored_logit(clf, Ztr, Zte)


def _get_sites(ctx, n):
    try:
        meta = ctx.get("meta_train")
        if meta is None:
            return None
        if hasattr(meta, "columns") and "site" in meta.columns:
            s = np.asarray(meta["site"]).astype(str)
        elif isinstance(meta, dict) and "site" in meta:
            s = np.asarray(meta["site"]).astype(str)
        else:
            s = np.asarray([m["site"] for m in meta]).astype(str)
        return s if len(s) == n else None
    except Exception:
        return None


def _site_auc(y, s, sites):
    if sites is None:
        return roc_auc_score(y, s) if len(np.unique(y)) > 1 else 0.5
    acc, tot = 0.0, 0
    for st in np.unique(sites):
        m = sites == st
        if len(np.unique(y[m])) < 2:
            continue
        acc += roc_auc_score(y[m], s[m]) * m.sum()
        tot += m.sum()
    if tot == 0:
        return roc_auc_score(y, s) if len(np.unique(y)) > 1 else 0.5
    return acc / tot


def _tune(T, A, y, ctx):
    n = len(y)
    groups = ctx.get("groups_train")
    groups = np.asarray(groups) if groups is not None and len(groups) == n else np.arange(n)
    if len(np.unique(groups)) < 5 or len(np.unique(y)) < 2:
        return C_CONN, C_AMP, W_AMP
    sites = _get_sites(ctx, n)
    oof_c = {C: np.zeros(n) for C in C_CONN_GRID}
    oof_a = {C: np.zeros(n) for C in C_AMP_GRID}
    for tr, te in GroupKFold(n_splits=5).split(T, y, groups):
        if len(np.unique(y[tr])) < 2:
            return C_CONN, C_AMP, W_AMP
        sc = StandardScaler().fit(T[tr])
        Ttr, Tte = sc.transform(T[tr]), sc.transform(T[te])
        for C in C_CONN_GRID:
            clf = LogisticRegression(C=C, max_iter=5000).fit(Ttr, y[tr])
            oof_c[C][te] = _zscored_logit(clf, Ttr, Tte)
        sca = StandardScaler().fit(A[tr])
        Atr, Ate = sca.transform(A[tr]), sca.transform(A[te])
        for C in C_AMP_GRID:
            clf = LogisticRegression(C=C, max_iter=5000).fit(Atr, y[tr])
            oof_a[C][te] = _zscored_logit(clf, Atr, Ate)
    best, best_score = (C_CONN, C_AMP, W_AMP), -np.inf
    for cc in C_CONN_GRID:
        for ca in C_AMP_GRID:
            for w in W_GRID:
                sc_ = _site_auc(y, (1.0 - w) * oof_c[cc] + w * oof_a[ca], sites)
                if sc_ > best_score + 1e-9:
                    best_score, best = sc_, (cc, ca, w)
    return best


def fit_predict(F_train, y_train, F_test, ctx):
    F_train = np.asarray(F_train, dtype=np.float64)
    F_test = np.asarray(F_test, dtype=np.float64)
    y_train = np.asarray(y_train).astype(int)
    n_amp = int(round(F_train[0, 0]))
    T_tr, A_tr = F_train[:, 1:-n_amp], F_train[:, -n_amp:]
    T_te, A_te = F_test[:, 1:-n_amp], F_test[:, -n_amp:]

    c_conn, c_amp, w_amp = _tune(T_tr, A_tr, y_train, ctx)

    z_conn = _fit_z(T_tr, y_train, T_te, c_conn)
    z_amp = _fit_z(A_tr, y_train, A_te, c_amp)

    z = (1.0 - w_amp) * z_conn + w_amp * z_amp
    return 1.0 / (1.0 + np.exp(-z))