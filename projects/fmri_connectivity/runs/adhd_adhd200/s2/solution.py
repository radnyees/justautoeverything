"""CC200 (athena) time series -> in-house global signal regression (mean of in-FOV ROIs) ->
Ledoit-Wolf correlation matrices -> conditional (Gaussian) imputation of out-of-FOV ROIs from
the training-fold population correlation -> tangent-space embedding (Riemannian reference fitted
on training fold only) -> isotropic (global) scaling that preserves the tangent-space metric ->
exact projection onto the training span (representer theorem) -> multi-task feature augmentation
[shared, alpha * site-specific copy] -> L2 logistic regression (shared + site-specific weights)."""
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from pyriemann.tangentspace import TangentSpace

SITE_ALPHA = 0.7  # site-specific copy scale: same-site kernel = (1 + alpha^2) x shared kernel


def _gsr(x):
    # regress the global signal (mean over in-FOV ROIs) out of every in-FOV ROI
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(0)
    good = x.std(0) > 1e-8
    if good.sum() < 2:
        return x
    g = x[:, good].mean(1)
    gg = float(g @ g)
    if gg < 1e-12:
        return x
    beta = (g @ x[:, good]) / gg
    x[:, good] = x[:, good] - np.outer(g, beta)
    return x


def _lw_corr(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(0)
    sd = x.std(0)
    good = sd > 1e-8
    z = np.zeros_like(x)
    z[:, good] = x[:, good] / sd[good]
    c = LedoitWolf(store_precision=False, assume_centered=True).fit(z).covariance_
    # dead ROIs (outside FOV): set to identity rows so the matrix stays SPD and uninformative
    d = np.sqrt(np.clip(np.diag(c), 1e-12, None))
    c = c / np.outer(d, d)
    c[~good, :] = 0.0
    c[:, ~good] = 0.0
    c[~good, ~good] = 1.0
    c = 0.5 * (c + c.T) + 1e-6 * np.eye(c.shape[0])
    return c


def featurize(items, ctx):
    strategy = "filt_noglobal" if "filt_noglobal" in ctx["strategies"] else ctx["strategies"][0]
    ts = ctx["load_timeseries"](items, "cc200", strategy)
    covs = [_lw_corr(_gsr(x)) for x in ts]
    return np.asarray(covs, dtype=np.float32)


def _dead_mask(C):
    # a dead ROI has all off-diagonal entries exactly zero
    off = np.abs(C).sum(2) - np.abs(np.diagonal(C, axis1=1, axis2=2))
    return off < 1e-6


def _population_corr(C, dead):
    # pairwise-available mean correlation over training subjects, projected to SPD correlation
    n = C.shape[1]
    S = np.zeros((n, n))
    N = np.zeros((n, n))
    for c, d in zip(C, dead):
        g = (~d).astype(np.float64)
        w = np.outer(g, g)
        S += c * w
        N += w
    M = np.where(N > 0, S / np.maximum(N, 1.0), 0.0)
    np.fill_diagonal(M, 1.0)
    M = 0.5 * (M + M.T)
    w, V = np.linalg.eigh(M)
    w = np.maximum(w, 1e-3)
    M = (V * w) @ V.T
    d = np.sqrt(np.diag(M))
    M = M / np.outer(d, d)
    return 0.5 * (M + M.T)


def _impute(C, dead, M):
    # conditional Gaussian imputation: x_D = A x_G + e, A = M_DG M_GG^-1, e ~ N(0, M_D|G)
    C = C.copy()
    for i in range(len(C)):
        D = dead[i]
        if not D.any():
            continue
        G = ~D
        if G.sum() < 2:
            continue
        Mgg = M[np.ix_(G, G)]
        Mdg = M[np.ix_(D, G)]
        Mdd = M[np.ix_(D, D)]
        A = np.linalg.solve(Mgg, Mdg.T).T
        Sc = Mdd - A @ Mdg.T
        Cgg = C[i][np.ix_(G, G)]
        Cdg = A @ Cgg
        Cdd = Sc + A @ Cgg @ A.T
        c = C[i]
        c[np.ix_(D, G)] = Cdg
        c[np.ix_(G, D)] = Cdg.T
        c[np.ix_(D, D)] = Cdd
        d = np.sqrt(np.clip(np.diag(c), 1e-12, None))
        c = c / np.outer(d, d)
        C[i] = 0.5 * (c + c.T) + 1e-6 * np.eye(c.shape[0])
    return C


def _sites(meta, n):
    if meta is None:
        return np.zeros(n, dtype=object)
    try:
        s = meta["site"]
        s = s.values if hasattr(s, "values") else s
        s = np.asarray(s).astype(str)
        if len(s) == n:
            return s
    except Exception:
        pass
    try:
        s = np.asarray([m["site"] for m in meta]).astype(str)
        if len(s) == n:
            return s
    except Exception:
        pass
    return np.zeros(n, dtype=object)


def _augment(Z, sites, levels, alpha):
    # [shared | alpha * Z restricted to own site] -> w_eff = w_shared + alpha * w_site
    blocks = [Z]
    for lv in levels:
        m = (sites == lv).astype(np.float64)[:, None]
        blocks.append(alpha * Z * m)
    return np.hstack(blocks)


def fit_predict(F_train, y_train, F_test, ctx):
    Ctr = np.asarray(F_train, dtype=np.float64)
    Cte = np.asarray(F_test, dtype=np.float64)
    dtr = _dead_mask(Ctr)
    dte = _dead_mask(Cte)
    M = _population_corr(Ctr, dtr)
    Ctr = _impute(Ctr, dtr, M)
    Cte = _impute(Cte, dte, M)
    ts = TangentSpace(metric="riemann").fit(Ctr)
    Xtr = ts.transform(Ctr)
    Xte = ts.transform(Cte)
    # isotropic scaling: centre on training mean, divide by one global scale (RMS of feature SDs)
    # so relative tangent-space geometry is preserved and C keeps the same effective meaning
    mu = Xtr.mean(0)
    scale = np.sqrt(np.mean(Xtr.var(0)))
    scale = scale if scale > 1e-12 else 1.0
    Xtr = (Xtr - mu) / scale
    Xte = (Xte - mu) / scale
    # exact projection onto the span of training rows: L2-LR solution lies in this span, so
    # inner products (and hence the fit / predictions) are unchanged, with n instead of p dims
    _, s, Vt = np.linalg.svd(Xtr, full_matrices=False)
    keep = s > 1e-8 * s.max()
    V = Vt[keep].T
    Ztr = Xtr @ V
    Zte = Xte @ V
    # multi-task (shared + site-specific) weights via feature augmentation
    str_ = _sites(ctx.get("meta_train"), len(Ztr))
    ste = _sites(ctx.get("meta_test"), len(Zte))
    levels = sorted(set(str_.tolist()))
    Atr = _augment(Ztr, str_, levels, SITE_ALPHA)
    Ate = _augment(Zte, ste, levels, SITE_ALPHA)  # unseen test sites fall back to shared weights
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(Atr, y_train)
    return clf.predict_proba(Ate)[:, 1]