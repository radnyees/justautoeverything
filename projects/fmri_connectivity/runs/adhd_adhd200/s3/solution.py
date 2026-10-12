"""Ledoit-Wolf covariance on CC400 -> per-site Riemannian re-centering at a class-balanced
log-Euclidean mean (average of class means, training labels only) -> tangent space,
per-site standardization with class-balanced mean / pooled within-class SD,
L2 logistic regression with per-site class-balanced sample weights (C scaled to feature count),
then low-rank spectral denoising of the ROI x ROI weight matrix (eigencomponents above the
random-matrix noise edge, with optimal eigenvalue shrinkage)."""
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression


def _lw_cov(x):
    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = x - x.mean(0, keepdims=True)
    sd = x.std(0)
    good = sd > 1e-8
    x[:, good] = x[:, good] / sd[good]
    x[:, ~good] = 0.0
    c = LedoitWolf(assume_centered=True).fit(x).covariance_
    c = 0.5 * (c + c.T)
    # guarantee strict positive definiteness (all-zero ROIs, very short scans)
    w = np.linalg.eigvalsh(c)
    if w[0] < 1e-6:
        c = c + (1e-6 - w[0] + 1e-6) * np.eye(c.shape[0])
    return c


def _sym_fun(c, fun):
    w, v = np.linalg.eigh(c)
    w = np.maximum(w, 1e-12)
    return (v * fun(w)) @ v.T


def _pick_atlas(ctx):
    atlases = list(ctx.get("atlases", []))
    for a in ("cc400", "cc200"):
        if a in atlases:
            return a
    return atlases[0] if atlases else "cc400"


def featurize(items, ctx):
    strategy = "filt_noglobal" if "filt_noglobal" in ctx["strategies"] else ctx["strategies"][0]
    ts = ctx["load_timeseries"](items, _pick_atlas(ctx), strategy)
    n = ts[0].shape[1]
    iu = np.triu_indices(n, k=0)
    feats = np.empty((len(ts), len(iu[0])), dtype=np.float32)
    for i, x in enumerate(ts):
        feats[i] = _lw_cov(x)[iu]
    return feats


class _Unpacker:
    def __init__(self, p):
        n = int(round((np.sqrt(8 * p + 1) - 1) / 2))
        self.n = n
        self.iu = np.triu_indices(n, k=0)

    def __call__(self, row):
        c = np.zeros((self.n, self.n))
        c[self.iu] = np.asarray(row, dtype=np.float64)
        c = c + c.T
        c[np.arange(self.n), np.arange(self.n)] *= 0.5
        return c


def _sites(meta, n):
    if meta is None:
        return np.zeros(n, dtype=object)
    try:
        import pandas as pd
        if isinstance(meta, pd.DataFrame):
            return meta["site"].astype(str).to_numpy()
        if isinstance(meta, pd.Series):
            return meta.astype(str).to_numpy()
    except Exception:
        pass
    if isinstance(meta, dict) and "site" in meta:
        return np.asarray([str(s) for s in meta["site"]], dtype=object)
    out = []
    for m in meta:
        out.append(str(m["site"]) if isinstance(m, dict) else str(m))
    return np.asarray(out, dtype=object)


def _balanced(mtr, y):
    """Return per-class index arrays if the site's training items contain both classes
    with at least 2 subjects each, else None."""
    if len(mtr) == 0:
        return None
    ys = y[mtr]
    classes = np.unique(ys)
    if len(classes) < 2:
        return None
    groups = [mtr[ys == c] for c in classes]
    if min(len(g) for g in groups) < 2:
        return None
    return groups


def _site_tangent(F_tr, F_te, s_train, s_test, y_train):
    """Re-center each site at a class-balanced log-Euclidean mean, then log-map."""
    unpack = _Unpacker(F_tr.shape[1])
    iu = unpack.iu
    coef = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    Ttr = np.empty((len(F_tr), len(iu[0])))
    Tte = np.empty((len(F_te), len(iu[0])))
    for s in np.unique(np.concatenate([s_train, s_test])):
        mtr = np.where(s_train == s)[0]
        mte = np.where(s_test == s)[0]
        groups = _balanced(mtr, y_train)
        if groups is not None:
            logs = np.mean([np.mean([_sym_fun(unpack(F_tr[i]), np.log) for i in g], axis=0)
                            for g in groups], axis=0)
        else:
            if len(mtr) >= 2:
                ref_rows = [F_tr[i] for i in mtr]
            else:
                ref_rows = [F_te[i] for i in mte]
            if len(ref_rows) < 1:
                continue
            logs = np.mean([_sym_fun(unpack(r), np.log) for r in ref_rows], axis=0)
        ref = _sym_fun(0.5 * (logs + logs.T), np.exp)
        isqrt = _sym_fun(ref, lambda w: 1.0 / np.sqrt(w))
        for idx, F, T in ((mtr, F_tr, Ttr), (mte, F_te, Tte)):
            for i in idx:
                m = isqrt @ unpack(F[i]) @ isqrt
                m = 0.5 * (m + m.T)
                T[i] = coef * _sym_fun(m, np.log)[iu]
    return Ttr, Tte


def _site_standardize(Xtr, Xte, s_train, s_test, y_train):
    """In-place z-scoring of features within each site using the class-balanced mean and
    pooled within-class SD of training items; falls back to plain (label-free) statistics
    for single-class sites, and to the test items' own statistics for unseen sites."""
    for s in np.unique(np.concatenate([s_train, s_test])):
        mtr = s_train == s
        mte = s_test == s
        groups = _balanced(np.where(mtr)[0], y_train)
        if groups is not None:
            mus = [Xtr[g].mean(0) for g in groups]
            mu = np.mean(mus, axis=0)
            ss = sum(((Xtr[g] - m) ** 2).sum(0) for g, m in zip(groups, mus))
            dof = sum(len(g) for g in groups) - len(groups)
            sd = np.sqrt(ss / max(dof, 1))
        else:
            ref = Xtr[mtr] if mtr.sum() >= 2 else Xte[mte]
            if ref.shape[0] < 2:
                ref = np.vstack([Xtr[mtr], Xte[mte]])
            mu = ref.mean(0)
            sd = ref.std(0) if ref.shape[0] >= 2 else np.ones_like(mu)
        sd = np.where(sd < 1e-6, 1.0, sd)
        if mtr.any():
            Xtr[mtr] = (Xtr[mtr] - mu) / sd
        if mte.any():
            Xte[mte] = (Xte[mte] - mu) / sd
    return Xtr, Xte


def _site_class_weights(s_train, y):
    """Within each site, give both classes equal total weight (n_site / 2 each) so the
    loss reflects within-site discrimination; single-class sites keep unit weights.
    Weights are normalized to mean 1 so the effective regularization is unchanged."""
    w = np.ones(len(y), dtype=np.float64)
    for s in np.unique(s_train):
        idx = np.where(s_train == s)[0]
        groups = _balanced(idx, y)
        if groups is None:
            continue
        for g in groups:
            w[g] = len(idx) / (len(groups) * len(g))
    return w / w.mean()


def _lowrank_weights(w):
    """Fold the upper-triangular weight vector into a symmetric ROI x ROI matrix, keep the
    eigencomponents whose magnitude exceeds the Wigner noise edge 2*sigma*sqrt(n) (sigma from
    a robust MAD of off-diagonal entries), shrink them to their de-biased signal strength,
    and unfold. Falls back to the original weights if no component clears the edge."""
    unpack = _Unpacker(len(w))
    n, iu = unpack.n, unpack.iu
    M = np.zeros((n, n))
    M[iu] = w
    M = 0.5 * (M + M.T)  # off-diagonal = w/2 on each side, diagonal = w
    off = M[np.triu_indices(n, k=1)]
    sigma = 1.4826 * np.median(np.abs(off - np.median(off)))
    if not np.isfinite(sigma) or sigma <= 0:
        return w
    edge = 2.0 * sigma * np.sqrt(n)
    lam, V = np.linalg.eigh(M)
    keep = np.abs(lam) > edge
    if not keep.any():
        return w
    theta = np.sign(lam[keep]) * 0.5 * (np.abs(lam[keep]) + np.sqrt(lam[keep] ** 2 - edge ** 2))
    Mk = (V[:, keep] * theta) @ V[:, keep].T
    out = Mk[iu].copy()
    out[iu[0] != iu[1]] *= 2.0
    return out


def fit_predict(F_train, y_train, F_test, ctx):
    F_train = np.asarray(F_train)
    F_test = np.asarray(F_test)
    n_tr, n_te = len(F_train), len(F_test)
    y = np.asarray(y_train)
    s_train = _sites(ctx.get("meta_train"), n_tr)
    s_test = _sites(ctx.get("meta_test"), n_te)
    Xtr, Xte = _site_tangent(F_train, F_test, s_train, s_test, y)
    Xtr, Xte = _site_standardize(Xtr, Xte, s_train, s_test, y)
    # global standardization (in place)
    mu = Xtr.mean(0)
    sd = Xtr.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Xtr -= mu
    Xtr /= sd
    Xte -= mu
    Xte /= sd
    sw = _site_class_weights(s_train, y)
    # keep the effective L2 strength of the CC200 model (C=0.001 at 20100 features)
    C = 0.001 * 20100.0 / Xtr.shape[1]
    clf = LogisticRegression(C=C, max_iter=5000)
    clf.fit(Xtr, y_train, sample_weight=sw)
    w = _lowrank_weights(clf.coef_.ravel().astype(np.float64))
    return Xte @ w + float(clf.intercept_[0])