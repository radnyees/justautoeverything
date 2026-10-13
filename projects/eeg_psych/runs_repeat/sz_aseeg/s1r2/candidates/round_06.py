"""Log relative band power (delta/theta/alpha/beta/gamma) per channel from Welch PSD of each
available condition + filter-bank Riemannian tangent-space features: trace-normalized,
average-referenced OAS spatial covariances for broadband 1-45 Hz and for delta/theta/alpha/beta
sub-bands (reference point fitted on the training fold only) + per-channel spectral shape:
aperiodic 1/f exponent (log-log fit 2-40 Hz excluding alpha) and individual alpha peak frequency
(centre of gravity of the periodic residual, 7-13 Hz) + per-channel complexity: Higuchi and Katz
fractal dimension of the 1-45 Hz average-referenced signal (median over 4-s epochs). Sub-band
blocks are down-weighted so that together they carry about one block's prior weight;
standardized, L2 logistic regression."""
import numpy as np
from scipy.signal import welch, butter, sosfiltfilt
from sklearn.covariance import OAS
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pyriemann.tangentspace import TangentSpace

BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
COV_BANDS = [(1, 45), (1, 4), (4, 8), (8, 13), (13, 30)]   # first = broadband
N_SHAPE = 2                                                 # aperiodic exponent, alpha peak freq
N_CPLX = 2                                                  # Higuchi FD, Katz FD
EPOCH_SEC = 4.0
KMAX = 10


def _bandpower(x, fs):
    f, p = welch(x, fs=fs, nperseg=int(2 * fs), axis=-1)
    tot = p[:, (f >= 1) & (f <= 45)].sum(axis=1, keepdims=True) + 1e-12
    return np.log(np.stack([p[:, (f >= a) & (f < b)].sum(axis=1) for a, b in BANDS], axis=1) / tot)


def _spectral_shape(x, fs):
    f, p = welch(x.astype(np.float64), fs=fs, nperseg=int(4 * fs), axis=-1)
    lp = np.log10(p + 1e-20)
    fit_m = (f >= 2) & (f <= 40) & ~((f >= 7) & (f <= 14))
    A = np.stack([np.ones(fit_m.sum()), np.log10(f[fit_m])], axis=1)
    coef, *_ = np.linalg.lstsq(A, lp[:, fit_m].T, rcond=None)   # (2, n_ch)
    expo = -coef[1]
    am = (f >= 7) & (f <= 13)
    resid = lp[:, am] - (coef[0][:, None] + coef[1][:, None] * np.log10(f[am])[None, :])
    w = np.clip(resid, 0, None)
    ws = w.sum(axis=1)
    paf = np.where(ws > 0, (w * f[am][None, :]).sum(axis=1) / np.maximum(ws, 1e-12), 10.0)
    return np.stack([expo, paf], axis=1)


def _higuchi(X, kmax=KMAX):
    """Higuchi fractal dimension along the last axis, vectorized over leading axes."""
    N = X.shape[-1]
    logL, logk = [], []
    for k in range(1, kmax + 1):
        Lm = []
        for m in range(k):
            idx = np.arange(m, N, k)
            n_max = len(idx) - 1
            if n_max < 1:
                continue
            d = np.abs(np.diff(X[..., idx], axis=-1)).sum(axis=-1)
            Lm.append(d * (N - 1) / (n_max * k) / k)
        logL.append(np.log(np.mean(Lm, axis=0) + 1e-20))
        logk.append(np.log(1.0 / k))
    logL = np.stack(logL, axis=-1)                       # (..., kmax)
    logk = np.asarray(logk)
    lk = logk - logk.mean()
    return ((logL - logL.mean(axis=-1, keepdims=True)) * lk).sum(axis=-1) / (lk ** 2).sum()


def _katz(X):
    dif = np.abs(np.diff(X, axis=-1))
    L = dif.sum(axis=-1) + 1e-20
    n = X.shape[-1] - 1
    d = np.abs(X - X[..., :1]).max(axis=-1) + 1e-20
    return np.log10(n) / (np.log10(n) + np.log10(d / L))


def _complexity(x, fs):
    x = x.astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)                       # average re-reference
    sos = butter(4, [1, min(45, 0.45 * fs)], btype="band", fs=fs, output="sos")
    xf = sosfiltfilt(sos, x, axis=-1)
    L = int(EPOCH_SEC * fs)
    n_ep = max(1, xf.shape[1] // L)
    if xf.shape[1] < L:
        E = xf[None]
    else:
        E = xf[:, :n_ep * L].reshape(xf.shape[0], n_ep, L).transpose(1, 0, 2)  # (ep, ch, L)
    hfd = np.median(_higuchi(E), axis=0)
    kfd = np.median(_katz(E), axis=0)
    return np.stack([hfd, kfd], axis=1)


def _spatial_covs(x, fs):
    x = x.astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)                       # average re-reference
    out = []
    for lo, hi in COV_BANDS:
        sos = butter(4, [lo, min(hi, 0.45 * fs)], btype="band", fs=fs, output="sos")
        xf = sosfiltfilt(sos, x, axis=-1)
        C = OAS(assume_centered=False).fit(xf.T).covariance_   # shrinkage -> full rank after avg ref
        out.append((C / np.trace(C)).ravel())                   # amplitude-scale invariant
    return np.concatenate(out)


def featurize(items, ctx):
    fs = ctx["fs"]
    bp_feats, cov_feats, sh_feats, cx_feats = [], [], [], []
    blocks = {c: ctx["load_eeg"](items, c) for c in ctx["conditions"]}
    for i in range(len(items)):
        fb, fc, fsh, fcx = [], [], [], []
        for c in ctx["conditions"]:
            x = blocks[c][i]
            if x is None:                      # missing condition -> fall back to eyes closed
                x = blocks[ctx["conditions"][0]][i]
            fb.append(_bandpower(x, fs).ravel())
            fc.append(_spatial_covs(x, fs))
            fsh.append(_spectral_shape(x, fs).ravel())
            fcx.append(_complexity(x, fs).ravel())
        bp_feats.append(np.concatenate(fb))
        cov_feats.append(np.concatenate(fc))
        sh_feats.append(np.concatenate(fsh))
        cx_feats.append(np.concatenate(fcx))
    return np.hstack([np.asarray(bp_feats), np.asarray(cov_feats),
                      np.asarray(sh_feats), np.asarray(cx_feats)]).astype(np.float64)


def _split(F, ctx):
    n_ch = len(ctx["channels"])
    n_cond = len(ctx["conditions"])
    nb = n_cond * n_ch * len(BANDS)
    nc = n_cond * len(COV_BANDS) * n_ch * n_ch
    ns = n_cond * n_ch * N_SHAPE
    bp = F[:, :nb]
    covs = F[:, nb:nb + nc].reshape(len(F), n_cond, len(COV_BANDS), n_ch, n_ch)
    shape = F[:, nb + nc:nb + nc + ns]
    cplx = F[:, nb + nc + ns:]
    return bp, covs, shape, cplx


def fit_predict(F_train, y_train, F_test, ctx):
    F_train = np.asarray(F_train, dtype=np.float64)
    F_test = np.asarray(F_test, dtype=np.float64)
    bp_tr, cov_tr, sh_tr, cx_tr = _split(F_train, ctx)
    bp_te, cov_te, sh_te, cx_te = _split(F_test, ctx)
    sc_bp = StandardScaler().fit(bp_tr)
    sc_sh = StandardScaler().fit(sh_tr)
    sc_cx = StandardScaler().fit(cx_tr)
    X_tr = [sc_bp.transform(bp_tr), sc_sh.transform(sh_tr), sc_cx.transform(cx_tr)]
    X_te = [sc_bp.transform(bp_te), sc_sh.transform(sh_te), sc_cx.transform(cx_te)]
    n_sub = len(COV_BANDS) - 1
    for k in range(cov_tr.shape[1]):
        for b in range(cov_tr.shape[2]):
            Ctr = (cov_tr[:, k, b] + cov_tr[:, k, b].transpose(0, 2, 1)) / 2
            Cte = (cov_te[:, k, b] + cov_te[:, k, b].transpose(0, 2, 1)) / 2
            ts = TangentSpace(metric="riemann").fit(Ctr)
            t_tr, t_te = ts.transform(Ctr), ts.transform(Cte)
            sc = StandardScaler().fit(t_tr)
            w = 1.0 if b == 0 else 1.0 / np.sqrt(n_sub)
            X_tr.append(w * sc.transform(t_tr))
            X_te.append(w * sc.transform(t_te))
    X_tr = np.hstack(X_tr)
    X_te = np.hstack(X_te)
    clf = LogisticRegression(C=0.1, max_iter=5000).fit(X_tr, y_train)
    return clf.predict_proba(X_te)[:, 1]