"""Log relative band power (delta/theta/alpha/beta/gamma) per channel from Welch PSD of each
available condition + per-channel aperiodic (1/f) exponent (log-log linear fit, alpha range
excluded) + per-channel individual alpha frequency (centre of gravity of the 7-13 Hz PSD)
+ per-channel Higuchi fractal dimension and Hjorth mobility/complexity of the 1-45 Hz signal
+ trace-normalized filter-bank spatial covariances (broadband 1-45 Hz, theta, alpha, beta) per
condition, each projected to its own Riemannian tangent space (reference = training-fold mean),
standardized, L2 logistic regression."""
import numpy as np
from scipy.signal import welch, butter, sosfiltfilt
from sklearn.covariance import OAS
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pyriemann.tangentspace import TangentSpace
from antropy import higuchi_fd

BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
COV_BANDS = [(1, 45), (4, 8), (8, 13), (13, 30)]


def _spectral(x, fs):
    f, p = welch(x, fs=fs, nperseg=int(2 * fs), axis=-1)
    tot = p[:, (f >= 1) & (f <= 45)].sum(axis=1, keepdims=True) + 1e-12
    bp = np.log(np.stack([p[:, (f >= a) & (f < b)].sum(axis=1) for a, b in BANDS], axis=1) / tot)
    # aperiodic exponent: log-log linear fit outside the alpha peak and below line noise
    m = ((f >= 2) & (f < 7)) | ((f > 14) & (f <= 40))
    lf = np.log10(f[m])
    lp = np.log10(p[:, m] + 1e-12)
    A = np.vstack([lf, np.ones_like(lf)]).T
    coef, *_ = np.linalg.lstsq(A, lp.T, rcond=None)
    slope = -coef[0]                           # one exponent per channel
    # individual alpha frequency: power-weighted centre of gravity in 7-13 Hz
    am = (f >= 7) & (f <= 13)
    pa = p[:, am]
    paf = (pa * f[am]).sum(axis=1) / (pa.sum(axis=1) + 1e-12)
    return bp.ravel(), slope, paf


def _complexity(x, fs):
    # Higuchi fractal dimension + Hjorth mobility/complexity of the 1-45 Hz signal
    sos = butter(4, [1, 45], btype="bandpass", fs=fs, output="sos")
    xf = sosfiltfilt(sos, x, axis=-1)
    fd = np.array([higuchi_fd(np.ascontiguousarray(ch), kmax=10) for ch in xf])
    d1 = np.diff(xf, axis=-1)
    d2 = np.diff(d1, axis=-1)
    v0 = xf.var(axis=-1) + 1e-12
    v1 = d1.var(axis=-1) + 1e-12
    v2 = d2.var(axis=-1) + 1e-12
    mob = np.sqrt(v1 / v0)
    comp = np.sqrt(v2 / v1) / mob
    return fd, np.concatenate([mob, comp])


def _cov(x, fs, lo, hi):
    sos = butter(4, [lo, hi], btype="bandpass", fs=fs, output="sos")
    xf = sosfiltfilt(sos, x.astype(np.float64), axis=-1)
    C = OAS(assume_centered=False).fit(xf.T).covariance_
    C = C / np.trace(C)
    C = C + 1e-6 * np.eye(C.shape[0])
    return C


def featurize(items, ctx):
    fs = ctx["fs"]
    feats = []
    blocks = {c: ctx["load_eeg"](items, c) for c in ctx["conditions"]}
    for i in range(len(items)):
        bp, sl, pf, fd, hj, cv = [], [], [], [], [], []
        for c in ctx["conditions"]:
            x = blocks[c][i]
            if x is None:                      # missing condition -> fall back to eyes closed
                x = blocks[ctx["conditions"][0]][i]
            x64 = x.astype(np.float64)
            b, s, a = _spectral(x64, fs)
            bp.append(b)
            sl.append(s)
            pf.append(a)
            h, hp = _complexity(x64, fs)
            fd.append(h)
            hj.append(hp)
            for lo, hi in COV_BANDS:
                cv.append(_cov(x, fs, lo, hi).ravel())
        feats.append(np.concatenate(bp + sl + pf + fd + hj + cv))
    return np.asarray(feats, dtype=np.float64)


def _split(F, ctx):
    n_ch = len(ctx["channels"])
    n_cond = len(ctx["conditions"])
    n_bp = n_ch * len(BANDS) * n_cond + 5 * n_ch * n_cond
    bp = F[:, :n_bp]
    covs = F[:, n_bp:].reshape(len(F), n_cond * len(COV_BANDS), n_ch, n_ch)
    return bp, covs


def fit_predict(F_train, y_train, F_test, ctx):
    F_train = np.asarray(F_train, dtype=np.float64)
    F_test = np.asarray(F_test, dtype=np.float64)
    bp_tr, cov_tr = _split(F_train, ctx)
    bp_te, cov_te = _split(F_test, ctx)
    ts_tr, ts_te = [bp_tr], [bp_te]
    for k in range(cov_tr.shape[1]):
        ts = TangentSpace(metric="riemann").fit(cov_tr[:, k])
        ts_tr.append(ts.transform(cov_tr[:, k]))
        ts_te.append(ts.transform(cov_te[:, k]))
    X_tr = np.hstack(ts_tr)
    X_te = np.hstack(ts_te)
    sc = StandardScaler().fit(X_tr)
    clf = LogisticRegression(C=0.1, max_iter=5000).fit(sc.transform(X_tr), y_train)
    return clf.predict_proba(sc.transform(X_te))[:, 1]