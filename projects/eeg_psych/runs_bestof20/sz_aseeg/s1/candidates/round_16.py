"""Log relative band power per channel (Welch) + trace-normalized broadband covariance in Riemannian
tangent space; two L2 logistic regressions whose probabilities are averaged."""
import numpy as np
from scipy.signal import welch, butter, sosfiltfilt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pyriemann.tangentspace import TangentSpace

BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]


def _bandpower(x, fs):
    f, p = welch(x, fs=fs, nperseg=int(2 * fs), axis=-1)
    tot = p[:, (f >= 1) & (f <= 45)].sum(axis=1, keepdims=True) + 1e-12
    return np.log(np.stack([p[:, (f >= a) & (f < b)].sum(axis=1) for a, b in BANDS], axis=1) / tot)


def _cov(x, fs):
    sos = butter(4, [1.0, min(45.0, 0.45 * fs)], btype="band", fs=fs, output="sos")
    xf = sosfiltfilt(sos, x.astype(np.float64), axis=-1)
    xf = xf - xf.mean(axis=1, keepdims=True)
    C = xf @ xf.T / xf.shape[1]
    C = C / np.trace(C)                         # amplitude-scale invariance
    n = C.shape[0]
    mu = np.trace(C) / n
    C = 0.9 * C + 0.1 * mu * np.eye(n)          # fixed shrinkage -> well-conditioned SPD
    return C


def featurize(items, ctx):
    fs = ctx["fs"]
    conds = ctx["conditions"]
    blocks = {c: ctx["load_eeg"](items, c) for c in conds}
    feats = []
    for i in range(len(items)):
        bp, cv = [], []
        for c in conds:
            x = blocks[c][i]
            if x is None:                      # missing condition -> fall back to first condition
                x = blocks[conds[0]][i]
            bp.append(_bandpower(x, fs).ravel())
            cv.append(_cov(x, fs).ravel())
        feats.append(np.concatenate(bp + cv))
    return np.asarray(feats, dtype=np.float64)


def _split(F, ctx):
    n_ch = len(ctx["channels"])
    n_c = len(ctx["conditions"])
    n_bp = n_c * n_ch * len(BANDS)
    bp = F[:, :n_bp]
    covs = F[:, n_bp:].reshape(len(F), n_c, n_ch, n_ch)
    return bp, covs


def fit_predict(F_train, y_train, F_test, ctx):
    F_train = np.asarray(F_train, dtype=np.float64)
    F_test = np.asarray(F_test, dtype=np.float64)
    bp_tr, cv_tr = _split(F_train, ctx)
    bp_te, cv_te = _split(F_test, ctx)

    # view 1: relative band power
    sc = StandardScaler().fit(bp_tr)
    clf = LogisticRegression(C=0.1, max_iter=5000).fit(sc.transform(bp_tr), y_train)
    p1 = clf.predict_proba(sc.transform(bp_te))[:, 1]

    # view 2: covariance tangent space (reference mean fitted on training fold only)
    T_tr, T_te = [], []
    for k in range(cv_tr.shape[1]):
        ts = TangentSpace(metric="riemann").fit(cv_tr[:, k])
        T_tr.append(ts.transform(cv_tr[:, k]))
        T_te.append(ts.transform(cv_te[:, k]))
    T_tr = np.concatenate(T_tr, axis=1)
    T_te = np.concatenate(T_te, axis=1)
    sc2 = StandardScaler().fit(T_tr)
    clf2 = LogisticRegression(C=0.05, max_iter=5000).fit(sc2.transform(T_tr), y_train)
    p2 = clf2.predict_proba(sc2.transform(T_te))[:, 1]

    return 0.5 * p1 + 0.5 * p2