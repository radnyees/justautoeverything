"""Log relative band power (delta/theta/alpha/beta/gamma) per channel from Welch PSD of each
available condition + trace-normalized filter-bank spatial covariances (broadband 1-45 Hz,
theta, alpha, beta) per condition, each projected to its own Riemannian tangent space
(reference = training-fold mean). One standardized L2 logistic regression per feature block
(band power block + each condition x band covariance block); block log-odds are averaged."""
import numpy as np
from scipy.signal import welch, butter, sosfiltfilt
from sklearn.covariance import OAS
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pyriemann.tangentspace import TangentSpace

BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
COV_BANDS = [(1, 45), (4, 8), (8, 13), (13, 30)]


def _bandpower(x, fs):
    f, p = welch(x, fs=fs, nperseg=int(2 * fs), axis=-1)
    tot = p[:, (f >= 1) & (f <= 45)].sum(axis=1, keepdims=True) + 1e-12
    return np.log(np.stack([p[:, (f >= a) & (f < b)].sum(axis=1) for a, b in BANDS], axis=1) / tot)


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
        bp, cv = [], []
        for c in ctx["conditions"]:
            x = blocks[c][i]
            if x is None:                      # missing condition -> fall back to eyes closed
                x = blocks[ctx["conditions"][0]][i]
            bp.append(_bandpower(x, fs).ravel())
            for lo, hi in COV_BANDS:
                cv.append(_cov(x, fs, lo, hi).ravel())
        feats.append(np.concatenate(bp + cv))
    return np.asarray(feats, dtype=np.float64)


def _split(F, ctx):
    n_ch = len(ctx["channels"])
    n_cond = len(ctx["conditions"])
    n_bp = n_ch * len(BANDS) * n_cond
    bp = F[:, :n_bp]
    covs = F[:, n_bp:].reshape(len(F), n_cond * len(COV_BANDS), n_ch, n_ch)
    return bp, covs


def _block_logit(X_tr, y_tr, X_te):
    sc = StandardScaler().fit(X_tr)
    clf = LogisticRegression(C=0.1, max_iter=5000).fit(sc.transform(X_tr), y_tr)
    return clf.decision_function(sc.transform(X_te))


def fit_predict(F_train, y_train, F_test, ctx):
    F_train = np.asarray(F_train, dtype=np.float64)
    F_test = np.asarray(F_test, dtype=np.float64)
    bp_tr, cov_tr = _split(F_train, ctx)
    bp_te, cov_te = _split(F_test, ctx)
    blocks_tr, blocks_te = [bp_tr], [bp_te]
    for k in range(cov_tr.shape[1]):
        ts = TangentSpace(metric="riemann").fit(cov_tr[:, k])
        blocks_tr.append(ts.transform(cov_tr[:, k]))
        blocks_te.append(ts.transform(cov_te[:, k]))
    scores = [_block_logit(a, y_train, b) for a, b in zip(blocks_tr, blocks_te)]
    return np.mean(scores, axis=0)