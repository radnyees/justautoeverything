"""Log relative band power (delta/theta/alpha/beta/gamma) per channel from Welch PSD of each
available condition + Riemannian tangent-space features of trace-normalized, average-referenced,
1-45 Hz band-passed OAS spatial covariance (reference point fitted on the training fold only);
standardized, L2 logistic regression."""
import numpy as np
from scipy.signal import welch, butter, sosfiltfilt
from sklearn.covariance import OAS
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pyriemann.tangentspace import TangentSpace

BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]


def _bandpower(x, fs):
    f, p = welch(x, fs=fs, nperseg=int(2 * fs), axis=-1)
    tot = p[:, (f >= 1) & (f <= 45)].sum(axis=1, keepdims=True) + 1e-12
    return np.log(np.stack([p[:, (f >= a) & (f < b)].sum(axis=1) for a, b in BANDS], axis=1) / tot)


def _spatial_cov(x, fs):
    x = x.astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)                       # average re-reference
    sos = butter(4, [1.0, min(45.0, 0.45 * fs)], btype="band", fs=fs, output="sos")
    x = sosfiltfilt(sos, x, axis=-1)
    C = OAS(assume_centered=False).fit(x.T).covariance_        # shrinkage -> full rank after avg ref
    return C / np.trace(C)                                      # amplitude-scale invariant


def featurize(items, ctx):
    fs = ctx["fs"]
    bp_feats, cov_feats = [], []
    blocks = {c: ctx["load_eeg"](items, c) for c in ctx["conditions"]}
    for i in range(len(items)):
        fb, fc = [], []
        for c in ctx["conditions"]:
            x = blocks[c][i]
            if x is None:                      # missing condition -> fall back to eyes closed
                x = blocks[ctx["conditions"][0]][i]
            fb.append(_bandpower(x, fs).ravel())
            fc.append(_spatial_cov(x, fs).ravel())
        bp_feats.append(np.concatenate(fb))
        cov_feats.append(np.concatenate(fc))
    return np.hstack([np.asarray(bp_feats), np.asarray(cov_feats)]).astype(np.float64)


def _split(F, ctx):
    n_ch = len(ctx["channels"])
    n_cond = len(ctx["conditions"])
    nb = n_cond * n_ch * len(BANDS)
    bp = F[:, :nb]
    covs = F[:, nb:].reshape(len(F), n_cond, n_ch, n_ch)
    return bp, covs


def fit_predict(F_train, y_train, F_test, ctx):
    F_train = np.asarray(F_train, dtype=np.float64)
    F_test = np.asarray(F_test, dtype=np.float64)
    bp_tr, cov_tr = _split(F_train, ctx)
    bp_te, cov_te = _split(F_test, ctx)
    ts_tr, ts_te = [], []
    for k in range(cov_tr.shape[1]):
        Ctr = (cov_tr[:, k] + cov_tr[:, k].transpose(0, 2, 1)) / 2
        Cte = (cov_te[:, k] + cov_te[:, k].transpose(0, 2, 1)) / 2
        ts = TangentSpace(metric="riemann").fit(Ctr)
        ts_tr.append(ts.transform(Ctr))
        ts_te.append(ts.transform(Cte))
    X_tr = np.hstack([bp_tr] + ts_tr)
    X_te = np.hstack([bp_te] + ts_te)
    sc = StandardScaler().fit(X_tr)
    clf = LogisticRegression(C=0.1, max_iter=5000).fit(sc.transform(X_tr), y_train)
    return clf.predict_proba(sc.transform(X_te))[:, 1]