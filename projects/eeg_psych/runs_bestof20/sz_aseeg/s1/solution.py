"""Log relative band power (per channel, each condition) + Riemannian tangent-space covariance
branch (avg-referenced, band-filtered EC epochs, trace-normalized), two L2 logistic regressions
whose probabilities are averaged."""
import numpy as np
from scipy.signal import welch, butter, sosfiltfilt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pyriemann.estimation import Covariances
from pyriemann.utils.mean import mean_riemann
from pyriemann.tangentspace import TangentSpace

BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
COV_BANDS = [(1, 4), (4, 8), (8, 13), (13, 30)]
EPOCH_SEC = 4


def _bandpower(x, fs):
    f, p = welch(x, fs=fs, nperseg=int(2 * fs), axis=-1)
    tot = p[:, (f >= 1) & (f <= 45)].sum(axis=1, keepdims=True) + 1e-12
    return np.log(np.stack([p[:, (f >= a) & (f < b)].sum(axis=1) for a, b in BANDS], axis=1) / tot)


def _band_covs(x, fs):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)          # average re-reference
    n_ch = x.shape[0]
    L = int(EPOCH_SEC * fs)
    est = Covariances(estimator="oas")
    out = []
    for a, b in COV_BANDS:
        sos = butter(4, [a, b], btype="bandpass", fs=fs, output="sos")
        xf = sosfiltfilt(sos, x, axis=-1)
        n_ep = max(1, xf.shape[1] // L)
        if xf.shape[1] >= L:
            ep = np.stack([xf[:, k * L:(k + 1) * L] for k in range(n_ep)])
        else:
            ep = xf[None]
        C = est.fit_transform(ep) + 1e-10 * np.eye(n_ch)[None]
        M = mean_riemann(C) if len(C) > 1 else C[0]
        M = M / np.trace(M) * n_ch                 # amplitude-scale invariance
        out.append(M.ravel())
    return np.concatenate(out)


def featurize(items, ctx):
    fs = ctx["fs"]
    conds = ctx["conditions"]
    feats = []
    blocks = {c: ctx["load_eeg"](items, c) for c in conds}
    for i in range(len(items)):
        f = []
        for c in conds:
            x = blocks[c][i]
            if x is None:                      # missing condition -> fall back to first condition
                x = blocks[conds[0]][i]
            f.append(_bandpower(x, fs).ravel())
        x0 = blocks[conds[0]][i]
        if x0 is None:
            x0 = next(blocks[c][i] for c in conds if blocks[c][i] is not None)
        f.append(_band_covs(x0, fs))
        feats.append(np.concatenate(f))
    return np.asarray(feats, dtype=np.float64)


def fit_predict(F_train, y_train, F_test, ctx):
    F_train = np.asarray(F_train, dtype=np.float64)
    F_test = np.asarray(F_test, dtype=np.float64)
    n_ch = len(ctx["channels"])
    n_bp = len(ctx["conditions"]) * n_ch * len(BANDS)

    # branch 1: relative band power
    Xtr, Xte = F_train[:, :n_bp], F_test[:, :n_bp]
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.1, max_iter=5000).fit(sc.transform(Xtr), y_train)
    p1 = clf.predict_proba(sc.transform(Xte))[:, 1]

    # branch 2: tangent-space covariances, one tangent space per band
    nb = len(COV_BANDS)
    Ctr = F_train[:, n_bp:].reshape(len(F_train), nb, n_ch, n_ch)
    Cte = F_test[:, n_bp:].reshape(len(F_test), nb, n_ch, n_ch)
    Ttr, Tte = [], []
    for k in range(nb):
        ts = TangentSpace(metric="riemann").fit(Ctr[:, k])
        Ttr.append(ts.transform(Ctr[:, k]))
        Tte.append(ts.transform(Cte[:, k]))
    Ttr, Tte = np.hstack(Ttr), np.hstack(Tte)
    sc2 = StandardScaler().fit(Ttr)
    clf2 = LogisticRegression(C=0.02, max_iter=5000).fit(sc2.transform(Ttr), y_train)
    p2 = clf2.predict_proba(sc2.transform(Tte))[:, 1]

    return 0.5 * p1 + 0.5 * p2