"""Log relative band power per channel + filter-bank covariance matrices (5 bands) projected to the
Riemannian tangent space (reference fitted on the training fold), standardized, L2 logistic regression
with C chosen by patient-grouped inner CV."""
import numpy as np
from scipy.signal import welch, butter, sosfiltfilt
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from pyriemann.tangentspace import TangentSpace

BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
CHUNK = 200


def _bandpower(x, fs):
    f, p = welch(x, fs=fs, nperseg=int(2 * fs), axis=-1)
    tot = p[:, (f >= 1) & (f <= 45)].sum(axis=1, keepdims=True) + 1e-12
    return np.log(np.stack([p[:, (f >= a) & (f < b)].sum(axis=1) for a, b in BANDS], axis=1) / tot)


def _fbcov(x, sos_list):
    x = x.astype(np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    n = x.shape[0]
    out = []
    for sos in sos_list:
        y = sosfiltfilt(sos, x, axis=-1)
        C = y @ y.T / y.shape[1]
        C += 1e-3 * np.trace(C) / n * np.eye(n)   # avg reference -> rank deficient; regularize
        out.append(C.ravel())
    return np.concatenate(out)


def featurize(items, ctx):
    fs = ctx["fs"]
    conds = ctx["conditions"]
    sos_list = [butter(4, [a, min(b, 0.49 * fs)], btype="band", fs=fs, output="sos") for a, b in BANDS]
    feats = []
    for s in range(0, len(items), CHUNK):
        sub = items[s:s + CHUNK]
        blocks = {c: ctx["load_eeg"](sub, c) for c in conds}
        for i in range(len(sub)):
            bp, cv = [], []
            for c in conds:
                x = blocks[c][i]
                if x is None:                      # missing condition -> fall back to first condition
                    x = blocks[conds[0]][i]
                bp.append(_bandpower(x, fs).ravel())
                cv.append(_fbcov(x, sos_list))
            feats.append(np.concatenate(bp + cv))
        del blocks
    return np.asarray(feats, dtype=np.float64)


def _split(F, ctx):
    n_ch = len(ctx["channels"])
    n_c = len(ctx["conditions"])
    nbp = n_c * n_ch * len(BANDS)
    bp = F[:, :nbp]
    covs = F[:, nbp:].reshape(len(F), n_c * len(BANDS), n_ch, n_ch)
    return bp, covs


def fit_predict(F_train, y_train, F_test, ctx):
    F_train = np.asarray(F_train, dtype=np.float64)
    F_test = np.asarray(F_test, dtype=np.float64)
    bp_tr, cov_tr = _split(F_train, ctx)
    bp_te, cov_te = _split(F_test, ctx)

    ts_tr, ts_te = [], []
    for k in range(cov_tr.shape[1]):
        ts = TangentSpace(metric="riemann").fit(cov_tr[:, k])
        ts_tr.append(ts.transform(cov_tr[:, k]))
        ts_te.append(ts.transform(cov_te[:, k]))
    X_tr = np.hstack([bp_tr] + ts_tr)
    X_te = np.hstack([bp_te] + ts_te)

    sc = StandardScaler().fit(X_tr)
    X_tr = sc.transform(X_tr)
    X_te = sc.transform(X_te)

    y_train = np.asarray(y_train)
    groups = ctx.get("groups_train")
    Cs = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
    if groups is not None:
        cv = list(GroupKFold(n_splits=5).split(X_tr, y_train, np.asarray(groups)))
        clf = LogisticRegressionCV(Cs=Cs, cv=cv, scoring="roc_auc", max_iter=5000,
                                   n_jobs=-1).fit(X_tr, y_train)
    else:
        clf = LogisticRegression(C=1e-2, max_iter=5000).fit(X_tr, y_train)
    return clf.predict_proba(X_te)[:, 1]