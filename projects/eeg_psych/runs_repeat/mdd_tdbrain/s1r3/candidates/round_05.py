"""Log relative band power (delta/theta/alpha/beta/gamma) per channel from Welch PSD of each
available condition -> standardized L2 logistic regression, averaged with a second L2 logistic
regression on Riemannian tangent-space features of trace-normalized filter-bank spatial
covariances (1-45 Hz broadband + theta, alpha, beta) per condition. The regularization strength
of each logistic regression is chosen by subject-grouped inner CV (AUROC) on the training fold."""
import numpy as np
from scipy.signal import welch, butter, sosfiltfilt
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from pyriemann.tangentspace import TangentSpace

BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
COV_BANDS = [(1, 45), (4, 8), (8, 13), (13, 30)]
CS = np.logspace(-4, 1, 11)


def _bandpower(x, fs):
    f, p = welch(x, fs=fs, nperseg=int(2 * fs), axis=-1)
    tot = p[:, (f >= 1) & (f <= 45)].sum(axis=1, keepdims=True) + 1e-12
    return np.log(np.stack([p[:, (f >= a) & (f < b)].sum(axis=1) for a, b in BANDS], axis=1) / tot)


def _cov(x, fs, lo, hi):
    sos = butter(4, [lo, hi], btype="bandpass", fs=fs, output="sos")
    xf = sosfiltfilt(sos, x.astype(np.float64), axis=-1)
    xf = xf - xf.mean(axis=1, keepdims=True)
    c = xf @ xf.T / xf.shape[1]
    c = c / np.trace(c)                                  # amplitude-scale invariance
    n = c.shape[0]
    c = c + 1e-3 * np.eye(n) / n                         # keep SPD (e.g. rank-deficient references)
    return c


def featurize(items, ctx):
    feats, covs = [], []
    fs = ctx["fs"]
    blocks = {c: ctx["load_eeg"](items, c) for c in ctx["conditions"]}
    for i in range(len(items)):
        f, cv = [], []
        for c in ctx["conditions"]:
            x = blocks[c][i]
            if x is None:                      # missing condition -> fall back to eyes closed
                x = blocks[ctx["conditions"][0]][i]
            f.append(_bandpower(x, fs).ravel())
            for lo, hi in COV_BANDS:
                cv.append(_cov(x, fs, lo, hi).ravel())
        feats.append(np.concatenate(f))
        covs.append(np.concatenate(cv))
    F = np.concatenate([np.asarray(feats, dtype=np.float64), np.asarray(covs, dtype=np.float64)], axis=1)
    return F


def _split(F, ctx):
    F = np.asarray(F, dtype=np.float64)
    nch = len(ctx["channels"])
    ncond = len(ctx["conditions"])
    nbp = ncond * nch * len(BANDS)
    bp = F[:, :nbp]
    covs = F[:, nbp:].reshape(len(F), ncond * len(COV_BANDS), nch, nch)
    covs = 0.5 * (covs + covs.transpose(0, 1, 3, 2))
    return bp, covs


def _inner_splits(y, ctx):
    groups = ctx.get("groups_train")
    if groups is None:
        groups = np.arange(len(y))
    groups = np.asarray(groups)
    k = int(min(5, len(np.unique(groups))))
    return list(GroupKFold(n_splits=k).split(np.zeros(len(y)), y, groups))


def _fit_lr(X_tr, y, X_te, splits):
    sc = StandardScaler().fit(X_tr)
    Xs_tr, Xs_te = sc.transform(X_tr), sc.transform(X_te)
    try:
        cv = LogisticRegressionCV(Cs=CS, cv=splits, scoring="roc_auc", max_iter=5000).fit(Xs_tr, y)
        C = float(cv.C_[0])
    except Exception:
        C = 0.1
    clf = LogisticRegression(C=C, max_iter=5000).fit(Xs_tr, y)
    return clf.predict_proba(Xs_te)[:, 1]


def fit_predict(F_train, y_train, F_test, ctx):
    y_train = np.asarray(y_train)
    bp_tr, cov_tr = _split(F_train, ctx)
    bp_te, cov_te = _split(F_test, ctx)
    splits = _inner_splits(y_train, ctx)

    p_bp = _fit_lr(bp_tr, y_train, bp_te, splits)

    T_tr, T_te = [], []
    for k in range(cov_tr.shape[1]):
        ts = TangentSpace(metric="riemann").fit(cov_tr[:, k])
        T_tr.append(ts.transform(cov_tr[:, k]))
        T_te.append(ts.transform(cov_te[:, k]))
    T_tr, T_te = np.concatenate(T_tr, axis=1), np.concatenate(T_te, axis=1)
    p_ts = _fit_lr(T_tr, y_train, T_te, splits)

    return 0.5 * (p_bp + p_ts)