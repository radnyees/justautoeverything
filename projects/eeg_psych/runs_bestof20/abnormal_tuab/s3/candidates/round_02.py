"""Log relative PSD in 1-Hz bins (1-45 Hz) per channel from median-averaged Welch (artifact-robust,
amplitude-scale invariant), standardized, L2 logistic regression with C tuned by inner grouped CV."""
import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

FMIN, FMAX = 1, 45
CHUNK = 150


def _logrel_spectrum(x, fs):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    f, p = welch(x, fs=fs, nperseg=int(4 * fs), noverlap=int(2 * fs), axis=-1, average="median")
    edges = np.arange(FMIN, FMAX + 1)
    bins = np.stack([p[:, (f >= a) & (f < a + 1)].mean(axis=1) for a in edges[:-1]], axis=1)
    tot = bins.sum(axis=1, keepdims=True) + 1e-20
    return np.log(bins / tot + 1e-12)


def featurize(items, ctx):
    fs = ctx["fs"]
    conds = ctx["conditions"]
    feats = []
    for s in range(0, len(items), CHUNK):
        sub = items[s:s + CHUNK]
        blocks = {c: ctx["load_eeg"](sub, c) for c in conds}
        for i in range(len(sub)):
            f = []
            for c in conds:
                x = blocks[c][i]
                if x is None:                  # missing condition -> fall back to first condition
                    x = blocks[conds[0]][i]
                f.append(_logrel_spectrum(x, fs).ravel())
            feats.append(np.concatenate(f))
        del blocks
    return np.asarray(feats, dtype=np.float32)


def fit_predict(F_train, y_train, F_test, ctx):
    F_train = np.asarray(F_train, dtype=np.float64)
    F_test = np.asarray(F_test, dtype=np.float64)
    y_train = np.asarray(y_train)
    sc = StandardScaler().fit(F_train)
    Xtr, Xte = sc.transform(F_train), sc.transform(F_test)

    groups = ctx.get("groups_train")
    if groups is not None and len(np.unique(groups)) >= 5:
        splits = list(GroupKFold(n_splits=5).split(Xtr, y_train, groups))
    else:
        splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(Xtr, y_train))
    Cs = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
    cv = LogisticRegressionCV(Cs=Cs, cv=splits, scoring="roc_auc", max_iter=5000,
                              solver="lbfgs").fit(Xtr, y_train)
    C = float(cv.C_[0])
    clf = LogisticRegression(C=C, max_iter=5000).fit(Xtr, y_train)
    return clf.predict_proba(Xte)[:, 1]