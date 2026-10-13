"""Log relative power in fine 1-Hz bins (1-45 Hz) per channel from a median-averaged Welch PSD
(robust to transient artifacts), standardized, L2 logistic regression with C tuned by grouped
inner CV on the training fold."""
import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

FMIN, FMAX = 1, 45
C_GRID = [0.003, 0.01, 0.03, 0.1, 0.3]


def _fine_relpower(x, fs):
    x = x - x.mean(axis=1, keepdims=True)
    f, p = welch(x, fs=fs, nperseg=int(2 * fs), noverlap=int(fs), axis=-1, average="median")
    edges = np.arange(FMIN, FMAX + 1)
    bins = np.stack([p[:, (f >= a) & (f < a + 1)].mean(axis=1) for a in edges[:-1]], axis=1)
    tot = bins.sum(axis=1, keepdims=True) + 1e-12
    return np.log(bins / tot + 1e-12)


def featurize(items, ctx):
    feats = []
    blocks = {c: ctx["load_eeg"](items, c) for c in ctx["conditions"]}
    for i in range(len(items)):
        f = []
        for c in ctx["conditions"]:
            x = blocks[c][i]
            if x is None:                      # missing condition -> fall back to first condition
                x = blocks[ctx["conditions"][0]][i]
            f.append(_fine_relpower(np.asarray(x, dtype=np.float64), ctx["fs"]).ravel())
        feats.append(np.concatenate(f))
    return np.nan_to_num(np.asarray(feats, dtype=np.float32))


def _fit(X, y, C):
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=C, max_iter=5000).fit(sc.transform(X), y)
    return sc, clf


def fit_predict(F_train, y_train, F_test, ctx):
    X, y = np.asarray(F_train, dtype=np.float64), np.asarray(y_train)
    groups = ctx.get("groups_train")
    if groups is None:
        groups = np.arange(len(y))
    groups = np.asarray(groups)
    best_C, best_s = C_GRID[2], -np.inf
    if len(np.unique(groups)) >= 3:
        cv = GroupKFold(n_splits=3)
        for C in C_GRID:
            scores = []
            for tr, va in cv.split(X, y, groups):
                if len(np.unique(y[va])) < 2:
                    continue
                sc, clf = _fit(X[tr], y[tr], C)
                scores.append(roc_auc_score(y[va], clf.predict_proba(sc.transform(X[va]))[:, 1]))
            s = np.mean(scores) if scores else -np.inf
            if s > best_s:
                best_s, best_C = s, C
    sc, clf = _fit(X, y, best_C)
    return clf.predict_proba(sc.transform(np.asarray(F_test, dtype=np.float64)))[:, 1]