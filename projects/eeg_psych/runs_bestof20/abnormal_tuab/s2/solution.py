"""Fine-resolution log spectrum (1-Hz bins, 1-45 Hz) per channel from median-averaged Welch PSD
(robust to transient artifacts) + per-channel log total power; standardized, L2 logistic
regression with C chosen by grouped inner CV on the training fold."""
import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

FMIN, FMAX = 1, 45
C_GRID = [0.0003, 0.001, 0.003, 0.01, 0.03]


def _spectrum(x, fs):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    f, p = welch(x, fs=fs, nperseg=int(2 * fs), noverlap=int(fs), axis=-1, average="median")
    bins = []
    for lo in range(FMIN, FMAX):
        m = (f >= lo) & (f < lo + 1)
        bins.append(p[:, m].mean(axis=1))
    bins = np.stack(bins, axis=1) + 1e-12                          # (ch, 44)
    tot = np.log(bins.sum(axis=1, keepdims=True))                  # (ch, 1)
    return np.concatenate([np.log(bins), tot], axis=1)


def featurize(items, ctx):
    feats = []
    blocks = {c: ctx["load_eeg"](items, c) for c in ctx["conditions"]}
    for i in range(len(items)):
        f = []
        for c in ctx["conditions"]:
            x = blocks[c][i]
            if x is None:                      # missing condition -> fall back to first condition
                x = blocks[ctx["conditions"][0]][i]
            f.append(_spectrum(x, ctx["fs"]).ravel())
        feats.append(np.concatenate(f))
    return np.asarray(feats, dtype=np.float32)


def _fit(X, y, C):
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=C, max_iter=5000).fit(sc.transform(X), y)
    return sc, clf


def fit_predict(F_train, y_train, F_test, ctx):
    X = np.asarray(F_train, dtype=np.float64)
    Xt = np.asarray(F_test, dtype=np.float64)
    y = np.asarray(y_train)
    groups = ctx.get("groups_train")
    if groups is None:
        groups = np.arange(len(y))
    best_C, best_s = C_GRID[len(C_GRID) // 2], -np.inf
    gkf = GroupKFold(n_splits=3)
    for C in C_GRID:
        scores = []
        for tr, va in gkf.split(X, y, groups):
            sc, clf = _fit(X[tr], y[tr], C)
            scores.append(roc_auc_score(y[va], clf.predict_proba(sc.transform(X[va]))[:, 1]))
        s = float(np.mean(scores))
        if s > best_s:
            best_s, best_C = s, C
    sc, clf = _fit(X, y, best_C)
    return clf.predict_proba(sc.transform(Xt))[:, 1]