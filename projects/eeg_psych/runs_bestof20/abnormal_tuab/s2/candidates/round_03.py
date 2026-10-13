"""Fine-resolution robust spectrum: per channel, log relative power in 1-Hz bins (1-45 Hz) plus
log total power, from median-averaged Welch PSD (robust to transient artifacts in unprocessed
clinical EEG). Standardized features, L2 logistic regression with C tuned by grouped inner CV."""
import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

EDGES = np.arange(1, 46)          # 1-Hz bins from 1 to 45 Hz -> 44 bins
CHUNK = 200


def _spectrum(x, fs):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    f, p = welch(x, fs=fs, nperseg=int(2 * fs), noverlap=int(fs), axis=-1, average="median")
    bins = np.stack([p[:, (f >= a) & (f < b)].mean(axis=1)
                     for a, b in zip(EDGES[:-1], EDGES[1:])], axis=1) + 1e-12
    tot = bins.sum(axis=1, keepdims=True)
    rel = np.log(bins / tot)
    return np.concatenate([rel, np.log(tot)], axis=1).ravel()


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
                f.append(_spectrum(x, fs))
            feats.append(np.concatenate(f))
        del blocks
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
    grid = [0.001, 0.003, 0.01, 0.03, 0.1]
    best_C, best = grid[0], -np.inf
    gkf = GroupKFold(n_splits=4)
    for C in grid:
        scores = []
        for tr, va in gkf.split(X, y, groups):
            sc, clf = _fit(X[tr], y[tr], C)
            scores.append(roc_auc_score(y[va], clf.predict_proba(sc.transform(X[va]))[:, 1]))
        m = float(np.mean(scores))
        if m > best:
            best, best_C = m, C
    sc, clf = _fit(X, y, best_C)
    return clf.predict_proba(sc.transform(Xt))[:, 1]