"""Fine-resolution log absolute spectrum (1 Hz bins, 1-45 Hz) per channel from a median-averaged Welch PSD
(robust to transient artifacts), plus log total power; standardized, L2 logistic regression with C chosen
by grouped inner CV."""
import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

EDGES = np.arange(1, 46, 1)  # 1..45 Hz -> 44 one-Hz bins


def _logspec(x, fs):
    x = np.asarray(x, dtype=np.float64)
    f, p = welch(x, fs=fs, nperseg=int(2 * fs), noverlap=int(fs), axis=-1, average="median")
    bins = np.stack([p[:, (f >= a) & (f < b)].mean(axis=1) for a, b in zip(EDGES[:-1], EDGES[1:])], axis=1)
    tot = p[:, (f >= 1) & (f <= 45)].sum(axis=1, keepdims=True)
    return np.log(np.concatenate([bins, tot], axis=1) + 1e-12)


def featurize(items, ctx):
    feats = []
    blocks = {c: ctx["load_eeg"](items, c) for c in ctx["conditions"]}
    for i in range(len(items)):
        f = []
        for c in ctx["conditions"]:
            x = blocks[c][i]
            if x is None:                      # missing condition -> fall back to first condition
                x = blocks[ctx["conditions"][0]][i]
            f.append(_logspec(x, ctx["fs"]).ravel())
        feats.append(np.concatenate(f))
    return np.asarray(feats, dtype=np.float32)


def _choose_C(X, y, groups, grid=(0.001, 0.003, 0.01, 0.03, 0.1)):
    if groups is None:
        groups = np.arange(len(y))
    gkf = GroupKFold(n_splits=4)
    best_C, best = grid[0], -np.inf
    for C in grid:
        oof = np.zeros(len(y))
        for tr, va in gkf.split(X, y, groups):
            sc = StandardScaler().fit(X[tr])
            m = LogisticRegression(C=C, max_iter=5000).fit(sc.transform(X[tr]), y[tr])
            oof[va] = m.decision_function(sc.transform(X[va]))
        s = roc_auc_score(y, oof)
        if s > best:
            best, best_C = s, C
    return best_C


def fit_predict(F_train, y_train, F_test, ctx):
    X = np.asarray(F_train, dtype=np.float64)
    Xt = np.asarray(F_test, dtype=np.float64)
    y = np.asarray(y_train)
    groups = ctx.get("groups_train")
    groups = None if groups is None else np.asarray(groups)
    C = _choose_C(X, y, groups)
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=C, max_iter=5000).fit(sc.transform(X), y)
    return clf.predict_proba(sc.transform(Xt))[:, 1]