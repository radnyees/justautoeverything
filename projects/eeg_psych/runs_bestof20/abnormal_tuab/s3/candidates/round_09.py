"""Log relative band power (5 classic bands) + fine-grained log relative spectrum (1-Hz bins, 1-45 Hz)
per channel from Welch PSD; standardized, L2 logistic regression with C tuned by grouped inner CV."""
import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
FINE = [(a, a + 1) for a in range(1, 45)]


def _spec_feats(x, fs):
    f, p = welch(x, fs=fs, nperseg=int(2 * fs), axis=-1)
    tot = p[:, (f >= 1) & (f <= 45)].sum(axis=1, keepdims=True) + 1e-12
    bp = np.stack([p[:, (f >= a) & (f < b)].sum(axis=1) for a, b in BANDS], axis=1) / tot
    fp = np.stack([p[:, (f >= a) & (f < b)].sum(axis=1) for a, b in FINE], axis=1) / tot
    return np.concatenate([np.log(bp + 1e-12).ravel(), np.log(fp + 1e-12).ravel()])


def featurize(items, ctx):
    feats = []
    blocks = {c: ctx["load_eeg"](items, c) for c in ctx["conditions"]}
    for i in range(len(items)):
        f = []
        for c in ctx["conditions"]:
            x = blocks[c][i]
            if x is None:                      # missing condition -> fall back to first condition
                x = blocks[ctx["conditions"][0]][i]
            f.append(_spec_feats(np.asarray(x, dtype=np.float64), ctx["fs"]))
        feats.append(np.concatenate(f))
    return np.asarray(feats, dtype=np.float32)


def _fit(X, y, C):
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=C, max_iter=5000).fit(sc.transform(X), y)
    return sc, clf


def fit_predict(F_train, y_train, F_test, ctx):
    X = np.asarray(F_train, dtype=np.float64)
    y = np.asarray(y_train)
    Xt = np.asarray(F_test, dtype=np.float64)
    groups = np.asarray(ctx["groups_train"])

    grid = [0.001, 0.003, 0.01, 0.03, 0.1]
    gkf = GroupKFold(n_splits=4)
    scores = np.zeros(len(grid))
    for tr, va in gkf.split(X, y, groups):
        for k, C in enumerate(grid):
            sc, clf = _fit(X[tr], y[tr], C)
            scores[k] += roc_auc_score(y[va], clf.predict_proba(sc.transform(X[va]))[:, 1])
    best_C = grid[int(np.argmax(scores))]

    sc, clf = _fit(X, y, best_C)
    return clf.predict_proba(sc.transform(Xt))[:, 1]