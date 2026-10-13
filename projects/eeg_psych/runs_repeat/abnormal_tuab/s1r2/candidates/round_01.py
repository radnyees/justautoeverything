"""Log relative power spectrum in 1-Hz bins (1-45 Hz) per channel from Welch PSD of each available
condition, standardized, L2 logistic regression with C chosen by inner grouped CV."""
import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

BINS = [(k, k + 1) for k in range(1, 45)]
CS = [0.001, 0.003, 0.01, 0.03, 0.1]


def _spectrum(x, fs):
    f, p = welch(x, fs=fs, nperseg=int(2 * fs), axis=-1)
    tot = p[:, (f >= 1) & (f < 45)].sum(axis=1, keepdims=True) + 1e-12
    return np.log(np.stack([p[:, (f >= a) & (f < b)].sum(axis=1) for a, b in BINS], axis=1) / tot + 1e-12)


def featurize(items, ctx):
    feats = []
    blocks = {c: ctx["load_eeg"](items, c) for c in ctx["conditions"]}
    for i in range(len(items)):
        f = []
        for c in ctx["conditions"]:
            x = blocks[c][i]
            if x is None:                      # missing condition -> fall back to first condition
                x = blocks[ctx["conditions"][0]][i]
            f.append(_spectrum(np.asarray(x, dtype=np.float64), ctx["fs"]).ravel())
        feats.append(np.concatenate(f))
    return np.asarray(feats, dtype=np.float32)


def _fit(X, y, C):
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=C, max_iter=5000).fit(sc.transform(X), y)
    return sc, clf


def fit_predict(F_train, y_train, F_test, ctx):
    X, y = np.asarray(F_train, dtype=np.float64), np.asarray(y_train)
    groups = ctx.get("groups_train")
    if groups is None:
        groups = np.arange(len(y))
    gkf = GroupKFold(n_splits=5)
    scores = np.zeros(len(CS))
    for tr, va in gkf.split(X, y, groups):
        for j, C in enumerate(CS):
            sc, clf = _fit(X[tr], y[tr], C)
            scores[j] += roc_auc_score(y[va], clf.predict_proba(sc.transform(X[va]))[:, 1])
    best_C = CS[int(np.argmax(scores))]
    sc, clf = _fit(X, y, best_C)
    return clf.predict_proba(sc.transform(np.asarray(F_test, dtype=np.float64)))[:, 1]