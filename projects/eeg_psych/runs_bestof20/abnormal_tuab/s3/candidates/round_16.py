"""Fine-resolution relative log-spectrum (1-Hz bins, 1-45 Hz) per channel from median-Welch PSD
(robust to transient artifacts) + log total power per channel; standardized L2 logistic regression
with C chosen by inner grouped CV on the training fold."""
import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

EDGES = np.arange(1, 46, 1)          # 1-Hz bins from 1 to 45 Hz -> 44 bins


def _spectrum(x, fs):
    x = x - x.mean(axis=1, keepdims=True)
    f, p = welch(x, fs=fs, nperseg=int(4 * fs), noverlap=int(2 * fs), axis=-1, average="median")
    bins = np.stack([p[:, (f >= a) & (f < b)].mean(axis=1) for a, b in zip(EDGES[:-1], EDGES[1:])],
                    axis=1) + 1e-12
    tot = bins.sum(axis=1, keepdims=True)
    rel = np.log(bins / tot)                       # scale-free spectral shape
    return np.concatenate([rel.ravel(), np.log(tot).ravel()])


def featurize(items, ctx):
    feats = []
    blocks = {c: ctx["load_eeg"](items, c) for c in ctx["conditions"]}
    for i in range(len(items)):
        f = []
        for c in ctx["conditions"]:
            x = blocks[c][i]
            if x is None:                      # missing condition -> fall back to first condition
                x = blocks[ctx["conditions"][0]][i]
            f.append(_spectrum(np.asarray(x, dtype=np.float64), ctx["fs"]))
        feats.append(np.concatenate(f))
    return np.asarray(feats, dtype=np.float32)


def _fit(X, y, C):
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=C, max_iter=5000).fit(sc.transform(X), y)
    return sc, clf


def fit_predict(F_train, y_train, F_test, ctx):
    X = np.asarray(F_train, dtype=np.float64)
    y = np.asarray(y_train)
    groups = ctx.get("groups_train")
    if groups is None:
        groups = np.arange(len(y))
    Cs = [0.0003, 0.001, 0.003, 0.01, 0.03]
    scores = np.zeros(len(Cs))
    for tr, va in GroupKFold(n_splits=4).split(X, y, groups):
        for k, C in enumerate(Cs):
            sc, clf = _fit(X[tr], y[tr], C)
            scores[k] += roc_auc_score(y[va], clf.predict_proba(sc.transform(X[va]))[:, 1])
    C = Cs[int(np.argmax(scores))]
    sc, clf = _fit(X, y, C)
    return clf.predict_proba(sc.transform(np.asarray(F_test, dtype=np.float64)))[:, 1]