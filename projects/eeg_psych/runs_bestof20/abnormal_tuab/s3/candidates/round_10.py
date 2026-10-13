"""Full log-spectral profile per channel (1-Hz bins, 1-45 Hz, relative to total power) plus log
total power per channel, from Welch PSD; standardized, L2 logistic regression with C chosen by
grouped inner CV on the training fold."""
import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

EDGES = np.arange(1, 46, 1.0)   # 1-Hz bins from 1 to 45 Hz


def _spectrum(x, fs):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    f, p = welch(x, fs=fs, nperseg=int(2 * fs), noverlap=int(fs), axis=-1)
    bins = np.stack([p[:, (f >= a) & (f < b)].sum(axis=1)
                     for a, b in zip(EDGES[:-1], EDGES[1:])], axis=1) + 1e-12
    tot = bins.sum(axis=1, keepdims=True)
    rel = np.log(bins / tot)                      # spectral shape (scale-invariant)
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
            f.append(_spectrum(x, ctx["fs"]))
        feats.append(np.concatenate(f))
    return np.asarray(feats, dtype=np.float32)


def _fit(X, y, C):
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=C, max_iter=5000).fit(sc.transform(X), y)
    return sc, clf


def fit_predict(F_train, y_train, F_test, ctx):
    X = np.asarray(F_train, dtype=np.float64)
    y = np.asarray(y_train)
    groups = np.asarray(ctx.get("groups_train", np.arange(len(y))))
    Cs = [0.0003, 0.001, 0.003, 0.01, 0.03]
    gkf = GroupKFold(n_splits=4)
    scores = {C: [] for C in Cs}
    for tr, va in gkf.split(X, y, groups):
        if len(np.unique(y[va])) < 2:
            continue
        for C in Cs:
            sc, clf = _fit(X[tr], y[tr], C)
            scores[C].append(roc_auc_score(y[va], clf.predict_proba(sc.transform(X[va]))[:, 1]))
    best_C = max(Cs, key=lambda c: np.mean(scores[c]) if scores[c] else -1)
    sc, clf = _fit(X, y, best_C)
    return clf.predict_proba(sc.transform(np.asarray(F_test, dtype=np.float64)))[:, 1]