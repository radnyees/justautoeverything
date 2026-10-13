"""Log relative spectrum in 1-Hz bins (1-45 Hz) per channel from median-averaged Welch PSD
(robust to transient artifacts), standardized, L2 logistic regression with C tuned by grouped inner CV."""
import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

FMIN, FMAX = 1, 45


def _spectrum(x, fs):
    x = np.asarray(x, dtype=np.float64)
    f, p = welch(x, fs=fs, nperseg=int(2 * fs), axis=-1, average="median")
    tot = p[:, (f >= FMIN) & (f < FMAX)].sum(axis=1, keepdims=True) + 1e-12
    bins = [p[:, (f >= a) & (f < a + 1)].sum(axis=1) for a in range(FMIN, FMAX)]
    return np.log(np.stack(bins, axis=1) / tot + 1e-12)


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
    X, y = np.asarray(F_train, dtype=np.float64), np.asarray(y_train)
    groups = ctx.get("groups_train")
    if groups is None:
        groups = np.arange(len(y))
    Cs = [0.001, 0.003, 0.01, 0.03, 0.1]
    scores = {C: [] for C in Cs}
    for tr, va in GroupKFold(n_splits=4).split(X, y, groups):
        if len(np.unique(y[va])) < 2:
            continue
        for C in Cs:
            sc, clf = _fit(X[tr], y[tr], C)
            scores[C].append(roc_auc_score(y[va], clf.predict_proba(sc.transform(X[va]))[:, 1]))
    best = max(Cs, key=lambda C: np.mean(scores[C]) if scores[C] else -1)
    sc, clf = _fit(X, y, best)
    return clf.predict_proba(sc.transform(np.asarray(F_test, dtype=np.float64)))[:, 1]