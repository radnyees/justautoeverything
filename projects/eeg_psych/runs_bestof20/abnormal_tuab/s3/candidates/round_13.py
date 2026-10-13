"""Log absolute power spectrum in fine bins (1-Hz bins 1-20 Hz, wider bins up to 45 Hz) per channel
from a median-averaged Welch PSD (robust to transient artifacts), standardized, L2 logistic
regression with C chosen by grouped inner CV."""
import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

EDGES = list(range(1, 21)) + [24, 28, 32, 38, 45]
BINS = list(zip(EDGES[:-1], EDGES[1:]))
CHUNK = 200


def _logspec(x, fs):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    f, p = welch(x, fs=fs, nperseg=int(4 * fs), noverlap=int(2 * fs), axis=-1, average="median")
    return np.log(np.stack([p[:, (f >= a) & (f < b)].mean(axis=1) for a, b in BINS], axis=1) + 1e-12)


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
                f.append(_logspec(x, fs).ravel())
            feats.append(np.concatenate(f))
        del blocks
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
    Cs = [0.001, 0.003, 0.01, 0.03, 0.1]
    gkf = GroupKFold(n_splits=4)
    scores = np.zeros(len(Cs))
    for tr, va in gkf.split(X, y, groups):
        for k, C in enumerate(Cs):
            sc, clf = _fit(X[tr], y[tr], C)
            scores[k] += roc_auc_score(y[va], clf.predict_proba(sc.transform(X[va]))[:, 1])
    C = Cs[int(np.argmax(scores))]
    sc, clf = _fit(X, y, C)
    return clf.predict_proba(sc.transform(np.asarray(F_test, dtype=np.float64)))[:, 1]