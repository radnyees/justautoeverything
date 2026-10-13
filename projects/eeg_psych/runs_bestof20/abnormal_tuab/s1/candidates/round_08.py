"""Full-resolution log-PSD (1-Hz bins, 1-45 Hz, median-Welch) per channel: absolute log power plus
log relative power summary, standardized, L2 logistic regression with C tuned by grouped inner CV."""
import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

FMIN, FMAX = 1, 45
BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]


def _spectrum(x, fs):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    f, p = welch(x, fs=fs, nperseg=int(2 * fs), axis=-1, average="median")
    p = p + 1e-12
    # 1-Hz bins of absolute log power
    bins = np.stack([p[:, (f >= lo) & (f < lo + 1)].mean(axis=1) for lo in range(FMIN, FMAX)], axis=1)
    logabs = np.log(bins)
    # log relative band power (scale-free summary)
    tot = p[:, (f >= FMIN) & (f <= FMAX)].sum(axis=1, keepdims=True)
    rel = np.log(np.stack([p[:, (f >= a) & (f < b)].sum(axis=1) for a, b in BANDS], axis=1) / tot)
    return np.concatenate([logabs.ravel(), rel.ravel()])


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
    Xt = np.asarray(F_test, dtype=np.float64)
    y = np.asarray(y_train)
    groups = np.asarray(ctx.get("groups_train", np.arange(len(y))))
    Cs = [0.001, 0.003, 0.01, 0.03, 0.1]
    gkf = GroupKFold(n_splits=4)
    scores = []
    for C in Cs:
        s = []
        for tr, va in gkf.split(X, y, groups):
            sc, clf = _fit(X[tr], y[tr], C)
            s.append(roc_auc_score(y[va], clf.predict_proba(sc.transform(X[va]))[:, 1]))
        scores.append(np.mean(s))
    best_C = Cs[int(np.argmax(scores))]
    sc, clf = _fit(X, y, best_C)
    return clf.predict_proba(sc.transform(Xt))[:, 1]