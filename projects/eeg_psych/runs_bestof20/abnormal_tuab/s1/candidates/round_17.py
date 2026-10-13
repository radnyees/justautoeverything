"""Log relative power spectrum (2-Hz bins, 1-45 Hz) per channel from median-averaged Welch PSD
(robust to transient artifacts), standardized, L2 logistic regression with C tuned by grouped inner CV."""
import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EDGES = np.arange(1, 46, 2)          # 1-3, 3-5, ..., 43-45 Hz -> 22 bins
C_GRID = [0.001, 0.003, 0.01, 0.03, 0.1]


def _relspec(x, fs):
    x = np.asarray(x, dtype=np.float64)
    f, p = welch(x, fs=fs, nperseg=int(2 * fs), axis=-1, average="median")
    tot = p[:, (f >= 1) & (f <= 45)].sum(axis=1, keepdims=True) + 1e-12
    bins = np.stack([p[:, (f >= a) & (f < b)].sum(axis=1) for a, b in zip(EDGES[:-1], EDGES[1:])],
                    axis=1)
    return np.log(bins / tot + 1e-12)


def featurize(items, ctx):
    feats = []
    blocks = {c: ctx["load_eeg"](items, c) for c in ctx["conditions"]}
    for i in range(len(items)):
        f = []
        for c in ctx["conditions"]:
            x = blocks[c][i]
            if x is None:                      # missing condition -> fall back to first condition
                x = blocks[ctx["conditions"][0]][i]
            f.append(_relspec(x, ctx["fs"]).ravel())
        feats.append(np.concatenate(f))
    return np.asarray(feats, dtype=np.float32)


def _model(C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))


def fit_predict(F_train, y_train, F_test, ctx):
    F_train = np.asarray(F_train, dtype=np.float64)
    F_test = np.asarray(F_test, dtype=np.float64)
    y_train = np.asarray(y_train)
    groups = ctx.get("groups_train")
    if groups is None:
        groups = np.arange(len(y_train))
    groups = np.asarray(groups)

    best_C, best_auc = 0.01, -np.inf
    cv = GroupKFold(n_splits=4)
    for C in C_GRID:
        aucs = []
        for tr, va in cv.split(F_train, y_train, groups):
            if len(np.unique(y_train[va])) < 2:
                continue
            m = _model(C).fit(F_train[tr], y_train[tr])
            aucs.append(roc_auc_score(y_train[va], m.predict_proba(F_train[va])[:, 1]))
        if aucs and np.mean(aucs) > best_auc:
            best_auc, best_C = np.mean(aucs), C

    clf = _model(best_C).fit(F_train, y_train)
    return clf.predict_proba(F_test)[:, 1]