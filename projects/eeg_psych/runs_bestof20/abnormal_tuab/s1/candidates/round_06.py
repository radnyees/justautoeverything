"""Log absolute PSD in 1-Hz bins (1-45 Hz) + log relative band power per channel, from a
median-averaged Welch PSD (robust to transient artifacts); standardized L2 logistic regression
with C chosen by grouped inner CV on the training fold."""
import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
BINS = [(a, a + 1) for a in range(1, 45)]
C_GRID = [0.001, 0.003, 0.01, 0.03, 0.1]


def _spectral(x, fs):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    f, p = welch(x, fs=fs, nperseg=int(2 * fs), axis=-1, average="median")
    tot = p[:, (f >= 1) & (f <= 45)].sum(axis=1, keepdims=True) + 1e-12
    rel = np.log(np.stack([p[:, (f >= a) & (f < b)].sum(axis=1) for a, b in BANDS], axis=1) / tot + 1e-12)
    absb = np.log(np.stack([p[:, (f >= a) & (f < b)].mean(axis=1) for a, b in BINS], axis=1) + 1e-12)
    return np.concatenate([rel.ravel(), absb.ravel()])


def featurize(items, ctx):
    feats = []
    blocks = {c: ctx["load_eeg"](items, c) for c in ctx["conditions"]}
    for i in range(len(items)):
        f = []
        for c in ctx["conditions"]:
            x = blocks[c][i]
            if x is None:                      # missing condition -> fall back to first condition
                x = blocks[ctx["conditions"][0]][i]
            f.append(_spectral(x, ctx["fs"]))
        feats.append(np.concatenate(f))
    F = np.asarray(feats, dtype=np.float32)
    return np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)


def _model(C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))


def fit_predict(F_train, y_train, F_test, ctx):
    F_train = np.asarray(F_train, dtype=np.float64)
    F_test = np.asarray(F_test, dtype=np.float64)
    y_train = np.asarray(y_train)
    groups = np.asarray(ctx["groups_train"])
    best_C, best_s = C_GRID[2], -np.inf
    cv = GroupKFold(n_splits=3)
    for C in C_GRID:
        scores = []
        for tr, va in cv.split(F_train, y_train, groups):
            if len(np.unique(y_train[va])) < 2:
                continue
            m = _model(C).fit(F_train[tr], y_train[tr])
            scores.append(roc_auc_score(y_train[va], m.predict_proba(F_train[va])[:, 1]))
        s = np.mean(scores) if scores else -np.inf
        if s > best_s:
            best_s, best_C = s, C
    clf = _model(best_C).fit(F_train, y_train)
    return clf.predict_proba(F_test)[:, 1]