"""Relative log-spectrum (2-Hz bins, 1-45 Hz) per channel from median-averaged Welch PSD (robust to
transient artifacts), standardized, L2 logistic regression with C tuned by patient-grouped inner CV."""
import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

EDGES = np.arange(1, 47, 2)  # 1-3, 3-5, ..., 43-45 Hz


def _spectrum(x, fs):
    f, p = welch(x, fs=fs, nperseg=int(4 * fs), noverlap=int(2 * fs), axis=-1, average="median")
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
            f.append(_spectrum(np.asarray(x, dtype=np.float64), ctx["fs"]).ravel())
        feats.append(np.concatenate(f))
    return np.asarray(feats, dtype=np.float32)


def fit_predict(F_train, y_train, F_test, ctx):
    sc = StandardScaler().fit(F_train)
    Xtr, Xte = sc.transform(F_train), sc.transform(F_test)
    y = np.asarray(y_train)
    groups = ctx.get("groups_train")
    if groups is None:
        groups = np.arange(len(y))
    cv = list(GroupKFold(n_splits=5).split(Xtr, y, groups))
    clf = LogisticRegressionCV(Cs=np.logspace(-4, 0, 9), cv=cv, scoring="roc_auc",
                               max_iter=5000, n_jobs=-1).fit(Xtr, y)
    return clf.predict_proba(Xte)[:, 1]