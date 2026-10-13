"""Fine-resolution log relative spectrum (2-Hz bins, 1-45 Hz) + log absolute total power per channel
from Welch PSD, standardized, L2 logistic regression with C chosen by inner grouped CV."""
import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

EDGES = np.arange(1, 47, 2)  # 1-3, 3-5, ..., 43-45 Hz
BINS = list(zip(EDGES[:-1], EDGES[1:]))


def _spectral(x, fs):
    f, p = welch(x, fs=fs, nperseg=int(4 * fs), noverlap=int(2 * fs), axis=-1)
    tot = p[:, (f >= 1) & (f < 45)].sum(axis=1) + 1e-12
    rel = np.stack([p[:, (f >= a) & (f < b)].sum(axis=1) for a, b in BINS], axis=1) / tot[:, None]
    return np.concatenate([np.log(rel + 1e-12).ravel(), np.log(tot)])


def featurize(items, ctx):
    feats = []
    blocks = {c: ctx["load_eeg"](items, c) for c in ctx["conditions"]}
    for i in range(len(items)):
        f = []
        for c in ctx["conditions"]:
            x = blocks[c][i]
            if x is None:                      # missing condition -> fall back to first condition
                x = blocks[ctx["conditions"][0]][i]
            f.append(_spectral(np.asarray(x, dtype=np.float64), ctx["fs"]))
        feats.append(np.concatenate(f))
    return np.asarray(feats, dtype=np.float32)


def fit_predict(F_train, y_train, F_test, ctx):
    sc = StandardScaler().fit(F_train)
    Xtr, Xte = sc.transform(F_train), sc.transform(F_test)
    groups = ctx.get("groups_train")
    if groups is None:
        groups = np.arange(len(y_train))
    cv = list(GroupKFold(n_splits=5).split(Xtr, y_train, groups))
    clf = LogisticRegressionCV(Cs=[0.001, 0.003, 0.01, 0.03, 0.1], cv=cv, scoring="roc_auc",
                               max_iter=5000, n_jobs=1).fit(Xtr, y_train)
    return clf.predict_proba(Xte)[:, 1]