"""Log relative spectrum in 1-Hz bins (1-45 Hz) per channel from median-averaged Welch PSD
(robust to transient artifacts), standardized, L2 logistic regression with C tuned by inner
grouped CV."""
import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

FMIN, FMAX = 1.0, 45.0


def _spectrum(x, fs):
    x = np.asarray(x, dtype=np.float64)
    f, p = welch(x, fs=fs, nperseg=int(2 * fs), axis=-1, average="median")
    sel = (f >= FMIN) & (f < FMAX)
    tot = p[:, sel].sum(axis=1, keepdims=True) + 1e-20
    edges = np.arange(FMIN, FMAX + 1e-9, 1.0)
    bins = [p[:, (f >= a) & (f < b)].sum(axis=1) for a, b in zip(edges[:-1], edges[1:])]
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


def fit_predict(F_train, y_train, F_test, ctx):
    y_train = np.asarray(y_train)
    sc = StandardScaler().fit(F_train)
    Xtr, Xte = sc.transform(F_train), sc.transform(F_test)
    groups = ctx.get("groups_train")
    if groups is not None:
        cv = list(GroupKFold(n_splits=4).split(Xtr, y_train, groups=np.asarray(groups)))
        cvm = LogisticRegressionCV(Cs=[0.001, 0.003, 0.01, 0.03, 0.1], cv=cv, scoring="roc_auc",
                                   max_iter=5000, n_jobs=-1).fit(Xtr, y_train)
        C = float(cvm.C_[0])
    else:
        C = 0.01
    clf = LogisticRegression(C=C, max_iter=5000).fit(Xtr, y_train)
    return clf.predict_proba(Xte)[:, 1]