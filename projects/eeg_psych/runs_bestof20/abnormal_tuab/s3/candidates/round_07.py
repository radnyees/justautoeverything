"""Log relative spectrum in 2-Hz bins (1-45 Hz) per channel from a median-averaged Welch PSD
(robust to transient artifacts), plus log total power per channel; standardized L2 logistic
regression with C chosen by grouped inner CV on the training fold."""
import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

EDGES = np.arange(1, 47, 2)  # 1-3, 3-5, ..., 43-45 Hz


def _spectrum(x, fs):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    f, p = welch(x, fs=fs, nperseg=int(2 * fs), axis=-1, average="median")
    bins = np.stack([p[:, (f >= a) & (f < b)].sum(axis=1) for a, b in zip(EDGES[:-1], EDGES[1:])],
                    axis=1) + 1e-12
    tot = bins.sum(axis=1, keepdims=True)
    rel = np.log(bins / tot)
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
    F = np.asarray(feats, dtype=np.float32)
    return np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)


def fit_predict(F_train, y_train, F_test, ctx):
    F_train = np.asarray(F_train)
    F_test = np.asarray(F_test)
    y_train = np.asarray(y_train)
    sc = StandardScaler().fit(F_train)
    Xtr, Xte = sc.transform(F_train), sc.transform(F_test)
    groups = ctx.get("groups_train")
    if groups is not None:
        cv = list(GroupKFold(n_splits=5).split(Xtr, y_train, np.asarray(groups)))
    else:
        cv = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(Xtr, y_train))
    clf = LogisticRegressionCV(Cs=np.logspace(-4, 0, 9), cv=cv, scoring="roc_auc",
                               max_iter=5000, n_jobs=1, random_state=0).fit(Xtr, y_train)
    return clf.predict_proba(Xte)[:, 1]