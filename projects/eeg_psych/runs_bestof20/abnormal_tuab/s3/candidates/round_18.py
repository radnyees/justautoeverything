"""Fine-grained robust spectrum: per-channel log absolute PSD in 2-Hz bins (1-45 Hz) from median-Welch
(artifact-robust), standardized, L2 logistic regression with C tuned by inner grouped CV."""
import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

EDGES = np.arange(1, 47, 2)  # 1-3, 3-5, ..., 43-45 Hz -> 22 bins
CHUNK = 200


def _spectrum(x, fs):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    f, p = welch(x, fs=fs, nperseg=int(2 * fs), noverlap=int(fs), axis=-1, average="median")
    bins = [p[:, (f >= a) & (f < b)].mean(axis=1) for a, b in zip(EDGES[:-1], EDGES[1:])]
    return np.log(np.stack(bins, axis=1) + 1e-6)


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
                f.append(_spectrum(x, fs).ravel())
            feats.append(np.concatenate(f))
        del blocks
    return np.asarray(feats, dtype=np.float32)


def fit_predict(F_train, y_train, F_test, ctx):
    F_train = np.nan_to_num(np.asarray(F_train, dtype=np.float64))
    F_test = np.nan_to_num(np.asarray(F_test, dtype=np.float64))
    sc = StandardScaler().fit(F_train)
    Xtr, Xte = sc.transform(F_train), sc.transform(F_test)
    groups = np.asarray(ctx["groups_train"])
    cv = list(GroupKFold(n_splits=5).split(Xtr, y_train, groups))
    clf = LogisticRegressionCV(Cs=np.logspace(-4, 0, 9), cv=cv, scoring="roc_auc",
                               max_iter=5000, n_jobs=1).fit(Xtr, y_train)
    return clf.predict_proba(Xte)[:, 1]