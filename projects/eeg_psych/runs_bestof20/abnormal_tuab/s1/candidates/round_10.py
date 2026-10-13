"""Log relative spectrum in 1-Hz bins (1-45 Hz) per channel + log total power per channel, from a
median-averaged Welch PSD (robust to transient artifacts); standardized, L2 logistic regression
with C chosen by grouped inner CV."""
import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

FMIN, FMAX = 1, 45
CHUNK = 200


def _spectrum(x, fs):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    f, p = welch(x, fs=fs, nperseg=int(2 * fs), noverlap=int(fs), average="median", axis=-1)
    edges = np.arange(FMIN, FMAX + 1)
    bins = np.stack([p[:, (f >= a) & (f < a + 1)].mean(axis=1) for a in edges[:-1]], axis=1)
    tot = bins.sum(axis=1, keepdims=True) + 1e-12
    rel = np.log(bins / tot + 1e-12)
    return np.concatenate([rel.ravel(), np.log(tot.ravel())])


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
                f.append(_spectrum(x, fs))
            feats.append(np.concatenate(f))
        del blocks
    F = np.asarray(feats, dtype=np.float32)
    return np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)


def fit_predict(F_train, y_train, F_test, ctx):
    F_train = np.asarray(F_train, dtype=np.float64)
    F_test = np.asarray(F_test, dtype=np.float64)
    y_train = np.asarray(y_train)
    sc = StandardScaler().fit(F_train)
    Xtr, Xte = sc.transform(F_train), sc.transform(F_test)
    groups = ctx.get("groups_train")
    if groups is None:
        groups = np.arange(len(y_train))
    splits = list(GroupKFold(n_splits=5).split(Xtr, y_train, groups))
    clf = LogisticRegressionCV(Cs=[1e-3, 3e-3, 1e-2, 3e-2, 1e-1], cv=splits, scoring="roc_auc",
                               max_iter=5000, n_jobs=1).fit(Xtr, y_train)
    return clf.predict_proba(Xte)[:, 1]