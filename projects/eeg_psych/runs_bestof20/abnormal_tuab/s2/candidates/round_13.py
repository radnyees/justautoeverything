"""Log relative power spectrum (1-Hz bins, 1-45 Hz, median Welch for artifact robustness) plus log total
power per channel; standardized L2 logistic regression with C chosen by grouped inner CV."""
import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

FMIN, FMAX = 1, 45
CHUNK = 200


def _spectrum(x, fs):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    f, p = welch(x, fs=fs, nperseg=int(2 * fs), noverlap=int(fs), axis=-1, average="median")
    edges = np.arange(FMIN, FMAX + 1)
    bins = np.stack([p[:, (f >= a) & (f < a + 1)].mean(axis=1) for a in edges[:-1]], axis=1) + 1e-12
    tot = bins.sum(axis=1, keepdims=True)
    return np.concatenate([np.log(bins / tot).ravel(), np.log(tot).ravel()])


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
    return np.asarray(feats, dtype=np.float32)


def fit_predict(F_train, y_train, F_test, ctx):
    sc = StandardScaler().fit(F_train)
    Xtr, Xte = sc.transform(F_train), sc.transform(F_test)
    groups = ctx.get("groups_train")
    Cs = [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]
    if groups is not None and len(np.unique(groups)) >= 5:
        splits = list(GroupKFold(n_splits=5).split(Xtr, y_train, groups))
        clf = LogisticRegressionCV(Cs=Cs, cv=splits, scoring="roc_auc", max_iter=5000,
                                   n_jobs=-1).fit(Xtr, y_train)
    else:
        clf = LogisticRegression(C=0.01, max_iter=5000).fit(Xtr, y_train)
    return clf.predict_proba(Xte)[:, 1]