"""Fine-resolution spectral profile: robust (median-Welch) log-PSD in 1-Hz bins 1-45 Hz per channel,
normalized by total power (relative spectrum) plus log total power per channel; standardized,
L2 logistic regression with C chosen by inner grouped CV."""
import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

EDGES = np.arange(1, 46)  # 1-Hz bins from 1 to 45 Hz


def _spectrum(x, fs):
    x = x - np.median(x, axis=1, keepdims=True)
    f, p = welch(x, fs=fs, nperseg=int(2 * fs), noverlap=int(fs), axis=-1, average="median")
    bins = np.stack([p[:, (f >= a) & (f < b)].mean(axis=1) for a, b in zip(EDGES[:-1], EDGES[1:])],
                    axis=1) + 1e-12
    tot = bins.sum(axis=1, keepdims=True)
    return np.concatenate([np.log(bins / tot).ravel(), np.log(tot).ravel()])


def featurize(items, ctx):
    feats = []
    blocks = {c: ctx["load_eeg"](items, c) for c in ctx["conditions"]}
    for i in range(len(items)):
        f = []
        for c in ctx["conditions"]:
            x = blocks[c][i]
            if x is None:                      # missing condition -> fall back to first condition
                x = blocks[ctx["conditions"][0]][i]
            f.append(_spectrum(np.asarray(x, dtype=np.float64), ctx["fs"]))
        feats.append(np.concatenate(f))
    return np.nan_to_num(np.asarray(feats, dtype=np.float32))


def fit_predict(F_train, y_train, F_test, ctx):
    sc = StandardScaler().fit(F_train)
    Xtr, Xte = sc.transform(F_train), sc.transform(F_test)
    groups = np.asarray(ctx["groups_train"])
    splits = list(GroupKFold(n_splits=4).split(Xtr, y_train, groups))
    clf = LogisticRegressionCV(Cs=[0.0003, 0.001, 0.003, 0.01, 0.03, 0.1], cv=splits,
                               scoring="roc_auc", max_iter=5000, n_jobs=-1).fit(Xtr, y_train)
    return clf.predict_proba(Xte)[:, 1]