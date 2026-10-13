"""Fine-resolution (1 Hz bins, 1-45 Hz) log relative spectrum + log total power per channel from a
median-averaged (artifact-robust) Welch PSD; standardized L2 logistic regression with C chosen by
inner grouped CV (AUROC)."""
import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

EDGES = np.arange(1, 46, 1.0)  # 1 Hz bins from 1 to 45 Hz


def _spectrum(x, fs):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    f, p = welch(x, fs=fs, nperseg=int(2 * fs), axis=-1, average="median")
    bins = np.stack([p[:, (f >= a) & (f < b)].sum(axis=1)
                     for a, b in zip(EDGES[:-1], EDGES[1:])], axis=1) + 1e-12
    tot = bins.sum(axis=1, keepdims=True)
    rel = np.log(bins / tot)                       # spectral shape (scale-free)
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
    return np.nan_to_num(np.asarray(feats, dtype=np.float32))


def fit_predict(F_train, y_train, F_test, ctx):
    F_train = np.asarray(F_train, dtype=np.float64)
    F_test = np.asarray(F_test, dtype=np.float64)
    y_train = np.asarray(y_train)
    sc = StandardScaler().fit(F_train)
    Xtr, Xte = sc.transform(F_train), sc.transform(F_test)
    groups = ctx.get("groups_train")
    if groups is None:
        groups = np.arange(len(y_train))
    cv = list(GroupKFold(n_splits=5).split(Xtr, y_train, groups))
    clf = LogisticRegressionCV(Cs=[0.0003, 0.001, 0.003, 0.01, 0.03, 0.1], cv=cv,
                               scoring="roc_auc", max_iter=5000, n_jobs=-1)
    clf.fit(Xtr, y_train)
    return clf.predict_proba(Xte)[:, 1]