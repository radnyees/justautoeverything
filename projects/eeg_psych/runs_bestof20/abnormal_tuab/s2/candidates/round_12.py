"""Fine-resolution log absolute spectrum (1 Hz bins, 1-45 Hz) per channel from a median-averaged
Welch PSD (robust to transient artifacts), standardized, L2 logistic regression with C chosen by
grouped inner CV."""
import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

EDGES = np.arange(1, 46)  # 1 Hz bins from 1 to 45 Hz


def _logspec(x, fs):
    x = np.asarray(x, dtype=np.float64)
    f, p = welch(x, fs=fs, nperseg=int(2 * fs), noverlap=int(fs), axis=-1, average="median")
    bins = [p[:, (f >= a) & (f < b)].mean(axis=1) for a, b in zip(EDGES[:-1], EDGES[1:])]
    return np.log(np.stack(bins, axis=1) + 1e-6)


def featurize(items, ctx):
    feats = []
    blocks = {c: ctx["load_eeg"](items, c) for c in ctx["conditions"]}
    for i in range(len(items)):
        f = []
        for c in ctx["conditions"]:
            x = blocks[c][i]
            if x is None:                      # missing condition -> fall back to first condition
                x = blocks[ctx["conditions"][0]][i]
            f.append(_logspec(x, ctx["fs"]).ravel())
        feats.append(np.concatenate(f))
    return np.asarray(feats, dtype=np.float32)


def fit_predict(F_train, y_train, F_test, ctx):
    F_train = np.nan_to_num(np.asarray(F_train, dtype=np.float64))
    F_test = np.nan_to_num(np.asarray(F_test, dtype=np.float64))
    y_train = np.asarray(y_train)
    sc = StandardScaler().fit(F_train)
    Xtr, Xte = sc.transform(F_train), sc.transform(F_test)
    groups = ctx.get("groups_train")
    if groups is not None:
        cv = list(GroupKFold(n_splits=4).split(Xtr, y_train, np.asarray(groups)))
    else:
        cv = list(StratifiedKFold(n_splits=4, shuffle=True, random_state=0).split(Xtr, y_train))
    clf = LogisticRegressionCV(Cs=[3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1], cv=cv, scoring="roc_auc",
                               max_iter=5000, n_jobs=1).fit(Xtr, y_train)
    return clf.predict_proba(Xte)[:, 1]