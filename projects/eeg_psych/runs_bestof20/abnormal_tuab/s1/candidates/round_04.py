"""Fine-resolution log relative spectrum (2-Hz bins, 1-45 Hz) per channel from Welch PSD of each
available condition, standardized, L2 logistic regression with C tuned by grouped inner CV."""
import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

EDGES = np.arange(1, 46, 2)                     # 1,3,...,45 -> 22 bins of 2 Hz
BANDS = list(zip(EDGES[:-1], EDGES[1:]))


def _bandpower(x, fs):
    f, p = welch(x, fs=fs, nperseg=int(2 * fs), axis=-1)
    tot = p[:, (f >= 1) & (f <= 45)].sum(axis=1, keepdims=True) + 1e-12
    bp = np.stack([p[:, (f >= a) & (f < b)].sum(axis=1) for a, b in BANDS], axis=1)
    return np.log(bp / tot + 1e-12)


def featurize(items, ctx):
    feats = []
    blocks = {c: ctx["load_eeg"](items, c) for c in ctx["conditions"]}
    for i in range(len(items)):
        f = []
        for c in ctx["conditions"]:
            x = blocks[c][i]
            if x is None:                      # missing condition -> fall back to first condition
                x = blocks[ctx["conditions"][0]][i]
            f.append(_bandpower(np.asarray(x, dtype=np.float64), ctx["fs"]).ravel())
        feats.append(np.concatenate(f))
    return np.asarray(feats, dtype=np.float32)


def fit_predict(F_train, y_train, F_test, ctx):
    F_train = np.asarray(F_train, dtype=np.float64)
    F_test = np.asarray(F_test, dtype=np.float64)
    y_train = np.asarray(y_train)
    sc = StandardScaler().fit(F_train)
    Xtr, Xte = sc.transform(F_train), sc.transform(F_test)
    groups = ctx.get("groups_train")
    if groups is not None:
        cv = list(GroupKFold(n_splits=5).split(Xtr, y_train, np.asarray(groups)))
    else:
        cv = 5
    clf = LogisticRegressionCV(Cs=[0.002, 0.005, 0.01, 0.03, 0.1, 0.3], cv=cv, scoring="roc_auc",
                               penalty="l2", solver="lbfgs", max_iter=5000, n_jobs=1)
    clf.fit(Xtr, y_train)
    return clf.predict_proba(Xte)[:, 1]