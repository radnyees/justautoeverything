"""Fine-resolution (1 Hz bins, 1-45 Hz) log relative spectrum per channel from median-averaged Welch
PSD (robust to transient artifacts), standardized, L2 logistic regression with C chosen by inner
grouped CV."""
import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

FMIN, FMAX = 1, 45


def _finespec(x, fs):
    x = np.asarray(x, dtype=np.float64)
    f, p = welch(x, fs=fs, nperseg=int(2 * fs), axis=-1, average="median")
    sel = (f >= FMIN) & (f < FMAX)
    tot = p[:, sel].sum(axis=1, keepdims=True) + 1e-12
    bins = np.stack([p[:, (f >= a) & (f < a + 1)].sum(axis=1) for a in range(FMIN, FMAX)], axis=1)
    return np.log(bins / tot + 1e-12)


def featurize(items, ctx):
    feats = []
    blocks = {c: ctx["load_eeg"](items, c) for c in ctx["conditions"]}
    for i in range(len(items)):
        f = []
        for c in ctx["conditions"]:
            x = blocks[c][i]
            if x is None:                      # missing condition -> fall back to first condition
                x = blocks[ctx["conditions"][0]][i]
            f.append(_finespec(x, ctx["fs"]).ravel())
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
        cv = list(GroupKFold(n_splits=5).split(Xtr, y_train, groups=np.asarray(groups)))
    else:
        cv = 5
    clf = LogisticRegressionCV(Cs=[0.0003, 0.001, 0.003, 0.01, 0.03, 0.1], cv=cv,
                               scoring="roc_auc", max_iter=5000, n_jobs=1)
    clf.fit(Xtr, y_train)
    return clf.predict_proba(Xte)[:, 1]