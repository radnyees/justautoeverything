"""Fine-resolution (1 Hz bins, 1-45 Hz) log relative PSD per channel from median-averaged Welch
(robust to transient artifacts, scale-invariant), standardized, L2 logistic regression with C chosen
by grouped inner CV on AUROC."""
import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

FMIN, FMAX = 1, 45


def _finepsd(x, fs):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    f, p = welch(x, fs=fs, nperseg=int(2 * fs), noverlap=int(fs), axis=-1, average="median")
    tot = p[:, (f >= FMIN) & (f < FMAX)].sum(axis=1, keepdims=True) + 1e-12
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
            f.append(_finepsd(x, ctx["fs"]).ravel())
        feats.append(np.concatenate(f))
    F = np.asarray(feats, dtype=np.float32)
    return np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)


def fit_predict(F_train, y_train, F_test, ctx):
    y_train = np.asarray(y_train)
    sc = StandardScaler().fit(F_train)
    Xtr, Xte = sc.transform(F_train), sc.transform(F_test)
    groups = ctx.get("groups_train")
    if groups is None:
        groups = np.arange(len(y_train))
    splits = list(GroupKFold(n_splits=5).split(Xtr, y_train, groups))
    clf = LogisticRegressionCV(Cs=[0.001, 0.003, 0.01, 0.03, 0.1], cv=splits, scoring="roc_auc",
                               penalty="l2", max_iter=5000, n_jobs=1).fit(Xtr, y_train)
    return clf.predict_proba(Xte)[:, 1]