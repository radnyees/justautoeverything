"""Fine-grained spectral shape: per-channel log-relative PSD in 1-Hz bins (1-45 Hz) plus log total
power, from Welch PSD of each available condition; standardized L2 logistic regression with C tuned
by inner grouped CV."""
import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FMIN, FMAX = 1, 45


def _spectrum(x, fs):
    f, p = welch(x, fs=fs, nperseg=int(2 * fs), axis=-1)
    bins = np.stack([p[:, (f >= a) & (f < a + 1)].sum(axis=1) for a in range(FMIN, FMAX)], axis=1)
    tot = bins.sum(axis=1, keepdims=True) + 1e-12
    return np.concatenate([np.log(bins / tot + 1e-12), np.log(tot)], axis=1)


def featurize(items, ctx):
    feats = []
    blocks = {c: ctx["load_eeg"](items, c) for c in ctx["conditions"]}
    for i in range(len(items)):
        f = []
        for c in ctx["conditions"]:
            x = blocks[c][i]
            if x is None:                      # missing condition -> fall back to first condition
                x = blocks[ctx["conditions"][0]][i]
            f.append(_spectrum(np.asarray(x, dtype=np.float64), ctx["fs"]).ravel())
        feats.append(np.concatenate(f))
    return np.asarray(feats, dtype=np.float32)


def fit_predict(F_train, y_train, F_test, ctx):
    F_train = np.asarray(F_train, dtype=np.float64)
    F_test = np.asarray(F_test, dtype=np.float64)
    y_train = np.asarray(y_train)
    groups = ctx.get("groups_train")
    if groups is None:
        groups = np.arange(len(y_train))
    splits = list(GroupKFold(n_splits=5).split(F_train, y_train, groups))
    sc = StandardScaler().fit(F_train)
    clf = LogisticRegressionCV(Cs=np.logspace(-4, 0, 9), cv=splits, scoring="roc_auc",
                               max_iter=5000, n_jobs=-1).fit(sc.transform(F_train), y_train)
    return clf.predict_proba(sc.transform(F_test))[:, 1]