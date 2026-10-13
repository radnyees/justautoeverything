"""Per-channel robust log spectrum (median-Welch, 1-Hz bins 1-45 Hz, normalized to total power) plus
log total power per channel, standardized, L2 logistic regression with C chosen by grouped inner CV."""
import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

FMIN, FMAX = 1, 45


def _spectrum(x, fs):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    f, p = welch(x, fs=fs, nperseg=int(4 * fs), noverlap=int(2 * fs), average="median", axis=-1)
    bins = []
    for lo in range(FMIN, FMAX):                      # 1-Hz bins: [1,2), ..., [44,45)
        m = (f >= lo) & (f < lo + 1)
        bins.append(p[:, m].mean(axis=1))
    s = np.stack(bins, axis=1) + 1e-12                # channels x 44
    tot = s.sum(axis=1, keepdims=True)
    rel = np.log(s / tot)                             # spectral shape per channel
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
    F = np.asarray(feats, dtype=np.float32)
    return np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)


def fit_predict(F_train, y_train, F_test, ctx):
    sc = StandardScaler().fit(F_train)
    Xtr, Xte = sc.transform(F_train), sc.transform(F_test)
    groups = np.asarray(ctx["groups_train"])
    splits = list(GroupKFold(n_splits=4).split(Xtr, y_train, groups))
    clf = LogisticRegressionCV(Cs=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1], cv=splits,
                               scoring="roc_auc", max_iter=5000, n_jobs=-1).fit(Xtr, y_train)
    return clf.predict_proba(Xte)[:, 1]