"""Fine-grained relative log-spectrum: per channel, Welch PSD (4 s windows) averaged into 1-Hz bins
from 1-45 Hz and normalized by total 1-45 Hz power (amplitude-scale invariant). This captures the
exact frequency of slowing, the alpha peak's position and the spectral slope, not just 5 coarse bands.
Standardized, L2 logistic regression with C chosen by grouped inner CV (AUROC)."""
import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

EDGES = np.arange(1, 46, 1)  # 1-Hz bins, 1..45 Hz -> 44 bins


def _logspec(x, fs):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    f, p = welch(x, fs=fs, nperseg=int(4 * fs), axis=-1)
    m = (f >= EDGES[0]) & (f < EDGES[-1])
    tot = p[:, m].sum(axis=1, keepdims=True) + 1e-12
    bins = np.stack([p[:, (f >= a) & (f < b)].sum(axis=1) for a, b in zip(EDGES[:-1], EDGES[1:])],
                    axis=1)
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
            f.append(_logspec(x, ctx["fs"]).ravel())
        feats.append(np.concatenate(f))
    return np.asarray(feats, dtype=np.float32)


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
    clf = LogisticRegressionCV(Cs=[1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1], cv=splits,
                               scoring="roc_auc", max_iter=5000, n_jobs=-1).fit(Xtr, y_train)
    return clf.predict_proba(Xte)[:, 1]