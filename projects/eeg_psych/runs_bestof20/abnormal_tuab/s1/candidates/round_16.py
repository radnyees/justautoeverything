"""High-resolution log absolute spectrum (1-Hz bins, 1-45 Hz) per channel from Welch PSD,
standardized, L2 logistic regression with C chosen by grouped inner CV (AUROC)."""
import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

FMIN, FMAX = 1, 45


def _logspec(x, fs):
    f, p = welch(x, fs=fs, nperseg=int(2 * fs), axis=-1)
    bins = [p[:, (f >= k) & (f < k + 1)].mean(axis=1) for k in range(FMIN, FMAX)]
    return np.log10(np.stack(bins, axis=1) + 1e-6)


def featurize(items, ctx):
    feats = []
    blocks = {c: ctx["load_eeg"](items, c) for c in ctx["conditions"]}
    for i in range(len(items)):
        f = []
        for c in ctx["conditions"]:
            x = blocks[c][i]
            if x is None:                      # missing condition -> fall back to first condition
                x = blocks[ctx["conditions"][0]][i]
            f.append(_logspec(np.asarray(x, dtype=np.float64), ctx["fs"]).ravel())
        feats.append(np.concatenate(f))
    return np.asarray(feats, dtype=np.float32)


def fit_predict(F_train, y_train, F_test, ctx):
    y_train = np.asarray(y_train)
    sc = StandardScaler().fit(F_train)
    Xtr, Xte = sc.transform(F_train), sc.transform(F_test)
    groups = ctx.get("groups_train")
    if groups is not None:
        cv = list(GroupKFold(n_splits=5).split(Xtr, y_train, groups))
    else:
        cv = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(Xtr, y_train))
    clf = LogisticRegressionCV(Cs=np.logspace(-4, 0, 9), cv=cv, scoring="roc_auc",
                               max_iter=5000, n_jobs=-1).fit(Xtr, y_train)
    return clf.predict_proba(Xte)[:, 1]