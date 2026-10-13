"""Log relative PSD at 1-Hz resolution (1-45 Hz, median-Welch for artifact robustness) per channel
plus per-channel log total power, standardized, L2 logistic regression with C tuned by grouped
inner CV."""
import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

FMIN, FMAX = 1.0, 45.0


def _spectrum(x, fs):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    f, p = welch(x, fs=fs, nperseg=int(fs), noverlap=int(fs) // 2, axis=-1, average="median")
    m = (f >= FMIN) & (f <= FMAX)
    p = p[:, m] + 1e-12
    tot = p.sum(axis=1, keepdims=True)
    rel = np.log(p / tot)                       # spectral shape, scale-invariant
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
    return np.asarray(feats, dtype=np.float32)


def fit_predict(F_train, y_train, F_test, ctx):
    F_train = np.nan_to_num(np.asarray(F_train, dtype=np.float64))
    F_test = np.nan_to_num(np.asarray(F_test, dtype=np.float64))
    y_train = np.asarray(y_train)
    sc = StandardScaler().fit(F_train)
    Xtr, Xte = sc.transform(F_train), sc.transform(F_test)

    groups = ctx.get("groups_train")
    if groups is not None and len(np.unique(groups)) >= 5:
        cv = list(GroupKFold(n_splits=5).split(Xtr, y_train, np.asarray(groups)))
    else:
        cv = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(Xtr, y_train))

    clf = LogisticRegressionCV(Cs=np.logspace(-4, -0.5, 8), cv=cv, scoring="roc_auc",
                               penalty="l2", max_iter=5000, n_jobs=-1, random_state=0)
    clf.fit(Xtr, y_train)
    return clf.predict_proba(Xte)[:, 1]