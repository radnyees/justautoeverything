"""Log relative band power (5 bands) + fine-resolution log absolute PSD (1-Hz bins, 1-45 Hz) per
channel from Welch PSD, standardized, L2 logistic regression with C tuned by grouped inner CV."""
import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
FINE_EDGES = np.arange(1, 46)  # 1-Hz bins from 1 to 45 Hz


def _features(x, fs):
    x = np.asarray(x, dtype=np.float64)
    f, p = welch(x, fs=fs, nperseg=int(4 * fs), axis=-1)
    tot = p[:, (f >= 1) & (f <= 45)].sum(axis=1, keepdims=True) + 1e-12
    rel = np.log(np.stack([p[:, (f >= a) & (f < b)].sum(axis=1) for a, b in BANDS], axis=1) / tot + 1e-12)
    fine = np.log(np.stack([p[:, (f >= a) & (f < a + 1)].mean(axis=1)
                            for a in FINE_EDGES[:-1]], axis=1) + 1e-12)
    return np.concatenate([rel.ravel(), fine.ravel()])


def featurize(items, ctx):
    feats = []
    blocks = {c: ctx["load_eeg"](items, c) for c in ctx["conditions"]}
    for i in range(len(items)):
        f = []
        for c in ctx["conditions"]:
            x = blocks[c][i]
            if x is None:                      # missing condition -> fall back to first condition
                x = blocks[ctx["conditions"][0]][i]
            f.append(_features(x, ctx["fs"]))
        feats.append(np.concatenate(f))
    F = np.asarray(feats, dtype=np.float32)
    return np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)


def fit_predict(F_train, y_train, F_test, ctx):
    F_train = np.asarray(F_train, dtype=np.float64)
    F_test = np.asarray(F_test, dtype=np.float64)
    y_train = np.asarray(y_train)
    sc = StandardScaler().fit(F_train)
    Xtr, Xte = sc.transform(F_train), sc.transform(F_test)
    groups = ctx.get("groups_train")
    if groups is not None:
        splits = list(GroupKFold(n_splits=5).split(Xtr, y_train, np.asarray(groups)))
    else:
        splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(Xtr, y_train))
    clf = LogisticRegressionCV(Cs=np.logspace(-4, 0, 9), cv=splits, scoring="roc_auc",
                               max_iter=5000, n_jobs=-1).fit(Xtr, y_train)
    return clf.predict_proba(Xte)[:, 1]