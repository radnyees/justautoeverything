"""Log relative band power (5 bands) + fine-grained log absolute PSD (1-Hz bins, 1-45 Hz) per channel,
from median-averaged Welch (artifact-robust); standardized, L2 logistic regression with C tuned by
inner grouped CV."""
import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
FINE = [(a, a + 1) for a in range(1, 45)]


def _spectral(x, fs):
    x = x - x.mean(axis=1, keepdims=True)
    f, p = welch(x, fs=fs, nperseg=int(2 * fs), axis=-1, average="median")
    p = p + 1e-12
    tot = p[:, (f >= 1) & (f <= 45)].sum(axis=1, keepdims=True)
    rel = np.log(np.stack([p[:, (f >= a) & (f < b)].sum(axis=1) for a, b in BANDS], axis=1) / tot)
    fine = np.log(np.stack([p[:, (f >= a) & (f < b)].mean(axis=1) for a, b in FINE], axis=1))
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
            f.append(_spectral(np.asarray(x, dtype=np.float64), ctx["fs"]))
        feats.append(np.concatenate(f))
    F = np.asarray(feats, dtype=np.float32)
    return np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)


def fit_predict(F_train, y_train, F_test, ctx):
    F_train = np.asarray(F_train, dtype=np.float64)
    F_test = np.asarray(F_test, dtype=np.float64)
    y_train = np.asarray(y_train)
    sc = StandardScaler().fit(F_train)
    Xtr, Xte = sc.transform(F_train), sc.transform(F_test)
    groups = np.asarray(ctx["groups_train"])
    cv = list(GroupKFold(n_splits=5).split(Xtr, y_train, groups))
    clf = LogisticRegressionCV(Cs=[0.0003, 0.001, 0.003, 0.01, 0.03, 0.1], cv=cv, scoring="roc_auc",
                               penalty="l2", max_iter=5000, n_jobs=-1).fit(Xtr, y_train)
    return clf.predict_proba(Xte)[:, 1]