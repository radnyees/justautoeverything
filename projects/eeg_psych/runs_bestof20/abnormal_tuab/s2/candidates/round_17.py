"""Fine-resolution (2 Hz bins, 1-45 Hz) log relative spectrum + log total power per channel from a
median-averaged Welch PSD (robust to transient artifacts), standardized, L2 logistic regression with
C tuned by grouped inner CV."""
import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EDGES = np.arange(1, 46, 2)  # 1-3, 3-5, ..., 43-45 Hz


def _spectrum(x, fs):
    f, p = welch(x, fs=fs, nperseg=int(2 * fs), axis=-1, average="median")
    bins = np.stack([p[:, (f >= a) & (f < b)].mean(axis=1) for a, b in zip(EDGES[:-1], EDGES[1:])],
                    axis=1) + 1e-12
    tot = p[:, (f >= 1) & (f <= 45)].sum(axis=1, keepdims=True) + 1e-12
    rel = np.log(bins) - np.log(tot)                      # spectral shape (scale-free)
    return np.concatenate([rel, np.log(tot)], axis=1)     # + overall amplitude per channel


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
    F = np.asarray(feats, dtype=np.float32)
    return np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)


def fit_predict(F_train, y_train, F_test, ctx):
    groups = ctx.get("groups_train")
    if groups is not None:
        cv = list(GroupKFold(n_splits=5).split(F_train, y_train, groups))
    else:
        cv = 5
    sc = StandardScaler().fit(F_train)
    clf = LogisticRegressionCV(Cs=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3], cv=cv, scoring="roc_auc",
                               max_iter=5000, n_jobs=-1).fit(sc.transform(F_train), y_train)
    return clf.predict_proba(sc.transform(F_test))[:, 1]