"""Fine-resolution (1 Hz bins, 1-45 Hz) relative log-spectrum per channel from a median-averaged
Welch PSD (robust to transient artifacts), plus scale-invariant spatial power distribution
(per-channel log total power minus recording mean). Standardized, L2 logistic regression with
C tuned by grouped inner CV (AUROC)."""
import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EDGES = np.arange(1, 46)          # 1 Hz bins: [1,2), ..., [44,45)
CHUNK = 128


def _spectral_feats(x, fs):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    f, p = welch(x, fs=fs, nperseg=int(4 * fs), noverlap=int(2 * fs), axis=-1, average="median")
    bins = np.stack([p[:, (f >= a) & (f < a + 1)].mean(axis=1) for a in EDGES[:-1]], axis=1) + 1e-12
    tot = bins.sum(axis=1, keepdims=True)
    rel = np.log(bins / tot)                              # channels x 44, scale-free
    ltot = np.log(tot[:, 0])
    spatial = ltot - ltot.mean()                          # channels, scale-free
    return np.concatenate([rel.ravel(), spatial])


def featurize(items, ctx):
    fs = ctx["fs"]
    conds = ctx["conditions"]
    feats = []
    for s in range(0, len(items), CHUNK):
        sub = items[s:s + CHUNK]
        blocks = {c: ctx["load_eeg"](sub, c) for c in conds}
        for i in range(len(sub)):
            f = []
            for c in conds:
                x = blocks[c][i]
                if x is None:                      # missing condition -> fall back to first condition
                    x = blocks[conds[0]][i]
                f.append(_spectral_feats(x, fs))
            feats.append(np.concatenate(f))
        del blocks
    F = np.asarray(feats, dtype=np.float32)
    return np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)


def fit_predict(F_train, y_train, F_test, ctx):
    y_train = np.asarray(y_train)
    groups = ctx.get("groups_train")
    if groups is None:
        groups = np.arange(len(y_train))
    cv = list(GroupKFold(n_splits=5).split(F_train, y_train, groups))
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegressionCV(Cs=np.logspace(-4, -1, 7), cv=cv, scoring="roc_auc",
                             max_iter=5000, penalty="l2"),
    )
    clf.fit(F_train, y_train)
    return clf.predict_proba(F_test)[:, 1]