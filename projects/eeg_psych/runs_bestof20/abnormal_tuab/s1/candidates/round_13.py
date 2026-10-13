"""Full-resolution spectral fingerprint: per channel, log Welch PSD (1 Hz bins, 1-45 Hz, median
across segments for artifact robustness), both absolute and total-power-normalized (relative),
standardized, L2 logistic regression with C tuned by grouped inner CV."""
import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

FMIN, FMAX = 1, 45


def _spectrum(x, fs):
    x = np.nan_to_num(np.asarray(x, dtype=np.float64))
    nper = int(fs)                                   # 1 s windows -> 1 Hz resolution
    f, p = welch(x, fs=fs, nperseg=nper, noverlap=nper // 2, axis=-1, average="median")
    m = (f >= FMIN) & (f <= FMAX)
    p = p[:, m] + 1e-12
    absl = np.log(p)
    rel = np.log(p / p.sum(axis=1, keepdims=True))
    return np.concatenate([absl.ravel(), rel.ravel()])


def featurize(items, ctx):
    feats = []
    conds = ctx["conditions"]
    blocks = {c: ctx["load_eeg"](items, c) for c in conds}
    for i in range(len(items)):
        f = []
        for c in conds:
            x = blocks[c][i]
            if x is None:
                x = next((blocks[d][i] for d in conds if blocks[d][i] is not None), None)
            if x is None:
                n_ch = len(ctx["channels"])
                f.append(np.zeros(2 * n_ch * (FMAX - FMIN + 1)))
            else:
                f.append(_spectrum(x, ctx["fs"]))
        feats.append(np.concatenate(f))
    F = np.asarray(feats, dtype=np.float32)
    return np.nan_to_num(F)


def fit_predict(F_train, y_train, F_test, ctx):
    y_train = np.asarray(y_train)
    sc = StandardScaler().fit(F_train)
    Xtr, Xte = sc.transform(F_train), sc.transform(F_test)
    groups = ctx.get("groups_train")
    if groups is None:
        groups = np.arange(len(y_train))
    splits = list(GroupKFold(n_splits=4).split(Xtr, y_train, groups))
    clf = LogisticRegressionCV(Cs=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2], cv=splits,
                               scoring="roc_auc", max_iter=5000, n_jobs=-1).fit(Xtr, y_train)
    return clf.predict_proba(Xte)[:, 1]