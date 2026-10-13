"""Log relative band power (delta/theta/alpha/beta/gamma) per channel from a robust (median-of-segments)
Welch PSD of each available condition after common-average re-referencing, standardized,
L2 logistic regression."""
import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]


def _bandpower(x, fs):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)      # common average reference (reference-independent)
    f, p = welch(x, fs=fs, nperseg=int(2 * fs), axis=-1, average="median")  # robust to transient artifacts
    tot = p[:, (f >= 1) & (f <= 45)].sum(axis=1, keepdims=True) + 1e-12
    bp = np.stack([p[:, (f >= a) & (f < b)].sum(axis=1) for a, b in BANDS], axis=1)
    return np.log(bp / tot + 1e-12)


def featurize(items, ctx):
    feats = []
    blocks = {c: ctx["load_eeg"](items, c) for c in ctx["conditions"]}
    for i in range(len(items)):
        f = []
        for c in ctx["conditions"]:
            x = blocks[c][i]
            if x is None:                      # missing condition -> fall back to another available one
                x = next(blocks[cc][i] for cc in ctx["conditions"] if blocks[cc][i] is not None)
            f.append(_bandpower(x, ctx["fs"]).ravel())
        feats.append(np.concatenate(f))
    return np.asarray(feats, dtype=np.float32)


def fit_predict(F_train, y_train, F_test, ctx):
    sc = StandardScaler().fit(F_train)
    clf = LogisticRegression(C=0.1, max_iter=5000).fit(sc.transform(F_train), y_train)
    return clf.predict_proba(sc.transform(F_test))[:, 1]