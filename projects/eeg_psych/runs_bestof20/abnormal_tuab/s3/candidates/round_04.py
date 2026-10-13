"""Log relative spectral power in fine bins (1-Hz bins 1-20 Hz, 2-Hz bins 20-30 Hz, 30-45 Hz) per
channel from Welch PSD of each available condition, standardized, L2 logistic regression.
Fine bins resolve slowing of the posterior dominant rhythm (alpha peak shift) and focal delta/theta."""
import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

EDGES = list(range(1, 21)) + [22, 24, 26, 28, 30, 45]
BANDS = list(zip(EDGES[:-1], EDGES[1:]))


def _bandpower(x, fs):
    f, p = welch(x, fs=fs, nperseg=int(4 * fs), axis=-1)
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
            if x is None:                      # missing condition -> fall back to first condition
                x = blocks[ctx["conditions"][0]][i]
            f.append(_bandpower(np.asarray(x, dtype=np.float64), ctx["fs"]).ravel())
        feats.append(np.concatenate(f))
    return np.asarray(feats, dtype=np.float32)


def fit_predict(F_train, y_train, F_test, ctx):
    sc = StandardScaler().fit(F_train)
    clf = LogisticRegression(C=0.1, max_iter=5000).fit(sc.transform(F_train), y_train)
    return clf.predict_proba(sc.transform(F_test))[:, 1]