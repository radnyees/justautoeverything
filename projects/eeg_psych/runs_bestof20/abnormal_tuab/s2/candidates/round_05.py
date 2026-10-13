"""Log absolute PSD in 1-Hz bins (1-45 Hz) per channel from Welch PSD of each available condition,
standardized, L2 logistic regression. Fine spectral resolution captures slowing / alpha-peak shifts /
beta excess that coarse relative bands blur; log-absolute lets the linear model form relative contrasts
itself while also seeing amplitude."""
import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

EDGES = np.arange(1, 46, 1)  # 1-Hz bins from 1 to 45 Hz -> 44 bins


def _logpsd(x, fs):
    x = np.nan_to_num(np.asarray(x, dtype=np.float64))
    f, p = welch(x, fs=fs, nperseg=int(2 * fs), axis=-1)
    bins = []
    for a, b in zip(EDGES[:-1], EDGES[1:]):
        m = (f >= a) & (f < b)
        bins.append(p[:, m].mean(axis=1))
    return np.log(np.stack(bins, axis=1) + 1e-6)


def featurize(items, ctx):
    feats = []
    blocks = {c: ctx["load_eeg"](items, c) for c in ctx["conditions"]}
    for i in range(len(items)):
        f = []
        for c in ctx["conditions"]:
            x = blocks[c][i]
            if x is None:                      # missing condition -> fall back to first condition
                x = blocks[ctx["conditions"][0]][i]
            f.append(_logpsd(x, ctx["fs"]).ravel())
        feats.append(np.concatenate(f))
    return np.asarray(feats, dtype=np.float32)


def fit_predict(F_train, y_train, F_test, ctx):
    sc = StandardScaler().fit(F_train)
    clf = LogisticRegression(C=0.01, max_iter=5000).fit(sc.transform(F_train), y_train)
    return clf.predict_proba(sc.transform(F_test))[:, 1]