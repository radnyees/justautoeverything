"""Log relative band power (delta/theta/alpha/beta/gamma) per channel from Welch PSD of each
available condition + individual alpha peak frequency (centre of gravity 7-13 Hz, median-averaged
high-resolution Welch) per channel and its across-channel median; standardized, L2 logistic regression."""
import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]


def _bandpower(x, fs):
    f, p = welch(x, fs=fs, nperseg=int(2 * fs), axis=-1)
    tot = p[:, (f >= 1) & (f <= 45)].sum(axis=1, keepdims=True) + 1e-12
    return np.log(np.stack([p[:, (f >= a) & (f < b)].sum(axis=1) for a, b in BANDS], axis=1) / tot)


def _alpha_cog(x, fs):
    # 4 s windows (0.25 Hz resolution), median across segments for robustness to transient artifacts
    nper = min(int(4 * fs), x.shape[-1])
    f, p = welch(x, fs=fs, nperseg=nper, axis=-1, average="median")
    m = (f >= 7) & (f <= 13)
    pa = p[:, m]
    cog = (pa * f[m]).sum(axis=1) / (pa.sum(axis=1) + 1e-12)
    return np.concatenate([cog, [np.median(cog)]])


def featurize(items, ctx):
    feats = []
    fs = ctx["fs"]
    blocks = {c: ctx["load_eeg"](items, c) for c in ctx["conditions"]}
    for i in range(len(items)):
        f = []
        for c in ctx["conditions"]:
            x = blocks[c][i]
            if x is None:                      # missing condition -> fall back to eyes closed
                x = blocks[ctx["conditions"][0]][i]
            x = np.asarray(x, dtype=np.float64)
            f.append(_bandpower(x, fs).ravel())
            f.append(_alpha_cog(x, fs))
        feats.append(np.concatenate(f))
    return np.asarray(feats, dtype=np.float32)


def fit_predict(F_train, y_train, F_test, ctx):
    sc = StandardScaler().fit(F_train)
    clf = LogisticRegression(C=0.1, max_iter=5000).fit(sc.transform(F_train), y_train)
    return clf.predict_proba(sc.transform(F_test))[:, 1]