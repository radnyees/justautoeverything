"""Log relative band power (delta/theta/alpha/beta/gamma) per channel from Welch PSD of each
available condition, plus per-channel individual alpha frequency (power-weighted centroid in
7-13 Hz), standardized, L2 logistic regression."""
import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
ALPHA_WIN = (7.0, 13.0)


def _spectral_feats(x, fs):
    f, p = welch(x, fs=fs, nperseg=int(4 * fs), axis=-1)
    tot = p[:, (f >= 1) & (f <= 45)].sum(axis=1, keepdims=True) + 1e-12
    rel = np.log(np.stack([p[:, (f >= a) & (f < b)].sum(axis=1) for a, b in BANDS], axis=1) / tot)
    # individual alpha frequency: centroid of the 7-13 Hz spectrum (scale/length invariant)
    m = (f >= ALPHA_WIN[0]) & (f <= ALPHA_WIN[1])
    pa = p[:, m]
    iaf = (pa * f[m][None, :]).sum(axis=1) / (pa.sum(axis=1) + 1e-12)
    return np.concatenate([rel.ravel(), iaf])


def featurize(items, ctx):
    feats = []
    blocks = {c: ctx["load_eeg"](items, c) for c in ctx["conditions"]}
    for i in range(len(items)):
        f = []
        for c in ctx["conditions"]:
            x = blocks[c][i]
            if x is None:                      # missing condition -> fall back to eyes closed
                x = blocks[ctx["conditions"][0]][i]
            x = np.asarray(x, dtype=np.float64)
            f.append(_spectral_feats(x, ctx["fs"]))
        feats.append(np.concatenate(f))
    return np.asarray(feats, dtype=np.float32)


def fit_predict(F_train, y_train, F_test, ctx):
    sc = StandardScaler().fit(F_train)
    clf = LogisticRegression(C=0.1, max_iter=5000).fit(sc.transform(F_train), y_train)
    return clf.predict_proba(sc.transform(F_test))[:, 1]