"""Log relative band power (delta/theta/alpha/beta/gamma) per channel from Welch PSD of each
available condition, plus per-channel aperiodic (1/f) slope from a log-log fit of the PSD
(2-40 Hz, alpha range excluded); standardized, L2 logistic regression."""
import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]


def _spectral(x, fs):
    f, p = welch(x, fs=fs, nperseg=int(2 * fs), axis=-1)
    tot = p[:, (f >= 1) & (f <= 45)].sum(axis=1, keepdims=True) + 1e-12
    rbp = np.log(np.stack([p[:, (f >= a) & (f < b)].sum(axis=1) for a, b in BANDS], axis=1) / tot)
    # aperiodic slope: linear fit of log10 power vs log10 freq, skipping the alpha peak
    m = (f >= 2) & (f <= 40) & ~((f >= 7) & (f <= 14))
    lf = np.log10(f[m])
    lp = np.log10(p[:, m] + 1e-12)
    A = np.stack([lf, np.ones_like(lf)], axis=1)
    coef, *_ = np.linalg.lstsq(A, lp.T, rcond=None)
    slope = coef[0][:, None]                   # channels x 1 (scale-invariant)
    return np.concatenate([rbp, slope], axis=1)


def featurize(items, ctx):
    feats = []
    blocks = {c: ctx["load_eeg"](items, c) for c in ctx["conditions"]}
    for i in range(len(items)):
        f = []
        for c in ctx["conditions"]:
            x = blocks[c][i]
            if x is None:                      # missing condition -> fall back to eyes closed
                x = blocks[ctx["conditions"][0]][i]
            f.append(_spectral(np.asarray(x, dtype=np.float64), ctx["fs"]).ravel())
        feats.append(np.concatenate(f))
    return np.asarray(feats, dtype=np.float32)


def fit_predict(F_train, y_train, F_test, ctx):
    sc = StandardScaler().fit(F_train)
    clf = LogisticRegression(C=0.1, max_iter=5000).fit(sc.transform(F_train), y_train)
    return clf.predict_proba(sc.transform(F_test))[:, 1]