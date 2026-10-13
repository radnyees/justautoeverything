"""Log relative band power (delta/theta/alpha/beta/gamma) per channel from Welch PSD of each
available condition + per-channel aperiodic (1/f) exponent (log-log fit 2-40 Hz excluding the
alpha peak), standardized, L2 logistic regression."""
import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]


def _features(x, fs):
    f, p = welch(x, fs=fs, nperseg=int(2 * fs), axis=-1)
    tot = p[:, (f >= 1) & (f <= 45)].sum(axis=1, keepdims=True) + 1e-12
    rel = np.log(np.stack([p[:, (f >= a) & (f < b)].sum(axis=1) for a, b in BANDS], axis=1) / tot)
    # aperiodic exponent: slope of log10 power vs log10 frequency, alpha peak excluded
    m = (f >= 2) & (f <= 40) & ~((f >= 7) & (f <= 14))
    lf = np.log10(f[m])
    lp = np.log10(p[:, m] + 1e-12)                       # channels x freqs
    A = np.stack([lf, np.ones_like(lf)], axis=1)
    coef, *_ = np.linalg.lstsq(A, lp.T, rcond=None)     # 2 x channels
    expo = -coef[0]                                      # positive = steeper 1/f
    return np.concatenate([rel.ravel(), expo])


def featurize(items, ctx):
    feats = []
    blocks = {c: ctx["load_eeg"](items, c) for c in ctx["conditions"]}
    for i in range(len(items)):
        f = []
        for c in ctx["conditions"]:
            x = blocks[c][i]
            if x is None:                      # missing condition -> fall back to eyes closed
                x = blocks[ctx["conditions"][0]][i]
            f.append(_features(np.asarray(x, dtype=np.float64), ctx["fs"]))
        feats.append(np.concatenate(f))
    F = np.asarray(feats, dtype=np.float32)
    return np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)


def fit_predict(F_train, y_train, F_test, ctx):
    sc = StandardScaler().fit(F_train)
    clf = LogisticRegression(C=0.1, max_iter=5000).fit(sc.transform(F_train), y_train)
    return clf.predict_proba(sc.transform(F_test))[:, 1]