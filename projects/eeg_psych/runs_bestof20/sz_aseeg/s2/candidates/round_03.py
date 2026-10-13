"""Log relative band power (delta/theta/alpha/beta/gamma) per channel from Welch PSD of each
available condition, plus per-channel aperiodic (1/f) slope fitted in log-log space excluding the
alpha peak; standardized, L2 logistic regression."""
import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
AP_RANGE = (2.0, 40.0)
AP_EXCLUDE = (7.0, 14.0)


def _features(x, fs):
    f, p = welch(x, fs=fs, nperseg=int(2 * fs), axis=-1)
    tot = p[:, (f >= 1) & (f <= 45)].sum(axis=1, keepdims=True) + 1e-12
    rel = np.log(np.stack([p[:, (f >= a) & (f < b)].sum(axis=1) for a, b in BANDS], axis=1) / tot)
    # aperiodic slope: linear fit of log10 power vs log10 frequency, alpha peak excluded
    m = (f >= AP_RANGE[0]) & (f <= AP_RANGE[1]) & ~((f >= AP_EXCLUDE[0]) & (f <= AP_EXCLUDE[1]))
    lf = np.log10(f[m])
    lp = np.log10(p[:, m] + 1e-12)
    lfc = lf - lf.mean()
    slope = ((lp - lp.mean(axis=1, keepdims=True)) * lfc).sum(axis=1) / (lfc ** 2).sum()
    return np.concatenate([rel.ravel(), slope])


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
    return np.asarray(feats, dtype=np.float32)


def fit_predict(F_train, y_train, F_test, ctx):
    sc = StandardScaler().fit(F_train)
    clf = LogisticRegression(C=0.1, max_iter=5000).fit(sc.transform(F_train), y_train)
    return clf.predict_proba(sc.transform(F_test))[:, 1]