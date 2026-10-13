"""Log relative band power (delta/theta/alpha/beta/gamma) per channel from Welch PSD of each
available condition, plus per-channel individual alpha frequency (centre of gravity 7-13 Hz) and
aperiodic 1/f slope (log-log fit 2-40 Hz excluding alpha), standardized, L2 logistic regression."""
import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]


def _spectral(x, fs):
    f, p = welch(x, fs=fs, nperseg=int(2 * fs), axis=-1)
    p = p + 1e-20
    tot = p[:, (f >= 1) & (f <= 45)].sum(axis=1, keepdims=True)
    rel = np.log(np.stack([p[:, (f >= a) & (f < b)].sum(axis=1) for a, b in BANDS], axis=1) / tot)
    # individual alpha frequency: power-weighted centre of gravity in 7-13 Hz
    am = (f >= 7) & (f <= 13)
    iaf = (p[:, am] * f[am]).sum(axis=1) / p[:, am].sum(axis=1)
    # aperiodic slope: linear fit of log10 power vs log10 frequency, alpha range excluded
    fm = (f >= 2) & (f <= 40) & ~((f >= 7) & (f <= 14))
    lf = np.log10(f[fm])
    lp = np.log10(p[:, fm])
    lfc = lf - lf.mean()
    slope = ((lp - lp.mean(axis=1, keepdims=True)) * lfc).sum(axis=1) / (lfc ** 2).sum()
    return np.concatenate([rel.ravel(), iaf, slope])


def featurize(items, ctx):
    feats = []
    blocks = {c: ctx["load_eeg"](items, c) for c in ctx["conditions"]}
    for i in range(len(items)):
        f = []
        for c in ctx["conditions"]:
            x = blocks[c][i]
            if x is None:                      # missing condition -> fall back to eyes closed
                x = blocks[ctx["conditions"][0]][i]
            f.append(_spectral(np.asarray(x, dtype=np.float64), ctx["fs"]))
        feats.append(np.concatenate(f))
    return np.asarray(feats, dtype=np.float32)


def fit_predict(F_train, y_train, F_test, ctx):
    sc = StandardScaler().fit(F_train)
    clf = LogisticRegression(C=0.1, max_iter=5000).fit(sc.transform(F_train), y_train)
    return clf.predict_proba(sc.transform(F_test))[:, 1]