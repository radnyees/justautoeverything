"""Log relative band power (delta/theta/alpha/beta/gamma) per channel plus per-channel aperiodic
(1/f) exponent from Welch PSD of each available condition, standardized, L2 logistic regression."""
import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
APER_RANGE = (2.0, 40.0)
APER_EXCLUDE = (7.0, 14.0)   # skip the alpha peak so it does not bias the 1/f fit


def _spectral_features(x, fs):
    f, p = welch(x, fs=fs, nperseg=int(2 * fs), axis=-1)
    p = p + 1e-12
    tot = p[:, (f >= 1) & (f <= 45)].sum(axis=1, keepdims=True)
    rel = np.log(np.stack([p[:, (f >= a) & (f < b)].sum(axis=1) for a, b in BANDS], axis=1) / tot)

    # aperiodic exponent: slope of log10 power vs log10 frequency (scale-invariant)
    m = (f >= APER_RANGE[0]) & (f <= APER_RANGE[1]) & ~((f >= APER_EXCLUDE[0]) & (f <= APER_EXCLUDE[1]))
    lf = np.log10(f[m])
    lp = np.log10(p[:, m])                                   # channels x freqs
    lf_c = lf - lf.mean()
    slope = (lp - lp.mean(axis=1, keepdims=True)) @ lf_c / (lf_c @ lf_c)
    expo = -slope
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
            f.append(_spectral_features(np.asarray(x, dtype=np.float64), ctx["fs"]))
        feats.append(np.concatenate(f))
    F = np.asarray(feats, dtype=np.float32)
    return np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)


def fit_predict(F_train, y_train, F_test, ctx):
    sc = StandardScaler().fit(F_train)
    clf = LogisticRegression(C=0.1, max_iter=5000).fit(sc.transform(F_train), y_train)
    return clf.predict_proba(sc.transform(F_test))[:, 1]