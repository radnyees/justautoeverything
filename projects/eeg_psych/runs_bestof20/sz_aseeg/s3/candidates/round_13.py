"""Log relative band power (delta/theta/alpha/beta/gamma) per channel plus per-channel aperiodic
(1/f) exponent and peak alpha frequency (centre of gravity of 1/f-corrected alpha) from Welch PSD of
each available condition, standardized, L2 logistic regression."""
import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]


def _bandpower(x, fs):
    f, p = welch(x, fs=fs, nperseg=int(2 * fs), axis=-1)
    tot = p[:, (f >= 1) & (f <= 45)].sum(axis=1, keepdims=True) + 1e-12
    return np.log(np.stack([p[:, (f >= a) & (f < b)].sum(axis=1) for a, b in BANDS], axis=1) / tot)


def _spectral_shape(x, fs):
    """Per channel: aperiodic exponent (log-log fit 2-40 Hz excluding 7-14 Hz) and peak alpha
    frequency (centre of gravity of the positive 1/f-corrected residual in 7-13 Hz)."""
    f, p = welch(x, fs=fs, nperseg=int(4 * fs), average="median", axis=-1)
    lf = np.log10(np.maximum(f, 1e-6))
    lp = np.log10(p + 1e-12)
    fit_m = (f >= 2) & (f <= 40) & ~((f >= 7) & (f <= 14))
    A = np.stack([lf[fit_m], np.ones(fit_m.sum())], axis=1)
    coef, *_ = np.linalg.lstsq(A, lp[:, fit_m].T, rcond=None)      # (2, n_ch)
    slope, icpt = coef[0], coef[1]
    am = (f >= 7) & (f <= 13)
    resid = lp[:, am] - (slope[:, None] * lf[am][None, :] + icpt[:, None])
    w = np.clip(resid, 0, None) + 1e-6
    paf = (w * f[am][None, :]).sum(axis=1) / w.sum(axis=1)
    return np.stack([-slope, paf], axis=1)


def featurize(items, ctx):
    feats = []
    blocks = {c: ctx["load_eeg"](items, c) for c in ctx["conditions"]}
    fs = ctx["fs"]
    for i in range(len(items)):
        f = []
        for c in ctx["conditions"]:
            x = blocks[c][i]
            if x is None:                      # missing condition -> fall back to eyes closed
                x = blocks[ctx["conditions"][0]][i]
            x = np.asarray(x, dtype=np.float64)
            f.append(_bandpower(x, fs).ravel())
            f.append(_spectral_shape(x, fs).ravel())
        feats.append(np.concatenate(f))
    F = np.asarray(feats, dtype=np.float32)
    return np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)


def fit_predict(F_train, y_train, F_test, ctx):
    sc = StandardScaler().fit(F_train)
    clf = LogisticRegression(C=0.1, max_iter=5000).fit(sc.transform(F_train), y_train)
    return clf.predict_proba(sc.transform(F_test))[:, 1]