"""Log relative band power (delta/theta/alpha/beta/gamma) per channel from Welch PSD, plus per-channel
aperiodic (1/f) exponent (log-log fit 2-40 Hz excluding alpha) and posterior individual alpha
frequency (centre of gravity 7-13 Hz); standardized, L2 logistic regression."""
import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
POSTERIOR = ["O1", "O2", "P3", "P4", "Pz", "T5", "T6", "P7", "P8"]


def _spectral_feats(x, fs, post_idx):
    f, p = welch(x, fs=fs, nperseg=int(2 * fs), axis=-1)
    p = p + 1e-12
    tot = p[:, (f >= 1) & (f <= 45)].sum(axis=1, keepdims=True)
    rel = np.log(np.stack([p[:, (f >= a) & (f < b)].sum(axis=1) for a, b in BANDS], axis=1) / tot)

    # aperiodic exponent: linear fit of log10 power vs log10 frequency, alpha peak excluded
    m = (f >= 2) & (f <= 40) & ~((f >= 7) & (f <= 14))
    lf = np.log10(f[m])
    lp = np.log10(p[:, m])
    A = np.vstack([lf, np.ones_like(lf)]).T
    coef, *_ = np.linalg.lstsq(A, lp.T, rcond=None)
    slope = -coef[0]                                   # per channel exponent

    # individual alpha frequency: centre of gravity of posterior 7-13 Hz power above 1/f fit
    am = (f >= 7) & (f <= 13)
    fit = (A_alpha := np.vstack([np.log10(f[am]), np.ones(am.sum())]).T) @ coef
    resid = np.clip(p[:, am] - 10 ** fit.T, 0, None)
    pr = resid[post_idx].mean(axis=0) if len(post_idx) else resid.mean(axis=0)
    paf = np.sum(f[am] * pr) / (pr.sum() + 1e-12) if pr.sum() > 0 else 10.0
    return np.concatenate([rel.ravel(), slope, [paf]])


def featurize(items, ctx):
    feats = []
    chans = list(ctx["channels"])
    post_idx = [chans.index(c) for c in POSTERIOR if c in chans]
    blocks = {c: ctx["load_eeg"](items, c) for c in ctx["conditions"]}
    for i in range(len(items)):
        f = []
        for c in ctx["conditions"]:
            x = blocks[c][i]
            if x is None:                      # missing condition -> fall back to first condition
                x = blocks[ctx["conditions"][0]][i]
            f.append(_spectral_feats(np.asarray(x, dtype=np.float64), ctx["fs"], post_idx))
        feats.append(np.concatenate(f))
    F = np.asarray(feats, dtype=np.float32)
    return np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)


def fit_predict(F_train, y_train, F_test, ctx):
    sc = StandardScaler().fit(F_train)
    clf = LogisticRegression(C=0.1, max_iter=5000).fit(sc.transform(F_train), y_train)
    return clf.predict_proba(sc.transform(F_test))[:, 1]