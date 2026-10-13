"""Log relative band power (delta/theta/alpha/beta/gamma) per channel from Welch PSD of each
available condition, plus scale-invariant per-channel aperiodic (1/f) slope and posterior
individual alpha peak frequency (centre of gravity), standardized, L2 logistic regression."""
import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
POSTERIOR = ["O1", "O2", "P3", "P4", "Pz", "T5", "T6", "P7", "P8"]


def _features(x, fs, post_idx):
    f, p = welch(x, fs=fs, nperseg=int(2 * fs), axis=-1)
    p = p + 1e-12
    tot = p[:, (f >= 1) & (f <= 45)].sum(axis=1, keepdims=True)
    rel = np.log(np.stack([p[:, (f >= a) & (f < b)].sum(axis=1) for a, b in BANDS], axis=1) / tot)

    # aperiodic slope: log-log linear fit over 2-40 Hz excluding the alpha peak region
    m = (f >= 2) & (f <= 40) & ~((f >= 7) & (f <= 14))
    lf = np.log10(f[m])
    lp = np.log10(p[:, m])
    A = np.vstack([lf, np.ones_like(lf)]).T
    coef, *_ = np.linalg.lstsq(A, lp.T, rcond=None)
    slope = coef[0]

    # individual alpha frequency: centre of gravity 7-14 Hz on posterior mean spectrum
    pm = p[post_idx].mean(axis=0)
    am = (f >= 7) & (f <= 14)
    paf = np.array([(f[am] * pm[am]).sum() / pm[am].sum()])

    return np.concatenate([rel.ravel(), slope, paf])


def featurize(items, ctx):
    chans = list(ctx["channels"])
    post_idx = [i for i, c in enumerate(chans) if c in POSTERIOR] or list(range(len(chans)))
    feats = []
    blocks = {c: ctx["load_eeg"](items, c) for c in ctx["conditions"]}
    for i in range(len(items)):
        f = []
        for c in ctx["conditions"]:
            x = blocks[c][i]
            if x is None:                      # missing condition -> fall back to eyes closed
                x = blocks[ctx["conditions"][0]][i]
            f.append(_features(np.asarray(x, dtype=np.float64), ctx["fs"], post_idx))
        feats.append(np.concatenate(f))
    F = np.asarray(feats, dtype=np.float32)
    return np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)


def fit_predict(F_train, y_train, F_test, ctx):
    sc = StandardScaler().fit(F_train)
    clf = LogisticRegression(C=0.1, max_iter=5000).fit(sc.transform(F_train), y_train)
    return clf.predict_proba(sc.transform(F_test))[:, 1]