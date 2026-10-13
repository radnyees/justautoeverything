"""Log relative + log absolute band power (delta/theta/alpha/beta/gamma) and log total power per
channel from median-averaged Welch PSD (robust to transient artifacts), standardized, L2 logistic
regression."""
import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]


def _bandpower(x, fs):
    x = np.asarray(x, dtype=np.float64)
    f, p = welch(x, fs=fs, nperseg=int(2 * fs), axis=-1, average="median")
    df = f[1] - f[0]
    tot = p[:, (f >= 1) & (f <= 45)].sum(axis=1) * df + 1e-12
    bp = np.stack([p[:, (f >= a) & (f < b)].sum(axis=1) * df for a, b in BANDS], axis=1) + 1e-12
    rel = np.log(bp / tot[:, None])
    absp = np.log(bp)
    return np.concatenate([rel, absp, np.log(tot)[:, None]], axis=1)


def featurize(items, ctx):
    feats = []
    blocks = {c: ctx["load_eeg"](items, c) for c in ctx["conditions"]}
    for i in range(len(items)):
        f = []
        for c in ctx["conditions"]:
            x = blocks[c][i]
            if x is None:                      # missing condition -> fall back to first condition
                x = blocks[ctx["conditions"][0]][i]
            f.append(_bandpower(x, ctx["fs"]).ravel())
        feats.append(np.concatenate(f))
    F = np.asarray(feats, dtype=np.float32)
    return np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)


def fit_predict(F_train, y_train, F_test, ctx):
    sc = StandardScaler().fit(F_train)
    clf = LogisticRegression(C=0.1, max_iter=5000).fit(sc.transform(F_train), y_train)
    return clf.predict_proba(sc.transform(F_test))[:, 1]