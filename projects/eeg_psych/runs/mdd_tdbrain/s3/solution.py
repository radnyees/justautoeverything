"""Log relative band power (delta/theta/alpha/beta/gamma + 55-95 Hz high-frequency/EMG band) per
channel from Welch PSD of each available condition after common-average re-referencing, plus the
scale-free spatial topography of total 1-45 Hz power (log total per channel minus subject mean
across channels), plus per-channel log kurtosis of the 1-45 Hz and 55-95 Hz band-passed signal
(scale-free burstiness of transients / EMG bursts), plus vigilance-dynamics features: the slope
over normalized time and the across-chunk SD of channel-averaged log relative band power over 8
equal fractions of each recording (hyperstable vigilance regulation in depression).
Classifier: equal-weight probability average of standardized L2 logistic regression and a random
forest."""
import numpy as np
from scipy.signal import welch, butter, sosfiltfilt
from scipy.stats import kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
HF_BAND = (55, 95)   # above 50 Hz line noise, below its 100 Hz harmonic; mostly scalp EMG
N_CHUNKS = 8


def _car(x):
    x = np.asarray(x, dtype=np.float64)
    return x - x.mean(axis=0, keepdims=True)


def _bands(fs):
    bands = list(BANDS)
    if fs / 2.0 > HF_BAND[1]:
        bands.append(HF_BAND)
    return bands


def _logrel(x, fs):
    f, p = welch(x, fs=fs, nperseg=int(2 * fs), axis=-1)
    tot = p[:, (f >= 1) & (f <= 45)].sum(axis=1, keepdims=True) + 1e-12
    bp = np.stack([p[:, (f >= a) & (f < b)].sum(axis=1) for a, b in _bands(fs)], axis=1)
    return np.log(bp / tot + 1e-12), tot[:, 0]


def _bandpower(x, fs):
    rel, tot = _logrel(x, fs)
    ltot = np.log(tot)
    topo = ltot - ltot.mean()                 # spatial distribution of power, amplitude-scale free
    return np.concatenate([rel.ravel(), topo])


def _vigilance(x, fs):
    """Time course of global log relative band power across equal fractions of the recording."""
    nb = len(_bands(fs))
    n = x.shape[1]
    edges = np.linspace(0, n, N_CHUNKS + 1).astype(int)
    rows = []
    for k in range(N_CHUNKS):
        seg = x[:, edges[k]:edges[k + 1]]
        if seg.shape[1] < int(4 * fs):
            continue
        rel, _ = _logrel(seg, fs)
        rows.append(rel.mean(axis=0))
    if len(rows) < 3:
        return np.zeros(2 * nb)
    R = np.asarray(rows)                      # (chunks, bands)
    t = np.linspace(0.0, 1.0, len(rows))
    t = t - t.mean()
    slope = (t[:, None] * (R - R.mean(axis=0))).sum(axis=0) / (t ** 2).sum()
    sd = R.std(axis=0)
    return np.concatenate([slope, sd])


def _kurt(x, fs):
    out = []
    bands = [(1, 45)]
    if fs / 2.0 > HF_BAND[1]:
        bands.append(HF_BAND)
    for a, b in bands:
        sos = butter(4, [a, b], btype="bandpass", fs=fs, output="sos")
        xf = sosfiltfilt(sos, x, axis=-1)
        k = kurtosis(xf, axis=-1, fisher=False)    # Pearson kurtosis (Gaussian = 3)
        out.append(np.log(np.maximum(k, 1e-3)))
    return np.concatenate(out)


def featurize(items, ctx):
    feats = []
    fs = ctx["fs"]
    blocks = {c: ctx["load_eeg"](items, c) for c in ctx["conditions"]}
    for i in range(len(items)):
        f = []
        for c in ctx["conditions"]:
            x = blocks[c][i]
            if x is None:                      # missing condition -> fall back to eyes closed
                x = blocks[ctx["conditions"][0]][i]
            xc = _car(x)
            f.append(_bandpower(xc, fs))
            f.append(_kurt(xc, fs))
            f.append(_vigilance(xc, fs))
        feats.append(np.concatenate(f))
    F = np.asarray(feats, dtype=np.float64)
    F[~np.isfinite(F)] = 0.0
    return F.astype(np.float32)


def fit_predict(F_train, y_train, F_test, ctx):
    F_train = np.asarray(F_train, dtype=np.float64)
    F_test = np.asarray(F_test, dtype=np.float64)
    sc = StandardScaler().fit(F_train)
    lr = LogisticRegression(C=0.1, max_iter=5000).fit(sc.transform(F_train), y_train)
    p_lr = lr.predict_proba(sc.transform(F_test))[:, 1]
    rf = RandomForestClassifier(n_estimators=1000, max_features="sqrt", min_samples_leaf=3,
                                class_weight="balanced_subsample", n_jobs=-1,
                                random_state=0).fit(F_train, y_train)
    p_rf = rf.predict_proba(F_test)[:, 1]
    return 0.5 * p_lr + 0.5 * p_rf