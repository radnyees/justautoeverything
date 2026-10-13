"""Log relative power in fine 2-Hz spectral bins (1-2, 2-4, ..., 28-30 Hz, plus 30-45 Hz gamma) per
channel from Welch PSD of each available condition, plus per-channel temporal variability (std over
2-s windows of log sub-band power in delta, theta1/2, alpha1/2, beta1/2 and gamma; vigilance /
non-stationarity marker), plus long-range temporal correlation (DFA exponent of amplitude envelopes
in the same sub-bands, 1-20 s scales), plus fast burstiness (coefficient of variation of the
amplitude envelope in the same sub-bands), plus burst lifetime (log mean duration of supra-75th-
percentile envelope episodes), plus frequency instability (log std over 0.2-s blocks of the Hilbert
instantaneous frequency in each sub-band), standardized, L2 logistic regression."""
import numpy as np
from scipy.signal import welch, spectrogram, butter, sosfiltfilt, hilbert
from scipy.fft import next_fast_len
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

EDGES = [1] + list(range(2, 31, 2)) + [45]
BANDS = list(zip(EDGES[:-1], EDGES[1:]))
VAR_BANDS = [(1, 4), (4, 6), (6, 8), (8, 10), (10, 13), (13, 20), (20, 30), (30, 45)]
DFA_BANDS = [(1, 4), (4, 6), (6, 8), (8, 10), (10, 13), (13, 20), (20, 30), (30, 45)]
ENV_FS = 25.0                                   # envelope sampling rate for DFA
DFA_SCALES_S = np.logspace(0, np.log10(20), 10)  # 1-20 s
BURST_Q = 75.0                                  # percentile threshold for burst detection
IF_BLOCK_S = 0.2                                # block length for instantaneous-frequency averaging


def _bandpower(x, fs):
    f, p = welch(x, fs=fs, nperseg=int(2 * fs), axis=-1)
    tot = p[:, (f >= 1) & (f <= 45)].sum(axis=1, keepdims=True) + 1e-12
    return np.log(np.stack([p[:, (f >= a) & (f < b)].sum(axis=1) for a, b in BANDS], axis=1) / tot
                  + 1e-12)


def _variability(x, fs):
    # std across 2-s windows (50% overlap) of log band power -> scale invariant
    f, _, s = spectrogram(x, fs=fs, nperseg=int(2 * fs), noverlap=int(fs), axis=-1)  # ch x f x t
    lp = np.stack([np.log(s[:, (f >= a) & (f < b), :].sum(axis=1) + 1e-12) for a, b in VAR_BANDS],
                  axis=1)                                                         # ch x band x t
    return lp.std(axis=-1)


def _dfa(env):
    # env: ch x T envelope at ENV_FS -> DFA exponent per channel
    y = np.cumsum(env - env.mean(axis=1, keepdims=True), axis=1)
    T = y.shape[1]
    logn, logF = [], []
    for s in DFA_SCALES_S:
        n = int(round(s * ENV_FS))
        k = T // n
        if k < 2:
            continue
        seg = y[:, :k * n].reshape(y.shape[0], k, n)
        t = np.arange(n, dtype=np.float64)
        X = np.stack([np.ones(n), t - t.mean()], axis=1)
        P = X @ np.linalg.pinv(X)                       # n x n projection onto linear trend
        res = seg - seg @ P.T
        logn.append(np.log(n))
        logF.append(0.5 * np.log(np.mean(res ** 2, axis=(1, 2)) + 1e-24))
    if len(logn) < 3:
        return np.full(y.shape[0], 0.5)
    logn = np.asarray(logn)
    logF = np.stack(logF, axis=1)                         # ch x scales
    ln = logn - logn.mean()
    return ((logF - logF.mean(axis=1, keepdims=True)) @ ln) / (ln @ ln)


def _burst_lifetime(e, fs):
    # log mean duration (s) of episodes with envelope above its own per-channel percentile
    thr = np.percentile(e, BURST_Q, axis=1, keepdims=True)
    above = e > thr
    onsets = np.sum(above[:, 1:] & ~above[:, :-1], axis=1) + above[:, 0]
    dur = above.sum(axis=1) / fs / np.maximum(onsets, 1)
    return np.log(dur + 1e-6)


def _freq_instability(z, fs):
    # z: ch x T analytic signal (edges trimmed). Block-averaged instantaneous frequency (Hz) is the
    # net phase advance per block; its std over time measures frequency wandering (scale invariant)
    dphi = np.angle(z[:, 1:] * np.conj(z[:, :-1])) * fs / (2 * np.pi)
    n = max(2, int(round(IF_BLOCK_S * fs)))
    k = dphi.shape[1] // n
    if k < 3:
        return np.zeros(z.shape[0])
    bf = dphi[:, :k * n].reshape(z.shape[0], k, n).mean(axis=2)
    return np.log(bf.std(axis=1) + 1e-6)


def _lrtc(x, fs):
    step = max(1, int(round(fs / ENV_FS)))
    dfa, cv, bl, fi = [], [], [], []
    for a, b in DFA_BANDS:
        sos = butter(4, [a, b], btype="bandpass", fs=fs, output="sos")
        xf = sosfiltfilt(sos, x, axis=-1)
        N = xf.shape[1]
        z = hilbert(xf, N=next_fast_len(N), axis=-1)[:, :N]
        env = np.abs(z)
        dfa.append(_dfa(env[:, ::step]))
        # burstiness: coefficient of variation of the envelope (edges trimmed), scale invariant
        if N > 4 * int(fs):
            e = env[:, int(fs):N - int(fs)]
            zt = z[:, int(fs):N - int(fs)]
        else:
            e, zt = env, z
        cv.append(e.std(axis=1) / (e.mean(axis=1) + 1e-12))
        bl.append(_burst_lifetime(e, fs))
        fi.append(_freq_instability(zt, fs))
    return (np.stack(dfa, axis=1), np.stack(cv, axis=1), np.stack(bl, axis=1),
            np.stack(fi, axis=1))  # ch x band each


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
            x = np.asarray(x, dtype=np.float64)
            f.append(_bandpower(x, fs).ravel())
            f.append(_variability(x, fs).ravel())
            dfa, cv, bl, fi = _lrtc(x, fs)
            f.append(dfa.ravel())
            f.append(cv.ravel())
            f.append(bl.ravel())
            f.append(fi.ravel())
        feats.append(np.concatenate(f))
    F = np.asarray(feats, dtype=np.float32)
    return np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)


def fit_predict(F_train, y_train, F_test, ctx):
    sc = StandardScaler().fit(F_train)
    clf = LogisticRegression(C=0.1, max_iter=5000).fit(sc.transform(F_train), y_train)
    return clf.predict_proba(sc.transform(F_test))[:, 1]