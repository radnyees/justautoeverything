"""Per-channel seizure detection. Per second, on 8 s windows per channel: line length, RMS,
4 band powers, spectral entropy, dominant-peak power fraction, post-zero-crossing autocorrelation
peak, plus centered 15 s / 61 s moving averages (temporal context). Each channel-second row also
carries its deviation from the cross-channel median and the cross-channel median / max (global
background context). Histogram gradient boosting scores every channel; the per-second output is
the max over channels -> 10 s moving-average smoothing."""
import numpy as np
from scipy.ndimage import uniform_filter1d
from sklearn.ensemble import HistGradientBoostingClassifier

BANDS = [(0.5, 4), (4, 8), (8, 13), (13, 30)]
CONTEXT = (15, 61)                                   # seconds, centered moving-average windows
TRAIN_STRIDE = 4                                     # subsample training seconds (autocorrelated)


def _features(x, fs, nsec):
    win = 8 * fs
    nfft = 2 * win                                   # zero-padded -> linear autocorrelation
    pad = np.pad(x, ((0, 0), (win // 2, win // 2)), mode="reflect")
    freqs = np.fft.rfftfreq(nfft, 1 / fs)
    masks = [(freqs >= a) & (freqs < b) for a, b in BANDS]
    full = (freqs >= 0.5) & (freqs < 30)
    low_idx = np.where((freqs >= 0.5) & (freqs < 8))[0]
    nfull = full.sum()
    hw = 4                                           # +-0.25 Hz around the peak bin
    max_lag = 2 * fs
    nch = x.shape[0]
    out = np.empty((nsec, nch, 9), dtype=np.float32)
    for s in range(nsec):
        seg = pad[:, s * fs: s * fs + win].astype(np.float64)
        seg = seg - seg.mean(axis=1, keepdims=True)
        ll = np.abs(np.diff(seg, axis=1)).mean(axis=1)
        rms = np.sqrt((seg ** 2).mean(axis=1))
        P = np.abs(np.fft.rfft(seg, n=nfft, axis=1)) ** 2
        bp = np.stack([P[:, m].sum(axis=1) for m in masks], axis=1)
        # spectral entropy over 0.5-30 Hz
        Pf = P[:, full]
        tot = Pf.sum(axis=1) + 1e-12
        p = Pf / tot[:, None]
        sent = -(p * np.log(p + 1e-20)).sum(axis=1) / np.log(nfull)
        # dominant-peak power fraction in 0.5-8 Hz
        cs = np.concatenate([np.zeros((P.shape[0], 1)), np.cumsum(P, axis=1)], axis=1)
        lo = np.maximum(low_idx - hw, 0)
        hi = np.minimum(low_idx + hw + 1, P.shape[1])
        peak = (cs[:, hi] - cs[:, lo]).max(axis=1)
        pfrac = peak / tot
        # autocorrelation of band-limited signal: max after first zero crossing, lags <= 2 s
        Pb = np.where(full[None, :], P, 0.0)
        r = np.fft.irfft(Pb, n=nfft, axis=1)[:, :max_lag + 1]
        r = r / (r[:, :1] + 1e-12)
        after = np.cumsum(r < 0, axis=1) > 0
        rhy = np.clip(np.where(after, r, -1.0).max(axis=1), 0, 1)
        out[s] = np.column_stack([np.log(np.column_stack([ll, rms, bp]) + 1e-6),
                                  sent, pfrac, rhy])     # 19 x 9
    # temporal context per channel: centered moving averages within this recording
    ctxf = [uniform_filter1d(out, size=w, axis=0, mode="nearest") for w in CONTEXT]
    return np.concatenate([out] + ctxf, axis=2).astype(np.float32)   # nsec x 19 x 27


def featurize(items, ctx):
    from joblib import Parallel, delayed
    fs = ctx["fs"]
    eegs = ctx["load_eeg"](items)
    return Parallel(n_jobs=8)(delayed(_features)(np.asarray(x), fs, ctx["n_seconds"](it))
                              for it, x in zip(items, eegs))


def _rows(F):
    """(n, ch, d) per-channel features -> (n*ch, 4d) rows: channel, channel - median, median, max."""
    F = np.asarray(F, dtype=np.float32)
    n, c, d = F.shape
    med = np.median(F, axis=1, keepdims=True)
    mx = F.max(axis=1, keepdims=True)
    X = np.concatenate([F, F - med, np.broadcast_to(med, F.shape),
                        np.broadcast_to(mx, F.shape)], axis=2)
    return X.reshape(n * c, 4 * d)


def fit_predict(F_train, y_train, F_test, ctx):
    Xs, ys = [], []
    for F, y in zip(F_train, y_train):
        Fs = np.asarray(F)[::TRAIN_STRIDE]
        ys.append(np.repeat(np.asarray(y)[::TRAIN_STRIDE], Fs.shape[1]))
        Xs.append(_rows(Fs))
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    del Xs, ys
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=0).fit(X, y)
    del X
    k = np.ones(10) / 10
    preds = []
    for F in F_test:
        F = np.asarray(F)
        n, c, _ = F.shape
        p = clf.predict_proba(_rows(F))[:, 1].reshape(n, c).max(axis=1)
        preds.append(np.convolve(p, k, mode="same"))
    return preds