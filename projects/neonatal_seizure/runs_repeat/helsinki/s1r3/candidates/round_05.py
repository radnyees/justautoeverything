"""Zero-phase 0.5-30 Hz band-pass per channel, then per-second features on 8 s windows (line length,
RMS, 4 band powers, plus rhythmicity features: autocorrelation peak, spectral peak fraction, negative
spectral entropy per channel; max and median over channels) -> histogram gradient boosting ->
10 s moving-average smoothing."""
import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.ensemble import HistGradientBoostingClassifier

BANDS = [(0.5, 4), (4, 8), (8, 13), (13, 30)]


def _bandpass(x, fs):
    # channel-by-channel to keep memory bounded on long recordings
    sos = butter(4, [0.5, 30], btype="bandpass", fs=fs, output="sos")
    y = np.empty(x.shape, dtype=np.float32)
    for c in range(x.shape[0]):
        y[c] = sosfiltfilt(sos, np.asarray(x[c], dtype=np.float64))
    return y


def _features(x, fs, nsec):
    x = _bandpass(x, fs)
    win = 8 * fs
    nfft = 2 * win                                   # zero-padded -> linear (non-circular) ACF
    pad = np.pad(x, ((0, 0), (win // 2, win // 2)), mode="reflect")
    freqs = np.fft.rfftfreq(nfft, 1 / fs)
    masks = [(freqs >= a) & (freqs < b) for a, b in BANDS]
    full = (freqs >= 0.5) & (freqs < 30)
    rhy = (freqs >= 0.5) & (freqs < 8)
    rhy_in_full = rhy[full]                          # positions of 0.5-8 Hz bins inside 0.5-30 Hz
    rhy_idx = np.where(rhy_in_full)[0]
    nb = int(full.sum())
    half = 4                                         # +/- 0.25 Hz around the peak (1/16 Hz bins)
    lag_lo, lag_hi = int(0.25 * fs), int(2 * fs)
    out = np.empty((nsec, 18), dtype=np.float32)
    for s in range(nsec):
        seg = pad[:, s * fs: s * fs + win].astype(np.float64)
        seg = seg - seg.mean(axis=1, keepdims=True)
        ll = np.abs(np.diff(seg, axis=1)).mean(axis=1)
        rms = np.sqrt((seg ** 2).mean(axis=1))
        P = np.abs(np.fft.rfft(seg, n=nfft, axis=1)) ** 2
        bp = np.stack([P[:, m].sum(axis=1) for m in masks], axis=1)
        per_ch = np.log(np.column_stack([ll, rms, bp]) + 1e-6)          # 19 x 6

        # rhythmicity: band-limited autocorrelation peak at 0.25-2 s lags
        Pb = np.where(full[None, :], P, 0.0)
        r = np.fft.irfft(Pb, n=nfft, axis=1)[:, :lag_hi + 1]
        acf = r[:, lag_lo:].max(axis=1) / (r[:, 0] + 1e-9)
        # spectral peak fraction: narrow peak in 0.5-8 Hz relative to 0.5-30 Hz power
        Pf = P[:, full]
        tot = Pf.sum(axis=1) + 1e-9
        pk = rhy_idx[np.argmax(Pf[:, rhy_in_full], axis=1)]
        cs = np.concatenate([np.zeros((Pf.shape[0], 1)), np.cumsum(Pf, axis=1)], axis=1)
        lo = np.maximum(pk - half, 0)
        hi = np.minimum(pk + half + 1, nb)
        rows = np.arange(Pf.shape[0])
        pfrac = (cs[rows, hi] - cs[rows, lo]) / tot
        # negative spectral entropy (higher = more concentrated spectrum)
        p = Pf / tot[:, None]
        nent = (p * np.log(p + 1e-12)).sum(axis=1) / np.log(nb)
        rh = np.column_stack([acf, pfrac, nent])                         # 19 x 3

        feats = np.column_stack([per_ch, rh])                            # 19 x 9
        out[s] = np.concatenate([feats.max(axis=0), np.median(feats, axis=0)])
    return out


def featurize(items, ctx):
    from joblib import Parallel, delayed
    fs = ctx["fs"]
    eegs = ctx["load_eeg"](items)
    return Parallel(n_jobs=8)(delayed(_features)(x, fs, ctx["n_seconds"](it))
                              for it, x in zip(items, eegs))


def fit_predict(F_train, y_train, F_test, ctx):
    X = np.vstack(F_train)
    y = np.concatenate(y_train)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=0).fit(X, y)
    k = np.ones(10) / 10
    return [np.convolve(clf.predict_proba(F)[:, 1], k, mode="same") for F in F_test]