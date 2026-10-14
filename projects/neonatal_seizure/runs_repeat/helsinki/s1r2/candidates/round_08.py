"""Per-second features on 8 s windows: line length, RMS, 4 band powers, plus rhythmicity features
(spectral entropy, dominant-peak power fraction, post-zero-crossing autocorrelation peak) and
1/f-detrended spectral-peak features (prominence above aperiodic fit, peak frequency, aperiodic slope)
per channel; max / median / min over channels, plus centered 15 s / 61 s moving averages of every
feature (temporal context) -> histogram gradient boosting -> 10 s moving-average smoothing."""
import numpy as np
from scipy.ndimage import uniform_filter1d
from sklearn.ensemble import HistGradientBoostingClassifier

BANDS = [(0.5, 4), (4, 8), (8, 13), (13, 30)]
CONTEXT = (15, 61)                                   # seconds, centered moving-average windows
NPC = 12                                             # per-channel features


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
    # 1/f fit on 0.5 Hz bins spanning 0.5-30 Hz
    df = freqs[1] - freqs[0]
    bin_w = int(round(0.5 / df))
    i0 = int(round(0.5 / df))
    nb = int(round(29.5 / 0.5))
    i1 = i0 + nb * bin_w
    centers = freqs[i0:i1].reshape(nb, bin_w).mean(axis=1)
    lf = np.log(centers)
    A = np.column_stack([np.ones(nb), lf])
    pinvA = np.linalg.pinv(A)                        # 2 x nb
    low_bins = centers < 8
    out = np.empty((nsec, 3 * NPC), dtype=np.float32)
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
        # aperiodic (1/f) fit in log-log space; residual peak = oscillation above background
        lp = np.log(P[:, i0:i1].reshape(P.shape[0], nb, bin_w).mean(axis=2) + 1e-12)
        coef = lp @ pinvA.T                          # 19 x 2 (intercept, slope)
        res = lp - coef @ A.T
        rl = res[:, low_bins]
        prom = rl.max(axis=1)
        pfreq = centers[low_bins][rl.argmax(axis=1)]
        slope = coef[:, 1]
        per_ch = np.column_stack([np.log(np.column_stack([ll, rms, bp]) + 1e-6),
                                  sent, pfrac, rhy, prom, pfreq, slope])     # 19 x 12
        out[s] = np.concatenate([per_ch.max(axis=0), np.median(per_ch, axis=0),
                                 per_ch.min(axis=0)])
    # temporal context: centered moving averages of each feature within this recording
    ctxf = [uniform_filter1d(out, size=w, axis=0, mode="nearest") for w in CONTEXT]
    return np.hstack([out] + ctxf).astype(np.float32)


def featurize(items, ctx):
    from joblib import Parallel, delayed
    fs = ctx["fs"]
    eegs = ctx["load_eeg"](items)
    return Parallel(n_jobs=8)(delayed(_features)(np.asarray(x), fs, ctx["n_seconds"](it))
                              for it, x in zip(items, eegs))


def fit_predict(F_train, y_train, F_test, ctx):
    X = np.vstack(F_train)
    y = np.concatenate(y_train)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=0).fit(X, y)
    k = np.ones(10) / 10
    return [np.convolve(clf.predict_proba(F)[:, 1], k, mode="same") for F in F_test]