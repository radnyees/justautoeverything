"""Per-second features on 8 s windows of the 18-channel bipolar double-banana montage (0.5-30 Hz
band-passed): line length, RMS, 4 band powers per channel; max and median over channels ->
histogram gradient boosting -> 10 s moving-average smoothing."""
import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.ensemble import HistGradientBoostingClassifier

BANDS = [(0.5, 4), (4, 8), (8, 13), (13, 30)]

BIPOLAR = [("Fp1", "F7"), ("F7", "T3"), ("T3", "T5"), ("T5", "O1"),
           ("Fp2", "F8"), ("F8", "T4"), ("T4", "T6"), ("T6", "O2"),
           ("Fp1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
           ("Fp2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
           ("Fz", "Cz"), ("Cz", "Pz")]


def _bipolar_filtered(x, fs, channels):
    idx = {c: i for i, c in enumerate(channels)}
    sos = butter(4, [0.5, 30], btype="bandpass", fs=fs, output="sos")
    out = np.empty((len(BIPOLAR), x.shape[1]), dtype=np.float32)
    for k, (a, b) in enumerate(BIPOLAR):
        d = np.asarray(x[idx[a]], dtype=np.float64) - np.asarray(x[idx[b]], dtype=np.float64)
        out[k] = sosfiltfilt(sos, d).astype(np.float32)
    return out


def _features(x, fs, nsec, channels):
    x = _bipolar_filtered(x, fs, channels)
    win = 8 * fs
    pad = np.pad(x, ((0, 0), (win // 2, win // 2)), mode="reflect")
    del x
    freqs = np.fft.rfftfreq(win, 1 / fs)
    masks = [(freqs >= a) & (freqs < b) for a, b in BANDS]
    out = np.empty((nsec, 12), dtype=np.float32)
    for s in range(nsec):
        seg = pad[:, s * fs: s * fs + win].astype(np.float64)
        seg = seg - seg.mean(axis=1, keepdims=True)
        ll = np.abs(np.diff(seg, axis=1)).mean(axis=1)
        rms = np.sqrt((seg ** 2).mean(axis=1))
        P = np.abs(np.fft.rfft(seg, axis=1)) ** 2
        bp = np.stack([P[:, m].sum(axis=1) for m in masks], axis=1)
        per_ch = np.log(np.column_stack([ll, rms, bp]) + 1e-6)          # 18 x 6
        out[s] = np.concatenate([per_ch.max(axis=0), np.median(per_ch, axis=0)])
    return out


def featurize(items, ctx):
    from joblib import Parallel, delayed
    fs = ctx["fs"]
    channels = list(ctx["channels"])
    eegs = ctx["load_eeg"](items)
    return Parallel(n_jobs=8)(delayed(_features)(x, fs, ctx["n_seconds"](it), channels)
                              for it, x in zip(items, eegs))


def fit_predict(F_train, y_train, F_test, ctx):
    X = np.vstack(F_train)
    y = np.concatenate(y_train)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=0).fit(X, y)
    k = np.ones(10) / 10
    return [np.convolve(clf.predict_proba(F)[:, 1], k, mode="same") for F in F_test]