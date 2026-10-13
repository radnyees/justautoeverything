"""Per-second features on 8 s windows (line length, RMS, 4 band powers per channel; max and
median over channels), augmented with per-recording robust z-scored copies (per-neonate
normalization) -> histogram gradient boosting -> 10 s moving-average smoothing."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

BANDS = [(0.5, 4), (4, 8), (8, 13), (13, 30)]


def _features(x, fs, nsec):
    win = 8 * fs
    pad = np.pad(x, ((0, 0), (win // 2, win // 2)), mode="reflect")
    freqs = np.fft.rfftfreq(win, 1 / fs)
    masks = [(freqs >= a) & (freqs < b) for a, b in BANDS]
    out = np.empty((nsec, 12), dtype=np.float32)
    for s in range(nsec):
        seg = pad[:, s * fs: s * fs + win]
        seg = seg - seg.mean(axis=1, keepdims=True)
        ll = np.abs(np.diff(seg, axis=1)).mean(axis=1)
        rms = np.sqrt((seg ** 2).mean(axis=1))
        P = np.abs(np.fft.rfft(seg, axis=1)) ** 2
        bp = np.stack([P[:, m].sum(axis=1) for m in masks], axis=1)
        per_ch = np.log(np.column_stack([ll, rms, bp]) + 1e-6)          # 19 x 6
        out[s] = np.concatenate([per_ch.max(axis=0), np.median(per_ch, axis=0)])
    # per-neonate robust normalization (label-free): deviation from the recording's own background
    med = np.median(out, axis=0)
    iqr = np.percentile(out, 75, axis=0) - np.percentile(out, 25, axis=0)
    z = (out - med) / (iqr + 1e-3)
    return np.hstack([out, z.astype(np.float32)])


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