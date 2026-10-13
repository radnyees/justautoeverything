"""Zero-phase 0.5-30 Hz band-pass per channel, then per-second features on 8 s windows computed on both
the 19 referential channels and the 18-channel neonatal bipolar double-banana montage (line length,
RMS, 4 band powers, rhythmicity features: autocorrelation peak, spectral peak fraction, negative
spectral entropy; amplitude-invariant spectral-shape features: relative band powers, Hjorth mobility
and complexity, 90% spectral edge; max and median over channels per montage) -> histogram gradient
boosting -> second-stage stacker (regularised gradient boosting) trained on grouped inner-CV
out-of-fold probabilities, using multi-scale smoothed probabilities, a rolling max and label-free
recording-level probability percentiles as temporal / neonate-level context."""
import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.ndimage import uniform_filter1d, maximum_filter1d
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

BANDS = [(0.5, 4), (4, 8), (8, 13), (13, 30)]
STACK_SCALES = [10, 30, 60, 120, 240, 480]
FALLBACK_W = 80
N_PER_CH = 16
BIPOLAR = [("Fp2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
           ("Fp1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
           ("Fp2", "F8"), ("F8", "T4"), ("T4", "T6"), ("T6", "O2"),
           ("Fp1", "F7"), ("F7", "T3"), ("T3", "T5"), ("T5", "O1"),
           ("Fz", "Cz"), ("Cz", "Pz")]


def _bipolar_matrix(channels):
    idx = {c.lower(): i for i, c in enumerate(channels)}
    B = np.zeros((len(BIPOLAR), len(channels)))
    for k, (a, b) in enumerate(BIPOLAR):
        B[k, idx[a.lower()]] = 1.0
        B[k, idx[b.lower()]] = -1.0
    return B


def _bandpass(x, fs):
    # channel-by-channel to keep memory bounded on long recordings
    sos = butter(4, [0.5, 30], btype="bandpass", fs=fs, output="sos")
    y = np.empty(x.shape, dtype=np.float32)
    for c in range(x.shape[0]):
        y[c] = sosfiltfilt(sos, np.asarray(x[c], dtype=np.float64))
    return y


def _features(x, fs, nsec, B):
    x = _bandpass(x, fs)
    n_ref = x.shape[0]
    win = 8 * fs
    nfft = 2 * win                                   # zero-padded -> linear (non-circular) ACF
    pad = np.pad(x, ((0, 0), (win // 2, win // 2)), mode="reflect")
    del x
    freqs = np.fft.rfftfreq(nfft, 1 / fs)
    masks = [(freqs >= a) & (freqs < b) for a, b in BANDS]
    full = (freqs >= 0.5) & (freqs < 30)
    freqs_full = freqs[full]
    rhy = (freqs >= 0.5) & (freqs < 8)
    rhy_in_full = rhy[full]                          # positions of 0.5-8 Hz bins inside 0.5-30 Hz
    rhy_idx = np.where(rhy_in_full)[0]
    nb = int(full.sum())
    half = 4                                         # +/- 0.25 Hz around the peak (1/16 Hz bins)
    lag_lo, lag_hi = int(0.25 * fs), int(2 * fs)
    out = np.empty((nsec, 4 * N_PER_CH), dtype=np.float32)
    for s in range(nsec):
        seg = pad[:, s * fs: s * fs + win].astype(np.float64)
        seg = seg - seg.mean(axis=1, keepdims=True)
        seg = np.vstack([seg, B @ seg])              # referential + bipolar double-banana
        dx = np.diff(seg, axis=1)
        ll = np.abs(dx).mean(axis=1)
        rms = np.sqrt((seg ** 2).mean(axis=1))
        P = np.abs(np.fft.rfft(seg, n=nfft, axis=1)) ** 2
        bp = np.stack([P[:, m].sum(axis=1) for m in masks], axis=1)
        per_ch = np.log(np.column_stack([ll, rms, bp]) + 1e-6)          # C x 6

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
        rh = np.column_stack([acf, pfrac, nent])                         # C x 3

        # amplitude-invariant spectral shape: relative band powers, Hjorth, spectral edge
        rel = np.log(bp / (bp.sum(axis=1, keepdims=True) + 1e-9) + 1e-6)
        ddx = np.diff(dx, axis=1)
        v0 = seg.var(axis=1) + 1e-9
        v1 = dx.var(axis=1) + 1e-9
        v2 = ddx.var(axis=1) + 1e-9
        mob = np.sqrt(v1 / v0)
        comp = np.sqrt(v2 / v1) / mob
        sef = freqs_full[np.argmax(cs[:, 1:] >= 0.9 * tot[:, None], axis=1)]
        shp = np.column_stack([rel, mob, comp, sef])                     # C x 7

        feats = np.column_stack([per_ch, rh, shp])                       # C x 16
        fr, fb = feats[:n_ref], feats[n_ref:]
        out[s] = np.concatenate([fr.max(axis=0), np.median(fr, axis=0),
                                 fb.max(axis=0), np.median(fb, axis=0)])
    return out


def featurize(items, ctx):
    from joblib import Parallel, delayed
    fs = ctx["fs"]
    B = _bipolar_matrix(list(ctx["channels"]))
    eegs = ctx["load_eeg"](items)
    return Parallel(n_jobs=8)(delayed(_features)(x, fs, ctx["n_seconds"](it), B)
                              for it, x in zip(items, eegs))


def _fit(F_list, y_list):
    X = np.vstack(F_list)
    y = np.concatenate(y_list)
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=0).fit(X, y)


def _smooth(p, w):
    if w <= 1:
        return p
    return uniform_filter1d(p, size=w, mode="nearest")


def _stack_feats(p):
    """Label-free temporal and recording-level context built from one recording's probabilities."""
    p = np.asarray(p, dtype=np.float64)
    cols = [p] + [_smooth(p, w) for w in STACK_SCALES]
    s30 = cols[2]
    mx = maximum_filter1d(s30, size=121, mode="nearest")
    q = np.percentile(s30, [50, 90, 99])
    rec = np.tile(q, (len(p), 1))
    rel = s30 - q[0]
    return np.column_stack(cols + [mx, rel, rec])


def _oof(F_train, y_train, groups):
    n = len(F_train)
    idx = np.arange(n)
    oof = [None] * n
    for tr, va in GroupKFold(n_splits=3).split(idx, groups=groups):
        clf = _fit([F_train[i] for i in tr], [y_train[i] for i in tr])
        for i in va:
            oof[i] = clf.predict_proba(F_train[i])[:, 1]
    return oof


def fit_predict(F_train, y_train, F_test, ctx):
    F_train = list(F_train)
    y_train = [np.asarray(y) for y in y_train]
    groups = np.asarray(ctx.get("groups_train", np.arange(len(F_train))))
    y_all = np.concatenate(y_train)
    clf = _fit(F_train, y_train)
    p_test = [clf.predict_proba(F)[:, 1] for F in F_test]
    if y_all.min() == y_all.max():
        return [_smooth(p, FALLBACK_W) for p in p_test]

    oof = _oof(F_train, y_train, groups)
    Z = np.vstack([_stack_feats(p) for p in oof])
    stacker = HistGradientBoostingClassifier(
        max_iter=150, learning_rate=0.05, max_leaf_nodes=8, max_depth=3,
        min_samples_leaf=1000, l2_regularization=1.0, early_stopping=False,
        random_state=0).fit(Z, y_all)
    return [stacker.predict_proba(_stack_feats(p))[:, 1] for p in p_test]