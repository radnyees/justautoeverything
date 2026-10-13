"""Per-second features on 8 s windows: line length, RMS, 4 band powers per channel (max and median
over channels) + rhythmicity features per channel (band-limited autocorrelation peak, spectral
entropy, dominant-peak power ratio; max/median/min over channels) + spatial-spread features
(across-channel std of the 9 per-channel features, fraction of strongly rhythmic channels,
average-reference 0.5-10 Hz inter-channel correlation: global, max local-neighbourhood, and
neighbourhood of the most rhythmic channel) + hemispheric asymmetry features (max homologous-pair
|left-right| difference and |hemisphere-mean difference| of the 9 per-channel features, mean
homologous-pair correlation) + regional focality features (9 per-channel features averaged over
each electrode's 10-20 neighbourhood: max over regions and max-minus-median regional value)
+ multi-scale temporal context (centred moving averages over 15 s and 61 s) + deviation from a
local 10-min running-median background (raw and 15 s averaged)
-> average of histogram gradient boosting and a large-leaf ExtraTrees forest
-> 10 s moving-average smoothing."""
import numpy as np
from scipy.ndimage import uniform_filter1d, median_filter
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier

BANDS = [(0.5, 4), (4, 8), (8, 13), (13, 30)]
CONTEXT = (15, 61)
BASE_WIN = 601      # seconds, centred running-median background
BASE_STEP = 10      # subsample step for the running median (speed)

# 10-20 spatial neighbours, channel order: Fp1 Fp2 F3 F4 C3 C4 P3 P4 O1 O2 F7 F8 T3 T4 T5 T6 Fz Cz Pz
NEIGH = [[1, 2, 10], [0, 3, 11], [0, 4, 10, 16], [1, 5, 11, 16], [2, 6, 12, 17], [3, 7, 13, 17],
         [4, 8, 14, 18], [5, 9, 15, 18], [6, 14, 9], [7, 15, 8], [0, 2, 12], [1, 3, 13],
         [10, 4, 14], [11, 5, 15], [12, 6, 8], [13, 7, 9], [2, 3, 17], [16, 4, 5, 18], [17, 6, 7]]
# homologous pairs: Fp1/Fp2, F3/F4, C3/C4, P3/P4, O1/O2, F7/F8, T3/T4, T5/T6
LEFT = np.array([0, 2, 4, 6, 8, 10, 12, 14])
RIGHT = np.array([1, 3, 5, 7, 9, 11, 13, 15])
N_FEAT = 71


def _neigh_matrix(n_ch, self_loop=False):
    W = np.zeros((n_ch, n_ch))
    for i, nb in enumerate(NEIGH[:n_ch]):
        for j in nb:
            if j < n_ch:
                W[i, j] = 1.0
        if self_loop:
            W[i, i] = 1.0
    W /= np.maximum(W.sum(axis=1, keepdims=True), 1.0)
    return W


def _features(x, fs, nsec):
    n_ch = x.shape[0]
    win = 8 * fs
    nfft2 = 2 * win                      # zero-padded FFT -> linear (non-circular) autocorrelation
    lag_max = 2 * fs
    pad = np.pad(x, ((0, 0), (win // 2, win // 2)), mode="reflect")
    freqs = np.fft.rfftfreq(win, 1 / fs)
    masks = [(freqs >= a) & (freqs < b) for a, b in BANDS]
    f2 = np.fft.rfftfreq(nfft2, 1 / fs)
    acf_band = ((f2 >= 0.5) & (f2 <= 10)).astype(np.float64)
    ent_mask = (freqs >= 0.5) & (freqs < 30)
    low_mask = (freqs >= 0.5) & (freqs < 8)
    sync_band = ((freqs >= 0.5) & (freqs <= 10)).astype(np.float64)
    n_ent = int(ent_mask.sum())
    W = _neigh_matrix(n_ch)
    R = _neigh_matrix(n_ch, self_loop=True)                              # regional averaging
    off = ~np.eye(n_ch, dtype=bool)
    out = np.empty((nsec, N_FEAT), dtype=np.float32)
    for s in range(nsec):
        seg = pad[:, s * fs: s * fs + win].astype(np.float64)
        seg = seg - seg.mean(axis=1, keepdims=True)
        ll = np.abs(np.diff(seg, axis=1)).mean(axis=1)
        rms = np.sqrt((seg ** 2).mean(axis=1))
        S = np.fft.rfft(seg, axis=1)
        P = np.abs(S) ** 2
        bp = np.stack([P[:, m].sum(axis=1) for m in masks], axis=1)
        per_ch = np.log(np.column_stack([ll, rms, bp]) + 1e-6)          # 19 x 6

        # band-limited autocorrelation: height of first repeat peak after first zero crossing
        P2 = (np.abs(np.fft.rfft(seg, n=nfft2, axis=1)) ** 2) * acf_band
        ac = np.fft.irfft(P2, n=nfft2, axis=1)[:, :lag_max + 1]
        ac = ac / (ac[:, :1] + 1e-12)
        after_neg = np.cumsum(ac < 0, axis=1) > 0
        acp = np.maximum(np.where(after_neg, ac, -1.0).max(axis=1), 0.0)

        # normalized spectral entropy over 0.5-30 Hz
        Pe = P[:, ent_mask]
        tot = Pe.sum(axis=1) + 1e-12
        pn = Pe / tot[:, None]
        ent = -(pn * np.log(pn + 1e-12)).sum(axis=1) / np.log(n_ent)

        # fraction of power in a narrow band around the dominant 0.5-8 Hz peak
        Pl = uniform_filter1d(P[:, low_mask], size=5, axis=1, mode="constant")
        pk = Pl.max(axis=1) * 5 / tot

        rh = np.column_stack([acp, ent, pk])                             # 19 x 3

        # spatial spread: average-reference, 0.5-10 Hz band-limited inter-channel correlation
        Sf = S * sync_band
        Sf = Sf - Sf.mean(axis=0, keepdims=True)                          # common average ref
        xf = np.fft.irfft(Sf, n=win, axis=1)
        xf = xf - xf.mean(axis=1, keepdims=True)
        xf = xf / (np.sqrt((xf ** 2).mean(axis=1, keepdims=True)) + 1e-9)
        C = np.abs(xf @ xf.T) / win
        glob_sync = C[off].mean()
        loc = (W * C).sum(axis=1)                                         # mean |corr| to neighbours
        loc_max = loc.max()
        loc_rh = loc[int(np.argmax(acp))]
        frac_rh = (acp > 0.5).mean()
        allf = np.column_stack([per_ch, rh])                              # 19 x 9
        spread = allf.std(axis=0)                                         # 9

        # hemispheric asymmetry: focal/unilateral seizures vs symmetric state changes/artifacts
        dlr = allf[LEFT] - allf[RIGHT]                                    # 8 x 9
        asym_pair = np.abs(dlr).max(axis=0)                               # 9
        asym_hemi = np.abs(dlr.mean(axis=0))                              # 9
        homo_corr = C[LEFT, RIGHT].mean()

        # regional focality: neighbourhood-averaged features (coherent regional discharge vs
        # isolated single-electrode artifact vs global state change)
        reg = R @ allf                                                    # 19 x 9
        reg_max = reg.max(axis=0)                                         # 9
        reg_foc = reg_max - np.median(reg, axis=0)                        # 9

        out[s] = np.concatenate([per_ch.max(axis=0), np.median(per_ch, axis=0),
                                 rh.max(axis=0), np.median(rh, axis=0), rh.min(axis=0),
                                 spread, [glob_sync, loc_max, loc_rh, frac_rh],
                                 asym_pair, asym_hemi, [homo_corr],
                                 reg_max, reg_foc])
    return out


def _local_baseline(F):
    """Centred running median over ~BASE_WIN seconds (computed on a subsampled grid)."""
    sub = F[::BASE_STEP]
    size = max(1, (BASE_WIN // BASE_STEP) | 1)
    med = median_filter(sub, size=(size, 1), mode="nearest")
    idx = np.minimum(np.arange(len(F)) // BASE_STEP, len(sub) - 1)
    return med[idx]


def _with_context(F):
    """Append centred moving averages over several time scales and the deviation from the
    local running-median background (per recording, label-free)."""
    F = np.asarray(F, dtype=np.float32)
    parts = [F] + [uniform_filter1d(F, size=w, axis=0, mode="nearest") for w in CONTEXT]
    D = F - _local_baseline(F)
    parts += [D, uniform_filter1d(D, size=CONTEXT[0], axis=0, mode="nearest")]
    return np.hstack(parts)


def featurize(items, ctx):
    from joblib import Parallel, delayed
    fs = ctx["fs"]
    eegs = ctx["load_eeg"](items)
    return Parallel(n_jobs=8)(delayed(_features)(np.asarray(x), fs, ctx["n_seconds"](it))
                              for it, x in zip(items, eegs))


def fit_predict(F_train, y_train, F_test, ctx):
    X = np.vstack([_with_context(F) for F in F_train]).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.concatenate(y_train)
    hgb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=0).fit(X, y)
    # low-variance randomized forest: large leaves resist memorising neonate-specific patterns
    et = ExtraTreesClassifier(n_estimators=200, max_features="sqrt", min_samples_leaf=100,
                              n_jobs=8, random_state=0).fit(X, y)
    del X
    k = np.ones(10) / 10
    preds = []
    for F in F_test:
        Xt = np.nan_to_num(_with_context(F), nan=0.0, posinf=0.0, neginf=0.0)
        p = 0.5 * hgb.predict_proba(Xt)[:, 1] + 0.5 * et.predict_proba(Xt)[:, 1]
        preds.append(np.convolve(p, k, mode="same"))
    return preds