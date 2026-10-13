"""Per-second features on 8 s windows: line length, RMS, 4 band powers, plus rhythmicity features
(spectral entropy, dominant-peak power fraction, post-zero-crossing autocorrelation peak) per channel;
max / median / min over channels, plus inter-channel synchrony features (largest-eigenvalue fraction
and mean |r| of the 0.5-30 Hz correlation matrix, per-channel mean |r| max/median/min), plus centered
15 s / 61 s moving averages of every feature (temporal context) -> histogram gradient boosting ->
learned temporal post-processing: a second-stage GBM on multi-scale context of out-of-fold (inner
grouped CV) first-stage probabilities -> 10 s moving-average smoothing."""
import numpy as np
from scipy.ndimage import uniform_filter1d, maximum_filter1d
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

BANDS = [(0.5, 4), (4, 8), (8, 13), (13, 30)]
CONTEXT = (15, 61)                                   # seconds, centered moving-average windows
NFEAT = 27 + 5
STACK_MEAN = (5, 15, 31, 61, 121, 301)               # seconds, context of first-stage logits
STACK_MAX = (15, 61)
N_INNER = 3


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
    out = np.empty((nsec, NFEAT), dtype=np.float32)
    for s in range(nsec):
        seg = pad[:, s * fs: s * fs + win].astype(np.float64)
        seg = seg - seg.mean(axis=1, keepdims=True)
        ll = np.abs(np.diff(seg, axis=1)).mean(axis=1)
        rms = np.sqrt((seg ** 2).mean(axis=1))
        Xf = np.fft.rfft(seg, n=nfft, axis=1)
        P = np.abs(Xf) ** 2
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
        per_ch = np.column_stack([np.log(np.column_stack([ll, rms, bp]) + 1e-6),
                                  sent, pfrac, rhy])     # 19 x 9
        # inter-channel synchrony of the 0.5-30 Hz band-limited signals (via Parseval)
        Xb = Xf[:, full]
        C = (Xb @ Xb.conj().T).real
        d = np.sqrt(np.clip(np.diag(C), 0, None)) + 1e-12
        R = C / np.outer(d, d)
        absR = np.abs(R)
        ch_r = (absR.sum(axis=1) - np.diag(absR)) / (nch - 1)
        lam = np.linalg.eigvalsh(R)[-1] / max(np.trace(R), 1e-12)
        sync = np.array([lam, ch_r.mean(), ch_r.max(), np.median(ch_r), ch_r.min()])
        out[s] = np.concatenate([per_ch.max(axis=0), np.median(per_ch, axis=0),
                                 per_ch.min(axis=0), sync])
    # temporal context: centered moving averages of each feature within this recording
    ctxf = [uniform_filter1d(out, size=w, axis=0, mode="nearest") for w in CONTEXT]
    return np.hstack([out] + ctxf).astype(np.float32)


def featurize(items, ctx):
    from joblib import Parallel, delayed
    fs = ctx["fs"]
    eegs = ctx["load_eeg"](items)
    return Parallel(n_jobs=8)(delayed(_features)(np.asarray(x), fs, ctx["n_seconds"](it))
                              for it, x in zip(items, eegs))


def _stage1():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=0)


def _prob_feats(p):
    """Multi-scale temporal context of one recording's first-stage seizure logits."""
    lp = np.log(p + 1e-4) - np.log(1 - p + 1e-4)
    cols = [lp]
    cols += [uniform_filter1d(lp, size=w, mode="nearest") for w in STACK_MEAN]
    cols += [maximum_filter1d(lp, size=w, mode="nearest") for w in STACK_MAX]
    return np.column_stack(cols).astype(np.float32)


def fit_predict(F_train, y_train, F_test, ctx):
    X = np.vstack(F_train)
    y = np.concatenate(y_train)
    n = len(F_train)
    groups = np.asarray(ctx.get("groups_train", np.arange(n)))
    if len(groups) != n:
        groups = np.arange(n)

    # out-of-fold first-stage probabilities on training recordings (grouped inner CV)
    oof = [None] * n
    k_in = min(N_INNER, len(np.unique(groups)))
    for tr, va in GroupKFold(n_splits=k_in).split(np.arange(n), groups=groups):
        m = _stage1().fit(np.vstack([F_train[i] for i in tr]),
                          np.concatenate([y_train[i] for i in tr]))
        for i in va:
            oof[i] = m.predict_proba(F_train[i])[:, 1]

    clf = _stage1().fit(X, y)

    # second stage: learned temporal post-processing of the probability trajectories
    S = np.vstack([_prob_feats(p) for p in oof])
    clf2 = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                          min_samples_leaf=500, random_state=0).fit(S, y)
    k = np.ones(10) / 10
    preds = []
    for F in F_test:
        p1 = clf.predict_proba(F)[:, 1]
        p2 = clf2.predict_proba(_prob_feats(p1))[:, 1]
        preds.append(np.convolve(p2, k, mode="same"))
    return preds