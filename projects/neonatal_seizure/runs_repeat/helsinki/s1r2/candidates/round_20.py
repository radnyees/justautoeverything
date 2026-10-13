"""Per-second features on 8 s windows: line length, RMS, 4 band powers, plus rhythmicity features
(spectral entropy, dominant-peak power fraction, post-zero-crossing autocorrelation peak) per channel;
max / 3rd-largest / median / min over channels (3rd-largest = robust focal statistic insensitive to a
single bad electrode), plus inter-channel synchrony features (largest-eigenvalue fraction and mean |r|
of the 0.5-30 Hz correlation matrix, per-channel mean |r| max/median/min), plus centered 15 s / 61 s /
181 s moving averages of every feature (multi-scale temporal context: minutes-long seizures) ->
histogram gradient boosting -> moving-average smoothing whose width is chosen by grouped inner CV
(concatenated AUROC)."""
import numpy as np
from scipy.ndimage import uniform_filter1d
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

BANDS = [(0.5, 4), (4, 8), (8, 13), (13, 30)]
CONTEXT = (15, 61, 181)                              # seconds, centered moving-average windows
NFEAT = 36 + 5
SMOOTH_GRID = (1, 5, 10, 20, 40, 80)                 # candidate post-smoothing widths (s)
TOPK = 3                                             # k-th largest channel as robust focal aggregate


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
        kth = np.partition(per_ch, nch - TOPK, axis=0)[nch - TOPK]
        out[s] = np.concatenate([per_ch.max(axis=0), np.median(per_ch, axis=0),
                                 per_ch.min(axis=0), sync, kth])
    # temporal context: centered moving averages of each feature within this recording
    ctxf = [uniform_filter1d(out, size=w, axis=0, mode="nearest") for w in CONTEXT]
    return np.hstack([out] + ctxf).astype(np.float32)


def featurize(items, ctx):
    from joblib import Parallel, delayed
    fs = ctx["fs"]
    eegs = ctx["load_eeg"](items)
    return Parallel(n_jobs=8)(delayed(_features)(np.asarray(x), fs, ctx["n_seconds"](it))
                              for it, x in zip(items, eegs))


def _model():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=0)


def _smooth(p, w):
    if w <= 1:
        return p
    return uniform_filter1d(p, size=w, mode="nearest")


def _choose_smoothing(F_train, y_train, ctx):
    n = len(F_train)
    groups = np.asarray(ctx.get("groups_train", np.arange(n)))
    if len(groups) != n:
        groups = np.arange(n)
    n_splits = min(3, len(np.unique(groups)))
    if n_splits < 2:
        return 10
    oof = [None] * n
    for tr, va in GroupKFold(n_splits=n_splits).split(np.arange(n), groups=groups):
        X = np.vstack([F_train[i][::2] for i in tr])
        y = np.concatenate([np.asarray(y_train[i])[::2] for i in tr])
        if len(np.unique(y)) < 2:
            return 10
        clf = _model().fit(X, y)
        for i in va:
            oof[i] = clf.predict_proba(F_train[i])[:, 1]
    yall = np.concatenate([np.asarray(v) for v in y_train])
    if len(np.unique(yall)) < 2:
        return 10
    best_w, best_auc = 10, -1.0
    for w in SMOOTH_GRID:
        auc = roc_auc_score(yall, np.concatenate([_smooth(p, w) for p in oof]))
        if auc > best_auc:
            best_w, best_auc = w, auc
    return best_w


def fit_predict(F_train, y_train, F_test, ctx):
    w = _choose_smoothing(F_train, y_train, ctx)
    X = np.vstack(F_train)
    y = np.concatenate(y_train)
    clf = _model().fit(X, y)
    return [_smooth(clf.predict_proba(F)[:, 1], w) for F in F_test]