"""Per-second features on 8 s windows (line length, RMS, 4 band powers, rhythmicity:
band-limited autocorrelation peak, dominant spectral-peak power fraction, negative spectral
entropy, plus artifact descriptors: relative EMG (30-48 Hz) and drift (<0.5 Hz) power; max and
median over channels; global mean inter-channel correlation of the 0.5-30 Hz signal)
+ multi-scale temporal context (rolling 15 s / 61 s means, 15 s max)
+ local-background contrast (15 s mean minus rolling 10-min median, per recording)
+ channel-consistent context: per-channel 15 s mean (and its contrast to that channel's own
  10-min background), THEN max AND 3rd-largest over channels -> sustained focal activity that
  involves a region of electrodes (not a single bad electrode)
+ long-scale channel-consistent contrast: per-channel 61 s mean minus that channel's own 10-min
  background, max and 3rd-largest over channels -> minute-long focal discharges
+ regional (scalp-neighbourhood) contrast: each channel's 15 s / 61 s contrast averaged with its
  adjacent 10-20 electrodes, max over channels -> contiguous focal discharges
+ seizure-burden-robust contrast: per-channel 61 s mean minus that channel's rolling 30-min
  20th percentile (stays at background level even when seizures fill most of the window, e.g.
  status epilepticus), max / 3rd-largest / regional max over channels
-> strongly regularized histogram gradient boosting
-> moving-average smoothing whose width is chosen by inner grouped CV (concatenated AUROC)."""
import numpy as np
from scipy.ndimage import uniform_filter1d, maximum_filter1d, median_filter
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

BANDS = [(0.5, 4), (4, 8), (8, 13), (13, 30)]
N_PER_CH = 11
SMOOTH_CANDIDATES = [1, 5, 10, 20, 30, 45, 60, 90, 120]
BG_WIN = 601                                                    # 10-min local background
BG_STRIDE = 30                                                  # per-channel background grid
LQ_WIN = 1801                                                   # 30-min low-quantile background
LQ_Q = 20                                                       # percentile for that background
TOP_K = 3                                                       # regional spread: k-th channel
LONG_WIN = 61                                                   # minute-scale channel context
NEIGHBOURS = {                                                  # adjacent 10-20 electrodes
    "Fp1": ["Fp2", "F3", "F7", "Fz"], "Fp2": ["Fp1", "F4", "F8", "Fz"],
    "F3": ["Fp1", "F7", "Fz", "C3"], "F4": ["Fp2", "F8", "Fz", "C4"],
    "C3": ["F3", "T3", "Cz", "P3"], "C4": ["F4", "T4", "Cz", "P4"],
    "P3": ["C3", "T5", "Pz", "O1"], "P4": ["C4", "T6", "Pz", "O2"],
    "O1": ["P3", "T5", "O2"], "O2": ["P4", "T6", "O1"],
    "F7": ["Fp1", "F3", "T3"], "F8": ["Fp2", "F4", "T4"],
    "T3": ["F7", "C3", "T5"], "T4": ["F8", "C4", "T6"],
    "T5": ["T3", "P3", "O1"], "T6": ["T4", "P4", "O2"],
    "Fz": ["Fp1", "Fp2", "F3", "F4", "Cz"], "Cz": ["Fz", "C3", "C4", "Pz"],
    "Pz": ["Cz", "P3", "P4"],
}


def _neighbour_matrix(channels):
    # row-normalized averaging over each electrode and its scalp neighbours
    names = [str(c) for c in channels]
    idx = {c.lower(): i for i, c in enumerate(names)}
    n = len(names)
    A = np.eye(n, dtype=np.float32)
    for i, c in enumerate(names):
        for nb in NEIGHBOURS.get(c, []):
            j = idx.get(nb.lower())
            if j is not None:
                A[i, j] = 1.0
    return A / A.sum(axis=1, keepdims=True)


def _channel_background(flat, nsec, win=BG_WIN, q=50):
    # rolling per-column percentile (default: 10-min median), evaluated every BG_STRIDE s and
    # linearly interpolated
    half = win // 2
    centers = np.arange(0, nsec, BG_STRIDE)
    bgc = np.stack([np.percentile(flat[max(0, c - half): c + half + 1], q, axis=0)
                    for c in centers]).astype(np.float32)
    pos = np.arange(nsec) / BG_STRIDE
    i0 = np.minimum(np.floor(pos).astype(int), len(centers) - 1)
    i1 = np.minimum(i0 + 1, len(centers) - 1)
    f = (pos - i0)[:, None].astype(np.float32)
    return bgc[i0] + (bgc[i1] - bgc[i0]) * f


def _features(x, fs, nsec, A):
    win = 8 * fs
    pad = np.pad(x, ((0, 0), (win // 2, win // 2)), mode="reflect")
    freqs = np.fft.rfftfreq(win, 1 / fs)
    masks = [(freqs >= a) & (freqs < b) for a, b in BANDS]
    m_tot = (freqs >= 0.5) & (freqs < 30)
    m_hf = (freqs >= 30) & (freqs < 48)                         # EMG, below 50 Hz mains
    m_vlf = (freqs > 0) & (freqs < 0.5)                         # drift / movement / respiration
    n_tot = int(m_tot.sum())
    dom_bins = np.where((freqs >= 0.5) & (freqs < 8))[0]
    half = int(round(0.5 / (freqs[1] - freqs[0])))              # +-0.5 Hz around the peak
    # zero-padded FFT for a linear (non-circular) band-limited autocorrelation
    nfft = 2 * win
    acmask = ((np.fft.rfftfreq(nfft, 1 / fs) >= 0.5) & (np.fft.rfftfreq(nfft, 1 / fs) < 15))
    lo, hi = int(0.12 * fs), int(2 * fs)                        # lags 0.12-2 s (~0.5-8 Hz)
    lag_idx = np.arange(hi - lo)[None, :]
    nch = x.shape[0]
    ch = np.arange(nch)
    iu = np.triu_indices(nch, k=1)
    out = np.empty((nsec, 2 * N_PER_CH + 1), dtype=np.float32)
    per_all = np.empty((nsec, nch, N_PER_CH), dtype=np.float32)
    for s in range(nsec):
        seg = pad[:, s * fs: s * fs + win]
        seg = seg - seg.mean(axis=1, keepdims=True)
        ll = np.abs(np.diff(seg, axis=1)).mean(axis=1)
        rms = np.sqrt((seg ** 2).mean(axis=1))
        Fc = np.fft.rfft(seg, axis=1)
        P = np.abs(Fc) ** 2
        bp = np.stack([P[:, m].sum(axis=1) for m in masks], axis=1)
        # spectral entropy (low = rhythmic)
        Pt = P[:, m_tot]
        tot = Pt.sum(axis=1) + 1e-12
        p = Pt / tot[:, None]
        negH = (p * np.log(p + 1e-20)).sum(axis=1) / np.log(n_tot)
        # dominant-peak power fraction
        k = dom_bins[np.argmax(P[:, dom_bins], axis=1)]
        cs = np.concatenate([np.zeros((P.shape[0], 1)), np.cumsum(P, axis=1)], axis=1)
        pk = (cs[ch, k + half + 1] - cs[ch, np.maximum(k - half, 0)]) / tot
        # autocorrelation rhythmicity: max of band-limited ACF after its first zero crossing
        P2 = np.abs(np.fft.rfft(seg, n=nfft, axis=1)) ** 2 * acmask
        ac = np.fft.irfft(P2, n=nfft, axis=1)[:, :hi]
        ac = ac / (ac[:, :1] + 1e-12)
        sub = ac[:, lo:hi]
        neg = sub < 0
        first = np.where(neg.any(axis=1), neg.argmax(axis=1), 0)
        rhy = np.where(lag_idx >= first[:, None], sub, -1.0).max(axis=1)
        # artifact descriptors: relative EMG and drift power
        hf = np.log(P[:, m_hf].sum(axis=1) / tot + 1e-8)
        vlf = np.log(P[:, m_vlf].sum(axis=1) / tot + 1e-8)
        # common-mode activity: mean inter-channel correlation of the 0.5-30 Hz signal
        xb = np.fft.irfft(Fc * m_tot, n=win, axis=1)
        xb = xb / (np.sqrt((xb ** 2).sum(axis=1, keepdims=True)) + 1e-12)
        corr = float((xb @ xb.T)[iu].mean())
        per_ch = np.column_stack([np.log(np.column_stack([ll, rms, bp]) + 1e-6),
                                  rhy, pk, negH, hf, vlf])           # 19 x 11
        per_all[s] = per_ch
        out[s, :2 * N_PER_CH] = np.concatenate([per_ch.max(axis=0), np.median(per_ch, axis=0)])
        out[s, -1] = corr
    # multi-scale temporal context: sustained/evolving activity vs transient events
    m15 = uniform_filter1d(out, size=15, axis=0, mode="nearest")
    m61 = uniform_filter1d(out, size=61, axis=0, mode="nearest")
    x15 = maximum_filter1d(out, size=15, axis=0, mode="nearest")
    # local-background contrast: sustained deviation from this neonate's surrounding 10 min
    bw = min(BG_WIN, nsec if nsec % 2 == 1 else max(nsec - 1, 1))
    bg = median_filter(out, size=(bw, 1), mode="nearest")
    # channel-consistent context: average each channel over time first, then aggregate channels
    pc = uniform_filter1d(per_all, size=15, axis=0, mode="nearest")
    flat = pc.reshape(nsec, -1)
    pc_bg = _channel_background(flat, nsec)
    con = (flat - pc_bg).reshape(nsec, nch, N_PER_CH)
    pc_max = pc.max(axis=1)
    pc_con = con.max(axis=1)
    # regional spread: k-th largest channel (high only if several electrodes are involved)
    kk = min(TOP_K, nch)
    pc_k = -np.partition(-pc, kk - 1, axis=1)[:, kk - 1]
    con_k = -np.partition(-con, kk - 1, axis=1)[:, kk - 1]
    # minute-scale channel-consistent contrast: focal activity persisting on the same electrodes
    flat61 = uniform_filter1d(per_all, size=LONG_WIN, axis=0, mode="nearest").reshape(nsec, -1)
    con61 = (flat61 - _channel_background(flat61, nsec)).reshape(nsec, nch, N_PER_CH)
    con61_max = con61.max(axis=1)
    con61_k = -np.partition(-con61, kk - 1, axis=1)[:, kk - 1]
    # scalp-neighbourhood contrast: contiguous electrodes deviating from their own background
    reg15 = np.einsum("tcf,dc->tdf", con, A).max(axis=1)
    reg61 = np.einsum("tcf,dc->tdf", con61, A).max(axis=1)
    # seizure-burden-robust contrast: 30-min low-percentile reference per channel, so that
    # long / recurrent seizures that dominate the 10-min median still stand out
    conq = (flat61 - _channel_background(flat61, nsec, LQ_WIN, LQ_Q)).reshape(nsec, nch,
                                                                              N_PER_CH)
    conq_max = conq.max(axis=1)
    conq_k = -np.partition(-conq, kk - 1, axis=1)[:, kk - 1]
    regq = np.einsum("tcf,dc->tdf", conq, A).max(axis=1)
    return np.hstack([out, m15, m61, x15, m15 - bg, pc_max, pc_con,
                      pc_k, con_k, con61_max, con61_k, reg15, reg61,
                      conq_max, conq_k, regq]).astype(np.float32)


def featurize(items, ctx):
    from joblib import Parallel, delayed
    fs = ctx["fs"]
    eegs = ctx["load_eeg"](items)
    chans = ctx.get("channels")
    if chans is None or len(chans) == 0:
        chans = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8",
                 "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz"]
    A = _neighbour_matrix(chans)
    return Parallel(n_jobs=8)(delayed(_features)(np.asarray(x), fs, ctx["n_seconds"](it), A)
                              for it, x in zip(items, eegs))


def _model():
    # neonate-independent generalization: autocorrelated seconds make small leaves memorize
    # individual recordings, so use large leaves, small trees, L2 and a slower learning rate
    return HistGradientBoostingClassifier(max_iter=600, learning_rate=0.05, max_leaf_nodes=15,
                                          min_samples_leaf=1000, l2_regularization=1.0,
                                          max_bins=255, random_state=0)


def _smooth(p, w):
    return p if w <= 1 else uniform_filter1d(p, size=w, mode="nearest")


def _choose_smoothing(F_train, y_train, groups):
    # inner grouped CV: out-of-fold raw probabilities per training recording, then pick the
    # moving-average width maximizing concatenated AUROC (seizures last tens of s to minutes)
    n = len(F_train)
    oof = [None] * n
    for tr, va in GroupKFold(n_splits=3).split(np.zeros(n), groups=groups):
        clf = _model().fit(np.vstack([F_train[i] for i in tr]),
                           np.concatenate([y_train[i] for i in tr]))
        for i in va:
            oof[i] = clf.predict_proba(F_train[i])[:, 1]
    y = np.concatenate(y_train)
    if y.min() == y.max():
        return 10
    scores = [roc_auc_score(y, np.concatenate([_smooth(p, w) for p in oof]))
              for w in SMOOTH_CANDIDATES]
    return SMOOTH_CANDIDATES[int(np.argmax(scores))]


def fit_predict(F_train, y_train, F_test, ctx):
    groups = ctx.get("groups_train")
    if groups is None or len(groups) != len(F_train):
        groups = np.arange(len(F_train))
    groups = np.asarray(groups)
    w = _choose_smoothing(F_train, y_train, groups) if len(np.unique(groups)) >= 3 else 10
    clf = _model().fit(np.vstack(F_train), np.concatenate(y_train))
    return [_smooth(clf.predict_proba(F)[:, 1], w) for F in F_test]