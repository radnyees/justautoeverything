"""Per-channel seizure detection. Per-second features on 8 s windows per channel (line length,
RMS, 4 band powers, 2 artifact-band powers (0.1-0.5 Hz drift/movement, 30-70 Hz muscle),
autocorrelation peak, dominant-peak power fraction, peak frequency, spectral entropy), plus
channel-minus-cross-channel-median (focality), the mean of the channel's scalp neighbours'
features (local spatial spread: focal seizures involve adjacent electrodes, single-lead artifacts
do not) and per-channel 15 s / 61 s / 181 s moving means (persistence; the 181 s scale captures
prolonged seizures / status lasting minutes in one region). Each (second, channel) row is
augmented with the global channel summaries (max/median over channels + 15 s / 61 s context +
max-over-channel persistence). A histogram gradient boosting model is trained on all channels
with the recording-level label (subsampled seconds). Each channel's probability is smoothed in
time, channels are aggregated by the mean of the top-k channel probabilities (k=1 is the max),
then the aggregate is smoothed again; both smoothing lengths and k are chosen by grouped inner CV
on the training neonates (out-of-fold channel probabilities, concatenated AUROC)."""
import numpy as np
from scipy.ndimage import uniform_filter1d
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

BANDS = [(0.5, 4), (4, 8), (8, 13), (13, 30)]
ART_BANDS = [(0.1, 0.5), (30, 70)]   # movement/drift and muscle (EMG) artifact bands
CONTEXT_WINDOWS = [15, 61]
PERSIST_WINDOWS = [15, 61, 181]
TRAIN_STRIDE = 5
CH_SMOOTH_GRID = [5, 10, 20, 40]
OUT_SMOOTH_GRID = [10, 20, 40, 80]
TOPK_GRID = [1, 2, 3, 5]
INNER_FOLDS = 3
DEFAULT_POST = (10, 10, 1)

# approximate 2D scalp grid positions of the 10-20 electrodes (x: left->right, y: back->front)
POS = {"Fp1": (-1, 2), "Fp2": (1, 2),
       "F7": (-2, 1), "F3": (-1, 1), "Fz": (0, 1), "F4": (1, 1), "F8": (2, 1),
       "T3": (-2, 0), "C3": (-1, 0), "Cz": (0, 0), "C4": (1, 0), "T4": (2, 0),
       "T5": (-2, -1), "P3": (-1, -1), "Pz": (0, -1), "P4": (1, -1), "T6": (2, -1),
       "O1": (-1, -2), "O2": (1, -2)}
ALIAS = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}


def _neighbour_matrix(channels):
    """Row-normalised adjacency (8-connected on the scalp grid, self excluded)."""
    n = len(channels)
    names = [ALIAS.get(str(c).strip(), str(c).strip()) for c in channels]
    W = np.zeros((n, n), dtype=np.float64)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i == j or a not in POS or b not in POS:
                continue
            (xa, ya), (xb, yb) = POS[a], POS[b]
            if max(abs(xa - xb), abs(ya - yb)) == 1:
                W[i, j] = 1.0
    for i in range(n):
        if W[i].sum() == 0:            # unknown channel: fall back to all other channels
            W[i] = 1.0
            W[i, i] = 0.0
    return W / W.sum(axis=1, keepdims=True)


def _features(x, fs, nsec, W):
    win = 8 * fs
    nfft2 = 2 * win
    pad = np.pad(x, ((0, 0), (win // 2, win // 2)), mode="reflect")
    freqs = np.fft.rfftfreq(win, 1 / fs)
    masks = [(freqs >= a) & (freqs < b) for a, b in BANDS + ART_BANDS]
    tot_mask = (freqs >= 0.5) & (freqs < 30)
    n_tot = tot_mask.sum()
    # dominant-peak search: 0.5 Hz bins between 0.5 and 13 Hz
    df = fs / win
    step = int(round(0.5 / df))
    i0 = int(round(0.5 / df))
    i1 = int(round(13 / df))
    nb = (i1 - i0) // step
    i1 = i0 + nb * step
    bin_centers = 0.5 + 0.5 * (np.arange(nb) + 0.5)
    # autocorrelation via zero-padded power spectrum, band-limited 0.5-15 Hz
    freqs2 = np.fft.rfftfreq(nfft2, 1 / fs)
    acf_mask = ((freqs2 >= 0.5) & (freqs2 <= 15)).astype(np.float64)
    lagmax = 2 * fs
    lag_idx = np.arange(lagmax)

    n_ch = x.shape[0]
    n_feat = 2 + len(masks) + 4
    pc = np.empty((nsec, n_ch, n_feat), dtype=np.float32)
    for s in range(nsec):
        seg = pad[:, s * fs: s * fs + win].astype(np.float64)
        seg = seg - seg.mean(axis=1, keepdims=True)
        ll = np.abs(np.diff(seg, axis=1)).mean(axis=1)
        rms = np.sqrt((seg ** 2).mean(axis=1))
        P = np.abs(np.fft.rfft(seg, axis=1)) ** 2
        bp = np.stack([P[:, m].sum(axis=1) for m in masks], axis=1)
        log_feats = np.log(np.column_stack([ll, rms, bp]) + 1e-6)          # 19 x 8

        Pt = P[:, tot_mask] + 1e-12
        tot = Pt.sum(axis=1)
        p = Pt / tot[:, None]
        ent = -(p * np.log(p)).sum(axis=1) / np.log(n_tot)
        binned = P[:, i0:i1].reshape(P.shape[0], nb, step).sum(axis=2)
        kmax = binned.argmax(axis=1)
        peak_frac = binned[np.arange(P.shape[0]), kmax] / tot
        peak_freq = bin_centers[kmax]

        P2 = np.abs(np.fft.rfft(seg, n=nfft2, axis=1)) ** 2 * acf_mask
        acf = np.fft.irfft(P2, n=nfft2, axis=1)[:, :lagmax]
        acf = acf / (acf[:, :1] + 1e-12)
        neg = acf < 0
        first = np.where(neg.any(axis=1), neg.argmax(axis=1), lagmax)
        masked = np.where(lag_idx[None, :] >= first[:, None], acf, -1.0)
        rhy = masked.max(axis=1)

        pc[s] = np.column_stack([log_feats, ent, peak_frac, peak_freq, rhy])  # 19 x 12

    med = np.median(pc, axis=1)
    out = np.concatenate([pc.max(axis=1), med], axis=1)

    # temporal context: centred moving means at multiple scales (within recording, label-free)
    ctx_feats = [uniform_filter1d(out, size=w, axis=0, mode="nearest") for w in CONTEXT_WINDOWS]
    # per-channel persistence: smooth each channel's features in time
    persist_ch = [uniform_filter1d(pc, size=w, axis=0, mode="nearest") for w in PERSIST_WINDOWS]
    persist = [p_.max(axis=1) for p_ in persist_ch]
    glob = np.hstack([out] + ctx_feats + persist).astype(np.float32)          # nsec x 108

    # local spatial spread: mean of each channel's scalp neighbours' features
    neigh = np.einsum("jk,skf->sjf", W.astype(np.float32), pc).astype(np.float32)

    # per-channel rows: own features, focality, neighbour mean, own persistence
    ch = np.concatenate([pc, pc - med[:, None, :], neigh] + persist_ch,
                        axis=2).astype(np.float32)
    return glob, ch                                                            # ch: nsec x 19 x 72


def featurize(items, ctx):
    from joblib import Parallel, delayed
    fs = ctx["fs"]
    W = _neighbour_matrix(ctx["channels"])
    eegs = ctx["load_eeg"](items)
    return Parallel(n_jobs=8)(delayed(_features)(np.asarray(x), fs, ctx["n_seconds"](it), W)
                              for it, x in zip(items, eegs))


def _rows(glob, ch, idx):
    n_ch = ch.shape[1]
    Xc = ch[idx].reshape(-1, ch.shape[2])
    Xg = np.repeat(glob[idx], n_ch, axis=0)
    return np.hstack([Xc, Xg])


def _fit(F, Y, rec_idx):
    Xs, ys = [], []
    for i in rec_idx:
        glob, ch = F[i]
        idx = np.arange(i % TRAIN_STRIDE, len(glob), TRAIN_STRIDE)
        Xs.append(_rows(glob, ch, idx))
        ys.append(np.repeat(np.asarray(Y[i])[idx], ch.shape[1]))
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    del Xs, ys
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                          random_state=0).fit(X, y)


def _channel_probs(clf, glob, ch):
    nsec, n_ch = ch.shape[0], ch.shape[1]
    return clf.predict_proba(_rows(glob, ch, np.arange(nsec)))[:, 1].reshape(nsec, n_ch)


def _smooth_ch(p, cs):
    return uniform_filter1d(p, size=cs, axis=0, mode="nearest")


def _topk(p, k):
    # mean of the k largest channel probabilities per second (k=1 -> max)
    if k <= 1:
        return p.max(axis=1)
    k = min(k, p.shape[1])
    return -np.partition(-p, k - 1, axis=1)[:, :k].mean(axis=1)


def _post(p, cs, os_, k):
    agg = _topk(_smooth_ch(p, cs), k)
    return uniform_filter1d(agg, size=os_, mode="nearest")


def _select_post(F_train, y_train, groups):
    """Grouped inner CV: out-of-fold channel probabilities -> best (channel smoothing,
    output smoothing, top-k channel aggregation)."""
    n = len(F_train)
    oof = [None] * n
    for tr, va in GroupKFold(n_splits=INNER_FOLDS).split(np.zeros(n), groups=groups):
        ytr = np.concatenate([np.asarray(y_train[i]) for i in tr])
        if ytr.min() == ytr.max():
            return DEFAULT_POST
        clf = _fit(F_train, y_train, tr)
        for i in va:
            oof[i] = _channel_probs(clf, *F_train[i]).astype(np.float32)
        del clf
    y_all = np.concatenate([np.asarray(y) for y in y_train])
    if y_all.min() == y_all.max():
        return DEFAULT_POST
    best, best_auc = DEFAULT_POST, -1.0
    for cs in CH_SMOOTH_GRID:
        sm = [_smooth_ch(p, cs) for p in oof]
        for k in TOPK_GRID:
            aggs = [_topk(p, k) for p in sm]
            for os_ in OUT_SMOOTH_GRID:
                s = np.concatenate([uniform_filter1d(a, size=os_, mode="nearest") for a in aggs])
                auc = roc_auc_score(y_all, s)
                if auc > best_auc:
                    best, best_auc = (cs, os_, k), auc
        del sm
    return best


def fit_predict(F_train, y_train, F_test, ctx):
    groups = np.asarray(ctx.get("groups_train", np.arange(len(F_train))))
    cs, os_, k = _select_post(F_train, y_train, groups)
    clf = _fit(F_train, y_train, np.arange(len(F_train)))
    return [_post(_channel_probs(clf, glob, ch), cs, os_, k) for glob, ch in F_test]