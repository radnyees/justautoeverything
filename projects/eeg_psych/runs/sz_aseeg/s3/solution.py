"""Two-model late fusion.
Model 1: log relative band power (delta/theta/alpha/beta/gamma) per channel from a median-averaged Welch PSD,
per-channel Higuchi fractal dimension, per-channel multiscale permutation entropy (order 4, delays 1 and 3),
per-channel multiscale sample entropy (m=2, r=0.2 SD, coarse-graining scales 1 and 4),
per-channel wPLI node strength (delta/theta/alpha/beta) plus wPLI graph topology (weighted clustering, global
efficiency on normalized graphs), and EEG microstate statistics (4 label-free group templates ordered to
canonical A/B/C/D; coverage, duration, occurrence, GEV); standardized L2 logistic regression.
Model 2: band-limited (1-8, 8-13, 13-30 Hz) spatial covariance matrices (average reference, artifact epochs
rejected, trace-normalized, shrunk) projected to the Riemannian tangent space at the training-fold mean;
standardized L2 logistic regression. Predictions = mean of the two probabilities."""
import numpy as np
from scipy.signal import welch, butter, sosfiltfilt
from scipy.ndimage import uniform_filter1d
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import shortest_path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import antropy as ant

BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
CONN_BANDS = [(1, 4), (4, 8), (8, 13), (13, 30)]   # gamma left out: dominated by muscle (EMG) coupling
COV_BANDS = [(1, 8), (8, 13), (13, 30)]             # slow, alpha, beta spatial covariance
PE_DELAYS = [1, 3]                                  # fast (beta/gamma) and slower (theta/alpha) ordinal scale
SE_SCALES = [1, 4]                                  # sample entropy: fine and coarse (<16 Hz) time scale
SE_EP_SEC = 8
SE_MAX_EP = 20
N_MS = 4                                            # classic microstate classes A-D


def _bandpower(x, fs):
    x = np.asarray(x, dtype=np.float64)
    # median across Welch segments: segments hit by blinks/movement/muscle bursts do not dominate
    f, p = welch(x, fs=fs, nperseg=int(2 * fs), axis=-1, average="median")
    tot = p[:, (f >= 1) & (f <= 45)].sum(axis=1, keepdims=True) + 1e-12
    return np.log(np.stack([p[:, (f >= a) & (f < b)].sum(axis=1) for a, b in BANDS], axis=1) / tot + 1e-12)


def _filt(x, fs):
    sos = butter(4, [1, 45], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, np.asarray(x, dtype=np.float64), axis=-1)


def _complexity(x, fs, ep_sec=4, kmax=10):
    # band-pass removes drift and line noise, which would otherwise dominate the complexity estimates
    x = _filt(x, fs)
    L = int(ep_sec * fs)
    n_ep = max(1, x.shape[1] // L)
    hfd = np.zeros(x.shape[0])
    pe = np.zeros((x.shape[0], len(PE_DELAYS)))
    for ch in range(x.shape[0]):
        hv, pv = [], []
        for e in range(n_ep):
            seg = x[ch, e * L:(e + 1) * L]
            if seg.size < 2 * kmax or np.std(seg) < 1e-9:
                continue
            seg = np.ascontiguousarray(seg)
            hv.append(ant.higuchi_fd(seg, kmax=kmax))
            pv.append([ant.perm_entropy(seg, order=4, delay=d, normalize=True) for d in PE_DELAYS])
        hfd[ch] = np.median(hv) if hv else 1.5
        pe[ch] = np.median(np.asarray(pv), axis=0) if pv else 0.9
    return hfd, pe.ravel()


def _sampen(x, r, m=2):
    """Sample entropy (Chebyshev distance, self-matches excluded, +1 smoothing for short series)."""
    n = len(x) - m
    if n < 10:
        return np.nan
    x = x.astype(np.float32)
    d = np.zeros((n, n), dtype=np.float32)
    for k in range(m):
        np.maximum(d, np.abs(x[k:k + n, None] - x[None, k:k + n]), out=d)
    B = np.count_nonzero(d <= r) - n
    np.maximum(d, np.abs(x[m:m + n, None] - x[None, m:m + n]), out=d)
    A = np.count_nonzero(d <= r) - n
    return -np.log((A + 1.0) / (B + 1.0))


def _mse(x, fs):
    """Per-channel multiscale sample entropy, median over evenly spaced epochs (tolerance relative to epoch SD)."""
    x = _filt(x, fs)
    L = int(SE_EP_SEC * fs)
    n_ep = x.shape[1] // L
    if n_ep < 1:
        L = x.shape[1]
        n_ep = 1
    idx = np.unique(np.linspace(0, n_ep - 1, min(n_ep, SE_MAX_EP)).round().astype(int))
    out = np.zeros((x.shape[0], len(SE_SCALES)))
    for ch in range(x.shape[0]):
        vals = []
        for e in idx:
            seg = x[ch, e * L:(e + 1) * L]
            sd = np.std(seg)
            if sd < 1e-9:
                continue
            seg = (seg - seg.mean()) / sd                # amplitude-scale free; r = 0.2 SD of the original epoch
            v = []
            for s in SE_SCALES:
                cg = seg[:(len(seg) // s) * s].reshape(-1, s).mean(axis=1)
                v.append(_sampen(cg, 0.2))
            vals.append(v)
        if vals:
            out[ch] = np.nanmedian(np.asarray(vals, dtype=np.float64), axis=0)
        else:
            out[ch] = 1.0
    return np.nan_to_num(out, nan=1.0).ravel()


def _keep_epochs(segs):
    # drop epochs whose largest channel std is extreme (blinks, movement)
    sd = np.log(segs.std(axis=-1) + 1e-9).max(axis=1)
    med = np.median(sd)
    mad = np.median(np.abs(sd - med)) + 1e-9
    keep = sd <= med + 3 * 1.4826 * mad
    return segs[keep] if keep.sum() >= 2 else segs


def _graph(w):
    """Weighted network topology on normalized graphs (independent of overall coupling strength):
    Onnela clustering coefficient of W/max and global efficiency of W/mean (distance = 1/weight)."""
    n = w.shape[0]
    W = np.clip(np.asarray(w, dtype=np.float64), 0, None)
    np.fill_diagonal(W, 0.0)
    off = ~np.eye(n, dtype=bool)
    if W.max() <= 1e-12 or n < 3:
        return np.zeros(2)
    cr = np.cbrt(W / W.max())
    tri = np.diag(cr @ cr @ cr)
    clust = np.mean(tri / ((n - 1) * (n - 2)))
    Wn = W / (W[off].mean() + 1e-12)
    D = np.where(off, 1.0 / (Wn + 1e-6), 0.0)
    sp = shortest_path(D, method="D", directed=False)
    eff = np.mean(1.0 / sp[off])
    return np.log(np.array([clust, eff]) + 1e-6)


def _wpli(x, fs, ep_sec=2):
    x = _filt(x, fs)
    n_ch = x.shape[0]
    L = int(ep_sec * fs)
    n_ep = x.shape[1] // L
    if n_ep < 2:
        return np.zeros(n_ch * len(CONN_BANDS)), np.zeros(2 * len(CONN_BANDS))
    segs = x[:, :n_ep * L].reshape(n_ch, n_ep, L).transpose(1, 0, 2)
    segs = segs - segs.mean(axis=-1, keepdims=True)
    # wPLI weights epochs by |Im S|, so artifact epochs are removed first
    segs = _keep_epochs(segs)
    X = np.fft.rfft(segs * np.hanning(L), axis=-1)
    f = np.fft.rfftfreq(L, 1.0 / fs)
    out, gr = [], []
    for a, b in CONN_BANDS:
        Xb = X[:, :, (f >= a) & (f < b)]
        im = (Xb[:, :, None, :] * np.conj(Xb[:, None, :, :])).imag
        w = np.abs(im.mean(axis=0)) / (np.abs(im).mean(axis=0) + 1e-12)
        w = w.mean(axis=-1)
        np.fill_diagonal(w, 0.0)
        out.append(w.sum(axis=1) / (n_ch - 1))
        gr.append(_graph(w))
    return np.concatenate(out), np.concatenate(gr)


def _covs(x, fs, ep_sec=2, shrink=0.05):
    """Band-limited spatial covariances: average reference, artifact epochs dropped, trace-normalized
    (amplitude-scale free) and shrunk toward identity (also regularizes the avg-ref rank deficiency)."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    n_ch = x.shape[0]
    L = int(ep_sec * fs)
    out = []
    for a, b in COV_BANDS:
        sos = butter(4, [a, b], btype="bandpass", fs=fs, output="sos")
        xb = sosfiltfilt(sos, x, axis=-1)
        n_ep = xb.shape[1] // L
        if n_ep < 2:
            C = np.cov(xb)
        else:
            segs = xb[:, :n_ep * L].reshape(n_ch, n_ep, L).transpose(1, 0, 2)
            segs = _keep_epochs(segs)
            C = np.einsum("eit,ejt->ij", segs, segs) / (segs.shape[0] * L)
        C = C / (np.trace(C) + 1e-12)
        C = (1 - shrink) * C + shrink * np.eye(n_ch) / n_ch
        out.append(C.ravel())
    return np.concatenate(out)


# ---------------------------------------------------------------- microstates
def _ms_prep(x, fs):
    # standard microstate band and average reference (topographies are reference-free after this)
    sos = butter(4, [2, 20], btype="bandpass", fs=fs, output="sos")
    x = sosfiltfilt(sos, np.asarray(x, dtype=np.float64), axis=-1)
    x = x - x.mean(axis=0, keepdims=True)
    return x, x.std(axis=0)


def _peak_maps(x, gfp, rng, n_max=1000):
    pk = np.where((gfp[1:-1] > gfp[:-2]) & (gfp[1:-1] >= gfp[2:]))[0] + 1
    if pk.size == 0:
        return np.zeros((0, x.shape[0]))
    g = np.log(gfp[pk] + 1e-12)
    med = np.median(g)
    mad = np.median(np.abs(g - med)) + 1e-9
    pk = pk[g <= med + 3 * 1.4826 * mad]           # artifact peaks (blinks, movement) excluded
    if pk.size > n_max:
        pk = np.sort(rng.choice(pk, n_max, replace=False))
    m = x[:, pk].T
    return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-12)


def _modkmeans(M, k=N_MS, n_init=5, n_iter=100, seed=0):
    rng = np.random.RandomState(seed)
    best = None
    for _ in range(n_init):
        T = M[rng.choice(len(M), k, replace=False)].copy()
        for _ in range(n_iter):
            lab = np.argmax(np.abs(M @ T.T), axis=1)
            T_new = T.copy()
            for j in range(k):
                Mj = M[lab == j]
                if len(Mj) < k:
                    continue
                _, v = np.linalg.eigh(Mj.T @ Mj)   # polarity-invariant centroid
                T_new[j] = v[:, -1]
            done = np.allclose(np.abs(np.sum(T_new * T, axis=1)), 1.0, atol=1e-6)
            T = T_new
            if done:
                break
        gev = np.sum(np.max(np.abs(M @ T.T), axis=1) ** 2)
        if best is None or gev > best[0]:
            best = (gev, T)
    return best[1]


def _canonical(ch_names):
    """Approximate canonical A/B/C/D microstate topographies from electrode positions."""
    import mne
    pos = mne.channels.make_standard_montage("standard_1020").get_positions()["ch_pos"]
    pos = {k.lower(): v for k, v in pos.items()}
    alias = {"t3": "t7", "t4": "t8", "t5": "p7", "t6": "p8", "t7": "t3", "t8": "t4", "p7": "t5", "p8": "t6"}
    P = []
    for ch in ch_names:
        c = ch.lower()
        p = pos.get(c, pos.get(alias.get(c, ""), None))
        P.append(np.zeros(3) if p is None else np.asarray(p, dtype=np.float64))
    P = np.asarray(P)
    P = P / (np.abs(P).max(axis=0, keepdims=True) + 1e-12)
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    C = np.stack([x + y, y - x, y, z])               # A: R-front/L-back, B: L-front/R-back, C: ant-post, D: fronto-central
    C = C - C.mean(axis=1, keepdims=True)
    return C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)


def _order_templates(T, ch_names):
    try:
        C = _canonical(ch_names)
        _, col = linear_sum_assignment(-np.abs(T @ C.T))
        order = np.argsort(col)                      # template assigned to canonical class 0..3
        return T[order]
    except Exception:
        return T


def _ms_feats(x, gfp, T, fs):
    xn = x / (np.linalg.norm(x, axis=0, keepdims=True) + 1e-12)
    corr = np.abs(T @ xn)
    n = x.shape[1]
    gev = np.sum((gfp * corr.max(axis=0)) ** 2) / (np.sum(gfp ** 2) + 1e-12)
    # temporal smoothing of the fit (~40 ms) suppresses single-sample label flicker
    c = uniform_filter1d(corr, size=max(1, int(round(0.04 * fs))), axis=1)
    lab = np.argmax(c, axis=0)
    change = np.flatnonzero(np.diff(lab)) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, n]
    seg_lab = lab[starts]
    seg_len = ends - starts
    dur_sec = n / fs
    cov, dur, occ = [], [], []
    for j in range(T.shape[0]):
        cov.append(np.mean(lab == j))
        sj = seg_len[seg_lab == j]
        dur.append(np.log(sj.mean() / fs + 1e-3) if sj.size else np.log(1e-3))
        occ.append(sj.size / dur_sec)
    return np.array(cov + dur + occ + [gev])


def _n_cov(ctx):
    return len(ctx["conditions"]) * len(COV_BANDS) * len(ctx["channels"]) ** 2


def featurize(items, ctx):
    fs = ctx["fs"]
    conds = ctx["conditions"]
    blocks = {c: ctx["load_eeg"](items, c) for c in conds}

    def get(c, i):
        x = blocks[c][i]
        return blocks[conds[0]][i] if x is None else x   # missing condition -> fall back to eyes closed

    # label-free group microstate templates from the first (eyes-closed) condition
    rng = np.random.RandomState(0)
    maps = []
    for i in range(len(items)):
        x = blocks[conds[0]][i]
        if x is None:
            continue
        xp, g = _ms_prep(x, fs)
        maps.append(_peak_maps(xp, g, rng))
    M = np.concatenate(maps, axis=0)
    T = _order_templates(_modkmeans(M), list(ctx["channels"]))

    feats = []
    for i in range(len(items)):
        f, fc = [], []
        for c in conds:
            x = get(c, i)
            f.append(_bandpower(x, fs).ravel())
            hfd, pe = _complexity(x, fs)
            f.append(hfd)
            f.append(pe)
            f.append(_mse(x, fs))
            ns, gr = _wpli(x, fs)
            f.append(ns)
            f.append(gr)
            xp, g = _ms_prep(x, fs)
            f.append(_ms_feats(xp, g, T, fs))
            fc.append(_covs(x, fs))
        feats.append(np.concatenate(f + fc))           # covariance block always last
    return np.nan_to_num(np.asarray(feats, dtype=np.float64))


def _tangent(C_tr, C_te):
    from pyriemann.tangentspace import TangentSpace
    ts = TangentSpace(metric="riemann").fit(C_tr)
    return ts.transform(C_tr), ts.transform(C_te)


def _lr(X_tr, y, X_te, C=0.1):
    sc = StandardScaler().fit(X_tr)
    clf = LogisticRegression(C=C, max_iter=5000).fit(sc.transform(X_tr), y)
    return clf.predict_proba(sc.transform(X_te))[:, 1]


def fit_predict(F_train, y_train, F_test, ctx):
    F_train = np.asarray(F_train, dtype=np.float64)
    F_test = np.asarray(F_test, dtype=np.float64)
    try:
        n_cov = _n_cov(ctx)
        n_ch = len(ctx["channels"])
    except Exception:
        return _lr(F_train, y_train, F_test)
    p1 = _lr(F_train[:, :-n_cov], y_train, F_test[:, :-n_cov])
    try:
        n_blk = n_cov // (n_ch * n_ch)
        eye = 1e-9 * np.eye(n_ch)
        Ctr = F_train[:, -n_cov:].reshape(len(F_train), n_blk, n_ch, n_ch)
        Cte = F_test[:, -n_cov:].reshape(len(F_test), n_blk, n_ch, n_ch)
        Ztr, Zte = [], []
        for b in range(n_blk):
            A = 0.5 * (Ctr[:, b] + Ctr[:, b].transpose(0, 2, 1)) + eye
            B = 0.5 * (Cte[:, b] + Cte[:, b].transpose(0, 2, 1)) + eye
            ztr, zte = _tangent(A, B)
            Ztr.append(ztr)
            Zte.append(zte)
        p2 = _lr(np.concatenate(Ztr, axis=1), y_train, np.concatenate(Zte, axis=1))
    except Exception:
        return p1
    return 0.5 * (p1 + p2)