"""Log relative band power (delta/theta/alpha/beta/gamma) per channel from Welch PSD of each
available condition + per-channel aperiodic (1/f) exponent (log-log linear fit, alpha range
excluded) + per-channel individual alpha frequency (centre of gravity of the 7-13 Hz PSD)
+ per-channel Higuchi fractal dimension of the 1-45 Hz signal + per-channel normalized spectral
entropy (1-45 Hz PSD) and permutation entropy (order 3, ~12 ms lag, 1-45 Hz signal)
+ per-channel Lempel-Ziv complexity (median-binarized 1-45 Hz signal, averaged over 10 s windows)
+ per-channel wPLI node strength
(theta, alpha, beta) + per-channel temporal variability (IQR across 2 s epochs) of log relative
theta and alpha power (vigilance stability) + per-channel DFA exponent of theta and alpha
amplitude envelopes (long-range temporal correlations) + per-channel node strength of
orthogonalized amplitude-envelope correlation (theta, alpha, beta; leakage-corrected power
coupling) + EEG microstate dynamics (4 group
templates from label-free polarity-invariant k-means on 2-20 Hz average-referenced GFP-peak
maps, canonically ordered against A/B/C/D prototypes; coverage, duration, occurrence, GEV and
transition probabilities per condition) + trace-normalized filter-bank spatial covariances
(broadband 1-45 Hz, theta, alpha, beta) per condition, each projected to its own Riemannian
tangent space (reference = training-fold mean), standardized, L2 logistic regression."""
import numpy as np
from scipy.signal import welch, butter, sosfiltfilt, hilbert, find_peaks
from scipy.optimize import linear_sum_assignment
from sklearn.covariance import OAS
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pyriemann.tangentspace import TangentSpace
from antropy import higuchi_fd, perm_entropy, lziv_complexity

BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
COV_BANDS = [(1, 45), (4, 8), (8, 13), (13, 30)]
WPLI_BANDS = [(4, 8), (8, 13), (13, 30)]
VIG_BANDS = [(4, 8), (8, 13)]
DFA_BANDS = [(4, 8), (8, 13)]
AEC_BANDS = [(4, 8), (8, 13), (13, 30)]
MS_K = 4
MS_NFEAT = 4 * MS_K + MS_K * (MS_K - 1)
N_PER_CH = 6                                   # slope, paf, hfd, spec ent, perm ent, LZC
LZ_WIN_S = 10.0


def _spectral(x, fs):
    f, p = welch(x, fs=fs, nperseg=int(2 * fs), axis=-1)
    tot = p[:, (f >= 1) & (f <= 45)].sum(axis=1, keepdims=True) + 1e-12
    bp = np.log(np.stack([p[:, (f >= a) & (f < b)].sum(axis=1) for a, b in BANDS], axis=1) / tot)
    # aperiodic exponent: log-log linear fit outside the alpha peak and below line noise
    m = ((f >= 2) & (f < 7)) | ((f > 14) & (f <= 40))
    lf = np.log10(f[m])
    lp = np.log10(p[:, m] + 1e-12)
    A = np.vstack([lf, np.ones_like(lf)]).T
    coef, *_ = np.linalg.lstsq(A, lp.T, rcond=None)
    slope = -coef[0]                           # one exponent per channel
    # individual alpha frequency: power-weighted centre of gravity in 7-13 Hz
    am = (f >= 7) & (f <= 13)
    pa = p[:, am]
    paf = (pa * f[am]).sum(axis=1) / (pa.sum(axis=1) + 1e-12)
    # normalized spectral (Shannon) entropy of the 1-45 Hz PSD
    pm = p[:, (f >= 1) & (f <= 45)]
    q = pm / (pm.sum(axis=1, keepdims=True) + 1e-12)
    sen = -(q * np.log(q + 1e-12)).sum(axis=1) / np.log(q.shape[1])
    return bp.ravel(), slope, paf, sen


def _hfd_pe(x, fs):
    # Higuchi fractal dimension, permutation entropy and Lempel-Ziv complexity of the
    # 1-45 Hz signal (drift and line noise removed); PE lag fixed in time (~12 ms) across
    # sampling rates; LZC on median-binarized fixed-length (10 s) windows, averaged, so it
    # does not depend on recording length
    sos = butter(4, [1, 45], btype="bandpass", fs=fs, output="sos")
    xf = sosfiltfilt(sos, x, axis=-1)
    lag = max(1, int(round(0.012 * fs)))
    hfd = np.array([higuchi_fd(np.ascontiguousarray(ch), kmax=10) for ch in xf])
    pe = np.array([perm_entropy(np.ascontiguousarray(ch), order=3, delay=lag, normalize=True)
                   for ch in xf])
    seg = int(LZ_WIN_S * fs)
    lz = []
    for ch in xf:
        ns = len(ch) // seg
        if ns < 1:
            wins = [ch]
        else:
            wins = [ch[s * seg:(s + 1) * seg] for s in range(ns)]
        vals = [lziv_complexity(np.ascontiguousarray(w > np.median(w)), normalize=True)
                for w in wins]
        lz.append(float(np.mean(vals)))
    return hfd, pe, np.asarray(lz)


def _epochs_fft(x, fs):
    n_ch = x.shape[0]
    n = int(2 * fs)
    ne = x.shape[1] // n
    ep = x[:, :ne * n].reshape(n_ch, ne, n).transpose(1, 0, 2)
    ep = ep - ep.mean(axis=-1, keepdims=True)
    X = np.fft.rfft(ep * np.hanning(n), axis=-1)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    return X, f


def _wpli(X, f, n_ch):
    # weighted phase-lag index per band from 2 s Hann-windowed epochs; summarized as
    # per-channel node strength (mean wPLI to all other channels)
    out = []
    for lo, hi in WPLI_BANDS:
        Xb = X[:, :, (f >= lo) & (f < hi)]                       # epochs x ch x freqs
        im = np.imag(Xb[:, :, None, :] * np.conj(Xb[:, None, :, :]))
        num = np.abs(im.sum(axis=0))
        den = np.abs(im).sum(axis=0) + 1e-12
        w = (num / den).mean(axis=-1)
        np.fill_diagonal(w, 0.0)
        out.append(w.sum(axis=1) / (n_ch - 1))
    return np.concatenate(out)


def _vigilance(X, f):
    # temporal stability of theta/alpha: IQR across 2 s epochs of log relative band power
    P = np.abs(X) ** 2                                           # epochs x ch x freqs
    tot = P[:, :, (f >= 1) & (f <= 45)].sum(axis=-1) + 1e-12
    out = []
    for lo, hi in VIG_BANDS:
        rb = np.log(P[:, :, (f >= lo) & (f < hi)].sum(axis=-1) / tot + 1e-12)
        q75, q25 = np.percentile(rb, [75, 25], axis=0)
        out.append(q75 - q25)
    return np.concatenate(out)


def _dfa(x, fs):
    # long-range temporal correlations: DFA exponent of the band amplitude envelope,
    # window sizes 1 s .. min(15 s, length/4), log-spaced (fixed in seconds across datasets)
    N = x.shape[1]
    max_s = max(2.0, min(15.0, N / fs / 4.0))
    sizes = np.unique(np.round(np.logspace(np.log10(fs * 1.0), np.log10(fs * max_s), 10))
                      .astype(int))
    ls = np.log10(sizes)
    out = []
    for lo, hi in DFA_BANDS:
        sos = butter(4, [lo, hi], btype="bandpass", fs=fs, output="sos")
        env = np.abs(hilbert(sosfiltfilt(sos, x, axis=-1), axis=-1))
        y = np.cumsum(env - env.mean(axis=-1, keepdims=True), axis=-1)
        flu = []
        for n in sizes:
            nw = N // n
            seg = y[:, :nw * n].reshape(y.shape[0], nw, n)
            t = np.arange(n, dtype=np.float64)
            t = t - t.mean()
            segc = seg - seg.mean(axis=-1, keepdims=True)
            b = (segc * t).sum(axis=-1, keepdims=True) / (t @ t)
            res = segc - b * t
            flu.append(np.sqrt((res ** 2).mean(axis=(1, 2)) + 1e-24))
        F = np.log10(np.array(flu))                              # sizes x ch
        out.append(np.polyfit(ls, F, 1)[0])
    return np.concatenate(out)


def _aec(x, fs):
    # orthogonalized amplitude-envelope correlation (Hipp et al. 2012): for each seed channel,
    # remove the zero-lag (volume-conducted) component of every other channel's analytic
    # signal, correlate envelopes; symmetrize and summarize as per-channel node strength
    n_ch = x.shape[0]
    edge = int(fs)                                               # drop 1 s filter edges
    out = []
    for lo, hi in AEC_BANDS:
        sos = butter(4, [lo, hi], btype="bandpass", fs=fs, output="sos")
        Z = hilbert(sosfiltfilt(sos, x, axis=-1), axis=-1)
        if Z.shape[1] > 4 * edge:
            Z = Z[:, edge:-edge]
        Z = Z[:, ::2]                                            # envelopes are low-pass
        A = np.abs(Z) + 1e-12
        W = np.zeros((n_ch, n_ch))
        for i in range(n_ch):
            yo = np.abs(np.imag(Z * (np.conj(Z[i]) / A[i])[None, :]))  # ch x time
            a = A[i] - A[i].mean()
            b = yo - yo.mean(axis=1, keepdims=True)
            W[i] = (b @ a) / (np.sqrt((b ** 2).sum(axis=1) * (a @ a)) + 1e-12)
        W = 0.5 * (W + W.T)
        np.fill_diagonal(W, 0.0)
        out.append(W.sum(axis=1) / (n_ch - 1))
    return np.concatenate(out)


def _cov(x, fs, lo, hi):
    sos = butter(4, [lo, hi], btype="bandpass", fs=fs, output="sos")
    xf = sosfiltfilt(sos, x.astype(np.float64), axis=-1)
    C = OAS(assume_centered=False).fit(xf.T).covariance_
    C = C / np.trace(C)
    C = C + 1e-6 * np.eye(C.shape[0])
    return C


def _ms_prep(x, fs):
    # microstate input: 2-20 Hz, average reference, maps at local maxima of global field power
    sos = butter(4, [2, 20], btype="bandpass", fs=fs, output="sos")
    xf = sosfiltfilt(sos, x, axis=-1)
    xf = xf - xf.mean(axis=0, keepdims=True)
    gfp = xf.std(axis=0)
    pk, _ = find_peaks(gfp)
    return pk, xf[:, pk].T.astype(np.float32), gfp[pk], x.shape[1]


def _ms_kmeans(M, K, n_init=5, n_iter=100, seed=0):
    # polarity-invariant modified k-means on unit-norm maps
    rng = np.random.RandomState(seed)
    best_T, best_s = None, -np.inf
    for _ in range(n_init):
        T = M[rng.choice(len(M), K, replace=False)].copy()
        prev = None
        for _ in range(n_iter):
            lab = np.argmax(np.abs(M @ T.T), axis=1)
            if prev is not None and np.array_equal(lab, prev):
                break
            prev = lab
            for k in range(K):
                Mk = M[lab == k]
                if len(Mk) == 0:
                    continue
                _, v = np.linalg.eigh(Mk.T @ Mk)
                T[k] = v[:, -1]
        s = np.mean(np.max((M @ T.T) ** 2, axis=1))
        if s > best_s:
            best_s, best_T = s, T.copy()
    return best_T


def _ms_order(T, channels):
    # order templates against canonical A/B/C/D prototype topographies
    try:
        from mne.channels import make_standard_montage
        pos = make_standard_montage("standard_1020").get_positions()["ch_pos"]
        low = {k.lower(): v for k, v in pos.items()}
        P = np.array([low.get(c.lower(), np.zeros(3))[:2] for c in channels], dtype=np.float64)
        P = P - P.mean(axis=0)
        P = P / (np.abs(P).max(axis=0) + 1e-12)
        xx, yy = P[:, 0], P[:, 1]
        protos = np.stack([xx + yy, yy - xx, yy, -(xx ** 2 + yy ** 2)])
        protos = protos - protos.mean(axis=1, keepdims=True)
        protos = protos / (np.linalg.norm(protos, axis=1, keepdims=True) + 1e-12)
        sim = np.abs(T @ protos.T)
        r, c = linear_sum_assignment(-sim)
        order = r[np.argsort(c)]
        return T[order]
    except Exception:
        return T


def _ms_feats(pk, maps, g, N, T, fs):
    K = T.shape[0]
    if len(pk) < 3:
        return np.zeros(MS_NFEAT)
    maps = maps.astype(np.float64)
    corr = (maps @ T.T) / (np.linalg.norm(maps, axis=1, keepdims=True) + 1e-12)
    lab = np.argmax(np.abs(corr), axis=1)
    mids = (pk[:-1] + pk[1:]) / 2.0
    sl = lab[np.searchsorted(mids, np.arange(N))]         # each sample -> nearest GFP peak
    ch = np.flatnonzero(np.diff(sl))
    starts = np.r_[0, ch + 1]
    ends = np.r_[ch + 1, N]
    rl = sl[starts]
    dur = (ends - starts) / fs
    total = N / fs
    g2 = g ** 2
    cmax2 = np.max(corr ** 2, axis=1)
    out = []
    for k in range(K):
        out.append((sl == k).mean())
        out.append((rl == k).sum() / total)
        out.append(dur[rl == k].mean() if np.any(rl == k) else 0.0)
        out.append((g2 * cmax2)[lab == k].sum() / (g2.sum() + 1e-12))
    tr = np.zeros((K, K))
    if len(rl) > 1:
        np.add.at(tr, (rl[:-1], rl[1:]), 1.0)
        tr /= (len(rl) - 1)
    out.extend(tr[~np.eye(K, dtype=bool)])
    return np.asarray(out, dtype=np.float64)


def featurize(items, ctx):
    fs = ctx["fs"]
    conds = ctx["conditions"]
    blocks = {c: ctx["load_eeg"](items, c) for c in conds}
    rng = np.random.RandomState(0)
    base, covs, ms_in, pool = [], [], [], []
    for i in range(len(items)):
        bp, sl, pf, fd, se, pe, lz, wp, vg, df, ae, cv, mi = ([] for _ in range(13))
        for c in conds:
            x = blocks[c][i]
            if x is None:                      # missing condition -> fall back to eyes closed
                x = blocks[conds[0]][i]
            x64 = x.astype(np.float64)
            b, s, a, e = _spectral(x64, fs)
            bp.append(b)
            sl.append(s)
            pf.append(a)
            se.append(e)
            h, p, z = _hfd_pe(x64, fs)
            fd.append(h)
            pe.append(p)
            lz.append(z)
            X, f = _epochs_fft(x64, fs)
            wp.append(_wpli(X, f, x64.shape[0]))
            vg.append(_vigilance(X, f))
            df.append(_dfa(x64, fs))
            ae.append(_aec(x64, fs))
            for lo, hi in COV_BANDS:
                cv.append(_cov(x, fs, lo, hi).ravel())
            pk, maps, g, N = _ms_prep(x64, fs)
            mi.append((pk, maps, g, N))
            # pool a bounded, artifact-trimmed sample of GFP-peak maps for template fitting
            keep = np.flatnonzero(g < 3.0 * np.median(g)) if len(g) else np.array([], int)
            if len(keep) > 300:
                keep = np.sort(rng.choice(keep, 300, replace=False))
            if len(keep):
                pool.append(maps[keep].astype(np.float64))
        base.append(np.concatenate(bp + sl + pf + fd + se + pe + lz + wp + vg + df + ae))
        covs.append(np.concatenate(cv))
        ms_in.append(mi)
    M = np.vstack(pool)
    M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    T = _ms_order(_ms_kmeans(M, MS_K), ctx["channels"])
    feats = []
    for i in range(len(items)):
        ms = [_ms_feats(pk, maps, g, N, T, fs) for pk, maps, g, N in ms_in[i]]
        feats.append(np.concatenate([base[i]] + ms + [covs[i]]))
    return np.asarray(feats, dtype=np.float64)


def _split(F, ctx):
    n_ch = len(ctx["channels"])
    n_cond = len(ctx["conditions"])
    n_bp = (n_ch * len(BANDS) * n_cond
            + (N_PER_CH + len(WPLI_BANDS) + len(VIG_BANDS) + len(DFA_BANDS) + len(AEC_BANDS))
            * n_ch * n_cond
            + MS_NFEAT * n_cond)
    bp = F[:, :n_bp]
    covs = F[:, n_bp:].reshape(len(F), n_cond * len(COV_BANDS), n_ch, n_ch)
    return bp, covs


def fit_predict(F_train, y_train, F_test, ctx):
    F_train = np.asarray(F_train, dtype=np.float64)
    F_test = np.asarray(F_test, dtype=np.float64)
    bp_tr, cov_tr = _split(F_train, ctx)
    bp_te, cov_te = _split(F_test, ctx)
    ts_tr, ts_te = [bp_tr], [bp_te]
    for k in range(cov_tr.shape[1]):
        ts = TangentSpace(metric="riemann").fit(cov_tr[:, k])
        ts_tr.append(ts.transform(cov_tr[:, k]))
        ts_te.append(ts.transform(cov_te[:, k]))
    X_tr = np.hstack(ts_tr)
    X_te = np.hstack(ts_te)
    sc = StandardScaler().fit(X_tr)
    clf = LogisticRegression(C=0.1, max_iter=5000).fit(sc.transform(X_tr), y_train)
    return clf.predict_proba(sc.transform(X_te))[:, 1]