"""Log relative band power (delta/theta/alpha/beta/gamma) per channel from Welch PSD of each
available condition + filter-bank Riemannian tangent-space features: trace-normalized,
average-referenced OAS spatial covariances for broadband 1-45 Hz and for delta/theta/alpha/beta/
low-gamma sub-bands, estimated only on artifact-free 2-s epochs (epochs whose broadband
log-variance is a robust outlier, |z| > 3 via median/MAD, on any channel are rejected; reference
point fitted on the training fold only) + per-channel spectral shape:
aperiodic 1/f exponent (log-log fit 2-40 Hz excluding alpha) and individual alpha peak frequency
(centre of gravity of the periodic residual, 7-13 Hz) + weighted phase-lag index (wPLI) for all
channel pairs in delta/theta/alpha/beta (Hilbert analytic signal, average reference) + EEG
microstates: 4 polarity-invariant template maps fitted label-free (modified k-means on GFP peaks
pooled over all items, 2-20 Hz, average reference), backfitted per subject to give coverage, mean
duration, occurrence rate, transition probabilities and GEV. Sub-band blocks are down-weighted
so that together they carry about one block's prior weight; standardized, L2 logistic regression."""
import numpy as np
from scipy.signal import welch, butter, sosfiltfilt, hilbert, find_peaks
from sklearn.covariance import OAS
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pyriemann.tangentspace import TangentSpace

BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
COV_BANDS = [(1, 45), (1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]   # first = broadband
WPLI_BANDS = [(1, 4), (4, 8), (8, 13), (13, 30)]
N_SHAPE = 2                                                 # aperiodic exponent, alpha peak freq
MS_K = 4
N_MS = 3 * MS_K + MS_K * (MS_K - 1) + 1                     # cov/dur/occ, transitions, GEV
EPOCH_S = 2.0
REJ_Z = 3.0


def _bandpower(x, fs):
    f, p = welch(x, fs=fs, nperseg=int(2 * fs), axis=-1)
    tot = p[:, (f >= 1) & (f <= 45)].sum(axis=1, keepdims=True) + 1e-12
    return np.log(np.stack([p[:, (f >= a) & (f < b)].sum(axis=1) for a, b in BANDS], axis=1) / tot)


def _spectral_shape(x, fs):
    f, p = welch(x.astype(np.float64), fs=fs, nperseg=int(4 * fs), axis=-1)
    lp = np.log10(p + 1e-20)
    fit_m = (f >= 2) & (f <= 40) & ~((f >= 7) & (f <= 14))
    A = np.stack([np.ones(fit_m.sum()), np.log10(f[fit_m])], axis=1)
    coef, *_ = np.linalg.lstsq(A, lp[:, fit_m].T, rcond=None)   # (2, n_ch)
    expo = -coef[1]
    am = (f >= 7) & (f <= 13)
    resid = lp[:, am] - (coef[0][:, None] + coef[1][:, None] * np.log10(f[am])[None, :])
    w = np.clip(resid, 0, None)
    ws = w.sum(axis=1)
    paf = np.where(ws > 0, (w * f[am][None, :]).sum(axis=1) / np.maximum(ws, 1e-12), 10.0)
    return np.stack([expo, paf], axis=1)


def _clean_epoch_mask(xb, fs):
    """Boolean sample mask keeping 2-s epochs without robust broadband-variance outliers."""
    n_s = xb.shape[1]
    L = int(EPOCH_S * fs)
    n_ep = n_s // L
    mask = np.ones(n_s, dtype=bool)
    if n_ep < 4:
        return mask
    ep = xb[:, :n_ep * L].reshape(xb.shape[0], n_ep, L)
    lv = np.log(ep.var(axis=-1) + 1e-12)                        # (n_ch, n_ep)
    med = np.median(lv, axis=1, keepdims=True)
    mad = 1.4826 * np.median(np.abs(lv - med), axis=1, keepdims=True) + 1e-6
    score = np.max(np.abs(lv - med) / mad, axis=0)              # worst channel per epoch
    keep = score <= REJ_Z
    if keep.sum() < n_ep // 2:                                  # never drop more than half
        keep = np.zeros(n_ep, dtype=bool)
        keep[np.argsort(score, kind="stable")[:max(n_ep // 2, 1)]] = True
    mask[:n_ep * L] = np.repeat(keep, L)
    return mask


def _spatial_covs(x, fs):
    x = x.astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)                       # average re-reference
    mask = None
    out = []
    for lo, hi in COV_BANDS:
        sos = butter(4, [lo, min(hi, 0.45 * fs)], btype="band", fs=fs, output="sos")
        xf = sosfiltfilt(sos, x, axis=-1)
        if mask is None:                                        # broadband defines clean epochs
            mask = _clean_epoch_mask(xf, fs)
        C = OAS(assume_centered=False).fit(xf[:, mask].T).covariance_
        out.append((C / np.trace(C)).ravel())                   # amplitude-scale invariant
    return np.concatenate(out)


def _wpli(x, fs):
    x = x.astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)                       # average re-reference
    iu = np.triu_indices(x.shape[0], 1)
    trim = int(fs)                                              # drop filter/Hilbert edge effects
    out = []
    for lo, hi in WPLI_BANDS:
        sos = butter(4, [lo, min(hi, 0.45 * fs)], btype="band", fs=fs, output="sos")
        z = hilbert(sosfiltfilt(sos, x, axis=-1), axis=-1)
        if z.shape[1] > 4 * trim:
            z = z[:, trim:-trim]
        im = np.imag(z[iu[0]] * np.conj(z[iu[1]]))
        out.append(np.abs(im.sum(axis=1)) / (np.abs(im).sum(axis=1) + 1e-12))
    return np.concatenate(out)


def _ms_filter(x, fs):
    x = x.astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)                       # average re-reference
    sos = butter(4, [2, 20], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=-1)


def _gfp_peak_maps(xf, rng, n_max=1000):
    gfp = xf.std(axis=0)
    pk, _ = find_peaks(gfp)
    if len(pk) > n_max:
        pk = np.sort(rng.choice(pk, n_max, replace=False))
    return xf[:, pk].T


def _modkmeans(X, k, n_init=10, n_iter=100, seed=0):
    """Polarity-invariant (modified) k-means on unit-norm maps."""
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    rng = np.random.RandomState(seed)
    best = None
    for _ in range(n_init):
        T = X[rng.choice(len(X), k, replace=False)].copy()
        for _ in range(n_iter):
            lab = np.argmax(np.abs(X @ T.T), axis=1)
            Tn = T.copy()
            for j in range(k):
                S = X[lab == j]
                if len(S) < 2:
                    continue
                _, v = np.linalg.eigh(S.T @ S)
                Tn[j] = v[:, -1]
            done = np.allclose(np.abs((Tn * T).sum(axis=1)), 1.0, atol=1e-8)
            T = Tn
            if done:
                break
        ev = np.mean(np.max(np.abs(X @ T.T), axis=1) ** 2)
        if best is None or ev > best[0]:
            best = (ev, T)
    T = best[1]
    lab = np.argmax(np.abs(X @ T.T), axis=1)
    return T[np.argsort(-np.bincount(lab, minlength=k), kind="stable")]


def _ms_features(xf, T, fs, min_len=3):
    k = T.shape[0]
    gfp = xf.std(axis=0)
    c = np.abs(T @ xf) / (np.linalg.norm(xf, axis=0) + 1e-12)  # (k, n_samples)
    lab = np.argmax(c, axis=0)
    gev = np.sum((gfp * c.max(axis=0)) ** 2) / (np.sum(gfp ** 2) + 1e-12)
    change = np.flatnonzero(np.diff(lab)) + 1
    starts = np.r_[0, change]
    seg_len = np.diff(np.r_[starts, len(lab)])
    seg_lab = lab[starts].copy()
    for i in range(1, len(seg_lab)):                             # absorb very short segments
        if seg_len[i] < min_len:
            seg_lab[i] = seg_lab[i - 1]
    lab = np.repeat(seg_lab, seg_len)
    change = np.flatnonzero(np.diff(lab)) + 1
    starts = np.r_[0, change]
    seg_len = np.diff(np.r_[starts, len(lab)])
    seg_lab = lab[starts]
    dur_total = len(lab) / fs
    cov = np.array([seg_len[seg_lab == j].sum() / len(lab) for j in range(k)])
    mdur = np.array([seg_len[seg_lab == j].mean() / fs if np.any(seg_lab == j) else 0.0
                     for j in range(k)])
    occ = np.array([np.sum(seg_lab == j) / dur_total for j in range(k)])
    tr = np.zeros((k, k))
    np.add.at(tr, (seg_lab[:-1], seg_lab[1:]), 1.0)
    rs = tr.sum(axis=1, keepdims=True)
    tr = np.where(rs > 0, tr / np.maximum(rs, 1e-12), 1.0 / (k - 1))
    off = tr[~np.eye(k, dtype=bool)]
    return np.concatenate([cov, mdur, occ, off, [gev]])


def featurize(items, ctx):
    fs = ctx["fs"]
    conds = ctx["conditions"]
    bp_feats, cov_feats, sh_feats, wp_feats = [], [], [], []
    blocks = {c: ctx["load_eeg"](items, c) for c in conds}
    rng = np.random.RandomState(0)
    peak_maps = []
    for i in range(len(items)):
        fb, fc, fsh, fw = [], [], [], []
        for c in conds:
            x = blocks[c][i]
            if x is None:                      # missing condition -> fall back to eyes closed
                x = blocks[conds[0]][i]
            fb.append(_bandpower(x, fs).ravel())
            fc.append(_spatial_covs(x, fs))
            fsh.append(_spectral_shape(x, fs).ravel())
            fw.append(_wpli(x, fs))
            if blocks[c][i] is not None:
                peak_maps.append(_gfp_peak_maps(_ms_filter(x, fs), rng))
        bp_feats.append(np.concatenate(fb))
        cov_feats.append(np.concatenate(fc))
        sh_feats.append(np.concatenate(fsh))
        wp_feats.append(np.concatenate(fw))
    T = _modkmeans(np.vstack(peak_maps), MS_K)   # label-free group templates
    ms_feats = []
    for i in range(len(items)):
        fm = []
        for c in conds:
            x = blocks[c][i]
            if x is None:
                x = blocks[conds[0]][i]
            fm.append(_ms_features(_ms_filter(x, fs), T, fs))
        ms_feats.append(np.concatenate(fm))
    F = np.hstack([np.asarray(bp_feats), np.asarray(cov_feats), np.asarray(sh_feats),
                   np.asarray(wp_feats), np.asarray(ms_feats)]).astype(np.float64)
    return np.nan_to_num(F)


def _split(F, ctx):
    n_ch = len(ctx["channels"])
    n_cond = len(ctx["conditions"])
    n_pair = n_ch * (n_ch - 1) // 2
    nb = n_cond * n_ch * len(BANDS)
    nc = n_cond * len(COV_BANDS) * n_ch * n_ch
    ns = n_cond * n_ch * N_SHAPE
    nw = n_cond * len(WPLI_BANDS) * n_pair
    bp = F[:, :nb]
    covs = F[:, nb:nb + nc].reshape(len(F), n_cond, len(COV_BANDS), n_ch, n_ch)
    shape = F[:, nb + nc:nb + nc + ns]
    o = nb + nc + ns
    wpli = F[:, o:o + nw].reshape(len(F), n_cond, len(WPLI_BANDS), n_pair)
    ms = F[:, o + nw:o + nw + n_cond * N_MS]
    return bp, covs, shape, wpli, ms


def fit_predict(F_train, y_train, F_test, ctx):
    F_train = np.asarray(F_train, dtype=np.float64)
    F_test = np.asarray(F_test, dtype=np.float64)
    bp_tr, cov_tr, sh_tr, wp_tr, ms_tr = _split(F_train, ctx)
    bp_te, cov_te, sh_te, wp_te, ms_te = _split(F_test, ctx)
    sc_bp = StandardScaler().fit(bp_tr)
    sc_sh = StandardScaler().fit(sh_tr)
    sc_ms = StandardScaler().fit(ms_tr)
    X_tr = [sc_bp.transform(bp_tr), sc_sh.transform(sh_tr), sc_ms.transform(ms_tr)]
    X_te = [sc_bp.transform(bp_te), sc_sh.transform(sh_te), sc_ms.transform(ms_te)]
    n_sub = len(COV_BANDS) - 1
    for k in range(cov_tr.shape[1]):
        for b in range(cov_tr.shape[2]):
            Ctr = (cov_tr[:, k, b] + cov_tr[:, k, b].transpose(0, 2, 1)) / 2
            Cte = (cov_te[:, k, b] + cov_te[:, k, b].transpose(0, 2, 1)) / 2
            ts = TangentSpace(metric="riemann").fit(Ctr)
            t_tr, t_te = ts.transform(Ctr), ts.transform(Cte)
            sc = StandardScaler().fit(t_tr)
            w = 1.0 if b == 0 else 1.0 / np.sqrt(n_sub)
            X_tr.append(w * sc.transform(t_tr))
            X_te.append(w * sc.transform(t_te))
    w_wp = 1.0 / np.sqrt(len(WPLI_BANDS))
    for k in range(wp_tr.shape[1]):
        for b in range(wp_tr.shape[2]):
            sc = StandardScaler().fit(wp_tr[:, k, b])
            X_tr.append(w_wp * sc.transform(wp_tr[:, k, b]))
            X_te.append(w_wp * sc.transform(wp_te[:, k, b]))
    X_tr = np.hstack(X_tr)
    X_te = np.hstack(X_te)
    clf = LogisticRegression(C=0.1, max_iter=5000).fit(X_tr, y_train)
    return clf.predict_proba(X_te)[:, 1]