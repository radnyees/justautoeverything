"""Log relative band power (delta/theta/alpha/beta/gamma) per channel from Welch PSD of each
available condition + per-channel wPLI node strength (delta/theta/alpha/beta, eyes closed,
average reference) + EEG microstate statistics (4 group templates fitted label-free on GFP-peak
maps, canonically ordered A-D; coverage, duration, occurrence, GEV) + per-channel spectral
parameterization (robust aperiodic 1/f exponent and alpha peak centre-of-gravity frequency above
the aperiodic fit, eyes closed, average reference) + per-channel DFA exponent of the alpha
amplitude envelope (long-range temporal correlations, 1-10 s windows, eyes closed, average
reference) + Riemannian tangent-space vector of the 4-13 Hz spatial covariance (eyes closed,
average reference, high-variance epochs dropped; per-subject log-Euclidean mean of
trace-normalized, shrunk per-epoch covariances; label-free log-Euclidean group mean as
reference), standardized, L2 logistic regression."""
import numpy as np
from scipy.signal import welch, detrend, butter, sosfiltfilt, find_peaks, hilbert
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
CONN_BANDS = [(1, 4), (4, 8), (8, 13), (13, 30)]
N_MS = 4
MS_PEAKS_PER_SUBJ = 500
COV_BAND = (4, 13)
COV_SHRINK = 0.05
DFA_BAND = (8, 13)
DFA_WIN_S = (1.0, 10.0)


def _bandpower(x, fs):
    f, p = welch(x, fs=fs, nperseg=int(2 * fs), axis=-1)
    tot = p[:, (f >= 1) & (f <= 45)].sum(axis=1, keepdims=True) + 1e-12
    return np.log(np.stack([p[:, (f >= a) & (f < b)].sum(axis=1) for a, b in BANDS], axis=1) / tot)


def _wpli_strength(x, fs):
    x = x.astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)              # common average reference
    n = int(2 * fs)
    nep = x.shape[1] // n
    C = x.shape[0]
    ep = x[:, :nep * n].reshape(C, nep, n).transpose(1, 0, 2)
    ep = detrend(ep, axis=-1) * np.hanning(n)
    X = np.fft.rfft(ep, axis=-1)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    out = []
    for a, b in CONN_BANDS:
        Xb = X[:, :, (f >= a) & (f < b)]
        im = np.imag(Xb[:, :, None, :] * np.conj(Xb[:, None, :, :]))   # nep x C x C x nf
        num = np.abs(im.mean(axis=0))
        den = np.abs(im).mean(axis=0) + 1e-12
        w = (num / den).mean(axis=-1)
        np.fill_diagonal(w, 0.0)
        out.append(w.sum(axis=1) / (C - 1))
    return np.stack(out, axis=1)


# ---------------------------------------------------------------- spectral parameterization
def _spec_param(x, fs):
    """Per channel: aperiodic exponent (robust log-log fit 2-40 Hz, alpha excluded) and
    alpha peak frequency (centre of gravity of positive residual above the fit, 7-13 Hz)."""
    x = x.astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)              # common average reference
    f, p = welch(x, fs=fs, nperseg=int(4 * fs), axis=-1)
    m = (f >= 2) & (f <= 40)
    fm = f[m]
    lf = np.log10(fm)
    lp = np.log10(p[:, m] + 1e-20)
    A = np.stack([lf, np.ones_like(lf)], axis=1)
    keep0 = ~((fm >= 7) & (fm <= 14))
    alpha = (fm >= 7) & (fm <= 13)
    out = []
    for c in range(lp.shape[0]):
        coef = np.linalg.lstsq(A[keep0], lp[c, keep0], rcond=None)[0]
        r = lp[c] - A @ coef
        keep = keep0 & (r <= np.median(r[keep0]))       # drop residual peaks, refit floor
        if keep.sum() >= 5:
            coef = np.linalg.lstsq(A[keep], lp[c, keep], rcond=None)[0]
        r = lp[c] - A @ coef
        ra = np.clip(r[alpha], 0, None)
        iaf = (fm[alpha] * ra).sum() / ra.sum() if ra.sum() > 0 else 10.0
        out.append([-coef[0], iaf])
    return np.asarray(out)


# ---------------------------------------------------------------- alpha envelope LRTC (DFA)
def _alpha_dfa(x, fs):
    """Per channel DFA exponent of the alpha amplitude envelope over 1-10 s windows."""
    x = x.astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)              # common average reference
    C = x.shape[0]
    sos = butter(4, list(DFA_BAND), btype="band", fs=fs, output="sos")
    x = sosfiltfilt(sos, x, axis=-1)
    env = np.abs(hilbert(x, axis=-1))
    trim = int(fs)
    if env.shape[1] > 4 * trim:
        env = env[:, trim:-trim]                        # drop filter edge effects
    y = np.cumsum(env - env.mean(axis=1, keepdims=True), axis=1)
    N = y.shape[1]
    sizes = np.unique(np.round(np.logspace(np.log10(DFA_WIN_S[0] * fs),
                                           np.log10(DFA_WIN_S[1] * fs), 10)).astype(int))
    sizes = sizes[sizes <= N // 4]
    if len(sizes) < 3:
        return np.full(C, 0.5)
    F = []
    for n in sizes:
        k = N // n
        seg = y[:, :k * n].reshape(C, k, n)
        r = detrend(seg, axis=-1)
        F.append(np.sqrt((r ** 2).mean(axis=(1, 2)) + 1e-30))
    lF = np.log(np.asarray(F))                          # S x C
    slope = np.polyfit(np.log(sizes.astype(np.float64)), lF, 1)[0]
    return slope


# ---------------------------------------------------------------- riemannian covariance
def _band_cov(x, fs):
    """Log-Euclidean mean of trace-normalized, shrunk per-epoch spatial covariances in COV_BAND;
    top 10% variance epochs dropped."""
    x = x.astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)              # common average reference
    sos = butter(4, list(COV_BAND), btype="band", fs=fs, output="sos")
    x = sosfiltfilt(sos, x, axis=-1)
    n = int(2 * fs)
    nep = x.shape[1] // n
    C = x.shape[0]
    ep = x[:, :nep * n].reshape(C, nep, n).transpose(1, 0, 2)
    v = ep.var(axis=(1, 2))
    ep = ep[v <= np.percentile(v, 90)] if nep >= 10 else ep
    Se = np.einsum("ecn,edn->ecd", ep, ep) / n
    tr = np.trace(Se, axis1=1, axis2=2) / C
    Se = Se / (tr[:, None, None] + 1e-20)
    Se = (1 - COV_SHRINK) * Se + COV_SHRINK * np.eye(C)[None]
    w, V = np.linalg.eigh(Se)
    L = np.einsum("eij,ej,ekj->eik", V, np.log(w), V).mean(axis=0)
    S = _fun_spd(L, np.exp)
    S = S / (np.trace(S) / C + 1e-20)
    return S


def _fun_spd(S, fun):
    w, V = np.linalg.eigh(S)
    return (V * fun(w)) @ V.T


def _tangent(covs):
    covs = np.asarray(covs)
    M = _fun_spd(np.mean([_fun_spd(S, np.log) for S in covs], axis=0), np.exp)
    Mi = _fun_spd(M, lambda w: 1.0 / np.sqrt(w))
    C = covs.shape[1]
    iu = np.triu_indices(C)
    coef = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    out = []
    for S in covs:
        L = _fun_spd(Mi @ S @ Mi, np.log)
        out.append(L[iu] * coef)
    return np.asarray(out)


# ---------------------------------------------------------------- microstates
def _ms_prep(x, fs):
    x = x.astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)              # average reference
    sos = butter(4, [2, 20], btype="band", fs=fs, output="sos")
    x = sosfiltfilt(sos, x, axis=-1)
    gfp = x.std(axis=0)
    pk, _ = find_peaks(gfp)
    return x, gfp, pk


def _unit(M):
    M = M - M.mean(axis=1, keepdims=True)
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)


def _mod_kmeans(X, k, n_init=10, iters=200, seed=0):
    rng = np.random.RandomState(seed)
    best, best_g = None, -1.0
    for _ in range(n_init):
        T = X[rng.choice(len(X), k, replace=False)].copy()
        for _ in range(iters):
            lab = np.argmax(np.abs(X @ T.T), axis=1)
            Tn = T.copy()
            for j in range(k):
                Xj = X[lab == j]
                if len(Xj) > 0:
                    _, v = np.linalg.eigh(Xj.T @ Xj)
                    Tn[j] = v[:, -1]
            done = np.allclose(np.abs(np.sum(Tn * T, axis=1)), 1.0, atol=1e-7)
            T = Tn
            if done:
                break
        g = np.mean(np.max(np.abs(X @ T.T), axis=1) ** 2)
        if g > best_g:
            best, best_g = T, g
    return best


def _canonical_order(T, channels):
    """Order templates as classic A (LP-RA diagonal), B (RP-LA diagonal), C (anterior-posterior),
    D (fronto-central) by matching to prototypes built from 10-20 electrode positions."""
    try:
        import mne
        pos = mne.channels.make_standard_montage("standard_1020").get_positions()["ch_pos"]
        low = {k.lower(): v for k, v in pos.items()}
        alias = {"t3": "t7", "t4": "t8", "t5": "p7", "t6": "p8", "t7": "t3", "t8": "t4",
                 "p7": "t5", "p8": "t6"}
        P = []
        for ch in channels:
            c = ch.lower()
            p = low.get(c, low.get(alias.get(c, ""), None))
            if p is None:
                return T
            P.append(p)
        P = np.asarray(P, dtype=np.float64)
        P = P - P.mean(axis=0)
        x, y, z = P[:, 0], P[:, 1], P[:, 2]
        protos = _unit(np.stack([x + y, y - x, y, z]))
        cost = -np.abs(T @ protos.T)
        r, c = linear_sum_assignment(cost)
        order = np.empty(len(T), dtype=int)
        order[c] = r
        return T[order]
    except Exception:
        return T


def _ms_features(x, gfp, pk, T, fs):
    n = x.shape[1]
    if len(pk) < 2:
        return np.zeros(3 * N_MS + 1)
    maps = _unit(x[:, pk].T)
    corr = np.abs(maps @ T.T)
    lab = np.argmax(corr, axis=1)
    g2 = gfp[pk] ** 2
    gev = np.sum(g2 * corr.max(axis=1) ** 2) / (np.sum(g2) + 1e-12)
    bounds = (pk[:-1] + pk[1:]) // 2
    L = lab[np.searchsorted(bounds, np.arange(n), side="right")]
    starts = np.concatenate([[0], np.flatnonzero(np.diff(L)) + 1])
    lengths = np.diff(np.concatenate([starts, [n]]))
    seglab = L[starts]
    if len(starts) > 2:                                  # drop truncated edge segments
        lengths, seglab = lengths[1:-1], seglab[1:-1]
    dur_s = n / fs
    cov, dur, occ = [], [], []
    for k in range(N_MS):
        cov.append(np.mean(L == k))
        m = seglab == k
        occ.append(m.sum() / dur_s)
        dur.append(np.log(lengths[m].mean() / fs if m.any() else 1.0 / fs))
    return np.array(cov + dur + occ + [gev])


def _microstates(xs, fs, channels):
    prepped = [_ms_prep(x, fs) for x in xs]
    pool = []
    for x, gfp, pk in prepped:
        if len(pk) < N_MS:
            continue
        g = gfp[pk]
        keep = pk[g <= np.percentile(g, 95)]           # drop artifact-dominated peaks
        sel = keep[np.linspace(0, len(keep) - 1, min(MS_PEAKS_PER_SUBJ, len(keep))).astype(int)]
        pool.append(_unit(x[:, sel].T))
    T = _mod_kmeans(np.concatenate(pool, axis=0), N_MS)
    T = _canonical_order(T, channels)
    return np.stack([_ms_features(x, gfp, pk, T, fs) for x, gfp, pk in prepped])


def featurize(items, ctx):
    feats = []
    covs = []
    fs = ctx["fs"]
    blocks = {c: ctx["load_eeg"](items, c) for c in ctx["conditions"]}
    c0 = ctx["conditions"][0]
    for i in range(len(items)):
        f = []
        for c in ctx["conditions"]:
            x = blocks[c][i]
            if x is None:                      # missing condition -> fall back to eyes closed
                x = blocks[c0][i]
            f.append(_bandpower(x, fs).ravel())
        f.append(_wpli_strength(blocks[c0][i], fs).ravel())
        f.append(_spec_param(blocks[c0][i], fs).ravel())
        f.append(_alpha_dfa(blocks[c0][i], fs).ravel())
        feats.append(np.concatenate(f))
        covs.append(_band_cov(blocks[c0][i], fs))
    ms = _microstates(blocks[c0], fs, ctx["channels"])
    ts = _tangent(covs)
    return np.concatenate([np.asarray(feats), ms, ts], axis=1).astype(np.float32)


def fit_predict(F_train, y_train, F_test, ctx):
    sc = StandardScaler().fit(F_train)
    clf = LogisticRegression(C=0.1, max_iter=5000).fit(sc.transform(F_train), y_train)
    return clf.predict_proba(sc.transform(F_test))[:, 1]