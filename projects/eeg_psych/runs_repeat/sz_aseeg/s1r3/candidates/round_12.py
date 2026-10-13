"""Log relative band power per channel (Welch) with split alpha (8-10 / 10-13 Hz) and beta
(13-20 / 20-30 Hz) sub-bands, plus EEG microstate dynamics (4 polarity-invariant group templates
from label-free modified k-means on GFP peaks; coverage, duration, occurrence, GEV, transitions)
-> L2 logistic regression, late-fused with Riemannian tangent-space models of inter-channel
correlation matrices computed broadband (1-45 Hz) and in theta / alpha / beta bands (one tangent
reference + classifier per band, fitted on the training fold only, probabilities averaged)."""
import numpy as np
from scipy.signal import welch, butter, sosfiltfilt, find_peaks
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pyriemann.tangentspace import TangentSpace

BANDS = [(1, 4), (4, 8), (8, 10), (10, 13), (13, 20), (20, 30), (30, 45)]
CORR_BANDS = [(1, 45), (4, 8), (8, 13), (13, 30)]
N_MS = 4
N_MS_FEAT = 4 * N_MS + 1 + N_MS * (N_MS - 1)


def _bandpower(x, fs):
    f, p = welch(x, fs=fs, nperseg=int(2 * fs), axis=-1)
    tot = p[:, (f >= 1) & (f <= 45)].sum(axis=1, keepdims=True) + 1e-12
    return np.log(np.stack([p[:, (f >= a) & (f < b)].sum(axis=1) for a, b in BANDS], axis=1) / tot)


def _corr(x, fs, lo, hi):
    sos = butter(4, [lo, hi], btype="band", fs=fs, output="sos")
    xf = sosfiltfilt(sos, np.asarray(x, dtype=np.float64), axis=-1)
    c = np.cov(xf)
    d = np.sqrt(np.diag(c)) + 1e-9
    r = c / np.outer(d, d)
    return 0.99 * r + 0.01 * np.eye(r.shape[0])   # keep strictly SPD


def _ms_prep(x, fs):
    """2-20 Hz, average reference (microstate topographies only), GFP and its peaks."""
    sos = butter(4, [2, 20], btype="band", fs=fs, output="sos")
    xf = sosfiltfilt(sos, np.asarray(x, dtype=np.float64), axis=-1)
    xf = xf - xf.mean(axis=0, keepdims=True)
    gfp = xf.std(axis=0)
    peaks, _ = find_peaks(gfp)
    return xf, gfp, peaks


def _modkmeans(V, k=N_MS, n_init=10, n_iter=100, seed=0):
    """Polarity-invariant modified k-means on unit-norm maps V (n x ch)."""
    rng = np.random.RandomState(seed)
    best, best_gev = None, -np.inf
    for _ in range(n_init):
        T = V[rng.choice(len(V), k, replace=False)].copy()
        prev = None
        for _ in range(n_iter):
            lab = np.argmax(np.abs(V @ T.T), axis=1)
            if prev is not None and np.array_equal(lab, prev):
                break
            prev = lab
            for j in range(k):
                S = V[lab == j]
                if len(S) == 0:
                    continue
                _, vec = np.linalg.eigh(S.T @ S)
                T[j] = vec[:, -1]
        gev = (np.max(np.abs(V @ T.T), axis=1) ** 2).sum()
        if gev > best_gev:
            best, best_gev = T.copy(), gev
    return best


def _ms_features(xf, gfp, peaks, T, fs):
    n = xf.shape[1]
    feat = np.zeros(N_MS_FEAT)
    if len(peaks) < 2:
        return feat
    X = xf[:, peaks]
    corr = (T @ X) / (np.linalg.norm(X, axis=0, keepdims=True) + 1e-12)
    lab_p = np.argmax(np.abs(corr), axis=0)
    g2 = gfp[peaks] ** 2
    gev = (g2 * np.max(corr ** 2, axis=0)).sum() / (g2.sum() + 1e-12)
    mids = (peaks[:-1] + peaks[1:]) / 2.0
    lab = lab_p[np.searchsorted(mids, np.arange(n))]
    starts = np.concatenate([[0], np.flatnonzero(np.diff(lab) != 0) + 1])
    lens = np.diff(np.concatenate([starts, [n]]))
    rl = lab[starts]
    dur_s = n / fs
    cov = np.array([np.mean(lab == j) for j in range(N_MS)])
    dur = np.array([lens[rl == j].mean() / fs if np.any(rl == j) else 0.0 for j in range(N_MS)])
    occ = np.array([np.sum(rl == j) / dur_s for j in range(N_MS)])
    M = np.zeros((N_MS, N_MS))
    np.add.at(M, (rl[:-1], rl[1:]), 1.0)
    M /= M.sum() + 1e-12
    trans = M[~np.eye(N_MS, dtype=bool)]
    return np.concatenate([cov, dur, occ, [gev], trans])


def featurize(items, ctx):
    conds = ctx["conditions"]
    blocks = {c: ctx["load_eeg"](items, c) for c in conds}
    fs = ctx["fs"]

    def get(c, i):
        x = blocks[c][i]
        return blocks[conds[0]][i] if x is None else x

    # label-free group microstate templates from GFP-peak maps (first condition)
    rng = np.random.RandomState(0)
    pooled = []
    for i in range(len(items)):
        x = get(conds[0], i)
        if x is None:
            continue
        xf, gfp, peaks = _ms_prep(x, fs)
        if len(peaks) == 0:
            continue
        g = gfp[peaks]
        peaks = peaks[g <= g.mean() + 3 * g.std()]           # drop artifact-sized peaks
        if len(peaks) > 500:
            peaks = np.sort(rng.choice(peaks, 500, replace=False))
        maps = xf[:, peaks].T
        pooled.append(maps / (np.linalg.norm(maps, axis=1, keepdims=True) + 1e-12))
    T = _modkmeans(np.vstack(pooled))

    feats = []
    for i in range(len(items)):
        bp, cv, ms = [], [], []
        for c in conds:
            x = get(c, i)
            bp.append(_bandpower(x, fs).ravel())
            for lo, hi in CORR_BANDS:
                cv.append(_corr(x, fs, lo, hi).ravel())
            ms.append(_ms_features(*_ms_prep(x, fs), T, fs))
        feats.append(np.concatenate(bp + cv + ms))
    return np.asarray(feats, dtype=np.float64)


def _split(F, ctx):
    """Returns local features (band powers + microstates) and covs[band_index] = list (per
    condition) of (n, ch, ch) arrays."""
    F = np.asarray(F, dtype=np.float64)
    n_ch, nc, nbands = len(ctx["channels"]), len(ctx["conditions"]), len(CORR_BANDS)
    nb = len(BANDS) * n_ch * nc
    m = n_ch * n_ch
    covs = [[] for _ in range(nbands)]
    for k in range(nc):
        for b in range(nbands):
            off = nb + (k * nbands + b) * m
            covs[b].append(F[:, off: off + m].reshape(-1, n_ch, n_ch))
    ms_off = nb + nc * nbands * m
    local = np.hstack([F[:, :nb], F[:, ms_off: ms_off + nc * N_MS_FEAT]])
    return local, covs


def fit_predict(F_train, y_train, F_test, ctx):
    bp_tr, cov_tr = _split(F_train, ctx)
    bp_te, cov_te = _split(F_test, ctx)

    sc = StandardScaler().fit(bp_tr)
    clf = LogisticRegression(C=0.1, max_iter=5000).fit(sc.transform(bp_tr), y_train)
    p_bp = clf.predict_proba(sc.transform(bp_te))[:, 1]

    p_bands = []
    for band_tr, band_te in zip(cov_tr, cov_te):
        T_tr, T_te = [], []
        for ctr, cte in zip(band_tr, band_te):
            ts = TangentSpace(metric="riemann").fit(ctr)
            T_tr.append(ts.transform(ctr))
            T_te.append(ts.transform(cte))
        T_tr, T_te = np.hstack(T_tr), np.hstack(T_te)
        sc2 = StandardScaler().fit(T_tr)
        clf2 = LogisticRegression(C=0.02, max_iter=5000).fit(sc2.transform(T_tr), y_train)
        p_bands.append(clf2.predict_proba(sc2.transform(T_te))[:, 1])
    p_ts = np.mean(p_bands, axis=0)

    return 0.5 * p_bp + 0.5 * p_ts