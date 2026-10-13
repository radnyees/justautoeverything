"""Log relative band power per channel (Welch) with split alpha (8-10 / 10-13 Hz) and beta
(13-20 / 20-30 Hz) sub-bands to capture alpha slowing -> L2 logistic regression, late-fused with
Riemannian tangent-space models of inter-channel correlation matrices computed broadband (1-45 Hz)
and in theta / alpha / beta bands (frequency-specific, scale-free functional connectivity; one
tangent reference + classifier per band, fitted on the training fold only), plus a weighted
phase-lag index (wPLI) model in theta / alpha / beta that captures lagged (non-zero-phase) coupling
insensitive to volume conduction and the recording reference. Connectivity probabilities are averaged."""
import numpy as np
from scipy.signal import welch, butter, sosfiltfilt
from scipy.signal.windows import hann
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pyriemann.tangentspace import TangentSpace

BANDS = [(1, 4), (4, 8), (8, 10), (10, 13), (13, 20), (20, 30), (30, 45)]
CORR_BANDS = [(1, 45), (4, 8), (8, 13), (13, 30)]
WPLI_BANDS = [(4, 8), (8, 13), (13, 30)]


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


def _wpli(x, fs):
    """wPLI per band from 2-s Hann-windowed epochs; returns upper-triangle values per band."""
    x = np.asarray(x, dtype=np.float64)
    n_ch = x.shape[0]
    n = int(2 * fs)
    nep = max(1, x.shape[1] // n)
    ep = x[:, : nep * n].reshape(n_ch, nep, n).transpose(1, 0, 2)
    ep = ep - ep.mean(axis=-1, keepdims=True)
    ep = ep * hann(n, sym=False)
    X = np.fft.rfft(ep, axis=-1)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    iu = np.triu_indices(n_ch, k=1)
    out = []
    for lo, hi in WPLI_BANDS:
        Xb = X[:, :, (f >= lo) & (f < hi)]
        S = np.einsum("eif,ejf->eijf", Xb, np.conj(Xb)).imag
        num = np.abs(S.mean(axis=0))
        den = np.abs(S).mean(axis=0) + 1e-12
        w = (num / den).mean(axis=-1)
        out.append(w[iu])
    return np.concatenate(out)


def featurize(items, ctx):
    feats = []
    conds = ctx["conditions"]
    blocks = {c: ctx["load_eeg"](items, c) for c in conds}
    fs = ctx["fs"]
    for i in range(len(items)):
        bp, cv, wp = [], [], []
        for c in conds:
            x = blocks[c][i]
            if x is None:                      # missing condition -> fall back to first condition
                x = blocks[conds[0]][i]
            bp.append(_bandpower(x, fs).ravel())
            for lo, hi in CORR_BANDS:
                cv.append(_corr(x, fs, lo, hi).ravel())
            wp.append(_wpli(x, fs))
        feats.append(np.concatenate(bp + cv + wp))
    return np.asarray(feats, dtype=np.float64)


def _split(F, ctx):
    """Returns band powers, covs[band_index] = list (per condition) of (n, ch, ch) arrays, wPLI."""
    F = np.asarray(F, dtype=np.float64)
    n_ch, nc, nbands = len(ctx["channels"]), len(ctx["conditions"]), len(CORR_BANDS)
    nb = len(BANDS) * n_ch * nc
    m = n_ch * n_ch
    covs = [[] for _ in range(nbands)]
    for k in range(nc):
        for b in range(nbands):
            off = nb + (k * nbands + b) * m
            covs[b].append(F[:, off: off + m].reshape(-1, n_ch, n_ch))
    off_w = nb + nc * nbands * m
    return F[:, :nb], covs, F[:, off_w:]


def fit_predict(F_train, y_train, F_test, ctx):
    bp_tr, cov_tr, w_tr = _split(F_train, ctx)
    bp_te, cov_te, w_te = _split(F_test, ctx)

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

    sc3 = StandardScaler().fit(w_tr)
    clf3 = LogisticRegression(C=0.02, max_iter=5000).fit(sc3.transform(w_tr), y_train)
    p_bands.append(clf3.predict_proba(sc3.transform(w_te))[:, 1])
    p_ts = np.mean(p_bands, axis=0)

    return 0.5 * p_bp + 0.5 * p_ts