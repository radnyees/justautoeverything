"""Log relative band power per channel (Welch) -> L2 logistic regression, late-fused with
Riemannian tangent-space models of inter-channel correlation matrices computed broadband (1-45 Hz)
and in theta / alpha / beta bands, plus amplitude-envelope correlation (AEC) matrices of the
Hilbert envelopes in theta / alpha / beta (phase-independent amplitude co-fluctuation coupling).
One tangent reference + classifier per matrix type, fitted on the training fold only,
probabilities averaged."""
import numpy as np
from scipy.signal import welch, butter, sosfiltfilt, hilbert
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pyriemann.tangentspace import TangentSpace

BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
CORR_BANDS = [(1, 45), (4, 8), (8, 13), (13, 30)]
ENV_BANDS = [(4, 8), (8, 13), (13, 30)]


def _bandpower(x, fs):
    f, p = welch(x, fs=fs, nperseg=int(2 * fs), axis=-1)
    tot = p[:, (f >= 1) & (f <= 45)].sum(axis=1, keepdims=True) + 1e-12
    return np.log(np.stack([p[:, (f >= a) & (f < b)].sum(axis=1) for a, b in BANDS], axis=1) / tot)


def _to_corr(z):
    c = np.cov(z)
    d = np.sqrt(np.diag(c)) + 1e-9
    r = c / np.outer(d, d)
    return 0.99 * r + 0.01 * np.eye(r.shape[0])   # keep strictly SPD


def _bandpass(x, fs, lo, hi):
    sos = butter(4, [lo, hi], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, np.asarray(x, dtype=np.float64), axis=-1)


def _corr(x, fs, lo, hi):
    return _to_corr(_bandpass(x, fs, lo, hi))


def _env_corr(x, fs, lo, hi):
    env = np.abs(hilbert(_bandpass(x, fs, lo, hi), axis=-1))
    return _to_corr(env)


def featurize(items, ctx):
    feats = []
    conds = ctx["conditions"]
    blocks = {c: ctx["load_eeg"](items, c) for c in conds}
    fs = ctx["fs"]
    for i in range(len(items)):
        bp, cv, ev = [], [], []
        for c in conds:
            x = blocks[c][i]
            if x is None:                      # missing condition -> fall back to first condition
                x = blocks[conds[0]][i]
            bp.append(_bandpower(x, fs).ravel())
            for lo, hi in CORR_BANDS:
                cv.append(_corr(x, fs, lo, hi).ravel())
            for lo, hi in ENV_BANDS:
                ev.append(_env_corr(x, fs, lo, hi).ravel())
        feats.append(np.concatenate(bp + cv + ev))
    return np.asarray(feats, dtype=np.float64)


def _split(F, ctx):
    """Returns band powers and mats[type_index] = list (per condition) of (n, ch, ch) arrays,
    types = CORR_BANDS followed by ENV_BANDS."""
    F = np.asarray(F, dtype=np.float64)
    n_ch, nc = len(ctx["channels"]), len(ctx["conditions"])
    ncorr, nenv = len(CORR_BANDS), len(ENV_BANDS)
    nb = len(BANDS) * n_ch * nc
    m = n_ch * n_ch
    mats = [[] for _ in range(ncorr + nenv)]
    for k in range(nc):
        for b in range(ncorr):
            off = nb + (k * ncorr + b) * m
            mats[b].append(F[:, off: off + m].reshape(-1, n_ch, n_ch))
    env0 = nb + nc * ncorr * m
    for k in range(nc):
        for b in range(nenv):
            off = env0 + (k * nenv + b) * m
            mats[ncorr + b].append(F[:, off: off + m].reshape(-1, n_ch, n_ch))
    return F[:, :nb], mats


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