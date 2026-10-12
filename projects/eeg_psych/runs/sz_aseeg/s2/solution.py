"""Log relative band power (delta/theta/alpha/beta/gamma) per channel, per-channel aperiodic
(1/f) exponent, per-channel individual alpha frequency (centre of gravity of the alpha power
above the aperiodic fit) from Welch PSD, per-channel wPLI node strength (mean weighted
phase-lag index to all other channels) in theta/alpha/beta, per-channel Higuchi fractal
dimension (median over 4-s epochs of the 1-45 Hz band-passed signal), plus trace-normalized
spatial covariances (broadband 1-45 Hz, theta 4-8 Hz, alpha 8-13 Hz, beta 13-30 Hz, low gamma
30-45 Hz; artifact epochs rejected on broadband power) each mapped to the Riemannian tangent
space at the training fold's log-Euclidean mean, for each available condition, standardized,
then each feature-family block rescaled to equal total variance (group-balanced L2 prior), L2
logistic regression with the penalty strength C chosen by repeated stratified-grouped inner CV
AUROC on the training fold."""
import numpy as np
from scipy.signal import welch, butter, sosfiltfilt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
CONN_BANDS = [(4, 8), (8, 13), (13, 30)]
COV_SUBBANDS = [(4, 8), (8, 13), (13, 30), (30, 45)]  # in addition to the broadband 1-45 Hz covariance
N_COV = 1 + len(COV_SUBBANDS)
HFD_KMAX = 8
COV_SHRINK = 0.05
C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0]
INNER_FOLDS = 5
INNER_REPEATS = 3


def _bandpass(x, fs, lo=1.0, hi=45.0):
    hi = min(hi, 0.45 * fs)
    lo = min(lo, 0.8 * hi)
    sos = butter(4, [lo, hi], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=-1)


def _spectral_feats(x, fs):
    f, p = welch(x, fs=fs, nperseg=int(2 * fs), axis=-1)
    tot = p[:, (f >= 1) & (f <= 45)].sum(axis=1, keepdims=True) + 1e-12
    rbp = np.log(np.stack([p[:, (f >= a) & (f < b)].sum(axis=1) for a, b in BANDS], axis=1) / tot)
    # aperiodic exponent: log-log linear fit over 2-40 Hz, excluding the alpha peak region
    m = (f >= 2) & (f <= 40) & ~((f >= 7) & (f <= 14))
    lf = np.log10(f[m])
    lp = np.log10(p[:, m] + 1e-20)
    lfm = lf.mean()
    lfc = lf - lfm
    lpm = lp.mean(axis=1, keepdims=True)
    slope = (lp - lpm) @ lfc / (lfc @ lfc)
    # individual alpha frequency: centre of gravity of the 7-13 Hz power above the 1/f fit
    am = (f >= 7) & (f <= 13)
    fa = f[am]
    fit = lpm + slope[:, None] * (np.log10(fa)[None, :] - lfm)
    w = np.clip(np.log10(p[:, am] + 1e-20) - fit, 0, None)
    ws = w.sum(axis=1)
    iaf = np.where(ws > 1e-9, (w @ fa) / np.maximum(ws, 1e-9), 10.0)
    return np.concatenate([rbp.ravel(), -slope, iaf])


def _wpli_feats(x, fs):
    # weighted phase-lag index from 2-s Hann-windowed epochs; node strength per channel and band
    n = int(2 * fs)
    n_ch = x.shape[0]
    n_ep = x.shape[1] // n
    if n_ep < 2:
        return np.zeros(n_ch * len(CONN_BANDS))
    ep = x[:, :n_ep * n].reshape(n_ch, n_ep, n).transpose(1, 0, 2)
    ep = ep - ep.mean(axis=-1, keepdims=True)
    X = np.fft.rfft(ep * np.hanning(n), axis=-1)            # (epochs, ch, freqs)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    off = ~np.eye(n_ch, dtype=bool)
    out = []
    for a, b in CONN_BANDS:
        Xb = X[:, :, (f >= a) & (f < b)]
        im = np.imag(Xb[:, :, None, :] * np.conj(Xb[:, None, :, :]))   # (e, c, c, f)
        num = np.abs(im.sum(axis=0))
        den = np.abs(im).sum(axis=0) + 1e-20
        wpli = (num / den).mean(axis=-1)                    # (c, c), averaged over band bins
        out.append((wpli * off).sum(axis=1) / (n_ch - 1))
    return np.concatenate(out)


def _epochs(xf, fs):
    n = int(4 * fs)
    n_ch = xf.shape[0]
    n_ep = xf.shape[1] // n
    if n_ep < 1:
        n, n_ep = xf.shape[1], 1
    return xf[:, :n_ep * n].reshape(n_ch, n_ep, n), n


def _hfd_feats(xf):
    # Higuchi fractal dimension per channel on 1-45 Hz band-passed 4-s epochs, median over epochs
    ep, n = xf
    ks = np.arange(1, HFD_KMAX + 1)
    logL = []
    for k in ks:
        Lm = []
        for m in range(k):
            d = np.abs(np.diff(ep[..., m::k], axis=-1))
            nm = d.shape[-1]
            Lm.append(d.sum(axis=-1) * (n - 1) / (nm * k) / k)
        logL.append(np.log(np.mean(Lm, axis=0) + 1e-20))
    logL = np.stack(logL, axis=-1)                          # (ch, ep, k)
    lk = np.log(1.0 / ks)
    lkc = lk - lk.mean()
    fd = (logL - logL.mean(axis=-1, keepdims=True)) @ lkc / (lkc @ lkc)
    return np.median(fd, axis=1)


def _epoch_covs(ep):
    e = ep.transpose(1, 0, 2)
    e = e - e.mean(axis=-1, keepdims=True)
    return np.einsum("ecn,edn->ecd", e, e) / e.shape[-1]


def _norm_cov(C):
    n_ch = C.shape[0]
    C = C / (np.trace(C) / n_ch + 1e-20)
    C = (1 - COV_SHRINK) * C + COV_SHRINK * np.eye(n_ch)
    return C[np.triu_indices(n_ch)]


def _cov_feats(x, xf, fs):
    # spatial covariances of 4-s epochs (broadband + sub-bands); epochs with high broadband
    # power (artifacts) are dropped from every band; each covariance is trace-normalized
    ep, _ = xf
    covs = _epoch_covs(ep)
    pw = np.trace(covs, axis1=1, axis2=2)
    keep = pw <= 3.0 * np.median(pw)
    if not keep.any():
        keep[:] = True
    out = [_norm_cov(covs[keep].mean(axis=0))]
    for a, b in COV_SUBBANDS:
        epb, _ = _epochs(_bandpass(x, fs, a, b), fs)
        cb = _epoch_covs(epb)
        out.append(_norm_cov(cb[keep].mean(axis=0)))
    return out


def featurize(items, ctx):
    feats, covf = [], []
    fs = ctx["fs"]
    blocks = {c: ctx["load_eeg"](items, c) for c in ctx["conditions"]}
    for i in range(len(items)):
        f, cv = [], []
        for c in ctx["conditions"]:
            x = blocks[c][i]
            if x is None:                      # missing condition -> fall back to eyes closed
                x = blocks[ctx["conditions"][0]][i]
            x = np.asarray(x, dtype=np.float64)
            f.append(_spectral_feats(x, fs))
            f.append(_wpli_feats(x, fs))
            xe = _epochs(_bandpass(x, fs), fs)
            f.append(_hfd_feats(xe))
            cv.extend(_cov_feats(x, xe, fs))
        feats.append(np.concatenate(f + cv))   # covariance blocks always last
    return np.asarray(feats, dtype=np.float64)


def _mfun(C, fn):
    w, V = np.linalg.eigh(C)
    return (V * fn(w)[..., None, :]) @ np.swapaxes(V, -1, -2)


def _tangent(cov_tr, cov_te, n_ch):
    iu = np.triu_indices(n_ch)

    def _mats(v):
        M = np.zeros((v.shape[0], n_ch, n_ch))
        M[:, iu[0], iu[1]] = v
        M = M + np.swapaxes(M, 1, 2)
        M[:, np.arange(n_ch), np.arange(n_ch)] /= 2
        return M

    Ctr, Cte = _mats(cov_tr), _mats(cov_te)
    floor = lambda w: np.log(np.maximum(w, 1e-10))
    ref = _mfun(_mfun(Ctr, floor).mean(axis=0), np.exp)
    isq = _mfun(ref, lambda w: 1.0 / np.sqrt(np.maximum(w, 1e-10)))
    wt = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))

    def _map(C):
        T = _mfun(isq @ C @ isq, floor)
        return T[:, iu[0], iu[1]] * wt

    return _map(Ctr), _map(Cte)


def _block_sizes(n_ch, n_cond, base):
    # per-condition family layout of the non-covariance features (order as in featurize)
    per = [len(BANDS) * n_ch, n_ch, n_ch, len(CONN_BANDS) * n_ch, n_ch]  # rbp, slope, iaf, wpli, hfd
    if sum(per) * n_cond != base:
        return [base]
    return per * n_cond


def _design(F_train, F_test, ctx):
    # tangent mapping, standardization and block balancing, all fitted on F_train only
    n_ch = len(ctx["channels"])
    n_cond = len(ctx["conditions"])
    m = n_ch * (n_ch + 1) // 2
    n_blk = n_cond * N_COV
    base = F_train.shape[1] - n_blk * m
    tr, te = [F_train[:, :base]], [F_test[:, :base]]
    for k in range(n_blk):
        s = slice(base + k * m, base + (k + 1) * m)
        a, b = _tangent(F_train[:, s], F_test[:, s], n_ch)
        tr.append(a)
        te.append(b)
    Xtr, Xte = np.hstack(tr), np.hstack(te)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    # equalize each feature family's total variance (keeps overall variance = n_features)
    sizes = _block_sizes(n_ch, n_cond, base) + [m] * n_blk
    P, B = Xtr.shape[1], len(sizes)
    w = np.concatenate([np.full(p, np.sqrt(P / (B * p))) for p in sizes])
    return Xtr * w, Xte * w


def _select_C(F_train, y, groups, ctx):
    # repeated stratified-grouped inner CV; the whole design is re-fitted inside each inner fold
    n_groups = len(np.unique(groups))
    k = min(INNER_FOLDS, n_groups, int(np.bincount(y).min()))
    if k < 2:
        return 0.1
    scores = np.zeros(len(C_GRID))
    n_ok = 0
    for r in range(INNER_REPEATS):
        cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=r)
        for itr, iva in cv.split(F_train, y, groups):
            if len(np.unique(y[iva])) < 2 or len(np.unique(y[itr])) < 2:
                continue
            Xa, Xb = _design(F_train[itr], F_train[iva], ctx)
            for j, C in enumerate(C_GRID):
                clf = LogisticRegression(C=C, max_iter=5000).fit(Xa, y[itr])
                scores[j] += roc_auc_score(y[iva], clf.decision_function(Xb))
            n_ok += 1
    if n_ok == 0:
        return 0.1
    best = np.flatnonzero(scores >= scores.max() - 1e-9)[0]   # ties -> stronger penalty
    return C_GRID[best]


def fit_predict(F_train, y_train, F_test, ctx):
    F_train = np.asarray(F_train, dtype=np.float64)
    F_test = np.asarray(F_test, dtype=np.float64)
    y = np.asarray(y_train).astype(int)
    groups = ctx.get("groups_train")
    groups = np.arange(len(y)) if groups is None else np.asarray(groups)
    C = _select_C(F_train, y, groups, ctx)
    Xtr, Xte = _design(F_train, F_test, ctx)
    clf = LogisticRegression(C=C, max_iter=5000).fit(Xtr, y)
    return clf.predict_proba(Xte)[:, 1]