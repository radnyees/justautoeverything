"""Log relative band power (delta/theta/alpha/beta/gamma) per channel from Welch PSD of each
available condition -> standardized L2 logistic regression, averaged with a second L2 logistic
regression on Riemannian tangent-space features of trace-normalized filter-bank spatial
covariances (1-45 Hz broadband + theta, alpha, beta) per condition, a third L2 logistic
regression on debiased weighted phase-lag index (wPLI) connectivity (all channel pairs; delta,
theta, alpha, beta) per condition, and a fourth L2 logistic regression on orthogonalized
amplitude-envelope correlation (AEC, Hipp 2012; all channel pairs; delta, theta, alpha, beta)
per condition. The regularization strength of each logistic regression is chosen by
subject-grouped inner CV (AUROC) on the training fold."""
import numpy as np
from scipy.signal import welch, butter, sosfiltfilt, hilbert
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from pyriemann.tangentspace import TangentSpace

BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
COV_BANDS = [(1, 45), (4, 8), (8, 13), (13, 30)]
WPLI_BANDS = [(1, 4), (4, 8), (8, 13), (13, 30)]
AEC_BANDS = [(1, 4), (4, 8), (8, 13), (13, 30)]
CS = np.logspace(-4, 1, 11)


def _bandpower(x, fs):
    f, p = welch(x, fs=fs, nperseg=int(2 * fs), axis=-1)
    tot = p[:, (f >= 1) & (f <= 45)].sum(axis=1, keepdims=True) + 1e-12
    return np.log(np.stack([p[:, (f >= a) & (f < b)].sum(axis=1) for a, b in BANDS], axis=1) / tot)


def _cov(x, fs, lo, hi):
    sos = butter(4, [lo, hi], btype="bandpass", fs=fs, output="sos")
    xf = sosfiltfilt(sos, x.astype(np.float64), axis=-1)
    xf = xf - xf.mean(axis=1, keepdims=True)
    c = xf @ xf.T / xf.shape[1]
    c = c / np.trace(c)                                  # amplitude-scale invariance
    n = c.shape[0]
    c = c + 1e-3 * np.eye(n) / n                         # keep SPD (e.g. rank-deficient references)
    return c


def _wpli(x, fs):
    """Debiased wPLI (Vinck 2011) per band over 2-s Hann-windowed epochs x frequency bins."""
    x = np.asarray(x, dtype=np.float64)
    nch = x.shape[0]
    seg = int(2 * fs)
    n = max(1, x.shape[1] // seg)
    if x.shape[1] < seg:
        x = np.pad(x, ((0, 0), (0, seg - x.shape[1])))
    X = x[:, :n * seg].reshape(nch, n, seg)
    X = X - X.mean(axis=-1, keepdims=True)
    X = np.fft.rfft(X * np.hanning(seg), axis=-1)
    f = np.fft.rfftfreq(seg, 1.0 / fs)
    iu = np.triu_indices(nch, 1)
    out = []
    for lo, hi in WPLI_BANDS:
        Xb = X[:, :, (f >= lo) & (f < hi)].reshape(nch, -1)
        im = np.imag(Xb[iu[0]] * np.conj(Xb[iu[1]]))     # pairs x (epochs*bins)
        s = im.sum(axis=1)
        s2 = (im ** 2).sum(axis=1)
        sa = np.abs(im).sum(axis=1)
        w = (s ** 2 - s2) / (sa ** 2 - s2 + 1e-20)
        out.append(np.clip(w, -1.0, 1.0))
    return np.concatenate(out)


def _aec(x, fs):
    """Orthogonalized amplitude-envelope correlation (Hipp 2012), symmetrized, Fisher-z."""
    x = np.asarray(x, dtype=np.float64)
    nch = x.shape[0]
    iu = np.triu_indices(nch, 1)
    edge = int(fs)                                       # trim filter/Hilbert edge effects
    out = []
    for lo, hi in AEC_BANDS:
        sos = butter(4, [lo, hi], btype="bandpass", fs=fs, output="sos")
        z = hilbert(sosfiltfilt(sos, x, axis=-1), axis=-1)
        if z.shape[1] > 4 * edge:
            z = z[:, edge:-edge]
        amp = np.abs(z) + 1e-12
        A = np.zeros((nch, nch))
        for i in range(nch):
            e_i = amp[i] - amp[i].mean()
            e_o = np.abs(np.imag(z * (np.conj(z[i]) / amp[i])[None, :]))   # leakage-free envelopes
            e_o = e_o - e_o.mean(axis=1, keepdims=True)
            num = e_o @ e_i
            den = np.sqrt((e_o ** 2).sum(axis=1) * (e_i ** 2).sum()) + 1e-20
            A[i] = num / den
        A = 0.5 * (A + A.T)
        out.append(np.arctanh(np.clip(A[iu], -0.999, 0.999)))
    return np.concatenate(out)


def featurize(items, ctx):
    feats, covs, wps, aecs = [], [], [], []
    fs = ctx["fs"]
    blocks = {c: ctx["load_eeg"](items, c) for c in ctx["conditions"]}
    for i in range(len(items)):
        f, cv, wp, ae = [], [], [], []
        for c in ctx["conditions"]:
            x = blocks[c][i]
            if x is None:                      # missing condition -> fall back to eyes closed
                x = blocks[ctx["conditions"][0]][i]
            f.append(_bandpower(x, fs).ravel())
            for lo, hi in COV_BANDS:
                cv.append(_cov(x, fs, lo, hi).ravel())
            wp.append(_wpli(x, fs))
            ae.append(_aec(x, fs))
        feats.append(np.concatenate(f))
        covs.append(np.concatenate(cv))
        wps.append(np.concatenate(wp))
        aecs.append(np.concatenate(ae))
    F = np.concatenate([np.asarray(feats, dtype=np.float64), np.asarray(covs, dtype=np.float64),
                        np.asarray(wps, dtype=np.float64), np.asarray(aecs, dtype=np.float64)],
                       axis=1)
    return F


def _split(F, ctx):
    F = np.asarray(F, dtype=np.float64)
    nch = len(ctx["channels"])
    ncond = len(ctx["conditions"])
    npair = nch * (nch - 1) // 2
    nbp = ncond * nch * len(BANDS)
    ncov = ncond * len(COV_BANDS) * nch * nch
    nwp = ncond * len(WPLI_BANDS) * npair
    bp = F[:, :nbp]
    covs = F[:, nbp:nbp + ncov].reshape(len(F), ncond * len(COV_BANDS), nch, nch)
    covs = 0.5 * (covs + covs.transpose(0, 1, 3, 2))
    wp = F[:, nbp + ncov:nbp + ncov + nwp]
    ae = F[:, nbp + ncov + nwp:]
    return bp, covs, wp, ae


def _inner_splits(y, ctx):
    groups = ctx.get("groups_train")
    if groups is None:
        groups = np.arange(len(y))
    groups = np.asarray(groups)
    k = int(min(5, len(np.unique(groups))))
    return list(GroupKFold(n_splits=k).split(np.zeros(len(y)), y, groups))


def _fit_lr(X_tr, y, X_te, splits):
    sc = StandardScaler().fit(X_tr)
    Xs_tr, Xs_te = sc.transform(X_tr), sc.transform(X_te)
    Xs_tr = np.nan_to_num(Xs_tr)
    Xs_te = np.nan_to_num(Xs_te)
    try:
        cv = LogisticRegressionCV(Cs=CS, cv=splits, scoring="roc_auc", max_iter=5000).fit(Xs_tr, y)
        C = float(cv.C_[0])
    except Exception:
        C = 0.1
    clf = LogisticRegression(C=C, max_iter=5000).fit(Xs_tr, y)
    return clf.predict_proba(Xs_te)[:, 1]


def fit_predict(F_train, y_train, F_test, ctx):
    y_train = np.asarray(y_train)
    bp_tr, cov_tr, wp_tr, ae_tr = _split(F_train, ctx)
    bp_te, cov_te, wp_te, ae_te = _split(F_test, ctx)
    splits = _inner_splits(y_train, ctx)

    p_bp = _fit_lr(bp_tr, y_train, bp_te, splits)

    T_tr, T_te = [], []
    for k in range(cov_tr.shape[1]):
        ts = TangentSpace(metric="riemann").fit(cov_tr[:, k])
        T_tr.append(ts.transform(cov_tr[:, k]))
        T_te.append(ts.transform(cov_te[:, k]))
    T_tr, T_te = np.concatenate(T_tr, axis=1), np.concatenate(T_te, axis=1)
    p_ts = _fit_lr(T_tr, y_train, T_te, splits)

    p_wp = _fit_lr(wp_tr, y_train, wp_te, splits)

    p_ae = _fit_lr(ae_tr, y_train, ae_te, splits)

    return (p_bp + p_ts + p_wp + p_ae) / 4.0