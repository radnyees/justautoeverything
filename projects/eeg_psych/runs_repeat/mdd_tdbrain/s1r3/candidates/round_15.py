"""Log relative band power (delta/theta/alpha/beta/gamma) per channel from Welch PSD of each
available condition, plus eyes-closed vs eyes-open reactivity (log ratio of absolute band power
and total 1-45 Hz power between the first and each other condition, per channel; scale-free
within subject) -> standardized L2 logistic regression, averaged with a second L2 logistic
regression on Riemannian tangent-space features of trace-normalized filter-bank spatial
covariances (1-45 Hz broadband + theta, alpha, beta) per condition, estimated only on clean
2-s epochs (epochs whose 1-45 Hz log-variance is a robust within-recording outlier on any channel
are rejected); this covariance model is trained on the full-recording covariances augmented with
covariances of N_CHUNK contiguous chunks of the clean epochs (within-subject augmentation,
subject-grouped inner CV) and predicts from the full-recording covariances; a third L2 logistic
regression on debiased weighted phase-lag index (wPLI) connectivity (all channel pairs; delta,
theta, alpha, beta) per condition, a fourth L2 logistic regression on orthogonalized
amplitude-envelope correlation (AEC, Hipp 2012; all channel pairs; delta, theta, alpha, beta)
per condition, and a fifth L2 logistic regression on spectral-shape features per channel and
condition (aperiodic 1/f exponent, individual alpha peak frequency, alpha peak height above the
aperiodic fit). The regularization strength of each logistic regression is chosen by
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
N_SHAPE = 3
N_CHUNK = 3
CS = np.logspace(-4, 1, 11)
REJ_Z = 3.0


def _bandpower(x, fs):
    f, p = welch(x, fs=fs, nperseg=int(2 * fs), axis=-1)
    tot = p[:, (f >= 1) & (f <= 45)].sum(axis=1, keepdims=True) + 1e-12
    return np.log(np.stack([p[:, (f >= a) & (f < b)].sum(axis=1) for a, b in BANDS], axis=1) / tot)


def _abspower(x, fs):
    """Log absolute band power + log total 1-45 Hz power per channel (nch x (len(BANDS)+1))."""
    f, p = welch(np.asarray(x, dtype=np.float64), fs=fs, nperseg=int(2 * fs), axis=-1)
    bp = [p[:, (f >= a) & (f < b)].sum(axis=1) for a, b in BANDS]
    bp.append(p[:, (f >= 1) & (f <= 45)].sum(axis=1))
    return np.log(np.stack(bp, axis=1) + 1e-12)


def _shape(x, fs):
    """Aperiodic exponent (log-log fit 2-40 Hz excluding 6-14 Hz, with one robust refit),
    alpha peak frequency (centroid of positive residual 7-14 Hz) and alpha peak height."""
    f, p = welch(np.asarray(x, dtype=np.float64), fs=fs, nperseg=int(4 * fs), axis=-1)
    lf = np.log10(f[(f >= 2) & (f <= 40)])
    lp = np.log10(p[:, (f >= 2) & (f <= 40)] + 1e-20)
    ff = f[(f >= 2) & (f <= 40)]
    fitm = (ff < 6) | (ff > 14)
    A = np.stack([np.ones(fitm.sum()), lf[fitm]], axis=1)
    out = []
    for ch in range(lp.shape[0]):
        yv = lp[ch, fitm]
        w = np.ones_like(yv)
        for _ in range(2):                                # robust reweighting against bumps
            coef = np.linalg.lstsq(A * w[:, None], yv * w, rcond=None)[0]
            r = yv - A @ coef
            s = 1.4826 * np.median(np.abs(r)) + 1e-9
            w = 1.0 / np.maximum(1.0, np.abs(r) / (2.0 * s))
        res = lp[ch] - (coef[0] + coef[1] * lf)
        am = (ff >= 7) & (ff <= 14)
        ra = res[am]
        pos = np.clip(ra, 0, None)
        paf = float((ff[am] * pos).sum() / pos.sum()) if pos.sum() > 1e-9 else 10.0
        out.append([-coef[1], paf, float(ra.max())])
    return np.asarray(out).ravel()


def _clean_epochs(x, fs):
    """Boolean mask of clean 2-s epochs: reject epochs whose 1-45 Hz log-variance is a robust
    (median/MAD) upward outlier on any channel; always keep at least half of the epochs."""
    seg = int(2 * fs)
    n = x.shape[1] // seg
    if n < 4:
        return None
    sos = butter(4, [1, 45], btype="bandpass", fs=fs, output="sos")
    xf = sosfiltfilt(sos, np.asarray(x, dtype=np.float64), axis=-1)
    lv = np.log(xf[:, :n * seg].reshape(x.shape[0], n, seg).var(axis=-1) + 1e-20)
    med = np.median(lv, axis=1, keepdims=True)
    mad = np.maximum(1.4826 * np.median(np.abs(lv - med), axis=1, keepdims=True), 0.05)
    score = ((lv - med) / mad).max(axis=0)
    keep = score <= REJ_Z
    if keep.sum() < n / 2.0:
        keep = score <= np.median(score)
    return keep


def _ncov(xf):
    xf = xf - xf.mean(axis=1, keepdims=True)
    c = xf @ xf.T / xf.shape[1]
    c = c / np.trace(c)                                  # amplitude-scale invariance
    n = c.shape[0]
    c = c + 1e-3 * np.eye(n) / n                         # keep SPD (e.g. rank-deficient references)
    return c


def _covs(x, fs, lo, hi, keep=None):
    """Full clean-epoch covariance plus N_CHUNK covariances of contiguous chunks of clean epochs."""
    sos = butter(4, [lo, hi], btype="bandpass", fs=fs, output="sos")
    xf = sosfiltfilt(sos, x.astype(np.float64), axis=-1)
    nch = xf.shape[0]
    seg = int(2 * fs)
    if keep is not None:                                 # clean epochs only
        n = len(keep)
        E = xf[:, :n * seg].reshape(nch, n, seg)[:, keep]
        full = _ncov(E.reshape(nch, -1))
    else:
        full = _ncov(xf)
        n = xf.shape[1] // seg
        E = xf[:, :n * seg].reshape(nch, n, seg) if n >= 1 else xf[:, None, :]
    m = E.shape[1]
    chunks = []
    for idx in np.array_split(np.arange(m), N_CHUNK):
        chunks.append(_ncov(E[:, idx].reshape(nch, -1)) if len(idx) >= 2 else full)
    return full, chunks


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
    feats, covs, ckcovs, wps, aecs, shps, rxs = [], [], [], [], [], [], []
    fs = ctx["fs"]
    conds = ctx["conditions"]
    blocks = {c: ctx["load_eeg"](items, c) for c in conds}
    for i in range(len(items)):
        f, cv, wp, ae, sh, ap = [], [], [], [], [], []
        ck = [[] for _ in range(N_CHUNK)]
        for c in conds:
            x = blocks[c][i]
            if x is None:                      # missing condition -> fall back to eyes closed
                x = blocks[conds[0]][i]
            f.append(_bandpower(x, fs).ravel())
            ap.append(_abspower(x, fs))
            keep = _clean_epochs(x, fs)
            for lo, hi in COV_BANDS:
                full, chunks = _covs(x, fs, lo, hi, keep)
                cv.append(full.ravel())
                for k in range(N_CHUNK):
                    ck[k].append(chunks[k].ravel())
            wp.append(_wpli(x, fs))
            ae.append(_aec(x, fs))
            sh.append(_shape(x, fs))
        # reactivity: log power ratio first condition (EC) vs each other condition (EO);
        # gain cancels within subject; a missing condition gives 0 (no reactivity info)
        rx = [(ap[0] - ap[j]).ravel() for j in range(1, len(conds))]
        feats.append(np.concatenate(f))
        covs.append(np.concatenate(cv))
        ckcovs.append(np.concatenate([np.concatenate(ck[k]) for k in range(N_CHUNK)]))
        wps.append(np.concatenate(wp))
        aecs.append(np.concatenate(ae))
        shps.append(np.concatenate(sh))
        rxs.append(np.concatenate(rx) if rx else np.zeros(0))
    F = np.concatenate([np.asarray(feats, dtype=np.float64), np.asarray(covs, dtype=np.float64),
                        np.asarray(ckcovs, dtype=np.float64),
                        np.asarray(wps, dtype=np.float64), np.asarray(aecs, dtype=np.float64),
                        np.asarray(shps, dtype=np.float64),
                        np.asarray(rxs, dtype=np.float64).reshape(len(items), -1)],
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
    nae = ncond * len(AEC_BANDS) * npair
    nsh = ncond * nch * N_SHAPE
    bp = F[:, :nbp]
    covs = F[:, nbp:nbp + ncov].reshape(len(F), ncond * len(COV_BANDS), nch, nch)
    covs = 0.5 * (covs + covs.transpose(0, 1, 3, 2))
    o = nbp + ncov
    ck = F[:, o:o + N_CHUNK * ncov].reshape(len(F), N_CHUNK, ncond * len(COV_BANDS), nch, nch)
    ck = 0.5 * (ck + ck.transpose(0, 1, 2, 4, 3))
    o += N_CHUNK * ncov
    wp = F[:, o:o + nwp]
    o += nwp
    ae = F[:, o:o + nae]
    o += nae
    sh = F[:, o:o + nsh]
    o += nsh
    rx = F[:, o:]
    if rx.shape[1] > 0:
        bp = np.concatenate([bp, rx], axis=1)
    return bp, covs, ck, wp, ae, sh


def _groups(y, ctx):
    groups = ctx.get("groups_train")
    if groups is None:
        groups = np.arange(len(y))
    return np.asarray(groups)


def _group_splits(y, groups):
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
    bp_tr, cov_tr, ck_tr, wp_tr, ae_tr, sh_tr = _split(F_train, ctx)
    bp_te, cov_te, _, wp_te, ae_te, sh_te = _split(F_test, ctx)
    groups = _groups(y_train, ctx)
    splits = _group_splits(y_train, groups)
    nch = cov_tr.shape[-1]

    p_bp = _fit_lr(bp_tr, y_train, bp_te, splits)

    # covariance model: train on full + chunk covariances (within-subject augmentation)
    T_tr, T_ck, T_te = [], [], []
    for k in range(cov_tr.shape[1]):
        ts = TangentSpace(metric="riemann").fit(cov_tr[:, k])
        T_tr.append(ts.transform(cov_tr[:, k]))
        T_ck.append(ts.transform(ck_tr[:, :, k].reshape(-1, nch, nch)))   # subject-major rows
        T_te.append(ts.transform(cov_te[:, k]))
    T_tr = np.concatenate(T_tr, axis=1)
    T_ck = np.concatenate(T_ck, axis=1)
    T_te = np.concatenate(T_te, axis=1)
    X_aug = np.vstack([T_tr, T_ck])
    y_aug = np.concatenate([y_train, np.repeat(y_train, N_CHUNK)])
    g_aug = np.concatenate([groups, np.repeat(groups, N_CHUNK)])
    p_ts = _fit_lr(X_aug, y_aug, T_te, _group_splits(y_aug, g_aug))

    p_wp = _fit_lr(wp_tr, y_train, wp_te, splits)

    p_ae = _fit_lr(ae_tr, y_train, ae_te, splits)

    p_sh = _fit_lr(sh_tr, y_train, sh_te, splits)

    return (p_bp + p_ts + p_wp + p_ae + p_sh) / 5.0