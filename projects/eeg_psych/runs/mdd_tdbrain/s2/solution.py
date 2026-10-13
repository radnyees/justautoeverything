"""Log relative band power (delta/theta/low-alpha/high-alpha/beta/gamma), spatially normalized log
total power (cordance-style topography of absolute power, gain-invariant), aperiodic (1/f) exponent
and individual alpha peak frequency (centre of gravity of the above-aperiodic PSD in 7-14 Hz) per
channel from a median-averaged (artifact-robust) Welch PSD of each available condition, plus
per-channel coherence strength (theta/low-alpha/high-alpha/beta) from artifact-screened 2-s
segments, per-channel amplitude-envelope correlation strength (slow co-fluctuation of band
amplitude, artifact-screened 0.5-s windows) and per-channel ORTHOGONALIZED envelope correlation
strength (zero-lag leakage removed, Hipp 2012), standardized, L2 logistic regression; late-fused
(probability average) with an L2 logistic regression on the Riemannian tangent-space projection of
the band-limited correlation matrices and the amplitude-envelope correlation matrices
(theta/low-alpha/high-alpha/beta)."""
import numpy as np
from scipy.signal import welch, butter, sosfiltfilt, hilbert
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pyriemann.tangentspace import TangentSpace

# alpha split for power too: low alpha (global, vigilance-related) vs high alpha (more local)
BANDS = [(1, 4), (4, 8), (8, 10), (10, 13), (13, 30), (30, 45)]
# alpha split: low alpha (global, vigilance-related) vs high alpha (more local) coupling
COH_BANDS = [(4, 8), (8, 10), (10, 13), (13, 30)]
SHRINK = 0.1


def _spectral(x, fs):
    # median across 2-s segments: robust to transient artifacts (blinks, movement, pops)
    f, p = welch(x, fs=fs, nperseg=int(2 * fs), axis=-1, average="median")
    tot = p[:, (f >= 1) & (f <= 45)].sum(axis=1, keepdims=True) + 1e-12
    rel = np.log(np.stack([p[:, (f >= a) & (f < b)].sum(axis=1) for a, b in BANDS], axis=1) / tot + 1e-12)
    # spatial topography of absolute power: log total power relative to the subject's channel
    # mean (global amplifier gain cancels); with relative power this spans cordance-type info
    ltot = np.log(tot[:, 0])
    topo = ltot - ltot.mean()
    # aperiodic exponent: log-log slope over 2-40 Hz excluding the alpha peak (scale-invariant)
    m = (f >= 2) & (f <= 40) & ~((f >= 7) & (f <= 14))
    lf = np.log10(f[m])
    lp = np.log10(p[:, m] + 1e-12)
    lf_c = lf - lf.mean()
    slope = ((lp - lp.mean(axis=1, keepdims=True)) * lf_c).sum(axis=1) / (lf_c ** 2).sum()
    intercept = lp.mean(axis=1) - slope * lf.mean()
    # individual alpha frequency: centre of gravity of the PSD excess above the aperiodic fit
    am = (f >= 7) & (f <= 14)
    fa = f[am]
    resid = np.log10(p[:, am] + 1e-12) - (intercept[:, None] + slope[:, None] * np.log10(fa)[None, :])
    w = np.clip(resid, 0.0, None)
    wsum = w.sum(axis=1)
    raw = p[:, am]
    cog_raw = (raw * fa).sum(axis=1) / (raw.sum(axis=1) + 1e-12)
    iaf = np.where(wsum > 1e-6, (w * fa).sum(axis=1) / (wsum + 1e-12), cog_raw)
    return np.concatenate([rel.ravel(), topo, slope, iaf])


def _coherence(x, fs):
    # per-channel mean magnitude-squared coherence with all other channels, per band,
    # plus the band-limited (co-spectral) correlation matrix for the Riemannian branch
    n = int(2 * fs)
    starts = np.arange(0, x.shape[1] - n + 1, n // 2)
    segs = np.stack([x[:, s:s + n] for s in starts])
    segs = segs - segs.mean(axis=-1, keepdims=True)
    Z = np.fft.rfft(segs * np.hanning(n), axis=-1)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    # drop segments whose broadband log power is a robust outlier (blinks, movement, pops)
    lp = np.log((np.abs(Z[:, :, (f >= 1) & (f <= 45)]) ** 2).sum(axis=-1) + 1e-12).mean(axis=1)
    med = np.median(lp)
    mad = np.median(np.abs(lp - med)) + 1e-12
    keep = lp <= med + 3 * 1.4826 * mad
    if keep.sum() >= 4:
        Z = Z[keep]
    nch = x.shape[0]
    off = ~np.eye(nch, dtype=bool)
    out, mats = [], []
    for a, b in COH_BANDS:
        zb = Z[:, :, (f >= a) & (f < b)]
        S = np.einsum("scf,sdf->fcd", zb, zb.conj()) / zb.shape[0]
        d = np.real(np.einsum("fcc->fc", S))
        coh = np.abs(S) ** 2 / (d[:, :, None] * d[:, None, :] + 1e-30)
        coh = coh.mean(axis=0)
        strength = (coh * off).sum(axis=1) / (nch - 1)
        out.append(np.arctanh(np.sqrt(np.clip(strength, 0.0, 0.999))))
        # band-limited correlation matrix (scale-invariant), shrunk towards identity for SPD
        Sb = np.real(S.sum(axis=0))
        db = np.sqrt(np.clip(np.diag(Sb), 1e-30, None))
        R = Sb / np.outer(db, db)
        R = (1 - SHRINK) * R + SHRINK * np.eye(nch)
        R = 0.5 * (R + R.T)
        mats.append(R.ravel())
    return np.concatenate(out), np.concatenate(mats)


def _zrows(a):
    a = a - a.mean(axis=-1, keepdims=True)
    sd = a.std(axis=-1, keepdims=True)
    return np.where(sd > 1e-12, a / (sd + 1e-12), 0.0)


def _envelope(x, fs):
    # amplitude-envelope correlation strength: slow co-fluctuation of band amplitude between
    # channels (log envelope averaged in 0.5-s windows; log makes it gain-invariant),
    # plus the full envelope-correlation matrix (shrunk, SPD) for the Riemannian branch,
    # plus orthogonalized envelope-correlation strength (zero-lag field spread removed)
    nch = x.shape[0]
    off = ~np.eye(nch, dtype=bool)
    x = x - x.mean(axis=1, keepdims=True)
    w = int(fs / 2)
    out, orth, mats = [], [], []
    for a, b in COH_BANDS:
        sos = butter(4, [a, b], btype="band", fs=fs, output="sos")
        an = hilbert(sosfiltfilt(sos, x, axis=-1), axis=-1)
        env = np.abs(an)
        nw = env.shape[1] // w
        T = nw * w
        e = np.log(env[:, :T].reshape(nch, nw, w).mean(axis=-1) + 1e-12)
        # drop windows whose mean log envelope is a robust outlier (artifacts)
        g = e.mean(axis=0)
        med = np.median(g)
        mad = np.median(np.abs(g - med)) + 1e-12
        keep = g <= med + 3 * 1.4826 * mad
        if keep.sum() < 20:
            keep = np.ones(nw, dtype=bool)
        e = e[:, keep]
        with np.errstate(invalid="ignore", divide="ignore"):
            R = np.nan_to_num(np.corrcoef(e))
        np.fill_diagonal(R, 1.0)
        z = np.arctanh(np.clip(R, -0.999, 0.999))
        out.append((z * off).sum(axis=1) / (nch - 1))
        Rs = (1 - SHRINK) * R + SHRINK * np.eye(nch)
        Rs = 0.5 * (Rs + Rs.T)
        mats.append(Rs.ravel())
        # orthogonalized AEC: amplitude of each channel's component orthogonal to the seed's
        # phase (removes instantaneous leakage / shared reference), correlated with seed envelope
        anw = an[:, :T]
        ez = _zrows(e)
        O = np.zeros((nch, nch))
        for i in range(nch):
            unit = np.conj(anw[i]) / (env[i, :T] + 1e-12)
            proj = np.abs(np.imag(anw * unit[None, :]))
            eo = np.log(proj.reshape(nch, nw, w).mean(axis=-1) + 1e-12)[:, keep]
            O[i] = (_zrows(eo) * ez[i][None, :]).mean(axis=1)
        O = 0.5 * (O + O.T)
        zo = np.arctanh(np.clip(O, -0.999, 0.999))
        orth.append((zo * off).sum(axis=1) / (nch - 1))
    return np.concatenate(out + orth), np.concatenate(mats)


def featurize(items, ctx):
    base_feats, cov_feats = [], []
    fs = ctx["fs"]
    blocks = {c: ctx["load_eeg"](items, c) for c in ctx["conditions"]}
    for i in range(len(items)):
        f, cv = [], []
        for c in ctx["conditions"]:
            x = blocks[c][i]
            if x is None:                      # missing condition -> fall back to eyes closed
                x = blocks[ctx["conditions"][0]][i]
            x = np.asarray(x, dtype=np.float64)
            f.append(_spectral(x, fs))
            coh, mats = _coherence(x, fs)
            f.append(coh)
            env, emats = _envelope(x, fs)
            f.append(env)
            cv.append(mats)
            cv.append(emats)
        base_feats.append(np.concatenate(f))
        cov_feats.append(np.concatenate(cv))
    base = np.nan_to_num(np.asarray(base_feats, dtype=np.float64))
    covs = np.asarray(cov_feats, dtype=np.float64)
    return np.hstack([base, covs])


def _nblk(ctx):
    # per condition: coherence-correlation blocks + envelope-correlation blocks
    return len(ctx["conditions"]) * 2 * len(COH_BANDS)


def _tangent(F_train, F_test, ctx):
    nch = len(ctx["channels"])
    nblk = _nblk(ctx)
    Ctr = np.asarray(F_train, dtype=np.float64).reshape(len(F_train), nblk, nch, nch)
    Cte = np.asarray(F_test, dtype=np.float64).reshape(len(F_test), nblk, nch, nch)
    Ttr, Tte = [], []
    for k in range(nblk):
        ts = TangentSpace(metric="riemann").fit(Ctr[:, k])
        Ttr.append(ts.transform(Ctr[:, k]))
        Tte.append(ts.transform(Cte[:, k]))
    return np.nan_to_num(np.hstack(Ttr)), np.nan_to_num(np.hstack(Tte))


def fit_predict(F_train, y_train, F_test, ctx):
    F_train = np.asarray(F_train, dtype=np.float64)
    F_test = np.asarray(F_test, dtype=np.float64)
    nch = len(ctx["channels"])
    covdim = _nblk(ctx) * nch * nch
    Btr, Bte = F_train[:, :-covdim], F_test[:, :-covdim]
    sc = StandardScaler().fit(Btr)
    clf = LogisticRegression(C=0.1, max_iter=5000).fit(sc.transform(Btr), y_train)
    p_base = clf.predict_proba(sc.transform(Bte))[:, 1]
    # Riemannian branch: tangent space at the training-fold mean, per band/condition block
    Ttr, Tte = _tangent(F_train[:, -covdim:], F_test[:, -covdim:], ctx)
    sc2 = StandardScaler().fit(Ttr)
    clf2 = LogisticRegression(C=0.1, max_iter=5000).fit(sc2.transform(Ttr), y_train)
    p_cov = clf2.predict_proba(sc2.transform(Tte))[:, 1]
    return 0.5 * (p_base + p_cov)