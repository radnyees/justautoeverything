"""Log relative band power (delta/theta/alpha/beta/gamma) per channel, the aperiodic (1/f)
exponent per channel, per-channel band-wise coherence node strength (mean Fisher-z
magnitude-squared coherence to all other channels), per-channel theta/alpha amplitude-envelope
correlation node strength, per-channel DFA exponent of the theta/alpha amplitude envelopes
(long-range temporal correlations), per-channel Higuchi fractal dimension and log Hjorth
mobility/complexity (medians over 4-s band-passed epochs), from each available condition,
standardized, L2 logistic regression."""
import numpy as np
from scipy.signal import welch, butter, sosfiltfilt, hilbert
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
AEC_BANDS = [(4, 8), (8, 13)]


def _features(x, fs):
    # median across 2-s segments: robust to transient artifacts (blinks, movement, electrode pops)
    f, p = welch(x, fs=fs, nperseg=int(2 * fs), axis=-1, average="median")
    tot = p[:, (f >= 1) & (f <= 45)].sum(axis=1, keepdims=True) + 1e-12
    rel = np.log(np.stack([p[:, (f >= a) & (f < b)].sum(axis=1) for a, b in BANDS], axis=1) / tot + 1e-12)
    # aperiodic exponent: log-log linear fit over 2-40 Hz, excluding the alpha peak (7-14 Hz)
    m = (f >= 2) & (f <= 40) & ~((f >= 7) & (f <= 14))
    lf = np.log10(f[m])
    A = np.stack([lf, np.ones_like(lf)], axis=1)
    lp = np.log10(p[:, m] + 1e-20).T                     # (freqs, channels)
    coef, *_ = np.linalg.lstsq(A, lp, rcond=None)
    slope = coef[0][:, None]                             # (channels, 1), scale-invariant
    S = _epochs(x, fs)
    envs = _envelopes(x, fs)
    return np.concatenate([rel, slope, _coherence(x, fs), _aec(envs, S), _dfa(envs, fs),
                           _hfd(S), _hjorth(S)], axis=1)


def _coherence(x, fs):
    # magnitude-squared coherence from 2-s Hann segments (50% overlap); segments with outlying
    # total power (artifacts) are dropped since complex cross-spectra cannot be median-averaged
    n = int(2 * fs)
    S = np.lib.stride_tricks.sliding_window_view(x, n, axis=-1)[:, :: n // 2, :]
    S = S - S.mean(axis=-1, keepdims=True)
    S = S * np.hanning(n)
    pw = (S ** 2).sum(axis=(0, 2))
    keep = pw <= 3 * np.median(pw)
    if keep.sum() < 5:
        keep = np.ones_like(keep)
    X = np.fft.rfft(S[:, keep], axis=-1)                 # (ch, seg, freq)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    nch = x.shape[0]
    out = []
    for a, b in BANDS:
        Xb = X[:, :, (f >= a) & (f < b)]
        C = np.einsum("isf,jsf->ijf", Xb, np.conj(Xb))   # cross-spectra summed over segments
        d = np.real(np.einsum("iif->if", C)) + 1e-20
        coh = np.abs(C) ** 2 / (d[:, None, :] * d[None, :, :])
        coh = coh.mean(axis=-1)
        z = np.arctanh(np.sqrt(np.clip(coh, 0, 0.999)))
        np.fill_diagonal(z, 0.0)
        out.append(z.sum(axis=1) / (nch - 1))            # node strength per channel
    return np.stack(out, axis=1)                         # (channels, bands)


def _envelopes(x, fs):
    # Hilbert amplitude envelopes of the theta and alpha band-passed signals
    envs = []
    for a, b in AEC_BANDS:
        sos = butter(4, [a, b], btype="band", fs=fs, output="sos")
        envs.append(np.abs(hilbert(sosfiltfilt(sos, x, axis=-1), axis=-1)))
    return envs


def _aec(envs, S):
    # amplitude-envelope correlation: Pearson correlation of log Hilbert envelopes (scale-invariant)
    # over 4-s epochs, dropping epochs with outlying broadband power (artifacts)
    nch, nseg, n = S.shape
    pw = (S ** 2).sum(axis=(0, 2))
    keep = pw <= 3 * np.median(pw)
    if keep.sum() < 3:
        keep = np.ones_like(keep)
    out = []
    for env in envs:
        E = env[:, : nseg * n].reshape(nch, nseg, n)[:, keep].reshape(nch, -1)
        R = np.corrcoef(np.log(E + 1e-12))
        z = np.arctanh(np.clip(np.nan_to_num(R), -0.999, 0.999))
        np.fill_diagonal(z, 0.0)
        out.append(z.sum(axis=1) / (nch - 1))            # node strength per channel
    return np.stack(out, axis=1)                         # (channels, bands)


def _dfa(envs, fs):
    # detrended fluctuation analysis exponent of each band envelope (long-range temporal
    # correlations, 1-20 s scales); scale-invariant since F(n) scales linearly with amplitude
    step = max(1, int(fs // 50))
    fe = fs / step
    out = []
    for env in envs:
        e = env[:, ::step]
        nch, T = e.shape
        Y = np.cumsum(e - e.mean(axis=1, keepdims=True), axis=1)
        hi = min(20 * fe, T / 4)
        sizes = np.unique(np.logspace(np.log10(fe), np.log10(max(hi, 2 * fe)), 10).astype(int))
        sizes = sizes[sizes * 2 <= T]
        if len(sizes) < 2:
            out.append(np.full(nch, 0.5))
            continue
        F = []
        for n in sizes:
            nw = T // n
            W = Y[:, : nw * n].reshape(nch, nw, n)
            t = np.arange(n) - (n - 1) / 2.0
            Wc = W - W.mean(axis=-1, keepdims=True)
            b = (Wc * t).sum(axis=-1, keepdims=True) / (t ** 2).sum()
            r = Wc - b * t
            F.append(np.sqrt((r ** 2).mean(axis=(1, 2))))
        lF = np.log(np.stack(F, axis=1) + 1e-20)         # (ch, sizes)
        ls = np.log(sizes.astype(float))
        lc = ls - ls.mean()
        out.append(((lF - lF.mean(axis=1, keepdims=True)) * lc).sum(axis=1) / (lc ** 2).sum())
    return np.stack(out, axis=1)                         # (channels, bands)


def _epochs(x, fs):
    # 1-45 Hz band-pass (removes drift and line noise), non-overlapping 4-s epochs
    hi = min(45.0, 0.45 * fs)
    sos = butter(4, [1.0, hi], btype="band", fs=fs, output="sos")
    xf = sosfiltfilt(sos, x, axis=-1)
    n = int(4 * fs)
    nseg = max(1, xf.shape[1] // n)
    if xf.shape[1] < n:
        n = xf.shape[1]
    return xf[:, : nseg * n].reshape(xf.shape[0], nseg, n)


def _hfd(S, kmax=10):
    # Higuchi fractal dimension per epoch; median over epochs is robust to artifact epochs
    n = S.shape[-1]
    lk = []
    for k in range(1, kmax + 1):
        Lm = []
        for m in range(k):
            sub = S[..., m::k]
            cnt = sub.shape[-1] - 1
            Lm.append(np.abs(np.diff(sub, axis=-1)).sum(axis=-1) * (n - 1) / (cnt * k) / k)
        lk.append(np.mean(Lm, axis=0))
    lk = np.log(np.stack(lk, axis=-1) + 1e-20)           # (ch, seg, k)
    xk = np.log(1.0 / np.arange(1, kmax + 1))
    xc = xk - xk.mean()
    d = (lk * xc).sum(axis=-1) / (xc ** 2).sum()         # slope = fractal dimension
    return np.median(d, axis=1)[:, None]                 # (channels, 1)


def _hjorth(S):
    # log Hjorth mobility (mean-frequency proxy) and complexity (bandwidth proxy) per epoch;
    # both are amplitude-scale invariant; median over epochs is robust to artifact epochs
    d1 = np.diff(S, axis=-1)
    d2 = np.diff(d1, axis=-1)
    v0 = S.var(axis=-1) + 1e-20
    v1 = d1.var(axis=-1) + 1e-20
    v2 = d2.var(axis=-1) + 1e-20
    mob = np.sqrt(v1 / v0)
    comp = np.sqrt(v2 / v1) / mob
    return np.stack([np.median(np.log(mob), axis=1), np.median(np.log(comp), axis=1)], axis=1)


def featurize(items, ctx):
    feats = []
    blocks = {c: ctx["load_eeg"](items, c) for c in ctx["conditions"]}
    for i in range(len(items)):
        f = []
        for c in ctx["conditions"]:
            x = blocks[c][i]
            if x is None:                      # missing condition -> fall back to eyes closed
                x = blocks[ctx["conditions"][0]][i]
            f.append(_features(np.asarray(x, dtype=np.float64), ctx["fs"]).ravel())
        feats.append(np.concatenate(f))
    return np.nan_to_num(np.asarray(feats, dtype=np.float32))


def fit_predict(F_train, y_train, F_test, ctx):
    sc = StandardScaler().fit(F_train)
    clf = LogisticRegression(C=0.1, max_iter=5000).fit(sc.transform(F_train), y_train)
    return clf.predict_proba(sc.transform(F_test))[:, 1]