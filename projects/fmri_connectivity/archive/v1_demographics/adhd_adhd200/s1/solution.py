"""Multi-atlas ensemble (CC200 + AAL + HO): tangent-space embedding of Ledoit-Wolf covariances per atlas
(log-Euclidean reference, label-free), plus a regional time-series block (log SD amplitude / ALFF proxy and
lag-1/lag-2 autocorrelation as low-frequency-dominance proxies, pooled across atlases), plus a node-strength
block (per-ROI global connectivity: mean signed and mean absolute Fisher-z correlation, pooled across atlases),
plus a dynamic-connectivity block (per-ROI sliding-window temporal variability of the connectivity profile and
SD of windowed node strength, pooled across atlases), plus a network-summary block (within/between-module mean
Fisher-z connectivity for label-free data-driven modules at two scales, k=7 and k=17, pooled across atlases),
plus a whole-brain block (global-signal amplitude/coupling/autocorrelation, DVARS residual-motion proxies and
effective dimensionality of the correlation spectrum, pooled across atlases), plus a scale-free dynamics block
(per-ROI short- and long-scale detrended fluctuation analysis exponents, pooled across atlases).
Per-site feature standardization (label-free, transductive) plus lightly-penalized site one-hot, sex one-hot
and age covariate columns, moderately-penalized site x sex interaction columns (site-specific recruitment),
L2 logistic regression per block, decision scores averaged across blocks."""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

_ATLASES = ["cc200", "aal", "ho"]
_MAX_BLOCKS = 9
_WIN = 30
_STEP = 3
_MODULE_K = (7, 17)
_N_GLOBAL = 12
_COV_SCALE = 30.0
_INTER_SCALE = 5.0
_INTER_MIN = 5
_DFA_SCALES = (4, 5, 6, 8, 11, 16, 22, 32, 45, 64)
_DFA_SPLIT = 8


def _spd_fn(C, f):
    w, V = np.linalg.eigh(C)
    w = np.maximum(w, 1e-10)
    return (V * f(w)) @ V.T


def _tangent_feats(ts):
    covs = []
    for x in ts:
        x = np.nan_to_num(np.asarray(x, dtype=np.float64))
        x = x - x.mean(axis=0)
        sd = x.std(axis=0)
        good = sd > 1e-8
        x[:, good] = x[:, good] / sd[good]
        x[:, ~good] = 0.0
        c = LedoitWolf(assume_centered=True).fit(x).covariance_
        # guarantee strictly positive definite (all-zero ROIs)
        c = c + 1e-3 * np.eye(c.shape[0])
        covs.append(c)
    # label-free log-Euclidean mean as tangent-space reference
    logs = np.mean([_spd_fn(c, np.log) for c in covs], axis=0)
    ref = _spd_fn(logs, np.exp)
    isqrt = _spd_fn(ref, lambda w: 1.0 / np.sqrt(w))
    iu = np.triu_indices(ref.shape[0], k=0)
    feats = []
    for c in covs:
        t = _spd_fn(isqrt @ c @ isqrt, np.log)
        feats.append(t[iu])
    return np.asarray(feats, dtype=np.float32)


def _impute_median(F):
    F = np.asarray(F, dtype=np.float64)
    med = np.nanmedian(F, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    bad = ~np.isfinite(F)
    F[bad] = np.take(med, np.where(bad)[1])
    return F.astype(np.float32)


def _ts_feats(ts):
    """Per-ROI log SD (amplitude, ALFF proxy) and lag-1/lag-2 autocorrelation (spectral slope proxy)."""
    rows = []
    for x in ts:
        x = np.nan_to_num(np.asarray(x, dtype=np.float64))
        x = x - x.mean(axis=0)
        sd = x.std(axis=0)
        good = sd > 1e-8
        n_roi = x.shape[1]
        logsd = np.full(n_roi, np.nan)
        ac1 = np.full(n_roi, np.nan)
        ac2 = np.full(n_roi, np.nan)
        if good.any():
            z = x[:, good] / sd[good]
            logsd[good] = np.log(sd[good])
            if z.shape[0] > 3:
                ac1[good] = np.mean(z[1:] * z[:-1], axis=0)
                ac2[good] = np.mean(z[2:] * z[:-2], axis=0)
        rows.append(np.concatenate([logsd, ac1, ac2]))
    # impute out-of-FOV ROIs with the per-feature median (label-free)
    return _impute_median(rows)


def _dfa_feats(ts):
    """Per-ROI detrended fluctuation analysis: scaling exponent of the RMS fluctuation of the integrated,
    linearly detrended signal over short (4-8 TR) and long (8 TR .. T/4) window sizes (Hurst-like
    long-range temporal correlation)."""
    rows = []
    for x in ts:
        x = np.nan_to_num(np.asarray(x, dtype=np.float64))
        x = x - x.mean(axis=0)
        sd = x.std(axis=0)
        good = sd > 1e-8
        n_t, n_roi = x.shape
        a_s = np.full(n_roi, np.nan)
        a_l = np.full(n_roi, np.nan)
        scales = [n for n in _DFA_SCALES if n <= n_t // 4]
        if good.any() and len(scales) >= 4:
            z = x[:, good] / sd[good]
            y = np.cumsum(z, axis=0)
            logF = []
            for n in scales:
                ns = n_t // n
                Y = y[:ns * n].reshape(ns, n, -1)
                tc = np.arange(n, dtype=np.float64) - (n - 1) / 2.0
                Yc = Y - Y.mean(axis=1, keepdims=True)
                b = np.einsum("t,stk->sk", tc, Yc) / np.sum(tc ** 2)
                res = Yc - tc[None, :, None] * b[:, None, :]
                F = np.sqrt(np.mean(res ** 2, axis=(0, 1)))
                logF.append(np.log(F + 1e-12))
            L = np.asarray(logF)  # (S, ng)
            ln = np.log(np.asarray(scales, dtype=np.float64))

            def _slope(mask):
                lx = ln[mask] - ln[mask].mean()
                Ly = L[mask] - L[mask].mean(axis=0)
                return (lx @ Ly) / np.sum(lx ** 2)

            short = np.asarray(scales) <= _DFA_SPLIT
            long_ = np.asarray(scales) >= _DFA_SPLIT
            if short.sum() >= 2:
                a_s[good] = _slope(short)
            if long_.sum() >= 2:
                a_l[good] = _slope(long_)
        rows.append(np.concatenate([a_s, a_l]))
    return _impute_median(rows)


def _strength_feats(ts):
    """Per-ROI global connectivity: mean signed and mean absolute Fisher-z correlation to all other ROIs."""
    rows = []
    for x in ts:
        x = np.nan_to_num(np.asarray(x, dtype=np.float64))
        x = x - x.mean(axis=0)
        sd = x.std(axis=0)
        good = sd > 1e-8
        n_roi = x.shape[1]
        s_signed = np.full(n_roi, np.nan)
        s_abs = np.full(n_roi, np.nan)
        ng = int(good.sum())
        if ng > 2:
            z = x[:, good] / sd[good]
            r = (z.T @ z) / z.shape[0]
            r = np.clip(r, -0.999, 0.999)
            fz = np.arctanh(r)
            np.fill_diagonal(fz, 0.0)
            s_signed[good] = fz.sum(axis=1) / (ng - 1)
            s_abs[good] = np.abs(fz).sum(axis=1) / (ng - 1)
        rows.append(np.concatenate([s_signed, s_abs]))
    return _impute_median(rows)


def _dynamic_feats(ts):
    """Per-ROI sliding-window dynamics: temporal variability of the connectivity profile
    (1 - mean between-window correlation of the ROI's FC profile) and SD of windowed node strength."""
    rows = []
    for x in ts:
        x = np.nan_to_num(np.asarray(x, dtype=np.float64))
        n_t, n_roi = x.shape
        sd = (x - x.mean(axis=0)).std(axis=0)
        good = sd > 1e-8
        ng = int(good.sum())
        tvar = np.full(n_roi, np.nan)
        sdstr = np.full(n_roi, np.nan)
        starts = list(range(0, n_t - _WIN + 1, _STEP))
        if ng > 3 and len(starts) >= 4:
            xg = x[:, good]
            prof = []
            for s in starts:
                w = xg[s:s + _WIN]
                w = w - w.mean(axis=0)
                wsd = w.std(axis=0)
                wsd[wsd < 1e-8] = 1.0
                w = w / wsd
                r = np.clip((w.T @ w) / _WIN, -0.999, 0.999)
                fz = np.arctanh(r)
                np.fill_diagonal(fz, 0.0)
                prof.append(fz)
            P = np.asarray(prof)  # (W, ng, ng)
            nw = P.shape[0]
            strength = P.sum(axis=2) / (ng - 1)  # (W, ng)
            sdstr[good] = strength.std(axis=0)
            # correlation of each ROI's profile between windows (diagonal zero entries included, harmless)
            Pc = P - P.mean(axis=2, keepdims=True)
            nrm = np.sqrt((Pc ** 2).sum(axis=2, keepdims=True))
            nrm[nrm < 1e-10] = 1.0
            Pc = Pc / nrm
            G = np.einsum("wik,vik->iwv", Pc, Pc)  # (ng, W, W)
            off = (G.sum(axis=(1, 2)) - np.trace(G, axis1=1, axis2=2)) / (nw * (nw - 1))
            tvar[good] = 1.0 - off
        rows.append(np.concatenate([tvar, sdstr]))
    return _impute_median(rows)


def _fz_full(x):
    """Full ROI x ROI Fisher-z correlation with NaN for out-of-FOV ROIs and on the diagonal."""
    x = np.nan_to_num(np.asarray(x, dtype=np.float64))
    x = x - x.mean(axis=0)
    sd = x.std(axis=0)
    good = sd > 1e-8
    n_roi = x.shape[1]
    fz = np.full((n_roi, n_roi), np.nan)
    if good.sum() > 2:
        z = x[:, good] / sd[good]
        r = np.clip((z.T @ z) / z.shape[0], -0.999, 0.999)
        fz[np.ix_(good, good)] = np.arctanh(r)
    np.fill_diagonal(fz, np.nan)
    return fz


def _module_feats(ts):
    """Network summaries: label-free modules from KMeans on group-mean FC profiles (two scales),
    then per-subject mean Fisher-z connectivity within and between each pair of modules."""
    n_roi = np.asarray(ts[0]).shape[1]
    S = np.zeros((n_roi, n_roi))
    Cnt = np.zeros((n_roi, n_roi))
    for x in ts:
        fz = _fz_full(x)
        v = np.isfinite(fz)
        S += np.where(v, fz, 0.0)
        Cnt += v
    G = S / np.maximum(Cnt, 1.0)
    np.fill_diagonal(G, 0.0)

    mats = []
    for k in _MODULE_K:
        k = int(min(k, max(2, n_roi // 3)))
        lab = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(G)
        M = np.zeros((n_roi, k))
        M[np.arange(n_roi), lab] = 1.0
        mats.append(M)

    rows = []
    for x in ts:
        fz = _fz_full(x)
        v = np.isfinite(fz).astype(np.float64)
        Z = np.where(v > 0, fz, 0.0)
        parts = []
        for M in mats:
            sums = M.T @ Z @ M
            cnts = M.T @ v @ M
            with np.errstate(invalid="ignore", divide="ignore"):
                means = np.where(cnts > 0, sums / np.maximum(cnts, 1.0), np.nan)
            iu = np.triu_indices(M.shape[1], k=0)
            parts.append(means[iu])
        rows.append(np.concatenate(parts))
    return _impute_median(rows)


def _global_feats(ts):
    """Whole-brain descriptors: global-signal amplitude, synchrony, ROI coupling, autocorrelation and
    kurtosis (vigilance/arousal markers); DVARS level, spikiness and spike fraction (residual-motion proxies);
    and leading-eigenvalue fraction, participation ratio and spectral entropy of the correlation matrix."""
    rows = []
    for x in ts:
        x = np.nan_to_num(np.asarray(x, dtype=np.float64))
        x = x - x.mean(axis=0)
        sd = x.std(axis=0)
        good = sd > 1e-8
        ng = int(good.sum())
        f = np.full(_N_GLOBAL, np.nan)
        n_t = x.shape[0]
        if ng > 3 and n_t > 5:
            xg = x[:, good]
            sg = sd[good]
            z = xg / sg
            gs_raw = xg.mean(axis=1)
            gsz = z.mean(axis=1)
            f[0] = np.log(gs_raw.std() + 1e-12)
            f[1] = np.log(np.mean(sg))
            f[2] = gsz.var()  # ~ mean pairwise correlation (global synchrony)
            gsd = gsz.std()
            if gsd > 1e-8:
                g = gsz / gsd
                r = (z.T @ g) / n_t
                f[3] = np.mean(r ** 2)
                f[4] = np.mean(g[1:] * g[:-1])
                f[5] = np.mean(g ** 4) - 3.0
            d = np.diff(z, axis=0)
            dv = np.sqrt(np.mean(d ** 2, axis=1))
            med = np.median(dv)
            if med > 1e-8:
                f[6] = np.log(np.mean(dv))
                f[7] = np.log(np.percentile(dv, 95) / med)
                f[8] = np.mean(dv > 1.5 * med)
            ev = np.maximum(np.linalg.eigvalsh((z.T @ z) / n_t), 0.0)
            s = ev.sum()
            if s > 1e-8:
                p = ev / s
                f[9] = ev.max() / s
                f[10] = (s ** 2 / np.sum(ev ** 2)) / min(ng, n_t)
                pp = p[p > 1e-12]
                f[11] = -np.sum(pp * np.log(pp)) / np.log(min(ng, n_t))
        rows.append(f)
    return _impute_median(rows)


def featurize(items, ctx):
    strategy = "filt_noglobal" if "filt_noglobal" in ctx["strategies"] else ctx["strategies"][0]
    atlases = [a for a in _ATLASES if a in ctx["atlases"]]
    if not atlases:
        atlases = [ctx["atlases"][0]]
    blocks = []
    ts_blocks = []
    st_blocks = []
    dyn_blocks = []
    mod_blocks = []
    glob_blocks = []
    dfa_blocks = []
    for a in atlases:
        ts = ctx["load_timeseries"](items, a, strategy)
        blocks.append(_tangent_feats(ts))
        ts_blocks.append(_ts_feats(ts))
        st_blocks.append(_strength_feats(ts))
        dyn_blocks.append(_dynamic_feats(ts))
        mod_blocks.append(_module_feats(ts))
        glob_blocks.append(_global_feats(ts))
        dfa_blocks.append(_dfa_feats(ts))
        del ts
    blocks.append(np.hstack(ts_blocks))
    blocks.append(np.hstack(st_blocks))
    blocks.append(np.hstack(dyn_blocks))
    blocks.append(np.hstack(mod_blocks))
    blocks.append(np.hstack(glob_blocks))
    blocks.append(np.hstack(dfa_blocks))
    n = len(items)
    # header: block sizes (constant per item) so fit_predict can split the blocks
    header = np.zeros((n, _MAX_BLOCKS), dtype=np.float32)
    for j, b in enumerate(blocks[:_MAX_BLOCKS]):
        header[:, j] = b.shape[1]
    return np.hstack([header] + blocks[:_MAX_BLOCKS]).astype(np.float32)


def _meta_col(meta, key, n):
    if meta is None:
        return None
    try:
        s = meta[key]
    except Exception:
        try:
            s = [m[key] for m in meta]
        except Exception:
            return None
    out = list(s)
    if len(out) != n:
        return None
    return out


def _sites(meta, n):
    s = _meta_col(meta, "site", n)
    if s is None:
        return None
    return np.array([str(v) for v in s], dtype=object)


def _sex(meta, n):
    s = _meta_col(meta, "sex", n)
    if s is None:
        return None
    out = []
    for v in s:
        try:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                out.append("nan")
                continue
        except Exception:
            pass
        t = str(v).strip().lower()
        if t in ("m", "male", "1", "1.0"):
            t = "m"
        elif t in ("f", "female", "0", "0.0", "2", "2.0"):
            t = "f"
        elif t in ("", "none", "nan", "<na>", "-999", "-999.0"):
            t = "nan"
        out.append(t)
    return np.array(out, dtype=object)


def _age(meta, n):
    s = _meta_col(meta, "age", n)
    if s is None:
        return None
    out = np.empty(n, dtype=np.float64)
    for i, v in enumerate(s):
        try:
            out[i] = float(v)
        except Exception:
            out[i] = np.nan
    out[(out < 0) | (out > 100)] = np.nan
    return out


def _site_standardize(X, sites):
    X = X.copy()
    for s in np.unique(sites):
        m = sites == s
        mu = X[m].mean(axis=0)
        sd = X[m].std(axis=0)
        sd[sd < 1e-8] = 1.0
        X[m] = (X[m] - mu) / sd
    return X


def _covariates(ctx, n_tr, n_te):
    m_tr, m_te = ctx.get("meta_train"), ctx.get("meta_test")
    extra_tr, extra_te = [], []
    s_tr = _sites(m_tr, n_tr)
    s_te = _sites(m_te, n_te)
    train_sites = None
    if s_tr is not None and s_te is not None:
        train_sites = np.unique(s_tr)
        extra_tr.append((s_tr.reshape(-1, 1) == train_sites.reshape(1, -1)).astype(np.float64))
        extra_te.append((s_te.reshape(-1, 1) == train_sites.reshape(1, -1)).astype(np.float64))

    x_tr = _sex(m_tr, n_tr)
    x_te = _sex(m_te, n_te)
    if x_tr is not None and x_te is not None:
        cats = np.array([c for c in np.unique(x_tr) if c != "nan"], dtype=object)
        if len(cats) >= 2:
            # drop one level to avoid collinearity with site one-hots
            cats = cats[1:]
            extra_tr.append((x_tr.reshape(-1, 1) == cats.reshape(1, -1)).astype(np.float64))
            extra_te.append((x_te.reshape(-1, 1) == cats.reshape(1, -1)).astype(np.float64))
            # site x sex interactions (site-specific recruitment of boys/girls per diagnostic group),
            # given a smaller scale so they are shrunk toward the main effects
            if train_sites is not None:
                it_tr, it_te = [], []
                for s in train_sites:
                    for c in cats:
                        col_tr = ((s_tr == s) & (x_tr == c)).astype(np.float64)
                        if col_tr.sum() < _INTER_MIN or col_tr.sum() > len(col_tr) - _INTER_MIN:
                            continue
                        it_tr.append(col_tr)
                        it_te.append(((s_te == s) & (x_te == c)).astype(np.float64))
                if it_tr:
                    r = _INTER_SCALE / _COV_SCALE
                    extra_tr.append(np.column_stack(it_tr) * r)
                    extra_te.append(np.column_stack(it_te) * r)

    a_tr = _age(m_tr, n_tr)
    a_te = _age(m_te, n_te)
    if a_tr is not None and a_te is not None and np.isfinite(a_tr).sum() > 10:
        med = np.nanmedian(a_tr)
        a_tr = np.where(np.isfinite(a_tr), a_tr, med)
        a_te = np.where(np.isfinite(a_te), a_te, med)
        mu, sd = a_tr.mean(), a_tr.std()
        sd = sd if sd > 1e-8 else 1.0
        z_tr = (a_tr - mu) / sd
        z_te = (a_te - mu) / sd
        extra_tr.append(np.column_stack([z_tr, z_tr ** 2 - 1.0]))
        extra_te.append(np.column_stack([z_te, z_te ** 2 - 1.0]))
    return s_tr, s_te, extra_tr, extra_te


def _fit_block(X_tr, X_te, y_train, s_tr, s_te, extra_tr, extra_te):
    if s_tr is not None and s_te is not None:
        n_tr = len(X_tr)
        X_all = np.vstack([X_tr, X_te])
        s_all = np.concatenate([s_tr, s_te])
        X_all = _site_standardize(X_all, s_all)
        X_tr, X_te = X_all[:n_tr], X_all[n_tr:]

    sc = StandardScaler().fit(X_tr)
    X_tr = np.clip(sc.transform(X_tr), -6.0, 6.0)
    X_te = np.clip(sc.transform(X_te), -6.0, 6.0)

    scale = _COV_SCALE
    if extra_tr:
        X_tr = np.hstack([X_tr] + [e * scale for e in extra_tr])
        X_te = np.hstack([X_te] + [e * scale for e in extra_te])

    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(X_tr, y_train)
    return clf.decision_function(X_te)


def fit_predict(F_train, y_train, F_test, ctx):
    F_tr = np.asarray(F_train, dtype=np.float32)
    F_te = np.asarray(F_test, dtype=np.float32)
    y_train = np.asarray(y_train)
    n_tr, n_te = len(F_tr), len(F_te)

    sizes = [int(round(v)) for v in F_tr[0, :_MAX_BLOCKS] if v > 0]
    s_tr, s_te, extra_tr, extra_te = _covariates(ctx, n_tr, n_te)

    scores = []
    start = _MAX_BLOCKS
    for sz in sizes:
        X_tr = F_tr[:, start:start + sz].astype(np.float64)
        X_te = F_te[:, start:start + sz].astype(np.float64)
        start += sz
        scores.append(_fit_block(X_tr, X_te, y_train, s_tr, s_te, extra_tr, extra_te))
    score = np.mean(scores, axis=0)
    return 1.0 / (1.0 + np.exp(-score))