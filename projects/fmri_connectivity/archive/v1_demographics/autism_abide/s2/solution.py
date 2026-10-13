"""CC200 (no GSR) -> Ledoit-Wolf covariance of standardized signals -> Riemannian tangent space
(reference = geometric mean of training fold) -> standardized L2 logistic regression.
Second base learner: per-ROI relative log-amplitude (ALFF proxy of band-passed signal, centred
within subject) -> L2 logistic regression.
Third base learner: CC400 (no GSR) Ledoit-Wolf covariance -> log-Euclidean tangent vector
(matrix log, computed label-free per subject) -> standardized L2 logistic regression.
Fourth base learner: CC200 per-ROI intrinsic-timescale proxy (lag-1/lag-2 autocorrelation,
Fisher-z, centred within subject) -> L2 logistic regression.
Fifth base learner: Harvard-Oxford (anatomical) Ledoit-Wolf covariance -> log-Euclidean tangent
vector -> standardized L2 logistic regression.
Sixth base learner: graph node strength (mean Fisher-z Pearson correlation of each ROI with all
others) on CC200 and CC400, centred within subject, plus the whole-brain mean connectivity
-> L2 logistic regression.
A small stacked logistic regression combines the inner-CV out-of-fold scores of the base
learners with covariates (log mean_fd, age, sex, site one-hot) and score x log-age interactions
(developmental modulation of atypical connectivity)."""
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from pyriemann.tangentspace import TangentSpace

_LAGS = (1, 2)
_HDR = 3


def _lw_cov_amp(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    valid = sd[0] >= 1e-8
    # relative log-amplitude (ALFF proxy); invalid ROIs set to subject median (=0 after centring)
    amp = np.zeros(x.shape[1])
    if valid.any():
        la = np.log(sd[0, valid])
        amp[valid] = la - np.median(la)
    sd[sd < 1e-8] = 1.0  # all-zero ROIs stay zero; LW shrinkage keeps matrix PD
    x = x / sd
    c = LedoitWolf(store_precision=False, assume_centered=True).fit(x).covariance_
    # guarantee strict positive definiteness
    c = c + 1e-6 * np.eye(c.shape[0])
    return c, amp


def _acf_feats(x):
    """Per-ROI lag-k autocorrelation (intrinsic timescale proxy), Fisher-z, centred within subject."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    den = (x * x).sum(axis=0)
    valid = den > 1e-12
    out = []
    for lag in _LAGS:
        z = np.zeros(x.shape[1])
        if valid.any() and x.shape[0] > lag + 1:
            num = (x[lag:] * x[:-lag]).sum(axis=0)
            r = num[valid] / den[valid]
            zz = np.arctanh(np.clip(r, -0.999, 0.999))
            z[valid] = zz - np.median(zz)
        out.append(z)
    return np.concatenate(out)


def _strength(x):
    """Per-ROI node strength (mean Fisher-z Pearson r with all other valid ROIs), centred within
    subject; also returns the whole-brain mean connectivity."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0)
    valid = sd >= 1e-8
    out = np.zeros(x.shape[1])
    nv = int(valid.sum())
    if nv < 3:
        return out, 0.0
    z = x[:, valid] / sd[valid]
    r = (z.T @ z) / z.shape[0]
    zr = np.arctanh(np.clip(r, -0.999, 0.999))
    np.fill_diagonal(zr, 0.0)
    s = zr.sum(axis=1) / (nv - 1)
    g = float(s.mean())
    out[valid] = s - g
    return out, g


def _logm_vec(c):
    """Log-Euclidean tangent vector: upper triangle of logm(C), off-diagonals scaled by sqrt(2)."""
    c = 0.5 * (c + c.T)
    w, v = np.linalg.eigh(c)
    w = np.log(np.clip(w, 1e-10, None))
    lm = (v * w) @ v.T
    r = lm.shape[0]
    iu = np.triu_indices(r)
    coef = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    return lm[iu] * coef


def featurize(items, ctx):
    strategy = "filt_noglobal" if "filt_noglobal" in ctx["strategies"] else ctx["strategies"][0]
    ts200 = ctx["load_timeseries"](items, "cc200", strategy)
    n = len(items)
    first_c, _ = _lw_cov_amp(ts200[0])
    r1 = first_c.shape[0]
    na = len(_LAGS) * r1
    ts400 = ctx["load_timeseries"](items, "cc400", strategy)
    r2 = np.asarray(ts400[0]).shape[1]
    d2 = r2 * (r2 + 1) // 2
    tsho = ctx["load_timeseries"](items, "ho", strategy)
    r3 = np.asarray(tsho[0]).shape[1]
    d3 = r3 * (r3 + 1) // 2
    ns = r1 + r2 + 2
    # layout per row: [R1, R2, R3, cov200 (R1*R1), amp200 (R1), acf200 (L*R1),
    #                  logm400 triu (d2), logmHO triu (d3), strength200 (R1), strength400 (R2),
    #                  global200, global400]
    out = np.empty((n, _HDR + r1 * r1 + r1 + na + d2 + d3 + ns), dtype=np.float32)
    out[:, 0] = r1
    out[:, 1] = r2
    out[:, 2] = r3
    for i in range(n):
        c, a = _lw_cov_amp(ts200[i])
        o = _HDR
        out[i, o:o + r1 * r1] = c.ravel()
        o += r1 * r1
        out[i, o:o + r1] = a
        o += r1
        out[i, o:o + na] = _acf_feats(ts200[i])
        o += na
        c4, _ = _lw_cov_amp(ts400[i])
        out[i, o:o + d2] = _logm_vec(c4)
        o += d2
        ch, _ = _lw_cov_amp(tsho[i])
        out[i, o:o + d3] = _logm_vec(ch)
        o += d3
        s2, g2 = _strength(ts200[i])
        s4, g4 = _strength(ts400[i])
        out[i, o:o + r1] = s2
        o += r1
        out[i, o:o + r2] = s4
        o += r2
        out[i, o] = g2
        out[i, o + 1] = g4
    del ts200, ts400, tsho
    return out


def _to_df(meta, n):
    if meta is None:
        return pd.DataFrame(index=range(n))
    if isinstance(meta, pd.DataFrame):
        return meta.reset_index(drop=True)
    return pd.DataFrame(meta).reset_index(drop=True)


def _sex_numeric(s):
    s = pd.Series(s)
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().mean() > 0.5:
        return num.astype(float)
    m = s.astype(str).str.strip().str.upper().map(
        {"M": 1.0, "MALE": 1.0, "1": 1.0, "F": 2.0, "FEMALE": 2.0, "2": 2.0})
    return m.astype(float)


def _covariates(mtr, mte):
    """Build covariate matrices (fit imputation / site levels on train only)."""
    cols_tr, cols_te = [], []
    for name in ("mean_fd", "age", "sex"):
        if name in mtr.columns and name in mte.columns:
            if name == "sex":
                a, b = _sex_numeric(mtr[name]), _sex_numeric(mte[name])
            else:
                a = pd.to_numeric(mtr[name], errors="coerce").astype(float)
                b = pd.to_numeric(mte[name], errors="coerce").astype(float)
            med = np.nanmedian(a.values) if np.isfinite(a.values).any() else 0.0
            a = a.fillna(med).values
            b = b.fillna(med).values
            if name == "mean_fd":
                a = np.log(np.clip(a, 1e-4, None))
                b = np.log(np.clip(b, 1e-4, None))
            cols_tr.append(a[:, None])
            cols_te.append(b[:, None])
    if "site" in mtr.columns and "site" in mte.columns:
        sites = sorted(pd.Series(mtr["site"]).astype(str).unique())
        str_tr = mtr["site"].astype(str).values
        str_te = mte["site"].astype(str).values
        cols_tr.append(np.stack([(str_tr == s).astype(float) for s in sites], axis=1))
        cols_te.append(np.stack([(str_te == s).astype(float) for s in sites], axis=1))
    if not cols_tr:
        return None, None
    return np.hstack(cols_tr), np.hstack(cols_te)


def _log_age_z(mtr, mte):
    """Standardized log-age (statistics from train only); None if unavailable."""
    if "age" not in mtr.columns or "age" not in mte.columns:
        return None, None
    a = pd.to_numeric(mtr["age"], errors="coerce").astype(float)
    b = pd.to_numeric(mte["age"], errors="coerce").astype(float)
    if not np.isfinite(a.values).any():
        return None, None
    med = np.nanmedian(a.values)
    a = np.log(np.clip(a.fillna(med).values, 1.0, None))
    b = np.log(np.clip(b.fillna(med).values, 1.0, None))
    mu, sd = a.mean(), a.std()
    if sd < 1e-8:
        return None, None
    return (a - mu) / sd, (b - mu) / sd


def fit_predict(F_train, y_train, F_test, ctx):
    y_train = np.asarray(y_train).astype(int)
    Ftr = np.asarray(F_train, dtype=np.float32)
    Fte = np.asarray(F_test, dtype=np.float32)
    r1 = int(round(float(Ftr[0, 0])))
    r2 = int(round(float(Ftr[0, 1])))
    r3 = int(round(float(Ftr[0, 2])))
    d2 = r2 * (r2 + 1) // 2
    d3 = r3 * (r3 + 1) // 2
    na = len(_LAGS) * r1
    o = _HDR
    Ctr = Ftr[:, o:o + r1 * r1].astype(np.float64).reshape(-1, r1, r1)
    Cte = Fte[:, o:o + r1 * r1].astype(np.float64).reshape(-1, r1, r1)
    o += r1 * r1
    Atr = Ftr[:, o:o + r1].astype(np.float64)
    Ate = Fte[:, o:o + r1].astype(np.float64)
    o += r1
    Ttr = Ftr[:, o:o + na].astype(np.float64)
    Tte = Fte[:, o:o + na].astype(np.float64)
    o += na
    Ltr = np.ascontiguousarray(Ftr[:, o:o + d2])
    Lte = np.ascontiguousarray(Fte[:, o:o + d2])
    o += d2
    Htr = Ftr[:, o:o + d3].astype(np.float64)
    Hte = Fte[:, o:o + d3].astype(np.float64)
    o += d3
    Gtr = Ftr[:, o:].astype(np.float64)
    Gte = Fte[:, o:].astype(np.float64)
    del Ftr, Fte

    # symmetrize after float32 round trip
    Ctr = 0.5 * (Ctr + np.transpose(Ctr, (0, 2, 1)))
    Cte = 0.5 * (Cte + np.transpose(Cte, (0, 2, 1)))
    ts = TangentSpace(metric="riemann").fit(Ctr)
    Xtr = ts.transform(Ctr)
    Xte = ts.transform(Cte)
    del Ctr, Cte
    sc = StandardScaler().fit(Xtr)
    Xtr = sc.transform(Xtr)
    Xte = sc.transform(Xte)

    sca = StandardScaler().fit(Atr)
    Atr = sca.transform(Atr)
    Ate = sca.transform(Ate)

    sct = StandardScaler().fit(Ttr)
    Ttr = sct.transform(Ttr)
    Tte = sct.transform(Tte)

    scl = StandardScaler().fit(Ltr)
    Ltr = scl.transform(Ltr).astype(np.float32)
    Lte = scl.transform(Lte).astype(np.float32)

    sch = StandardScaler().fit(Htr)
    Htr = sch.transform(Htr)
    Hte = sch.transform(Hte)

    scg = StandardScaler().fit(Gtr)
    Gtr = scg.transform(Gtr)
    Gte = scg.transform(Gte)

    def _clf():
        return LogisticRegression(C=0.001, max_iter=5000)

    def _clf_amp():
        return LogisticRegression(C=0.01, max_iter=5000)

    def _clf_acf():
        # ~2x as many features as the ALFF view -> slightly stronger L2
        return LogisticRegression(C=0.005, max_iter=5000)

    def _clf_le():
        # ~4x more features than CC200 tangent space -> proportionally stronger L2
        return LogisticRegression(C=0.0003, max_iter=5000)

    def _clf_ho():
        # ~1/3 the features of CC200 tangent space -> proportionally weaker L2
        return LogisticRegression(C=0.003, max_iter=5000)

    def _clf_str():
        # ~3x as many features as the ALFF view -> slightly stronger L2
        return LogisticRegression(C=0.003, max_iter=5000)

    learners = [(Xtr, Xte, _clf), (Atr, Ate, _clf_amp), (Ltr, Lte, _clf_le),
                (Ttr, Tte, _clf_acf), (Htr, Hte, _clf_ho), (Gtr, Gte, _clf_str)]

    te_scores = [mk().fit(A, y_train).decision_function(B) for A, B, mk in learners]

    mtr = _to_df(ctx.get("meta_train"), len(y_train))
    mte = _to_df(ctx.get("meta_test"), Xte.shape[0])
    Ztr, Zte = _covariates(mtr, mte)
    age_tr, age_te = _log_age_z(mtr, mte)

    # out-of-fold base-learner scores on the training fold (grouped by subject)
    groups = ctx.get("groups_train")
    if groups is None:
        groups = np.arange(len(y_train))
    groups = np.asarray(groups)
    n_splits = min(5, len(np.unique(groups)))
    oofs = [np.zeros(len(y_train)) for _ in learners]
    for itr, iva in GroupKFold(n_splits=n_splits).split(Xtr, y_train, groups):
        if len(np.unique(y_train[itr])) < 2:
            continue
        for k, (A, _, mk) in enumerate(learners):
            oofs[k][iva] = mk().fit(A[itr], y_train[itr]).decision_function(A[iva])

    parts_tr = [v[:, None] for v in oofs]
    parts_te = [v[:, None] for v in te_scores]
    if Ztr is not None:
        parts_tr.append(Ztr)
        parts_te.append(Zte)
    if age_tr is not None:
        # developmental modulation: let each view's weight vary linearly with log-age
        O_tr = np.column_stack(oofs)
        O_te = np.column_stack(te_scores)
        so = StandardScaler().fit(O_tr)
        parts_tr.append(so.transform(O_tr) * age_tr[:, None])
        parts_te.append(so.transform(O_te) * age_te[:, None])
    S_tr = np.hstack(parts_tr)
    S_te = np.hstack(parts_te)
    sc2 = StandardScaler().fit(S_tr)
    stack = LogisticRegression(C=1.0, max_iter=5000)
    stack.fit(sc2.transform(S_tr), y_train)
    return stack.predict_proba(sc2.transform(S_te))[:, 1]