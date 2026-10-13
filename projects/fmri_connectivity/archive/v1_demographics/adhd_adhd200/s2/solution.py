"""Ledoit-Wolf shrinkage covariance on CC200 (z-scored series) -> Riemannian re-centering per
site (each site's covariances whitened by that site's geometric mean, label-free) -> tangent space
at identity, upper triangle -> per-site feature-wise variance standardization (ComBat-style scale
correction, shrunk toward pooled within-site variance) -> per-edge split-half reliability weighting
(label-free test-retest prior: features scaled by sqrt of Spearman-Brown reliability) -> L2 logistic
regression with C chosen by grouped inner CV. Stacked with demographic covariates (sex, age, site)
via inner grouped CV out-of-fold scores."""
import numpy as np
import pandas as pd
from sklearn.covariance import ledoit_wolf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

MIN_SITE = 10
VAR_PRIOR = 10.0  # pseudo-count shrinking per-site feature variance toward pooled within-site variance
C_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
C_DEFAULT = 1e-3
MIN_HALF = 20  # minimum timepoints per half for split-half reliability
REL_FLOOR = 0.01


def _sym_fn(m, fn):
    w, v = np.linalg.eigh(m)
    w = np.maximum(w, 1e-10)
    return (v * fn(w)) @ v.T


def _geometric_mean(covs, n_iter=30, tol=1e-6):
    # Start from log-Euclidean mean, then Karcher iterations
    G = _sym_fn(np.mean([_sym_fn(c, np.log) for c in covs], axis=0), np.exp)
    for _ in range(n_iter):
        Gh = _sym_fn(G, np.sqrt)
        Gih = _sym_fn(G, lambda w: 1.0 / np.sqrt(w))
        T = np.mean([_sym_fn(Gih @ c @ Gih, np.log) for c in covs], axis=0)
        G = Gh @ _sym_fn(T, np.exp) @ Gh
        G = (G + G.T) / 2
        if np.linalg.norm(T) < tol:
            break
    return G


def _item_sites(items, ctx):
    """Site label per item from ctx['meta'] (label-free); None if unavailable."""
    meta = ctx.get("meta")
    if meta is None:
        return None
    try:
        df = meta if isinstance(meta, pd.DataFrame) else pd.DataFrame(meta)
    except Exception:
        return None
    if "site" not in df.columns:
        return None
    keys = [str(i) for i in items]
    idx = df.index.astype(str)
    if set(keys).issubset(set(idx)):
        s = df["site"].copy()
        s.index = idx
        s = s[~s.index.duplicated()]
        return s.loc[keys].astype(str).values
    for col in ("subject", "subject_id", "id", "item", "sub_id"):
        if col in df.columns:
            ids = df[col].astype(str).values
            if set(keys).issubset(set(ids)):
                s = pd.Series(df["site"].astype(str).values, index=ids)
                s = s[~s.index.duplicated()]
                return s.loc[keys].values
    if len(df) == len(items):
        return df["site"].astype(str).values
    return None


def _site_scale(F, sites):
    """Per-site feature-wise centering and variance standardization (label-free).
    Site variances are shrunk toward the pooled within-site variance; small sites use pooled."""
    F = F.copy()
    big = []
    for s in np.unique(sites):
        idx = np.where(sites == s)[0]
        if len(idx) >= MIN_SITE:
            big.append(idx)
    if not big:
        return F
    resid = np.concatenate([F[idx] - F[idx].mean(0) for idx in big], axis=0)
    pooled_var = resid.var(0)
    pooled_var[pooled_var < 1e-12] = 1e-12
    for idx in big:
        mu = F[idx].mean(0)
        v = F[idx].var(0)
        n = len(idx)
        v_shr = (n * v + VAR_PRIOR * pooled_var) / (n + VAR_PRIOR)
        F[idx] = (F[idx] - mu) / np.sqrt(np.maximum(v_shr, 1e-12))
    big_set = set(np.concatenate(big).tolist())
    small = np.array([i for i in range(len(F)) if i not in big_set], dtype=int)
    if len(small):
        F[small] = F[small] / np.sqrt(pooled_var)
    return F


def _lw_cov(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(0)
    sd = x.std(0)
    sd[sd < 1e-8] = 1.0
    x = x / sd
    c, _ = ledoit_wolf(x, assume_centered=True)
    c = (c + c.T) / 2
    c += 1e-6 * np.eye(c.shape[0])
    return c


def _center_by_site(H, sites):
    if sites is None:
        return H - H.mean(0)
    H = H.copy()
    rest = np.ones(len(H), dtype=bool)
    for s in np.unique(sites):
        idx = np.where(sites == s)[0]
        if len(idx) >= MIN_SITE:
            H[idx] -= H[idx].mean(0)
            rest[idx] = False
    if rest.any():
        H[rest] -= H[rest].mean(0)
    return H


def featurize(items, ctx):
    strategy = "filt_noglobal" if "filt_noglobal" in ctx["strategies"] else ctx["strategies"][0]
    ts = ctx["load_timeseries"](items, "cc200", strategy)
    covs, halves = [], []
    for x in ts:
        x = np.asarray(x, dtype=np.float64)
        covs.append(_lw_cov(x))
        T = x.shape[0]
        if T >= 2 * MIN_HALF:
            h = T // 2
            halves.append((_lw_cov(x[:h]), _lw_cov(x[h:])))
        else:
            halves.append(None)
    covs = np.asarray(covs)

    G_all = _geometric_mean(covs)
    Gih_all = _sym_fn(G_all, lambda w: 1.0 / np.sqrt(w))
    whiten = [Gih_all] * len(covs)

    sites = _item_sites(items, ctx)
    if sites is not None:
        sites = np.asarray(sites)
        for s in np.unique(sites):
            idx = np.where(sites == s)[0]
            if len(idx) < MIN_SITE:
                continue
            Gs = _geometric_mean(covs[idx])
            Gih_s = _sym_fn(Gs, lambda w: 1.0 / np.sqrt(w))
            for i in idx:
                whiten[i] = Gih_s

    n = covs.shape[1]
    iu = np.triu_indices(n, k=0)
    wts = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    feats = []
    H1, H2, hok = [], [], []
    for i, (c, W) in enumerate(zip(covs, whiten)):
        t = _sym_fn(W @ c @ W, np.log)
        feats.append(t[iu] * wts)
        if halves[i] is not None:
            a, b = halves[i]
            H1.append((_sym_fn(W @ a @ W, np.log)[iu] * wts).astype(np.float32))
            H2.append((_sym_fn(W @ b @ W, np.log)[iu] * wts).astype(np.float32))
            hok.append(i)
    F = np.asarray(feats, dtype=np.float64)
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    if sites is not None:
        F = _site_scale(F, sites)

    # Label-free split-half reliability per edge -> feature weights sqrt(Spearman-Brown reliability)
    if len(hok) >= 30:
        hs = sites[np.asarray(hok)] if sites is not None else None
        A = _center_by_site(np.nan_to_num(np.asarray(H1, dtype=np.float64)), hs)
        B = _center_by_site(np.nan_to_num(np.asarray(H2, dtype=np.float64)), hs)
        num = (A * B).sum(0)
        den = np.sqrt((A * A).sum(0) * (B * B).sum(0))
        r = np.where(den > 1e-12, num / np.maximum(den, 1e-12), 0.0)
        r = np.clip(r, 0.0, 1.0)
        r_sb = 2 * r / (1 + r)
        w = np.sqrt(np.clip(r_sb, REL_FLOOR, 1.0))
        F = F * w[None, :]
        del A, B

    F = F.astype(np.float32)
    return np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)


def _conn_fit_score(Xtr, ytr, Xte, C):
    # Center only: features are already per-site variance standardized and reliability-weighted.
    sc = StandardScaler(with_std=False).fit(Xtr)
    clf = LogisticRegression(C=C, max_iter=5000).fit(sc.transform(Xtr), ytr)
    return clf.decision_function(sc.transform(Xte))


def _to_df(m, n):
    if m is None:
        return None
    try:
        df = m if isinstance(m, pd.DataFrame) else pd.DataFrame(m)
    except Exception:
        return None
    if len(df) != n:
        return None
    return df.reset_index(drop=True)


def _sex_col(s):
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().mean() > 0.5:
        return num.astype(float)
    st = s.astype(str).str.strip().str.lower()
    return st.map(lambda v: 1.0 if v.startswith("m") else (0.0 if v.startswith("f") else np.nan))


def _covariates(mtr, mte):
    """Build covariate matrices (train, test) from meta frames; fitted on train only."""
    cols_tr, cols_te = [], []
    if "sex" in mtr.columns and "sex" in mte.columns:
        a, b = _sex_col(mtr["sex"]), _sex_col(mte["sex"])
        fill = a.mean() if a.notna().any() else 0.0
        cols_tr.append(a.fillna(fill).values)
        cols_te.append(b.fillna(fill).values)
    if "age" in mtr.columns and "age" in mte.columns:
        a = pd.to_numeric(mtr["age"], errors="coerce")
        b = pd.to_numeric(mte["age"], errors="coerce")
        fill = a.median() if a.notna().any() else 0.0
        a, b = a.fillna(fill).values, b.fillna(fill).values
        cols_tr += [a, a ** 2]
        cols_te += [b, b ** 2]
    if "site" in mtr.columns and "site" in mte.columns:
        s_tr = mtr["site"].astype(str).values
        s_te = mte["site"].astype(str).values
        for s in sorted(set(s_tr)):
            cols_tr.append((s_tr == s).astype(float))
            cols_te.append((s_te == s).astype(float))
    if not cols_tr:
        return None, None
    return np.column_stack(cols_tr), np.column_stack(cols_te)


def _inner_oof(Xtr, y, groups):
    """Out-of-fold connectivity scores for each C in the grid; returns (best_C, oof_of_best)."""
    if groups is not None and len(np.unique(groups)) >= 5:
        splits = list(GroupKFold(n_splits=5).split(Xtr, y, np.asarray(groups)))
    else:
        splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(Xtr, y))
    best_C, best_auc, best_oof = None, -np.inf, None
    for C in C_GRID:
        oof = np.zeros(len(y))
        for tr, va in splits:
            oof[va] = _conn_fit_score(Xtr[tr], y[tr], Xtr[va], C)
        try:
            auc = roc_auc_score(y, oof)
        except ValueError:
            auc = 0.5
        if auc > best_auc + 1e-4:
            best_C, best_auc, best_oof = C, auc, oof
    if best_C is None:
        best_C = C_DEFAULT
        best_oof = np.zeros(len(y))
        for tr, va in splits:
            best_oof[va] = _conn_fit_score(Xtr[tr], y[tr], Xtr[va], best_C)
    return best_C, best_oof


def fit_predict(F_train, y_train, F_test, ctx):
    Xtr = np.asarray(F_train, dtype=np.float64)
    Xte = np.asarray(F_test, dtype=np.float64)
    y = np.asarray(y_train).astype(int)

    best_C, oof = _inner_oof(Xtr, y, ctx.get("groups_train"))
    test_score = _conn_fit_score(Xtr, y, Xte, best_C)

    mtr = _to_df(ctx.get("meta_train"), len(y))
    mte = _to_df(ctx.get("meta_test"), len(Xte))
    if mtr is None or mte is None:
        return test_score
    Ctr, Cte = _covariates(mtr, mte)
    if Ctr is None:
        return test_score

    Ztr = np.column_stack([oof, Ctr])
    Zte = np.column_stack([test_score, Cte])
    sc = StandardScaler().fit(Ztr)
    meta_clf = LogisticRegression(C=1.0, max_iter=5000)
    meta_clf.fit(sc.transform(Ztr), y)
    return meta_clf.predict_proba(sc.transform(Zte))[:, 1]