"""Ledoit-Wolf shrinkage covariance on CC200 (z-scored series) -> Riemannian re-centering per
site (each site's covariances whitened by that site's geometric mean, label-free) -> tangent space
at identity, upper triangle, L2 logistic regression. Stacked with demographic covariates
(sex, age, site) via inner grouped CV out-of-fold scores."""
import numpy as np
import pandas as pd
from sklearn.covariance import ledoit_wolf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

MIN_SITE = 10


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


def featurize(items, ctx):
    strategy = "filt_noglobal" if "filt_noglobal" in ctx["strategies"] else ctx["strategies"][0]
    ts = ctx["load_timeseries"](items, "cc200", strategy)
    covs = []
    for x in ts:
        x = np.asarray(x, dtype=np.float64)
        x = x - x.mean(0)
        sd = x.std(0)
        sd[sd < 1e-8] = 1.0
        x = x / sd
        c, _ = ledoit_wolf(x, assume_centered=True)
        c = (c + c.T) / 2
        c += 1e-6 * np.eye(c.shape[0])
        covs.append(c)
    covs = np.asarray(covs)

    G_all = _geometric_mean(covs)
    Gih_all = _sym_fn(G_all, lambda w: 1.0 / np.sqrt(w))
    whiten = [Gih_all] * len(covs)

    sites = _item_sites(items, ctx)
    if sites is not None:
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
    for c, W in zip(covs, whiten):
        t = _sym_fn(W @ c @ W, np.log)
        feats.append(t[iu] * wts)
    F = np.asarray(feats, dtype=np.float32)
    return np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)


def _conn_model():
    return LogisticRegression(C=0.001, max_iter=5000)


def _conn_fit_score(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    clf = _conn_model().fit(sc.transform(Xtr), ytr)
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


def fit_predict(F_train, y_train, F_test, ctx):
    Xtr = np.asarray(F_train, dtype=np.float64)
    Xte = np.asarray(F_test, dtype=np.float64)
    y = np.asarray(y_train).astype(int)

    test_score = _conn_fit_score(Xtr, y, Xte)

    mtr = _to_df(ctx.get("meta_train"), len(y))
    mte = _to_df(ctx.get("meta_test"), len(Xte))
    if mtr is None or mte is None:
        return test_score
    Ctr, Cte = _covariates(mtr, mte)
    if Ctr is None:
        return test_score

    # Out-of-fold connectivity scores on the training fold (grouped inner CV)
    groups = ctx.get("groups_train")
    oof = np.zeros(len(y))
    if groups is not None and len(np.unique(groups)) >= 5:
        splitter = GroupKFold(n_splits=5).split(Xtr, y, np.asarray(groups))
    else:
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(Xtr, y)
    for tr, va in splitter:
        oof[va] = _conn_fit_score(Xtr[tr], y[tr], Xtr[va])

    Ztr = np.column_stack([oof, Ctr])
    Zte = np.column_stack([test_score, Cte])
    sc = StandardScaler().fit(Ztr)
    meta_clf = LogisticRegression(C=1.0, max_iter=5000)
    meta_clf.fit(sc.transform(Ztr), y)
    return meta_clf.predict_proba(sc.transform(Zte))[:, 1]