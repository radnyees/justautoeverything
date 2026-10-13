"""Pearson connectivity on CC200, Fisher z, upper triangle, per-site z-scoring (harmonization), L2 logistic regression."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def featurize(items, ctx):
    strategy = "filt_noglobal" if "filt_noglobal" in ctx["strategies"] else ctx["strategies"][0]
    ts = ctx["load_timeseries"](items, "cc200", strategy)
    iu = None
    feats = []
    for x in ts:
        c = np.corrcoef(x.T)
        c = np.nan_to_num(c, nan=0.0)
        if iu is None:
            iu = np.triu_indices(c.shape[0], k=1)
        feats.append(np.arctanh(np.clip(c[iu], -0.999, 0.999)))
    return np.asarray(feats, dtype=np.float32)


def _sites(meta, n):
    if meta is None:
        return np.zeros(n, dtype=object)
    try:
        import pandas as pd
        if isinstance(meta, pd.DataFrame):
            return meta["site"].astype(str).values
        if isinstance(meta, pd.Series):
            return meta.astype(str).values
    except Exception:
        pass
    if isinstance(meta, dict) and "site" in meta:
        return np.asarray([str(s) for s in meta["site"]], dtype=object)
    out = []
    for m in meta:
        if isinstance(m, dict):
            out.append(str(m.get("site", "")))
        else:
            out.append(str(m))
    if len(out) != n:
        return np.zeros(n, dtype=object)
    return np.asarray(out, dtype=object)


def _site_standardize(F_tr, F_te, s_tr, s_te):
    """Label-free per-site z-scoring using pooled (train+test) items of each site."""
    F_tr = F_tr.astype(np.float64).copy()
    F_te = F_te.astype(np.float64).copy()
    all_F = np.vstack([F_tr, F_te])
    all_s = np.concatenate([s_tr, s_te])
    g_mu = all_F.mean(0)
    g_sd = all_F.std(0) + 1e-6
    for s in np.unique(all_s):
        m = all_s == s
        if m.sum() >= 5:
            mu = all_F[m].mean(0)
            sd = all_F[m].std(0) + 1e-6
        else:
            mu, sd = g_mu, g_sd
        mtr = s_tr == s
        mte = s_te == s
        F_tr[mtr] = (F_tr[mtr] - mu) / sd
        F_te[mte] = (F_te[mte] - mu) / sd
    return F_tr, F_te


def fit_predict(F_train, y_train, F_test, ctx):
    F_train = np.asarray(F_train)
    F_test = np.asarray(F_test)
    s_tr = _sites(ctx.get("meta_train"), len(F_train))
    s_te = _sites(ctx.get("meta_test"), len(F_test))
    X_tr, X_te = _site_standardize(F_train, F_test, s_tr, s_te)
    sc = StandardScaler().fit(X_tr)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(X_tr), y_train)
    return clf.predict_proba(sc.transform(X_te))[:, 1]