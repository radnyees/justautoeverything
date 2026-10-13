"""Pearson connectivity on CC200, Fisher z, upper triangle, per-site feature standardization, L2 logistic regression."""
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
        s = meta["site"]
    except Exception:
        try:
            s = [m["site"] for m in meta]
        except Exception:
            return np.zeros(n, dtype=object)
    s = np.asarray(list(s) if not hasattr(s, "values") else s.values).astype(str)
    if len(s) != n:
        return np.zeros(n, dtype=object)
    return s


def _site_standardize(F_tr, F_te, s_tr, s_te):
    """Label-free per-site z-scoring, using all (train+test) items of each site."""
    F_tr = np.array(F_tr, dtype=np.float64, copy=True)
    F_te = np.array(F_te, dtype=np.float64, copy=True)
    for s in np.unique(np.concatenate([s_tr, s_te])):
        mtr, mte = s_tr == s, s_te == s
        X = np.vstack([F_tr[mtr], F_te[mte]])
        if X.shape[0] < 2:
            continue
        mu = X.mean(0)
        sd = X.std(0)
        sd[sd < 1e-8] = 1.0
        F_tr[mtr] = (F_tr[mtr] - mu) / sd
        F_te[mte] = (F_te[mte] - mu) / sd
    return F_tr, F_te


def fit_predict(F_train, y_train, F_test, ctx):
    F_train = np.asarray(F_train)
    F_test = np.asarray(F_test)
    s_tr = _sites(ctx.get("meta_train"), len(F_train))
    s_te = _sites(ctx.get("meta_test"), len(F_test))
    F_train, F_test = _site_standardize(F_train, F_test, s_tr, s_te)
    sc = StandardScaler().fit(F_train)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(F_train), y_train)
    return clf.predict_proba(sc.transform(F_test))[:, 1]