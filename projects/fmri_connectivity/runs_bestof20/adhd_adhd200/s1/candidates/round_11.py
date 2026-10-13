"""Pearson connectivity on CC200, Fisher z, upper triangle, per-site standardization, L2 logistic regression."""
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
    s = np.asarray(list(s)).astype(str)
    if len(s) != n:
        return np.zeros(n, dtype=object)
    return s


def _site_standardize(F_train, F_test, s_train, s_test, min_n=5):
    """Z-score each feature within site. Stats come from the site's training items;
    if a site is missing or too small in training, use its own (unlabeled) test items."""
    Xtr = np.asarray(F_train, dtype=np.float64).copy()
    Xte = np.asarray(F_test, dtype=np.float64).copy()
    g_mu = Xtr.mean(0)
    g_sd = Xtr.std(0) + 1e-6
    for site in np.unique(np.concatenate([s_train, s_test])):
        mtr = s_train == site
        mte = s_test == site
        if mtr.sum() >= min_n:
            ref = Xtr[mtr]
        elif mte.sum() >= min_n:
            ref = Xte[mte]
        else:
            ref = None
        if ref is None:
            mu, sd = g_mu, g_sd
        else:
            mu = ref.mean(0)
            sd = ref.std(0) + 1e-6
        if mtr.any():
            Xtr[mtr] = (Xtr[mtr] - mu) / sd
        if mte.any():
            Xte[mte] = (Xte[mte] - mu) / sd
    return Xtr, Xte


def fit_predict(F_train, y_train, F_test, ctx):
    s_train = _sites(ctx.get("meta_train"), len(F_train))
    s_test = _sites(ctx.get("meta_test"), len(F_test))
    Xtr, Xte = _site_standardize(F_train, F_test, s_train, s_test)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(Xtr), y_train)
    return clf.predict_proba(sc.transform(Xte))[:, 1]