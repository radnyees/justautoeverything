"""Pearson connectivity on CC200, Fisher z, upper triangle, per-site feature standardization
(harmonization), L2 logistic regression."""
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
    if hasattr(meta, "columns"):
        return meta["site"].astype(str).values
    if isinstance(meta, dict):
        return np.asarray([str(s) for s in meta["site"]])
    return np.asarray([str(m["site"]) if isinstance(m, dict) else str(m) for m in meta])


def _site_standardize(F_train, s_train, F_test, s_test):
    """Z-score each feature within site, using training-fold statistics of that site
    (falls back to the test items' own label-free statistics for unseen sites)."""
    Ztr = np.empty_like(F_train, dtype=np.float64)
    Zte = np.empty_like(F_test, dtype=np.float64)
    for s in np.unique(np.concatenate([s_train, s_test])):
        mtr = s_train == s
        mte = s_test == s
        ref = F_train[mtr] if mtr.sum() >= 2 else F_test[mte]
        if len(ref) < 2:
            ref = F_train
        mu = ref.mean(0)
        sd = ref.std(0) + 1e-6
        if mtr.any():
            Ztr[mtr] = (F_train[mtr] - mu) / sd
        if mte.any():
            Zte[mte] = (F_test[mte] - mu) / sd
    return Ztr, Zte


def fit_predict(F_train, y_train, F_test, ctx):
    F_train = np.asarray(F_train, dtype=np.float64)
    F_test = np.asarray(F_test, dtype=np.float64)
    s_train = _sites(ctx.get("meta_train"), len(F_train))
    s_test = _sites(ctx.get("meta_test"), len(F_test))
    Xtr, Xte = _site_standardize(F_train, s_train, F_test, s_test)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(Xtr), y_train)
    return clf.predict_proba(sc.transform(Xte))[:, 1]