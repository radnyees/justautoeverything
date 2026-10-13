"""Pearson connectivity on CC200, Fisher z, upper triangle, per-site standardization
(location/scale harmonization from training-fold site statistics), L2 logistic regression."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def featurize(items, ctx):
    strategies = ctx["strategies"]
    strategy = "filt_noglobal" if "filt_noglobal" in strategies else strategies[0]
    ts = ctx["load_timeseries"](items, "cc200", strategy)
    iu = None
    feats = []
    for x in ts:
        x = np.asarray(x, dtype=np.float64)
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
        s = np.asarray(meta["site"])
    except Exception:
        try:
            s = np.asarray([m["site"] for m in meta])
        except Exception:
            return np.zeros(n, dtype=object)
    if len(s) != n:
        return np.zeros(n, dtype=object)
    return s.astype(str)


def _site_standardize(F_train, F_test, s_train, s_test):
    Xtr = np.array(F_train, dtype=np.float64)
    Xte = np.array(F_test, dtype=np.float64)
    gmu = Xtr.mean(0)
    gsd = Xtr.std(0) + 1e-6
    for site in np.unique(np.concatenate([s_train, s_test])):
        mtr = s_train == site
        mte = s_test == site
        if mtr.sum() >= 5:
            ref = Xtr[mtr]
        elif (mtr.sum() + mte.sum()) >= 5:
            # label-free fallback: use all available items of this site
            ref = np.vstack([Xtr[mtr], Xte[mte]])
        else:
            mu, sd = gmu, gsd
            Xtr[mtr] = (Xtr[mtr] - mu) / sd
            Xte[mte] = (Xte[mte] - mu) / sd
            continue
        mu = ref.mean(0)
        sd = ref.std(0) + 1e-6
        Xtr[mtr] = (Xtr[mtr] - mu) / sd
        Xte[mte] = (Xte[mte] - mu) / sd
    return Xtr, Xte


def fit_predict(F_train, y_train, F_test, ctx):
    n_tr, n_te = len(F_train), len(F_test)
    s_train = _sites(ctx.get("meta_train"), n_tr)
    s_test = _sites(ctx.get("meta_test"), n_te)
    Xtr, Xte = _site_standardize(F_train, F_test, s_train, s_test)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(Xtr), y_train)
    return clf.predict_proba(sc.transform(Xte))[:, 1]