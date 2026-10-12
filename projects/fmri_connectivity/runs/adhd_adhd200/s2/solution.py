"""CC200 (athena) Ledoit-Wolf correlation matrices -> tangent-space embedding (Riemannian
reference fitted on training fold only) -> isotropic (global) scaling that preserves the
tangent-space metric -> L2 logistic regression."""
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from pyriemann.tangentspace import TangentSpace


def _lw_corr(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(0)
    sd = x.std(0)
    good = sd > 1e-8
    z = np.zeros_like(x)
    z[:, good] = x[:, good] / sd[good]
    c = LedoitWolf(store_precision=False, assume_centered=True).fit(z).covariance_
    # dead ROIs (outside FOV): set to identity rows so the matrix stays SPD and uninformative
    d = np.sqrt(np.clip(np.diag(c), 1e-12, None))
    c = c / np.outer(d, d)
    c[~good, :] = 0.0
    c[:, ~good] = 0.0
    c[~good, ~good] = 1.0
    c = 0.5 * (c + c.T) + 1e-6 * np.eye(c.shape[0])
    return c


def featurize(items, ctx):
    strategy = "filt_noglobal" if "filt_noglobal" in ctx["strategies"] else ctx["strategies"][0]
    ts = ctx["load_timeseries"](items, "cc200", strategy)
    covs = [_lw_corr(x) for x in ts]
    return np.asarray(covs, dtype=np.float32)


def fit_predict(F_train, y_train, F_test, ctx):
    Ctr = np.asarray(F_train, dtype=np.float64)
    Cte = np.asarray(F_test, dtype=np.float64)
    ts = TangentSpace(metric="riemann").fit(Ctr)
    Xtr = ts.transform(Ctr)
    Xte = ts.transform(Cte)
    # isotropic scaling: centre on training mean, divide by one global scale (RMS of feature SDs)
    # so relative tangent-space geometry is preserved and C keeps the same effective meaning
    mu = Xtr.mean(0)
    scale = np.sqrt(np.mean(Xtr.var(0)))
    scale = scale if scale > 1e-12 else 1.0
    Xtr = (Xtr - mu) / scale
    Xte = (Xte - mu) / scale
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(Xtr, y_train)
    return clf.predict_proba(Xte)[:, 1]