"""CC200 Ledoit-Wolf correlation -> tangent-space embedding (log-Euclidean reference), L2 logistic regression."""
import numpy as np
from sklearn.covariance import ledoit_wolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _sym_fn(M, fn):
    w, V = np.linalg.eigh((M + M.T) / 2.0)
    return (V * fn(w)) @ V.T


def featurize(items, ctx):
    strategies = ctx["strategies"]
    strategy = "filt_noglobal" if "filt_noglobal" in strategies else strategies[0]
    ts = ctx["load_timeseries"](items, "cc200", strategy)

    covs = []
    for x in ts:
        x = np.asarray(x, dtype=np.float64)
        x = x - x.mean(axis=0)
        sd = x.std(axis=0)
        good = sd > 1e-8
        z = np.zeros_like(x)
        z[:, good] = x[:, good] / sd[good]
        c, _ = ledoit_wolf(z, assume_centered=True)
        # Dead (all-zero) ROIs: set to identity so they carry no connectivity.
        c[~good, :] = 0.0
        c[:, ~good] = 0.0
        c[~good, ~good] = 1.0
        c = c + 1e-3 * np.eye(c.shape[0])
        covs.append(c)

    # Label-free reference: log-Euclidean mean of all subjects' matrices.
    logs = [_sym_fn(c, lambda w: np.log(np.maximum(w, 1e-10))) for c in covs]
    ref = _sym_fn(np.mean(logs, axis=0), np.exp)
    W = _sym_fn(ref, lambda w: 1.0 / np.sqrt(np.maximum(w, 1e-10)))

    n = covs[0].shape[0]
    iu = np.triu_indices(n, k=0)
    coef = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    feats = []
    for c in covs:
        t = _sym_fn(W @ c @ W, lambda w: np.log(np.maximum(w, 1e-10)))
        feats.append(t[iu] * coef)
    return np.asarray(feats, dtype=np.float32)


def fit_predict(F_train, y_train, F_test, ctx):
    sc = StandardScaler().fit(F_train)
    clf = LogisticRegression(C=0.001, max_iter=5000)
    clf.fit(sc.transform(F_train), y_train)
    return clf.predict_proba(sc.transform(F_test))[:, 1]