"""Genus-level relative abundance + OTU-level alpha diversity (Shannon, inverse Simpson,
analytically rarefied richness) -> random forest."""
import numpy as np
import pandas as pd
from scipy.special import gammaln
from sklearn.ensemble import RandomForestClassifier

RARE_DEPTH = 1000


def _rarefied_richness(counts, depth):
    """Expected number of observed OTUs when subsampling `depth` reads without replacement."""
    c = counts[counts > 0]
    N = c.sum()
    if N <= 0:
        return 0.0
    n = min(depth, N)
    rest = N - c
    ok = rest >= n
    p_absent = np.zeros_like(c, dtype=float)
    r = rest[ok]
    p_absent[ok] = np.exp(gammaln(r + 1) - gammaln(r - n + 1) - gammaln(N + 1) + gammaln(N - n + 1))
    return float(np.sum(1.0 - p_absent))


def _alpha(items):
    X = np.rint(np.clip(items.values.astype(float), 0, None))
    depth = X.sum(axis=1)
    P = X / np.clip(depth, 1, None)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        logP = np.where(P > 0, np.log(P), 0.0)
    shannon = -(P * logP).sum(axis=1)
    simpson = (P ** 2).sum(axis=1)
    inv_simpson = np.where(simpson > 0, 1.0 / np.clip(simpson, 1e-12, None), 0.0)
    rich = np.array([_rarefied_richness(row, RARE_DEPTH) for row in X])
    return pd.DataFrame({"__alpha_shannon": shannon,
                         "__alpha_invsimpson": inv_simpson,
                         "__alpha_rare_richness": rich}, index=items.index)


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    G = rel.T.groupby(np.array(genus)).sum().T              # samples x genera
    return pd.concat([G, _alpha(items)], axis=1)


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]