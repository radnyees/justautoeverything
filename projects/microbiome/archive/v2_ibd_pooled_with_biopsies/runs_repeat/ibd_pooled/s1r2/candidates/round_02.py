"""Genus-level relative abundance + alpha diversity, per-study percentile-rank harmonization + random forest."""
import numpy as np
import pandas as pd
from scipy.special import gammaln
from sklearn.ensemble import RandomForestClassifier


def _study_labels(items, ctx):
    meta = ctx.get("meta") if isinstance(ctx, dict) else None
    if meta is None or "study" not in getattr(meta, "columns", []):
        return np.zeros(len(items), dtype=object)
    try:
        s = meta["study"].reindex(items.index)
        if s.isna().any():
            raise KeyError
        return s.astype(str).values
    except Exception:
        if len(meta) == len(items):
            return meta["study"].astype(str).values
        return np.zeros(len(items), dtype=object)


def _shannon(P):
    with np.errstate(divide="ignore", invalid="ignore"):
        L = np.where(P > 0, P * np.log(P), 0.0)
    return -L.sum(axis=1)


def _inv_simpson(P):
    return 1.0 / np.clip((P ** 2).sum(axis=1), 1e-12, None)


def _expected_richness(C, depth):
    """Analytic rarefied richness (Hurlbert) of each row of count matrix C at given depth."""
    out = np.zeros(C.shape[0])
    for r in range(C.shape[0]):
        n = C[r][C[r] > 0]
        N = n.sum()
        if N <= depth:
            out[r] = len(n)
            continue
        rest = N - n
        ok = rest >= depth
        logp = np.full(len(n), -np.inf)
        logp[ok] = (gammaln(rest[ok] + 1) - gammaln(rest[ok] - depth + 1)
                    - gammaln(N + 1) + gammaln(N - depth + 1))
        out[r] = np.sum(1.0 - np.exp(logp))
    return out


def featurize(items, ctx):
    C = items.values.astype(float)
    depth = C.sum(axis=1)
    rel = items.div(np.clip(depth, 1, None), axis=0)
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    G = rel.T.groupby(np.array(genus)).sum().T              # samples x genera
    studies = _study_labels(items, ctx)

    div = pd.DataFrame(index=items.index)
    P = rel.values
    div["__shannon_otu"] = _shannon(P)
    div["__invsimpson_otu"] = _inv_simpson(P)
    div["__shannon_genus"] = _shannon(G.values)
    div["__invsimpson_genus"] = _inv_simpson(G.values)
    rich = np.zeros(len(items))
    for s in pd.unique(studies):
        m = studies == s
        d = max(int(np.percentile(depth[m], 10)), 50)        # common within-study depth
        rich[m] = _expected_richness(C[m], d)
    div["__rarefied_richness"] = rich

    feats = pd.concat([G, div], axis=1).astype(float)
    out = feats.copy()
    for s in pd.unique(studies):
        m = studies == s
        sub = feats.loc[m]
        # within-study percentile ranks (ties averaged); robust to depth/region/protocol shifts
        out.loc[m] = (sub.rank(axis=0, method="average") / max(m.sum(), 1)).values
    return out


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test features to training features
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]