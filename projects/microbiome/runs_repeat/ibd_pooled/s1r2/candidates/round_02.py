"""Genus-level relative abundance + alpha diversity, per-study percentile-rank harmonization + random forest."""
import numpy as np
import pandas as pd
from scipy.special import gammaln
from sklearn.ensemble import RandomForestClassifier


def _study_labels(items, ctx):
    meta = ctx.get("meta") if isinstance(ctx, dict) else None
    if meta is None or "study" not in getattr(meta, "columns", []):
        return pd.Series("all", index=items.index)
    try:
        s = meta.loc[items.index, "study"]
        return pd.Series(s.values, index=items.index).astype(str)
    except Exception:
        if len(meta) == len(items):
            return pd.Series(meta["study"].values, index=items.index).astype(str)
        return pd.Series("all", index=items.index)


def _shannon(P):
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.nansum(np.where(P > 0, P * np.log(P), 0.0), axis=1)


def _rarefied_richness(C, depth=1000):
    """Expected number of observed OTUs when subsampling n reads without replacement."""
    C = np.floor(np.asarray(C, dtype=float))
    N = C.sum(axis=1)
    out = np.zeros(C.shape[0])
    for i in range(C.shape[0]):
        Ni = C[i][C[i] > 0]
        tot = N[i]
        if tot <= 0:
            continue
        n = min(depth, tot)
        # P(OTU absent) = C(tot - Ni, n) / C(tot, n); zero when tot - Ni < n
        ok = (tot - Ni) >= n
        logp = np.full(Ni.shape, -np.inf)
        a = tot - Ni[ok]
        logp[ok] = (gammaln(a + 1) - gammaln(a - n + 1)
                    - gammaln(tot + 1) + gammaln(tot - n + 1))
        out[i] = np.sum(1.0 - np.exp(logp))
    return out


def featurize(items, ctx):
    counts = items.values.astype(float)
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    G = rel.T.groupby(np.array(genus)).sum().T              # samples x genera

    # alpha diversity (IBD: reduced diversity is a robust cross-cohort signal)
    P = rel.values
    A = pd.DataFrame({
        "alpha__shannon_otu": _shannon(P),
        "alpha__shannon_genus": _shannon(G.values),
        "alpha__invsimpson_otu": 1.0 / np.clip((P ** 2).sum(axis=1), 1e-12, None),
        "alpha__rich_rare1000": _rarefied_richness(counts, 1000),
    }, index=items.index)
    X = pd.concat([G, A], axis=1)

    study = _study_labels(items, ctx)
    # per-study percentile ranks: label-free harmonization of study-specific scale shifts
    R = X.groupby(study.values).rank(pct=True, method="average")
    return R.reindex(index=X.index, columns=X.columns).fillna(0.5)


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test features to training features
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0.5).values
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, y_train)
    return rf.predict_proba(Xte)[:, 1]