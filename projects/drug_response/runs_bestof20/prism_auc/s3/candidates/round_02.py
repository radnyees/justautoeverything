"""Top-5,000 most variable genes, standardized, plus lightly-penalized one-hot lineage indicators;
one multi-output ridge regression for all drugs (missing training AUCs filled with the drug's
training mean)."""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

LINEAGE_WEIGHT = 20.0  # effective ridge penalty on lineage coefs = alpha / w^2 ~ 12.5


def _lineage(meta, index):
    """Return OncotreeLineage labels aligned to `index` (or None if unavailable)."""
    if meta is None:
        return None
    if isinstance(meta, pd.DataFrame):
        col = next((c for c in meta.columns if c.lower() == "oncotreelineage"), None)
        if col is None:
            return None
        s = meta[col]
    elif isinstance(meta, pd.Series):
        s = meta
    else:
        return None
    if len(s) == len(index) and not s.index.equals(index):
        if set(index).issubset(set(s.index)):
            s = s.loc[index]
        else:
            s = pd.Series(s.values, index=index)
    elif not s.index.equals(index):
        s = s.reindex(index)
    return s.fillna("Unknown").astype(str).values


def fit_predict(F_train, y_train, F_test, ctx):
    genes = F_train.var().sort_values(ascending=False).index[:5000]
    sc = StandardScaler().fit(F_train[genes].values)
    Xtr, Xte = sc.transform(F_train[genes].values), sc.transform(F_test[genes].values)

    lin_tr = _lineage(ctx.get("meta_train"), F_train.index)
    lin_te = _lineage(ctx.get("meta_test"), F_test.index)
    if lin_tr is not None and lin_te is not None:
        cats = sorted(set(lin_tr))
        Ltr = np.stack([(lin_tr == c).astype(float) for c in cats], axis=1)
        Lte = np.stack([(lin_te == c).astype(float) for c in cats], axis=1)
        Xtr = np.hstack([Xtr, LINEAGE_WEIGHT * Ltr])
        Xte = np.hstack([Xte, LINEAGE_WEIGHT * Lte])

    Y = y_train.fillna(y_train.mean()).values
    model = Ridge(alpha=5000.0).fit(Xtr, Y)
    return model.predict(Xte)