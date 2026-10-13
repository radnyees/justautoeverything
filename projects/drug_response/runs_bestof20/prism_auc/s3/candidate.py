"""Top-5,000 most variable genes, standardized, plus lightly-penalized one-hot lineage indicators;
one multi-output ridge regression for all drugs (missing training AUCs filled with drug mean)."""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ALPHA = 5000.0
LINEAGE_SCALE = 10.0   # effective penalty on lineage coefs = ALPHA / LINEAGE_SCALE**2
MIN_LINEAGE_N = 5


def _lineage(meta, index):
    if meta is None:
        return pd.Series("unknown", index=index)
    if isinstance(meta, pd.DataFrame) and "OncotreeLineage" in meta.columns:
        s = meta["OncotreeLineage"]
        if s.index.isin(index).all() and len(s) == len(index):
            s = s.reindex(index)
        else:
            s = pd.Series(s.values, index=index)
        return s.fillna("unknown").astype(str)
    return pd.Series("unknown", index=index)


def fit_predict(F_train, y_train, F_test, ctx):
    genes = F_train.var().sort_values(ascending=False).index[:5000]
    sc = StandardScaler().fit(F_train[genes].values)
    Xtr, Xte = sc.transform(F_train[genes].values), sc.transform(F_test[genes].values)

    lin_tr = _lineage(ctx.get("meta_train"), F_train.index)
    lin_te = _lineage(ctx.get("meta_test"), F_test.index)
    counts = lin_tr.value_counts()
    keep = sorted(counts[counts >= MIN_LINEAGE_N].index)
    if len(keep) > 1:
        def onehot(lin):
            L = np.zeros((len(lin), len(keep) + 1))
            pos = {k: i for i, k in enumerate(keep)}
            for r, v in enumerate(lin.values):
                L[r, pos.get(v, len(keep))] = 1.0
            return L
        Ltr, Lte = onehot(lin_tr), onehot(lin_te)
        mu = Ltr.mean(0)
        Xtr = np.hstack([Xtr, (Ltr - mu) * LINEAGE_SCALE])
        Xte = np.hstack([Xte, (Lte - mu) * LINEAGE_SCALE])

    Y = y_train.fillna(y_train.mean()).values
    model = Ridge(alpha=ALPHA).fit(Xtr, Y)
    return model.predict(Xte)