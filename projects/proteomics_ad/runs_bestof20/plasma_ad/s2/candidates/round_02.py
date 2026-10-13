"""log2 + per-sample median centring + per-batch per-protein median centring + half-min imputation + L2 LR."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def featurize(items, ctx):
    L = np.log2(items.astype(float))
    return L.sub(L.median(axis=1), axis=0)                 # per-sample median centring


def _batch_vector(meta, n):
    if meta is None:
        return None
    try:
        if isinstance(meta, pd.DataFrame):
            if "batch" not in meta.columns:
                return None
            b = meta["batch"].values
        else:
            b = np.asarray(meta)
        if len(b) != n:
            return None
        return np.asarray(b).astype(str)
    except Exception:
        return None


def _batch_center(F_train, F_test, ctx):
    """Subtract each protein's median within each batch (label-free, pooled train+test per batch)."""
    btr = _batch_vector(ctx.get("meta_train"), len(F_train))
    bte = _batch_vector(ctx.get("meta_test"), len(F_test))
    if btr is None or bte is None:
        return F_train, F_test
    A = pd.concat([F_train, F_test], axis=0, ignore_index=True)
    b = np.concatenate([btr, bte])
    if len(np.unique(b)) < 2:
        return F_train, F_test
    vals = A.values.copy()
    grand = np.nanmedian(vals, axis=0)
    for g in np.unique(b):
        m = b == g
        if m.sum() < 3:
            continue
        sub = vals[m]
        cnt = np.sum(~np.isnan(sub), axis=0)
        with np.errstate(all="ignore"):
            med = np.nanmedian(sub, axis=0)
        shift = np.where(cnt >= 3, med - grand, 0.0)
        shift = np.nan_to_num(shift)
        vals[m] = sub - shift
    A = pd.DataFrame(vals, columns=A.columns)
    ntr = len(F_train)
    Ftr = A.iloc[:ntr].set_axis(F_train.index, axis=0)
    Fte = A.iloc[ntr:].set_axis(F_test.index, axis=0)
    return Ftr, Fte


def fit_predict(F_train, y_train, F_test, ctx):
    keep = F_train.columns[F_train.notna().mean() >= 0.7]  # quantified in >= 70% of train samples
    Ftr, Fte = _batch_center(F_train[keep], F_test[keep], ctx)
    fill = Ftr.min() - 1.0                                  # half-minimum on the log2 scale
    Xtr = Ftr.fillna(fill).values
    Xte = Fte.fillna(fill).values
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.05, max_iter=5000)
    clf.fit(sc.transform(Xtr), y_train)
    return clf.predict_proba(sc.transform(Xte))[:, 1]