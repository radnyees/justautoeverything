"""log2 + per-sample median centring + per-batch protein median centring + half-min imputation + L2 LR."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def featurize(items, ctx):
    L = np.log2(items.astype(float))
    return L.sub(L.median(axis=1), axis=0)                 # per-sample median centring


def _batches(meta, n):
    if meta is None:
        return np.zeros(n, dtype=object)
    if isinstance(meta, pd.DataFrame) and "batch" in meta.columns:
        return np.asarray(meta["batch"]).astype(str)
    return np.zeros(n, dtype=object)


def _batch_centre(Ftr, Fte, btr, bte):
    """Label-free per-batch, per-protein median centring (NaN-aware), using all samples of each batch."""
    Ftr = Ftr.copy()
    Fte = Fte.copy()
    allF = pd.concat([Ftr, Fte], axis=0)
    allb = np.concatenate([btr, bte])
    gmed = allF.median(axis=0)
    for b in np.unique(allb):
        m = allb == b
        bmed = allF.loc[m].median(axis=0)
        shift = (bmed - gmed).fillna(0.0)
        # require a few observed values in the batch for a reliable shift
        nobs = allF.loc[m].notna().sum(axis=0)
        shift[nobs < 5] = 0.0
        mtr = btr == b
        mte = bte == b
        if mtr.any():
            Ftr.loc[mtr] = Ftr.loc[mtr] - shift.values
        if mte.any():
            Fte.loc[mte] = Fte.loc[mte] - shift.values
    return Ftr, Fte


def fit_predict(F_train, y_train, F_test, ctx):
    keep = F_train.columns[F_train.notna().mean() >= 0.7]  # quantified in >= 70% of train samples
    Ftr, Fte = F_train[keep], F_test[keep]
    btr = _batches(ctx.get("meta_train"), len(Ftr))
    bte = _batches(ctx.get("meta_test"), len(Fte))
    Ftr, Fte = _batch_centre(Ftr, Fte, btr, bte)
    fill = Ftr.min() - 1.0                                  # half-minimum on the log2 scale
    Xtr = Ftr.fillna(fill).values
    Xte = Fte.fillna(fill).values
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.05, max_iter=5000)
    clf.fit(sc.transform(Xtr), y_train)
    return clf.predict_proba(sc.transform(Xte))[:, 1]