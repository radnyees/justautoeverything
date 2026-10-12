"""Genus-level relative abundance, percentile-ranked within study (batch correction; absent taxa pinned to 0
so absence means the same thing across studies) + study one-hot + a supervised dysbiosis score (median
distance to same-study training controls) + extremely randomized trees."""
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.ensemble import ExtraTreesClassifier


def _study_series(items, ctx):
    meta = ctx.get("meta") if isinstance(ctx, dict) else None
    if meta is None or "study" not in getattr(meta, "columns", []):
        return pd.Series("all", index=items.index)
    st = meta["study"]
    if st.index.equals(items.index) or set(items.index).issubset(set(st.index)):
        st = st.reindex(items.index)
    elif len(st) == len(items):
        st = pd.Series(st.values, index=items.index)
    else:
        return pd.Series("all", index=items.index)
    return st.fillna("unknown").astype(str)


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    G = rel.T.groupby(np.array(genus)).sum().T              # samples x genera
    study = _study_series(items, ctx)
    # percentile rank of each genus within its study -> removes study-specific depth/protocol scale
    R = G.groupby(study.values).rank(pct=True, method="average")
    # zero-abundance ties otherwise get a study-prevalence-dependent mid-rank; pin absence to 0
    R = R.where(G.values > 0, 0.0)
    R.columns = ["rk_" + str(c) for c in R.columns]
    S = pd.get_dummies(study, prefix="study").astype(float)
    S.index = items.index
    return pd.concat([R, S], axis=1)


def _dysbiosis(Xq, sq, gq, Xc, sc, gc):
    """Median L1 distance (per feature) from each query sample to reference controls of the same study,
    excluding controls from the same subject (so training items never see themselves)."""
    D = cdist(Xq, Xc, metric="cityblock") / max(Xq.shape[1], 1)
    out = np.empty(len(Xq))
    for i in range(len(Xq)):
        other = np.ones(len(Xc), bool) if gq is None else (gc != gq[i])
        m = other & (sc == sq[i])
        if m.sum() < 5:                                     # too few same-study controls -> all controls
            m = other
        out[i] = np.median(D[i, m]) if m.any() else np.median(D[i])
    return out


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test features to training features
    Ftr = F_train[cols]
    Fte = F_test.reindex(columns=cols, fill_value=0)
    y = np.asarray(y_train).astype(int)

    rk_cols = [c for c in cols if str(c).startswith("rk_")]
    st_cols = [c for c in cols if str(c).startswith("study_")]
    if st_cols:
        s_tr = Ftr[st_cols].values.argmax(axis=1)
        s_te = Fte[st_cols].values.argmax(axis=1)
    else:
        s_tr = np.zeros(len(Ftr), int)
        s_te = np.zeros(len(Fte), int)
    groups = ctx.get("groups_train") if isinstance(ctx, dict) else None
    g_tr = np.asarray(groups) if groups is not None and len(groups) == len(Ftr) else np.arange(len(Ftr))

    Rtr = Ftr[rk_cols].values
    Rte = Fte[rk_cols].values
    ctrl = y == 0
    d_tr = _dysbiosis(Rtr, s_tr, g_tr, Rtr[ctrl], s_tr[ctrl], g_tr[ctrl])
    d_te = _dysbiosis(Rte, s_te, None, Rtr[ctrl], s_tr[ctrl], g_tr[ctrl])

    Xtr = np.column_stack([Ftr.values, d_tr])
    Xte = np.column_stack([Fte.values, d_te])
    # random split thresholds smooth over noisy, tie-heavy within-study rank cut points
    et = ExtraTreesClassifier(n_estimators=1000, min_samples_leaf=2, max_features="sqrt",
                              random_state=0, n_jobs=-1)
    et.fit(Xtr, y)
    return et.predict_proba(Xte)[:, 1]