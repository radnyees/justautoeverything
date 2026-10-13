"""Genus-level relative abundance + Gevers-style dysbiosis index (log-ratio of IBD-enriched vs
IBD-depleted taxa) + random forest, with study x class balanced sample weights so that study
identity (confounded with case/control ratio) carries no label information."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Gevers et al. 2014 (RISK cohort) microbial dysbiosis index taxa
_UP_FAMILIES = {"Enterobacteriaceae", "Pasteurellaceae", "Veillonellaceae", "Fusobacteriaceae",
                "Neisseriaceae", "Gemellaceae"}
_DOWN_ORDERS = {"Bacteroidales", "Clostridiales"}
_DOWN_FAMILIES = {"Erysipelotrichaceae", "Bifidobacteriaceae"}


def _level(c, i):
    parts = c.split(";")
    if len(parts) <= i:
        return ""
    p = parts[i]
    return p.split("__", 1)[1] if "__" in p else p


def _dysbiosis(rel):
    fam = np.array([_level(c, 4) for c in rel.columns])
    order = np.array([_level(c, 3) for c in rel.columns])
    up = np.isin(fam, list(_UP_FAMILIES))
    down = (np.isin(order, list(_DOWN_ORDERS)) | np.isin(fam, list(_DOWN_FAMILIES))) & ~up
    eps = 1e-4
    up_ab = rel.values[:, up].sum(axis=1)
    down_ab = rel.values[:, down].sum(axis=1)
    return pd.DataFrame({
        "__MDI": np.log((up_ab + eps) / (down_ab + eps)),
        "__log_up": np.log(up_ab + eps),
        "__log_down": np.log(down_ab + eps),
    }, index=rel.index)


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    genus = [c.split(";")[5] if len(c.split(";")) > 5 else "g__" for c in rel.columns]
    G = rel.T.groupby(np.array(genus)).sum().T             # samples x genera
    return pd.concat([G, _dysbiosis(rel)], axis=1)


def _balanced_weights(y, study):
    """Within each study, cases and controls get equal total weight; each study keeps
    total weight proportional to its size. Removes the study <-> label confound."""
    y = np.asarray(y)
    study = np.asarray(study)
    w = np.ones(len(y), dtype=float)
    for s in np.unique(study):
        ms = study == s
        n_s = ms.sum()
        classes = np.unique(y[ms])
        if len(classes) < 2:
            # single-class study gives no within-study contrast; down-weight it
            w[ms] = 0.5
            continue
        for c in classes:
            m = ms & (y == c)
            w[m] = n_s / (len(classes) * m.sum())
    return w * len(y) / w.sum()


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test genera to training genera
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    meta_tr = ctx.get("meta_train", None)
    if meta_tr is not None and "study" in meta_tr and len(meta_tr) == len(Xtr):
        w = _balanced_weights(y_train, meta_tr["study"].astype(str).values)
    else:
        w = None
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
    rf.fit(Xtr, np.asarray(y_train), sample_weight=w)
    return rf.predict_proba(Xte)[:, 1]