"""Lineage-aware family+genus relative abundance, percentile-ranked within study (batch harmonization)
+ Gevers-style microbial dysbiosis index, random forest blended with the index (weight chosen by OOB)."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# Gevers 2014 (RISK) microbial dysbiosis index: taxa increased vs decreased in Crohn's disease.
_UP_FAMILIES = {"Enterobacteriaceae", "Pasteurellaceae", "Veillonellaceae", "Fusobacteriaceae",
                "Neisseriaceae", "Gemellaceae"}
_DOWN_ORDERS = {"Bacteroidales", "Clostridiales"}
_DOWN_FAMILIES = {"Erysipelotrichaceae", "Bifidobacteriaceae"}
_MDI = "__dysbiosis_index__"


def _study_labels(items, ctx, key="meta"):
    meta = ctx.get(key) if isinstance(ctx, dict) else None
    if meta is None or "study" not in getattr(meta, "columns", []):
        return np.zeros(len(items), dtype=object)
    if meta.index.isin(items.index).all() and items.index.isin(meta.index).all():
        return meta.loc[items.index, "study"].astype(str).values
    if len(meta) == len(items):
        return meta["study"].astype(str).values
    return np.zeros(len(items), dtype=object)


def _lineage(col, depth):
    # Full lineage prefix up to `depth` levels (k,p,c,o,f = 5; +g = 6), so that
    # unannotated genera ('g__') stay distinct per family instead of being lumped together.
    parts = col.split(";")
    parts = parts + [""] * max(0, depth - len(parts))
    return ";".join(parts[:depth])


def _level(col, prefix):
    for p in col.split(";"):
        p = p.strip()
        if p.startswith(prefix):
            return p[len(prefix):].strip("[]")
    return ""


def _dysbiosis_index(rel):
    fam = np.array([_level(c, "f__") for c in rel.columns])
    order = np.array([_level(c, "o__") for c in rel.columns])
    up = np.isin(fam, list(_UP_FAMILIES))
    down = (np.isin(order, list(_DOWN_ORDERS)) | np.isin(fam, list(_DOWN_FAMILIES))) & ~up
    eps = 1e-5
    return np.log(rel.loc[:, up].sum(axis=1) + eps) - np.log(rel.loc[:, down].sum(axis=1) + eps)


def _rank_within(values, studies):
    return pd.Series(np.asarray(values, dtype=float)).groupby(np.asarray(studies)).rank(pct=True).values


def _within_study_auc(y, score, studies):
    aucs, ws = [], []
    for s in pd.unique(studies):
        m = studies == s
        if len(np.unique(y[m])) == 2:
            aucs.append(roc_auc_score(y[m], score[m]))
            ws.append(m.sum())
    return np.average(aucs, weights=ws) if aucs else 0.5


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    blocks = []
    for depth in (5, 6):                                    # family, genus
        keys = np.array([_lineage(c, depth) for c in rel.columns])
        blocks.append(rel.T.groupby(keys).sum().T)
    G = pd.concat(blocks, axis=1)
    G = G.loc[:, ~G.columns.duplicated()]
    G[_MDI] = _dysbiosis_index(rel).values
    # Per-study percentile ranks: removes study-specific depth/region scale shifts,
    # keeps within-study ordering (all zeros within a study share one tied rank).
    studies = _study_labels(G, ctx)
    out = pd.DataFrame(0.0, index=G.index, columns=G.columns)
    for s in pd.unique(studies):
        m = studies == s
        out.loc[m] = G.loc[m].rank(axis=0, method="average", pct=True).values
    return out


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns                                  # align test taxa to training taxa
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    y = np.asarray(y_train).astype(int)
    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1, oob_score=True)
    rf.fit(Xtr, y)

    s_tr = _study_labels(F_train, ctx, "meta_train")
    s_te = _study_labels(F_test, ctx, "meta_test")

    # Choose blend weight for the dysbiosis index from OOB within-study AUROC (conservative: ties -> RF).
    oob = rf.oob_decision_function_[:, 1]
    oob = np.nan_to_num(oob, nan=np.nanmean(oob))
    r_oob = _rank_within(oob, s_tr)
    r_mdi_tr = _rank_within(F_train[_MDI].values, s_tr)
    best_w, best_auc = 0.0, _within_study_auc(y, r_oob, s_tr)
    for w in (0.15, 0.3, 0.5):
        auc = _within_study_auc(y, (1 - w) * r_oob + w * r_mdi_tr, s_tr)
        if auc > best_auc + 0.002:
            best_w, best_auc = w, auc

    p = rf.predict_proba(Xte)[:, 1]
    if best_w == 0.0:
        return p
    r_p = _rank_within(p, s_te)
    r_mdi_te = _rank_within(F_test.reindex(columns=cols, fill_value=0)[_MDI].values, s_te)
    return (1 - best_w) * r_p + best_w * r_mdi_te