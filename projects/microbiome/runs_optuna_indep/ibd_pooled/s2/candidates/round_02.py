PARAMS = {'level': 'genus', 'model': 'lr', 'C': 0.3514713523828802, 'rf_trees': 1000, 'rf_leaf': 3, 'hgb_lr': 0.17499133166217226, 'hgb_depth': 3, 'hgb_iter': 1000}
"""Seed-derived (independent) Optuna template: the seed (relative abundance at a taxonomic level ->
random forest) with its constants and choices made tunable plus the generic model list
(docs/preregistration_phase3.md). PARAMS is prepended by core/search.py."""
import numpy as np
from core.generic_models import make_classifier

LEVEL_IDX = {"family": 4, "genus": 5}


def featurize(items, ctx):
    rel = items.div(items.sum(axis=1).clip(lower=1), axis=0)
    if PARAMS["level"] == "otu":
        return rel
    k = LEVEL_IDX[PARAMS["level"]]
    empty = {4: "f__", 5: "g__"}[k]                          # same fallback label as the seed
    lab = [c.split(";")[k] if len(c.split(";")) > k else empty for c in rel.columns]
    return rel.T.groupby(np.array(lab)).sum().T


def fit_predict(F_train, y_train, F_test, ctx):
    cols = F_train.columns
    Xtr, Xte = F_train[cols].values, F_test.reindex(columns=cols, fill_value=0).values
    clf = make_classifier(PARAMS).fit(Xtr, y_train)
    return clf.predict_proba(Xte)[:, 1]
