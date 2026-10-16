PARAMS = {'center': False, 'min_present': 0.5, 'fill_offset': 2.0, 'model': 'hgb', 'C': 0.0051206204549250406, 'rf_trees': 1000, 'rf_leaf': 3, 'hgb_lr': 0.12044555687833554, 'hgb_depth': 3, 'hgb_iter': 1000}
"""Seed-derived (independent) Optuna template: the seed (log2, per-sample median centring, presence
filter, half-minimum imputation, L2 logistic regression) with its constants/steps tunable plus the generic
model list (docs/preregistration_phase3.md). PARAMS is prepended by core/search.py."""
import numpy as np
from core.generic_models import make_classifier


def featurize(items, ctx):
    L = np.log2(items.astype(float))
    return L.sub(L.median(axis=1), axis=0) if PARAMS["center"] else L


def fit_predict(F_train, y_train, F_test, ctx):
    keep = F_train.columns[F_train.notna().mean() >= PARAMS["min_present"]]
    fill = F_train[keep].min() - PARAMS["fill_offset"]
    Xtr = F_train[keep].fillna(fill).values
    Xte = F_test[keep].fillna(fill).values
    clf = make_classifier(PARAMS).fit(Xtr, y_train)
    return clf.predict_proba(Xte)[:, 1]
