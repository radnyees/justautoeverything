PARAMS = {'center': True, 'min_present': 0.9, 'fill_offset': 1.0, 'model': 'lr', 'C': 0.9037654206093024, 'rf_trees': 1000, 'rf_leaf': 10, 'hgb_lr': 0.029585835634035187, 'hgb_depth': 2, 'hgb_iter': 100}
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
