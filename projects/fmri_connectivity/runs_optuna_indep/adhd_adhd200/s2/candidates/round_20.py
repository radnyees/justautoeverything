PARAMS = {'atlas': 'ho', 'strategy': 0, 'model': 'lr', 'C': 0.00015385851447008916, 'rf_trees': 200, 'rf_leaf': 1, 'hgb_lr': 0.2442025347575869, 'hgb_depth': 2, 'hgb_iter': 1000}
"""Seed-derived (independent) Optuna template: the seed (Pearson connectivity, Fisher z, upper triangle,
standardization + L2 logistic regression) with atlas / denoising strategy chosen among the options the
task exposes, plus the generic model list (docs/preregistration_phase3.md). PARAMS prepended by search.py."""
import numpy as np
from core.generic_models import make_classifier


def featurize(items, ctx):
    strategy = ctx["strategies"][min(PARAMS["strategy"], len(ctx["strategies"]) - 1)]
    atlas = PARAMS["atlas"] if PARAMS["atlas"] in ctx["atlases"] else "cc200"
    feats, iu = [], None
    for x in ctx["load_timeseries"](items, atlas, strategy):
        c = np.nan_to_num(np.corrcoef(x.T), nan=0.0)
        if iu is None:
            iu = np.triu_indices(c.shape[0], k=1)
        feats.append(np.arctanh(np.clip(c[iu], -0.999, 0.999)))
    return np.asarray(feats, dtype=np.float32)


def fit_predict(F_train, y_train, F_test, ctx):
    clf = make_classifier(PARAMS).fit(F_train, y_train)
    return clf.predict_proba(F_test)[:, 1]
