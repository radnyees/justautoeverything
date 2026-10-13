"""core/metrics.py -- shared metrics.

within_site_auroc: size-weighted mean of per-site AUROCs over sites that contain both classes.
Used for multi-site / multi-study tasks so that a model cannot gain by learning which site has
more cases (site/study prevalence differs); sites are still available for harmonization.
"""
import numpy as np
from sklearn.metrics import roc_auc_score


def within_site_auroc(y_true, y_pred, sites):
    y, p, s = np.asarray(y_true), np.asarray(y_pred, dtype=float), np.asarray(sites).astype(str)
    num = den = 0.0
    for site in np.unique(s):
        m = s == site
        if len(np.unique(y[m])) == 2:
            num += m.sum() * roc_auc_score(y[m], p[m])
            den += m.sum()
    if den == 0:
        raise ValueError("no site contains both classes")
    return float(num / den)
