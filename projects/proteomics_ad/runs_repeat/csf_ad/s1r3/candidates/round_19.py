"""log2 + per-sample median centring on a near-completely-quantified reference protein set
+ label-free removal of erythrocyte (blood-contamination) signal
+ half-minimum imputation + L2 logistic regression, ensembled with a prior-biology panel model
(known CSF AD markers), a model on proteins selected by within-site meta-analytic effect size,
a shallow, heavily regularised gradient-boosting model and a heavily bagged ExtraTrees model on
selected + panel proteins, augmented with up-marker / down-marker log-ratio features and
label-free co-expression module scores."""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from lightgbm import LGBMClassifier

# Known CSF AD-associated proteins (Bader 2020 tau-correlated set, glycolysis / 14-3-3 module,
# synaptic, glial / inflammatory, complement, lipid transport).
PANEL = {
    "SMOC1", "YWHAG", "YWHAZ", "YWHAB", "YWHAE", "YWHAQ", "PKM", "ALDOA", "ENO1", "ENO2",
    "GAPDH", "LDHA", "LDHB", "MDH1", "PGK1", "TPI1", "PGAM1", "PPIA", "SOD1", "PARK7",
    "TMSB10", "PEBP1", "CFL1", "CALM1", "CALM2", "CALM3", "CHI3L1", "CHI3L2", "SPP1",
    "GFAP", "NEFL", "NEFM", "NPTX1", "NPTX2", "NPTXR", "VGF", "NRGN", "SNAP25", "GAP43",
    "BASP1", "MAPT", "APOE", "CLU", "C1QA", "C1QB", "C1QC", "C3", "C4A", "C4B", "CFH",
    "TREM2", "CD44", "ITGB2", "SCG2", "CHGA", "PDYN", "VSTM2A", "NCAN", "MIF",
}

# Markers reported to rise in AD CSF (glycolysis / 14-3-3 / matrisome / glial) and to fall
# (neuronal pentraxins, neurosecretory proteins); their log-ratios (e.g. YWHAG:NPTX2) are among
# the strongest CSF AD signals and are hard for single-feature tree splits to represent.
UP = ["YWHAG", "YWHAZ", "YWHAB", "SMOC1", "PKM", "ALDOA", "ENO1", "GAPDH", "LDHA", "PPIA",
      "SPP1", "CHI3L1", "MIF", "PARK7"]
DOWN = ["NPTX2", "NPTX1", "NPTXR", "VGF", "SCG2", "PDYN"]

# Erythrocyte-specific proteins: their CSF levels track blood contamination at lumbar puncture
# (Geyer / Bader sample-quality marker panel), not disease.
ERY = {
    "HBA1", "HBA2", "HBB", "HBD", "HBG1", "HBG2", "HBQ1", "HBM", "CA1", "CA2", "PRDX2", "CAT",
    "SLC4A1", "SPTA1", "SPTB", "ALAS2", "BLVRB", "EPB42", "ANK1", "GYPA", "HMBS", "SELENBP1",
}


def _gene_match(cols, genes_set):
    out = []
    for c in cols:
        genes = str(c).split("|")[0].split(";")
        if any(g.strip().upper() in genes_set for g in genes):
            out.append(c)
    return out


def _remove_contamination(L):
    det = L.notna().mean()
    ery = _gene_match(L.columns, ERY)
    ery_ok = [c for c in ery if det[c] >= 0.5]
    if len(ery_ok) < 3:
        return L
    E = L[ery_ok]
    Z = (E - E.mean()) / (E.std() + 1e-9)
    s = Z.mean(axis=1, skipna=True)
    s = s.fillna(s.min())
    s = (s - s.mean()).values

    X = L.drop(columns=ery).values
    M = ~np.isnan(X)
    n = M.sum(axis=0).astype(float)
    S = np.where(M, s[:, None], 0.0)
    sm = S.sum(axis=0) / np.maximum(n, 1)
    xm = np.nanmean(np.where(M, X, np.nan), axis=0)
    ds = np.where(M, s[:, None] - sm, 0.0)
    dx = np.where(M, X - xm, 0.0)
    var = (ds ** 2).sum(axis=0)
    cov = (ds * dx).sum(axis=0)
    slope = np.where((n >= 10) & (var > 1e-9), cov / np.maximum(var, 1e-9), 0.0)
    Xr = X - slope[None, :] * s[:, None]  # NaN stays NaN
    return pd.DataFrame(Xr, index=L.index, columns=L.columns.drop(ery))


def featurize(items, ctx):
    L = np.log2(items.astype(float))
    # Reference set: proteins quantified in (almost) every sample, so the per-sample median is
    # not biased by abundance-dependent missingness of low-abundance proteins.
    det = L.notna().mean()
    ref = L.columns[det >= 0.95]
    if len(ref) < 50:
        ref = det.sort_values(ascending=False).index[:max(50, len(det) // 5)]
    L = L.sub(L[ref].median(axis=1), axis=0)
    return _remove_contamination(L)


def _panel_cols(cols):
    return _gene_match(cols, PANEL)


def _fit_score(Xtr, y, Xte, C):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=C, max_iter=5000)
    clf.fit(sc.transform(Xtr), y)
    s_tr = clf.decision_function(sc.transform(Xtr))
    s_te = clf.decision_function(sc.transform(Xte))
    return (s_te - s_tr.mean()) / (s_tr.std() + 1e-9)


def _ztest(s_te):
    # Tree models overfit their own training scores, so standardise on the (label-free) test scores.
    if len(s_te) < 2 or s_te.std() < 1e-9:
        return np.zeros(len(s_te))
    return (s_te - s_te.mean()) / s_te.std()


def _gbm_score(Xtr, y, Xte):
    clf = LGBMClassifier(
        n_estimators=300, learning_rate=0.02, num_leaves=4, max_depth=2,
        min_child_samples=8, subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
        reg_lambda=5.0, random_state=0, n_jobs=1, verbose=-1,
    )
    clf.fit(Xtr, y)
    return _ztest(clf.predict(Xte, raw_score=True))


def _et_score(Xtr, y, Xte):
    clf = ExtraTreesClassifier(
        n_estimators=1000, max_features=0.3, min_samples_leaf=3, bootstrap=False,
        class_weight="balanced", random_state=0, n_jobs=1,
    )
    clf.fit(Xtr, y)
    return _ztest(clf.predict_proba(Xte)[:, 1])


def _ratio_feats(Xtr_df, Xte_df):
    """Label-free log-ratio features between up- and down-in-AD markers (train-fold z-scale)."""
    mu = Xtr_df.mean()
    sd = Xtr_df.std() + 1e-9
    Ztr = (Xtr_df - mu) / sd
    Zte = (Xte_df - mu) / sd

    def collect(genes):
        tr, te = [], []
        for g in genes:
            cols = _gene_match(Xtr_df.columns, {g})
            if cols:
                tr.append(Ztr[cols].mean(axis=1).values)
                te.append(Zte[cols].mean(axis=1).values)
        return tr, te

    up_tr, up_te = collect(UP)
    dn_tr, dn_te = collect(DOWN)
    if len(up_tr) < 2 or len(dn_tr) < 1:
        return None, None
    up_m_tr, up_m_te = np.mean(up_tr, axis=0), np.mean(up_te, axis=0)
    dn_m_tr, dn_m_te = np.mean(dn_tr, axis=0), np.mean(dn_te, axis=0)
    R_tr = [u - dn_m_tr for u in up_tr] + [up_m_tr - d for d in dn_tr] + [up_m_tr - dn_m_tr]
    R_te = [u - dn_m_te for u in up_te] + [up_m_te - d for d in dn_te] + [up_m_te - dn_m_te]
    return np.column_stack(R_tr), np.column_stack(R_te)


def _module_feats(Xtr, Xte, cut=0.6, min_size=5, max_modules=40):
    """Label-free co-expression modules (average-linkage clustering on 1 - r, train fold only);
    each module score is the mean train-fold z-score of its member proteins."""
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0) + 1e-9
    Ztr = (Xtr - mu) / sd
    Zte = (Xte - mu) / sd
    with np.errstate(invalid="ignore", divide="ignore"):
        R = np.corrcoef(Ztr, rowvar=False)
    R = np.nan_to_num(R)
    D = np.clip(1.0 - (R + R.T) / 2.0, 0.0, 2.0)
    np.fill_diagonal(D, 0.0)
    lab = fcluster(linkage(squareform(D, checks=False), method="average"),
                   t=cut, criterion="distance")
    ids, counts = np.unique(lab, return_counts=True)
    order = np.argsort(-counts)
    tr, te = [], []
    for i in order[:max_modules]:
        if counts[i] < min_size:
            break
        m = lab == ids[i]
        tr.append(Ztr[:, m].mean(axis=1))
        te.append(Zte[:, m].mean(axis=1))
    if not tr:
        return None, None
    return np.column_stack(tr), np.column_stack(te)


def _train_sites(ctx, n):
    meta = ctx.get("meta_train") if isinstance(ctx, dict) else None
    if meta is None:
        return np.zeros(n)
    try:
        if isinstance(meta, pd.DataFrame) and "site" in meta.columns:
            s = np.asarray(meta["site"]).astype(str)
        else:
            s = np.asarray(pd.DataFrame(meta)["site"]).astype(str)
        return s if len(s) == n else np.zeros(n)
    except Exception:
        return np.zeros(n)


def _within_site_effect(X, y, sites):
    """Sample-size-weighted mean of within-site Hedges' g (AD minus non-AD) per protein."""
    num = np.zeros(X.shape[1])
    wsum = 0.0
    for s in np.unique(sites):
        m = sites == s
        y_s = y[m]
        n1, n0 = int((y_s == 1).sum()), int((y_s == 0).sum())
        if n1 < 2 or n0 < 2:
            continue
        X1, X0 = X[m][y_s == 1], X[m][y_s == 0]
        sp = np.sqrt(((n1 - 1) * X1.var(axis=0, ddof=1) + (n0 - 1) * X0.var(axis=0, ddof=1))
                     / (n1 + n0 - 2))
        d = (X1.mean(axis=0) - X0.mean(axis=0)) / (sp + 1e-9)
        d *= 1.0 - 3.0 / (4.0 * (n1 + n0) - 9.0)
        w = n1 * n0 / (n1 + n0)
        num += w * d
        wsum += w
    if wsum == 0:
        return None
    return num / wsum


def fit_predict(F_train, y_train, F_test, ctx):
    y = np.asarray(y_train).astype(int)
    keep = F_train.columns[F_train.notna().mean() >= 0.7]  # quantified in >= 70% of train samples
    fill = F_train[keep].min() - 1.0                        # half-minimum on the log2 scale
    Xtr_df = F_train[keep].fillna(fill)
    Xte_df = F_test[keep].fillna(fill)

    scores = [_fit_score(Xtr_df.values, y, Xte_df.values, C=0.05)]

    pcols = _panel_cols(keep)
    if len(pcols) >= 3:
        scores.append(_fit_score(Xtr_df[pcols].values, y, Xte_df[pcols].values, C=0.1))

    # Proteins with a consistent diagnostic effect inside every cohort (not between cohorts).
    sites = _train_sites(ctx, len(y))
    g = _within_site_effect(Xtr_df.values, y, sites)
    idx = np.array([], dtype=int)
    if g is not None:
        k = min(50, len(keep))
        idx = np.argsort(-np.abs(g))[:k]
        scores.append(_fit_score(Xtr_df.values[:, idx], y, Xte_df.values[:, idx], C=0.1))

    # Non-linear, low-variance components on the site-consistent + prior-biology proteins,
    # plus up/down marker log-ratios that single-feature splits cannot form, plus label-free
    # co-expression module scores (pathway-level, less noisy than single proteins).
    pidx = [keep.get_loc(c) for c in pcols]
    gidx = np.unique(np.concatenate([idx, np.asarray(pidx, dtype=int)])).astype(int)
    if len(gidx) >= 5 and len(F_test) >= 2:
        Xg_tr, Xg_te = Xtr_df.values[:, gidx], Xte_df.values[:, gidx]
        R_tr, R_te = _ratio_feats(Xtr_df, Xte_df)
        if R_tr is not None:
            Xg_tr = np.hstack([Xg_tr, R_tr])
            Xg_te = np.hstack([Xg_te, R_te])
        M_tr, M_te = _module_feats(Xtr_df.values, Xte_df.values)
        if M_tr is not None:
            Xg_tr = np.hstack([Xg_tr, M_tr])
            Xg_te = np.hstack([Xg_te, M_te])
        scores.append(_gbm_score(Xg_tr, y, Xg_te))
        scores.append(_et_score(Xg_tr, y, Xg_te))

    return np.mean(scores, axis=0)