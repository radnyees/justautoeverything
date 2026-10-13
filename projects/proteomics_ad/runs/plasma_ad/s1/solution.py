"""log2 + per-sample median centring + per-batch protein median centring
+ label-free removal of erythrocyte/platelet contamination axes (Geyer et al. 2019 quality panels)
+ half-min imputation + train-fold rank-based inverse-normal transform per protein
+ batch-centred missingness indicators (down-weighted block)
+ prior-informed penalty (known AD proteins up-weighted) + L2 LR."""
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Sample-quality marker panels for plasma proteomics (Geyer et al., EMBO Mol Med 2019)
_ERY = ["HBA1", "HBA2", "HBB", "HBD", "CA1", "CA2", "PRDX2", "CAT", "BLVRB",
        "SLC4A1", "SPTA1", "SPTB", "ALAD", "PRDX1", "SOD1", "PGK1"]
_PLT = ["PF4", "PPBP", "TLN1", "MYL6", "ITGA2B", "ITGB3", "GP1BA", "FLNA",
        "TPM4", "VCL", "ACTN1", "THBS1", "MYH9", "CFL1", "GP9", "TUBB1"]

# Proteins with prior evidence for AD in plasma / CSF / brain proteomics
# (APOE/lipid transport, complement, glial/neuronal injury, matrisome module, synaptic, inflammation)
_AD_PRIOR = set([
    "APOE", "CLU", "APOA1", "APOA4", "APOC1", "APOD", "APOJ", "TTR", "A2M",
    "C1QA", "C1QB", "C1QC", "C1R", "C1S", "C3", "C4A", "C4B", "C4BPA", "CFH", "CFB", "CFI", "C5", "C9",
    "NEFL", "NEFM", "GFAP", "YWHAG", "YWHAZ", "MAPT", "APP", "SNAP25", "NRGN",
    "SMOC1", "SPP1", "CHI3L1", "CHIT1", "MDK", "PTN", "NTN1", "SPARCL1", "TREM2",
    "NPTX1", "NPTX2", "VGF", "PKM", "ALDOA", "LDHB", "GDA", "CST3", "IGFBP2",
    "SERPINA3", "CP", "HP", "AMBP", "VCAM1", "ICAM1", "CRP", "PPY", "BDNF",
])
_PRIOR_W = 2.0
_CLIP = 3.0


def _batch_vector(items, ctx):
    meta = ctx.get("meta") if isinstance(ctx, dict) else None
    if meta is None:
        return None
    try:
        if isinstance(meta, pd.DataFrame):
            if "batch" not in meta.columns:
                return None
            if meta.index.equals(items.index):
                return meta["batch"].values
            if len(meta) == len(items):
                return meta["batch"].values
            return meta["batch"].reindex(items.index).values
    except Exception:
        return None
    return None


def _meta_batch(ctx, key, n):
    meta = ctx.get(key) if isinstance(ctx, dict) else None
    try:
        if isinstance(meta, pd.DataFrame) and "batch" in meta.columns and len(meta) == n:
            return meta["batch"].astype(str).values
    except Exception:
        pass
    return None


def _panel_score(L, genes, min_present=0.8):
    gset = set(genes)
    cols = []
    for c in L.columns:
        g = str(c).split("|")[0]
        parts = [p for p in g.split(";") if p]
        if parts and all(p in gset for p in parts):
            cols.append(c)
    if not cols:
        return None
    sub = L[cols]
    sub = sub.loc[:, sub.notna().mean() >= min_present]
    if sub.shape[1] < 2:
        return None
    z = (sub - sub.mean()) / sub.std().replace(0, np.nan)
    s = z.mean(axis=1, skipna=True)
    if s.isna().any():
        s = s.fillna(0.0)
    if s.std() < 1e-8:
        return None
    return ((s - s.mean()) / s.std()).values


def _remove_contamination(L):
    scores = [s for s in (_panel_score(L, _ERY), _panel_score(L, _PLT)) if s is not None]
    if not scores:
        return L
    S = np.column_stack(scores)
    V = L.values.copy()
    for j in range(V.shape[1]):
        obs = ~np.isnan(V[:, j])
        if obs.sum() < 10:
            continue
        Sj = S[obs]
        Sc = Sj - Sj.mean(axis=0)
        y = V[obs, j]
        yc = y - y.mean()
        beta, *_ = np.linalg.lstsq(Sc, yc, rcond=None)
        V[obs, j] = y - Sc @ beta                          # remove slope only, keep level
    return pd.DataFrame(V, index=L.index, columns=L.columns)


def featurize(items, ctx):
    L = np.log2(items.astype(float))
    L = L.sub(L.median(axis=1), axis=0)                    # per-sample median centring
    b = _batch_vector(items, ctx)
    if b is not None:
        b = pd.Series(b, index=L.index).astype(str)
        if b.nunique() > 1:
            # label-free per-batch, per-protein median centring (removes batch offsets)
            for lev in b.unique():
                rows = (b == lev).values
                if rows.sum() < 5:
                    continue
                med = L.loc[rows].median(axis=0)
                L.loc[rows] = L.loc[rows].sub(med.fillna(0.0), axis=1)
    L = _remove_contamination(L)                           # erythrocyte / platelet axes
    return L


def _indicators(F_train, F_test, ctx, lo=0.2, hi=0.8):
    rate = F_train.notna().mean()
    cols = F_train.columns[(rate >= lo) & (rate <= hi)]
    if len(cols) == 0:
        return None, None
    Mtr = F_train[cols].notna().values.astype(float)
    Mte = F_test[cols].notna().values.astype(float)
    btr = _meta_batch(ctx, "meta_train", len(F_train))
    bte = _meta_batch(ctx, "meta_test", len(F_test))
    glob = Mtr.mean(axis=0)
    if btr is not None and bte is not None:
        Ctr = Mtr - glob
        Cte = Mte - glob
        for lev in np.unique(btr):
            r = btr == lev
            if r.sum() < 5:
                continue
            m = Mtr[r].mean(axis=0)                         # train-only per-batch detection rate
            Ctr[r] = Mtr[r] - m
            rt = bte == lev
            if rt.any():
                Cte[rt] = Mte[rt] - m
        return Ctr, Cte
    return Mtr - glob, Mte - glob


def _prior_weights(cols):
    w = np.ones(len(cols))
    for i, c in enumerate(cols):
        g = str(c).split("|")[0]
        parts = [p for p in g.split(";") if p]
        if any(p in _AD_PRIOR for p in parts):
            w[i] = _PRIOR_W                                 # lower effective L2 penalty (1/w^2)
    return w


def _rank_inverse_normal(Xtr, Xte):
    """Per-column inverse-normal transform using the training fold's empirical CDF.
    Mid-rank (ties averaged) mapping; identical formula for train and test values."""
    n = Xtr.shape[0]
    Ztr = np.empty_like(Xtr, dtype=float)
    Zte = np.empty_like(Xte, dtype=float)
    lo, hi = 0.5 / n, 1.0 - 0.5 / n
    for j in range(Xtr.shape[1]):
        xs = np.sort(Xtr[:, j])
        for X, Z in ((Xtr, Ztr), (Xte, Zte)):
            v = X[:, j]
            u = 0.5 * (np.searchsorted(xs, v, side="left") + np.searchsorted(xs, v, side="right")) / n
            Z[:, j] = norm.ppf(np.clip(u, lo, hi))
    return Ztr, Zte


def fit_predict(F_train, y_train, F_test, ctx):
    keep = F_train.columns[F_train.notna().mean() >= 0.7]  # quantified in >= 70% of train samples
    fill = F_train[keep].min() - 1.0                        # half-minimum on the log2 scale
    Xtr = F_train[keep].fillna(fill).values
    Xte = F_test[keep].fillna(fill).values
    Xtr, Xte = _rank_inverse_normal(Xtr, Xte)               # robust Gaussianization (train CDF)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    Xtr = np.clip(Xtr, -_CLIP, _CLIP)
    Xte = np.clip(Xte, -_CLIP, _CLIP)
    pw = _prior_weights(keep)
    Xtr, Xte = Xtr * pw, Xte * pw

    Mtr, Mte = _indicators(F_train, F_test, ctx)
    if Mtr is not None:
        sd = Mtr.std(axis=0)
        ok = sd > 1e-8
        w = 0.5                                             # down-weight the indicator block
        Xtr = np.hstack([Xtr, w * Mtr[:, ok] / sd[ok]])
        Xte = np.hstack([Xte, w * Mte[:, ok] / sd[ok]])

    clf = LogisticRegression(C=0.05, max_iter=5000)
    clf.fit(Xtr, y_train)
    return clf.predict_proba(Xte)[:, 1]