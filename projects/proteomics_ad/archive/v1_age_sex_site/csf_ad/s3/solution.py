"""log2 + per-sample median centring on a fixed, completely-quantified reference set
+ label-free removal of sample-quality contamination axes (erythrocyte, platelet,
coagulation; Geyer et al. 2019 quality-marker panels) + half-minimum imputation.
Ensemble: whole-proteome L2 logistic regression + prior-biology CSF AD panel L2 logistic regression
with coefficient signs constrained to the a-priori direction of change in AD
(Bader 2020 panel extended with the Higginbotham/Johnson 2020 CSF glycolysis and
neuronal-secretory/synaptic modules) + two fixed-direction log-ratio scores:
14-3-3/SMOC1 vs neuronal pentraxins (Oh et al. 2025 YWHAG:NPTX2) and the glycolysis
module vs the neuroendocrine secretory-granule module (Higginbotham/Johnson 2020)."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Prior-knowledge CSF AD panel (tau/14-3-3/SMOC1, neuroaxonal, glial/inflammation,
# synaptic, and the glycolysis/oxidative-stress proteins reported by Bader et al. 2020)
PANEL = {
    "MAPT", "SMOC1", "YWHAG", "YWHAZ", "YWHAB", "YWHAE", "YWHAQ", "NEFL", "NEFM",
    "NRGN", "SPP1", "CHI3L1", "CHI3L2", "CHIT1", "GFAP", "TREM2", "NPTX2", "VGF",
    "GAP43", "SNAP25", "PKM", "ALDOA", "ENO1", "ENO2", "MDH1", "GDA", "SOD1",
    "PARK7", "PEBP1", "LDHB", "GOT1", "PPIA", "FABP3",
}
# Higginbotham et al. 2020 / Johnson et al. 2020 CSF modules: glycolytic enzymes
# elevated in AD, and neuronal pentraxin / secretory-granule proteins reduced in AD
PANEL |= {
    "LDHA", "GAPDH", "PGK1", "TPI1", "PGAM1", "MIF",
    "NPTX1", "NPTXR", "SCG2", "PCSK1", "PCSK1N", "CHGA", "CHGB",
}

# A-priori direction of change in AD CSF for the panel proteins (used as sign constraints).
# Decreased: neuronal pentraxins, VGF and neuroendocrine secretory-granule proteins.
PANEL_DOWN = {"NPTX2", "NPTX1", "NPTXR", "VGF", "SCG2", "PCSK1", "PCSK1N", "CHGA", "CHGB"}
# Increased: tau, 14-3-3, SMOC1, neuroaxonal/synaptic leakage, glial activation, glycolysis.
PANEL_UP = PANEL - PANEL_DOWN

# Synaptic-stress ratio (Oh et al. 2025, CSF YWHAG:NPTX2): numerator proteins rise in AD,
# denominator synaptic pentraxins / VGF fall in AD. Directions fixed a priori.
RATIO_UP = {"YWHAG", "SMOC1"}
RATIO_DOWN = {"NPTX2", "NPTX1", "NPTXR", "VGF"}

# Metabolic-vs-secretory ratio (Higginbotham 2020 / Johnson 2020 CSF modules): glycolytic
# enzymes rise in AD CSF, neuroendocrine secretory-granule proteins fall. Directions fixed a priori.
GLYCO_UP = {"PKM", "ALDOA", "ENO1", "GAPDH", "LDHA", "PGK1", "TPI1", "PGAM1"}
SECRETORY_DOWN = {"SCG2", "PCSK1", "PCSK1N", "CHGA", "CHGB"}

# Erythrocyte-specific proteins: markers of blood contamination at lumbar puncture
RBC = {"HBA1", "HBA2", "HBB", "HBD", "HBG1", "HBG2", "CA1", "PRDX2", "CAT",
       "SLC4A1", "ALAS2", "BLVRB", "SPTA1", "SPTB"}

# Platelet-specific proteins: platelet contamination / activation during sampling
PLATELET = {"PF4", "PPBP", "TLN1", "MYL9", "TPM4", "ITGA2B", "ITGB3", "GP1BA", "GP9",
            "VCL", "FLNA", "TAGLN2", "ZYX", "ACTN1", "PFN1", "THBS1", "F13A1", "GP5"}

# Fibrinogen chains: degree of (partial) coagulation of the sample
COAG = {"FGA", "FGB", "FGG"}


def _gene_set(col):
    gene = str(col).split("|")[0]
    return {g.strip().upper() for g in gene.split(";") if g.strip()}


def _remove_contamination(N, markers):
    """Regress a per-sample contamination score (from a marker panel) out of proteins that track it."""
    det = N.notna().mean(axis=0)
    mcols = [c for c in N.columns if _gene_set(c) & markers and det[c] >= 0.8]
    if len(mcols) < 2:
        return N
    R = N[mcols]
    R = R.fillna(R.min() - 1.0)
    Z = (R - R.mean()) / (R.std(ddof=0) + 1e-9)
    s = Z.mean(axis=1).values
    s = (s - s.mean()) / (s.std() + 1e-9)
    out = N.copy()
    for c in N.columns:
        v = N[c].values
        m = ~np.isnan(v)
        if m.sum() < 20:
            continue
        sv, vv = s[m], v[m]
        sc = sv - sv.mean()
        denom = (sc ** 2).sum()
        if denom <= 0:
            continue
        vc = vv - vv.mean()
        r = (sc * vc).sum() / np.sqrt(denom * (vc ** 2).sum() + 1e-12)
        if abs(r) < 0.2:
            continue
        beta = (sc * vc).sum() / denom
        adj = v.copy()
        adj[m] = vv - beta * sc
        out[c] = adj
    return out


def featurize(items, ctx):
    L = np.log2(items.astype(float))
    # Missingness is abundance-dependent, so a per-sample median over "whatever was
    # quantified" is biased upward in samples with more missing low-abundance proteins.
    # Normalise on a fixed reference set quantified in (nearly) every sample instead.
    det = L.notna().mean(axis=0)
    ref = L.columns[det >= 1.0]
    if len(ref) < 50:
        ref = L.columns[det >= 0.9]
    if len(ref) < 50:
        ref = L.columns
    N = L.sub(L[ref].median(axis=1), axis=0)
    # Sequentially remove pre-analytical quality axes (erythrocyte, platelet, coagulation)
    for markers in (RBC, PLATELET, COAG):
        N = _remove_contamination(N, markers)
    return N


def _fit_lr(F_train, y_train, F_test, cols, C):
    fill = F_train[cols].min() - 1.0                        # half-minimum on the log2 scale
    Xtr = F_train[cols].fillna(fill).values
    Xte = F_test[cols].fillna(fill).values
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=C, max_iter=5000)
    clf.fit(sc.transform(Xtr), y_train)
    return clf.predict_proba(sc.transform(Xte))[:, 1]


def _fit_signed_lr(F_train, y_train, F_test, cols, C):
    """L2 logistic regression (sklearn C convention) whose coefficients are bounded to the
    a-priori AD direction of each protein (>= 0 for AD-elevated, <= 0 for AD-reduced)."""
    fill = F_train[cols].min() - 1.0                        # half-minimum on the log2 scale
    sc = StandardScaler().fit(F_train[cols].fillna(fill).values)
    X = sc.transform(F_train[cols].fillna(fill).values)
    Xte = sc.transform(F_test[cols].fillna(fill).values)
    y = np.asarray(y_train, dtype=float)
    n, p = X.shape

    bounds = []
    for c in cols:
        gs = _gene_set(c)
        up, down = bool(gs & PANEL_UP), bool(gs & PANEL_DOWN)
        if up and not down:
            bounds.append((0.0, None))
        elif down and not up:
            bounds.append((None, 0.0))
        else:
            bounds.append((None, None))
    bounds.append((None, None))                             # unpenalised, unbounded intercept

    def obj(theta):
        w, b = theta[:p], theta[p]
        z = X @ w + b
        loss = np.sum(np.logaddexp(0.0, z) - y * z)
        g = expit(z) - y
        val = C * loss + 0.5 * w @ w
        grad = np.concatenate([C * (X.T @ g) + w, [C * g.sum()]])
        return val, grad

    res = minimize(obj, np.zeros(p + 1), jac=True, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 5000})
    w, b = res.x[:p], res.x[p]
    return expit(Xte @ w + b)


def _ratio_score(F_train, F_test, up, down):
    """Mean z of a priori AD-elevated proteins minus mean z of AD-reduced proteins."""
    def zmean(markers):
        cols = [c for c in F_train.columns
                if _gene_set(c) & markers and F_train[c].notna().mean() >= 0.5]
        if not cols:
            return None, None
        fill = F_train[cols].min() - 1.0
        Xtr = F_train[cols].fillna(fill)
        Xte = F_test[cols].fillna(fill)
        mu, sd = Xtr.mean(), Xtr.std(ddof=0) + 1e-9
        return ((Xtr - mu) / sd).mean(axis=1).values, ((Xte - mu) / sd).mean(axis=1).values
    up_tr, up_te = zmean(up)
    dn_tr, dn_te = zmean(down)
    if up_tr is None or dn_tr is None:
        return None, None
    return up_tr - dn_tr, up_te - dn_te


def _ratio_member(F_train, y_train, F_test, up, down):
    s_tr, s_te = _ratio_score(F_train, F_test, up, down)
    if s_tr is None:
        return None
    # Direction is fixed by biology; the 1-D logistic fit only calibrates scale/offset.
    clf = LogisticRegression(C=1.0, max_iter=5000)
    clf.fit(s_tr.reshape(-1, 1), y_train)
    if clf.coef_[0, 0] > 0:
        return clf.predict_proba(s_te.reshape(-1, 1))[:, 1]
    # never flip the prior direction; fall back to a monotone squashing of the score
    return 1.0 / (1.0 + np.exp(-s_te))


def fit_predict(F_train, y_train, F_test, ctx):
    y_train = np.asarray(y_train)
    keep = F_train.columns[F_train.notna().mean() >= 0.7]  # quantified in >= 70% of train samples
    p_all = _fit_lr(F_train, y_train, F_test, keep, C=0.05)

    panel_cols = [c for c in F_train.columns
                  if _gene_set(c) & PANEL and F_train[c].notna().mean() >= 0.5]
    if len(panel_cols) < 3:
        return p_all
    members = [p_all, _fit_signed_lr(F_train, y_train, F_test, panel_cols, C=0.1)]

    for up, down in ((RATIO_UP, RATIO_DOWN), (GLYCO_UP, SECRETORY_DOWN)):
        p = _ratio_member(F_train, y_train, F_test, up, down)
        if p is not None:
            members.append(p)
    return np.mean(members, axis=0)