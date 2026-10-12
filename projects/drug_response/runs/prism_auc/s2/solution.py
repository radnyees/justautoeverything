"""Multi-scale ensemble of shared multi-output ridges (several variance-filtered gene pools x penalties,
averaged as z-scored predictions) blended with an ensemble of per-drug ridges on drug-specific
correlation-selected genes (averaged over several panel sizes / penalties), then smoothed across drugs
with similar response profiles (kNN in drug-response-correlation space); blend weight / smoothing
strength chosen by grouped inner CV."""
import numpy as np
from sklearn.model_selection import GroupKFold, KFold

POOL = 8000
KS = (50, 100, 250, 500, 1000)
LAMS = (0.5, 2.0, 8.0)       # alpha = lam * k
BASE_POOLS = (2000, 5000, 10000)
BASE_A = (0.5, 1.0, 2.0)     # shared-ridge alpha = a * n_genes (a=1, p=5000 is the old model)
WS = (0.0, 0.33, 0.5, 0.67, 1.0)  # weight on the per-drug filtered ensemble
GS = (0.0, 0.25, 0.5)        # weight on the drug-neighbour-smoothed prediction
NNB = 10                     # neighbours per drug
POW = 2.0                    # similarity sharpening


def _base(Ftr, Y, Fte):
    """Average of z-scored multi-output ridge predictions over gene pools x penalties (dual form)."""
    order = Ftr.var().sort_values(ascending=False).index
    Yf = np.where(np.isnan(Y), np.nanmean(Y, 0), Y)
    Yf = np.nan_to_num(Yf)
    Ym = Yf.mean(0)
    Yc = Yf - Ym
    acc = np.zeros((len(Fte), Y.shape[1]))
    for p in BASE_POOLS:
        genes = order[:min(p, len(order))]
        Xtr, Xte = Ftr[genes].values, Fte[genes].values
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
        Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
        ev, U = np.linalg.eigh(Xtr @ Xtr.T)
        ev = np.clip(ev, 0, None)
        UY = U.T @ Yc
        Kte = Xte @ Xtr.T
        for a in BASE_A:
            P = Kte @ (U @ (UY / (ev + a * len(genes))[:, None])) + Ym
            acc += _z(P)
    return acc


def _filtered(Ftr, Y, Fte):
    """Average of z-scored per-drug ridge predictions over all (k, lam) configurations."""
    genes = Ftr.var().sort_values(ascending=False).index[:POOL]
    Xtr, Xte = Ftr[genes].values, Fte[genes].values
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    D = Y.shape[1]
    out = np.zeros((Xte.shape[0], D))
    for d in range(D):
        obs = ~np.isnan(Y[:, d])
        if obs.sum() < 10:
            continue
        yo = Y[obs, d]
        ym = yo.mean()
        Xo = Xtr[obs]
        xm = Xo.mean(0)
        Xc = Xo - xm
        yc = yo - ym
        r = (Xc.T @ yc) / (np.sqrt((Xc ** 2).sum(0)) * np.sqrt((yc ** 2).sum()) + 1e-8)
        order = np.argsort(-np.abs(r))
        acc = np.zeros(Xte.shape[0])
        for k in KS:
            idx = order[:k]
            A = Xc[:, idx]
            At = Xte[:, idx] - xm[idx]
            ev, U = np.linalg.eigh(A @ A.T)
            ev = np.clip(ev, 0, None)
            Uy = U.T @ yc
            Kte = At @ A.T
            for l in LAMS:
                p = Kte @ (U @ (Uy / (ev + l * k)))
                s = p.std()
                if s > 1e-12:
                    acc += (p - p.mean()) / s
        out[:, d] = acc
    return out


def _neighbours(Y):
    """Row-normalised drug x drug weight matrix from response correlations (training rows only)."""
    D = Y.shape[1]
    Yf = np.where(np.isnan(Y), np.nanmean(Y, 0), Y)
    Yf = np.nan_to_num(Yf)
    Yz = (Yf - Yf.mean(0)) / (Yf.std(0) + 1e-12)
    C = (Yz.T @ Yz) / Yz.shape[0]
    np.fill_diagonal(C, 0.0)
    C = np.clip(C, 0, None) ** POW
    W = np.zeros((D, D))
    for d in range(D):
        nb = np.argsort(-C[d])[:NNB]
        w = C[d, nb]
        s = w.sum()
        if s > 1e-12:
            W[d, nb] = w / s
        else:
            W[d, d] = 1.0
    return W


def _smooth(P, W, g):
    if g == 0.0:
        return P
    return (1 - g) * P + g * (P @ W.T)


def _z(P):
    return (P - P.mean(0)) / (P.std(0) + 1e-12)


def _mean_pearson(P, Y):
    rs = []
    for d in range(Y.shape[1]):
        m = ~np.isnan(Y[:, d])
        if m.sum() < 3:
            continue
        a, b = P[m, d], Y[m, d]
        if a.std() < 1e-12 or b.std() < 1e-12:
            rs.append(0.0)
            continue
        rs.append(np.corrcoef(a, b)[0, 1])
    return float(np.mean(rs)) if rs else 0.0


def fit_predict(F_train, y_train, F_test, ctx):
    Y = y_train.values.astype(float)
    n = len(F_train)
    groups = ctx.get("groups_train", None)
    if groups is not None and len(np.unique(groups)) >= 3:
        splits = list(GroupKFold(n_splits=3).split(np.zeros(n), groups=np.asarray(groups)))
    else:
        splits = list(KFold(n_splits=3, shuffle=True, random_state=0).split(np.zeros(n)))

    # inner CV: collect OOF z-scored predictions
    oof_base = np.zeros_like(Y)
    oof_filt = np.zeros_like(Y)
    fold_W = []
    for tr, te in splits:
        Ftr, Fte = F_train.iloc[tr], F_train.iloc[te]
        oof_base[te] = _z(_base(Ftr, Y[tr], Fte))
        oof_filt[te] = _z(_filtered(Ftr, Y[tr], Fte))
        fold_W.append(_neighbours(Y[tr]))

    best, best_s = (0.0, 0.0), -np.inf
    for w in WS:
        B = (1 - w) * oof_base + w * oof_filt
        for g in GS:
            if g == 0.0:
                Bs = B
            else:
                Bs = np.zeros_like(B)
                for (tr, te), Wf in zip(splits, fold_W):
                    Bs[te] = _smooth(B[te], Wf, g)
            s = _mean_pearson(Bs, Y)
            if s > best_s:
                best_s, best = s, (w, g)
    w, g = best

    Z = _z(_base(F_train, Y, F_test))
    if w > 0.0:
        Pf = _filtered(F_train, Y, F_test)
        Z = (1 - w) * Z + w * _z(Pf)
    if g > 0.0:
        Z = _z(_smooth(Z, _neighbours(Y), g))
    else:
        Z = _z(Z)
    # rescale to the training AUC scale (Pearson is invariant; keeps outputs in AUC units)
    ym = np.nan_to_num(np.nanmean(Y, 0))
    ys = np.nan_to_num(np.nanstd(Y, 0))
    return Z * (0.3 * ys + 1e-6) + ym