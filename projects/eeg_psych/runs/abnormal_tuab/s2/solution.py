"""Fine-resolution log spectrum (per channel log absolute power in 2-Hz bins 1-45 Hz + log total power)
plus a fine filter-bank of spatial covariance matrices (9 physiological sub-bands: low/high delta, low/high theta,
low/high alpha, low/high beta, gamma) projected to the Riemannian tangent space (reference = Riemannian mean of
the training fold). Because the data are average-referenced, covariances are rank-deficient along the all-ones
direction; each covariance is therefore projected onto the orthonormal (n-1)-dim complement of the ones vector
(full rank there), and only a tiny ridge is added instead of heavy shrinkage.
Both are estimated only on artifact-free data:
4-s segments (50% overlap) whose channel-averaged log broadband power is a robust outlier (|z| > 3.5, median/MAD)
or that contain clipped samples are rejected (at most half of the segments).
The log spectrum uses the per-bin MEDIAN over kept segments (median Welch) rather than the mean, so residual
transients (eye movements, muscle bursts, arousals) do not dominate the estimate of the background rhythm.
Multi-view (crop) augmentation: features are computed on the full recording and on N_CROPS contiguous crops.
Hemispheric mirror augmentation: abnormality is label-invariant to left/right mirroring, so every view is also
mirrored (homologous 10-20 channels swapped; projected covariances permuted exactly via Q = B^T P B, Q C Q^T).
Training uses all views (grouped by patient in inner CV) so the models learn to ignore features that fluctuate
with brain state within a recording and side of a lesion; test predictions are averaged over all views.
Ensemble: standardized L2 logistic regression (C chosen by grouped inner CV on the training fold) + LightGBM
(shallow, heavily subsampled), combined by rank averaging."""
import numpy as np
from scipy.linalg import null_space
from scipy.signal import spectrogram, butter, sosfiltfilt
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from pyriemann.tangentspace import TangentSpace
import lightgbm as lgb

EDGES = np.arange(1, 46, 2)            # 1,3,...,45 -> 22 bins of 2 Hz
COV_BANDS = [(1, 2.5), (2.5, 4), (4, 6), (6, 8), (8, 10), (10, 13), (13, 20), (20, 30), (30, 45)]
SHRINK = 0.001                         # tiny ridge; rank deficiency is removed by projection
Z_THR = 3.5                            # robust z threshold for segment rejection
CLIP_UV = 790.0                        # data are clipped at +/-800 uV
MAX_REJECT = 0.5
N_CROPS = 3                            # contiguous crops per recording (in addition to the full view)
N_VIEWS = N_CROPS + 1
MIRROR_PAIRS = [("FP1", "FP2"), ("F7", "F8"), ("F3", "F4"), ("C3", "C4"), ("P3", "P4"), ("O1", "O2"),
                ("T3", "T4"), ("T5", "T6"), ("T7", "T8"), ("P7", "P8"), ("A1", "A2")]


def _basis(n):
    """Orthonormal basis (n x n-1) of the complement of the all-ones (average-reference null) direction."""
    return null_space(np.ones((1, n)))


def _segments(x, fs):
    """Welch-style segment PSDs + boolean mask of artifact-free segments."""
    L = int(4 * fs)
    step = int(2 * fs)
    f, _, P = spectrogram(x, fs=fs, window="hann", nperseg=L, noverlap=L - step,
                          detrend="constant", scaling="density", mode="psd")   # (ch, f, seg)
    nseg = P.shape[-1]
    starts = np.arange(nseg) * step
    keep = np.ones(nseg, dtype=bool)
    if nseg >= 4:
        band = (f >= 1) & (f <= 45)
        lp = np.log(P[:, band, :].sum(axis=1) + 1e-12).mean(axis=0)
        med = np.median(lp)
        mad = 1.4826 * np.median(np.abs(lp - med)) + 1e-6
        z = np.abs(lp - med) / mad
        clip = np.array([np.abs(x[:, s:s + L]).max() >= CLIP_UV for s in starts])
        score = z + 100.0 * clip
        bad = score > Z_THR
        max_bad = int(np.floor(MAX_REJECT * nseg))
        if bad.sum() > max_bad:                 # keep the cleanest half
            order = np.argsort(score)
            bad = np.ones(nseg, dtype=bool)
            bad[order[:nseg - max_bad]] = False
        keep = ~bad
    return f, P, starts, L, keep


def _logspec(f, P, keep):
    p = P[:, :, keep]                                     # (ch, f, seg)
    bins = [np.median(p[:, (f >= a) & (f < b), :].mean(axis=1), axis=-1)
            for a, b in zip(EDGES[:-1], EDGES[1:])]
    tot = np.median(p[:, (f >= 1) & (f <= 45), :].sum(axis=1), axis=-1)
    return np.log(np.column_stack(bins + [tot]) + 1e-12)


def _filterbank(x, fs):
    out = []
    for lo, hi in COV_BANDS:
        hi = min(hi, 0.49 * fs)
        sos = butter(4, [lo, hi], btype="band", fs=fs, output="sos")
        out.append(sosfiltfilt(sos, x, axis=-1))
    return out


def _covs(filt, mask, B):
    k = B.shape[1]
    iu = np.triu_indices(k)
    out = []
    for xb in filt:
        xf = xb[:, mask]
        xf = xf - xf.mean(axis=1, keepdims=True)
        c = xf @ xf.T / xf.shape[1]
        c = B.T @ c @ B                                   # full-rank (n-1)-dim subspace
        c = 0.5 * (c + c.T)
        c = (1 - SHRINK) * c + SHRINK * np.trace(c) / k * np.eye(k) + 1e-10 * np.eye(k)
        out.append(c[iu])
    return np.concatenate(out)


def _views(x, fs, B):
    """List of N_VIEWS (spec, cov) feature pairs: full recording + contiguous crops."""
    f, P, starts, L, keep = _segments(x, fs)
    T = x.shape[1]
    clean = np.ones(T, dtype=bool)
    for s in starts[~keep]:
        clean[s:s + L] = False
    if clean.sum() < L:
        clean[:] = True
    filt = _filterbank(x, fs)
    views = [(_logspec(f, P, keep).ravel(), _covs(filt, clean, B))]
    for k in range(N_CROPS):
        lo, hi = k * T // N_CROPS, (k + 1) * T // N_CROPS
        segsel = (starts >= lo) & (starts + L <= hi)
        seg_k = keep & segsel
        if seg_k.sum() == 0:
            seg_k = segsel
        if seg_k.sum() == 0:
            seg_k = keep
        m = clean.copy()
        m[:lo] = False
        m[hi:] = False
        if m.sum() < L:
            m = np.zeros(T, dtype=bool)
            m[lo:hi] = True
        if m.sum() < L:
            m = clean
        views.append((_logspec(f, P, seg_k).ravel(), _covs(filt, m, B)))
    return views


def featurize(items, ctx):
    fs = ctx["fs"]
    B = _basis(len(ctx["channels"]))
    feats = []
    blocks = {c: ctx["load_eeg"](items, c) for c in ctx["conditions"]}
    for i in range(len(items)):
        per_cond = []
        for c in ctx["conditions"]:
            x = blocks[c][i]
            if x is None:                      # missing condition -> fall back to first condition
                x = blocks[ctx["conditions"][0]][i]
            per_cond.append(_views(np.asarray(x, dtype=np.float64), fs, B))
        vec = []
        for v in range(N_VIEWS):              # each view: [spec per cond] + [cov per cond]
            vec.extend([pc[v][0] for pc in per_cond])
            vec.extend([pc[v][1] for pc in per_cond])
        feats.append(np.concatenate(vec))
    return np.asarray(feats, dtype=np.float64)


def _mirror_perm(channels):
    """Index permutation swapping homologous left/right 10-20 channels (midline maps to itself)."""
    def norm(c):
        c = str(c).upper().replace("EEG", "").strip()
        return c.split("-")[0].strip()
    names = [norm(c) for c in channels]
    idx = {c: i for i, c in enumerate(names)}
    perm = np.arange(len(names))
    for a, b in MIRROR_PAIRS:
        if a in idx and b in idx:
            perm[idx[a]], perm[idx[b]] = idx[b], idx[a]
    return perm


def _to_mats(V, k, iu):
    M = np.zeros((len(V), k, k))
    M[:, iu[0], iu[1]] = V
    M = M + np.transpose(M, (0, 2, 1))
    M[:, np.arange(k), np.arange(k)] /= 2
    return M


def _mirror(R, perm, n, n_spec):
    """Mirror feature rows: permute channel spectra and transform projected covariances by Q = B^T P B."""
    out = R.copy()
    nb_spec = len(EDGES)
    S = R[:, :n_spec].reshape(len(R), -1, n, nb_spec)[:, :, perm, :]
    out[:, :n_spec] = S.reshape(len(R), -1)
    B = _basis(n)
    P = np.eye(n)[perm]
    Q = B.T @ P @ B
    k = n - 1
    iu = np.triu_indices(k)
    m = len(iu[0])
    C = R[:, n_spec:]
    for b in range(C.shape[1] // m):
        M = _to_mats(C[:, b * m:(b + 1) * m], k, iu)
        M2 = Q[None] @ M @ Q.T[None]
        out[:, n_spec + b * m:n_spec + (b + 1) * m] = M2[:, iu[0], iu[1]]
    return out


def _tangent(Ctr, Cte, n):
    """Ctr/Cte: (rows, n_bands_total * n(n+1)/2) upper triangles -> tangent-space features."""
    iu = np.triu_indices(n)
    m = len(iu[0])
    nb = Ctr.shape[1] // m
    tr, te = [], []
    for b in range(nb):
        Mtr = _to_mats(Ctr[:, b * m:(b + 1) * m], n, iu)
        Mte = _to_mats(Cte[:, b * m:(b + 1) * m], n, iu)
        ts = TangentSpace(metric="riemann").fit(Mtr)
        tr.append(ts.transform(Mtr))
        te.append(ts.transform(Mte))
    return np.hstack(tr), np.hstack(te)


def _rank01(p):
    return (rankdata(p) - 1) / max(len(p) - 1, 1)


def fit_predict(F_train, y_train, F_test, ctx):
    F_train = np.asarray(F_train, dtype=np.float64)
    F_test = np.asarray(F_test, dtype=np.float64)
    y_train = np.asarray(y_train)
    n_tr, n_te = len(F_train), len(F_test)
    D = F_train.shape[1] // N_VIEWS
    n = len(ctx["channels"])
    n_cond = len(ctx["conditions"])
    n_spec = n_cond * n * len(EDGES)        # 22 bins + total = len(EDGES) per channel

    Rtr = F_train.reshape(n_tr * N_VIEWS, D)       # item-major: views of an item are consecutive
    Rte = F_test.reshape(n_te * N_VIEWS, D)
    perm = _mirror_perm(ctx["channels"])
    if np.any(perm != np.arange(n)):
        Mtr = _mirror(Rtr, perm, n, n_spec)
        Mte = _mirror(Rte, perm, n, n_spec)
        Rtr = np.concatenate([Rtr.reshape(n_tr, N_VIEWS, D), Mtr.reshape(n_tr, N_VIEWS, D)],
                             axis=1).reshape(-1, D)
        Rte = np.concatenate([Rte.reshape(n_te, N_VIEWS, D), Mte.reshape(n_te, N_VIEWS, D)],
                             axis=1).reshape(-1, D)
        nv = 2 * N_VIEWS
    else:
        nv = N_VIEWS
    y_rep = np.repeat(y_train, nv)

    Ttr, Tte = _tangent(Rtr[:, n_spec:], Rte[:, n_spec:], n - 1)
    Xtr = np.hstack([Rtr[:, :n_spec], Ttr])
    Xte = np.hstack([Rte[:, :n_spec], Tte])
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

    groups = ctx.get("groups_train")
    if groups is None:
        groups = np.arange(n_tr)
    groups_rep = np.repeat(np.asarray(groups), nv)
    cv = list(GroupKFold(n_splits=5).split(Xtr, y_rep, groups_rep))
    clf = LogisticRegressionCV(Cs=np.logspace(-4, 0, 9), cv=cv, scoring="roc_auc",
                               penalty="l2", max_iter=5000, n_jobs=-1).fit(Xtr, y_rep)
    p_lr = clf.decision_function(Xte).reshape(n_te, nv).mean(axis=1)

    gbm = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=15, max_depth=5,
                             min_child_samples=20 * nv, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.3, reg_lambda=1.0, random_state=0,
                             deterministic=True, force_row_wise=True, n_jobs=4, verbose=-1)
    gbm.fit(Xtr, y_rep)
    p_gb = gbm.predict_proba(Xte)[:, 1].reshape(n_te, nv).mean(axis=1)

    return 0.5 * _rank01(p_lr) + 0.5 * _rank01(p_gb)