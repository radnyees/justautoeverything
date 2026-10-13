PARAMS = {'welch': 'median', 'conds': 'first', 'absbp': False, 'ratios': False, 'asym': False, 'aperiodic': True, 'coherence': True, 'hjorth': False, 'entropy': True, 'model': 'rf', 'C': 0.0021967063707060584}
"""Non-LLM search baseline template (PARAMS is prepended by core/search.py)."""
import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]


def _spectra(x, fs):
    n = int(2 * fs)
    x = x - x.mean(axis=1, keepdims=True)
    segs = np.stack([x[:, i:i + n] for i in range(0, x.shape[1] - n + 1, n // 2)])   # seg x ch x n
    if len(segs) < 2:
        segs = np.stack([x[:, :n], x[:, -n:]])
    win = np.hanning(n)
    Z = np.fft.rfft(segs * win, axis=-1)
    f = np.fft.rfftfreq(n, 1 / fs)
    P = np.abs(Z) ** 2
    psd = np.median(P, axis=0) if PARAMS["welch"] == "median" else P.mean(axis=0)
    return f, psd, Z


def _features(x, fs, chans):
    f, psd, Z = _spectra(x, fs)
    out = []
    band = np.stack([psd[:, (f >= a) & (f < b)].sum(axis=1) for a, b in BANDS], axis=1) + 1e-12
    tot = psd[:, (f >= 1) & (f < 45)].sum(axis=1, keepdims=True) + 1e-12
    out.append(np.log(band / tot).ravel())
    if PARAMS["absbp"]:
        out.append(np.log(band).ravel())
    if PARAMS["ratios"]:
        out.append(np.log(band[:, 1] / band[:, 3]))                   # theta/beta
        out.append(np.log(band[:, 2] / band[:, 1]))                   # alpha/theta
    if PARAMS["asym"]:
        pairs = [("F4", "F3"), ("F8", "F7"), ("P4", "P3"), ("O2", "O1")]
        out.append(np.array([np.log(band[chans.index(a), 2]) - np.log(band[chans.index(b), 2])
                             for a, b in pairs if a in chans and b in chans]))
    if PARAMS["aperiodic"]:
        m = (f >= 2) & (f <= 40) & ~((f >= 7) & (f <= 14))
        lf, lp = np.log10(f[m]), np.log10(psd[:, m] + 1e-12)
        out.append(np.polyfit(lf, lp.T, 1)[0])
    if PARAMS["coherence"]:
        S = np.einsum("sif,sjf->ijf", Z, np.conj(Z)) / Z.shape[0]
        d = np.real(np.einsum("iif->if", S))
        coh = np.abs(S) ** 2 / (d[:, None, :] * d[None, :, :] + 1e-12)
        iu = np.triu_indices(len(chans), 1)
        for a, b in [(4, 8), (8, 13), (13, 30)]:
            m = (f >= a) & (f < b)
            out.append(np.arctanh(np.clip(np.sqrt(coh[:, :, m].mean(axis=-1))[iu], 0, 0.999)))
    if PARAMS["hjorth"]:
        d1, d2 = np.diff(x, axis=1), np.diff(x, n=2, axis=1)
        v0, v1, v2 = x.var(axis=1) + 1e-12, d1.var(axis=1) + 1e-12, d2.var(axis=1) + 1e-12
        out.append(np.log(np.sqrt(v1 / v0)))
        out.append(np.log(np.sqrt(v2 / v1) / np.sqrt(v1 / v0)))
    if PARAMS["entropy"]:
        p = psd[:, (f >= 1) & (f < 45)]
        p = p / p.sum(axis=1, keepdims=True)
        out.append(-(p * np.log(p + 1e-12)).sum(axis=1))
    return np.concatenate([np.ravel(o) for o in out])


def featurize(items, ctx):
    conds = ctx["conditions"] if PARAMS["conds"] == "all" else ctx["conditions"][:1]
    fs, chans = ctx["fs"], list(ctx["channels"])
    blocks = {c: ctx["load_eeg"](items, c) for c in conds}

    def one(i):
        fts = []
        for c in conds:
            x = blocks[c][i]
            if x is None:
                x = blocks[conds[0]][i]
            fts.append(_features(np.asarray(x, dtype=np.float64), fs, chans))
        return np.concatenate(fts)
    return np.asarray(Parallel(n_jobs=8)(delayed(one)(i) for i in range(len(items))), dtype=np.float32)


def fit_predict(F_train, y_train, F_test, ctx):
    Xtr, Xte = np.nan_to_num(F_train), np.nan_to_num(F_test)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    m = PARAMS["model"]
    if m == "rf":
        clf = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)
    elif m == "hgb":
        clf = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300, random_state=0)
    elif m == "svm_rbf":
        clf = SVC(C=PARAMS["C"] * 10, gamma="scale").fit(Xtr, y_train)
        return clf.decision_function(Xte)
    else:
        clf = LogisticRegression(C=PARAMS["C"], max_iter=5000)
    clf.fit(Xtr, y_train)
    return clf.predict_proba(Xte)[:, 1]
