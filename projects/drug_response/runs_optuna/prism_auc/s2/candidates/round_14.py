PARAMS = {'n_genes': 5000, 'hallmark': False, 'lineage': False, 'model': 'pca_ridge', 'alpha': 2681.6372641138983, 'n_pca': 50}
"""Non-LLM search baseline template (PARAMS is prepended by core/search.py)."""
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def _hallmark(ctx):
    sets = {}
    for line in open(os.path.join(ctx["genesets_dir"], "hallmark.gmt")):
        p = line.rstrip("\n").split("\t")
        sets[p[0]] = p[2:]
    return sets


def fit_predict(F_train, y_train, F_test, ctx):
    genes = F_train.var().sort_values(ascending=False).index[:PARAMS["n_genes"]]
    sc = StandardScaler().fit(F_train[genes].values)
    Xtr, Xte = sc.transform(F_train[genes].values), sc.transform(F_test[genes].values)
    blocks_tr, blocks_te = [Xtr], [Xte]
    if PARAMS["hallmark"]:
        mu, sd = F_train.mean(), F_train.std() + 1e-6
        Ztr, Zte = (F_train - mu) / sd, (F_test - mu) / sd
        hs = [[g for g in v if g in F_train.columns] for v in _hallmark(ctx).values()]
        hs = [h for h in hs if len(h) >= 5]
        blocks_tr.append(np.column_stack([Ztr[h].mean(axis=1) for h in hs]))
        blocks_te.append(np.column_stack([Zte[h].mean(axis=1) for h in hs]))
    if PARAMS["lineage"]:
        lt = pd.get_dummies(ctx["meta_train"]["OncotreeLineage"].astype(str))
        le = pd.get_dummies(ctx["meta_test"]["OncotreeLineage"].astype(str)).reindex(columns=lt.columns, fill_value=0)
        blocks_tr.append(lt.values * 3.0)
        blocks_te.append(le.values * 3.0)
    Xtr, Xte = np.hstack(blocks_tr), np.hstack(blocks_te)
    Y = y_train.fillna(y_train.mean()).values
    m = PARAMS["model"]
    if m == "pca_ridge":
        pca = PCA(n_components=min(PARAMS["n_pca"], Xtr.shape[0] - 1), random_state=0).fit(Xtr)
        Xtr, Xte = pca.transform(Xtr), pca.transform(Xte)
        model = Ridge(alpha=PARAMS["alpha"])
    elif m == "kernel_ridge":
        model = KernelRidge(alpha=PARAMS["alpha"] / 1000, kernel="rbf", gamma=1.0 / Xtr.shape[1])
    else:
        model = Ridge(alpha=PARAMS["alpha"])
    return model.fit(Xtr, Y).predict(Xte)
