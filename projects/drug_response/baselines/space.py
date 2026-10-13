def suggest(trial, task):
    return {"n_genes": trial.suggest_categorical("n_genes", [1000, 2000, 5000, 10000]),
            "hallmark": trial.suggest_categorical("hallmark", [False, True]),
            "lineage": trial.suggest_categorical("lineage", [False, True]),
            "model": trial.suggest_categorical("model", ["ridge", "pca_ridge", "kernel_ridge"]),
            "alpha": trial.suggest_float("alpha", 10, 1e5, log=True),
            "n_pca": trial.suggest_categorical("n_pca", [50, 100, 200])}
