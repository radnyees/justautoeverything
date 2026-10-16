def suggest(trial, task):
    return {"n_genes": trial.suggest_categorical("n_genes", [1000, 2500, 5000, 10000]),   # seed 5000
            "standardize": trial.suggest_categorical("standardize", [True, False]),       # seed True
            "model": trial.suggest_categorical("model", ["ridge", "rf"]),                 # seed ridge
            "alpha": trial.suggest_float("alpha", 5000 / 30, 5000 * 30, log=True),        # seed 5000
            "rf_trees": trial.suggest_categorical("rf_trees", [200, 500, 1000]),
            "rf_leaf": trial.suggest_categorical("rf_leaf", [1, 3, 10])}
