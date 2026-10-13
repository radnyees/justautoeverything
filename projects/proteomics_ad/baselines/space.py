def suggest(trial, task):
    return {"min_present": trial.suggest_categorical("min_present", [0.5, 0.7, 0.9]),
            "norm": trial.suggest_categorical("norm", ["none", "median", "quantile"]),
            "batch_center": trial.suggest_categorical("batch_center", [False, True]),
            "impute": trial.suggest_categorical("impute", ["halfmin", "median"]),
            "select_k": trial.suggest_categorical("select_k", [0, 25, 100, 400]),
            "model": trial.suggest_categorical("model", ["lr_l2", "lr_l1", "svm_linear", "rf", "hgb"]),
            "C": trial.suggest_float("C", 1e-3, 10, log=True)}
