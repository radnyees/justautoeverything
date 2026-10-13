def suggest(trial, task):
    return {"atlas": trial.suggest_categorical("atlas", ["aal", "ho", "cc200", "cc400"]),
            "strategy": trial.suggest_categorical("strategy", [0, 1]),
            "conn": trial.suggest_categorical("conn", ["corr", "partial", "tangent"]),
            "site_std": trial.suggest_categorical("site_std", [False, True]),
            "model": trial.suggest_categorical("model", ["lr_l2", "svm_linear", "ridge"]),
            "C": trial.suggest_float("C", 1e-3, 10, log=True)}
