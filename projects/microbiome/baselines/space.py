def suggest(trial, task):
    return {"level": trial.suggest_categorical("level", ["otu", "genus", "family"]),
            "norm": trial.suggest_categorical("norm", ["relab", "sqrt_relab", "log_relab", "clr"]),
            "prevalence": trial.suggest_categorical("prevalence", [0.0, 0.05, 0.1, 0.2]),
            "study_std": trial.suggest_categorical("study_std", [False, True]) if task == "ibd_pooled" else False,
            "model": trial.suggest_categorical("model", ["rf", "hgb", "lr_l2", "lr_l1"]),
            "n_estimators": trial.suggest_categorical("n_estimators", [300, 1000]),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "C": trial.suggest_float("C", 1e-3, 10, log=True)}
