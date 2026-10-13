def suggest(trial, task):
    return {"welch": trial.suggest_categorical("welch", ["mean", "median"]),
            "conds": trial.suggest_categorical("conds", ["first", "all"]),
            "absbp": trial.suggest_categorical("absbp", [False, True]),
            "ratios": trial.suggest_categorical("ratios", [False, True]),
            "asym": trial.suggest_categorical("asym", [False, True]),
            "aperiodic": trial.suggest_categorical("aperiodic", [False, True]),
            "coherence": trial.suggest_categorical("coherence", [False, True]),
            "hjorth": trial.suggest_categorical("hjorth", [False, True]),
            "entropy": trial.suggest_categorical("entropy", [False, True]),
            "model": trial.suggest_categorical("model", ["lr_l2", "svm_rbf", "rf", "hgb"]),
            "C": trial.suggest_float("C", 1e-3, 10, log=True)}
