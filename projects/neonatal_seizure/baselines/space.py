def suggest(trial, task):
    return {"montage": trial.suggest_categorical("montage", ["referential", "bipolar"]),
            "win": trial.suggest_categorical("win", [4, 8, 16]),
            "ll": trial.suggest_categorical("ll", [False, True]),
            "rms": trial.suggest_categorical("rms", [False, True]),
            "bp": trial.suggest_categorical("bp", [False, True]),
            "entropy": trial.suggest_categorical("entropy", [False, True]),
            "rhythm": trial.suggest_categorical("rhythm", [False, True]),
            "neonate_norm": trial.suggest_categorical("neonate_norm", [False, True]),
            "model": trial.suggest_categorical("model", ["hgb", "rf", "lr"]),
            "iters": trial.suggest_categorical("iters", [200, 500]),
            "smooth": trial.suggest_categorical("smooth", [1, 5, 11, 21])}
