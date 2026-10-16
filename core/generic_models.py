"""core/generic_models.py -- the ONE generic model list of the seed-derived ("independent") Optuna
baseline (docs/preregistration_phase3.md): L2 logistic regression, random forest, histogram gradient
boosting (classification); ridge and random forest (regression). Same fixed ranges for every task."""


def suggest_classifier(trial, seed_model, seed_c):
    """seed_model: name of the seed's model ('lr' or 'rf' or 'hgb'); seed_c: its C if it is 'lr'."""
    import math
    m = trial.suggest_categorical("model", sorted({seed_model, "lr", "rf", "hgb"}))
    c0 = seed_c if seed_c else 1.0
    return {"model": m,
            "C": trial.suggest_float("C", c0 / 30, c0 * 30, log=True),
            "rf_trees": trial.suggest_categorical("rf_trees", [200, 500, 1000]),
            "rf_leaf": trial.suggest_categorical("rf_leaf", [1, 3, 10]),
            "hgb_lr": trial.suggest_float("hgb_lr", 0.02, 0.3, log=True),
            "hgb_depth": trial.suggest_categorical("hgb_depth", [2, 3, 0]),       # 0 = no limit
            "hgb_iter": trial.suggest_categorical("hgb_iter", [100, 300, 1000])}


def make_classifier(P):
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if P["model"] == "rf":
        return RandomForestClassifier(n_estimators=P["rf_trees"], min_samples_leaf=P["rf_leaf"],
                                      random_state=0, n_jobs=-1)
    if P["model"] == "hgb":
        return HistGradientBoostingClassifier(learning_rate=P["hgb_lr"], max_iter=P["hgb_iter"],
                                              max_depth=P["hgb_depth"] or None, random_state=0)
    return make_pipeline(StandardScaler(), LogisticRegression(C=P["C"], max_iter=5000))
