from sklearn.linear_model import LogisticRegression

def build_estimator(*, C=1.0, penalty="l2", max_iter=1000, class_weight=None, random_state=42):
    if penalty not in {"l1", "l2"}:
        raise ValueError("penalty must be 'l1' or 'l2'")
    solver = "lbfgs" if penalty == "l2" else "saga"
    return LogisticRegression(
        C=C, penalty=penalty, solver=solver,
        max_iter=max_iter, class_weight=class_weight,
        random_state=random_state,
    )