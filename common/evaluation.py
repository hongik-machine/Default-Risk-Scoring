# evaluation - 평가 코드
# AUC, Recall, ACC, F-1 Score

"""Evaluation utilities shared by all models.
Provides:
- build_cv: StratifiedKFold factory
- cross_validate_model: run CV with unified scoring (ROC-AUC, PR-AUC, F1, Recall, Precision, ACC)
- evaluate_holdout: fit on train and evaluate on a holdout set; returns metrics and predicted probabilities
"""
from __future__ import annotations
from typing import Dict, Any
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    recall_score, precision_score, accuracy_score
)

DEFAULT_SCORING = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "f1": "f1",
    "recall": "recall",
    "precision": "precision",
    "accuracy": "accuracy",
}


def build_cv(n_splits: int = 5, seed: int = 42) -> StratifiedKFold:
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def cross_validate_model(pipe, X: pd.DataFrame, y: np.ndarray, cv: StratifiedKFold | None = None,
                          scoring: Dict[str, Any] | None = None, n_jobs: int = -1) -> pd.Series:
    cv = cv or build_cv()
    scoring = scoring or DEFAULT_SCORING
    cv_res = cross_validate(pipe, X, y, cv=cv, scoring=scoring, return_train_score=False, n_jobs=n_jobs)
    out = {k.replace("test_", ""): float(np.mean(v)) for k, v in cv_res.items() if k.startswith("test_")}
    return pd.Series(out)


def evaluate_holdout(pipe, X_train, y_train, X_test, y_test) -> Dict[str, float]:
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None
    pred = pipe.predict(X_test)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, proba)) if proba is not None else np.nan,
        "pr_auc": float(average_precision_score(y_test, proba)) if proba is not None else np.nan,
        "f1": float(f1_score(y_test, pred)),
        "recall": float(recall_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred)),
        "accuracy": float(accuracy_score(y_test, pred)),
    }
    return metrics