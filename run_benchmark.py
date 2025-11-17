# ml-project/run_benchmark.py
"""
공통 러너:
- config.yaml 읽고 데이터 로드
- 전처리 + 샘플러 + 모델 파이프라인 구성
- CV 또는 Holdout 평가
- 최종 성능 그래프 + Threshold Tuning Curve 출력
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
import importlib
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix
)

from imblearn.pipeline import Pipeline

from common import preprocessing as prep_mod


# ---------------------------------------------------------------
# Utils
# ---------------------------------------------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def load_config(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def set_global_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------
# Data
# ---------------------------------------------------------------

def load_data(cfg: Dict[str, Any]) -> tuple[pd.DataFrame, pd.Series]:
    data_cfg = cfg["data"]
    path = data_cfg["file_path"]
    target = data_cfg["target"]

    df = pd.read_excel(path, header=1)

    if target not in df.columns:
        raise ValueError(f"target '{target}' not found")
    return df.drop(columns=[target]), df[target]


# ---------------------------------------------------------------
# Build pipeline components
# ---------------------------------------------------------------

def build_preprocessor(cfg: Dict[str, Any], X: pd.DataFrame):
    prep_mod = importlib.import_module("common.preprocessing")
    prep = prep_mod.make_preprocessor(X)
    build_sampler = getattr(prep_mod, "build_sampler", None)
    sampler = build_sampler(cfg.get("preprocessing", {})) if callable(build_sampler) else None
    return prep, sampler

def build_model(model_cfg: Dict[str, Any]):
    module_name = model_cfg["module"]
    params = model_cfg.get("params", {})
    mod = importlib.import_module(module_name)
    return mod.build_model(**params)


# ---------------------------------------------------------------
# Threshold Tuning Curve
# ---------------------------------------------------------------

def plot_threshold_curve(y_true, y_proba, model_name: str):
    thresholds = np.linspace(0, 1, 301)
    precisions, recalls, f1s = [], [], []

    best_t = 0.5
    best_f1 = -1

    for t in thresholds:
        pred = (y_proba >= t).astype(int)
        p = precision_score(y_true, pred, zero_division=0)
        r = recall_score(y_true, pred, zero_division=0)
        f = f1_score(y_true, pred, zero_division=0)

        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

        if f > best_f1:
            best_f1 = f
            best_t = t

    plt.figure(figsize=(7, 5))
    plt.plot(thresholds, precisions, label="Precision")
    plt.plot(thresholds, recalls, label="Recall")
    plt.plot(thresholds, f1s, label="F1")

    plt.axvline(best_t, color="red", linestyle="--",
                label=f"Best F1 Threshold = {best_t:.3f}")
    plt.scatter([best_t], [best_f1], color="red")

    plt.title(f"Threshold Tuning Curve – {model_name}")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return best_t, best_f1


# ---------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------

def compute_metrics(y_true, y_pred, y_proba=None, threshold=0.5) -> Dict[str, Any]:
    if y_proba is not None and threshold != 0.5:
        y_pred = (y_proba >= threshold).astype(int)

    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
        "pos_rate": float(np.mean(y_true)),
        "threshold": threshold
    }

    if y_proba is not None:
        out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        out["pr_auc"] = float(average_precision_score(y_true, y_proba))

    return out


# ---------------------------------------------------------------
# Holdout
# ---------------------------------------------------------------

def run_holdout(pipe: Pipeline, X, y, eval_cfg):
    test_size = eval_cfg.get("test_size", 0.2)
    seed = eval_cfg.get("random_state", 42)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)
    y_proba = pipe.predict_proba(X_te)[:, 1] if hasattr(pipe, "predict_proba") else None

    metrics = compute_metrics(y_te, y_pred, y_proba)

    # Threshold tuning curve
    if y_proba is not None:
        best_t, best_f1 = plot_threshold_curve(
            y_te, y_proba,
            pipe.named_steps["clf"].__class__.__name__
        )
        metrics["best_threshold"] = best_t
        metrics["best_f1_at_threshold"] = best_f1

    return metrics


# ---------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------

def run_cv(pipe: Pipeline, X, y, eval_cfg):
    n_splits = eval_cfg.get("cv", 5)
    seed = eval_cfg.get("random_state", 42)
    threshold = eval_cfg.get("threshold", 0.5)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    scores = {
        "accuracy": [], "precision": [], "recall": [], "f1": [],
        "roc_auc": [], "pr_auc": []
    }

    last_fold = None

    for tr_idx, te_idx in skf.split(X, y):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        proba = pipe.predict_proba(X_te)[:, 1] if hasattr(pipe, "predict_proba") else None

        m = compute_metrics(y_te, pred, proba, threshold)
        for k in ["accuracy", "precision", "recall", "f1"]:
            scores[k].append(m[k])
        if proba is not None:
            scores["roc_auc"].append(m.get("roc_auc"))
            scores["pr_auc"].append(m.get("pr_auc"))

        last_fold = (y_te, proba)

    agg = {f"{k}_mean": float(np.nanmean(v)) for k, v in scores.items()}
    agg.update({f"{k}_std": float(np.nanstd(v)) for k, v in scores.items()})

    if last_fold and last_fold[1] is not None:
        y_te_last, proba_last = last_fold
        best_t, best_f1 = plot_threshold_curve(
            y_te_last, proba_last,
            pipe.named_steps["clf"].__class__.__name__
        )
        agg["best_threshold"] = best_t
        agg["best_f1_at_threshold"] = best_f1

    return agg


# ---------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------

def visualize_results(results: List[Dict[str, Any]]) -> None:

    def _get(m: Dict[str, Any], key: str):
        return m.get(f"{key}_mean", m.get(key, np.nan))

    rows = []
    for r in results:
        m = r["metrics"]
        rows.append({
            "model": r["model"],
            "accuracy": _get(m, "accuracy"),
            "precision": _get(m, "precision"),
            "recall": _get(m, "recall"),
            "f1": _get(m, "f1"),
            "roc_auc": _get(m, "roc_auc"),
            "pr_auc": _get(m, "pr_auc"),
        })

    df = pd.DataFrame(rows)

    print("\n=== Summary (rounded) ===")
    with pd.option_context("display.max_columns", None, "display.width", 120):
        print(df.round(4).to_string(index=False))

    def _plot_metric(metric: str, title: str):
        sub = df[["model", metric]].dropna()
        if sub.empty:
            return
        sub = sub.sort_values(by=metric, ascending=False)
        plt.figure()
        plt.bar(sub["model"], sub[metric])
        plt.title(title)
        plt.xlabel("Model")
        plt.ylabel(metric)
        plt.xticks(rotation=20)
        plt.tight_layout()

    _plot_metric("roc_auc", "ROC-AUC by Model")
    _plot_metric("pr_auc", "PR-AUC by Model")
    _plot_metric("f1", "F1 Score by Model")

    plt.show()


# ---------------------------------------------------------------
# Runner
# ---------------------------------------------------------------

def main(cfg_path: str):

    cfg = load_config(cfg_path)
    set_global_seed(cfg.get("random_state", 42))

    X, y = load_data(cfg)

    # 2) X, y 같이 클리닝
    df = X.copy()
    df[cfg["data"]["target"]] = y
    df = prep_mod.clean_data(df)

    X = df.drop(columns=[cfg["data"]["target"]])
    y = df[cfg["data"]["target"]]

    prep, sampler = build_preprocessor(cfg, X)

    results = []

    for model_cfg in cfg["models"]:
        name = model_cfg["name"]
        clf = build_model(model_cfg)

        steps = [("prep", prep)]
        if sampler is not None:
            steps.append(("sampler", sampler))
        steps.append(("clf", clf))

        pipe = Pipeline(steps)
        eval_cfg = cfg.get("evaluation", {})

        if eval_cfg.get("mode", "cv") == "cv":
            metrics = run_cv(pipe, X, y, eval_cfg)
        else:
            metrics = run_holdout(pipe, X, y, eval_cfg)

        record = {
            "model": name,
            "module": model_cfg["module"],
            "params": model_cfg.get("params", {}),
            "metrics": metrics
        }
        results.append(record)

        print(f"\n=== {name} ===")
        print(json.dumps(metrics, indent=2, ensure_ascii=False))

    visualize_results(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)