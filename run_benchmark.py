# ml-project/run_benchmark.py
"""
공통 러너:
- config.yaml 읽어서 data 경로, 모델 리스트, 평가 방식 등 불러오기 - yaml을 바꾸면 여러 실험 가능
- 전처리 파이프라인(prep) + (선택) 샘플러(sampler) + 모델(clf)로 imblearn.Pipeline 구성
- CV 혹은 holdout 선택 평가
- 결과와 모델을 artifacts/ 아래 저장

1. config.yaml 읽기
2. preprocessing.py 호출 -> 전처리
3. models 폴더의 모델 생성
4. imblearn.Pipeline 구성 - [(prep), (sampler), (clf)] 형태로 자동 결합
5. 학습 + 평가 -> run_cv() 또는 run_holdout()을 통해 모델별로 자동 평가
6. 결과 저장 - 결과를 metrics 딕셔너리로 수집 → results.json, results.csv로 저장
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

from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix
)

# imblearn Pipeline (sampler 포함 가능)
from imblearn.pipeline import Pipeline


# -----------------------------
# Utils
# -----------------------------
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
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)


# -----------------------------
# Data
# -----------------------------
def load_data(cfg: Dict[str, Any]) -> tuple[pd.DataFrame, pd.Series]:
    data_cfg = cfg["data"]
    path = data_cfg["csv_path"]
    target = data_cfg["target"]
    df = pd.read_csv(path)
    if target not in df.columns:
        raise ValueError(f"target '{target}' not in columns: {list(df.columns)[:10]}...")
    y = df[target]
    X = df.drop(columns=[target])
    return X, y


# -----------------------------
# Build pipeline components
# -----------------------------
def build_preprocessor(cfg: Dict[str, Any]):
    """
    common/preprocessing.py 에 다음 함수가 있다고 가정:
      - build_preprocessor(cfg) -> sklearn/ColumnTransformer or Pipeline
      - build_sampler(cfg) -> imblearn sampler or None (예: SMOTE/None)

    프로젝트 구현체와 이름만 맞추면 됨.
    """
    prep_mod = importlib.import_module("common.preprocessing")
    prep = prep_mod.build_preprocessor(cfg.get("preprocessing", {}))
    build_sampler = getattr(prep_mod, "build_sampler", None)
    sampler = build_sampler(cfg.get("preprocessing", {})) if callable(build_sampler) else None
    return prep, sampler


def build_model(model_cfg: Dict[str, Any]):
    """
    models/*.py 파일은 반드시 build_model(params: dict | **kwargs) -> estimator 를 제공한다고 가정.
    예: models/logistic.py의 build_model(C=1.0, penalty="l2", ...)
    """
    module_name = model_cfg["module"]            # "models.logistic" 같은 경로
    params = model_cfg.get("params", {})         # dict
    mod = importlib.import_module(module_name)
    return mod.build_model(**params)


# -----------------------------
# Metrics helpers
# -----------------------------
def compute_metrics(y_true, y_pred, y_proba=None) -> Dict[str, Any]:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred),
        "pos_rate": float(np.mean(y_true)),
    }
    if y_proba is not None:
        out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        out["pr_auc"] = float(average_precision_score(y_true, y_proba))
    return out


# -----------------------------
# Train & Evaluate (Holdout)
# -----------------------------
def run_holdout(pipe: Pipeline, X, y, eval_cfg: Dict[str, Any]) -> Dict[str, Any]:
    test_size = eval_cfg.get("test_size", 0.2)
    seed = eval_cfg.get("random_state", 42)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)
    y_proba = pipe.predict_proba(X_te)[:, 1] if hasattr(pipe, "predict_proba") else None
    return compute_metrics(y_te, y_pred, y_proba)


# -----------------------------
# Train & Evaluate (CV)
# -----------------------------
def run_cv(pipe: Pipeline, X, y, eval_cfg: Dict[str, Any]) -> Dict[str, Any]:
    n_splits = eval_cfg.get("cv", 5)
    seed = eval_cfg.get("random_state", 42)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # cross_validate로 기본 점수 + 수작업 proba 메트릭까지 산출
    scores = {
        "accuracy": [], "precision": [], "recall": [], "f1": [],
        "roc_auc": [], "pr_auc": []
    }

    for tr_idx, te_idx in skf.split(X, y):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)
        y_proba = pipe.predict_proba(X_te)[:, 1] if hasattr(pipe, "predict_proba") else None

        m = compute_metrics(y_te, y_pred, y_proba)
        for k in ["accuracy", "precision", "recall", "f1"]:
            scores[k].append(m[k])
        if y_proba is not None:
            scores["roc_auc"].append(m.get("roc_auc", np.nan))
            scores["pr_auc"].append(m.get("pr_auc", np.nan))

    agg = {f"{k}_mean": float(np.nanmean(v)) for k, v in scores.items()}
    agg.update({f"{k}_std": float(np.nanstd(v)) for k, v in scores.items()})
    agg["n_splits"] = n_splits
    return agg


# -----------------------------
# Runner
# -----------------------------
def main(cfg_path: str):
    cfg = load_config(cfg_path)
    set_global_seed(cfg.get("random_state", 42))

    # 출력 폴더
    out_dir = cfg.get("output_dir", "artifacts")
    run_dir = os.path.join(out_dir, timestamp())
    ensure_dir(run_dir)

    # 데이터
    X, y = load_data(cfg)

    # 전처리/샘플러
    prep, sampler = build_preprocessor(cfg)

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

        record = {"model": name, "module": model_cfg["module"], "params": model_cfg.get("params", {}), "metrics": metrics}
        results.append(record)
        print(f"\n=== {name} ===")
        print(json.dumps(record["metrics"], indent=2, ensure_ascii=False))

        # 각 모델 파이프라인 저장 (원한다면 마지막 fold 학습 파이프라인 저장)
        try:
            import joblib
            joblib.dump(pipe, os.path.join(run_dir, f"{name}_pipeline.joblib"))
        except Exception as e:
            print(f"[WARN] Failed to save pipeline for {name}: {e}")

    # 전체 결과 저장
    with open(os.path.join(run_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 표 형태 CSV도 저장
    flat_rows = []
    for r in results:
        flat = {"model": r["model"]}
        for k, v in r["metrics"].items():
            flat[k] = v
        flat_rows.append(flat)
    pd.DataFrame(flat_rows).to_csv(os.path.join(run_dir, "results.csv"), index=False, encoding="utf-8-sig")

    print(f"\nArtifacts saved to: {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)