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
import matplotlib.pyplot as plt

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

# 중복 생성 에러 방지
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

# 실행 시각을 폴더/파일명에 작성
def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# config.yaml을 딕셔너리로 로드
def load_config(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# 파이썬/넘파이 난수 고정
def set_global_seed(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)


# -----------------------------
# Data
# -----------------------------

# 데이터셋 로드 -> 타깃 컬럼 존재 체크 후 X/y 분리해서 반환
def load_data(cfg: Dict[str, Any]) -> tuple[pd.DataFrame, pd.Series]:
    data_cfg = cfg["data"]
    path = data_cfg["file_path"]
    target = data_cfg["target"]

    # header=1 : 엑셀 파일의 두 번째 줄(index 1)을 헤더(컬럼명)로 읽어옴
    df = pd.read_excel(path, header=1)

    if target not in df.columns:
        raise ValueError(f"target '{target}' not in columns: {list(df.columns)[:10]}...")
    y = df[target]
    X = df.drop(columns=[target])
    return X, y


# -----------------------------
# Build pipeline components
# -----------------------------

# preprocessing 파일을 모듈화해 import
def build_preprocessor(cfg: Dict[str, Any], X: pd.DataFrame):

    prep_mod = importlib.import_module("common.preprocessing")
    prep = prep_mod.make_preprocessor(X)
    build_sampler = getattr(prep_mod, "build_sampler", None)
    sampler = build_sampler(cfg.get("preprocessing", {})) if callable(build_sampler) else None
    return prep, sampler



# 모델 생성
# config.yaml의 각 항목에 정의된 "module": "models.logistic" 같은 경로를 동적 임포트
# 그 모듈의 build_model(**params)를 호출해 sklearn 호환 추정기를 반환
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

# 메트릭 계산
def compute_metrics(y_true, y_pred, y_proba=None, threshold=0.5) -> Dict[str, Any]:
    # y_proba가 있고 threshold가 0.5가 아니면 y_pred를 재계산
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
    }
    if y_proba is not None:
        out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        out["pr_auc"] = float(average_precision_score(y_true, y_proba))
    # 결과 확인용: 적용된 threshold 기록
    out["threshold"] = threshold
    return out


# -----------------------------
# Train & Evaluate (Holdout)
# -----------------------------

# 홀드아웃 평가
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

# 교차검증(CV) 평가
# 층화 K-fold로 수동 루프를 돌며 각 폴드에서 파이프라인 학습/예측.
def run_cv(pipe: Pipeline, X, y, eval_cfg: Dict[str, Any]) -> Dict[str, Any]:
    n_splits = eval_cfg.get("cv", 5)
    seed = eval_cfg.get("random_state", 42)

    # Config에서 threshold 읽어오기 (없으면 기본 0.5)
    threshold = eval_cfg.get("threshold", 0.5)

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

        m = compute_metrics(y_te, y_pred, y_proba, threshold=threshold)
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
# Visualization
# -----------------------------
def visualize_results(results: List[Dict[str, Any]]) -> None:
    """
    결과 리스트를 요약 DataFrame으로 만들고,
    주요 지표(ROC-AUC, PR-AUC, F1)를 막대그래프로 시각화한다.
    """

    def _get(m: Dict[str, Any], key: str):
        # CV면 *_mean, holdout이면 key를 사용
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

    # 콘솔에 요약표
    print("\n=== Summary (rounded) ===")
    with pd.option_context("display.max_columns", None, "display.width", 120):
        print(df.round(4).to_string(index=False))

    # 막대그래프 유틸
    def _plot_metric(metric: str, title: str):
        sub = df[["model", metric]].copy().dropna()
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

    # 우선순위 높은 지표들
    _plot_metric("roc_auc", "ROC-AUC by Model")
    _plot_metric("pr_auc", "PR-AUC (Average Precision) by Model")
    _plot_metric("f1", "F1 Score by Model")

    plt.show()


# -----------------------------
# Runner
# -----------------------------
def main(cfg_path: str):

    # 설정 로딩
    cfg = load_config(cfg_path)
    # 시드 고정
    set_global_seed(cfg.get("random_state", 42))

    # 데이터
    X, y = load_data(cfg)

    # 전처리/샘플러
    prep, sampler = build_preprocessor(cfg, X)

    results = []
    for model_cfg in cfg["models"]:
        # 특정 모델만 돌리고 싶을 때 임시로 사용
        #if "neural_network" not in model_cfg["name"]:
            #continue
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
        print(json.dumps(record["metrics"], indent=2, ensure_ascii=False))


    visualize_results(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)