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
import os
from datetime import datetime
import importlib
from typing import Any, Dict, List
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 화면에 띄우지 않고 내부적으로만 처리하도록 설정 (또는 'TkAgg')
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

def build_preprocessor(cfg: Dict[str, Any], x: pd.DataFrame):
    prep_mod = importlib.import_module("common.preprocessing")
    prep = prep_mod.make_preprocessor(x)

    # preprocessing 공통 설정 읽어오기
    prep_cfg = cfg.get("preprocessing", {})
    random_state = cfg.get("random_state", 42)

    # sampler 생성
    build_sampler = getattr(prep_mod, "build_sampler", None)
    sampler = None
    if callable(build_sampler):
        sampler = build_sampler(prep_cfg, random_state=random_state)
    print(f"✅ 현재 적용된 Sampler: {sampler}")

    # PCA
    build_pca = getattr(prep_mod, "build_pca", None)
    pca = build_pca(prep_cfg, random_state=random_state) if callable(build_pca) else None

    return prep, sampler, pca

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
# Holdout 버전의 학습
# ---------------------------------------------------------------

def run_holdout(pipe: Pipeline, X_train, y_train, X_test, y_test, eval_cfg):

    threshold = eval_cfg.get("threshold", 0.5)

    # 1) Train 전체로 모델 fit
    pipe.fit(X_train, y_train)

    # 2) Test로 예측
    y_pred = pipe.predict(X_test)
    y_proba = (
        pipe.predict_proba(X_test)[:, 1]
        if hasattr(pipe, "predict_proba")
        else None
    )

    # 3) 기본 Metrics 계산
    metrics = compute_metrics(y_test, y_pred, y_proba, threshold)

    return metrics


# ---------------------------------------------------------------
# Cross-validation 버전의 학습
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
    rows = []
    for r in results:
        name = r["model"]
        sampler_name = r.get("sampler", "None")
        pca_status = r.get("pca", "N/A")

        m = r["metrics"]
        cv = m.get("cv", {})
        te = m.get("test", {})

        rows.append({
            "model": name,
            "sampler": sampler_name,
            "pca": pca_status,

            # CV 기준
            "cv_f1": cv.get("f1_mean", np.nan),
            "cv_recall": cv.get("recall_mean", np.nan),
            "cv_roc_auc": cv.get("roc_auc_mean", np.nan),
            "cv_pr_auc": cv.get("pr_auc_mean", np.nan),

            # Test 기준
            "test_f1": te.get("f1", np.nan),
            "test_recall": te.get("recall", np.nan),
            "test_roc_auc": te.get("roc_auc", np.nan),
            "test_pr_auc": te.get("pr_auc", np.nan)
        })

    df = pd.DataFrame(rows)

    print("\n=== Summary (rounded) ===")
    cols = ["model", "sampler", "pca", "cv_f1", "cv_recall", "cv_roc_auc", "test_f1", "test_recall", "test_roc_auc"]
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(df[cols].round(4).to_string(index=False))

    # CV 기준 플롯
    def _plot(metric: str, title: str):
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

    # CV 기준
    _plot("cv_roc_auc", "CV ROC-AUC by Model")
    _plot("cv_pr_auc", "CV PR-AUC by Model")
    _plot("cv_f1", "CV F1 by Model")
    _plot("cv_recall", "CV Recall by Model")

    # Test 기준
    _plot("test_roc_auc", "Test ROC-AUC by Model")
    _plot("test_pr_auc", "Test PR-AUC by Model")
    _plot("test_f1", "Test F1 by Model")
    _plot("test_recall", "Test Recall by Model")

    plt.show()



def print_metrics_pretty(name: str, metrics: dict):
    print(f"\n=== {name} ===")

    # 1) CV 결과 (있으면)
    if "cv" in metrics:
        cv = metrics["cv"]
        print("[CV] mean ± std")
        print(f"  accuracy : {cv['accuracy_mean']:.4f} ± {cv['accuracy_std']:.4f}")
        print(f"  precision: {cv['precision_mean']:.4f} ± {cv['precision_std']:.4f}")
        print(f"  recall   : {cv['recall_mean']:.4f} ± {cv['recall_std']:.4f}")
        print(f"  f1       : {cv['f1_mean']:.4f} ± {cv['f1_std']:.4f}")
        if "roc_auc_mean" in cv:
            print(f"  roc_auc  : {cv['roc_auc_mean']:.4f} ± {cv['roc_auc_std']:.4f}")
        if "pr_auc_mean" in cv:
            print(f"  pr_auc   : {cv['pr_auc_mean']:.4f} ± {cv['pr_auc_std']:.4f}")
        if "best_threshold" in cv:
            print(f"  best_th  : {cv['best_threshold']:.3f} (F1={cv['best_f1_at_threshold']:.4f})")

    # 2) Test 결과 (있으면)
    if "test" in metrics:
        te = metrics["test"]
        print("\n[Test]")
        print(f"  accuracy : {te['accuracy']:.4f}")
        print(f"  precision: {te['precision']:.4f}")
        print(f"  recall   : {te['recall']:.4f}")
        print(f"  f1       : {te['f1']:.4f}")
        if "roc_auc" in te:
            print(f"  roc_auc  : {te['roc_auc']:.4f}")
        if "pr_auc" in te:
            print(f"  pr_auc   : {te['pr_auc']:.4f}")
        print(f"  pos_rate : {te['pos_rate']:.4f}")
        print(f"  threshold: {te['threshold']:.3f}")

        # 혼동행렬 & classification report는 따로 보기 좋게
        print("\n  Confusion matrix:")
        cm = np.array(te["confusion_matrix"])
        print(cm)

        print("\n  Classification report:")
        print(te["classification_report"])


# ---------------------------------------------------------------
# Runner
# ---------------------------------------------------------------

# [NEW] 파라미터 추가
def main(cfg_path: str, use_tuned: bool = False):

    # 1) 데이터셋 load + clean
    cfg = load_config(cfg_path)
    set_global_seed(cfg.get("random_state", 42))
    X, y = load_data(cfg)

    df = X.copy()
    df[cfg["data"]["target"]] = y
    df = prep_mod.clean_data(df)

    X = df.drop(columns=[cfg["data"]["target"]])
    y = df[cfg["data"]["target"]]


    # 2) train/test split (1회)
    eval_cfg = cfg.get("evaluation", {})
    test_size = eval_cfg.get("test_size", 0.25)
    seed = eval_cfg.get("random_state", 42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    # 3) X_train만 이용해 preprocessor fit 준비
    prep, sampler, pca = build_preprocessor(cfg, X_train)

    results = []


    # =========================================================
    # [NEW] 튜닝된 하이퍼파라미터 불러오기(조건부 실행
    # =========================================================
    tuned_params_dict = {}

    # ★ use_tuned가 True일 때만 파일을 찾습니다!
    if use_tuned:
        best_params_path = "models/best_hyperparameters.json"
        if os.path.exists(best_params_path):
            print(f"\n [Tuned Mode] 튜닝된 파라미터 파일을 로드합니다: {best_params_path}")
            try:
                with open(best_params_path, "r") as f:
                    tuned_params_dict = json.load(f)
            except Exception as e:
                print(f"  파라미터 로드 중 오류 발생: {e}")
        else:
            print("\n [Tuned Mode] 파일이 없어 기본 설정을 사용합니다.")
    else:
        print("\n [Baseline Mode] 튜닝 없이 기본 설정(config.yaml)으로 실행합니다.")
    # =========================================================

    # 4) 4개의 모델을 평가하는 loop
    for model_cfg in cfg["models"]:
        name = model_cfg["name"]

        # =========================================================
        # [NEW] 튜닝 파라미터 적용 로직 (딕셔너리에 값이 있을 때만)
        # =========================================================
        # 만약 JSON 파일에 해당 모델 이름(name)이 있다면 파라미터를 덮어씌웁니다.
        if name in tuned_params_dict:
            print(f"    [{name}] 튜닝된 최적 파라미터를 적용합니다.")

            # 기존 config의 params를 가져와서
            current_params = model_cfg.get("params", {})

            # 튜닝된 값으로 업데이트 (덮어쓰기)
            # tuned_params_dict[name]['best_params'] 구조라고 가정
            best_p = tuned_params_dict[name].get("best_params", {})
            current_params.update(best_p)

            # 모델 설정에 다시 저장
            model_cfg["params"] = current_params

        else:
            # 튜닝 모드가 꺼져있거나(use_tuned=False), 해당 모델의 튜닝 결과가 없으면 여기로 옴
            if use_tuned:
                print(f"  [{name}] 튜닝 정보가 없어 기본 설정(config.yaml)을 사용합니다.")
            else:
            # Baseline 모드일 때는 조용히 기본값 사용
                pass
        # =========================================================



        clf = build_model(model_cfg)

        # pipeline 단계 조립
        steps = [("prep", prep)]

        # PCA
        if pca is not None:
            steps.append(("pca", pca))


        # Sampler
        if sampler is not None:
            steps.append(("sampler", sampler))

        # 모델 적용
        steps.append(("clf", clf))

        pipe = Pipeline(steps)


        eval_cfg = cfg.get("evaluation", {})
        threshold = eval_cfg.get("threshold", 0.5)

        # CV 모드
        if eval_cfg.get("mode", "cv") == "cv":
            cv_metrics = run_cv(pipe, X_train, y_train, eval_cfg)

            # CV 끝난 뒤, train 전체로 재학습 → test로 최종 점수 계산 (새로 추가된 부분)

            pipe.fit(X_train, y_train)
            y_proba_test = pipe.predict_proba(X_test)[:, 1]
            y_pred_test = (y_proba_test >= threshold).astype(int)

            test_metrics = compute_metrics(
                y_test, y_pred_test, y_proba_test, threshold
            )

            metrics = {
                "cv": cv_metrics,
                "test": test_metrics,
            }

        # holdout 모드
        else:
            pipe.fit(X_train, y_train)

            y_proba_test = pipe.predict_proba(X_test)[:, 1]
            y_pred_test = (y_proba_test >= threshold).astype(int)

            metrics = compute_metrics(y_test, y_pred_test, y_proba_test, threshold)

        # sampler 이름 추출 (없으면 "None")
        sampler_name = sampler.__class__.__name__ if sampler else "None"

        # PCA 적용 여부 저장 (원하는대로 'Used', 'None'으로 저장)
        pca_status = "Used" if pca else "None"

        record = {
            "model": name,
            "sampler": sampler_name,
            "pca": pca_status,
            "module": model_cfg["module"],
            "params": model_cfg.get("params", {}),
            "metrics": metrics
        }


        results.append(record)
        print_metrics_pretty(name, metrics)

    visualize_results(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default="config.yaml")
    # [NEW] 튜닝된 파라미터를 쓸지 말지 결정하는 스위치 추가
    parser.add_argument("--tuned", action="store_true", help="Use tuned hyperparameters if available")

    args = parser.parse_args()

    #main 함수에 tuned 옵션값(True/False)도 같이 넘겨줌
    main(args.config, args.tuned)