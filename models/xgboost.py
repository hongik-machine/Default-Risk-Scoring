from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# XGBoost / LightGBM
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# -------- 2) 모델 본체 --------
def make_xgb_classifier(**overrides) -> XGBClassifier:
    """XGBoost 기본값 + 필요한 것만 덮어쓰기"""
    params: Dict[str, Any] = dict(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",    # 빠른 히스토그램 분할
        eval_metric="logloss"  # 기본 지표(추후 AUC/PR-AUC로 교체 가능)
    )
    params.update(overrides)
    return XGBClassifier(**params)

'''
최적 성능을 위한 패러미터 값
model = XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    max_depth=4,
    reg_lambda=10,
    subsample=0.75,
    colsample_bytree=0.75,
    min_child_weight=2,
    learning_rate=0.025,
    n_estimators=1000,
    random_state=0,
    n_jobs=8
)'''


def make_lgbm_classifier(**overrides) -> LGBMClassifier:
    """LightGBM 기본값 + 필요한 것만 덮어쓰기"""
    params: Dict[str, Any] = dict(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=5,        # leaf-wise라 과적합 방지에 유리
        num_leaves=31,      # 트리 복잡도 핵심 파라미터
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        objective="binary",
        metric="binary_logloss"
    )
    params.update(overrides)
    return LGBMClassifier(**params)