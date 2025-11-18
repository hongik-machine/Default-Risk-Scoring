# common/preprocessing.py

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Tuple, List
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids


# =========================================================
# 1. data type별로 각 column 나누기
# =========================================================
# 연속형(스케일링 대상)
CONTINUOUS_COLS = [
    "LIMIT_BAL",
    "AGE",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
    "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3",
    "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]

# 범주형(원-핫 인코딩)
CATEGORICAL_COLS = [
    "SEX",
    "EDUCATION",
    "MARRIAGE",
]

# 순서형(PAY 상태) - clean_data에서 PAY_0 → PAY_1 으로 rename 함
ORDINAL_COLS = [
    "PAY_1", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
]


# column의 data type - continuous / categorical / ordinal 구분
def infer_columns(X: pd.DataFrame):

    cols = set(X.columns)

    cont_cols = [c for c in CONTINUOUS_COLS if c in cols]
    cat_cols = [c for c in CATEGORICAL_COLS if c in cols]
    ord_cols = [c for c in ORDINAL_COLS if c in cols]

    return cont_cols, cat_cols, ord_cols





# =========================================================
# 2. IQR 기반 outlier clipping 하기 위한 class 별도 정의
# =========================================================
class IQRClipper(BaseEstimator, TransformerMixin):
    def __init__(self, factor: float = 1.5):
        self.factor = factor

    def fit(self, X, y=None):
        # X: 2D array or DataFrame (continuous 컬럼만 들어온다고 가정)
        X_df = pd.DataFrame(X)
        q1 = X_df.quantile(0.25)
        q3 = X_df.quantile(0.75)
        iqr = q3 - q1

        self.lower_ = q1 - self.factor * iqr
        self.upper_ = q3 + self.factor * iqr
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X)
        X_clipped = X_df.clip(lower=self.lower_, upper=self.upper_, axis=1)
        return X_clipped.to_numpy()





# ===========================================================
# 3. 이상한 값(데이터의 오류) 제거 - Nan 값으로 데이터가 없는 것과는 다름
# ===========================================================
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ID 제거
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    # EDUCATION 잘못된 값 제거
    df = df[df["EDUCATION"].isin([1, 2, 3, 4])]

    # MARRIAGE 잘못된 값 제거
    df = df[df["MARRIAGE"].isin([1, 2, 3])]

    # PAY 컬럼들
    pay_cols = [c for c in df.columns if c.startswith("PAY_")]

    # PAY_0 → PAY_1 이름 변경 (UCI 문서상 PAY_0 = 최근 9월)
    if "PAY_0" in df.columns and "PAY_1" not in df.columns:
        df = df.rename(columns={"PAY_0": "PAY_1"})
        pay_cols = [c for c in df.columns if c.startswith("PAY_")]

    # PAY_n 값 정상 납부 통합
    for col in pay_cols:
        df[col] = df[col].replace({-2: 0, -1: 0, 0: 0})

    return df



# ===========================================================
# 4. 각 데이터의 특성에 맞는 전처리 helper 함수
# ===========================================================
# sklearn 버전에 따라 OneHotEncoder의 인자가 달라 안정화 래퍼 사용
# One-Hot_Encoding
def _make_ohe():
    from sklearn.preprocessing import OneHotEncoder
    try:
        # sklearn >= 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # sklearn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


# ===========================================================
# 5. PCA 적용 - 열 줄여주기
# ===========================================================

# config를 기준으로 PCA transformer 혹은 None을 return
def build_pca(prep_cfg: dict, random_state: int | None = None):

    pca_cfg = prep_cfg.get("pca", {})
    use_pca = pca_cfg.get("use", False) or pca_cfg.get("use_pca", False)

    if not use_pca:
        return None

    n_components = pca_cfg.get("n_components", None)
    return PCA(
        n_components=n_components,
        random_state=random_state
    )



# ===========================================================
# 6. ColumnTransformer 형태의 preprocessor 만들기
# ===========================================================
def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    - continuous: mean impute -> IQR outlier clipping -> StandardScaler
    - categorical: most_frequent impute + OneHotEncoder
    - ordinal(PAY_*): most_frequent impute (값 그대로 사용)
    """

    # X 안에서 연속형, 범주형, 순서형 column을 자동으로 골라낸다.
    cont_cols, cat_cols, ord_cols = infer_columns(X)

    # 1) continuous
    continuous_proc = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("iqr_clip", IQRClipper(factor=1.5)),
        ("scaler", StandardScaler()),
    ])

    # 2) categorical
    categorical_proc = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", _make_ohe()),
    ])

    # 3) ordinal (PAY_*): scaling 없이 원 값으로
    ordinal_proc = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # 스케일링/원핫 안 하고 값 그대로
    ])

    # 서로 다른 column 그룹에 서로 다른 전처리를 적용
    preprocessor = ColumnTransformer(
        transformers=[
            ("cont", continuous_proc, cont_cols),
            ("cat", categorical_proc, cat_cols),
            ("ord", ordinal_proc, ord_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


# ===========================================================
# 7. Sampler 적용 (SMOTE / ClusterCentroids)
# ===========================================================
def build_sampler(prep_cfg: dict, random_state: int | None = None):

    # prep_cfg['sampler'] 설정을 읽어서 SMOTE, ClusterCentroids 혹은 None을 반환
    sampler_cfg = prep_cfg.get("sampler", {})
    method = sampler_cfg.get("method",  None)  # 'smote', 'cluster_centroids' 등

    # sampler를 사용하지 않는 경우
    if not method or method == "none":
        return None

    params = sampler_cfg.get("params", {}).copy()

    # random_state가 params에 없으면 기본값 주입
    if "random_state" not in params and random_state is not None:
        params["random_state"] = random_state

    # SMOTE (오버샘플링)
    if method == "smote":
        # 필요하다면 k_neighbors 등의 파라미터를 sampler_cfg에서 꺼내서 쓸 수 있음
        # strategy = sampler_cfg.get("strategy", "auto")
        return SMOTE(**params)

    # Cluster Centroids (언더샘플링)
    elif method == "cluster_centroids":
        params.pop("k_neighbors", None)

        return ClusterCentroids(**params)

    # 그 외 정의되지 않은 method
    else:
        raise ValueError(f"Unknown sampler method: {method}")