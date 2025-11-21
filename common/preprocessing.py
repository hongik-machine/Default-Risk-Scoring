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
    "LIMIT_BAL",  # 신용한도
    "AGE",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",  # 월별 청구 금액
    "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3",  # 월별 납부 금액
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

    # 미리 정의된 리스트를 바탕으로, 현재 DataFrame에 포함된 컬럼만 선택
    cont_cols = [c for c in CONTINUOUS_COLS if c in cols]
    cat_cols = [c for c in CATEGORICAL_COLS if c in cols]
    ord_cols = [c for c in ORDINAL_COLS if c in cols]

    return cont_cols, cat_cols, ord_cols


# =========================================================
# 2. IQR 기반 outlier clipping 하기 위한 class 별도 정의
# =========================================================
class IQRClipper(BaseEstimator, TransformerMixin):  # BaseEstimator, TransformerMixin을 상속받아 Sklearn 파이프라인에 사용할 수 있게 함.
    def __init__(self, factor: float = 1.5):
        self.factor = factor  # 이상치 경계를 설정하는 계수(factor)를 설정(1.5)

    def fit(self, X, y=None):  # 훈련 데이터(X)의 분포를 학습하여 이상치 경계를 계산
        # X: 2D array or DataFrame (continuous 컬럼만 들어온다고 가정)
        X_df = pd.DataFrame(X)
        q1 = X_df.quantile(0.25)
        q3 = X_df.quantile(0.75)
        iqr = q3 - q1

        self.lower_ = q1 - self.factor * iqr
        self.upper_ = q3 + self.factor * iqr
        return self

    def transform(self, X):  # 새로운 데이터(X)에 학습된 이상치 경계를 적용
        X_df = pd.DataFrame(X)
        X_clipped = X_df.clip(lower=self.lower_, upper=self.upper_, axis=1)  # .clip() 함수를 사용하여 경계 밖의 값을 경계 값으로 대체
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

# 설정 파일(config)을 기준으로 PCA 트랜스포머 객체를 생성하거나 None을 반환
def build_pca(prep_cfg: dict, random_state: int | None = None):
    pca_cfg = prep_cfg.get("pca", {})
    use_pca = pca_cfg.get("use", False) or pca_cfg.get("use_pca", False)

    if not use_pca:
        return None

    n_components = pca_cfg.get("n_components", None)
    return PCA(
        n_components=n_components,
        random_state=random_state  # 재현성을 위한 시드 설정
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

    # 1단계: X 안에서 연속형, 범주형, 순서형 column을 자동으로 골라냄
    cont_cols, cat_cols, ord_cols = infer_columns(X)

    # 2단계: 연속형 데이터 처리 파이프라인
    continuous_proc = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),  # 결측치를 평균으로 채움
        ("iqr_clip", IQRClipper(factor=1.5)),  # IQR 기준으로 이상치를 경계 값으로 자름
        ("scaler", StandardScaler()),  # 데이터를 표준화함
    ])

    # 3단계: 범주형 데이터 처리 파이프라인
    categorical_proc = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),  # 결측치를 최빈값으로 채움
        ("onehot", _make_ohe()),  # One-Hot Encoding을 적용
    ])

    # 4단계: 순서형(PAY_*) 데이터 처리 파이프라인
    ordinal_proc = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),  # 결측치를 최빈값으로 채움
        # 스케일링 / 원핫 인코딩 없이 값 그대로 유지
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
    method = sampler_cfg.get("method", None)  # 'smote', 'cluster_centroids' 등

    # sampler를 사용하지 않는 경우
    if not method or method == "none":
        return None

    params = sampler_cfg.get("params", {}).copy()

    # 재현성을 위해 random_state가 파라미터에 없으면 기본값을 주입
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

    # 정의되지 않은 샘플링 방법인 경우 오류 발생
    else:
        raise ValueError(f"Unknown sampler method: {method}")