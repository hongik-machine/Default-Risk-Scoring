# common/preprocessing.py

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Tuple, List
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


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

# sklearn 버전에 따라 OneHotEncoder의 인자가 달라 안정화 래퍼 사용
def _make_ohe():
    from sklearn.preprocessing import OneHotEncoder
    try:
        # sklearn >= 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # sklearn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)



# 이상한 값(데이터의 오류) 제거
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

# 안쓰이고 있는 함수
def split_xy(df: pd.DataFrame, target: str):
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


# column의 data type - continuous / categorical / ordinal 구분
def infer_columns(X: pd.DataFrame):

    cols = set(X.columns)

    cont_cols = [c for c in CONTINUOUS_COLS if c in cols]
    cat_cols = [c for c in CATEGORICAL_COLS if c in cols]
    ord_cols = [c for c in ORDINAL_COLS if c in cols]

    return cont_cols, cat_cols, ord_cols




def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    - continuous: mean impute + StandardScaler
    - categorical: most_frequent impute + OneHotEncoder
    - ordinal(PAY_*): most_frequent impute (값 그대로 사용)
    """

    # X 안에서 연속형, 범주형, 순서형 column을 자동으로 골라낸다.
    cont_cols, cat_cols, ord_cols = infer_columns(X)

    # 1) continuous
    continuous_proc = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ])

    # 2) categorical
    categorical_proc = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", _make_ohe()),
    ])

    # 3) ordinal (PAY_*): 결측만 최빈값으로 채우고 숫자는 그대로 사용
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