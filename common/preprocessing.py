# common/preprocessing.py

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Tuple, List
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# sklearn 버전에 따라 OneHotEncoder의 인자가 달라 안정화 래퍼 사용
def _make_ohe():
    from sklearn.preprocessing import OneHotEncoder
    try:
        # sklearn >= 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # sklearn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


# 안쓰고 있는 함수
def basic_clean(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Unnamed 계열 컬럼 제거, target 정수화, 중복 제거"""
    to_drop = [c for c in df.columns if c.lower().startswith("unnamed")]
    if to_drop:
        df = df.drop(columns=to_drop)

    if target not in df.columns:
        raise ValueError(f"타깃 컬럼 '{target}' 이(가) 존재하지 않음. 실제 컬럼들: {list(df.columns)}")

    # 타깃 0/1 정수화
    df[target] = pd.to_numeric(df[target], errors="coerce").fillna(0).astype(int)

    # 중복 제거
    df = df.drop_duplicates()

    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

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

def infer_columns(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    return num_cols, cat_cols




def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """숫자: 평균 대치 / 범주: 최빈값 대치 + 원핫"""
    num_cols, cat_cols = infer_columns(X)
    num_cols, cat_cols = infer_columns(X)

    numeric_proc = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])
    categorical_proc = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", _make_ohe())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_proc, num_cols),
            ("cat", categorical_proc, cat_cols)
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )
    return preprocessor