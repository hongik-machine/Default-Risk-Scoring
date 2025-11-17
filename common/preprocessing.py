"""
Baseline preprocessing helpers aligned with the sample logistic+SMOTE script.

제공 기능:
- load_table: CSV/XLS/XLSX 읽기
- split_data: 학습/테스트 분리 (stratify 옵션)
- build_numeric_preprocessor: SimpleImputer(median) + StandardScaler
- fit_transform_preprocessor / transform_preprocessor: 전처리 헬퍼
- smote_resample: (학습셋 한정) SMOTE 적용
- make_pipeline: [preprocess] -> [model]  (샘플과 동일하게 SMOTE는 파이프라인 밖에서)

※ 전제: 피처가 전부 숫자(UCI Default 데이터처럼)라는 가정.
   범주형이 생기면 OneHot 혹은 SMOTENC로 확장 필요.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, List, Union
from enum import Enum, auto
from dataclasses import dataclass, field

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import TransformerMixin, BaseEstimator

from imblearn.over_sampling import SMOTENC


# =========================================================
# Scaling 전략 Enum
# =========================================================
class ScalingStrategy(Enum):
    STANDARD = auto()  # Logistic Regression, SVM
    MINMAX = auto()  # Neural Network, Deep Learning
    ROBUST = auto()  # Outlier-resistant
    NONE = auto()  # Tree-based models


# =========================================================
# 전처리 옵션 dataclass
# =========================================================
@dataclass
class PreprocessConfig:
    use_payment_status_mapping: bool = True
    payment_status_as_categorical: bool = False
    force_categorical: Optional[List[str]] = field(default_factory=list)
    outlier_clip: bool = True
    scaling_strategy: ScalingStrategy = ScalingStrategy.ROBUST


# =========================================================
# 1. 파일 로더
# =========================================================
def load_table(path: str, *, sheet_name: Optional[str] = None, **kwargs) -> pd.DataFrame:
    low = path.lower()
    if low.endswith(".csv"):
        return pd.read_csv(path, **kwargs)
    if low.endswith(".xls") or low.endswith(".xlsx"):
        return pd.read_excel(path, sheet_name=sheet_name, **kwargs)
    raise ValueError(f"Unsupported file format: {path}")


# =========================================================
# 2. 컬럼 타입 자동 감지
# =========================================================
def detect_feature_types(df: pd.DataFrame, force_categorical: Optional[List[str]] = None):
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    cat = df.select_dtypes(include=["object"]).columns.tolist()

    if force_categorical:
        for c in force_categorical:
            if c in num:
                num.remove(c)
            if c not in cat:
                cat.append(c)

    num = list(set(num))
    cat = list(set(cat))
    return num, cat


# =========================================================
# 3. payment_status 매핑
# =========================================================
def map_payment_status(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    mapping = {
        "Payed duly": 0,
        "Payment delayed 1 month": 1,
        "Payment delayed 2 months": 2,
        "Payment delayed 3 months": 3,
        "Payment delayed 4 months": 4,
        "Payment delayed 5 months": 5,
        "Payment delayed 6 months": 6,
        "Unknown": -1,
    }

    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].map(mapping)
            df[c] = df[c].replace(-1, np.nan)
    return df


# =========================================================
# 4. IQR 기반 outlier clipping
# =========================================================
def clip_outliers(df: pd.DataFrame, cols: List[str], factor=1.5) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR

        original_dtype = df[col].dtype

        df[col] = df[col].clip(lower, upper)

        if pd.api.types.is_integer_dtype(original_dtype) or original_dtype == np.int64:
            df[col] = df[col].round().astype(original_dtype, errors="ignore")
    return df


# =========================================================
# 5. Train/Test split
# =========================================================
def split_data(X, y, *, test_size=0.2, stratify=True, random_state=42):
    return train_test_split(
        X, y,
        test_size=test_size,
        stratify=y if stratify else None,
        random_state=random_state
    )


# =========================================================
# 6. Train/Val/Test split
# =========================================================
def three_way_split(
        X, y,
        *, test_size=0.2, val_size=0.2,
        stratify=True, random_state=42
):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=(test_size + val_size),
        stratify=y if stratify else None,
        random_state=random_state
    )

    rel_test = test_size / (test_size + val_size)

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=rel_test,
        stratify=y_temp if stratify else None,
        random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# =========================================================
# 7. SMOTENC (OneHot 이전)
# =========================================================
def smote_before_encoding(
        df: pd.DataFrame,
        target_col: str,
        categorical_cols: List[str],
        numerical_cols: List[str],
        *,
        random_state=42,
        k_neighbors=5
):
    df = df.copy()

    X = df[categorical_cols + numerical_cols]
    y = df[target_col].astype(int)

    classes, counts = np.unique(y, return_counts=True)
    minority = counts.min()

    if minority < 2:
        print("[WARN] Minority class < 2 → SMOTE 불가능")
        return df

    k_neighbors = max(1, min(k_neighbors, minority - 1))

    temp_X = X.copy()
    for col in categorical_cols:
        temp_X[col] = temp_X[col].fillna(temp_X[col].mode()[0])
    for col in numerical_cols:
        temp_X[col] = temp_X[col].fillna(temp_X[col].median())

    cat_indices = [temp_X.columns.get_loc(c) for c in categorical_cols]

    smote = SMOTENC(
        categorical_features=cat_indices,
        random_state=random_state,
        k_neighbors=k_neighbors
    )

    X_res, y_res = smote.fit_resample(temp_X, y)

    X_res = pd.DataFrame(X_res, columns=X.columns)
    y_res = pd.Series(y_res, name=target_col)

    return pd.concat([X_res, y_res], axis=1)


# =========================================================
# 8. ColumnTransformer 구성
# =========================================================
def build_preprocessor(
        X: pd.DataFrame,
        config: Optional[PreprocessConfig] = None
):
    if config is None:
        config = PreprocessConfig()
    df = X.copy()

    ps_cols = [c for c in df.columns if "payment_status" in c]

    if config.use_payment_status_mapping:
        df = map_payment_status(df, ps_cols)
        if config.payment_status_as_categorical:
            config.force_categorical += ps_cols

    num_cols, cat_cols = detect_feature_types(df, config.force_categorical)

    if config.outlier_clip:
        df = clip_outliers(df, num_cols)

    if config.scaling_strategy == ScalingStrategy.STANDARD:
        scaler = StandardScaler()
    elif config.scaling_strategy == ScalingStrategy.MINMAX:
        scaler = MinMaxScaler()
    elif config.scaling_strategy == ScalingStrategy.ROBUST:
        scaler = RobustScaler()
    else:
        scaler = "passthrough"

    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", scaler)
    ])

    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_cols),
            ("cat", categorical_pipeline, cat_cols)
        ],
        remainder="passthrough",
        verbose_feature_names_out=True
    )

    return preprocessor, df


# =========================================================
# 9. 전처리 + 모델 Pipeline
# =========================================================
def build_full_pipeline(model: Union[BaseEstimator, TransformerMixin], preprocessor: ColumnTransformer) -> Pipeline:
    return Pipeline([
        ("preprocess", preprocessor),
        ("model", model)
    ])


# =========================================================
# 10. 변환된 피처 이름
# =========================================================
def get_transformed_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    return list(preprocessor.get_feature_names_out())


# =========================================================
# ⭐⭐ (테스트벤치 필수) make_preprocessor
# =========================================================
def make_preprocessor(X, cfg: dict | None = None):
    preprocessor, _ = build_preprocessor(X)
    return preprocessor


# =========================================================
# ⭐⭐ (테스트벤치 필수) build_sampler
# =========================================================
def build_sampler(prep_cfg: dict):
    # 기본 프로젝트에서는 sampler 없음
    return None
