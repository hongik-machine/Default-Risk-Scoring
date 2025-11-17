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
    # 리스트 기본값은 field(default_factory=list)로 설정해야 안정적입니다.
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
            # 숫자로 매핑된 범주형도 'object'가 아닌 'number'로 감지되므로,
            # 강제로 범주형으로 취급해야 할 경우, num에서 제거하고 cat에 추가.
            if c not in cat:
                cat.append(c)

    # 중복 제거 및 최종 확정
    num = list(set(num))
    cat = list(set(cat))
    return num, cat


# =========================================================
# 3. payment_status 매핑 (Ordinal 처리 및 결측치 명확화) 🌟 개선
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
            # 'Unknown'(-1)을 명시적으로 np.nan으로 변환하여
            # 이후 SimpleImputer(median/mode)가 처리하도록 함.
            df[c] = df[c].replace(-1, np.nan)
    return df


# =========================================================
# 4. IQR 기반 outlier clipping 🌟 개선
# =========================================================
def clip_outliers(df: pd.DataFrame, cols: List[str], factor=1.5) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        # 데이터에 결측치가 있을 경우, quantile 계산 전에 무시 (Imputer가 처리 예정)
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR

        # 원래 데이터 타입 저장 (클리핑 후 정수형 복원 위함)
        original_dtype = df[col].dtype

        df[col] = df[col].clip(lower, upper)

        # 원래 dtype이 정수형 계열이었으면, float로 변환된 값을 다시 정수형으로 복원
        if pd.api.types.is_integer_dtype(original_dtype) or original_dtype == np.int64:
            # 클리핑된 값은 실수형이 되었을 수 있지만, round 후 정수로 복원 (NaN은 유지)
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
# 7. SMOTENC (OneHot 이전 단계에서 실행)
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
    # SMOTENC는 범주형 변수가 정수형이어야 함 (object/str는 불가능)
    y = df[target_col].astype(int)

    classes, counts = np.unique(y, return_counts=True)
    minority = counts.min()

    if minority < 2:
        print("[WARN] Minority class < 2 → SMOTE 불가능. 원본 데이터 반환.")
        return df

    # k_neighbors 조정
    k_neighbors = max(1, min(k_neighbors, minority - 1))

    # SMOTENC는 결측치를 허용하지 않으므로, 임시로 최빈값/중앙값 대체 후 SMOTE 적용
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

    X_res, y_res = smote.fit_resample(temp_X, y)  # 임시 결측치 처리된 데이터 사용

    X_res = pd.DataFrame(X_res, columns=X.columns)
    y_res = pd.Series(y_res, name=target_col)

    # SMOTE 후 결측치 재도입: SMOTE가 결측치를 포함하지 않는 데이터를 생성했으므로,
    # 이 결과를 그대로 사용하고, 이후 ColumnTransformer에서 결측치 처리를 진행.

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

    # -------------------------------------------
    # payment_status 컬럼 처리
    # -------------------------------------------
    ps_cols = [c for c in df.columns if "payment_status" in c]

    if config.use_payment_status_mapping:
        # map_payment_status 함수 내에서 -1이 np.nan으로 변환됨
        df = map_payment_status(df, ps_cols)

        if config.payment_status_as_categorical:
            # 강제로 범주형으로 취급하도록 force_categorical에 추가
            if config.force_categorical is None:
                config.force_categorical = []
            config.force_categorical += ps_cols

    # -------------------------------------------
    # 컬럼 타입 감지
    # -------------------------------------------
    num_cols, cat_cols = detect_feature_types(df, config.force_categorical)

    # -------------------------------------------
    # Outlier clipping
    # -------------------------------------------
    if config.outlier_clip:
        df = clip_outliers(df, num_cols)

    # -------------------------------------------
    # 스케일러 선택
    # -------------------------------------------
    if config.scaling_strategy == ScalingStrategy.STANDARD:
        scaler = StandardScaler()
    elif config.scaling_strategy == ScalingStrategy.MINMAX:
        scaler = MinMaxScaler()
    elif config.scaling_strategy == ScalingStrategy.ROBUST:
        scaler = RobustScaler()
    else:
        scaler = "passthrough"

    # -------------------------------------------
    # 각각의 파이프라인 구성
    # -------------------------------------------
    numeric_pipeline = Pipeline([
        # 결측치 처리: Outlier clipping 후 발생할 수 있는 nan과 원래 nan 처리
        ("impute", SimpleImputer(strategy="median")),
        ("scale", scaler)
    ])

    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        # handle_unknown="ignore"는 새로운 범주 등장 시 에러 대신 0 벡터 반환
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # -------------------------------------------
    # ColumnTransformer 조립
    # -------------------------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_cols),
            ("cat", categorical_pipeline, cat_cols)
        ],
        remainder="passthrough",  # 나머지 컬럼은 그대로 통과(passthrough)
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
# 10. 변환된 피처 이름 헬퍼 함수 🌟 추가
# =========================================================
def get_transformed_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """
    ColumnTransformer를 통과한 후의 최종 피처 이름 목록을 반환한다.
    (One-Hot Encoding으로 인해 늘어난 피처 이름 포함)
    """
    return list(preprocessor.get_feature_names_out())