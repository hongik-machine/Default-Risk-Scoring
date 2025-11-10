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

from typing import Optional, Tuple
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# 선택적: 불균형 보정 (사용 흐름은 샘플 코드와 동일—학습셋에만 적용)
from imblearn.over_sampling import SMOTE


# ------------------------------ IO ------------------------------
def load_table(path: str, *, sheet_name: Optional[str] = None, **read_kwargs) -> pd.DataFrame:
    """CSV/XLS/XLSX를 DataFrame으로 로드."""
    low = path.lower()
    if low.endswith(".csv"):
        return pd.read_csv(path, **read_kwargs)
    if low.endswith(".xls") or low.endswith(".xlsx"):
        return pd.read_excel(path, sheet_name=sheet_name, **read_kwargs)
    raise ValueError(f"Unsupported file type: {path}")


# ----------------------------- split -----------------------------
def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """학습/테스트 분할 (불균형이면 stratify=True 권장)."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if stratify else None,
    )


# ----------------------- numeric preprocessor -----------------------
def build_numeric_preprocessor(scale: bool = True) -> Pipeline:
    """숫자 전용 전처리 파이프라인.
    - SimpleImputer(strategy="median")
    - (옵션) StandardScaler()
    """
    steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps)


def fit_transform_preprocessor(preprocessor: Pipeline, X_train: pd.DataFrame) -> np.ndarray:
    """학습셋에 전처리를 학습/적용."""
    return preprocessor.fit_transform(X_train)


def transform_preprocessor(preprocessor: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """이미 학습된 전처리를 다른 데이터에 적용."""
    return preprocessor.transform(X)


# --------------------------- SMOTE utility ---------------------------
def smote_resample(
    X_train_scaled: np.ndarray,
    y_train: pd.Series,
    *,
    random_state: int = 42,
    k_neighbors: int = 5,
):
    """학습셋(전처리/스케일 완료 상태)에만 SMOTE 적용."""
    sm = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    X_res, y_res = sm.fit_resample(X_train_scaled, y_train)
    return X_res, y_res


# ------------------------ pipeline assembly ------------------------
def make_pipeline(model, *, preprocessor: Pipeline) -> Pipeline:
    """[preprocess] -> [model] 파이프라인 구성 (SMOTE는 파이프라인 밖에서)."""
    return Pipeline([
        ("preprocess", preprocessor),
        ("model", model),
    ])


# ----------------------------- example -----------------------------
if __name__ == "__main__":
    # 예시 (파일/타깃명만 맞춰서 사용):
    # df = load_table("./data/credit_default.csv")
    # target = "default.payment.next.month"
    # X, y = df.drop(columns=[target]), df[target]
    # X_tr, X_te, y_tr, y_te = split_data(X, y, test_size=0.2, stratify=True)
    # pre = build_numeric_preprocessor(scale=True)
    # X_tr_s = fit_transform_preprocessor(pre, X_tr)
    # X_te_s = transform_preprocessor(pre, X_te)
    # X_tr_res, y_tr_res = smote_resample(X_tr_s, y_tr)
    # from sklearn.linear_model import LogisticRegression
    # clf = LogisticRegression(max_iter=1000)
    # clf.fit(X_tr_res, y_tr_res)
    # print("ready")
    print("baseline numeric preprocessing ready")