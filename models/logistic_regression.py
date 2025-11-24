from typing import Optional, Union
from sklearn.linear_model import LogisticRegression

def build_model(
    *,
    C: float = 1.0,
    penalty: str = "l2",
    solver: str = "auto",   # [변경] 유저가 직접 정할 수도 있게 'auto'로 설정
    max_iter: int = 3000, 
    class_weight: Optional[Union[dict, str]] = None,
    random_state: int = 42,
    tol: float = 1e-4,
    l1_ratio: Optional[float] = None, # [추가] ElasticNet용 비율
    n_jobs: int = -1  # [추가] 속도 향상 (모든 CPU 코어 사용)
) -> LogisticRegression:

    # 1. Solver 자동 선택 로직
    if solver == "auto":
        if penalty == "l1":
            solver = "liblinear" # L1은 liblinear가 안정적이고 빠름
        elif penalty == "elasticnet":
            solver = "saga"      # ElasticNet은 saga만 지원함
        else:
            solver = "lbfgs"     # L2 기본값

    # 2. ElasticNet 안전장치
    if penalty == "elasticnet" and l1_ratio is None:
        raise ValueError("penalty='elasticnet' requires 'l1_ratio' to be set (0 < l1_ratio < 1).")

    return LogisticRegression(
        C=C,
        penalty=penalty,
        solver=solver,
        max_iter=max_iter,
        class_weight=class_weight,
        random_state=random_state,
        tol=tol,
        l1_ratio=l1_ratio,
        n_jobs=n_jobs
    )



