# ------------------------------------------------------------
# 4) 모델: RandomForestClassifier
#    - class_weight="balanced"로 소수 클래스 가중 (불균형 완화)
#    - max_depth, min_samples_leaf 조절로 과적합 제어 가능(필요 시)
# ------------------------------------------------------------
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(
    n_estimators=400, # 트리 개수
    max_depth=None, # 과적합 보이면 10~20 등으로 제한
    min_samples_split=2, # 분할 최소 샘플 수
    min_samples_leaf=1, # 리프 최소 샘플 수
    max_features="sqrt",  # 각 분할 시 고려할 특성 수
    class_weight="balanced",  # 클래스 불균형 대비
    random_state=42,  # 재현성 고정
    n_jobs=-1  # 멀티코어 활용
)

# 파이프라인: 전처리 → 모델 (누설 방지, 재사용/서빙에도 용이)
'''pipe = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", clf)
])'''
