# Default-Risk-Scoring
기계학습기초 팀플 레포입니다.



## 폴더링

```

ml-project/
├─ data/                       # 원본/가공 데이터 (필요시)
├─ common/
│  ├─ preprocessing.py         # 공통 ColumnTransformer/SMOTE 스위치 등
│  ├─ evaluation.py            # 공통 CV, 메트릭(ROC-AUC, PR-AUC, 리포트)
│  └─ utils.py                 # 공통 유틸(로깅, seed, 타이머)
├─ models/
│  ├─ logistic.py              # build_model() 구현
│  ├─ random_forest.py         # build_model() 구현
│  ├─ xgboost_lightgbm.py      # build_model() 구현(둘 다 리턴 가능)
│  └─ mlp.py                   # build_model() 구현
├─ run_benchmark.py            # 모델들 불러와 공통 평가 러너
└─ config.yaml                 # 공통 하이퍼파라미터/실험 설정(선택)

```
