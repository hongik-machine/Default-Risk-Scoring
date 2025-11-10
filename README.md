# Default-Risk-Scoring
기계학습기초 팀플 레포입니다.



## 폴더링

```

ml-project/
├─ data/                       # 원본/가공 데이터 (필요시)
├─ common/
│  ├─ preprocessing.py         # 공통 ColumnTransformer/SMOTE 스위치 등
│  └─ evaluation.py            # 공통 CV, 메트릭(ROC-AUC, PR-AUC, 리포트)
│  
├─ models/
│  ├─ logistic.py              # build_model() 구현
│  ├─ random_forest.py         # build_model() 구현
│  ├─ xgboost_lightgbm.py      # build_model() 구현(둘 다 리턴 가능)
│  └─ mlp.py                   # build_model() 구현
├─ run_benchmark.py            # 모델들 불러와 공통 평가 러너
└─ config.yaml                 # 공통 하이퍼파라미터/실험 설정(선택)

```


---

### Commit 컨벤션

- **feat**: 새로운 기능 추가
- **fix**: 버그 수정
- **docs**: 문서 변경 (예: README 수정)
- **style**: 코드 포맷팅, 세미콜론 누락 등 비즈니스 로직에 영향을 주지 않는 변경
- **refactor**: 코드 리팩토링
- **test**: 테스트 추가 또는 기존 테스트 수정
- 🔧: 빌드 프로세스 또는 보조 도구와 관련된 변경 (예: 패키지 매니저 설정)

---

## 🔖 브랜치 컨벤션
* `main` - main 브랜치
* `feat/xx` - 모델 단위로 독립적인 개발 환경을 위해 작성
* `refac/xx` - 개발된 기능을 리팩토링 하기 위해 작성
* `chore/xx` - 빌드 작업, 패키지 매니저 설정 등


