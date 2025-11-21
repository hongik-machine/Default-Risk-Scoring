# src/tuning.py
import pandas as pd
import numpy as np
import json
import os
import sys

# 프로젝트 루트 경로에 common 모듈이 있다고 가정하고 임포트 경로를 설정
# (실행 환경에 따라 이 부분이 필요할 수 있으므로 필요 시 주석 해제하고 사용해주세요)
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 모델 및 유틸리티 라이브러리
from sklearn.model_selection import train_test_split, RandomizedSearchCV
# from sklearn.preprocessing import StandardScaler # 이제 ColumnTransformer 내에서 처리되므로 제거했습니다
from sklearn.pipeline import Pipeline

# 사용 모델
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# common/preprocessing.py에서 전처리 유틸리티 임포트
from common import preprocessing as prep_mod


# ==========================================
# 1. 설정 및 데이터 로드 (클리닝 로직 추가됨)
# ==========================================
def load_data_and_clean(cfg):
    data_cfg = cfg["data"]
    path = data_cfg["file_path"]
    target = data_cfg["target"]

    df = pd.read_excel(path, header=1)

    if target not in df.columns:
        raise ValueError(f"target '{target}' not found")

    # X, y로 분리하기 전에 clean_data를 적용하기 위해 target 컬럼을 임시로 추가
    df[target] = df[target]

    # prep_mod.clean_data 적용
    df = prep_mod.clean_data(df)

    # target 분리
    return df.drop(columns=[target]), df[target]


# ==========================================
# 2. 메인 실행 함수
# ==========================================
def main():
    # ---------------------------------------------------------
    # 설정(Config) 정의
    # ---------------------------------------------------------
    config = {
        "data": {
            "file_path": "default of credit card clients.xls",
            "target": "default payment next month"
        },
        "random_state": 42
    }

    print(">>> 데이터를 로드하고 전처리합니다...")
    try:
        # load_data_and_clean 함수 호출
        X, y = load_data_and_clean(config)
        print(f"데이터 로드 및 클리닝 성공! Shape: {X.shape}")
    except FileNotFoundError:
        print(f"오류: 파일을 찾을 수 없습니다. 경로를 확인해주세요: {config['data']['file_path']}")
        return
    except Exception as e:
        print(f"오류 발생: {e}")
        return

    # ---------------------------------------------------------
    # 데이터 분리 (Train / Validation)
    # ---------------------------------------------------------
    seed = config['random_state']
    # Stratify를 사용하여 타겟 비율 유지
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)

    # 불균형 데이터 가중치 계산 (XGBoost용)
    neg_count = len(y_train) - sum(y_train)
    pos_count = sum(y_train)
    scale_pos_weight_val = neg_count / pos_count

    # ---------------------------------------------------------
    # ColumnTransformer 기반 Preprocessor 준비
    # ---------------------------------------------------------
    preprocessor = prep_mod.make_preprocessor(X_train)

    # ==========================================
    # 3. 모델 및 파라미터 그리드 정의
    # ==========================================
    model_configs = {
        # 1. logistic_l2
        'logistic_l2': {
            'model': LogisticRegression(solver='liblinear', random_state=seed),
            'params': {
                'clf__C': [0.01, 0.1, 1, 10, 100],
                'clf__penalty': ['l2'],
                'clf__class_weight': ['balanced', None]
            }
        },

        # 2. random_forest_baseline
        'random_forest_baseline': {
            'model': RandomForestClassifier(random_state=seed),
            'params': {
                'clf__n_estimators': [100, 200, 300, 400],
                'clf__max_depth': [None, 10, 20],
                'clf__min_samples_split': [2, 5],
                'clf__class_weight': ['balanced', 'balanced_subsample', None]
            }
        },

        # 3. xgb_baseline
        'xgb_baseline': {
            'model': XGBClassifier(eval_metric='logloss', random_state=seed, use_label_encoder=False),
            'params': {
                'clf__n_estimators': [100, 300, 400],
                'clf__learning_rate': [0.01, 0.05, 0.1],
                'clf__max_depth': [3, 5, 7],
                'clf__reg_lambda': [0.1, 1.0, 10.0],
                'clf__scale_pos_weight': [1, scale_pos_weight_val]
            }
        },

        # 4. neural_network_baseline (MLPClassifier)
        'neural_network_baseline': {
            'model': MLPClassifier(random_state=seed, early_stopping=True),
            'params': {
                'clf__hidden_layer_sizes': [(50,), (100,), (128, 64)],
                'clf__activation': ['relu', 'tanh'],
                'clf__alpha': [0.0001, 0.001, 0.01],
                'clf__learning_rate_init': [0.001, 0.01]
            }
        }
    }

    best_results = {}

    # models 폴더가 없으면 생성
    save_dir = './models'
    os.makedirs(save_dir, exist_ok=True)

    # ---------------------------------------------------------
    # 튜닝 루프 실행
    # ---------------------------------------------------------
    for model_name, model_conf in model_configs.items():
        print(f"\n>>> Tuning {model_name} ...")

        # 파이프라인: 전처리 (ColumnTransformer) -> 모델
        pipeline = Pipeline([
            ('prep', preprocessor),  # 복합 전처리 단계
            ('clf', model_conf['model'])
        ])

        search = RandomizedSearchCV(
            pipeline,
            model_conf['params'],
            n_iter=10,  # 랜덤 탐색 횟수
            scoring='f1',  # 평가 지표
            cv=3,
            n_jobs=-1,
            verbose=1,
            random_state=seed
        )

        # 전처리까지 포함된 파이프라인으로 훈련 데이터에 fit
        search.fit(X_train, y_train)

        print(f"Best Score: {search.best_score_:.4f}")
        print(f"Best Params: {search.best_params_}")

        # 'clf__' 접두사 제거 후 저장
        clean_params = {k.replace('clf__', ''): v for k, v in search.best_params_.items()}

        best_results[model_name] = {
            'best_params': clean_params,
            'best_score': search.best_score_
        }

    # ---------------------------------------------------------
    # 결과 저장
    # ---------------------------------------------------------
    json_path = f"{save_dir}/best_hyperparameters.json"
    with open(json_path, 'w') as f:
        json.dump(best_results, f, indent=4)

    print(f"\n[완료] 튜닝 결과가 {json_path} 에 저장되었습니다.")


if __name__ == "__main__":
    main()