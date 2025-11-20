# src/tuning.py
import pandas as pd
import numpy as np
import json
import os

# 모델 및 유틸리티 라이브러리
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 사용 모델
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier


# ==========================================
# 1. 설정 및 데이터 로드 (작성해주신 코드 그대로)
# ==========================================
def load_data(cfg):  # 타입 힌트는 선택 사항
    data_cfg = cfg["data"]
    path = data_cfg["file_path"]
    target = data_cfg["target"]

    # 경로가 맞는지 꼭 확인하세요! tuning.py 위치 기준 상대 경로입니다.
    df = pd.read_excel(path, header=1)

    if target not in df.columns:
        raise ValueError(f"target '{target}' not found")
    return df.drop(columns=[target]), df[target]


# ==========================================
# 2. 메인 실행 함수
# ==========================================
def main():
    # ---------------------------------------------------------
    # 설정(Config) 정의
    # ---------------------------------------------------------
    # ★ 중요: 여기 파일 이름이 실제 데이터 파일명과 같은지 꼭 확인하세요!
    config = {
        "data": {
            "file_path": "../default of credit card clients.xls",
            "target": "default payment next month"
        }
    }

    print(">>> 데이터를 로드합니다...")
    try:
        # 작성하신 load_data 함수 호출
        X, y = load_data(config)
        print(f"데이터 로드 성공! Shape: {X.shape}")
    except FileNotFoundError:
        print(f"오류: 파일을 찾을 수 없습니다. 경로를 확인해주세요: {config['data']['file_path']}")
        return
    except Exception as e:
        print(f"오류 발생: {e}")
        return

    # ---------------------------------------------------------
    # 데이터 분리 (Train / Validation)
    # ---------------------------------------------------------
    # Stratify를 사용하여 타겟 비율 유지
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # 불균형 데이터 가중치 계산 (XGBoost용)
    neg_count = len(y_train) - sum(y_train)
    pos_count = sum(y_train)
    scale_pos_weight_val = neg_count / pos_count

    # ==========================================
    # 3. 모델 및 파라미터 그리드 정의 (이름 수정됨!)
    # ==========================================
    model_configs = {
        # 1. config.yaml의 name: "logistic_l2" 와 일치시킴
        'logistic_l2': {
            'model': LogisticRegression(solver='liblinear', random_state=42),
            'params': {
                'clf__C': [0.01, 0.1, 1, 10, 100],
                'clf__penalty': ['l2'],  # config에서 l2를 쓴다고 했으므로 l2 위주로 탐색
                'clf__class_weight': ['balanced', None]
            }
        },

        # 2. config.yaml의 name: "random_forest_baseline" 와 일치시킴
        'random_forest_baseline': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'clf__n_estimators': [100, 200, 300, 400],  # 400이 기준이니 포함
                'clf__max_depth': [None, 10, 20],
                'clf__min_samples_split': [2, 5],
                'clf__class_weight': ['balanced', 'balanced_subsample', None]
            }
        },

        # 3. config.yaml의 name: "xgb_baseline" 와 일치시킴
        'xgb_baseline': {
            'model': XGBClassifier(eval_metric='logloss', random_state=42, use_label_encoder=False),
            'params': {
                'clf__n_estimators': [100, 300, 400],
                'clf__learning_rate': [0.01, 0.05, 0.1],
                'clf__max_depth': [3, 5, 7],
                'clf__reg_lambda': [0.1, 1.0, 10.0],  # config에 있는 파라미터 참고
                'clf__scale_pos_weight': [1, scale_pos_weight_val]
            }
        },

        # 4. config.yaml의 name: "neural_network_baseline" 와 일치시킴
        # 주의: config 파일의 파라미터(hidden_units 등)는 Keras 스타일이고,
        # 여기는 sklearn MLP 스타일이라 파라미터 이름이 다를 수 있습니다.
        # 일단 이름만이라도 맞춰서 매칭되게 합니다.
        'neural_network_baseline': {
            'model': MLPClassifier(random_state=42, early_stopping=True),
            'params': {
                'clf__hidden_layer_sizes': [(50,), (100,), (128, 64)],  # config의 구조 참고
                'clf__activation': ['relu', 'tanh'],
                'clf__alpha': [0.0001, 0.001, 0.01],
                'clf__learning_rate_init': [0.001, 0.01]
            }
        }
    }

    best_results = {}

    # models 폴더가 없으면 생성
    save_dir = '../models'
    os.makedirs(save_dir, exist_ok=True)

    # ---------------------------------------------------------
    # 튜닝 루프 실행
    # ---------------------------------------------------------
    for model_name, model_conf in model_configs.items():
        print(f"\n>>> Tuning {model_name} ...")

        # 파이프라인: 스케일링 -> 모델
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', model_conf['model'])
        ])

        search = RandomizedSearchCV(
            pipeline,
            model_conf['params'],
            n_iter=10,  # 랜덤 탐색 횟수 (늘리면 더 오래 걸리지만 정확해짐)
            scoring='f1',  # 평가 지표
            cv=3,
            n_jobs=-1,
            verbose=1,
            random_state=42
        )

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