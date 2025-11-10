import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 모델 구성 함수 (MLP 아키텍처)

def build_mlp(input_dim: int, dropout=0.3, lr=1e-3):
    """
    구조: Dense(128)→BN→ReLU→Dropout → Dense(64)→BN→ReLU→Dropout → Dense(32)→BN→ReLU → Dense(1)→Sigmoid
    - 목적: 비선형 분류 / 과적합 완화 / 수렴 안정
    """
    inp = keras.Input(shape=(input_dim,))
    x = layers.Dense(128, activation=None)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Dense(64, activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Dense(32, activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    out = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")]
    )
    return model
