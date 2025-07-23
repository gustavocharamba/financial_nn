import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

def build_model(input_shape):
    model = Sequential([
        Dense(2048, activation='relu', input_shape=(input_shape,)),
        Dropout(0.15),
        Dense(1024, activation='relu'),
        Dropout(0.1),
        Dense(3, activation='softmax')  # ✅ 3 classes para classificação
    ])

    optimizer = Adam(learning_rate=0.00005)

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',  # ✅ Para classificação multi-classe
        metrics=['accuracy']
    )

    return model
