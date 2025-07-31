import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam


def nn_model(input_shape):

    model = Sequential([
        Input(shape=(input_shape,)),
        Dense(512, activation='relu'),
        Dropout(0.45),
        Dense(256, activation='relu'),
        Dropout(0.4),
        Dense(128, activation='relu'),
        Dropout(0.35),
        Dense(3, activation='softmax')
    ])

    optimizer = Adam(learning_rate=0.00001)

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
