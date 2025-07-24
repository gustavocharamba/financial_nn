import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam


def build_model(input_shape, complexity='complex'):

    if complexity == 'simple':
        # Modelo simplificado para validação temporal (poucos dados)
        model = Sequential([
            Input(shape=(input_shape,)),
            Dense(256, activation='relu'),
            Dropout(0.4),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(3, activation='softmax')
        ])
        learning_rate = 0.001

    elif complexity == 'medium':
        # Modelo balanceado
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
        learning_rate = 0.00001

    else:
        # Modelo complexo original
        model = Sequential([
            Input(shape=(input_shape,)),
            Dense(2048, activation='relu'),
            Dropout(0.15),
            Dense(1024, activation='relu'),
            Dropout(0.1),
            Dense(3, activation='softmax')
        ])
        learning_rate = 0.00007

    optimizer = Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']  # ✅ Apenas accuracy por enquanto
    )

    return model
