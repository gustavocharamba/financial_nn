import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model
import joblib  # Para salvar o scaler

from model.nn_model_ensemble import nn_model  # Assumindo que isso é o seu modelo
from processing.load_economic import load_economic_datasets
from classifications.class_bitcoin_economic import btc_classification_economic

# 1. Carregar e preparar dados
df = load_economic_datasets("BTC")
df = btc_classification_economic(df)  # Garante coluna 'Target'

if 'Date' in df.columns:
    df = df.sort_values('Date').reset_index(drop=True)

# Separar features e target
feature_cols = [col for col in df.columns if col not in ['Date', 'Target']]
X = df[feature_cols].values
y = df['Target'].values

# Remapear rótulos: -1 (estável) -> 0, 0 (queda) -> 1, 1 (subida) -> 2
y = np.where(y == -1, 0, np.where(y == 0, 1, np.where(y == 1, 2, y)))

# Remover NaNs e rótulos inválidos (se houver fora de 0-2)
valid_mask = ~(pd.DataFrame(X).isna().any(axis=1) | pd.isna(y) | (y < 0) | (y > 2))
X = X[valid_mask]
y = y[valid_mask]

# Imprimir distribuição de rótulos
print("Distribuição de rótulos após remapeamento:", pd.Series(y).value_counts())

# 2. Parâmetros de validação rolling window
window_size = 1260
step_size = 126
min_test_size = 63

max_start = len(X) - window_size - min_test_size
n_folds = min(8, max(1, max_start // step_size))
step_size = max_start // (n_folds - 1) if n_folds > 1 else step_size

results = []

# 3. Executar rolling window
for k in range(n_folds):
    start = k * step_size
    train_end = start + window_size
    test_end = train_end + step_size
    if test_end > len(X):
        break

    X_train, y_train = X[start:train_end], y[start:train_end]
    X_test, y_test = X[train_end:test_end], y[train_end:test_end]

    # Padronização
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Calcular pesos de classes para lidar com desbalanceamento
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))

    # Modelo
    model = nn_model(input_shape=X_train.shape[1])

    early_stop = EarlyStopping(monitor='val_accuracy', patience=50, restore_best_weights=True, mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=15, min_lr=5e-5, verbose=0)

    split = int(len(X_train_scaled) * 0.85)
    history = model.fit(
        X_train_scaled[:split], y_train[:split],
        validation_data=(X_train_scaled[split:], y_train[split:]),
        epochs=300,
        batch_size=32,
        callbacks=[early_stop, reduce_lr],
        class_weight=class_weight_dict,  # Adicionado para desbalanceamento
        verbose=0
    )

    # Avaliação
    test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    y_pred_probs = model.predict(X_test_scaled, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Impressões por fold
    print(f"\n📅 Fold {k+1}")
    print(f"   Treino: {len(X_train)} amostras | Teste: {len(X_test)} amostras")
    print(f"   Accuracy teste: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
    print(f"   Loss teste:     {test_loss:.4f}")
    print(f"   Épocas:         {len(history.history['loss'])}")
    print(f"   Matriz de confusão:\n{confusion_matrix(y_test, y_pred)}")

    results.append({
        'accuracy': test_accuracy,
        'loss': test_loss,
        'epochs': len(history.history['loss']),
        'y_true': y_test,
        'y_pred': y_pred
    })

# 4. Análise final
all_y_true = np.concatenate([r['y_true'] for r in results])
all_y_pred = np.concatenate([r['y_pred'] for r in results])
accuracies = [r['accuracy'] for r in results]
losses = [r['loss'] for r in results]

print("\n" + "="*80)
print(f"🎯 Acurácia média:  {np.mean(accuracies):.4f}")
print(f"📉 Desvio padrão:   {np.std(accuracies):.4f}")
print(f"📈 Melhor acurácia: {np.max(accuracies):.4f}")
print(f"📉 Pior acurácia:   {np.min(accuracies):.4f}")
print(f"\n📋 Relatório classificação:")
print(classification_report(all_y_true, all_y_pred, target_names=["Estável", "Queda", "Subida"]))
print(f"🧮 Matriz de confusão total:\n{confusion_matrix(all_y_true, all_y_pred)}")

# 5. Treinamento final no dataset completo e salvamento para previsões futuras
print("\nTreinando modelo final no dataset completo...")
final_scaler = StandardScaler().fit(X)
X_scaled = final_scaler.transform(X)
model.fit(X_scaled, y, epochs=100, batch_size=32, verbose=0, class_weight=class_weight_dict)
model.save('btc_model.h5')
joblib.dump(final_scaler, 'btc_scaler.save')
print("Modelo e scaler salvos para previsões futuras!")
