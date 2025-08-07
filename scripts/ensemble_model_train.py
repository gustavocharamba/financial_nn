import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from model.nn_model_ensemble import nn_model
from processing.load_economic import load_economic_datasets
from classifications.class_bitcoin_economic import btc_classification_economic


# 📦 1. Carregar e preparar dados
df = load_economic_datasets("BTC")
df = btc_classification_economic(df, time=3, ref=0.05)  # Garante que exista coluna "Target"

if 'Date' in df.columns:
    df = df.sort_values('Date').reset_index(drop=True)

# Separar features e target
feature_cols = [col for col in df.columns if col not in ['Date', 'Target']]
X = df[feature_cols].values
y = df['Target'].values

# ✂️ Remover NaNs
valid_mask = ~(pd.DataFrame(X).isna().any(axis=1) | pd.isna(y))
X = X[valid_mask]
y = y[valid_mask]

# 🔁 Parâmetros de validação temporal
window_size = 1260     # Ex: ~5 anos
step_size = 126        # Ex: ~6 meses
min_test_size = 63     # Ex: ~1 trimestre

max_start = len(X) - window_size - min_test_size
n_folds = min(8, max(1, max_start // step_size))
step_size = max_start // (n_folds - 1) if n_folds > 1 else step_size

results = []

# 🚀 Executar Rolling Window
for k in range(n_folds):
    start = k * step_size
    train_end = start + window_size
    test_end = train_end + step_size
    if test_end > len(X): break

    X_train, y_train = X[start:train_end], y[start:train_end]
    X_test, y_test = X[train_end:test_end], y[train_end:test_end]

    # ⚙️ Padronização
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 🧠 Modelo
    model = nn_model(input_shape=X_train.shape[1])

    early_stop = EarlyStopping(monitor='val_accuracy', patience=30, restore_best_weights=True, mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=15, min_lr=5e-5, verbose=0)

    split = int(len(X_train_scaled) * 0.85)
    history = model.fit(
        X_train_scaled[:split], y_train[:split],
        validation_data=(X_train_scaled[split:], y_train[split:]),
        epochs=300,
        batch_size=32,
        callbacks=[early_stop, reduce_lr],
        verbose=0
    )

    # 🎯 Avaliação
    test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    y_pred_probs = model.predict(X_test_scaled, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # 📋 Informações por fold
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


# 📊 Análise final
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
print(classification_report(all_y_true, all_y_pred, target_names=["Baixa", "Estável", "Alta"]))
print(f"🧮 Matriz de confusão total:\n{confusion_matrix(all_y_true, all_y_pred)}")
