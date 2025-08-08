import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib

from model.nn_model_indicators_btc import nn_model
from processing.load_indicators import load_indicators_datasets
from classifications.class_nasdaq_indicators import nasdaq_classification_indicators

# --- Carregar e preparar dados ---
df = load_indicators_datasets("NASDAQ")
df = nasdaq_classification_indicators(df)

if 'Date' in df.columns:
    df = df.sort_values('Date').reset_index(drop=True)

feature_cols = [col for col in df.columns if col not in ['Date', 'Target']]
X = df[feature_cols].values
y = df['Target'].values

valid_mask = ~(pd.DataFrame(X).isna().any(axis=1) | pd.isna(y) | (y < 0) | (y > 1))
X = X[valid_mask]
y = y[valid_mask]

# --- Parâmetros ---
window_size = 800  # ~3 meses de dados para treino
step_size = 150     # avanço da janela para folds
min_test_size = 100 # tamanho mínimo do teste

max_start = len(X) - window_size - min_test_size
n_folds = min(10, max(1, max_start // step_size + 1))

results = []

# --- Rolling window ---
for k in range(n_folds):
    start = k * step_size
    train_end = min(start + window_size, len(X))
    test_end = min(train_end + step_size, len(X))
    if test_end - train_end < min_test_size:
        break

    X_train, y_train = X[start:train_end], y[start:train_end]
    X_test, y_test = X[train_end:test_end], y[train_end:test_end]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))

    model = nn_model(input_shape=X_train.shape[1])

    early_stop = EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True, mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=15, min_lr=5e-5, verbose=0)

    split = int(len(X_train_scaled) * 0.85)
    history = model.fit(
        X_train_scaled[:split], y_train[:split],
        validation_data=(X_train_scaled[split:], y_train[split:]),
        epochs=300,
        batch_size=16,
        callbacks=[early_stop, reduce_lr],
        class_weight=class_weight_dict,
        verbose=0
    )

    test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    y_pred_probs = model.predict(X_test_scaled, verbose=0)
    y_pred = (y_pred_probs > 0.7).astype(int).flatten()

    auc = roc_auc_score(y_test, y_pred_probs.flatten()) if len(np.unique(y_test)) > 1 else 0.7

    print(f"\n📅 Fold {k + 1}: Acurácia {test_accuracy:.4f}, Loss {test_loss:.4f}, AUC {auc:.4f}")
    print(f"Matriz: \n{confusion_matrix(y_test, y_pred)}")

    results.append({
        'accuracy': test_accuracy,
        'loss': test_loss,
        'auc': auc,
        'y_true': y_test,
        'y_pred': y_pred
    })

# --- Análise final ---
all_y_true = np.concatenate([r['y_true'] for r in results])
all_y_pred = np.concatenate([r['y_pred'] for r in results])
accuracies = [r['accuracy'] for r in results]
losses = [r['loss'] for r in results]
aucs = [r['auc'] for r in results]

print("\n" + "=" * 80)
print(f"🎯 Acurácia média: {np.mean(accuracies):.4f}")
print(f"📉 Desvio: {np.std(accuracies):.4f}")
print(f"🔍 Loss média: {np.mean(losses):.4f}")
print(f"📊 AUC médio: {np.mean(aucs):.4f}")
print(f"\nRelatório:\n{classification_report(all_y_true, all_y_pred, target_names=['Queda', 'Subida'])}")
print(f"Matriz total:\n{confusion_matrix(all_y_true, all_y_pred)}")

# --- Treinamento final e salvamento ---
print("\nTreinando final...")
final_scaler = StandardScaler().fit(X)
X_scaled = final_scaler.transform(X)
model.fit(X_scaled, y, epochs=100, batch_size=16, verbose=0, class_weight=class_weight_dict)

# Configurar caminho para a pasta saves fora da pasta atual
current_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.abspath(os.path.join(current_dir, '..', 'saves'))

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

model_path = os.path.join(save_dir, 'nasdaq_swing_model.keras')
scaler_path = os.path.join(save_dir, 'nasdaq_swing_scaler.save')

model.save(model_path)
joblib.dump(final_scaler, scaler_path)

print(f"Modelo salvo em: {model_path}")
print(f"Scaler salvo em: {scaler_path}")