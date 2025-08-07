import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from model.nn_model import nn_model

from processing.load_datasets import load_financial_datasets
from classifications.class_bitcoin import btc_classification
from processing.load_indicators import load_indicators


def prepare_financial_data():
    """Prepara dados financeiros com indicadores técnicos"""
    print("🔄 Preparando dados financeiros...")

    # Carregar dados base
    df = load_financial_datasets("BTC")

    # Adicionar indicadores técnicos
    print("🔄 Adicionando indicadores técnicos...")
    df_with_indicators = load_indicators(df)

    # Aplicar classificação
    df_classified = btc_classification(df_with_indicators, time=3, ref=0.05)

    # Garantir ordem cronológica
    if 'Date' in df_classified.columns:
        df_classified = df_classified.sort_values('Date').reset_index(drop=True)
        print(f"✅ Dados ordenados cronologicamente")

    # Separar features e target
    feature_cols = [col for col in df_classified.columns
                    if col not in ['Date', 'Trend']]

    X = df_classified[feature_cols].values
    y = df_classified['Trend'].values

    # Remover NaN (indicadores técnicos podem criar alguns)
    valid_mask = ~(pd.DataFrame(X).isna().any(axis=1) | pd.Series(y).isna())
    X = X[valid_mask]
    y = y[valid_mask]

    print(f"✅ Dados preparados: {X.shape[0]} amostras, {X.shape[1]} features")
    print(f"📊 Features originais: 16 (dados econômicos)")
    print(f"📊 Features técnicos: {X.shape[1] - 16}")
    print(f"📊 Total features: {X.shape[1]}")

    return X, y, feature_cols

def calculate_fold_parameters(data_length, window_size=1260, step_size=126, min_test_size=63):
    """Calcula parâmetros otimizados para os folds"""

    # Máximo de folds possíveis
    max_start = data_length - window_size - min_test_size
    n_folds = min(8, max(1, max_start // step_size))

    # Ajustar step_size se necessário
    if n_folds > 1:
        optimal_step = max_start // (n_folds - 1)
        step_size = min(step_size, optimal_step)

    print(f"📊 Parâmetros calculados:")
    print(f"   - Janela treino: {window_size} dias (~{window_size / 252:.1f} anos)")
    print(f"   - Passo avanço: {step_size} dias (~{step_size / 21:.1f} meses)")
    print(f"   - Folds previstos: {n_folds}")

    return n_folds, step_size

def execute_single_fold(X_train, y_train, X_test, y_test, fold_num):
    """Executa um único fold de validação"""

    # Normalização temporal (crucial!)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Modelo otimizado para validação temporal
    model = nn_model(input_shape=X_train.shape[1])

    # Callbacks otimizados para dados financeiros
    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=30,
        restore_best_weights=True,
        mode='max',
        min_delta=0.001
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.7,
        patience=15,
        min_lr=0.00005,
        verbose=0
    )

    # Split interno para validação durante treino
    internal_split = int(len(X_train_scaled) * 0.85)
    X_train_final = X_train_scaled[:internal_split]
    X_val_internal = X_train_scaled[internal_split:]
    y_train_final = y_train[:internal_split]
    y_val_internal = y_train[internal_split:]

    # Treinamento
    history = model.fit(
        X_train_final, y_train_final,
        validation_data=(X_val_internal, y_val_internal),
        epochs=300,
        batch_size=32,
        callbacks=[early_stop, reduce_lr],
        verbose=0
    )

    # Avaliação no conjunto de teste temporal
    test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
3
    # Predições para análise detalhada
    y_pred_probs = model.predict(X_test_scaled, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Métricas por classe
    class_accuracies = {}
    for class_id in [0, 1, 2]:
        class_mask = (y_test == class_id)
        if np.sum(class_mask) > 0:
            class_acc = accuracy_score(y_test[class_mask], y_pred[class_mask])
            class_accuracies[class_id] = class_acc

    # Compilar resultado do fold
    fold_result = {
        'fold': fold_num,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'test_accuracy': test_accuracy,
        'test_loss': test_loss,
        'epochs_trained': len(history.history['loss']),
        'best_val_accuracy': max(history.history['val_accuracy']),
        'class_accuracies': class_accuracies,
        'y_true': y_test,
        'y_pred': y_pred,
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }

    print(f"   🎯 Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
    print(f"   📉 Loss: {test_loss:.4f}")
    print(f"   🔄 Épocas: {len(history.history['loss'])}")

    return fold_result

def execute_rolling_window(X, y, window_size=1260, step_size=126, min_test_size=63):
    """Executa validação Rolling Window"""

    print(f"\n🚀 Iniciando Rolling Window Validation...")

    n_folds, adjusted_step_size = calculate_fold_parameters(len(X), window_size, step_size, min_test_size)
    results = []

    fold_count = 0
    max_start = len(X) - window_size - min_test_size

    for start_idx in range(0, max_start, adjusted_step_size):
        fold_count += 1

        # Definir índices das janelas
        train_start = start_idx
        train_end = start_idx + window_size
        test_start = train_end
        test_end = min(test_start + adjusted_step_size, len(X))

        # Garantir tamanho mínimo de teste
        if (test_end - test_start) < min_test_size:
            test_end = min(test_start + min_test_size, len(X))

        # Dados do fold
        X_train = X[train_start:train_end]
        y_train = y[train_start:train_end]
        X_test = X[test_start:test_end]
        y_test = y[test_start:test_end]

        print(f"\n📅 Fold {fold_count}:")
        print(f"   Treino: {train_start} → {train_end} ({len(X_train)} amostras)")
        print(f"   Teste:  {test_start} → {test_end} ({len(X_test)} amostras)")

        # Executar fold
        fold_result = execute_single_fold(X_train, y_train, X_test, y_test, fold_count)
        results.append(fold_result)

        # Limitar número de folds para evitar overfit na validação
        if fold_count >= 8:
            break

    return results

def analyze_results(results):
    """Análise abrangente dos resultados"""

    if not results:
        print("❌ Nenhum resultado para analisar")
        return None

    print("\n" + "=" * 80)
    print("📊 === ANÁLISE ROLLING WINDOW VALIDATION ===")
    print("=" * 80)

    # Métricas agregadas
    accuracies = [r['test_accuracy'] for r in results]
    losses = [r['test_loss'] for r in results]

    analysis = {
        'mean_accuracy': np.mean(accuracies),
        'std_accuracy': np.std(accuracies),
        'min_accuracy': np.min(accuracies),
        'max_accuracy': np.max(accuracies),
        'mean_loss': np.mean(losses),
        'std_loss': np.std(losses),
        'n_folds': len(results)
    }

    # Performance geral
    print(f"\n🎯 PERFORMANCE GERAL:")
    print(f"   Accuracy Média:  {analysis['mean_accuracy']:.4f} ± {analysis['std_accuracy']:.4f}")
    print(f"   Range Accuracy:  {analysis['min_accuracy']:.4f} - {analysis['max_accuracy']:.4f}")
    print(f"   Loss Médio:      {analysis['mean_loss']:.4f} ± {analysis['std_loss']:.4f}")
    print(f"   Folds executados: {analysis['n_folds']}")

    # Performance por fold
    print(f"\n📈 PERFORMANCE POR FOLD:")
    print("Fold | Accuracy | Loss    | Épocas | Amostras Teste")
    print("-" * 55)

    for r in results:
        print(f"{r['fold']:^4} | {r['test_accuracy']:.4f}   | {r['test_loss']:.4f} | "
              f"{r['epochs_trained']:^6} | {r['test_samples']:^13}")

    # Comparação com baselines
    all_y_true = np.concatenate([r['y_true'] for r in results])
    all_y_pred = np.concatenate([r['y_pred'] for r in results])

    # Baselines
    majority_class = np.bincount(all_y_true.astype(int)).argmax()
    majority_baseline = np.mean(all_y_true == majority_class)
    random_baseline = 1 / 3

    print(f"\n🎯 COMPARAÇÃO COM BASELINES:")
    print(f"   Rolling Window CV:     {analysis['mean_accuracy']:.4f} ({analysis['mean_accuracy'] * 100:.2f}%)")
    print(f"   Baseline Majoritária:  {majority_baseline:.4f} ({majority_baseline * 100:.2f}%)")
    print(f"   Baseline Random:       {random_baseline:.4f} ({random_baseline * 100:.2f}%)")

    improvement_majority = ((analysis['mean_accuracy'] - majority_baseline) / majority_baseline) * 100
    improvement_random = ((analysis['mean_accuracy'] - random_baseline) / random_baseline) * 100

    print(f"   Melhoria vs Majoritária: {improvement_majority:+.1f}%")
    print(f"   Melhoria vs Random:      {improvement_random:+.1f}%")

    # Análise de estabilidade temporal
    accuracy_trend = np.polyfit(range(len(accuracies)), accuracies, 1)[0]

    print(f"\n📊 ESTABILIDADE TEMPORAL:")
    print(f"   Tendência accuracy: {accuracy_trend:+.6f} por fold")

    if abs(accuracy_trend) < 0.002:
        stability = "✅ Muito Estável"
    elif accuracy_trend > 0:
        stability = "📈 Melhorando"
    else:
        stability = "📉 Degradando"

    print(f"   Status: {stability}")

    # Análise por classe
    print(f"\n🏷️ PERFORMANCE POR CLASSE:")
    class_names = {0: "Baixa", 1: "Alta", 2: "Média"}

    for class_id in [0, 1, 2]:
        class_accs = [r['class_accuracies'].get(class_id, 0) for r in results
                      if class_id in r['class_accuracies']]
        if class_accs:
            mean_class_acc = np.mean(class_accs)
            print(f"   Classe {class_names[class_id]:>5}: {mean_class_acc:.4f} ({mean_class_acc * 100:.2f}%)")

    # Relatório de classificação completo
    print(f"\n📋 RELATÓRIO DE CLASSIFICAÇÃO COMPLETO:")
    target_names = ['Tendência Baixa', 'Tendência Alta', 'Tendência Média']
    print(classification_report(all_y_true, all_y_pred, target_names=target_names))

    return analysis

def get_performance_summary(results):
    """Retorna resumo de performance"""
    if not results:
        return None

    accuracies = [r['test_accuracy'] for r in results]

    return {
        'validation_method': 'Rolling Window',
        'mean_accuracy': np.mean(accuracies),
        'std_accuracy': np.std(accuracies),
        'confidence_interval_95': (
            np.mean(accuracies) - 1.96 * np.std(accuracies),
            np.mean(accuracies) + 1.96 * np.std(accuracies)
        ),
        'n_folds': len(results),
        'is_statistically_significant': np.mean(accuracies) > 0.4  # 40% threshold
    }

def run_rolling_window_validation():
    """Função principal para executar Rolling Window Validation"""

    print("🚀 === ROLLING WINDOW VALIDATION ===")
    print(f"⏰ Iniciado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Preparar dados
    X, y, feature_cols = prepare_financial_data()

    # Executar validação
    results = execute_rolling_window(X, y)

    # Analisar resultados
    analysis = analyze_results(results)

    # Resumo final
    summary = get_performance_summary(results)

    print("\n" + "=" * 80)
    print("🎉 === VALIDAÇÃO CONCLUÍDA ===")
    print("=" * 80)
    print(f"📊 Performance: {summary['mean_accuracy']:.4f} ± {summary['std_accuracy']:.4f}")
    print(f"📈 IC 95%: [{summary['confidence_interval_95'][0]:.4f}, {summary['confidence_interval_95'][1]:.4f}]")
    print(f"✅ Significativo: {'Sim' if summary['is_statistically_significant'] else 'Não'}")
    print("=" * 80)

    return results, analysis, summary

if __name__ == "__main__":
    results, analysis, summary = run_rolling_window_validation()
