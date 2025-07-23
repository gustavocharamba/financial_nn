import os
import joblib
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
from model.nn_model import build_model
from model.nn_model_deep import build_model_deep
from process import load_and_split_data
from load_datasets import load_financial_datasets
from classifications.class_bitcoin import btc_classification


def train():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    print("🔄 Carregando dados...")
    df = load_financial_datasets("BTC")
    print(f"✅ Dataset carregado: {df.shape} (linhas, colunas)")
    print(f"📊 Primeiras linhas:\n{df.head()}")

    print("\n🔄 Aplicando classificação...")
    df = btc_classification(df)
    print(f"✅ Dataset após classificação: {df.shape}")
    print(f"📈 Distribuição da coluna Trend:\n{df['Trend'].value_counts()}")

    print("\n🔄 Preparando dados para treinamento...")
    X_train, X_test, y_train, y_test, scaler = load_and_split_data(
        data_source=df,
        target_column="Trend",
        test_size=0.3,
        random_state=42,
        scale=True
    )

    print(f"✅ Shapes dos dados:")
    print(f"   - X_train: {X_train.shape}")
    print(f"   - X_test: {X_test.shape}")
    print(f"   - y_train: {y_train.shape}")
    print(f"   - y_test: {y_test.shape}")

    print(f"\n📊 Distribuição do y_train:")
    print(pd.Series(y_train).value_counts().sort_index())

    print(f"\n🔄 Construindo modelo com input_shape={X_train.shape[1]}...")
    model = build_model(input_shape=X_train.shape[1])
    print("✅ Modelo construído")
    model.summary()

    print("\n🚀 Iniciando treinamento...")
    early_stop = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=500,
        batch_size=8,
        callbacks=[early_stop],
        verbose=1
    )

    print("✅ Treinamento concluído!")

    # 📊 Métricas Finais Detalhadas
    print("\n" + "=" * 60)
    print("📊 === RELATÓRIO COMPLETO DE TREINAMENTO ===")
    print("=" * 60)

    # Métricas finais
    final_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    final_accuracy = history.history['accuracy'][-1]
    final_val_accuracy = history.history['val_accuracy'][-1]

    print(f"\n🎯 MÉTRICAS FINAIS:")
    print(f"   Loss Treino:      {final_loss:.4f}")
    print(f"   Loss Validação:   {final_val_loss:.4f}")
    print(f"   Accuracy Treino:  {final_accuracy:.4f} ({final_accuracy * 100:.2f}%)")
    print(f"   Accuracy Valid.:  {final_val_accuracy:.4f} ({final_val_accuracy * 100:.2f}%)")

    # Métricas médias
    avg_loss = sum(history.history['loss']) / len(history.history['loss'])
    avg_val_loss = sum(history.history['val_loss']) / len(history.history['val_loss'])
    avg_accuracy = sum(history.history['accuracy']) / len(history.history['accuracy'])
    avg_val_accuracy = sum(history.history['val_accuracy']) / len(history.history['val_accuracy'])

    print(f"\n📈 MÉTRICAS MÉDIAS (todas as {len(history.history['loss'])} épocas):")
    print(f"   Loss Treino Médio:      {avg_loss:.4f}")
    print(f"   Loss Validação Médio:   {avg_val_loss:.4f}")
    print(f"   Accuracy Treino Médio:  {avg_accuracy:.4f} ({avg_accuracy * 100:.2f}%)")
    print(f"   Accuracy Valid. Médio:  {avg_val_accuracy:.4f} ({avg_val_accuracy * 100:.2f}%)")

    # Melhores métricas
    best_loss_idx = history.history['val_loss'].index(min(history.history['val_loss']))
    best_acc_idx = history.history['val_accuracy'].index(max(history.history['val_accuracy']))

    print(f"\n🏆 MELHORES RESULTADOS:")
    print(f"   Melhor Loss Valid.:     {min(history.history['val_loss']):.4f} (época {best_loss_idx + 1})")
    print(
        f"   Melhor Accuracy Valid.: {max(history.history['val_accuracy']):.4f} ({max(history.history['val_accuracy']) * 100:.2f}%) (época {best_acc_idx + 1})")

    # Melhoria durante o treinamento
    initial_loss = history.history['loss'][0]
    initial_accuracy = history.history['accuracy'][0]
    initial_val_loss = history.history['val_loss'][0]
    initial_val_accuracy = history.history['val_accuracy'][0]

    loss_improvement = ((initial_loss - final_loss) / initial_loss) * 100
    accuracy_improvement = ((final_accuracy - initial_accuracy) / initial_accuracy) * 100
    val_loss_improvement = ((initial_val_loss - final_val_loss) / initial_val_loss) * 100
    val_accuracy_improvement = ((final_val_accuracy - initial_val_accuracy) / initial_val_accuracy) * 100

    print(f"\n🚀 EVOLUÇÃO DO TREINAMENTO:")
    print(f"   Épocas executadas:        {len(history.history['loss'])}/500")
    print(f"   Melhoria Loss Treino:     {loss_improvement:.2f}%")
    print(f"   Melhoria Loss Valid.:     {val_loss_improvement:.2f}%")
    print(f"   Melhoria Accuracy Treino: {accuracy_improvement:.2f}%")
    print(f"   Melhoria Accuracy Valid.: {val_accuracy_improvement:.2f}%")

    # Comparação com baseline e distribuição de classes
    print(f"\n📊 ANÁLISE DE CLASSES:")
    class_counts = pd.Series(y_train).value_counts().sort_index()
    total_samples = len(y_train)

    for class_label, count in class_counts.items():
        percentage = (count / total_samples) * 100
        print(f"   Classe {int(class_label)}: {count:4d} amostras ({percentage:.1f}%)")

    # Baseline accuracy (classe majoritária)
    majority_class_percentage = max(class_counts) / total_samples
    baseline_accuracy = 1 / len(class_counts)  # Random baseline

    print(f"\n🎯 PERFORMANCE vs BASELINE:")
    print(f"   Baseline Random:       {baseline_accuracy:.4f} ({baseline_accuracy * 100:.2f}%)")
    print(f"   Baseline Majoritária:  {majority_class_percentage:.4f} ({majority_class_percentage * 100:.2f}%)")
    print(f"   Modelo Atual:          {final_val_accuracy:.4f} ({final_val_accuracy * 100:.2f}%)")

    improvement_over_random = ((final_val_accuracy - baseline_accuracy) / baseline_accuracy) * 100
    improvement_over_majority = ((final_val_accuracy - majority_class_percentage) / majority_class_percentage) * 100

    print(f"   Melhoria sobre Random: {improvement_over_random:.1f}%")
    if improvement_over_majority > 0:
        print(f"   Melhoria sobre Maior.: {improvement_over_majority:.1f}%")
    else:
        print(f"   vs Classe Majoritária: {improvement_over_majority:.1f}%")

    # Gap entre treino e validação (overfitting check)
    accuracy_gap = final_accuracy - final_val_accuracy
    loss_gap = final_val_loss - final_loss

    print(f"\n🔍 ANÁLISE DE OVERFITTING:")
    print(f"   Gap Accuracy:  {accuracy_gap:.4f} ({accuracy_gap * 100:.2f}%)")
    print(f"   Gap Loss:      {loss_gap:.4f}")

    if accuracy_gap < 0.05 and loss_gap < 0.1:
        print("   Status: ✅ Modelo bem balanceado")
    elif accuracy_gap < 0.1 and loss_gap < 0.2:
        print("   Status: ⚠️ Leve overfitting")
    else:
        print("   Status: ❌ Overfitting detectado")

    # Salvar modelo e scaler
    print(f"\n💾 SALVANDO MODELO:")
    model_path = "saved_models/btc_trend_model.h5"
    scaler_path = "saved_models/btc_scaler.pkl"

    os.makedirs("saved_models", exist_ok=True)
    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    print(f"   Modelo salvo em: {model_path}")
    print(f"   Scaler salvo em: {scaler_path}")

    print("\n" + "=" * 60)
    print("🎉 TREINAMENTO FINALIZADO COM SUCESSO!")
    print("=" * 60)

    return history, model, scaler


if __name__ == "__main__":
    history, model, scaler = train()
    print(f"\n📊 Objetos retornados:")
    print(f"   - history: {type(history)}")
    print(f"   - model: {type(model)}")
    print(f"   - scaler: {type(scaler)}")
