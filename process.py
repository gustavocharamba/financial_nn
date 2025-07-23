import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from load_datasets import load_financial_datasets
from classifications.class_bitcoin import btc_classification

# Carrega os dados e aplica a classificação
df = load_financial_datasets("BTC")
df = btc_classification(df)


def load_and_split_data(data_source, target_column, test_size=0.2, random_state=42, scale=True):
    # Carrega dados
    if isinstance(data_source, str):
        df = pd.read_csv(data_source)
    else:
        df = data_source.copy()

    # ✅ CRUCIAL: Remover coluna Date ANTES de qualquer processamento
    if 'Date' in df.columns:
        print(f"🗓️ Removendo coluna Date")
        df = df.drop('Date', axis=1)

    # Verificar tipos de dados restantes
    print(f"📊 Tipos de dados após remoção da Date:")
    print(df.dtypes)

    # Remover outras colunas não-numéricas se existirem
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_columns) != len(df.columns):
        non_numeric = [col for col in df.columns if col not in numeric_columns]
        print(f"⚠️ Colunas não-numéricas encontradas: {non_numeric}")
        df = df[numeric_columns]

    # Separar features e target
    y = df[target_column]
    X = df.drop(target_column, axis=1)

    print(f"✅ Dados preparados:")
    print(f"   - Features (X): {X.shape} - {X.dtypes.unique()}")
    print(f"   - Target (y): {y.shape} - {y.dtype}")

    # Divisão treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )

    # Normalização (apenas se solicitada)
    scaler = None
    if scale:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        print("🔄 Aplicando normalização...")
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        print("✅ Normalização aplicada")

    return X_train, X_test, y_train, y_test, scaler