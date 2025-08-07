import numpy as np

from sklearn.model_selection import train_test_split

def process_data(df, test_size=0.2, random_state=42, scale=True):
    if 'Date' in df.columns:
        df = df.drop('Date', axis=1)

    # Separar features e target
    y = df["Target"]
    X = df.drop("Target", axis=1)

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
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler
