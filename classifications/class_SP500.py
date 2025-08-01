import pandas as pd

def sp500_classification(df, time=9, ref=0.035):
    var_perc = (df['Close'].shift(-time) - df['Close']) / df['Close']

    def classify_trend(x):
        if pd.isna(x):
            return None
        if x > ref:
            return 2 # UP
        elif x < -ref:
            return 0 # DOWN
        else:
            return 1 # STABLE

    df['Trend'] = var_perc.apply(classify_trend)

    # Remove all NaN cells
    df = df.dropna()

    return df
