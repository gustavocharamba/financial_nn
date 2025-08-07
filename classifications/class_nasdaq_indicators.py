import pandas as pd

def nasdaq_classification_indicators(df, time=5, ref=0.025):
    var_perc = (df['Close'].shift(-time) - df['Close']) / df['Close']

    def classify_trend(x):
        if pd.isna(x):
            return None
        if x > ref:
            return 1 # UP
        elif x < -ref:
            return 0 # DOWN

    df['Target'] = var_perc.apply(classify_trend)

    # Remove all NaN cells
    df = df.dropna()

    return df
