import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')


def add_technical_indicators(df):
    """
    Adiciona indicadores técnicos completos ao dataset
    Implementação pura em pandas/numpy - SEM TA-Lib
    Compatível com seu modelo atual de Bitcoin
    """

    print("🔄 Adicionando indicadores técnicos (versão pandas)...")

    # Verificar se temos as colunas necessárias
    required_cols = ['Open', 'High', 'Low', 'Close']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Dataset deve conter: {required_cols}")

    # Extrair dados OHLC
    high = df['High'].astype(float)
    low = df['Low'].astype(float)
    close = df['Close'].astype(float)
    open_price = df['Open'].astype(float)

    # Volume (criar se não existir)
    if 'Volume' in df.columns:
        volume = df['Volume'].astype(float)
    else:
        # Volume sintético baseado na volatilidade
        volume = ((high - low) / close * 1000000).fillna(1000000)

    # === INDICADORES DE MOMENTUM ===

    # RSI (Relative Strength Index) - múltiplos períodos
    df['RSI_14'] = calculate_rsi(close, 14)
    df['RSI_7'] = calculate_rsi(close, 7)
    df['RSI_21'] = calculate_rsi(close, 21)

    # MACD (Moving Average Convergence Divergence)
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    df['MACD_Cross'] = np.where(df['MACD'] > df['MACD_Signal'], 1, 0)

    # Estocástico
    stoch_k, stoch_d = calculate_stochastic(high, low, close, 14, 3)
    df['Stoch_K'] = stoch_k
    df['Stoch_D'] = stoch_d
    df['Stoch_Cross'] = np.where(stoch_k > stoch_d, 1, 0)

    # Williams %R
    df['Williams_R'] = calculate_williams_r(high, low, close, 14)

    # Rate of Change (ROC)
    df['ROC_10'] = calculate_roc(close, 10)
    df['ROC_20'] = calculate_roc(close, 20)

    # === INDICADORES DE TENDÊNCIA ===

    # Médias Móveis Simples
    df['SMA_10'] = close.rolling(window=10).mean()
    df['SMA_20'] = close.rolling(window=20).mean()
    df['SMA_50'] = close.rolling(window=50).mean()

    # Médias Móveis Exponenciais
    df['EMA_12'] = close.ewm(span=12).mean()
    df['EMA_26'] = close.ewm(span=26).mean()
    df['EMA_50'] = close.ewm(span=50).mean()

    # Sinais de cruzamento de médias
    df['SMA_Cross_20_50'] = np.where(df['SMA_20'] > df['SMA_50'], 1, 0)
    df['EMA_Cross_12_26'] = np.where(df['EMA_12'] > df['EMA_26'], 1, 0)

    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close, 20, 2)
    df['BB_Upper'] = bb_upper
    df['BB_Middle'] = bb_middle
    df['BB_Lower'] = bb_lower
    df['BB_Width'] = (bb_upper - bb_lower) / bb_middle
    df['BB_Position'] = (close - bb_lower) / (bb_upper - bb_lower)

    # ADX (Average Directional Index)
    adx, di_plus, di_minus = calculate_adx(high, low, close, 14)
    df['ADX'] = adx
    df['DI_Plus'] = di_plus
    df['DI_Minus'] = di_minus
    df['DI_Diff'] = di_plus - di_minus

    # === INDICADORES DE VOLATILIDADE ===

    # Average True Range
    df['ATR'] = calculate_atr(high, low, close, 14)
    df['ATR_Percent'] = (df['ATR'] / close) * 100

    # Volatilidade realizada
    df['Volatility_10'] = close.pct_change().rolling(10).std() * np.sqrt(252) * 100
    df['Volatility_20'] = close.pct_change().rolling(20).std() * np.sqrt(252) * 100

    # === INDICADORES DE VOLUME ===

    # Volume médio
    df['Volume_SMA_10'] = volume.rolling(window=10).mean()
    df['Volume_SMA_20'] = volume.rolling(window=20).mean()
    df['Volume_Ratio'] = volume / df['Volume_SMA_20']

    # On Balance Volume
    df['OBV'] = calculate_obv(close, volume)
    df['OBV_SMA'] = df['OBV'].rolling(window=10).mean()

    # Money Flow Index
    df['MFI'] = calculate_mfi(high, low, close, volume, 14)

    # === INDICADORES ADICIONAIS ===

    # Commodity Channel Index
    df['CCI'] = calculate_cci(high, low, close, 14)

    # Parabolic SAR
    df['SAR'] = calculate_parabolic_sar(high, low, close, 0.02, 0.2)
    df['SAR_Signal'] = np.where(close > df['SAR'], 1, 0)

    # True Strength Index
    df['TSI'] = calculate_tsi(close)

    # Price relative to moving averages
    df['Price_vs_SMA20'] = (close - df['SMA_20']) / df['SMA_20']
    df['Price_vs_EMA50'] = (close - df['EMA_50']) / df['EMA_50']

    # Momentum personalizado
    df['Momentum_5'] = close / close.shift(5) - 1
    df['Momentum_10'] = close / close.shift(10) - 1

    # === FEATURES DERIVADAS ===

    # Combinações úteis para Bitcoin
    df['RSI_MA'] = df['RSI_14'].rolling(5).mean()  # RSI suavizado
    df['MACD_Strength'] = np.abs(df['MACD_Histogram'])  # Força do MACD
    df['BB_Squeeze'] = np.where(df['BB_Width'] < df['BB_Width'].rolling(20).mean(), 1, 0)

    # Contagem de features adicionadas
    new_features = [col for col in df.columns if col not in
                    ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Trend']]

    print(f"✅ Indicadores técnicos adicionados: {len(new_features)} novos features")
    print(f"📊 Features técnicos criados:")

    # Agrupar por categoria
    momentum_features = [f for f in new_features if
                         any(x in f for x in ['RSI', 'MACD', 'Stoch', 'Williams', 'ROC', 'Momentum'])]
    trend_features = [f for f in new_features if any(x in f for x in ['SMA', 'EMA', 'BB', 'ADX', 'DI', 'SAR'])]
    volatility_features = [f for f in new_features if any(x in f for x in ['ATR', 'Volatility'])]
    volume_features = [f for f in new_features if any(x in f for x in ['Volume', 'OBV', 'MFI'])]
    other_features = [f for f in new_features if
                      f not in momentum_features + trend_features + volatility_features + volume_features]

    print(f"   📈 Momentum: {len(momentum_features)} features")
    print(f"   📊 Tendência: {len(trend_features)} features")
    print(f"   📉 Volatilidade: {len(volatility_features)} features")
    print(f"   📦 Volume: {len(volume_features)} features")
    print(f"   🔧 Outros: {len(other_features)} features")

    return df


# === FUNÇÕES AUXILIARES PARA CÁLCULO DOS INDICADORES ===

def calculate_rsi(prices, period=14):
    """Calcula RSI usando apenas pandas"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    """Calcula Estocástico usando apenas pandas"""
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_period).mean()
    return k_percent, d_percent


def calculate_williams_r(high, low, close, period=14):
    """Calcula Williams %R"""
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    wr = -100 * ((highest_high - close) / (highest_high - lowest_low))
    return wr


def calculate_roc(prices, period=10):
    """Calcula Rate of Change"""
    return ((prices - prices.shift(period)) / prices.shift(period)) * 100


def calculate_bollinger_bands(close, period=20, std_dev=2):
    """Calcula Bollinger Bands"""
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, sma, lower_band


def calculate_atr(high, low, close, period=14):
    """Calcula Average True Range"""
    tr1 = high - low
    tr2 = np.abs(high - close.shift())
    tr3 = np.abs(low - close.shift())
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    return true_range.rolling(window=period).mean()


def calculate_adx(high, low, close, period=14):
    """Calcula ADX e Directional Indicators"""
    # True Range
    tr = calculate_atr(high, low, close, 1)

    # Directional Movement
    dm_plus = np.where((high.diff() > low.diff().abs()) & (high.diff() > 0), high.diff(), 0)
    dm_minus = np.where((low.diff().abs() > high.diff()) & (low.diff() < 0), low.diff().abs(), 0)

    # Smoothed values
    tr_smooth = pd.Series(tr).rolling(window=period).mean()
    dm_plus_smooth = pd.Series(dm_plus).rolling(window=period).mean()
    dm_minus_smooth = pd.Series(dm_minus).rolling(window=period).mean()

    # Directional Indicators
    di_plus = 100 * (dm_plus_smooth / tr_smooth)
    di_minus = 100 * (dm_minus_smooth / tr_smooth)

    # ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = dx.rolling(window=period).mean()

    return adx, di_plus, di_minus


def calculate_obv(close, volume):
    """Calcula On Balance Volume"""
    obv = np.where(close > close.shift(), volume,
                   np.where(close < close.shift(), -volume, 0))
    return pd.Series(obv).cumsum()


def calculate_mfi(high, low, close, volume, period=14):
    """Calcula Money Flow Index"""
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume

    positive_flow = np.where(typical_price > typical_price.shift(), money_flow, 0)
    negative_flow = np.where(typical_price < typical_price.shift(), money_flow, 0)

    positive_mf = pd.Series(positive_flow).rolling(window=period).sum()
    negative_mf = pd.Series(negative_flow).rolling(window=period).sum()

    mfi = 100 - (100 / (1 + (positive_mf / negative_mf)))
    return mfi


def calculate_cci(high, low, close, period=14):
    """Calcula Commodity Channel Index"""
    typical_price = (high + low + close) / 3
    sma_tp = typical_price.rolling(window=period).mean()
    mean_deviation = typical_price.rolling(window=period).apply(
        lambda x: np.mean(np.abs(x - x.mean()))
    )
    cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
    return cci


def calculate_parabolic_sar(high, low, close, af_start=0.02, af_max=0.2):
    """Calcula Parabolic SAR"""
    sar = np.zeros(len(close))
    trend = np.zeros(len(close))
    af = af_start
    ep = 0

    # Inicialização
    sar[0] = low.iloc[0]
    trend[0] = 1  # 1 para uptrend, -1 para downtrend

    for i in range(1, len(close)):
        if trend[i - 1] == 1:  # Uptrend
            sar[i] = sar[i - 1] + af * (ep - sar[i - 1])

            if high.iloc[i] > ep:
                ep = high.iloc[i]
                af = min(af + af_start, af_max)

            if low.iloc[i] <= sar[i]:
                trend[i] = -1
                sar[i] = ep
                af = af_start
                ep = low.iloc[i]
            else:
                trend[i] = 1

        else:  # Downtrend
            sar[i] = sar[i - 1] + af * (ep - sar[i - 1])

            if low.iloc[i] < ep:
                ep = low.iloc[i]
                af = min(af + af_start, af_max)

            if high.iloc[i] >= sar[i]:
                trend[i] = 1
                sar[i] = ep
                af = af_start
                ep = high.iloc[i]
            else:
                trend[i] = -1

    return pd.Series(sar, index=close.index)


def calculate_tsi(close, long_period=25, short_period=13):
    """Calcula True Strength Index"""
    momentum = close.diff()
    abs_momentum = momentum.abs()

    # Dupla suavização
    momentum_smooth1 = momentum.ewm(span=long_period).mean()
    momentum_smooth2 = momentum_smooth1.ewm(span=short_period).mean()

    abs_momentum_smooth1 = abs_momentum.ewm(span=long_period).mean()
    abs_momentum_smooth2 = abs_momentum_smooth1.ewm(span=short_period).mean()

    tsi = 100 * momentum_smooth2 / abs_momentum_smooth2
    return tsi
