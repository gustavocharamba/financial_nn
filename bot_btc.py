import os
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import plotly.graph_objects as go

from indicators.adx import getADX
from indicators.bollinger import getBollinger
from indicators.macd import getMACD
from indicators.rsi import getRSI
from indicators.sma import getSMA
from indicators.stochastic import getStochastic

def fix_series_1d(df_ind):
    for col in df_ind.columns:
        if isinstance(df_ind[col].values, np.ndarray) and df_ind[col].values.ndim > 1:
            df_ind[col] = df_ind[col].values.flatten()
    return df_ind

def build_indicators(df):
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    adx        = fix_series_1d(getADX(df))
    bollinger  = fix_series_1d(getBollinger(df))
    macd       = fix_series_1d(getMACD(df))
    rsi        = fix_series_1d(getRSI(df))
    sma        = fix_series_1d(getSMA(df))
    stochastic = fix_series_1d(getStochastic(df))
    indicators_dfs = [adx, bollinger, macd, rsi, sma, stochastic]
    indicators_cols = [col for indi_df in indicators_dfs for col in indi_df.columns]
    df_full = pd.concat([df[['Open', 'High', 'Low', 'Close', 'Volume']]] + indicators_dfs, axis=1)
    df_full[indicators_cols] = df_full[indicators_cols].shift(1)
    df_full = df_full.resample('D').ffill()
    df_full = df_full.dropna().reset_index()
    extras = {'Feature_Extra1': 0, 'Feature_Extra2': 0, 'Feature_Extra3': 0, 'Feature_Extra4': 0, 'Feature_Extra5': 0}
    for feat, val in extras.items():
        df_full[feat] = val
    return df_full

def backtesting_with_yfinance():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(current_dir, 'saves')
    model_path = os.path.join(save_dir, 'btc_swing_model.keras')
    scaler_path = os.path.join(save_dir, 'btc_swing_scaler.save')
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)

    print("Baixando dados do BTC com yfinance...")
    btc_data = yf.download("BTC-USD", period="5y", interval="1d", auto_adjust=True)
    btc_data.columns = [col if not isinstance(col, tuple) else col[0] for col in btc_data.columns]
    print("Calculando indicadores técnicos...")
    df_features = build_indicators(btc_data)

    feature_cols = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'ADX', '+DI', '-DI', 'BB_Middle', 'BB_Upper', 'BB_Lower',
        'MACD', 'MACD_Signal', 'RSI', 'SMA_3', 'SMA_7', 'SMA_21', 'SMA_50',
        'Stoch_K', 'Stoch_D'
    ]
    X = df_features[feature_cols].values

    print("\n==== COLUNAS DO BACKTESTING ====")
    print(feature_cols)
    print(f"Shape de X do backtest: {X.shape}")

    try:
        with open('features_train.txt') as f:
            features_train = [line.strip() for line in f.readlines()]
        print("\n==== COLUNAS DO TREINAMENTO (salvas em features_train.txt) ====")
        print(features_train)
        print(f"Número de colunas do treinamento: {len(features_train)}")
        print("\n== DIFERENÇAS ENTRE TREINAMENTO E BACKTEST ==")
        print("No treino e NÃO no backtest:", set(features_train) - set(feature_cols))
        print("No backtest e NÃO no treino:", set(feature_cols) - set(features_train))
    except Exception as e:
        print("\n[AVISO] Não foi possível comparar as features do treino:", e)

    print("\nNormalizando dados...")
    X_scaled = scaler.transform(X)
    print("Gerando previsões...")
    y_pred_probs = model.predict(X_scaled, verbose=0).flatten()
    y_pred = (y_pred_probs > 0.5).astype(int)
    df_features['Pred_Class'] = y_pred
    df_features['Date'] = pd.to_datetime(df_features['Date'])
    df_features = df_features.reset_index(drop=True)

    # ==== Estratégia: Alavancada + Stop Gain/Stop Loss (com lógica corrigida) ====
    start_balance = 10000.0
    position_size = 3000.0
    leverage = 3
    fee_pct = 0.001
    target_profit_pct = 0.3
    stop_loss_pct = 0.1

    balance = start_balance
    btc_units = 0.0
    last_buy_price = 0.0
    margin_held = 0.0
    in_trade = False

    buy_dates = []
    buy_prices = []
    sell_dates = []
    sell_prices = []
    trade_results = []
    portfolio_value_history = []

    for i, row in df_features.iterrows():
        price = row['Close']
        signal = row['Pred_Class']

        action_happened = False

        # VENDA: por stop gain ou stop loss (alavancado)
        if in_trade:
            current_profit = (price - last_buy_price) / last_buy_price
            if current_profit >= target_profit_pct or current_profit <= -stop_loss_pct:
                sell_value = btc_units * price * (1 - fee_pct)
                total_position_cost = leverage * position_size * (1 + fee_pct)
                profit = sell_value - total_position_cost
                balance += margin_held + profit
                sell_dates.append(row['Date'])
                sell_prices.append(price)
                trade_results.append(profit)
                btc_units = 0
                margin_held = 0
                in_trade = False
                action_happened = True  # Impede nova ação no mesmo candle

        # COMPRA: sinal do modelo, abre posição alavancada (somente se não teve venda nesse candle)
        if not in_trade and signal == 1 and not action_happened:
            buy_value = leverage * position_size * (1 + fee_pct)
            btc_units = (leverage * position_size) / price * (1 - fee_pct)
            margin_held = position_size * (1 + fee_pct)
            balance -= margin_held
            last_buy_price = price
            in_trade = True
            buy_dates.append(row['Date'])
            buy_prices.append(price)
            if balance < 0:
                print(f"\n[LIQUIDAÇÃO] Saldo negativo em {row['Date']}. Resetando saldo para zero.")
                balance = 0

        # Atualiza patrimônio considerando margem, saldo e posição
        current_position_value = btc_units * price if in_trade else 0
        total_value = balance + margin_held + current_position_value
        portfolio_value_history.append(total_value)

    # Encerra posição aberta no último preço
    if in_trade:
        price = df_features['Close'].iloc[-1]
        sell_value = btc_units * price * (1 - fee_pct)
        total_position_cost = leverage * position_size * (1 + fee_pct)
        profit = sell_value - total_position_cost
        balance += margin_held + profit
        sell_dates.append(df_features['Date'].iloc[-1])
        sell_prices.append(price)
        trade_results.append(profit)

    df_features['Portfolio_Value'] = portfolio_value_history
    total_return = (portfolio_value_history[-1] / start_balance - 1)
    if len(portfolio_value_history) > 1:
        max_drawdown = max(
            (max(portfolio_value_history[:i+1]) - v) / max(portfolio_value_history[:i+1])
            for i, v in enumerate(portfolio_value_history)
        )
    else:
        max_drawdown = 0

    win_trades = [g for g in trade_results if g > 0]
    loss_trades = [g for g in trade_results if g <= 0]
    win_rate = len(win_trades) / len(trade_results) if trade_results else 0

    print(f"\nRetorno total no período: {total_return*100:.2f}%")
    print(f"Trades finalizados: {len(trade_results)}")
    print(f"Win rate: {win_rate*100:.2f}%")
    print(f"Lucro médio trade vencedores: {np.mean(win_trades) if win_trades else 0:.2f}")
    print(f"Prejuízo médio trade perdedores: {np.mean(loss_trades) if loss_trades else 0:.2f}")
    print(f"Máximo drawdown: {max_drawdown*100:.2f}%")

    # --------- PLOT INTERATIVO: CANDLESTICK + ZOOM, FUNDO PRETO -----------
    df_features['Date_str'] = df_features['Date'].dt.strftime('%Y-%m-%d')

    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df_features['Date_str'],
        open=df_features['Open'],
        high=df_features['High'],
        low=df_features['Low'],
        close=df_features['Close'],
        name='Candlestick BTC'
    ))

    # Sinais de compra: 1% acima do close para não colar no venda
    if buy_dates:
        buy_points = df_features[df_features['Date'].isin(buy_dates)]
        fig.add_trace(go.Scatter(
            x=buy_points['Date_str'],
            y=buy_points['Close'] * 1.01,
            mode='markers', name='Compra',
            marker=dict(symbol='triangle-up', size=12, color='limegreen', opacity=0.93)
        ))

    # Sinais de venda: 1% abaixo do close para não colar no compra
    if sell_dates:
        sell_points = df_features[df_features['Date'].isin(sell_dates)]
        fig.add_trace(go.Scatter(
            x=sell_points['Date_str'],
            y=sell_points['Close'] * 0.99,
            mode='markers', name='Venda',
            marker=dict(symbol='triangle-down', size=12, color='red', opacity=0.93)
        ))

    fig.update_layout(
        title='BTC Swing Trade Alavancado (stop gain/stop loss) — Interativo',
        xaxis_title='Data',
        yaxis_title='Preço (USD)',
        xaxis_rangeslider_visible=False,
        legend=dict(font=dict(size=14), bgcolor='rgba(0,0,0,0.3)'),
        height=800,
        template='plotly_dark'
    )

    fig.show()

if __name__ == "__main__":
    backtesting_with_yfinance()
