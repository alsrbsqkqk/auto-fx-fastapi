import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math

# ====== 지표 계산 ======
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = -delta.clip(upper=0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series):
    ema12 = series.ewm(span=12).mean()
    ema26 = series.ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    return macd, signal

def calculate_stoch_rsi(rsi, period=14):
    min_rsi = rsi.rolling(window=period).min()
    max_rsi = rsi.rolling(window=period).max()
    return (rsi - min_rsi) / (max_rsi - min_rsi)

def calculate_bollinger_bands(series, window=20):
    mid = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper = mid + 2 * std
    lower = mid - 2 * std
    return upper, mid, lower

def detect_candle_pattern(row):
    body = abs(row['close'] - row['open'])
    upper_wick = row['high'] - max(row['close'], row['open'])
    lower_wick = min(row['close'], row['open']) - row['low']
    if lower_wick > 2 * body and upper_wick < body:
        return "HAMMER"
    elif upper_wick > 2 * body and lower_wick < body:
        return "SHOOTING_STAR"
    return "NEUTRAL"

def detect_trend(df):
    ema20 = df['close'].ewm(span=20).mean()
    ema50 = df['close'].ewm(span=50).mean()
    mid = df['boll_mid']
    latest = df.iloc[-1]
    if ema20.iloc[-1] > ema50.iloc[-1] and latest['close'] > mid.iloc[-1]:
        return "UPTREND"
    elif ema20.iloc[-1] < ema50.iloc[-1] and latest['close'] < mid.iloc[-1]:
        return "DOWNTREND"
    return "NEUTRAL"

# ====== 실전 예약주문 시뮬레이션 엔진 ======
def backtest(df):
    df['rsi'] = calculate_rsi(df['close'])
    df['macd'], df['macd_signal'] = calculate_macd(df['close'])
    df['stoch_rsi'] = calculate_stoch_rsi(df['rsi'])
    df['boll_up'], df['boll_mid'], df['boll_low'] = calculate_bollinger_bands(df['close'])
    df['pattern'] = df.apply(detect_candle_pattern, axis=1)

    results = []
    active_orders = []
    last_trade_time = None

    for i in range(50, len(df)):
        row = df.iloc[i]
        now_time = pd.to_datetime(row['time'])
        atlanta_hour = (now_time - timedelta(hours=4)).hour
        
        if atlanta_hour >= 22 or atlanta_hour <= 6:
            continue  # 거래 제한시간 필터
        
        trend = detect_trend(df.iloc[i-50:i])
        signal = None

        if row['rsi'] < 35 and row['pattern'] == 'HAMMER':
            signal = 'BUY'
        elif row['rsi'] > 70 and row['pattern'] == 'SHOOTING_STAR':
            signal = 'SELL'
        
        # MACD 필터
        if signal == 'BUY' and (row['macd'] - row['macd_signal']) < -0.001:
            signal = None
        if signal == 'SELL' and (row['macd_signal'] - row['macd']) < -0.001:
            signal = None
        
        # 최소 2시간 간격
        if signal and (last_trade_time is None or (now_time - last_trade_time) >= timedelta(hours=2)):
            entry_price = row['close']
            tp = entry_price + 0.0010 if signal == 'BUY' else entry_price - 0.0010
            sl = entry_price - 0.0007 if signal == 'BUY' else entry_price + 0.0007
            active_orders.append({
                'time': row['time'], 'signal': signal, 'entry': entry_price, 'tp': tp, 'sl': sl, 'status': 'open'
            })
            last_trade_time = now_time

        # 기존 활성 주문 처리 (예약주문)
        for order in active_orders:
            if order['status'] == 'closed':
                continue
            price = row['close']
            if order['signal'] == 'BUY':
                if price >= order['tp']:
                    order['status'] = 'tp'
                elif price <= order['sl']:
                    order['status'] = 'sl'
            if order['signal'] == 'SELL':
                if price <= order['tp']:
                    order['status'] = 'tp'
                elif price >= order['sl']:
                    order['status'] = 'sl'

    # 결과 저장
    for order in active_orders:
        results.append(order)
    
    return pd.DataFrame(results)

# ====== 실행 ======
data = pd.read_csv('backtest_data.csv')
data['time'] = pd.to_datetime(data['time'])
result = backtest(data)

print(result)
result.to_csv("backtest_result.csv", index=False)
