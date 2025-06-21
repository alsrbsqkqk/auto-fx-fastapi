import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math

# ====== 시뮬레이션용 주요 지표 계산 함수 ======

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

def candle_psychology_score(row, signal):
    score = 0
    reasons = []
    body = abs(row['close'] - row['open'])
    upper_wick = row['high'] - max(row['close'], row['open'])
    lower_wick = min(row['close'], row['open']) - row['low']
    total_range = row['high'] - row['low']
    body_ratio = body / total_range if total_range != 0 else 0

    if body_ratio >= 0.7:
        if row['close'] > row['open'] and signal == "BUY":
            score += 1
            reasons.append("장대양봉")
        elif row['close'] < row['open'] and signal == "SELL":
            score += 1
            reasons.append("장대음봉")
    if lower_wick > 2 * body and signal == "BUY":
        score += 1
        reasons.append("아래꼬리 길다")
    if upper_wick > 2 * body and signal == "SELL":
        score += 1
        reasons.append("위꼬리 길다")
    return score, reasons

# ====== 핵심 시뮬레이션 엔진 ======

def backtest(df):
    results = []
    
    df['rsi'] = calculate_rsi(df['close'])
    df['macd'], df['macd_signal'] = calculate_macd(df['close'])
    df['stoch_rsi'] = calculate_stoch_rsi(df['rsi'])
    df['boll_up'], df['boll_mid'], df['boll_low'] = calculate_bollinger_bands(df['close'])
    df['pattern'] = df.apply(detect_candle_pattern, axis=1)

    last_trade_time = None
    for i in range(50, len(df)):
        row = df.iloc[i]
        now_time = pd.to_datetime(row['time'])
        atlanta_hour = (now_time - timedelta(hours=4)).hour
        
        if atlanta_hour >= 22 or atlanta_hour <= 6:
            continue  # 거래 제한 시간 필터
        
        trend = detect_trend(df.iloc[i-50:i])
        signal = "BUY" if row['rsi'] < 35 else "SELL" if row['rsi'] > 70 else "WAIT"
        if signal == "WAIT":
            continue
        
        score = 0
        reasons = []
        
        if signal == "BUY" and row['pattern'] == "HAMMER":
            score += 2
            reasons.append("RSI+HAMMER")
        if signal == "SELL" and row['pattern'] == "SHOOTING_STAR":
            score += 2
            reasons.append("RSI+SHOOTING")
        
        if (row['macd'] - row['macd_signal']) > 0.001 and signal == "BUY":
            score += 1
            reasons.append("MACD GC")
        if (row['macd_signal'] - row['macd']) > 0.001 and signal == "SELL":
            score += 1
            reasons.append("MACD DC")
        
        if row['stoch_rsi'] > 0.8 and trend == "UPTREND" and signal == "BUY":
            score += 1
            reasons.append("Stoch RSI 과열")
        if row['stoch_rsi'] < 0.2 and trend == "DOWNTREND" and signal == "SELL":
            score += 1
            reasons.append("Stoch RSI 과매도")

        psy_score, psy_reasons = candle_psychology_score(row, signal)
        score += psy_score
        reasons += psy_reasons
        
        # 뉴스 필터 (시뮬레이션: 항상 안전)
        score += 0
        
        # 마지막 거래 후 최소 2시간 간격
        if last_trade_time and (now_time - last_trade_time) < timedelta(hours=2):
            continue
        
        # 진입 조건
        if score >= 4:
            entry_price = row['close']
            tp = entry_price + 0.0010 if signal == "BUY" else entry_price - 0.0010
            sl = entry_price - 0.0007 if signal == "BUY" else entry_price + 0.0007
            
            # 백테스트 결과 저장
            results.append({
                'time': row['time'], 'signal': signal, 'entry': entry_price, 
                'tp': tp, 'sl': sl, 'reasons': "; ".join(reasons)
            })
            last_trade_time = now_time
    
    return pd.DataFrame(results)

# ====== 샘플 데이터로 실행 ======

# 실제로는 OANDA에서 candle 가져와서 아래 df를 채우면 됨
# 임시 예시 (이후 OANDA 연동시 자동으로 df 준비됨)
data = pd.read_csv('oanda_sample_data.csv')
data['time'] = pd.to_datetime(data['time'])
result = backtest(data)

print(result)
result.to_csv("backtest_result.csv", index=False)
