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

def detect_trend(df):
    ema20 = df['close'].ewm(span=20).mean()
    ema50 = df['close'].ewm(span=50).mean()
    if ema20.iloc[-1] > ema50.iloc[-1]:
        return "UPTREND"
    elif ema20.iloc[-1] < ema50.iloc[-1]:
        return "DOWNTREND"
    else:
        return "NEUTRAL"

def detect_candle_pattern(row):
    body = abs(row['close'] - row['open'])
    upper_wick = row['high'] - max(row['close'], row['open'])
    lower_wick = min(row['close'], row['open']) - row['low']
    if lower_wick > 2 * body and upper_wick < body:
        return "HAMMER"
    elif upper_wick > 2 * body and lower_wick < body:
        return "SHOOTING_STAR"
    return "NEUTRAL"

# ====== 박스권 분석 (Main 엔진 전용) ======
def detect_box_breakout(df, box_window=10, pip_filter=0.0020):
    recent = df.tail(box_window)
    high = recent['high'].max()
    low = recent['low'].min()
    box_size = high - low
    breakout = None
    last_close = recent['close'].iloc[-1]
    if box_size < pip_filter:
        if last_close > high:
            breakout = "UP"
        elif last_close < low:
            breakout = "DOWN"
    return breakout

# ====== Main 시뮬레이션 엔진 (EURUSD, GBPUSD) ======
def backtest_main(df, pair):
    results = []
    df['rsi'] = calculate_rsi(df['close'])
    df['macd'], df['macd_signal'] = calculate_macd(df['close'])
    df['stoch_rsi'] = calculate_stoch_rsi(df['rsi'])
    df['boll_up'], df['boll_mid'], df['boll_low'] = calculate_bollinger_bands(df['close'])
    df['pattern'] = df.apply(detect_candle_pattern, axis=1)
    
    last_trade_time = None
    for i in range(50, len(df)):
        row = df.iloc[i]
        now = pd.to_datetime(row['time'])
        atlanta_hour = (now - timedelta(hours=4)).hour
        
        # 유동성 시간대 필터
        if pair in ['EURUSD','GBPUSD'] and (atlanta_hour >= 22 or atlanta_hour <= 6):
            continue
        
        trend = detect_trend(df.iloc[i-50:i])
        signal = "BUY" if row['rsi'] < 30 else "SELL" if row['rsi'] > 70 else "WAIT"
        if signal == "WAIT": continue
        
        score = 0
        reasons = []
        
        # RSI + 캔들 패턴
        if signal == "BUY" and row['pattern'] in ["HAMMER", "BULLISH_ENGULFING"]:
            score += 2
            reasons.append("RSI<30 + Hammer")
        if signal == "SELL" and row['pattern'] in ["SHOOTING_STAR", "BEARISH_ENGULFING"]:
            score += 2
            reasons.append("RSI>70 + ShootingStar")
        
        # MACD 확인
        if (row['macd'] - row['macd_signal']) > 0.0001 and signal == "BUY":
            score += 1
            reasons.append("MACD 골든크로스")
        if (row['macd_signal'] - row['macd']) > 0.0001 and signal == "SELL":
            score += 1
            reasons.append("MACD 데드크로스")
        
        # Stoch RSI + Trend
        if row['stoch_rsi'] > 0.8 and trend == "UPTREND" and signal == "BUY":
            score += 1
            reasons.append("Stoch과열 + 상승추세")
        if row['stoch_rsi'] < 0.2 and trend == "DOWNTREND" and signal == "SELL":
            score += 1
            reasons.append("Stoch과매도 + 하락추세")

        # 박스권 돌파 보정
        breakout = detect_box_breakout(df.iloc[i-10:i])
        if breakout == "UP" and signal == "BUY":
            score += 2
            reasons.append("박스권 상단 돌파")
        if breakout == "DOWN" and signal == "SELL":
            score += 2
            reasons.append("박스권 하단 돌파")

        # 최소 2시간 간격 유지
        if last_trade_time and (now - last_trade_time) < timedelta(hours=2):
            continue

        if score >= 4:
            entry = row['close']
            tp = entry + 0.0010 if signal == "BUY" else entry - 0.0010
            sl = entry - 0.0007 if signal == "BUY" else entry + 0.0007
            results.append({
                'time': row['time'], 'pair': pair, 'signal': signal, 
                'entry': entry, 'tp': tp, 'sl': sl, 'score': score, 'reason': "; ".join(reasons)
            })
            last_trade_time = now
    return pd.DataFrame(results)

# ====== FastFury 엔진 (USDJPY) ======
def backtest_fastfury(df):
    results = []
    df['rsi'] = calculate_rsi(df['close'])
    df['macd'], df['macd_signal'] = calculate_macd(df['close'])
    df['stoch_rsi'] = calculate_stoch_rsi(df['rsi'])
    
    last_trade_time = None
    for i in range(30, len(df)):
        row = df.iloc[i]
        now = pd.to_datetime(row['time'])
        atlanta_hour = (now - timedelta(hours=4)).hour
        
        if atlanta_hour >= 22 or atlanta_hour <= 6:
            continue

        score = 0
        reasons = []

        if abs(row['macd'] - row['macd_signal']) > 0.00008:
            if row['macd'] > row['macd_signal']:
                signal = "BUY"
            else:
                signal = "SELL"
            score += 1
            reasons.append("MACD 강도 통과")
        else:
            continue
        
        ema9 = df['close'].ewm(span=9).mean().iloc[i]
        ema21 = df['close'].ewm(span=21).mean().iloc[i]
        if signal == "BUY" and ema9 > ema21:
            score += 1
            reasons.append("EMA 상승추세")
        if signal == "SELL" and ema9 < ema21:
            score += 1
            reasons.append("EMA 하락추세")
        
        candle_body = row['close'] - row['open']
        if signal == "BUY" and candle_body > 0:
            score += 1
            reasons.append("캔들 양봉")
        if signal == "SELL" and candle_body < 0:
            score += 1
            reasons.append("캔들 음봉")

        if last_trade_time and (now - last_trade_time) < timedelta(hours=1):
            continue

    
        if score >= 3:
            entry = row['close']
            tp = entry + 0.10 if signal == "BUY" else entry - 0.10
            sl = entry - 0.07 if signal == "BUY" else entry + 0.07
            results.append({
                'time': row['time'], 'pair': "USDJPY", 'signal': signal, 
                'entry': entry, 'tp': tp, 'sl': sl, 'score': score, 'reason': "; ".join(reasons)
            })
            last_trade_time = now

    return pd.DataFrame(results)

# ====== 통합 실행 ======
if __name__ == "__main__":
    final = []

    # EURUSD
    df_eur = pd.read_csv("EURUSD.csv")
    df_eur['time'] = pd.to_datetime(df_eur['time'])
    res_eur = backtest_main(df_eur, "EURUSD")
    final.append(res_eur)

    # GBPUSD
    df_gbp = pd.read_csv("GBPUSD.csv")
    df_gbp['time'] = pd.to_datetime(df_gbp['time'])
    res_gbp = backtest_main(df_gbp, "GBPUSD")
    final.append(res_gbp)

    # USDJPY (Fast Fury)
    df_jpy = pd.read_csv("USDJPY.csv")
    df_jpy['time'] = pd.to_datetime(df_jpy['time'])
    res_jpy = backtest_fastfury(df_jpy)
    final.append(res_jpy)

    full_result = pd.concat(final)
    full_result.to_csv("full_backtest_results.csv", index=False)
    print("✅ 통합 백테스트 완료!")
