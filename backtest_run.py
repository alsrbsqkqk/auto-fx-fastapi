import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math

# 거래 비용 설정
SPREAD_EURUSD = 0.0001
SPREAD_GBPUSD = 0.00012
SPREAD_USDJPY = 0.012

# 주요 지표 계산 함수
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

# 캔들 패턴 및 심리 필터
def detect_candle_pattern(row):
    body = abs(row['close'] - row['open'])
    upper_wick = row['high'] - max(row['close'], row['open'])
    lower_wick = min(row['close'], row['open']) - row['low']
    if lower_wick > 2 * body and upper_wick < body:
        return "HAMMER"
    elif upper_wick > 2 * body and lower_wick < body:
        return "SHOOTING_STAR"
    return "NEUTRAL"

def candle_psychology_score(row, signal):
    score = 0
    body = abs(row['close'] - row['open'])
    total_range = row['high'] - row['low']
    if total_range == 0:
        return 0
    body_ratio = body / total_range
    if body_ratio > 0.7:
        if signal == "BUY" and row['close'] > row['open']:
            score += 1
        if signal == "SELL" and row['close'] < row['open']:
            score += 1
    return score

# Fast Fury (USDJPY 전용)
def fast_fury_signal(row):
    body = abs(row['close'] - row['open'])
    upper_wick = row['high'] - max(row['close'], row['open'])
    lower_wick = min(row['close'], row['open']) - row['low']
    if body >= (row['high'] - row['low']) * 0.65:
        if row['close'] > row['open']:
            return "BUY"
        elif row['close'] < row['open']:
            return "SELL"
    return "WAIT"

# 실전 알림 시뮬레이션 (알림 발생 → 지표 재평가)
def process_alert(df, pair_name):
    results = []
    spread = SPREAD_EURUSD if pair_name=="EURUSD" else SPREAD_GBPUSD if pair_name=="GBPUSD" else SPREAD_USDJPY

    for i in range(50, len(df)):
        row = df.iloc[i]
        time = row['time']
        macd, macd_signal = row['macd'], row['macd_signal']
        rsi, stoch_rsi = row['rsi'], row['stoch_rsi']
        ema9, ema21 = row['ema9'], row['ema21']
        volume = row['volume']
        candle_pattern = row['pattern']
        trendConfirmLong = ema9 > ema21
        trendConfirmShort = ema9 < ema21

        # === MACD CROSS SNIPER ===
        if macd > macd_signal and trendConfirmLong and abs(macd - macd_signal) > 0.00008:
            score = 1
            if stoch_rsi > 0.8:
                score += 1
            if candle_pattern == "HAMMER":
                score += 1
            score += candle_psychology_score(row, "BUY")
            if score >= 4:
                entry = row['close'] + spread
                tp = entry + 0.0010
                sl = entry - 0.0007
                results.append({"time": time, "pair": pair_name, "signal": "BUY", "entry": entry, "tp": tp, "sl": sl})

        if macd < macd_signal and trendConfirmShort and abs(macd - macd_signal) > 0.00008:
            score = 1
            if stoch_rsi < 0.2:
                score += 1
            if candle_pattern == "SHOOTING_STAR":
                score += 1
            score += candle_psychology_score(row, "SELL")
            if score >= 4:
                entry = row['close'] - spread
                tp = entry - 0.0010
                sl = entry + 0.0007
                results.append({"time": time, "pair": pair_name, "signal": "SELL", "entry": entry, "tp": tp, "sl": sl})

        # === STOCH BOUNCE FAST ===
        if rsi < 50 and stoch_rsi < 0.35 and trendConfirmLong:
            entry = row['close'] + spread
            tp = entry + 0.0010
            sl = entry - 0.0007
            results.append({"time": time, "pair": pair_name, "signal": "BUY", "entry": entry, "tp": tp, "sl": sl})

        if rsi > 50 and stoch_rsi > 0.65 and trendConfirmShort:
            entry = row['close'] - spread
            tp = entry - 0.0010
            sl = entry + 0.0007
            results.append({"time": time, "pair": pair_name, "signal": "SELL", "entry": entry, "tp": tp, "sl": sl})

        # === VOLUME BOOM REVERSAL ===
        avg_vol = df['volume'].rolling(window=20).mean().iloc[i]
        if volume > avg_vol * 1.5:
            if row['close'] > row['open'] and trendConfirmLong:
                entry = row['close'] + spread
                tp = entry + 0.0010
                sl = entry - 0.0007
                results.append({"time": time, "pair": pair_name, "signal": "BUY", "entry": entry, "tp": tp, "sl": sl})
            if row['close'] < row['open'] and trendConfirmShort:
                entry = row['close'] - spread
                tp = entry - 0.0010
                sl = entry + 0.0007
                results.append({"time": time, "pair": pair_name, "signal": "SELL", "entry": entry, "tp": tp, "sl": sl})

        # === TREND CONTINUATION 추가 ===
        if trendConfirmLong and row['close'] > row['open']:
            entry = row['close'] + spread
            tp = entry + 0.0010
            sl = entry - 0.0007
            results.append({"time": time, "pair": pair_name, "signal": "BUY", "entry": entry, "tp": tp, "sl": sl})
        if trendConfirmShort and row['close'] < row['open']:
            entry = row['close'] - spread
            tp = entry - 0.0010
            sl = entry + 0.0007
            results.append({"time": time, "pair": pair_name, "signal": "SELL", "entry": entry, "tp": tp, "sl": sl})

        # === FAST FURY (USDJPY 전용) ===
        if pair_name == "USDJPY":
            fury = fast_fury_signal(row)
            if fury in ["BUY", "SELL"]:
                entry = row['close'] + spread if fury == "BUY" else row['close'] - spread
                tp = entry + 0.0015 if fury == "BUY" else entry - 0.0015
                sl = entry - 0.0010 if fury == "BUY" else entry + 0.0010
                results.append({"time": time, "pair": pair_name, "signal": fury, "entry": entry, "tp": tp, "sl": sl})

    return pd.DataFrame(results)

# 메인 실행 파트
def run_full_backtest():
    all_results = []
    for pair_file, pair_name in [
        ("EURUSD.csv", "EURUSD"),
        ("GBPUSD.csv", "GBPUSD"),
        ("USDJPY.csv", "USDJPY")
    ]:
        df = pd.read_csv(pair_file)
        df['time'] = pd.to_datetime(df['time'])
        df['rsi'] = calculate_rsi(df['close'])
        df['macd'], df['macd_signal'] = calculate_macd(df['close'])
        df['stoch_rsi'] = calculate_stoch_rsi(df['rsi'])
        df['boll_up'], df['boll_mid'], df['boll_low'] = calculate_bollinger_bands(df['close'])
        df['ema9'] = df['close'].ewm(span=9).mean()
        df['ema21'] = df['close'].ewm(span=21).mean()
        df['pattern'] = df.apply(detect_candle_pattern, axis=1)

        pair_results = process_alert(df, pair_name)
        all_results.append(pair_results)

    final = pd.concat(all_results)
    final.to_csv("full_backtest_results.csv", index=False)
    print("✅ 전체 백테스트 완료. 결과 저장: full_backtest_results.csv")

if __name__ == "__main__":
    run_full_backtest()
