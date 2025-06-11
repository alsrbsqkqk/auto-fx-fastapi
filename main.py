from fastapi import FastAPI, Request
import os
from dotenv import load_dotenv
load_dotenv()
import requests
import json
import pandas as pd
from datetime import datetime
import openai
from openai import OpenAI
import numpy as np
import csv

app = FastAPI()

OANDA_API_KEY = os.getenv("OANDA_API_KEY")
ACCOUNT_ID = os.getenv("ACCOUNT_ID")
openai.api_key = os.getenv("OPENAI_API_KEY")

precision_by_pair = {
    "EUR_USD": 5,
    "USD_JPY": 3
}

entry_offset_by_pair = {
    "USD_JPY": 0.03,
    "EUR_USD": 0.0003
}

def fetch_forex_news():
    try:
        response = requests.get("https://www.forexfactory.com/", timeout=5)
        if "High Impact Expected" in response.text:
            return "⚠️ 고위험 뉴스 존재"
        return "🟢 뉴스 영향 적음"
    except:
        return "뉴스 필터 오류 또는 연결 실패"

@app.get("/")
def home():
    return {"message": "🚀 FastAPI 서버 정상 작동 중"}

@app.post("/webhook")
async def webhook(request: Request):
    try:
        data = await request.json()
        pair = data["pair"]
        price = float(data["price"])
        signal = data["signal"]
        strategy = data.get("strategy", "N/A")
    except Exception as e:
        return {"status": "error", "message": str(e)}

    now = datetime.utcnow()
    if now.hour < 4 or now.hour >= 20:
        return {"message": "현재는 유동성 낮은 시간대로, 전략 판단 신뢰도 저하. 관망 권장."}

    candles = get_candles(pair, "M30", 200)
    close = candles["close"]
    rsi = calculate_rsi(close)
    macd, macd_signal = calculate_macd(close)
    stoch_rsi = calculate_stoch_rsi(rsi)
    support_resistance = detect_support_resistance(candles)
    fibo_levels = calculate_fibonacci_levels(candles["high"].max(), candles["low"].min())

    latest_rsi = rsi.iloc[-1]
    latest_macd = macd.iloc[-1]
    latest_signal = macd_signal.iloc[-1]
    latest_stoch_rsi = stoch_rsi.iloc[-1]

    pattern = detect_candle_pattern(candles, pair)
    trend = detect_trend(candles)
    volatility = is_volatile(candles)
    extreme_volatility = is_extremely_volatile(candles)
    hhll = detect_hh_ll(candles)
    liquidity = estimate_liquidity(candles)
    news_risk = fetch_forex_news()

    if (latest_macd > latest_signal and signal == "SELL") or (latest_macd < latest_signal and signal == "BUY"):
        log_trade_result(pair, signal, "WAIT", 0, "지표 해석 충돌")
        return {"message": "지표 간 해석 충돌 감지. 관망 필터 적용."}

    signal_score = 0
    reasons = []
    if latest_rsi < 30:
        signal_score += 1
        reasons.append("RSI < 30")
    if latest_macd > latest_signal:
        signal_score += 1
        reasons.append("MACD 골든크로스")
    if latest_stoch_rsi > 0.8:
        signal_score += 1
        reasons.append("Stoch RSI 과열")
    if trend == "UPTREND" and signal == "BUY":
        signal_score += 1
        reasons.append("추세 상승 + 매수 일치")
    if trend == "DOWNTREND" and signal == "SELL":
        signal_score += 1
        reasons.append("추세 하락 + 매도 일치")
    if liquidity == "좋음":
        signal_score += 1
        reasons.append("유동성 충분")
    if pattern in ["HAMMER", "BULLISH_ENGULFING"]:
        signal_score += 1
        reasons.append(f"캔들패턴: {pattern}")
    if hhll["HH"] or hhll["LL"]:
        signal_score += 1
        reasons.append("고점/저점 갱신 감지")
    if volatility and not extreme_volatility:
        signal_score += 1
        reasons.append("적절한 변동성")

    decision = "BUY" if signal_score >= 5 and signal == "BUY" else "SELL" if signal_score >= 5 and signal == "SELL" else "WAIT"
    adjustment_reason = ""
    result = {}

    if decision in ["BUY", "SELL"]:
        units = 50000 if decision == "BUY" else -50000
        digits = precision_by_pair.get(pair, 5)
        offset = entry_offset_by_pair.get(pair, 0.0003)
        tp = round(price + offset, digits) if decision == "BUY" else round(price - offset, digits)
        sl = round(price - offset, digits) if decision == "BUY" else round(price + offset, digits)

        if decision == "BUY" and (tp < support_resistance["resistance"] or tp < fibo_levels["0.382"]):
            tp = round(price + 1.5 * offset, digits)
            adjustment_reason = "TP 보정: S/R 또는 피보나치 저항 고려"
        if decision == "SELL" and (tp > support_resistance["support"] or tp > fibo_levels["0.618"]):
            tp = round(price - 1.5 * offset, digits)
            adjustment_reason = "TP 보정: S/R 또는 피보나치 지지 고려"

        result = place_order(pair, units, tp, sl, digits)
        log_trade_result(pair, signal, decision, signal_score, ",".join(reasons) + (" | " + adjustment_reason if adjustment_reason else ""))
    else:
        log_trade_result(pair, signal, "WAIT", signal_score, ",".join(reasons))

    return {
        "rsi": round(latest_rsi, 2),
        "stoch_rsi": round(latest_stoch_rsi, 2),
        "macd": round(latest_macd, 5),
        "macd_signal": round(latest_signal, 5),
        "pattern": pattern,
        "trend": trend,
        "liquidity": liquidity,
        "volatility": volatility,
        "extreme_volatility": extreme_volatility,
        "hhll": hhll,
        "support_resistance": support_resistance,
        "fibonacci_levels": fibo_levels,
        "score": signal_score,
        "decision": decision,
        "reasons": reasons,
        "adjustment_reason": adjustment_reason,
        "news": news_risk,
        "order_result": result
    }

def get_candles(pair="EUR_USD", granularity="M30", count=200):
    url = f"https://api-fxpractice.oanda.com/v3/instruments/{pair}/candles"
    headers = {"Authorization": f"Bearer {OANDA_API_KEY}"}
    params = {"granularity": granularity, "count": count, "price": "M"}
    r = requests.get(url, headers=headers, params=params)
    data = r.json()
    candles = data.get("candles", [])
    df = pd.DataFrame([
        {
            "time": c["time"],
            "open": float(c["mid"]["o"]),
            "high": float(c["mid"]["h"]),
            "low": float(c["mid"]["l"]),
            "close": float(c["mid"]["c"]),
            "volume": c.get("volume", 0)
        } for c in candles if c.get("complete", False)
    ])
    return df

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_stoch_rsi(rsi_series, period=14):
    min_rsi = rsi_series.rolling(window=period).min()
    max_rsi = rsi_series.rolling(window=period).max()
    stoch_rsi = (rsi_series - min_rsi) / (max_rsi - min_rsi)
    return stoch_rsi

def detect_support_resistance(candles, window=10):
    highs = candles["high"].tail(window)
    lows = candles["low"].tail(window)
    return {
        "support": round(lows.min(), 5),
        "resistance": round(highs.max(), 5)
    }

def calculate_fibonacci_levels(high, low):
    diff = high - low
    return {
        "0.0": round(high, 5),
        "0.236": round(high - diff * 0.236, 5),
        "0.382": round(high - diff * 0.382, 5),
        "0.5": round(high - diff * 0.5, 5),
        "0.618": round(high - diff * 0.618, 5),
        "1.0": round(low, 5)
    }

def detect_candle_pattern(candles, symbol="EUR_USD"):
    if len(candles) < 3:
        return "NOT_ENOUGH_DATA"
    last = candles.iloc[-1]
    prev = candles.iloc[-2]
    o, h, l, c = last["open"], last["high"], last["low"], last["close"]
    p_o, p_c = prev["open"], prev["close"]
    wick_tolerance = 0.35 if symbol == "USD_JPY" else 0.2
    if o > c and (o - c) > ((h - l) * 0.7):
        return "BEARISH_ENGULFING"
    elif c > o and (c - o) > ((h - l) * 0.7):
        return "BULLISH_ENGULFING"
    elif (h - l) > 2 * abs(o - c) and abs(o - c) < ((h - l) * wick_tolerance):
        return "DOJI"
    elif (c > o) and (l == min(l, p_o, p_c)) and (c == max(c, p_o, p_c)):
        return "HAMMER"
    elif (h - max(o, c)) > 2 * abs(o - c) and (min(o, c) - l) < abs(o - c):
        return "INVERTED_HAMMER"
    elif (h - max(o, c)) > 2 * abs(o - c) and (min(o, c) - l) < abs(o - c) and p_c < c:
        return "SHOOTING_STAR"
    elif p_c < p_o and abs(o - c) < (h - l) * 0.3 and c > o and c > p_o:
        return "MORNING_STAR"
    return "NEUTRAL"

def detect_trend(candles):
    if len(candles) < 3:
        return "NEUTRAL"
    highs = candles["high"].tail(3).values
    lows = candles["low"].tail(3).values
    if highs[2] > highs[1] > highs[0] and lows[2] > lows[1] > lows[0]:
        return "UPTREND"
    elif highs[2] < highs[1] < highs[0] and lows[2] < lows[1] < lows[0]:
        return "DOWNTREND"
    return "NEUTRAL"

def detect_hh_ll(candles):
    recent_highs = candles["high"].tail(20)
    recent_lows = candles["low"].tail(20)
    return {
        "HH": bool(recent_highs.is_monotonic_increasing),
        "LL": bool(recent_lows.is_monotonic_decreasing)
    }

def estimate_liquidity(candles):
    recent_volume = candles["volume"].tail(10).mean()
    return "좋음" if recent_volume > 100 else "나쁨"

def is_volatile(candles, threshold=0.002):
    last = candles.iloc[-1]
    return (last["high"] - last["low"]) / last["close"] > threshold

def is_extremely_volatile(candles, window=5, threshold=2.0):
    if len(candles) < window + 1:
        return False
    last = candles.iloc[-1]
    wick_size = abs(last["high"] - last["low"])
    avg_wick = candles.tail(window).apply(lambda x: abs(x["high"] - x["low"]), axis=1).mean()
    return wick_size > avg_wick * threshold

def place_order(symbol, units, tp, sl, digits):
    url = f"https://api-fxpractice.oanda.com/v3/accounts/{ACCOUNT_ID}/orders"
    headers = {
        "Authorization": f"Bearer {OANDA_API_KEY}",
        "Content-Type": "application/json"
    }
    order = {
        "order": {
            "units": units,
            "instrument": symbol,
            "type": "MARKET",
            "positionFill": "DEFAULT",
            "takeProfitOnFill": {"price": f"{tp:.{digits}f}"},
            "stopLossOnFill": {"price": f"{sl:.{digits}f}"}
        }
    }
    try:
        response = requests.post(url, headers=headers, data=json.dumps(order))
        print("📤 OANDA 주문 응답:", response.status_code, response.text)
        return {"status": response.status_code, "response": response.json()}
    except Exception as e:
        return {"status": "error", "message": str(e)}
import os

def log_trade_result(pair, signal, decision, score, notes):
    file_exists = os.path.exists("trade_results.csv")
    with open("trade_results.csv", "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "pair", "signal", "decision", "score", "notes"])
        writer.writerow([datetime.utcnow(), pair, signal, decision, score, notes])    
