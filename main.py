from fastapi import FastAPI, Request
import os
import requests
import json
import pandas as pd
from datetime import datetime

app = FastAPI()

OANDA_API_KEY = os.getenv("OANDA_API_KEY")
ACCOUNT_ID = os.getenv("ACCOUNT_ID")

strategy_settings = {
    "RSI-FLUSH-LITE": {"tp": 0.0016, "sl": 0.0010},
    "MACD-WAVE-CATCH-LITE": {"tp": 0.0020, "sl": 0.0012},
    "RSI-EXTREME-PULLBACK-LITE": {"tp": 0.0017, "sl": 0.0010},
    "ST-RSI-SNAP-LITE": {"tp": 0.0015, "sl": 0.0010},
    "EMA-POWER-ZONE-LITE": {"tp": 0.0019, "sl": 0.0012},
    "LONDON-FADE-IN-LITE": {"tp": 0.0022, "sl": 0.0013},
    "NY-BREAK-RUN-LITE": {"tp": 0.0024, "sl": 0.0014},
    "RANGE-SCALP-MID-LITE": {"tp": 0.0014, "sl": 0.0010},
    "VOLUME-EXPLOSION-LITE": {"tp": 0.0021, "sl": 0.0013},
    "CANDLE-ENGULF-TRAP-LITE": {"tp": 0.0018, "sl": 0.0011},
    "BREAKOUT-MOMENTUM-LITE": {"tp": 0.0023, "sl": 0.0014},
    "NEWS-SPIKE-TRAP-LITE": {"tp": 0.0015, "sl": 0.0009}
}

@app.get("/")
def home():
    return {"message": "🚀 FastAPI 서버 정상 작동 중"}

@app.post("/webhook")
async def webhook(request: Request):
    data = await request.json()
    symbol = data.get("pair", "EUR_USD")
    price = float(data.get("price", 0.0)) if data.get("price") else None
    if price is None or price <= 0:
        return {"status": "error", "reason": "유효하지 않은 가격 정보"}

    signal = data.get("signal")
    strategy_id = data.get("strategy", "UNKNOWN")
    settings = strategy_settings.get(strategy_id)
    if not settings:
        return {"status": "ignored", "reason": f"정의되지 않은 전략 ID: {strategy_id}"}

    tp_gap = settings["tp"]
    sl_gap = settings["sl"]
    candles = get_candles(symbol, "M30", 50)
    pattern = detect_candle_pattern(candles)
    volatility = is_volatile(candles)
    extreme_volatility = is_extremely_volatile(candles)
    trend = detect_trend(candles)
    sr_levels = detect_support_resistance(candles)

    score = 0
    if pattern in ["HAMMER", "BULLISH_ENGULFING", "MORNING_STAR"]: score += 1
    if trend == "UPTREND" and signal == "BUY": score += 1
    if trend == "DOWNTREND" and signal == "SELL": score += 1
    if not extreme_volatility and volatility: score += 1

    if score < 2:
        log_order(symbol, strategy_id, signal, score, pattern, trend, volatility, extreme_volatility, "IGNORED", 0, "점수 부족")
        return {"status": "ignored", "reason": f"점수 부족: {score}/3"}

    if signal == "BUY":
        entry = round(price - 0.0003, 5)
        tp = round(price + tp_gap, 5)
        sl = round(price - sl_gap, 5)
    elif signal == "SELL":
        entry = round(price + 0.0003, 5)
        tp = round(price - tp_gap, 5)
        sl = round(price + sl_gap, 5)
    else:
        return {"status": "ignored", "reason": "알 수 없는 시그널"}

    order = {
        "order": {
            "units": "1000",
            "instrument": symbol,
            "type": "LIMIT",
            "positionFill": "DEFAULT",
            "price": str(entry),
            "takeProfitOnFill": {"price": str(tp)},
            "stopLossOnFill": {"price": str(sl)}
        }
    }

    headers = {
        "Authorization": f"Bearer {OANDA_API_KEY}",
        "Content-Type": "application/json"
    }
    url = f"https://api-fxpractice.oanda.com/v3/accounts/{ACCOUNT_ID}/orders"
    r = requests.post(url, headers=headers, data=json.dumps(order))

    log_order(symbol, strategy_id, signal, score, pattern, trend, volatility, extreme_volatility, "SENT", r.status_code, r.text)
    return {"status": "order sent", "response": r.json()}


def log_order(symbol, strategy, signal, score, pattern, trend, is_vol, is_extreme, status, code, response):
    with open("order_log.csv", "a", encoding="utf-8") as f:
        f.write(f"{datetime.now()},{symbol},{strategy},{signal},{score},{pattern},{trend},{is_vol},{is_extreme},{status},{code},{response.replace(',', ' ')}\n")


def get_candles(pair="EUR_USD", granularity="M30", count=50):
    url = f"https://api-fxpractice.oanda.com/v3/instruments/{pair}/candles"
    headers = {"Authorization": f"Bearer {OANDA_API_KEY}"}
    params = {"granularity": granularity, "count": count, "price": "M"}
    r = requests.get(url, headers=headers, params=params)
    data = r.json()
    print("📡 OANDA 응답 확인:", json.dumps(data, indent=2))
    df = pd.DataFrame([
        {
            "time": c["time"],
            "open": float(c["mid"]["o"]),
            "high": float(c["mid"]["h"]),
            "low": float(c["mid"]["l"]),
            "close": float(c["mid"]["c"]),
            "volume": c.get("volume", 0)
        } for c in data if c.get("complete", False)
    ])
    return df


def detect_candle_pattern(candles):
    if len(candles) < 3:
        return "NOT_ENOUGH_DATA"
    last = candles.iloc[-1]
    prev = candles.iloc[-2]
    o, h, l, c = last["open"], last["high"], last["low"], last["close"]
    p_o, p_c = prev["open"], prev["close"]

    if o > c and (o - c) > ((h - l) * 0.7):
        return "BEARISH_ENGULFING"
    elif c > o and (c - o) > ((h - l) * 0.7):
        return "BULLISH_ENGULFING"
    elif (h - l) > 2 * abs(o - c) and abs(o - c) < ((h - l) * 0.3):
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


def detect_support_resistance(candles, sensitivity=3):
    levels = []
    highs = list(candles["high"].round(4))
    lows = list(candles["low"].round(4))
    combined = highs + lows
    for price in combined:
        if sum([1 for p in combined if p == price]) >= sensitivity:
            if price not in levels:
                levels.append(price)
    levels = sorted(set(levels))
    return levels[-2:] if len(levels) >= 2 else levels


def detect_trend(candles):
    if len(candles) < 3:
        return "NEUTRAL"
    highs = candles["high"].tail(3).values
    lows = candles["low"].tail(3).values
    if highs[2] > highs[1] > highs[0] and lows[2] > lows[1] > lows[0]:
        return "UPTREND"
    elif highs[2] < highs[1] < highs[0] and lows[2] < lows[1] < lows[0]:
        return "DOWNTREND"
    else:
        return "NEUTRAL"


@app.get("/test-candle")
def test_candle(symbol: str = "EUR_USD"):
    try:
        df = get_candles(symbol, "M30", 10)
        return df.tail(5).to_dict(orient="records")
    except Exception as e:
        return {"error": str(e)}


@app.get("/test-candle2")
def test_candle2(symbol: str = "EUR_USD"):
    candles = get_candles(symbol, "M30", 50)
    pattern = detect_candle_pattern(candles)
    return {
        "symbol": symbol,
        "granularity": "M30",
        "candles": candles.tail(5).to_dict(orient="records"),
        "detected_pattern": pattern
    }


@app.get("/test-candle3")
def test_candle3(symbol: str = "EUR_USD"):
    try:
        candles = get_candles(symbol, "M30", 50)
        pattern = detect_candle_pattern(candles)
        volatility = is_volatile(candles)
        extreme_volatility = is_extremely_volatile(candles)
        trend = detect_trend(candles)
        support_resistance = detect_support_resistance(candles)

        return {
            "symbol": symbol,
            "granularity": "M30",
            "detected_pattern": pattern,
            "is_volatile": bool(volatility),
            "is_extremely_volatile": bool(extreme_volatility),
            "trend": trend,
            "support_resistance": support_resistance,
            "candles": candles.tail(5).to_dict(orient="records")
        }
    except Exception as e:
        return {"error": str(e)}
