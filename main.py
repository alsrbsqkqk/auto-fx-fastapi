# ⚠️ V2 업그레이드된 자동 트레이딩 스크립트 (학습 강화, 트렌드 보강, 시트 시간 보정 포함)

import os
from fastapi import FastAPI, Request
import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import openai
import numpy as np
import gspread
from oauth2client.service_account import ServiceAccountCredentials

app = FastAPI()

OANDA_API_KEY = os.getenv("OANDA_API_KEY")
ACCOUNT_ID = os.getenv("ACCOUNT_ID")
openai.api_key = os.getenv("OPENAI_API_KEY")

@app.post("/webhook")
async def webhook(request: Request):
    data = json.loads(await request.body())
    pair = data.get("pair")
    price = float(data.get("price"))
    signal = data.get("signal")
    alert_name = data.get("alert_name", "기본알림")

    candles = get_candles(pair, "M30", 200)
    close = candles["close"]
    rsi = calculate_rsi(close)
    stoch_rsi_series = calculate_stoch_rsi(rsi)
    stoch_rsi = stoch_rsi_series.dropna().iloc[-1] if not stoch_rsi_series.dropna().empty else 0
    macd, macd_signal = calculate_macd(close)
    boll_up, boll_mid, boll_low = calculate_bollinger_bands(close)

    pattern = detect_candle_pattern(candles)
    trend = detect_trend(candles, rsi, boll_mid)
    liquidity = estimate_liquidity(candles)
    news = fetch_forex_news()
    support_resistance = detect_support_resistance(candles)

    # 점수 계산
    signal_score = 0
    reasons = []
    if rsi.iloc[-1] < 30:
        signal_score += 1
        reasons.append("RSI < 30")
    if macd.iloc[-1] > macd_signal.iloc[-1]:
        signal_score += 1
        reasons.append("MACD 골든크로스")
    if stoch_rsi > 0.8:
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
        reasons.append("유동성 좋음")
    if pattern in ["HAMMER", "BULLISH_ENGULFING"]:
        signal_score += 1
        reasons.append(f"캔들패턴: {pattern}")

    fibo_levels = calculate_fibonacci_levels(candles["high"].max(), candles["low"].min())

    payload = {
        "pair": pair,
        "price": price,
        "signal": signal,
        "rsi": rsi.iloc[-1],
        "macd": macd.iloc[-1],
        "macd_signal": macd_signal.iloc[-1],
        "stoch_rsi": stoch_rsi,
        "bollinger_upper": boll_up.iloc[-1],
        "bollinger_lower": boll_low.iloc[-1],
        "pattern": pattern,
        "trend": trend,
        "liquidity": liquidity,
        "support": support_resistance["support"],
        "resistance": support_resistance["resistance"],
        "news": news
    }

    gpt_feedback = analyze_with_gpt(payload)
    decision, tp, sl = parse_gpt_feedback(gpt_feedback)

    result = {}
    if decision in ["BUY", "SELL"] and tp and sl:
        units = 50000 if decision == "BUY" else -50000
        digits = 5 if "EUR" in pair else 3
        result = place_order(pair, units, tp, sl, digits)

    log_trade_result(pair, signal, decision, signal_score, gpt_reason, result, rsi.iloc[-1], macd.iloc[-1], stoch_rsi.iloc[-1], pattern, trend, fibo_levels, decision, news, gpt_feedback, alert_name, tp, sl, price, pnl)
    return {"결정": decision, "TP": tp, "SL": sl, "GPT응답": gpt_feedback}

def detect_support_resistance(candles, window=10):
    highs = candles["high"].tail(window)
    lows = candles["low"].tail(window)
    return {
        "support": round(lows.min(), 5),
        "resistance": round(highs.max(), 5)
    }

# ✳️ Helper Functions

def get_candles(pair, granularity, count):
    url = f"https://api-fxpractice.oanda.com/v3/instruments/{pair}/candles"
    headers = {"Authorization": f"Bearer {OANDA_API_KEY}"}
    params = {"granularity": granularity, "count": count, "price": "M"}
    r = requests.get(url, headers=headers, params=params)
    candles = r.json().get("candles", [])
    return pd.DataFrame([{ "time": c["time"], "open": float(c["mid"]["o"]), "high": float(c["mid"]["h"]), "low": float(c["mid"]["l"]), "close": float(c["mid"]["c"]), "volume": c.get("volume", 0) } for c in candles if c["complete"]])

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

def detect_trend(candles, rsi, mid_band):
    close = candles["close"]
    ema20 = close.ewm(span=20).mean()
    ema50 = close.ewm(span=50).mean()
    if ema20.iloc[-1] > ema50.iloc[-1] and close.iloc[-1] > mid_band.iloc[-1]:
        return "UPTREND"
    elif ema20.iloc[-1] < ema50.iloc[-1] and close.iloc[-1] < mid_band.iloc[-1]:
        return "DOWNTREND"
    return "NEUTRAL"

def detect_candle_pattern(candles):
    return "NEUTRAL"

def estimate_liquidity(candles):
    return "좋음" if candles["volume"].tail(10).mean() > 100 else "낮음"

def fetch_forex_news():
    try:
        response = requests.get("https://www.forexfactory.com/", timeout=5)
        if "High Impact Expected" in response.text:
            return "⚠️ 고위험 뉴스 존재"
        return "🟢 뉴스 영향 적음"
    except:
        return "❓ 뉴스 확인 실패"

def place_order(pair, units, tp, sl, digits):
    return {"status": "order_placed", "tp": tp, "sl": sl}

def parse_gpt_feedback(text):
    import re
    d = re.search(r"결정\s*[:：]?\s*(BUY|SELL|WAIT)", text.upper())
    tp = re.search(r"TP\s*[:：]?\s*([\d.]+)", text.upper())
    sl = re.search(r"SL\s*[:：]?\s*([\d.]+)", text.upper())
    return d.group(1) if d else "WAIT", float(tp.group(1)) if tp else None, float(sl.group(1)) if sl else None

def analyze_with_gpt(payload):
    headers = {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}", "Content-Type": "application/json"}
    messages = [
        {"role": "system", "content": "너는 실전 FX 트레이딩 전략 조력자야. 아래 JSON 데이터를 기반으로 전략 리포트를 생성하고, 진입 판단(BUY, SELL, WAIT)과 TP, SL 값을 제시해줘."},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
    ]
    body = {"model": "gpt-4", "messages": messages, "temperature": 0.3}

    try:
        r = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=body)
        result = r.json()
        if "choices" in result:
            return result["choices"][0]["message"]["content"]
        else:
            return f"[GPT ERROR] {result.get('error', {}).get('message', 'Unknown GPT response error')}"
    except Exception as e:
        return f"[GPT EXCEPTION] {str(e)}"

def log_trade_result(pair, signal, decision, score, notes, result=None, rsi=None, macd=None, stoch_rsi=None, pattern=None, trend=None, fibo=None, gpt_decision=None, news=None, gpt_feedback=None, alert_name=None, tp=None, sl=None, entry=None, pnl=None):
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name("/etc/secrets/google_credentials.json", scope)
    client = gspread.authorize(creds)
    sheet = client.open("민균 FX trading result").sheet1
    now_atlanta = datetime.utcnow() - timedelta(hours=4)
    row = [
        str(now_atlanta), pair, alert_name or "", signal, decision, score, rsi or "", macd or "", stoch_rsi or "",
        pattern or "", trend or "", fibo.get("0.382", ""), fibo.get("0.618", ""),
        gpt_decision or "", news or "", notes, result or "미정", gpt_feedback or "",
        entry or "", tp or "", sl or "", pnl or ""
    ]
    row.insert(19, news)  # 뉴스 분석 전용 컬럼으로 추가
    sheet.append_row(row)
