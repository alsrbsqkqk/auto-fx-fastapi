import os
from fastapi import FastAPI, Request
import requests
import json
import pandas as pd
from datetime import datetime
import openai
from openai import OpenAI
import numpy as np
import csv

print("✅ Render에서 OANDA_API_KEY =", os.getenv("OANDA_API_KEY"))
print("✅ Loaded OANDA_API_KEY =", os.getenv("OANDA_API_KEY"))
print("✅ Loaded ACCOUNT_ID =", os.getenv("ACCOUNT_ID"))

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
        raw_data = await request.body()
        try:
            data = json.loads(raw_data) if isinstance(raw_data, bytes) else raw_data
            if isinstance(data, str):
                data = json.loads(data)
        except Exception as e:
            print("❌ Webhook 처리 실패:", str(e))
            return {"status": "error", "message": f"JSON 파싱 실패: {str(e)}"}

        pair = data.get("pair")
        price_raw = data.get("price")
        signal = data.get("signal")
        strategy = data.get("strategy")

        try:
            price = float(price_raw)
        except Exception as e:
            return {"status": "error", "message": f"price 변환 실패: {str(e)}"}

        now = datetime.utcnow()
        if now.hour < 4 or now.hour >= 20:
            return {"message": "현재는 유동성 낮은 시간대로, 전략 판단 신뢰도 저하. 관망 권장."}

        candles = get_candles(pair, "M30", 200)
        print("📊 캔들 데이터 길이:", len(candles))
        print(candles.head())
        if candles.empty:
            return {"status": "error", "message": f"{pair}에 대한 캔들 데이터 없음"}

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
            print("⚠️ 지표 간 충돌 조건으로 인해 기록 시도")
            log_trade_result(pair, signal, "WAIT", 0, "지표 해석 충돌")
            print("📌 log_trade_result 호출 완료: 기록 시도 완료됨")
            return {"message": "지표 간 해석 충돌로 인해 관망 처리됨"}
            

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
        # 기존 signal_score 판단 후 payload 구성
        payload = {
            "pair": pair,
            "signal": signal,
            "price": price,
            "rsi": round(latest_rsi, 2),
            "macd": round(latest_macd, 5),
            "macd_signal": round(latest_signal, 5),
            "stoch_rsi": round(latest_stoch_rsi, 2),
            "pattern": pattern,
            "trend": trend,
            "liquidity": liquidity,
            "volatility": volatility,
            "extreme_volatility": extreme_volatility,
            "hhll": hhll,
            "support_resistance": support_resistance,
            "fibonacci_levels": fibo_levels,
            "news": news_risk,
            "score": signal_score,
            "reasons": reasons
        }

        gpt_feedback = analyze_with_gpt(payload)

        # GPT 응답 파싱
        import re
        match = re.search(r"결정\s*[:：]?\s*(BUY|SELL|WAIT)", gpt_feedback, re.IGNORECASE)
        gpt_decision = match.group(1).upper() if match else "WAIT"

        tp_match = re.search(r"TP\s*[:：]?\s*([\d.]+)", gpt_feedback)
        sl_match = re.search(r"SL\s*[:：]?\s*([\d.]+)", gpt_feedback)
        tp = float(tp_match.group(1)) if tp_match else None
        sl = float(sl_match.group(1)) if sl_match else None

        # 기존 decision 무시하고 GPT 판단 적용
        decision = gpt_decision
        adjustment_reason = "GPT 전략 판단 반영"

        if decision in ["BUY", "SELL"] and tp and sl:
            units = 50000 if decision == "BUY" else -50000
            digits = precision_by_pair.get(pair, 5)
            result = place_order(pair, units, tp, sl, digits)
            print("📥 GPT 판단에 따른 기록 시작:", gpt_decision)
            log_trade_result(pair, signal, decision, signal_score, ",".join(reasons) + " | GPT결정")
        else:
            log_trade_result(pair, signal, "WAIT", signal_score, ",".join(reasons) + " | GPT WAIT")
            print("📌 log_trade_result 호출 완료: 기록 시도 완료됨")
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
            print("📌 log_trade_result 호출 완료: 기록 시도 완료됨")
            
        print("✅ 최종 return 직전: 모든 계산 완료, 결과 반환 시작")
        print("✅ 최종 결과 반환 준비 완료:")
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
            "hhll_HH": bool(hhll.get("HH", False)),
            "hhll_LL": bool(hhll.get("LL", False)),
            "support_resistance": support_resistance,
            "fibonacci_levels": fibo_levels,
            "score": signal_score,
            "decision": decision,
            "reasons": reasons,
            "adjustment_reason": adjustment_reason,
            "news": news_risk,
            "order_result": result
        }
    except Exception as e:
        return {"status": "error", "message": f"처리 실패: {str(e)}"}

def get_candles(pair="EUR_USD", granularity="M30", count=200):
    api_key = os.getenv("OANDA_API_KEY")
    url = f"https://api-fxpractice.oanda.com/v3/instruments/{pair}/candles"
    headers = {"Authorization": f"Bearer {OANDA_API_KEY}"}
    params = {"granularity": granularity, "count": count, "price": "M"}
    r = requests.get(url, headers=headers, params=params)

    try:
        data = r.json()
    except Exception as e:
        raise ValueError(f"캔들 JSON 파싱 실패: {e}")
        
    candles = data.get("candles", [])
    if not candles:
        raise ValueError(f"OANDA에서 받은 {pair}의 캔들 데이터가 비어 있습니다. 응답: {data}")
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
    if df.empty:
        raise ValueError(f"{pair}의 유효한 캔들 데이터가 없습니다 (모두 'complete=False')")
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

import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

def log_trade_result(pair, signal, decision, score, notes, result=None):
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name("/etc/secrets/google_credentials.json", scope)
        client = gspread.authorize(creds)
        sheet = client.open("민균 FX trading result").sheet1
        row = [str(datetime.utcnow()), pair, signal, decision, score, notes, result or "미정"]
        sheet.append_row(row)
        print("📄 구글 시트에 트레이드 기록 저장 완료:", row)
    except Exception as e:
        print("❌ 구글 시트 기록 실패:", str(e))

          

@app.get("/oanda-auth-test")
def oanda_auth_test():
    api_key = os.getenv("OANDA_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}"}
    url = "https://api-fxpractice.oanda.com/v3/accounts"

    r = requests.get(url, headers=headers)
    return {"status": r.status_code, "response": r.text}

def analyze_with_gpt(payload):
    import requests
    import os

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        "Content-Type": "application/json"
    }
    messages = [
        {
            "role": "system",
            "content": "너는 실전 FX 트레이딩 전략 조력자야. 아래 JSON 데이터를 기반으로 전략 리포트를 생성하고, 진입 판단(BUY, SELL, WAIT)과 TP, SL 값을 제시해줘."
        },
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False)
        }
    ]
    body = {
        "model": "gpt-4",
        "messages": messages,
        "temperature": 0.3
    }

    try:
        response = requests.post(url, headers=headers, json=body, timeout=10)
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"GPT 요청 실패: {str(e)}"
