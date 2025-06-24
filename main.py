# ⚠️ V2 업그레이드된 자동 트레이딩 스크립트 (학습 강화, 트렌드 보강, 시트 시간 보정 포함)
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import os
import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import openai
import numpy as np
import gspread
from oauth2client.service_account import ServiceAccountCredentials


def conflict_check(rsi, pattern, trend, signal):
    """
    추세-패턴-시그널 충돌 방지 필터 (V2 최종)
    """

    # 1️⃣ 기본 추세-패턴 충돌 방지
    if rsi > 80 and pattern in ["SHOOTING_STAR", "BEARISH_ENGULFING"] and trend == "UPTREND":
        return True
    if rsi < 20 and pattern in ["HAMMER", "BULLISH_ENGULFING"] and trend == "DOWNTREND":
        return True

    # 2️⃣ 캔들패턴이 없는데 시그널과 추세가 역방향이면 관망
    if pattern == "NEUTRAL":
        if trend == "UPTREND" and signal == "SELL" and rsi > 80:
            return True
        if trend == "DOWNTREND" and signal == "BUY" and rsi < 20:
            return True

    # 3️⃣ 기타 보수적 예외 추가
    if trend == "UPTREND" and signal == "SELL" and rsi > 80:
        return True
    if trend == "DOWNTREND" and signal == "BUY" and rsi < 20:
        return True

    return False
    
def check_recent_opposite_signal(pair, current_signal, within_minutes=30):
    """
    최근 동일 페어에서 반대 시그널이 있으면 True 반환
    """
    log_path = f"/tmp/{pair}_last_signal.txt"
    now = datetime.utcnow()

    # 기존 기록 읽기
    if os.path.exists(log_path):
        try:
            with open(log_path, "r") as f:
                last_record = f.read().strip().split(",")
                last_time = datetime.fromisoformat(last_record[0])
                last_signal = last_record[1]
            if (now - last_time).total_seconds() < within_minutes * 60:
                if last_signal != current_signal:
                    return True
        except Exception as e:
            print("❗ 최근 시그널 기록 불러오기 실패:", e)

    # 현재 시그널 기록 갱신
    try:
        with open(log_path, "w") as f:
            f.write(f"{now.isoformat()},{current_signal}")
    except Exception as e:
        print("❗ 시그널 기록 저장 실패:", e)

    return False



def score_signal_with_filters(rsi, macd, macd_signal, stoch_rsi, trend, signal, liquidity, pattern, pair, candles):
    signal_score = 0
    reasons = []

    # ✅ 거래 제한 시간 필터 (애틀랜타 기준)
    now_utc = datetime.utcnow()
    now_atlanta = now_utc - timedelta(hours=4)

    if now_atlanta.hour >= 22 or now_atlanta.hour <= 4:
        if pair in ["EUR_USD", "GBP_USD"]:
            reasons.append("❌ 심야 유동성 부족 → EURUSD, GBPUSD 거래 제한")
            return 0, reasons
    

    if conflict_check(rsi, pattern, trend, signal):
        reasons.append("⚠️ 추세와 패턴이 충돌 → 관망 권장")
        return 0, reasons   

    if rsi < 30:
        if pattern in ["HAMMER", "BULLISH_ENGULFING"]:
            signal_score += 2
            reasons.append("RSI < 30 + 캔들 패턴 확인")
        else:
            reasons.append("RSI < 30 but 캔들 패턴 없음 → 관망")

    if rsi > 70:
        if pattern in ["SHOOTING_STAR", "BEARISH_ENGULFING"]:
            signal_score += 2
            reasons.append("RSI > 70 + 캔들 패턴 확인")
        else:
            reasons.append("RSI > 70 but 캔들 패턴 없음 → 관망")
    if 40 <= rsi <= 60:
        reasons.append("RSI 중립구간 (보수 관망 추천)")

    if pattern in ["LONG_BODY_BULL", "LONG_BODY_BEAR"]:
        signal_score += 2
        reasons.append(f"장대바디 캔들 추가 가점: {pattern}")

    box_info = detect_box_breakout(candles, pair)

    if box_info["in_box"] and box_info["breakout"] == "UP" and signal == "BUY":
        signal_score += 3
        reasons.append("📦 박스권 상단 돌파 + 매수 신호 일치 (breakout 가점 강화)")
    elif box_info["in_box"] and box_info["breakout"] == "DOWN" and signal == "SELL":
        signal_score += 3
        reasons.append("📦 박스권 하단 돌파 + 매도 신호 일치")
    elif box_info["in_box"] and box_info["breakout"] is None:
        reasons.append("📦 박스권 유지 중 → 관망 경계")
    

    if (macd - macd_signal) > 0.0005 and trend == "UPTREND":
        signal_score += 3
        reasons.append("MACD 골든크로스 + 상승추세 일치 → breakout 강세")
    elif (macd_signal - macd) > 0.0005 and trend == "DOWNTREND":
        signal_score += 3
        reasons.append("MACD 데드크로스 + 하락추세 일치 → 하락 강화")
    elif abs(macd - macd_signal) > 0.0005:
        signal_score += 1
        reasons.append("MACD 교차 발생 (추세불명확)")
    else:
        reasons.append("MACD 미세변동 → 가점 보류")

    if stoch_rsi > 0.8 and trend == "UPTREND":
        signal_score += 1
        reasons.append("Stoch RSI 과열 + 상승추세 일치")
    elif stoch_rsi < 0.2 and trend == "DOWNTREND":
        signal_score += 1
        reasons.append("Stoch RSI 과매도 + 하락추세 일치")
    else:
        reasons.append("Stoch RSI 단독 과열/과매도 → 보류")

    if trend == "UPTREND" and signal == "BUY":
        signal_score += 1
        reasons.append("추세 상승 + 매수 일치")

    if trend == "DOWNTREND" and signal == "SELL":
        signal_score += 1
        reasons.append("추세 하락 + 매도 일치")

    if liquidity == "좋음":
        signal_score += 1
        reasons.append("유동성 좋음")

    if pattern in ["HAMMER", "BULLISH_ENGULFING", "SHOOTING_STAR", "BEARISH_ENGULFING"]:
        signal_score += 1
        reasons.append(f"캔들패턴 추가 가점: {pattern}")


    return signal_score, reasons

app = FastAPI()

OANDA_API_KEY = os.getenv("OANDA_API_KEY")
ACCOUNT_ID = os.getenv("ACCOUNT_ID")
openai.api_key = os.getenv("OPENAI_API_KEY")


def analyze_highs_lows(candles, window=20):
    highs = candles['high'].tail(window).dropna()
    lows = candles['low'].tail(window).dropna()

    if highs.empty or lows.empty:
        return {"new_high": False, "new_low": False}

    new_high = highs.iloc[-1] > highs.max()
    new_low = lows.iloc[-1] < lows.min()
    return {
        "new_high": new_high,
        "new_low": new_low
    }

@app.post("/webhook")
async def webhook(request: Request):
    print("✅ STEP 1: 웹훅 진입")
    data = json.loads(await request.body())
    pair = data.get("pair")
    signal = data.get("signal")
    print(f"✅ STEP 2: 데이터 수신 완료 | pair: {pair}")

    if check_recent_opposite_signal(pair, signal):    
        print("🚫 양방향 충돌 감지 → 관망")      
        return JSONResponse(content={"status": "WAIT", "reason": "conflict_with_recent_opposite_signal"})
        
    price_raw = data.get("price")
    try:
        price = float(price_raw)
    except (TypeError, ValueError):
        import re
        numeric_match = re.search(r"\d+\.?\d*", str(price_raw))
        price = float(numeric_match.group()) if numeric_match else None
    print(f"✅ STEP 3: 가격 파싱 완료 | price: {price}")

    if price is None:
        return JSONResponse(
            content={"error": "price 필드를 float으로 변환할 수 없습니다"},
            status_code=400
        )

    signal = data.get("signal")
    alert_name = data.get("alert_name", "기본알림")

    candles = get_candles(pair, "M30", 200)
    print("✅ STEP 4: 캔들 데이터 수신")
    # ✅ 최근 10봉 기준으로 지지선/저항선 다시 설정
    candles_recent = candles.tail(10)
    support_resistance = {
        "support": candles_recent["low"].min(),
        "resistance": candles_recent["high"].max()
    }
    
    if candles is None or candles.empty:
        return JSONResponse(content={"error": "캔들 데이터를 불러올 수 없음"}, status_code=400)

    close = candles["close"]
    rsi = calculate_rsi(close)
    stoch_rsi_series = calculate_stoch_rsi(rsi)
    stoch_rsi = stoch_rsi_series.dropna().iloc[-1] if not stoch_rsi_series.dropna().empty else 0
    macd, macd_signal = calculate_macd(close)
    print(f"✅ STEP 5: 보조지표 계산 완료 | RSI: {rsi.iloc[-1]}")
    boll_up, boll_mid, boll_low = calculate_bollinger_bands(close)

    pattern = detect_candle_pattern(candles)
    trend = detect_trend(candles, rsi, boll_mid)
    liquidity = estimate_liquidity(candles)
    support_resistance = {
        "support": candles["low"].min(),
        "resistance": candles["high"].max()
    }

    high_low_analysis = analyze_highs_lows(candles)
    atr = calculate_atr(candles).iloc[-1]
    fibo_levels = calculate_fibonacci_levels(candles["high"].max(), candles["low"].min())

    signal_score = 0
    reasons = []
    # ✅ 여기에 새 뉴스 필터 삽입
    news_risk_score, news_message = fetch_and_score_forex_news(pair)
    signal_score += news_risk_score
    reasons.append(news_message)
    
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
        "news": news_message,
        "new_high": bool(high_low_analysis["new_high"]),
        "new_low": bool(high_low_analysis["new_low"]),
        "atr": atr
    }
    psych_score, psych_reasons = calculate_candle_psychology_score(candles, signal)
    indicator_score, indicator_reasons = score_signal_with_filters(
        rsi.iloc[-1], macd.iloc[-1], macd_signal.iloc[-1], stoch_rsi,
        trend, signal, liquidity, pattern, pair, candles
    )
    signal_score += indicator_score
    reasons += indicator_reasons
    signal_score += psych_score
    reasons += psych_reasons
            
    recent_trade_time = get_last_trade_time()
    time_since_last = datetime.utcnow() - recent_trade_time if recent_trade_time else timedelta(hours=999)
    allow_conditional_trade = time_since_last > timedelta(hours=2)

    gpt_feedback = "GPT 분석 생략: 점수 미달"
    decision, tp, sl = "WAIT", None, None
    executed_price = None

    if signal_score >= 3:
        gpt_feedback = analyze_with_gpt(payload)
        print("✅ STEP 6: GPT 응답 수신 완료")
        decision, tp, sl = parse_gpt_feedback(gpt_feedback, pair)
    else:
        print("🚫 GPT 분석 생략: 점수 3점 미만")
    
    
    print(f"✅ STEP 7: GPT 해석 완료 | decision: {decision}, TP: {tp}, SL: {sl}")
   
    
    # ❌ GPT가 WAIT이면 주문하지 않음
    if decision == "WAIT":
        print("🚫 GPT 판단: WAIT → 주문 실행하지 않음")
        # 시트 기록도 남기기
        outcome_analysis = "WAIT 또는 주문 미실행"
        adjustment_suggestion = ""
        print(f"✅ STEP 10: 전략 요약 저장 호출 | decision: {decision}, TP: {tp}, SL: {sl}")
        log_trade_result(
            pair, signal, decision, signal_score,
            "\n".join(reasons) + f"\nATR: {round(atr or 0, 5)}",
            {}, rsi.iloc[-1], macd.iloc[-1], stoch_rsi,
            pattern, trend, fibo_levels, decision, news_message, gpt_feedback,
            alert_name, tp, sl, executed_price, None,
            outcome_analysis, adjustment_suggestion, [],
            atr
        )
        
        return JSONResponse(content={"status": "WAIT", "message": "GPT가 WAIT 판단"})
        
    #if is_recent_loss(pair) and recent_loss_within_cooldown(pair, window=60):
        #print(f"🚫 쿨다운 적용: 최근 {pair} 손실 후 반복 진입 차단")
        #return JSONResponse(content={"status": "COOLDOWN"})

    
    # ✅ TP/SL 값이 없을 경우 기본 설정 (15pip/10pip 기준)
    effective_decision = decision if decision in ["BUY", "SELL"] else signal
    if (tp is None or sl is None) and price is not None:
        pip_value = 0.01 if "JPY" in pair else 0.0001

        # ATR 기반 보정 추가
        if atr < 0.0007:
            tp_pips = pip_value * 10
            sl_pips = pip_value * 7
        else:
            tp_pips = pip_value * 15
            sl_pips = pip_value * 10

        if effective_decision == "BUY":
            tp = round(price + tp_pips, 5)
            sl = round(price - sl_pips, 5)
        elif effective_decision == "SELL":
            tp = round(price - tp_pips, 5)
            sl = round(price + sl_pips, 5)
        gpt_feedback += "\n⚠️ TP/SL 추출 실패 → ATR 기반 기본값 적용"           
      
        # ✅ 안전 거리 필터 (너무 가까운 주문 방지)
        if not is_min_distance_ok(pair, price, tp, sl):
            print("🚫 TP/SL이 현재가에 너무 가까움 → 주문 취소")
            return JSONResponse(content={"status": "WAIT", "message": "Too close TP/SL, skipped"})


    
    should_execute = False
    # 1️⃣ 기본 진입 조건: GPT가 BUY/SELL 판단 + 점수 4점 이상
    if decision in ["BUY", "SELL"] and signal_score >= 4:
        should_execute = True

    # 2️⃣ 조건부 진입: 최근 2시간 거래 없으면 점수 4점 미만이어도 진입 허용
    elif allow_conditional_trade and signal_score >= 4 and decision in ["BUY", "SELL"]:
        gpt_feedback += "\n⚠️ 조건부 진입: 최근 2시간 거래 없음 → 4점 이상 기준 만족하여 진입 허용"
        should_execute = True
        
    if should_execute:
        units = 100000 if decision == "BUY" else -100000
        digits = 3 if pair.endswith("JPY") else 5
        print(f"[DEBUG] 조건 충족 → 실제 주문 실행: {pair}, units={units}, tp={tp}, sl={sl}, digits={digits}")
        result = place_order(pair, units, tp, sl, digits)
        

    price_movements = []
    pnl = None
    if decision in ["BUY", "SELL"] and tp and sl:
        units = 100000 if decision == "BUY" else -100000
        digits = 3 if pair.endswith("JPY") else 5

        executed_time = datetime.utcnow()
        candles_post = get_candles(pair, "M30", 8)
        price_movements = candles_post[["high", "low"]].to_dict("records")

        try:
            executed_price = float(result['orderFillTransaction']['price'])
        except:
            executed_price = price  # 혹시 못읽으면 기존 price 유지

    if decision in ["BUY", "SELL"] and isinstance(result, dict) and "order_placed" in result.get("status", ""):
        if pnl is not None:
            if pnl > 0:
                if abs(tp - price) < abs(sl - price):
                    outcome_analysis = "성공: TP 우선 도달"
                else:
                    outcome_analysis = "성공: 수익 실현"
            elif pnl < 0:
                if abs(sl - price) < abs(tp - price):
                    outcome_analysis = "실패: SL 우선 터치"
                else:
                    outcome_analysis = "실패: 손실 발생"
            else:
                outcome_analysis = "보류: 실현손익 미확정"
        else:
            outcome_analysis = "보류: 실현손익 미확정"
    else:
        outcome_analysis = "WAIT 또는 주문 미실행"

    adjustment_suggestion = ""
    if outcome_analysis.startswith("실패"):
        if abs(sl - price) < abs(tp - price):
            adjustment_suggestion = "SL 터치 → SL 너무 타이트했을 수 있음, 다음 전략에서 완화 필요"
        elif abs(tp - price) < abs(sl - price):
            adjustment_suggestion = "TP 거의 닿았으나 실패 → TP 약간 보수적일 필요 있음"
            
    print(f"✅ STEP 10: 전략 요약 저장 호출 | decision: {decision}, TP: {tp}, SL: {sl}")
    log_trade_result(
        pair, signal, decision, signal_score,
        "\n".join(reasons) + f"\nATR: {round(atr or 0, 5)}",
        result, rsi.iloc[-1], macd.iloc[-1], stoch_rsi,
        pattern, trend, fibo_levels, decision, news_message, gpt_feedback,
        alert_name, tp, sl, executed_price, pnl, None,
        outcome_analysis, adjustment_suggestion, price_movements,
        atr
         )
    return JSONResponse(content={"status": "completed", "decision": decision})


def calculate_atr(candles, period=14):
    high_low = candles['high'] - candles['low']
    high_close = np.abs(candles['high'] - candles['close'].shift())
    low_close = np.abs(candles['low'] - candles['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def calculate_fibonacci_levels(high, low):
    diff = high - low
    return {
        "0.0": low,
        "0.382": high - 0.382 * diff,
        "0.618": high - 0.618 * diff,
        "1.0": high
    }

def get_candles(pair, granularity, count):
    url = f"https://api-fxpractice.oanda.com/v3/instruments/{pair}/candles"
    headers = {"Authorization": f"Bearer {OANDA_API_KEY}"}
    params = {"granularity": granularity, "count": count, "price": "M"}
    r = requests.get(url, headers=headers, params=params)
    candles = r.json().get("candles", [])

    if not candles:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
         
    return pd.DataFrame([
        {
            "time": c["time"],
            "open": float(c["mid"]["o"]),
            "high": float(c["mid"]["h"]),
            "low": float(c["mid"]["l"]),
            "close": float(c["mid"]["c"]),
            "volume": c.get("volume", 0)
        }
        for c in candles
    ])

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

def detect_box_breakout(candles, pair, box_window=10, box_threshold_pips=30):
    """
    박스권 돌파 감지 (상향/하향 돌파 모두 반환)
    """
    pip_value = 0.01 if pair.endswith("JPY") else 0.0001
    recent_candles = candles.tail(box_window)
    high_max = recent_candles['high'].max()
    low_min = recent_candles['low'].min()
    box_range = (high_max - low_min) / pip_value

    if box_range > box_threshold_pips:
        return {"in_box": False, "breakout": None}

    last_close = recent_candles['close'].iloc[-1]

    if last_close > high_max:
        return {"in_box": True, "breakout": "UP"}
    elif last_close < low_min:
        return {"in_box": True, "breakout": "DOWN"}
    else:
        return {"in_box": True, "breakout": None}

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
    if candles is None or candles.empty:
        return "NEUTRAL"

    last = candles.iloc[-1]
    if pd.isna(last['open']) or pd.isna(last['close']) or pd.isna(last['high']) or pd.isna(last['low']):
        return "NEUTRAL"

    body = abs(last['close'] - last['open'])
    upper_wick = last['high'] - max(last['close'], last['open'])
    lower_wick = min(last['close'], last['open']) - last['low']

    if lower_wick > 2 * body and upper_wick < body:
        return "HAMMER"
    elif upper_wick > 2 * body and lower_wick < body:
        return "SHOOTING_STAR"
    elif body / (last['high'] - last['low']) >= 0.7:
        if last['close'] > last['open']:
            return "LONG_BODY_BULL"
        elif last['close'] < last['open']:
            return "LONG_BODY_BEAR"
    return "NEUTRAL"

def calculate_candle_psychology_score(candles, signal):
    """
    시장 심리 점수화 시스템: 캔들 바디/꼬리 비율 기반으로 정량 심리 점수 반환
    """
    score = 0
    reasons = []

    last = candles.iloc[-1]
    body = abs(last['close'] - last['open'])
    upper_wick = last['high'] - max(last['close'], last['open'])
    lower_wick = min(last['close'], last['open']) - last['low']
    total_range = last['high'] - last['low']
    body_ratio = body / total_range if total_range != 0 else 0

    # ① 장대바디 판단
    if body_ratio >= 0.7:
        if last['close'] > last['open'] and signal == "BUY":
            score += 1
            reasons.append("✅ 강한 장대양봉 → 매수 심리 강화")
        elif last['close'] < last['open'] and signal == "SELL":
            score += 1
            reasons.append("✅ 강한 장대음봉 → 매도 심리 강화")

    # ② 꼬리 비율 심리
    if lower_wick > 2 * body and signal == "BUY":
        score += 1
        reasons.append("✅ 아래꼬리 길다 → 매수 지지 심리 강화")
    if upper_wick > 2 * body and signal == "SELL":
        score += 1
        reasons.append("✅ 위꼬리 길다 → 매도 압력 심리 강화")

    return score, reasons

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

def fetch_and_score_forex_news(pair):
    """
    뉴스 이벤트 위험 점수화 (단계 1+2 통합)
    """
    score = 0
    message = ""

    try:
        response = requests.get("https://www.forexfactory.com/", timeout=5)
        text = response.text

        if "High Impact Expected" in text:
            score -= 2
            message = "⚠️ 고위험 뉴스 존재"
        elif "Medium Impact Expected" in text:
            score -= 1
            message = "⚠️ 중간위험 뉴스"
        elif "Low Impact Expected" in text:
            message = "🟢 낮은 영향 뉴스"

        if pair.startswith("USD") and "Fed Chair" in text:
            score -= 1
            message += " | Fed 연설 포함"
        if pair.endswith("JPY") and "BoJ" in text:
            score -= 1
            message += " | 일본은행 관련 뉴스"

        if message == "":
            message = "🟢 뉴스 영향 적음"
    except Exception as e:
        score = 0
        message = "❓ 뉴스 확인 실패"

    return score, message


def place_order(pair, units, tp, sl, digits):
    url = f"https://api-fxpractice.oanda.com/v3/accounts/{ACCOUNT_ID}/orders"
    headers = {
        "Authorization": f"Bearer {OANDA_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "order": {
            "instrument": pair,
            "units": str(units),
            "type": "MARKET",
            "positionFill": "DEFAULT",
            "takeProfitOnFill": {
                "price": str(round(tp, digits))
            },
            "stopLossOnFill": {
                "price": str(round(sl, digits))
            }
        }
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": str(e)}

import re

# ✅ TP/SL 너무 가까운 거리 제한 필터
def is_min_distance_ok(pair, price, tp, sl, min_distance_pip=8):
    pip_value = 0.01 if pair.endswith("JPY") else 0.0001
    min_distance = pip_value * min_distance_pip

    if abs(price - tp) < min_distance or abs(price - sl) < min_distance:
        return False
    return True


def parse_gpt_feedback(text, pair):
    import re

    decision = "WAIT"
    tp = None
    sl = None

    # ✅ 명확한 판단 패턴 탐색 (정규식 우선)
    decision_patterns = [
        r"(결정|진입\s*판단|신호|방향)\s*(은|:|：)?\s*['\"]?(BUY|SELL|WAIT)['\"]?",
        r"진입\s*방향\s*(은|:|：)?\s*['\"]?(BUY|SELL|WAIT)['\"]?",
        r"판단\s*(은|:|：)?\s*['\"]?(BUY|SELL|WAIT)['\"]?",
    ]

    for pat in decision_patterns:
        d = re.search(pat, text.upper())
        if d:
            decision = d.group(3)
            break

    # ✅ fallback: "BUY" 또는 "SELL" 단독 등장 시 인식
    if decision == "WAIT":
        if "BUY" in text.upper() and "SELL" not in text.upper():
            decision = "BUY"
        elif "SELL" in text.upper() and "BUY" not in text.upper():
            decision = "SELL"

    # ✅ TP/SL 추출 (가장 마지막 숫자 사용)
    tp_line = next((line for line in text.splitlines() if "TP:" in line.upper() or "TP 제안 값" in line or "목표" in line), "")
    sl_line = next((line for line in text.splitlines() if re.search(r"\bSL\s*:?\s*\d+\.\d{4,5}", line.upper())), "")
    if not sl_line:
        print("❗ SL 라인 탐색 실패 → GPT 파서에서 예외로 처리")
        decision = "WAIT"
        return decision, None, None


    def extract_avg_price(line):
        matches = re.findall(r"\b\d{1,5}\.\d{1,5}\b", line)  # 가격 패턴만 추출
        if len(matches) >= 2:
            return (float(matches[0]) + float(matches[1])) / 2
        elif matches:
            return float(matches[0])
        else:
            return None

    tp = extract_avg_price(tp_line)
    sl = extract_avg_price(sl_line)

    # ✅ JPY 페어일 경우 자리수 자동 변환
    if "JPY" in pair:
        if tp is not None:
            tp = round(tp, 3)
        if sl is not None:
            sl = round(sl, 3)
    else:
        if tp is not None:
            tp = round(tp, 5)
        if sl is not None:
            sl = round(sl, 5)

    return decision, tp, sl
    
def analyze_with_gpt(payload):
    headers = {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}", "Content-Type": "application/json"}
    messages = [
        {"role": "system", "content": "너는 실전 FX 트레이딩 전략 조력자야. (1)아래 JSON 데이터를 기반으로 전략 리포트를 생성하고, 진입 판단(BUY, SELL, WAIT)과 TP, SL 값을 제시해줘. RSI, MACD, Stoch RSI, 추세 점수, 캔들 패턴 점수의 총합이 6점 이상인 경우에는 보수적 WAIT 대신 진입(BUY 또는 SELL) 판단을 조금 더 적극적으로 검토하라. (2)거래는 기본 1~2시간 내에 청산하는것을 목표로 너무 TP,SL을 멀리 떨어지지 않게 7 PIP~10 PIP이내로 설정하자 (tp:sl 2:1비율) (3)지지선(support)과 저항선(resistance)은 최근 1시간봉 기준 마지막 10봉에서의 고점/저점 기준으로 이미 계산된 사용하고, 아래 데이터에 포함되어 있다 그러니 분석 시에는 반드시 이 숫자만 기준으로 판단해라. 그 외 고점/저점은 무시해라. (4)분석할땐 캔들의 추세뿐만 아니라, 보조 지표들의 추세&흐름도 꼭 같이 파악해서 추세를 파악해서 분석해.  (5)그리고 너의 분석의 마지막은 항상 진입판단: BUY/SELL/WAIT 이라고 명료하게 보여줘 저 형식으로 (6) SL와 TP도 범위형 표현은 절대 사용하지 말고 단일수치값으로 명료하게 보여줘. (7) 최근 지지/저항을 중심으로, 현재가가 저항 근처면 짧은 TP 설정, 지지 멀다면 넓은 SL 허용한다 대신에 너무 많이 멀어지지 않도록. (8)피보나치 수렴 또는 확장 여부를 참고하여 돌파 가능성 있으면 TP를 과감하게 약간 확장 가능. 캔들패턴 뿐만 아니라 최근 파동(신고점/신저점 여부), 박스권 유지 여부까지 참고.ATR과 볼린저 폭을 함께 참고하여 변동성이 급격히 축소되는 경우에는 보수적으로 TP/SL 설정한다. 나의 최종목표는 거래 하나당 50~100불정도 가져가는게 목표이다."},
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
        
import math

def safe_float(val):
    try:
        if val is None:
            return ""
        val = float(val)
        if math.isnan(val) or math.isinf(val):
            return ""
        return round(val, 5)
    except:
        return ""


def log_trade_result(pair, signal, decision, score, notes, result=None, rsi=None, macd=None, stoch_rsi=None, pattern=None, trend=None, fibo=None, gpt_decision=None, news=None, gpt_feedback=None, alert_name=None, tp=None, sl=None, entry=None, price=None, pnl=None, outcome_analysis=None, adjustment_suggestion=None, price_movements=None, atr=None):
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name("/etc/secrets/google_credentials.json", scope)
    client = gspread.authorize(creds)
    sheet = client.open("민균 FX trading result").sheet1
    now_atlanta = datetime.utcnow() - timedelta(hours=4)
    if isinstance(price_movements, list):
        try:
            filtered_movements = [
                {
                    "high": float(p["high"]),
                    "low": float(p["low"])
                }
                for p in price_movements
                if isinstance(p, dict)
                and "high" in p and "low" in p
                and isinstance(p["high"], (float, int)) and isinstance(p["low"], (float, int))
                and not math.isnan(p["high"]) and not math.isnan(p["low"])
                and not math.isinf(p["high"]) and not math.isinf(p["low"])
            ]
        except Exception as e:
            print("❗ price_movements 정제 실패:", e)
            filtered_movements = []
    else:
        filtered_movements = []

    # ✅ 분석용 filtered_movements로 신고점/신저점 판단
    is_new_high = ""
    is_new_low = ""
    if len(filtered_movements) > 1:
        try:
            highs = [p["high"] for p in filtered_movements[:-1]]
            lows = [p["low"] for p in filtered_movements[:-1]]
            last = filtered_movements[-1]
            if "high" in last and highs and last["high"] > max(highs):
                is_new_high = "신고점"
            if "low" in last and lows and last["low"] < min(lows):
                is_new_low = "신저점"
        except Exception as e:
            print("❗ 신고점/신저점 계산 실패:", e)

    # ✅ Google Sheet 저장용 문자열로 변환
    

    filtered_movement_str = ", ".join([
        f"H: {round(p['high'], 5)} / L: {round(p['low'], 5)}"
        for p in filtered_movements[-5:]
        if isinstance(p, dict) and "high" in p and "low" in p
    ])


    try:
        filtered_movement_str = ", ".join([
            f"H: {round(p['high'], 5)} / L: {round(p['low'], 5)}"
            for p in filtered_movements[-5:]
            if isinstance(p, dict) and "high" in p and "low" in p and
               isinstance(p['high'], (float, int)) and isinstance(p['low'], (float, int)) and
               not math.isnan(p['high']) and not math.isnan(p['low']) and
               not math.isinf(p['high']) and not math.isinf(p['low'])
        ])
    except Exception as e:
        print("❌ filtered_movement_str 변환 실패:", e)
        filtered_movement_str = "error_in_conversion"
    
        if not filtered_movement_str:
            filtered_movement_str = "no_data"
   
    row = [
      
        str(now_atlanta), pair, alert_name or "", signal, decision, score,
        safe_float(rsi), safe_float(macd), safe_float(stoch_rsi),
        pattern or "", trend or "", fibo.get("0.382", ""), fibo.get("0.618", ""),
        gpt_decision or "", news or "", notes,
        json.dumps(result, ensure_ascii=False) if isinstance(result, dict) else (result or "미정"),
        gpt_feedback or "",        
        safe_float(price), safe_float(tp), safe_float(sl), safe_float(pnl),
        is_new_high,
        is_new_low,
        safe_float(atr),
        news,
        outcome_analysis or "",
        adjustment_suggestion or "",
        gpt_feedback or "",
        filtered_movement_str
    ]

    clean_row = []
    for v in row:
        if isinstance(v, (dict, list)):
            clean_row.append(json.dumps(v, ensure_ascii=False))
        elif isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            clean_row.append("")
        else:
            clean_row.append(v)

    print("✅ STEP 8: 시트 저장 직전", clean_row)
    for idx, val in enumerate(clean_row):
         if isinstance(val, (dict, list)):
            print(f"❌ [오류] clean_row[{idx}]에 dict 또는 list가 남아 있음 → {val}")
    
    for idx, val in enumerate(clean_row):
        if isinstance(val, (dict, list)):
            print(f"❌ [디버그] clean_row[{idx}]는 dict 또는 list → {val}")
    print(f"🧪 최종 clean_row 길이: {len(clean_row)}")

    try:
        sheet.append_row(clean_row)
    except Exception as e:
        print("❌ Google Sheet append_row 실패:", e)
        print("🧨 clean_row 전체 내용:\n", clean_row)


def get_last_trade_time():
    try:
        with open("/tmp/last_trade_time.txt", "r") as f:
            return datetime.fromisoformat(f.read().strip())
            
    except:
        return None













@app.post("/fastfury_webhook")
async def fastfury_webhook(request: Request):
    data = await request.json()

    pair_raw = data.get("pair")  # 예: "USD_JPY"
    signal = data.get("signal")  # BUY / SELL
    alert_name = data.get("alert_name", "")
    price_raw = data.get("price")

    # ✅ 변환: USD_JPY → USDJPY (OANDA용으로)
    pair = pair_raw.replace("_", "")

    try:
        price = float(price_raw)
    except:
        import re
        numeric_match = re.search(r"\d+\.?\d*", str(price_raw))
        price = float(numeric_match.group()) if numeric_match else None

    if price is None:
        return {"status": "error", "message": "가격 변환 실패"}

    print(f"✅ FAST FURY ALGO 진입: {pair} | {signal} | {price}")

    # 👉 여기에 GPT 간이필터 또는 본 전략 로직 연결 가능
    # ✅ 보조지표 계산 시작 (15분봉 기준)
    candles = get_candles(pair, "M5", 100)
    close = candles["close"]

    rsi = calculate_rsi(close)
    macd, macd_signal = calculate_macd(close)
    stoch_rsi_series = calculate_stoch_rsi(rsi)
    stoch_rsi = stoch_rsi_series.dropna().iloc[-1] if not stoch_rsi_series.dropna().empty else 0

    boll_up, boll_mid, boll_low = calculate_bollinger_bands(close)
    pattern = detect_candle_pattern(candles)
    trend = detect_trend(candles, rsi, boll_mid)
    liquidity = estimate_liquidity(candles)

    # ✅ GPT 호출 (TP/SL 없이 판단만 요청)
    payload = {
        "pair": pair, "price": price, "signal": signal,
        "rsi": rsi.iloc[-1], "macd": macd.iloc[-1], "macd_signal": macd_signal.iloc[-1],
        "stoch_rsi": stoch_rsi, "bollinger_upper": boll_up.iloc[-1], "bollinger_lower": boll_low.iloc[-1],
        "pattern": pattern, "trend": trend, "liquidity": liquidity
    }

    gpt_result = analyze_with_gpt(payload)

    # GPT 결과 파싱 (BUY/SELL/WAIT)
    if "BUY" in gpt_result:
        decision = "BUY"
    elif "SELL" in gpt_result:
        decision = "SELL"
    else:
        decision = "WAIT"

    if decision == "WAIT":
        return {"status": "WAIT", "message": "GPT 판단으로 관망"} 

    # 이제 GPT 최종 decision을 기준으로 진입
    tp = None
    sl = None

    pip_value = 0.01
    tp_pips = pip_value * 7
    sl_pips = pip_value * 4

    if decision == "BUY":
        units = 100000
        tp = round(price + tp_pips, 3)
        sl = round(price - sl_pips, 3)
    elif decision == "SELL":
        units = -100000
        tp = round(price - tp_pips, 3)
        sl = round(price + sl_pips, 3)
    else:
        return {"status": "NO_ACTION"}

    print(f"🚀 주문 실행: {pair} {decision} {units} @ {price} TP: {tp} SL: {sl}")
    result = place_order(pair, units, tp=tp, sl=sl, digits=3)
    print("✅ 주문 실행 완료:", result)

    # 스프레드시트 기록 호출
    log_trade_result(
        pair=pair, 
        signal=signal, 
        decision=decision, 
        score=None,  # 지금 이 버전엔 승점 없음
        notes="FastFury Hybrid 실전진입", 
        result=result, 
        rsi=rsi.iloc[-1], 
        macd=macd.iloc[-1], 
        stoch_rsi=stoch_rsi, 
        pattern=pattern, 
        trend=trend, 
        fibo={},  # 피보나치 안씀
        gpt_decision=decision, 
        news=None, 
        gpt_feedback=gpt_result, 
        alert_name=alert_name, 
        tp=tp, 
        sl=sl, 
        entry=price, 
        price=price, 
        pnl=None, 
        outcome_analysis=None, 
        adjustment_suggestion=None, 
        price_movements=None, 
        atr=None
    )


    
    return result


