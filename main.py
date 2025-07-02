  
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

# score_signal_with_filters 위쪽에 추가
def must_capture_opportunity(rsi, stoch_rsi, macd, macd_signal, pattern, candles):
    opportunity_score = 0
    reasons = []

    if stoch_rsi < 0.05 and rsi > 50 and macd > macd_signal:
        opportunity_score += 2
        reasons.append("💡 Stoch RSI 극단 과매도 + RSI 50 상단 돌파 + MACD 상승 → 강력한 BUY 기회")

    if stoch_rsi > 0.95 and rsi < 50 and macd < macd_signal:
        opportunity_score += 2
        reasons.append("💡 Stoch RSI 극단 과매수 + RSI 50 이탈 + MACD 하락 → 강력한 SELL 기회")

    if pattern in ["BULLISH_ENGULFING", "BEARISH_ENGULFING"]:
        opportunity_score += 1
        reasons.append(f"💡 {pattern} 발생 → 심리 반전 확률↑")

    if 48 < rsi < 52:
        opportunity_score += 1
        reasons.append("💡 RSI 50 근접 – 심리 경계선 전환 주시")

    return opportunity_score, reasons

def additional_opportunity_score(rsi, stoch_rsi, macd, macd_signal, pattern, trend):
    """ 기존 필터 이후, 추가 가중치 기반 보완 점수 """
    score = 0
    reasons = []

    # RSI 30 이하
    if rsi < 30:
        score += 1.5
        reasons.append("🔴 RSI 30 이하 (추가 기회 요인)")

    # Stoch RSI 극단
    if stoch_rsi < 0.05:
        score += 1.5
        reasons.append("🟢 Stoch RSI 0.05 이하 (반등 기대)")

    # MACD 상승 전환
    if macd > 0 and macd > macd_signal:
        score += 1
        reasons.append("🟢 MACD 상승 전환 (추가 확인 요인)")

    # 캔들 패턴
    if pattern in ["BULLISH_ENGULFING", "BEARISH_ENGULFING"]:
        score += 1
        reasons.append(f"📊 {pattern} 발생 (심리 반전)")

    # 추세가 중립일 때: 추가 감점
    if trend == "NEUTRAL":
        score -= 0.5
        reasons.append("⚠ 중립 추세 → 추세 부재로 감점")

    return score, reasons



def conflict_check(rsi, pattern, trend, signal):
    """
    추세-패턴-시그널 충돌 방지 필터 (V2 최종)
    """

    # 1️⃣ 기본 추세-패턴 충돌 방지
    if rsi > 85 and pattern in ["SHOOTING_STAR", "BEARISH_ENGULFING"] and trend == "UPTREND":
        return True
    if rsi < 15 and pattern in ["HAMMER", "BULLISH_ENGULFING"] and trend == "DOWNTREND":
        return True

    # 2️⃣ 캔들패턴이 없는데 시그널과 추세가 역방향이면 관망
    if pattern == "NEUTRAL":
        if signal == "BUY" and trend == "UPTREND":
            return False
        if signal == "SELL" and trend == "DOWNTREND":
            return False

    return False
    
def check_recent_opposite_signal(pair, current_signal, within_minutes=12):
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

    score, base_reasons = must_capture_opportunity(rsi, stoch_rsi, macd, macd_signal, pattern, candles)
    extra_score, extra_reasons = additional_opportunity_score(rsi, stoch_rsi, macd, macd_signal, pattern, trend)

    signal_score += score + extra_score
    reasons.extend(base_reasons + extra_reasons)

    
    # ✅ 거래 제한 시간 필터 (애틀랜타 기준)
    now_utc = datetime.utcnow()
    now_atlanta = now_utc - timedelta(hours=4)

    if now_atlanta.hour >= 22 or now_atlanta.hour <= 4:
        if pair in ["EUR_USD", "GBP_USD"]:
            reasons.append("❌ 심야 유동성 부족 → EURUSD, GBPUSD 거래 제한")
            return 0, reasons
    

    conflict_flag = conflict_check(rsi, pattern, trend, signal)

    # 보완 조건 정의: 극단적 RSI + Stoch RSI or MACD 반전 조짐
    extreme_buy = signal == "BUY" and rsi < 25 and stoch_rsi < 0.2
    extreme_sell = signal == "SELL" and rsi > 75 and stoch_rsi > 0.8
    macd_reversal_buy = signal == "BUY" and macd > macd_signal and trend == "DOWNTREND"
    macd_reversal_sell = signal == "SELL" and macd < macd_signal and trend == "UPTREND"

    # 완화된 조건: 강력한 역추세 진입 근거가 있을 경우 관망 무시
    if conflict_flag:
        if extreme_buy or extreme_sell or macd_reversal_buy or macd_reversal_sell:
            reasons.append("🔄 추세-패턴 충돌 BUT 강한 역추세 조건 충족 → 진입 허용")
        else:
            reasons.append("⚠️ 추세-패턴 충돌 + 보완 조건 미충족 → 관망")
            return 0, reasons

    # ✅ V3 과매도 SELL 방어 필터 추가
    if signal == "SELL" and rsi < 40:
        if macd > macd_signal and stoch_rsi > 0.5:
            signal_score += 1
            reasons.append("❗ 과매도 SELL 경계지만 MACD + Stoch RSI 상승 → 조건부 진입 허용")
        else:
            reasons.append("❗ 과매도 SELL 방어 → 관망 강제 (V3 강화)")
            return 0, reasons
        
    if rsi < 30 and pattern not in ["HAMMER", "BULLISH_ENGULFING"]:
        if macd < macd_signal and trend == "DOWNTREND":
            reasons.append("RSI < 30 but MACD & Trend 약세 지속 → 진입 허용")
        else:
            return 0, ["RSI < 30 but 반등 조건 미약 → 관망"]

    if rsi > 70 and pattern not in ["SHOOTING_STAR", "BEARISH_ENGULFING"]:
        if macd > macd_signal and trend == "UPTREND":
            reasons.append("RSI > 70 but MACD & Trend 강세 → 진입 허용")
        else:
            return 0, ["RSI > 70 but 캔들/지표 약함 → 관망"]
        
    # === 눌림목 BUY 강화: GBPUSD 한정 ===
    if pair == "GBP_USD" and signal == "BUY":
        if trend == "UPTREND":
            signal_score += 1
            reasons.append("GBPUSD 강화: UPTREND 유지 → 매수 기대")
        if 40 <= rsi <= 50:
            signal_score += 1
            reasons.append("GBPUSD 강화: RSI 40~50 눌림목 영역")
        if 0.1 <= stoch_rsi <= 0.3:
            signal_score += 1
            reasons.append("GBPUSD 강화: Stoch RSI 바닥 반등 초기")
        if pattern in ["HAMMER", "LONG_BODY_BULL"]:
            signal_score += 1
            reasons.append("GBPUSD 강화: 매수 캔들 패턴 확인")
        if macd > 0:
            signal_score += 1
            reasons.append("GBPUSD 강화: MACD 양수 유지 (상승 흐름 유지)")
    
    if 45 <= rsi <= 60 and signal == "BUY":
        signal_score += 1
        reasons.append("RSI 중립구간 (45~60) → 반등 기대 가점")

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
    

    if pair == "USD_JPY":
        if (macd - macd_signal) > 0.0002 and trend == "UPTREND":
            signal_score += 4
            reasons.append("USDJPY 강화: MACD 골든크로스 + 상승추세 일치 → breakout 강세")
        elif (macd_signal - macd) > 0.0002 and trend == "DOWNTREND":
            signal_score += 4
            reasons.append("USDJPY 강화: MACD 데드크로스 + 하락추세 일치 → 하락 강화")
        elif abs(macd - macd_signal) > 0.0005:
            signal_score += 1
            reasons.append("USDJPY MACD 교차 발생 (추세불명확)")
        else:
            reasons.append("USDJPY MACD 미세변동 → 가점 보류")

            # ✅ 히스토그램 증가 보조 판단 (미세하지만 상승 흐름일 경우)
            macd_hist = macd - macd_signal
            if macd_hist > 0:
                signal_score += 1
                reasons.append("MACD 미세하지만 히스토그램 증가 → 상승 초기 흐름")
      
            
    else:
        if (macd - macd_signal) > 0.0002 and trend == "UPTREND":
            signal_score += 3
            reasons.append("MACD 골든크로스 + 상승추세 일치 → breakout 강세")
        elif (macd_signal - macd) > 0.0002 and trend == "DOWNTREND":
            signal_score += 3
            reasons.append("MACD 데드크로스 + 하락추세 일치 → 하락 강화")
        elif abs(macd - macd_signal) > 0.0005:
            signal_score += 1
            reasons.append("MACD 교차 발생 (추세불명확)")
        if macd < macd_signal and trend == "DOWNTREND":
            signal_score += 1
            reasons.append("MACD 약한 데드 + 하락추세 → 약한 SELL 지지")
        else:
            reasons.append("MACD 미세변동 → 가점 보류")

    
    if stoch_rsi > 0.8:
        if trend == "UPTREND" and rsi < 70:
            if pair == "USD_JPY":
                signal_score += 3  # USDJPY만 강화
                reasons.append("USDJPY 강화: Stoch RSI 과열 + 상승추세 일치")
            else:
                signal_score += 2
                reasons.append("Stoch RSI 과열 + 상승추세 일치")
        elif trend == "NEUTRAL" and signal == "SELL" and rsi > 60:
            signal_score += 1
            reasons.append("Stoch RSI 과열 + neutral 매도 조건 → 피로 누적 매도 가능성")
        else:
            reasons.append("Stoch RSI 과열 → 고점 피로, 관망")
    elif stoch_rsi < 0.2:
        if trend == "DOWNTREND" and rsi > 30:
            signal_score += 2
            reasons.append("Stoch RSI 과매도 + 하락추세 일치")
        elif trend == "NEUTRAL" and signal == "SELL" and rsi > 50:
            signal_score += 1
            reasons.append("Stoch RSI 과매도 + neutral 매도 전환 조건")
        elif trend == "DOWNTREND":
            signal_score += 2
            reasons.append("Stoch RSI 과매도 + 하락추세 일치 (보완조건 포함)")
        elif trend == "NEUTRAL" and rsi < 50:
            signal_score += 1
            reasons.append("Stoch RSI 과매도 + RSI 50 이하 → 약세 유지 SELL 가능")
        
        if stoch_rsi < 0.1:
            signal_score += 1
            reasons.append("Stoch RSI 0.1 이하 → 극단적 과매도 가점")
        
        else:
            reasons.append("Stoch RSI 과매도 → 저점 피로, 관망")
    else:
        reasons.append("Stoch RSI 중립")

    if trend == "UPTREND" and signal == "BUY":
        signal_score += 1
        reasons.append("추세 상승 + 매수 일치")

    if trend == "DOWNTREND" and signal == "SELL":
        signal_score += 1
        reasons.append("추세 하락 + 매도 일치")

    if liquidity == "좋음":
        signal_score += 1
        reasons.append("유동성 좋음")
    last_3 = candles.tail(3)
    if all(last_3["close"] < last_3["open"]) and trend == "DOWNTREND" and pattern == "NEUTRAL":
        signal_score += 1
        reasons.append("최근 3봉 연속 음봉 + 하락추세 → 패턴 부재 보정 SELL 가점")
    
    if pattern in ["BULLISH_ENGULFING", "HAMMER"]:
        signal_score += 1  # 강력 패턴은 유지
    elif pattern in ["LONG_BODY_BULL"]:
        signal_score += 0.5  # 장대양봉은 소폭만 가점 (이번 케이스 반영)
    elif pattern in ["SHOOTING_STAR", "BEARISH_ENGULFING"]:
        signal_score -= 1  # 반전 패턴은 역가점
    # 교과서적 기회 포착 보조 점수
    op_score, op_reasons = must_capture_opportunity(rsi, stoch_rsi, macd, macd_signal, pattern, candles)
    if op_score > 0:
        signal_score += op_score
        reasons += op_reasons

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
        # 보정 적용
        if decision in ["BUY", "SELL"] and tp and sl:
            tp, sl = adjust_tp_sl_distance(price, tp, sl, atr, pair)
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

    
    # ✅ TP/SL 값이 없을 경우 기본 설정 (ATR 기반 세분화 보정)
    if (tp is None or sl is None) and price is not None:
        pip_value = 0.01 if "JPY" in pair else 0.0001

        # 더 세분화된 ATR 기반 설정
        if atr >= 0.18:
            tp_pips = pip_value * 25
            sl_pips = pip_value * 12
        elif atr >= 0.13:
            tp_pips = pip_value * 20
            sl_pips = pip_value * 10
        elif atr >= 0.08:
            tp_pips = pip_value * 15
            sl_pips = pip_value * 10
        else:
            tp_pips = pip_value * 10
            sl_pips = pip_value * 7

        if decision == "BUY":
            tp = round(price + tp_pips, 5 if pip_value == 0.0001 else 3)
            sl = round(price - sl_pips, 5 if pip_value == 0.0001 else 3)
        elif decision == "SELL":
            tp = round(price - tp_pips, 5 if pip_value == 0.0001 else 3)
            sl = round(price + sl_pips, 5 if pip_value == 0.0001 else 3)      
      
        # ✅ 안전 거리 필터 (너무 가까운 주문 방지)
        if not is_min_distance_ok(pair, price, tp, sl, atr):
            print(f"🚫 TP/SL 거리 미달 → TP: {tp}, SL: {sl}, 현재가: {price}, ATR: {atr}")
            return JSONResponse(content={"status": "WAIT", "message": "Too close TP/SL, skipped"})


    result = None  # 🧱 주문 실행 여부와 무관하게 선언 (에러 방지용)
    
    should_execute = False
    # 1️⃣ 기본 진입 조건: GPT가 BUY/SELL 판단 + 점수 4점 이상
    if decision in ["BUY", "SELL"] and signal_score >= 4:
        should_execute = True

    # 2️⃣ 조건부 진입: 최근 2시간 거래 없으면 점수 4점 미만이어도 진입 허용
    elif allow_conditional_trade and signal_score >= 4 and decision in ["BUY", "SELL"]:
        gpt_feedback += "\n⚠️ 조건부 진입: 최근 2시간 거래 없음 → 4점 이상 기준 만족하여 진입 허용"
        should_execute = True

    print(f"🚀 주문 조건 충족 | 페어: {pair}, 결정: {decision}, 점수: {signal_score}")
    print(f"🔧 TP: {tp}, SL: {sl}, 현재가: {price}, ATR: {atr}")  
    if should_execute:
        units = 100000 if decision == "BUY" else -100000
        digits = 3 if pair.endswith("JPY") else 5
        print(f"[DEBUG] 조건 충족 → 실제 주문 실행: {pair}, units={units}, tp={tp}, sl={sl}, digits={digits}")
        result = place_order(pair, units, tp, sl, digits)  # ⬅ 여기서 꼭 할당
        

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

    if result and decision in ["BUY", "SELL"] and isinstance(result, dict) and "order_placed" in result.get("status", ""):
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
        result = response.json()
        print("📦 OANDA 주문 응답:", result)
        return result
    except requests.exceptions.RequestException as e:
        print("❌ OANDA 요청 실패:", str(e))
        return {"status": "error", "message": str(e)}

import re

# ✅ 페어별 ATR 기반 TP/SL 거리 필터 (A안 적용)
def is_min_distance_ok(pair, price, tp, sl, atr):
    """
    페어별 ATR factor 적용
    """
    if pair == "USD_JPY":
        atr_factor = max(0.35, 0.05 / atr) 
    else:
        atr_factor = max(0.6, 0.0010 / atr)

    min_distance = atr * atr_factor
    if abs(price - tp) < min_distance or abs(price - sl) < min_distance:
        return False
    return True
  
def allow_narrow_tp_sl(signal_score, atr, liquidity, pair, tp, sl, min_gap_pips=5):
    pip_value = 0.01 if "JPY" in pair else 0.0001
    min_tp_sl_gap = pip_value * min_gap_pips

    if abs(tp - sl) < min_tp_sl_gap:
        if signal_score >= 7 and atr < 0.2 and liquidity == "좋음":
            print("✅ 신호 강도 & 유동성 조건 만족 → 좁은 TP-SL 예외 허용")
            return True
        else:
            print("❌ TP-SL 간격 부족 & 조건 미충족 → 진입 차단")
            return False
    return True


def parse_gpt_feedback(text, pair):
    import re

    decision = "WAIT"
    tp = None
    sl = None

    # ✅ 명확한 판단 패턴 탐색 (정규식 우선)
    decision_patterns = [
        r"(결정|판단)\s*(판단|신호|방향)?\s*(은|:|：)?\s*[\"']?(BUY|SELL|WAIT)[\"']?",
        r"진입\s*방향\s*(은|:|：)?\s*['\"]?(BUY|SELL|WAIT)['\"]?",
        r"판단\s*(은|:|：)?\s*['\"]?(BUY|SELL|WAIT)['\"]?",
    ]

    for pat in decision_patterns:
        d = re.search(pat, text.upper())
        if d:
            decision = d.group(3)
            break

    if decision == "BUY" or decision == "SELL":
        if not allow_narrow_tp_sl(signal_score, atr, liquidity, pair, tp, sl):
            return "WAIT", None, None
    
    # ✅ fallback: "BUY" 또는 "SELL" 단독 등장 시 인식
    if decision == "WAIT":
        if "BUY" in text.upper() and "SELL" not in text.upper():
            decision = "BUY"
        elif "SELL" in text.upper() and "BUY" not in text.upper():
            decision = "SELL"

    # GPT가 제시한 TP/SL이 너무 가까울 경우 보정
    def adjust_tp_sl_distance(price, tp, sl, atr, pair):
        if atr is None or tp is None or sl is None:
            return tp, sl

        pip_value = 0.01 if "JPY" in pair else 0.0001
        min_gap_pips = 5
        min_sl_distance = atr * 0.5  # SL과 현재가 간 거리 최소 확보
        min_tp_sl_gap = pip_value * min_gap_pips  # TP-SL 간 최소 거리

        # SL 보정
        if abs(price - sl) < min_sl_distance:
            if price > sl:
                sl = round(price - min_sl_distance, 3 if pair.endswith("JPY") else 5)
            else:
                sl = round(price + min_sl_distance, 3 if pair.endswith("JPY") else 5)

        # TP/SL 간 거리 보정
        if abs(tp - sl) < min_tp_sl_gap and not allow_narrow_tp_sl(signal_score, atr, liquidity, pair, tp, sl):
            print("⚠️ TP와 SL 간격이 부족하지만 진입 강행 (조건 완화)")
            # 보정 불가능하면 None 반환
        # ✅ TP가 현재가에 너무 가까운 경우 → 진입 제한

        print(f"[PARSE 최종] 결정: {decision}, TP: {tp}, SL: {sl}")
        return tp, sl
    

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

    # ✅ fallback: SL 없을 경우 자동 계산 보완
    if sl is None and decision in ["BUY", "SELL"] and tp is not None:
        atr_match = re.search(r"ATR\s*[:=]\s*([\d\.]+)", text.upper())
        if atr_match:
            atr = float(atr_match.group(1))
            if decision == "BUY":
                sl = round(tp - (atr * 2), 3 if "JPY" in pair else 5)
            elif decision == "SELL":
                sl = round(tp + (atr * 2), 3 if "JPY" in pair else 5)

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
        {"role": "system", "content": "너는 실전 FX 트레이딩 전략 조력자야. (1)아래 JSON 데이터를 기반으로 전략 리포트를 생성하고, 진입 판단(BUY, SELL, WAIT)과 TP, SL 값을 제시해줘. RSI, MACD, Stoch RSI, 추세 점수, 캔들 패턴 점수의 총합이 4점 이상인 경우에는 보수적 WAIT 대신 진입(BUY 또는 SELL) 판단을 조금 더 적극적으로 검토하라. (2) 거래는 기본적으로 1~2시간 내 청산을 목표로 하되, SL은 너무 짧지 않도록 ATR의 최소 50% 이상 거리로 설정해야 한다. SL과 TP는 너무 짧으면 OANDA 서버에서 주문이 거절되므로 반드시 현재가보다 8PIP정돈느 차이나게 설정한다. TP는 SL보다 넓게 설정하되, TP와 SL 사이의 간격도 최소 10 PIP 이상 확보해야 한다. 전략의 특성상 거래는 1~2시간 이내 청산을 목표로 하며, TP와 sL은 현재가(현재가격) 에서 20 pips를 초과하지 않도록 설정하라.   (3)지지선(support)과 저항선(resistance)은 최근 1시간봉 기준 마지막 6봉에서의 고점/저점 기준으로 이미 계산된 사용하고, 아래 데이터에 포함되어 있다 그러니 분석 시에는 반드시 이 숫자만 기준으로 판단해라. 그 외 고점/저점은 무시해라. (4)분석할땐 캔들의 추세뿐만 아니라, 보조 지표들의 추세&흐름도 꼭 같이 파악해서 추세를 파악해서 분석해.  (5)그리고 너의 분석의 마지막은 항상 진입판단: BUY/SELL/WAIT 이라고 명료하게 이 형식으로 보여줘 (6) SL와 TP도 범위형 표현은 절대 사용하지 말고 단일수치값으로 명료하게 보여주고 숫자 외에는 다른 말은 추가로 보여주지마. 왜냐하면 그 숫자만 함수로 불러와서 거래 할 것이기 때문에 (7) 최근 지지/저항을 중심으로, 현재가가 저항 근처면 짧은 TP 설정, 지지 멀다면 넓은 SL 허용한다 대신에 너무 많이 멀어지지 않도록. (8)피보나치 수렴 또는 확장 여부를 참고하여 돌파 가능성 있으면 TP를 과감하게 약간 확장 가능. 캔들패턴 뿐만 아니라 최근 파동(신고점/신저점 여부), 박스권 유지 여부까지 참고.ATR과 볼린저 폭을 함께 참고하여 변동성이 급격히 축소되는 경우에는 보수적으로 TP/SL 설정한다. 나의 최종목표는 거래 하나당 50~100불정도 가져가는게 목표이다."},
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
    
    # ✅ 여기를 새로 추가하세요 (row 정의 바로 위)
    result = "미정"  # OANDA 주문 결과 기본값
    filtered_movement_str = "no_data"
    rejection_reason = ""
    too_close_to_SL = False
    signal_score = score if 'score' in locals() else 0
    effective_decision = decision if 'decision' in locals() else ""
    

    def conflict_check():                  # 추세/패턴 충돌 필터 더미 함수
        return False
    
    
    row = [
      
        str(now_atlanta), pair, alert_name or "", signal, decision, score,
        safe_float(rsi), safe_float(macd), safe_float(stoch_rsi),
        pattern or "", trend or "", fibo.get("0.382", ""), fibo.get("0.618", ""),
        gpt_decision or "", news or "", notes,
        rejection_reason,    # ✅ 여기 새로 추가
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

    print("🧾 row 길이:", len(row))
    print("📋 row 내용:\n", row)
    rejection_reasons = []
    row[12] = " / ".join(rejection_reasons) if rejection_reasons else ""

    if too_close_to_SL:  # SL이 최소 거리보다 가까운 경우
        rejection_reasons.append("SL이 OANDA 최소거리 미달")

    if signal_score < 3:  # 점수가 부족한 경우
        rejection_reasons.append("전략 점수 미달")


    # ... 다른 조건들도 여기에 추가

    # 이유가 하나라도 있으면 문자열로 합치고 row에 기록
    if rejection_reasons:
        row.append(" / ".join(rejection_reasons))
    else:
        row.append("")

    
    clean_row = []
    for v in row:
        if isinstance(v, (dict, list)):
            clean_row.append(json.dumps(v, ensure_ascii=False))  # ✅ dict, list를 JSON 문자열로
        elif isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            clean_row.append("")  # NaN, inf → 빈 문자열
        elif v is None:
            clean_row.append("")  # ✅ NoneType도 명시 처리
        else:
            clean_row.append(str(v))  # ✅ 문자열로 변환해서 누락 방지

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
    # ✅ (이 위치에 추가)
    signal_score = 0
    reasons = []
   
    if trend == 'UPTREND' and macd > 0 and rsi > 65:
        reasons.append("상승추세 + MACD 강세 → RSI SELL 무효화")
        rsi_sell_score = 0
    

    # RSI + Stoch RSI 콤보
    if 50 <= rsi.iloc[-1] <= 60 and stoch_rsi < 0.2 and signal == "BUY":
        signal_score += 2
        reasons.append("RSI 중립 + Stoch RSI 과매도 → 상승 기대")
    if 50 <= rsi.iloc[-1] <= 60 and stoch_rsi > 0.8 and signal == "SELL":
        signal_score += 2
        reasons.append("RSI 중립 + Stoch RSI 과열 → 하락 기대")

    # MACD 민감도 완화
    if abs(macd.iloc[-1] - macd_signal.iloc[-1]) > 0.0001:
        signal_score += 1
        reasons.append("MACD 교차 (민감도 완화 적용)")

    # 박스권 하단 반복 지지 가점
    box_info = detect_box_breakout(candles, pair)
    recent_lows = candles['low'].tail(15)
    support_count = sum(recent_lows <= box_info['support'] * 1.001)
    if support_count >= 3 and signal == "BUY":
        signal_score += 2
        reasons.append("박스권 하단 반복 지지 → 상승 강화")

    # 장대바디 캔들 심리
    last = candles.iloc[-1]
    body = abs(last['close'] - last['open'])
    total_range = last['high'] - last['low']
    if total_range > 0 and (body / total_range) > 0.6:
        if signal == "BUY" and last['close'] > last['open']:
            signal_score += 1
            reasons.append("장대 양봉 → 매수 심리")
        elif signal == "SELL" and last['close'] < last['open']:
            signal_score += 1
            reasons.append("장대 음봉 → 매도 심리")

    # 미국장 초반 유동성 가점
    now_utc = datetime.utcnow()
    if 16 <= now_utc.hour <= 18:
        signal_score += 1
        reasons.append("미국 개장 초반 유동성 증가")

    print("📝 FastFury 내부 점수:", signal_score, reasons)

    # ✅ GPT 호출 (TP/SL 없이 판단만 요청)
    payload = {
        "pair": pair, "price": price, "signal": signal,
        "rsi": rsi.iloc[-1], "macd": macd.iloc[-1], "macd_signal": macd_signal.iloc[-1],
        "stoch_rsi": stoch_rsi, "bollinger_upper": boll_up.iloc[-1], "bollinger_lower": boll_low.iloc[-1],
        "pattern": pattern, "trend": trend, "liquidity": liquidity
    }

    gpt_result = analyze_with_gpt(payload)

    
    # GPT 결과 파싱 (BUY/SELL/WAIT)
    if "BUY" in gpt_result and trend == "UPTREND":
        decision = "BUY"
    elif "SELL" in gpt_result and trend == "DOWNTREND":
        decision = "SELL"
    else:
        decision = "WAIT"

    if decision == "WAIT":
        return {"status": "WAIT", "message": "GPT 판단으로 관망"} 

    # 이제 GPT 최종 decision을 기준으로 진입
    tp = None
    sl = None

    pip_value = 0.01
    tp_pips = pip_value * 12
    sl_pips = pip_value * 6

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
        score=signal_score,
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
