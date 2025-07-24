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
def must_capture_opportunity(rsi, stoch_rsi, macd, macd_signal, pattern, candles, trend, atr, price, bollinger_upper, bollinger_lower, support, resistance, support_distance, resistance_distance, pip_size):
    opportunity_score = 0
    reasons = []

    if stoch_rsi < 0.05 and rsi > 50 and macd > macd_signal:
        opportunity_score += 2
        reasons.append("💡 Stoch RSI 극단 과매도 + RSI 50 상단 돌파 + MACD 상승 → 강력한 BUY 기회")

    if stoch_rsi > 0.95 and rsi < 50 and macd < macd_signal:
        opportunity_score += 2
        reasons.append("💡 Stoch RSI 극단 과매수 + RSI 50 이탈 + MACD 하락 → 강력한 SELL 기회")


    if 48 < rsi < 52:
        opportunity_score += 0.5
        reasons.append("💡 RSI 50 근접 – 심리 경계선 전환 주시")
    if 60 < rsi < 65:
        opportunity_score += 0.5
        reasons.append("🔴 RSI 60~65: 과매수 초기 피로감 (SELL 경계)")
    if rsi >= 70:
        opportunity_score -= 1
        reasons.append("❌ RSI 70 이상: 과매수로 진입 위험 높음 → 관망 권장")
    
    if 40 < rsi < 60 and stoch_rsi > 0.8:
        opportunity_score += 0.5
        reasons.append("⚙ RSI 중립 + Stoch 과열 → 가중 진입 조건")
    if stoch_rsi > 0.8 and rsi > 60:
        opportunity_score -= 2
        reasons.append("⚠️ Stoch RSI 과열 + RSI 상승 피로 → 진입 주의 필요")
        
    if 35 < rsi < 40:
        opportunity_score += 0.5
        reasons.append("🟢 RSI 35~40: 중립 돌파 초기 시도 (기대 영역)")
    if trend == "UPTREND":
        opportunity_score += 0.5
        reasons.append("🟢 상승추세 지속: 매수 기대감 강화")
    elif trend == "DOWNTREND":
        opportunity_score += 0.5
        reasons.append("🔴 하락추세 지속: 매도 기대감 강화")
    # ✅ 중립 추세일 때 추가 조건
    elif trend == "NEUTRAL":
        if (45 < rsi < 60) and (macd > macd_signal) and (0.2 < stoch_rsi < 0.8):
            opportunity_score += 0.5
            reasons.append("🟡 중립 추세 + 조건 충족 → 약한 기대감")
        else:
            opportunity_score -= 0.5
            reasons.append("⚠️ 중립 추세 + 신호 불충분 → 신뢰도 낮음 (감점)")

    
    if pattern in ["HAMMER", "SHOOTING_STAR"]:
        opportunity_score += 0.5
        reasons.append(f"🕯 {pattern} 캔들: 심리 반전 가능성")
    if atr < 0.0005:
        opportunity_score -= 0.5
        reasons.append("📉 ATR 낮음 → 변동성 부족, 시그널 신뢰도 약화")
        # 1. RSI와 추세가 충돌
    if trend == "DOWNTREND" and rsi > 50:
        opportunity_score -= 0.5
        reasons.append("⚠️ 하락 추세 중 RSI 매수 신호 → 조건 충돌 감점")

    # 2. MACD 약세인데 RSI/Stoch RSI가 강세면 경고
    if macd < macd_signal and (rsi > 50 or stoch_rsi > 0.6):
        opportunity_score -= 0.5
        reasons.append("⚠️ MACD 하락 중 RSI or Stoch RSI 매수 신호 → 조건 불일치 감점")


    if macd > macd_signal:
        opportunity_score += 0.5
    else:
        opportunity_score += 0.0  # 감점 없음

    
    # 3. 추세 중립 + MACD 약세 = 확신 부족
    if trend == "NEUTRAL" and macd < macd_signal:
        opportunity_score -= 0.0
        reasons.append("⚠️ 추세 중립 + MACD 하락 → 확신 부족한 시그널")

    # 4. ATR 극저 (강한 무변동장)
    if atr < 0.001:
        opportunity_score -= 0.5
        reasons.append("⚠️ ATR 매우 낮음 → 변동성 매우 부족한 장세")
    if abs(macd - macd_signal) < 0.0002:
        opportunity_score -= 0.2
        reasons.append("⚠️ MACD 신호 미약 → 방향성 부정확으로 감점")
    if 40 < rsi < 50:
        opportunity_score -= 0.2
        reasons.append("⚠️ RSI 중립구간 (40~50) → 방향성 모호, 진입 보류")
    if atr < 0.0012:
        opportunity_score -= 0.5
        reasons.append("⚠️ ATR 낮음 → 진입 후 변동 부족, 리스크 대비 비효율")
    
    return opportunity_score, reasons


    # 강한 반전 신호 (1점)
    strong_reversal_patterns = [
        "BULLISH_ENGULFING", "BEARISH_ENGULFING",
        "MORNING_STAR", "EVENING_STAR",
        "PIERCING_LINE", "DARK_CLOUD_COVER"
    ]

    # 보조 반전 신호 (0.5점)
    supportive_patterns = [
        "HAMMER", "INVERTED_HAMMER",
        "SHOOTING_STAR", "SPINNING_TOP",
        "DOJI"
    ]

    if pattern in strong_reversal_patterns:
        opportunity_score += 1
        reasons.append(f"🟢 강력한 반전 캔들 패턴: {pattern}")
    elif pattern in supportive_patterns:
        opportunity_score += 0.5
        reasons.append(f"🟢 보조 캔들 패턴: {pattern}")
    else:
        reasons.append("⚪ 주요 캔들 패턴 없음")

    return opportunity_score, reasons
    
def get_enhanced_support_resistance(candles, price, atr, window=20, min_touch_count=2):
    highs = candles["high"].tail(window).astype(float)
    lows = candles["low"].tail(window).astype(float)

    support_zone = lows[lows < price].round(2).value_counts()
    resistance_zone = highs[highs > price].round(2).value_counts()

    support_candidates = support_zone[support_zone >= min_touch_count]
    resistance_candidates = resistance_zone[resistance_zone >= min_touch_count]

    # Support
    if not support_candidates.empty:
        support_value = support_candidates.idxmax()
        support_rows = candles[candles["low"].round(2) == support_value]
        if not support_rows.empty:
            support_price = float(support_rows["low"].iloc[-1])
        else:
            support_price = float(lows.min())
    else:
        support_price = float(lows.min())

    # Resistance
    if not resistance_candidates.empty:
        resistance_value = resistance_candidates.index.min()
        resistance_rows = candles[candles["high"].round(2) == resistance_value]
        if not resistance_rows.empty:
            resistance_price = float(resistance_rows["high"].iloc[0])
        else:
            resistance_price = float(highs.max())
    else:
        resistance_price = float(highs.max())

    # Ensure all are floats
    price = float(price)
    min_distance = max(0.1, float(atr.iloc[-1]) * 1.5)

    if price - support_price < min_distance:
        support_price = price - min_distance
    if resistance_price - price < min_distance:
        resistance_price = price + min_distance

    return round(support_price, 5), round(resistance_price, 5)


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
        
    if pattern in ["DOJI", "MORNING_STAR", "EVENING_STAR"]:
        score += 0.4
        reasons.append(f"🕯 {pattern} 패턴 → 반전 가능성 강화로 가점 (+0.4)")


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



def score_signal_with_filters(rsi, macd, macd_signal, stoch_rsi, trend, signal, liquidity, pattern, pair, candles, atr, price, bollinger_upper, bollinger_lower, support, resistance, support_distance, resistance_distance, pip_size):
    signal_score = 0
    opportunity_score = 0  
    reasons = []

    score, base_reasons = must_capture_opportunity(rsi, stoch_rsi, macd, macd_signal, pattern, candles, trend, atr, price, bollinger_upper, bollinger_lower, support, resistance, support_distance, resistance_distance, pip_size)
    extra_score, extra_reasons = additional_opportunity_score(rsi, stoch_rsi, macd, macd_signal, pattern, trend)

    signal_score += score + extra_score
    reasons.extend(base_reasons + extra_reasons)
    # ✅ 캔들 패턴과 추세 강한 일치 시 보너스 점수 부여
    if signal == "BUY" and trend == "UPTREND" and pattern in ["BULLISH_ENGULFING", "HAMMER", "PIERCING_LINE"]:
        signal_score += 1
        opportunity_score += 0.5  # ✅ 패턴-추세 일치 시 추가 점수
        reasons.append("✅ 강한 상승추세 + 매수 캔들 패턴 일치 → 보너스 + 기회 점수 강화")

    elif signal == "SELL" and trend == "DOWNTREND" and pattern in ["BEARISH_ENGULFING", "SHOOTING_STAR", "DARK_CLOUD_COVER"]:
        signal_score += 1
        opportunity_score += 0.5  # ✅ 패턴-추세 일치 시 추가 점수
        reasons.append("✅ 강한 하락추세 + 매도 캔들 패턴 일치 → 보너스 + 기회 점수 강화")
    
    # ✅ 거래 제한 시간 필터 (애틀랜타 기준)
    now_utc = datetime.utcnow()
    now_atlanta = now_utc - timedelta(hours=4)
    # ✅ 전략 시간대: 오전 09~14시 또는 저녁 19~22시
    if not ((9 <= now_atlanta.hour <= 14) or (19 <= now_atlanta.hour <= 22)):
        reasons.append("🕒 전략 외 시간대 → 유동성 부족 / 성공률 저하로 관망")
        return 0, reasons
    
    # ✅ 저항선과 너무 가까운 거리에서의 BUY 진입 방지 (구조상 불리한 진입 회피)
    if signal == "BUY" and resistance_distance / pip_size < 6:
        reasons.append("⚠️ 저항선 10pip 이내 → 구조상 불리 → 관망")
        return 0, reasons

    # ✅ 지지선과 너무 가까운 거리에서의 SELL 진입 방지 (구조상 불리한 진입 회피)
    if signal == "SELL" and abs(price - support) / pip_size < 6:
        reasons.append("⚠️ 지지선 10pip 이내 → 구조상 불리 → 관망")
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

    # ✅ RSI, MACD, Stoch RSI 모두 중립 + Trend도 NEUTRAL → 횡보장 진입 방어
    if trend == "NEUTRAL":
        if 45 <= rsi <= 55 and -0.05 < macd < 0.05 and 0.3 < stoch_rsi < 0.7:
            reasons.append("📉 지표 중립 + 트렌드 NEUTRAL → 횡보장 진입 방지")
            return 0, reasons
  
    # ✅ BUY 과열 진입 방어 (SELL의 대칭 조건)
    if signal == "BUY" and rsi > 60:
        if macd < macd_signal and stoch_rsi > 0.85:
            reasons.append("🛑 과매수 BUY 방어: MACD 하락 전환 + Stoch RSI 과열 → 관망")
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

    if price >= bollinger_upper:
        signal_score -= 1
        reasons.append("🔴 가격이 볼린저밴드 상단 돌파 ➔ 과매수 경계")
    elif price <= bollinger_lower:
        signal_score += 1
        reasons.append("🟢 가격이 볼린저밴드 하단 터치 ➔ 반등 가능성↑")

    if pattern in ["LONG_BODY_BULL", "LONG_BODY_BEAR"]:
        signal_score += 2
        reasons.append(f"장대바디 캔들 추가 가점: {pattern}")

    box_info = detect_box_breakout(candles, pair)
    
    high_low_flags = analyze_highs_lows(candles)
    if high_low_flags["new_high"]:
        reasons.append("📈 최근 고점 갱신 → 상승세 유지 가능성↑")
    if high_low_flags["new_low"]:
        reasons.append("📉 최근 저점 갱신 → 하락세 지속 가능성↑")
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
            macd_diff = macd - macd_signal
            if abs(macd_diff) < 0.0001:
                reasons.append("⚠️ MACD 미세변동 → 신뢰도 낮음")
            elif macd_diff > 0 and macd > 0:
                reasons.append("🟢 MACD 양수 유지 → 상승 흐름 유지")
            elif macd_diff < 0 and macd < 0:
                reasons.append("🔴 MACD 음수 지속 → 약세 흐름 유지")
      
            
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
        macd_diff = macd - macd_signal
        if abs(macd_diff) < 0.0001:
            reasons.append("⚠️ MACD 미세변동 → 신뢰도 낮음")
        elif macd_diff > 0 and macd > 0:
            reasons.append("🟢 MACD 양수 유지 → 상승 흐름 유지")
        elif macd_diff < 0 and macd < 0:
            reasons.append("🔴 MACD 음수 지속 → 약세 흐름 유지")


    if stoch_rsi == 0.0:
        signal_score += 1
        reasons.append("🟢 Stoch RSI 0.0 → 극단적 과매도 → 반등 기대")
    elif stoch_rsi == 1.0:
        signal_score -= 1
        reasons.append("🔴 Stoch RSI 1.0 → 극단적 과매수 → 피로감 주의")
    
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
    if (
        all(last_3["close"] < last_3["open"]) 
        and trend == "DOWNTREND" 
        and pattern in ["NEUTRAL", "SHOOTING_STAR", "LONG_BODY_BEAR"]
    ):
        signal_score += 1
        reasons.append("🔻최근 3봉 연속 음봉 + 하락추세 + 약세형 패턴 포함 → SELL 강화")

    # 상승 연속 양봉 패턴 보정 BUY
    if (
        all(last_3["close"] > last_3["open"]) 
        and trend == "UPTREND" 
        and pattern in ["NEUTRAL", "LONG_BODY_BULL", "INVERTED_HAMMER"]
    ):
        signal_score += 1
        reasons.append("🟢 최근 3봉 연속 양봉 + 상승추세 + 약세 미발견 → BUY 강화")
    if pattern in ["BULLISH_ENGULFING", "HAMMER", "MORNING_STAR"]:
        signal_score += 2
        reasons.append(f"🟢 강한 매수형 패턴 ({pattern}) → 진입 근거 강화")
    elif pattern in ["LONG_BODY_BULL"]:
        signal_score += 1
        reasons.append(f"🟢 양봉 확장 캔들 ({pattern}) → 상승 흐름 가정")
    elif pattern in ["SHOOTING_STAR", "BEARISH_ENGULFING", "HANGING_MAN", "EVENING_STAR"]:
        signal_score -= 2
        reasons.append(f"🔴 반전형 패턴 ({pattern}) → 매도 고려 필요")
    # 교과서적 기회 포착 보조 점수
    op_score, op_reasons = must_capture_opportunity(rsi, stoch_rsi, macd, macd_signal, pattern, candles, trend, atr, price, bollinger_upper, bollinger_lower, support, resistance, support_distance, resistance_distance, pip_size)
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
    # 동적 지지/저항선 계산 (파동 기반)
    if candles is not None and not candles.empty:
        current_price = candles.iloc[-1]['close']
    else:
        current_price = None
    # ✅ ATR 먼저 계산
    atr = calculate_atr(candles)  # 또는 고정값으로 테스트: atr = 0.2

    # ✅ 지지/저항 계산
    support, resistance = get_enhanced_support_resistance(candles, price=current_price, atr=atr)
    support_resistance = {
        "support": support,
        "resistance": resistance
    }
    support_distance = abs(price - support)
    resistance_distance = abs(resistance - price)
    # ✅ 현재가와 저항선 거리 계산 (pip 기준 거리 필터 적용을 위함)
    pip_size = 0.01 if "JPY" in pair else 0.0001
    resistance_distance = abs(resistance - price)

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
    news = fetch_forex_news()
    support_resistance = {
        "support": candles["low"].min(),
        "resistance": candles["high"].max()
    }
    news_score, news_msg = news_risk_score(pair)
    high_low_analysis = analyze_highs_lows(candles)
    atr = calculate_atr(candles).iloc[-1]
    fibo_levels = calculate_fibonacci_levels(candles["high"].max(), candles["low"].min())
    # 📌 현재가 계산
    price = candles["close"].iloc[-1]
    signal_score, reasons = score_signal_with_filters(
        rsi.iloc[-1],
        macd.iloc[-1],
        macd_signal.iloc[-1],
        stoch_rsi,
        trend,
        signal,
        liquidity,
        pattern,
        pair,
        candles,
        atr,
        price,
        boll_up.iloc[-1], 
        boll_low.iloc[-1],
        support,
        resistance,
        support_distance,
        resistance_distance,
        pip_size
    )

    # 📦 Payload 구성
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
        "news": f"{news} | {news_msg}",
        "new_high": bool(high_low_analysis["new_high"]),
        "new_low": bool(high_low_analysis["new_low"]),
        "atr": atr,
        "signal_score": signal_score,
        "score_components": reasons
    }



    signal_score, reasons = score_signal_with_filters(
    rsi.iloc[-1], macd.iloc[-1], macd_signal.iloc[-1], stoch_rsi,
    trend, signal, liquidity, pattern, pair, candles, atr, price, boll_up.iloc[-1], boll_low.iloc[-1], support, resistance, support_distance, resistance_distance, pip_size
    )
    # 🎯 뉴스 리스크 점수 추가 반영
    signal_score += news_score
    reasons.append(f"📰 뉴스 리스크: {news_msg} (점수 {news_score})")
            
    recent_trade_time = get_last_trade_time()
    time_since_last = datetime.utcnow() - recent_trade_time if recent_trade_time else timedelta(hours=999)
    allow_conditional_trade = time_since_last > timedelta(hours=2)

    gpt_feedback = "GPT 분석 생략: 점수 미달"
    decision, tp, sl = "WAIT", None, None

    if signal_score >= 5:
        gpt_feedback = analyze_with_gpt(payload)
        print("✅ STEP 6: GPT 응답 수신 완료")
        decision, tp, sl = parse_gpt_feedback(gpt_feedback)
    else:
        print("🚫 GPT 분석 생략: 점수 5점 미만")
    
    
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
            pattern, trend, fibo_levels, decision, news, gpt_feedback,
            alert_name, tp, sl, price, None,
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

        gpt_feedback += "\n⚠️ TP/SL 추출 실패 → ATR 기반 기본값 적용"

    
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
        

    result = {}
    price_movements = []
    pnl = None
    if decision in ["BUY", "SELL"] and tp and sl:
        units = 100000 if decision == "BUY" else -100000
        digits = 3 if pair.endswith("JPY") else 5
        result = place_order(pair, units, tp, sl, digits)
        print("✅ STEP 9: 주문 결과 확인 |", result)

        executed_time = datetime.utcnow()
        candles_post = get_candles(pair, "M30", 8)
        price_movements = candles_post[["high", "low"]].to_dict("records")

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
        pattern, trend, fibo_levels, decision, news, gpt_feedback,
        alert_name, tp, sl, price, pnl, None,
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

import feedparser
import pytz

def fetch_news_events():
    url = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
    feed = feedparser.parse(url)
    events = []
    for entry in feed.entries:
        events.append({
            "title": entry.title,
            "summary": entry.summary,
            "published": entry.published,
        })
    return events

def filter_relevant_news(pair, within_minutes=90):
    currency = pair.split("_")[0] if pair.startswith("USD") else pair.split("_")[1]
    now_utc = datetime.utcnow().replace(tzinfo=pytz.UTC)
    events = fetch_news_events()
    relevant = []

    for e in events:
        if currency not in e["title"]:
            continue
        try:
            event_time = datetime.strptime(e["published"], "%a, %d %b %Y %H:%M:%S %Z").replace(tzinfo=pytz.UTC)
        except Exception:
            continue
        delta = abs((event_time - now_utc).total_seconds()) / 60
        if delta < within_minutes:
            relevant.append(e["title"])
    return relevant

def news_risk_score(pair):
    relevant = filter_relevant_news(pair)
    if any("High" in title for title in relevant):
        return -2, "⚠️ 고위험 뉴스 임박"
    elif any("Medium" in title for title in relevant):
        return -1, "⚠️ 중간위험 뉴스 임박"
    elif relevant:
        return 0, "🟢 뉴스 있음 (낮은 영향)"
    else:
        return 0, "🟢 영향 있는 뉴스 없음"

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


def parse_gpt_feedback(text):
    import re

    decision = "WAIT"
    tp = None
    sl = None

    # ✅ 명확한 판단 패턴 탐색 (정규식 우선)
    decision_patterns = [
        r"(결정|판단)\s*(판단|신호|방향)?\s*(은|:|：)?\s*[\"']?(BUY|SELL|WAIT)[\"']?",
        r"진입\s*방향\s*(은|:|：)?\s*['\"]?(BUY|SELL|WAIT)['\"]?",
        r"판단\s*(은|:|：)?\s*['\"]?(BUY|SELL|WAIT)['\"]?",
        r"진입판단\s*(은|:|：)?\s*['\"]?(BUY|SELL|WAIT)['\"]?",
    ]

    for pat in decision_patterns:
        d = re.search(pat, text.upper())
        if d:
            decision = d.group(4)
            break

    # ✅ fallback: "BUY" 또는 "SELL" 단독 등장 시 인식
    if decision == "WAIT":
        if "BUY" in text.upper() and "SELL" not in text.upper():
            decision = "BUY"
        elif "SELL" in text.upper() and "BUY" not in text.upper():
            decision = "SELL"

    # ✅ TP/SL 추출 (가장 마지막 숫자 사용)
    tp_line = next((line for line in text.splitlines() if "TP:" in line.upper() or "TP 제안 값" in line or "목표" in line), "")
    sl_line = next((line for line in text.splitlines() if "SL:" in line.upper() and re.search(r"\d+\.\d+", line)), "")
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

    return decision, tp, sl
    
def analyze_with_gpt(payload):
    headers = {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}", "Content-Type": "application/json"}
    messages = [
        {"role": "system", 
         "content": (
            "너는 실전 FX 트레이딩 전략 조력자야.\n"
            "(1) 아래 JSON 데이터를 기반으로 전략 리포트를 작성해. score_components 리스트는 각각의 전략 요소가 신호 판단에 어떤 기여를 했는지를 설명한 거야. "
            "모든 요소를 종합적으로 분석해서 진입 판단(BUY, SELL, WAIT)과 TP, SL 값을 제시해.너의 판단이 관망일때는 그냥 wait으로 판단해\n"
            

            "(2) 거래는 기본적으로 1~2시간 내 청산을 목표로 하고, SL과 TP는 ATR의 최소 50% 이상 거리로 설정해. "
            "최근 5개 캔들의 고점/저점도 참고해서 너가 설정한 TP/SL이 REASONABLE한지 꼭 검토해.\n"
            "TP와 SL은 현재가에서 각각 8pip 이상 차이나야 하고, TP는 SL보다 넓게 잡아. "
            "TP:SL 비율은 2:1 이상이면서 최소 10pip 이상 차이 나야 해. 비율은 TP가 2이고 SL이 1이다. BUY일 땐 TP > 진입가, SL < 진입가 / SELL일 땐 TP < 진입가, SL > 진입가를 반드시 지켜.\n\n"

            "(3) 지지선(support), 저항선(resistance)은 최근 1시간봉 기준 마지막 6봉의 고점/저점에서 이미 계산되어 JSON에 포함되어 있어. support와 resistance는 주어진 숫자만을 사용하며, 수치를 임의로 변경하지 마십시오. "
            "이 숫자만 참고하고 그 외 고점/저점은 무시해.\n\n"

            "(4) 추세 판단 시 캔들 패턴뿐 아니라 보조지표(RSI, MACD, Stoch RSI)의 흐름과 방향성도 함께 고려해.\n\n"

            "(5) 리포트 마지막에는 아래 형식으로 진입판단을 명확하게 작성해:\n"
            "진입판단: BUY (또는 SELL, WAIT)\n"
            "TP: 1.08752\n"
            "SL: 1.08214\n\n"

            "(6) TP와 SL은 반드시 **단일 수치값만 제시**하고, '약'이나 '~부근' 같은 표현은 절대 쓰지 마. 숫자만 있어야 거래 자동화가 가능해.\n\n"

            "(7) 현재가가 저항선에 가까우면 TP는 짧게, 지지선에서 멀다면 SL은 조금 여유롭게 허용해. 하지만 너무 과도하게 넓지 않게 조정해.\n\n"

            "(8) 피보나치 수렴 또는 확장 여부도 참고하고, 돌파 가능성이 높다면 TP를 약간 확장해도 돼. "
            "캔들패턴, 신고점/신저점 흐름, 박스권 유지 여부, ATR, 볼린저밴드 폭 등을 종합해서 TP/SL 변동폭을 보수적으로 또는 공격적으로 조정해.\n\n"

            "나의 최종 목표는 거래당 약 $150 수익을 내는 것이고, 손실은 거래당 $100을 넘지 않도록 설정하는 것이야."
         )
        },
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
