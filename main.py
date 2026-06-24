    # ⚠️ V2 업그레이드된 자동 트레이딩 스크립트 (학습 강화, 트렌드 보강, 시트 시간 보정 포함)
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from zoneinfo import ZoneInfo
import os
import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import openai
import numpy as np
import gspread
import threading
from concurrent.futures import ThreadPoolExecutor
import ta
import time as _t
import math
import base64
import os
import asyncio
from playwright.sync_api import sync_playwright
import time
import time as _t
print("🔥 CURRENT OPENAI KEY:", os.getenv("OPENAI_API_KEY"))
_gpt_lock = threading.Lock()
_gpt_last_ts = 0.0
_gpt_cooldown_until = 0.0
_gpt_rate_lock = threading.Lock()
_gpt_next_slot = 0.0
_last_execution_time = 0.0  # 마지막 실행 시간을 저장할 변수
# 🟦 OpenAI Tier 3 기준 gpt-4o 한도가 5,000 RPM이라 20은 너무 낮았음(슬롯 대기가 불필요한 지연의 큰 원인).
#    안전마진 두고 3000으로 상향. 필요시 환경변수로 재조정 가능.
GPT_RPM = int(os.getenv("GPT_RPM", "3000"))
_SLOT = 60.0 / GPT_RPM
from oauth2client.service_account import ServiceAccountCredentials

# ===== (NEW) 글로벌 레이트/토큰 상태 =====
_tpm_remaining = 1e9
_tpm_reset_ts  = 0.0
_rpm_remaining = 1e9
_rpm_reset_ts  = 0.0

def _approx_tokens(msgs: list[dict]) -> int:
    """메시지 리스트의 대략적 토큰 수 추정(문자수/4)"""
    import json
    s = json.dumps(msgs, ensure_ascii=False)
    return max(1, int(len(s) / 4))

def _preflight_gate(need_tokens: int):
    """요청 보내기 직전에 남은 토큰/RPM으로 선대기"""
    import time as _t, random
    global _tpm_remaining, _tpm_reset_ts, _rpm_remaining, _rpm_reset_ts
    now = _t.time()
    wait_until = now
    # TPM 부족 시 토큰 리셋까지 대기
    if (_tpm_remaining - need_tokens) < 0 and now < _tpm_reset_ts:
        wait_until = max(wait_until, _tpm_reset_ts)
    # RPM 0이면 요청 리셋까지 대기
    if (_rpm_remaining - 1) < 0 and now < _rpm_reset_ts:
        wait_until = max(wait_until, _rpm_reset_ts)
    if wait_until > now:
        _t.sleep((wait_until - now) + random.uniform(0.05, 0.2))
# 1. 트레이딩뷰 차트를 캡처하는 함수
def capture_tradingview_chart(pair):
    print(f"📸 {pair} 차트 캡처 프로세스 시작...")
    with sync_playwright() as p:
        try:
            # 브라우저 실행
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(viewport={'width': 1920, 'height': 1080})
            page = context.new_page()

            # ✅ 중요: 아래 URL을 사용자님의 실제 차트 레이아웃 주소로 바꾸세요!
            # 주소 끝에 ?symbol=FX:USDJPY 처럼 종목을 붙여주면 해당 종목 차트가 열립니다.
            # 🟦 주식은 FX: 대신 거래소 prefix(NASDAQ/NYSE 등)가 필요. 정확한 거래소를 모를 땐
            #     TradingView가 prefix 없이도 종목명만으로 자동 매칭해주는 경우가 많아 우선 prefix 없이 시도.
            if is_stock_pair(pair):
                target_url = f"https://www.tradingview.com/chart/iHBYFrNs/?symbol={pair.replace('/', '')}"
            else:
                target_url = f"https://www.tradingview.com/chart/iHBYFrNs/?symbol=FX:{pair.replace('/', '')}"
            
            page.goto(target_url, wait_until="networkidle")
            print("⏳ 지표와 신호가 차트에 나타날 때까지 10초 대기...")
            _t.sleep(10) # 지표가 많을수록 로딩 시간이 필요하므로 넉넉히 줍니다.

            # 파일명 설정 및 저장
            filename = f"chart_{pair.replace('/', '_')}.png"
            page.screenshot(path=filename)
            browser.close()
            
            return filename
        except Exception as e:
            print(f"❌ 캡처 실패: {e}")
            return None

def encode_image(image_path):
    """이미지를 GPT가 읽을 수 있는 문자열로 변환"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def _save_rate_headers(h: dict) -> None:
    """
    OpenAI 응답 헤더에서 남은 요청/토큰 수와 리셋까지 남은 초를 읽어
    전역 상태(_rpm_remaining/_tpm_remaining/_rpm_reset_ts/_tpm_reset_ts)에 반영한다.
    키 대소문자/변형에 관대하게 처리.
    """
    import time as _t
    global _tpm_remaining, _tpm_reset_ts, _rpm_remaining, _rpm_reset_ts

    if not h:
        return

    # 헤더 키를 관대하게 조회 (소문자/TitleCase 모두 허용)
    def _hget(*keys):
        for k in keys:
            v = h.get(k)
            if v is None:  # requests가 소문자로 줄 수도 있음
                v = h.get(k.lower())
            if v is None:  # 일부 프록시는 TitleCase로 줄 수도 있음
                v = h.get(k.title())
            if v is not None:
                return v
        return None

    now = _t.time()

    try:
        # 남은 개수(요청/토큰)
        rem_req = _hget("x-ratelimit-remaining-requests", "X-RateLimit-Remaining-Requests")
        rem_tok = _hget("x-ratelimit-remaining-tokens",   "X-RateLimit-Remaining-Tokens")
        if rem_req is not None:
            _rpm_remaining = float(rem_req)
        if rem_tok is not None:
            _tpm_remaining = float(rem_tok)

        # 리셋까지 남은 초(요청/토큰)
        rst_req = _hget("x-ratelimit-reset-requests", "X-RateLimit-Reset-Requests")
        rst_tok = _hget("x-ratelimit-reset-tokens",   "X-RateLimit-Reset-Tokens")
        if rst_req is not None:
            _rpm_reset_ts = now + float(rst_req)
        if rst_tok is not None:
            _tpm_reset_ts = now + float(rst_tok)

    except Exception:
        # 형식이 이상해도 전체 흐름 멈추지 않음
        pass
        
# === OpenAI 공통 설정 & 세션 ===
OPENAI_URL = "https://api.openai.com/v1/responses"
OPENAI_HEADERS = {
    "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
    "Content-Type": "application/json",
}
_openai_sess = requests.Session()  # keep-alive로 커넥션 재사용 (429 억제에 도움)

# === 간단 디버그 (알림 한 건 추적용) ===
import uuid, time as _t, random
def dbg(tag, **k):
    try:
        pairs = " ".join(f"{a}={b}" for a, b in k.items())
    except Exception:
        pairs = str(k)
    print(f"[DBG] {tag} {pairs}")
    
def gpt_rate_gate():
    """계정 단위 요청 슬롯(=RPM) 대기"""
    global _gpt_next_slot, _gpt_rate_lock, _SLOT
    with _gpt_rate_lock:
        now = _t.time()                 # ← time.time() 말고 _t.time()
        if _gpt_next_slot < now:
            _gpt_next_slot = now
        slot = _gpt_next_slot
        _gpt_next_slot += _SLOT         # 다음 슬롯 예약

    wait = slot - now
    if wait > 0:
        _t.sleep(wait) 
def recent_high_break(highs, last_n=2):
    if not highs or last_n <= 0:
        return False
    if len(highs) < last_n + 1:
        return False
    prev_high = max(highs[:-last_n])
    recent_high = max(highs[-last_n:])
    return recent_high > prev_high

    
def recent_low_break(lows, last_n=2):
    if not lows or last_n <= 0:
        return False
    if len(lows) < last_n + 1:
        return False
    prev_low = min(lows[:-last_n])
    recent_low = min(lows[-last_n:])
    return recent_low < prev_low

def must_capture_opportunity(rsi, stoch_rsi, macd, macd_signal, pattern, candles, trend, atr, price, bollinger_upper, bollinger_lower, support, resistance, support_distance, resistance_distance, pip_size, expected_direction=None):
    opportunity_score = 0
    reasons = []

    
    is_buy = expected_direction == "BUY"
    is_sell = expected_direction == "SELL"

    # 🟦 버그 수정: breakout_confirmed가 정의된 적 없이 사용되고 있었음(stoch_rsi>0.9일 때 NameError로 크래시).
    #    "저항을 이미 뚫고 올라간 상태인가"를 의미하므로, price>resistance로 정의.
    breakout_confirmed = (price is not None and resistance is not None and price > resistance)

    # === macd_signal fallback ===
    if macd_signal is None:
        macd_signal = macd
        reasons.append("⚠️ macd_signal 없음 → macd 자체 사용")

    # ==================================================
    # 1️⃣ 강한 기회 포착 (기존 로직 유지)
    # ==================================================
    if stoch_rsi < 0.05 and rsi > 50 and macd > macd_signal and is_buy:
        opportunity_score += 2
        reasons.append("💡 Stoch RSI 극단 과매도 + RSI 상단 + MACD 상승 → 강한 BUY (+2)")

    if stoch_rsi < 0.1 and rsi < 40 and macd < 0 and is_sell:
        opportunity_score += 0.5
        reasons.append("⚠️ 약한 SELL 조건 충족 (+0.5)")

    if stoch_rsi > 0.9:
    
        if (
            resistance_distance < atr * 0.3
            and not breakout_confirmed
        ):
    
            opportunity_score -= 2
            reasons.append(
                "🔴 과열 + 저항 근접 + breakout 실패 위험"
            )
    
        elif trend == "UPTREND" and macd > macd_signal:
    
            opportunity_score -= 0.3
            reasons.append(
                "⚠️ 과열이지만 continuation 유지"
            )


    # ==============================
    # 🔥 2순위 방어: 극단 영역 + 패턴 없음 (칼날/천장 방어)
    # 위치: '강한 기회 포착' 끝나고, '추세 필터' 시작 바로 위
    # ==============================
    
    # BUY 방어: 극단 과매도 + 반등 패턴 없음 (+ MACD 약화) = 하락 가속(칼날) 위험
    if is_buy and stoch_rsi < 0.1:
        if (pattern is None or pattern == "NEUTRAL") and macd < macd_signal:
            opportunity_score -= 2.0
            reasons.append("🔴 (방어) Stoch RSI 극단 과매도(<0.1) + 반등 패턴 없음 + MACD 약화 → 하락 가속 위험 (opportunity -2)")
        elif (pattern is None or pattern == "NEUTRAL"):
            opportunity_score -= 1.0
            reasons.append("⚠️ (방어) Stoch RSI 극단 과매도(<0.1) + 반등 패턴 없음 → 반등 신뢰도 낮음 (opportunity -1)")
    
    # SELL 방어(미러): 극단 과매수 + 반전 패턴 없음 (+ MACD 강세 유지) = 상승 추세 속 역방향 SELL 말림 위험
    if is_sell and stoch_rsi > 0.9:
        # 여기서는 BUY와 반대로, MACD가 "여전히 강세"일 때가 더 위험
        if (pattern is None or pattern == "NEUTRAL") and macd > macd_signal:
            opportunity_score -= 2.0
            reasons.append("🔴 (방어) Stoch RSI 극단 과매수(>0.9) + 반전 패턴 없음 + MACD 강세 → 상승 지속 위험(SELL 말림) (opportunity -2)")
        elif (pattern is None or pattern == "NEUTRAL"):
            opportunity_score -= 1.0
            reasons.append("⚠️ (방어) Stoch RSI 극단 과매수(>0.9) + 반전 패턴 없음 → 반전 신뢰도 낮음 (opportunity -1)")
    # ==================================================
    # 4️⃣ 추세 필터 (가장 중요)
    # ==================================================
    highs = list(candles["high"].tail(20).astype(float).values)
    lows  = list(candles["low"].tail(20).astype(float).values)
    if is_buy and trend == "DOWNTREND":
    
        opportunity_score -= 1.5
    
        reasons.append(
            "🟠 하락 추세 + BUY 역방향 → continuation 신뢰도 낮음 (-1.5)"
        )

    if is_sell and trend == "UPTREND":
    
        opportunity_score -= 1.5
    
        reasons.append(
            "🟠 상승 추세 + SELL 역방향 → continuation 신뢰도 낮음 (-1.5)"
        )

    # BUY mirror penalty: overbought + no higher-high recently
    if is_buy and trend == "UPTREND":
        if rsi > 65:
            if not recent_high_break(highs, last_n=2):
                opportunity_score -= 0.5
                reasons.append(
                    "⚠️ 과매수 이후 고점 갱신 실패 → 되밀림 위험 BUY 감점 (-0.5)"
                )
    
    # SELL mirror penalty: oversold + no lower-low recently
    if is_sell and trend == "DOWNTREND":
        if rsi < 35:
            if not recent_low_break(lows, last_n=2):
                opportunity_score -= 0.5
                reasons.append(
                    "⚠️ 과매도 이후 저점 갱신 실패 → 반등 위험 SELL 감점 (-0.5)"
                )
    # ==================================================
    # 6️⃣ 캔들 패턴
    # ==================================================
    if is_buy and pattern in ["HAMMER", "BULLISH_ENGULFING", "PIERCING_LINE"]:
        opportunity_score += 0.5
        reasons.append(f"🕯 BUY 패턴 {pattern} (0.5)")

    if is_sell and pattern in ["SHOOTING_STAR", "BEARISH_ENGULFING", "DARK_CLOUD_COVER"]:
        opportunity_score += 0.5
        reasons.append(f"🕯 SELL 패턴 {pattern} (0.5)")

    # ==================================================
    # 7️⃣ ATR 필터
    # ==================================================
    if atr is not None and atr < 0.001:
        opportunity_score -= 0.5
        reasons.append("⚠️ ATR 매우 낮음 → 변동성 부족 (-0.5)")

    # ==================================================
    # 8️⃣ 최종 방향 충돌 필터 (조기 차단)
    # ==================================================
    if is_buy and opportunity_score < 0:
        opportunity_score -= 1.5
        reasons.append("⚠️ BUY 기대 방향 대비 opportunity_score 역행 → 신호 약화 (-1.5)")
    
    if is_sell and opportunity_score < 0:
        opportunity_score -= 1.5
        reasons.append("⚠️ SELL 기대 방향 대비 opportunity_score 역행 → 신호 약화 (-1.5)")

    return opportunity_score, reasons
    
def get_enhanced_support_resistance(candles, price, atr, timeframe, pair, window=20, min_touch_count=2):
    # 단타(3h/10pip) 최적화된 창 길이
    window_map = {'M5': 72, 'M15': 32, 'M30': 48, 'H1': 48, 'H4': 60}
    window = max(window_map.get(timeframe, window), 32)  # 최소 32봉 보장
    
    if price is None:
        return None, None
    highs = candles["high"].tail(window).astype(float)
    lows = candles["low"].tail(window).astype(float)
    df = candles.tail(window).copy()

    pip = pip_value_for(pair)
    round_digits = int(abs(np.log10(pip)))

    last_atr = float(atr.iloc[-1]) if hasattr(atr, "iloc") else float(atr)

    # 🟦 클러스터링(군집화) 임계치용 pip. 주식은 가격비례(pip) 대신 ATR 비례로 보정.
    #    예: TSLA 300달러 → pip=0.03 → threshold=6*0.03=0.18달러(ATR 10~15달러짜리 종목엔 의미 없음)
    #    → ATR*0.1을 6pip 폭으로 환산한 값과 비교해 더 큰 쪽을 사용
    cluster_pip = pip
    if is_stock_pair(pair) and last_atr:
        cluster_pip = max(pip, (last_atr * 0.1) / 6.0)
    
    # --- 동적 order: 창의 1/10 수준, 2~3로 클램프(반응성 확보) ---
    order = max(2, min(3, window // 10))
    if window < (2 * order + 1):  # 이론적 안전 장치
        order = max(2, (window - 1) // 2)
    
    # 초기화 (UnboundLocalError 방지)
    support_rows = pd.DataFrame(columns=candles.columns)
    resistance_rows = pd.DataFrame(columns=candles.columns)


    # 기본값
    price = float(price)
    price_rounded = round(price, round_digits)

    # 🔍 스윙 고점/저점 기반 지지선/저항선 추출
    def find_local_extrema(candles, order=3):
        highs = candles["high"].values
        lows = candles["low"].values
        resistance = []
        support = []

        for i in range(order, len(highs) - order):
            if highs[i] == max(highs[i - order:i + order + 1]):
                resistance.append(highs[i])
            if lows[i] == min(lows[i - order:i + order + 1]):
                support.append(lows[i])
        return support, resistance

    # 🎯 가까운 레벨 병합 (군집화)
    def cluster_levels(levels, *, pip: float, threshold_pips: int = 6, min_touch_count: int = 2):
        """
        인접 레벨 병합(클러스터) + 최소 터치 수 필터
        - threshold_pips: 단타는 6~8pip 권장(기본 6)
        - 통화쌍/가격 스케일에 무관하게 동작
        """
        if not levels:
            return []

        threshold = threshold_pips * pip
        buckets = []  # [{ "val": float, "cnt": int }]

        for lv in sorted(levels):
            if not buckets or abs(buckets[-1]["val"] - lv) > threshold:
                # 새 클러스터 시작
                buckets.append({"val": lv, "cnt": 1})
            else:
                # 가까우면 평균으로 병합 + 터치 수 증가
                buckets[-1]["val"] = (buckets[-1]["val"] + lv) / 2.0
                buckets[-1]["cnt"] += 1

        # 최소 터치 수 필터 적용
        return [b["val"] for b in buckets if b["cnt"] >= min_touch_count]
   

    # 📌 스윙 지지/저항 구하기
    support_levels, resistance_levels = find_local_extrema(df, order=order)
    support_levels    = cluster_levels(support_levels,    pip=cluster_pip, threshold_pips=6, min_touch_count=min_touch_count)
    resistance_levels = cluster_levels(resistance_levels, pip=cluster_pip, threshold_pips=6, min_touch_count=min_touch_count)
    
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # [A] 후보 부족 시 창을 2배로 확장해 1회 재시도 (단타용)
    if (not support_levels) or (not resistance_levels):
        df2 = candles.tail(window * 2).copy()
        order2 = max(2, min(3, (window * 2) // 10))
        if (window * 2) >= (2 * order2 + 1):
            s2, r2 = find_local_extrema(df2, order=order2)
            s2 = cluster_levels(s2, pip=cluster_pip, threshold_pips=6, min_touch_count=min_touch_count)
            r2 = cluster_levels(r2, pip=cluster_pip, threshold_pips=6, min_touch_count=min_touch_count)
            if s2: support_levels = s2
            if r2: resistance_levels = r2
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    min_distance = max(6 * pip, 0.8 * last_atr)  # 기존 10*pip, 1.2*ATR → 6*pip, 0.8*ATR


    
    # 🔽 현재가 아래 지지선 중 가장 가까운 것
    support_price = max([s for s in support_levels if s < price], default=price - min_distance)
    # 🔼 현재가 위 저항선 중 가장 가까운 것
    resistance_price = min([r for r in resistance_levels if r > price], default=price + min_distance)

    return round(support_price, round_digits), round(resistance_price, round_digits)


def additional_opportunity_score(rsi, stoch_rsi, macd, macd_signal, pattern, trend, signal):
    """ 기존 필터 이후, 추가 가중치 기반 보완 점수 """
    score = 0
    reasons = []
    is_buy = signal == "BUY"
    is_sell = signal == "SELL"

    if macd_signal is None:
        macd_signal = macd
        reasons.append("⚠️ macd_signal 없음 → macd 사용")

    # BUY 측 (미러링)
    if is_buy and (macd > 0) and (macd < macd_signal):
        if (stoch_rsi >= 0.80) or (rsi >= 65):
            score -= 0.5
            reasons.append("⚠️ BUY 중 MACD 약화 + 과열 구간 → 되돌림 위험 (감점 -0.5)")
    
    # SELL 측 (미러링)
    if is_sell and (macd < 0) and (macd > macd_signal):
        if (stoch_rsi <= 0.25) or (rsi <= 45):
            score -= 0.5
            reasons.append("⚠️ SELL 중 MACD 반등 + 과매도 구간 → 되돌림 위험 (감점 -0.5)")

        # ✅ NEUTRAL 구간 하락 재개(continuation) SELL 가점
    # - trend는 NEUTRAL이라도 "되돌림 후 재하락"이면 숏 기회로 봄
    # - 조건: MACD 약세 유지 + RSI 되돌림(50+) + Stoch 중립~상단(되돌림 완료 구간)
    if is_sell and (trend == "NEUTRAL"):
        if (macd < 0) and (macd < macd_signal) and (rsi >= 50) and (stoch_rsi >= 0.55):
            score += 1.0
            reasons.append("✅ NEUTRAL이지만 되돌림 후 하락 재개(continuation) → SELL 가점 +1.0")

        # ✅ NEUTRAL continuation BUY 가점 (미러링)
    # - trend는 NEUTRAL이어도 "되돌림 후 상승 재개"면 롱 기회로 봄
    if is_buy and (trend == "NEUTRAL"):
        if (macd > 0) and (macd > macd_signal) and (rsi <= 50) and (stoch_rsi <= 0.45):
            score += 1.0
            reasons.append("✅ NEUTRAL이지만 되돌림 후 상승 재개(continuation) → BUY 가점 +1.0")

    return score, reasons


# === pip/거리 헬퍼 ===
def pip_value_for(pair: str) -> float:
    """
    통화쌍/종목별 '1 pip 등가' 가격 크기 반환.
    - 주식: 현재가의 0.01%를 1pip 등가로 취급 (가격대가 천차만별이므로 비율 기반)
            → 캐시된 최근가가 없으면 1센트(0.01)로 폴백
    - JPY 쿼트: 0.01
    - 그 외 FX: 0.0001
    """
    if is_stock_pair(pair):
        last_price = _last_price_cache.get(pair)
        if last_price:
            return max(0.01, round(last_price * 0.0001, 6))
        return 0.01

    p = (pair or "").upper().replace("_", "/")
    # EUR/USD, GBP/USD, ...
    if p.endswith("/JPY") or p.endswith("JPY"):
        return 0.01
    return 0.0001
    
# ★ 추가: ATR을 pips로 변환
def atr_in_pips(atr_value: float, pair: str) -> float:
    pv = pip_value_for(pair)
    try:
        return float(atr_value) / pv if atr_value is not None else 0.0
    except:
        return 0.0

# ★ 추가: 통합 임계치(모든 페어 공통)
def dynamic_thresholds(pair: str, atr_value: float):
    pv = pip_value_for(pair)

    # 🟦 주식 전용 분기 (FX 로직은 아래 그대로 유지/미변경)
    #    FX는 'ATR-in-pips'가 보통 10~50 범위라 near_pips<=14, box_threshold_pips<=30 같은
    #    캡(상한)이 의미 있었지만, 주식은 pip_value_for()가 가격비례(price*0.0001)라
    #    ATR/price 비율이 큰 종목(TSLA 등)에서는 'ATR-in-pips'가 수백대로 나와 캡에 눌려버림.
    #    예: TSLA ATR=10, pv=0.03 → ATR-in-pips=333 → near_pips가 캡(14)에 눌려 14*pv=0.42달러
    #        (ATR 10달러짜리 종목에 0.42달러 임계치는 무의미)
    #    → 주식은 캡을 풀고 ATR 비율 그대로 사용(최소 하한만 유지)
    if is_stock_pair(pair):
        ap_stock = atr_in_pips(atr_value, pair)  # = ATR/price*10000 (가격 스케일 무관 변동성 비율)
        near_pips_stock          = max(8.0,  0.35 * ap_stock)
        box_threshold_pips_stock = max(12.0, 0.80 * ap_stock)
        breakout_buf_pips_stock  = max(1.0,  0.10 * ap_stock)
        macd_strong_stock = 20 * pv  # 참고용 값. 실제 MACD 채점은 score_signal_with_filters의 ATR 기반 strong/weak를 사용.
        macd_weak_stock   = 10 * pv

        return {
            "near_pips": near_pips_stock,
            "box_threshold_pips": box_threshold_pips_stock,
            "breakout_buf_pips": breakout_buf_pips_stock,
            "macd_strong": macd_strong_stock,
            "macd_weak": macd_weak_stock,
            "pip_value": pv,
        }

    # ===== 기존 FX 로직 (변경 없음) =====
    ap = max(6.0, atr_in_pips(atr_value, pair))     # ATR(pips), 최소 8pip

    # 🔧 변경: EUR/USD, GBP/USD는 근접 금지 하한 6 pip, 나머지는 8 pip
    min_near = 6 if pair in ("EUR_USD", "GBP_USD") else 8

    near_pips          = int(max(min_near, min(14, 0.35 * ap)))  # 지지/저항 근접 금지
    box_threshold_pips = int(max(12,     min(30, 0.80 * ap)))    # 박스 폭 임계
    breakout_buf_pips  = int(max(1,      min(3,  0.10 * ap))) 

    # MACD 교차 임계: pip 기준(강=20pip, 약=10pip)
    macd_strong = 20 * pv
    macd_weak   = 10 * pv

    return {
        "near_pips": near_pips,
        "box_threshold_pips": box_threshold_pips,
        "breakout_buf_pips": breakout_buf_pips,
        "macd_strong": macd_strong,
        "macd_weak": macd_weak,
        "pip_value": pv
    }




def pips_between(a: float, b: float, pair: str) -> float:
    return abs(a - b) / pip_value_for(pair)
    
def calculate_realistic_tp_sl(price, atr, pip_value, risk_reward_ratio=1, min_pips=8):
    """
    현실적인 TP/SL 계산 함수
    """
    atr_pips = max(min_pips, atr / pip_value * 0.5)  # ATR 절반 이상
    sl_price = price - (atr_pips * pip_value)
    tp_price = price + (atr_pips * pip_value * risk_reward_ratio)
    return round(tp_price, 5), round(sl_price, 5), atr_pips

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
    
def check_recent_opposite_signal(pair, current_signal, within_minutes=30, *,
                                 strategy=None, timeframe=None, score=None):
    """
    최근 within_minutes 안에 같은 pair(+옵션: strategy/timeframe)에서
    '반대 방향' 신호가 있었으면 True(관망), 아니면 False.
    항상 '현재 신호'를 기록하고 종료한다. (연속 관망 방지)
    """
    os.makedirs("/tmp", exist_ok=True)
    # 키를 넓히려면 전략/타프 포함
    key = f"{pair}:{strategy or 'ANY'}:{timeframe or 'ANY'}".replace(":", "_")
    log_path = f"/tmp/{key}_last_signal.json"
    now = datetime.utcnow()

    last_signal = None
    last_time = None

    # 1) 읽기
    if os.path.exists(log_path):
        try:
            with open(log_path, "r") as f:
                rec = json.load(f)
                last_signal = rec.get("signal")
                ts = rec.get("ts")
                if ts:
                    last_time = datetime.fromisoformat(ts)
        except Exception as e:
            print("[oppo-filter] read fail:", e)

    # 2) 충돌 판정
    conflict = False
    if last_time and (now - last_time) < timedelta(minutes=within_minutes):
        if last_signal and last_signal != current_signal:
            conflict = True

    # 3) 항상 현재 신호 기록 (연속 관망 방지의 핵심)
    try:
        with open(log_path, "w") as f:
            json.dump({
                "ts": now.isoformat(),
                "pair": pair,
                "signal": current_signal,
                "strategy": strategy,
                "timeframe": timeframe,
                "score": score
            }, f)
    except Exception as e:
        print("[oppo-filter] write fail:", e)

    return conflict



def calculate_structured_sl_tp(entry_price, direction, symbol, support, resistance, pip_size, atr=None):
    buffer = get_buffer_by_symbol(symbol, atr=atr)
    
    if direction == 'BUY':
        sl = support - buffer
        tp = entry_price + abs(entry_price - sl) * 1.8
    else:
        sl = resistance + buffer
        tp = entry_price - abs(entry_price - sl) * 1.8

    r_ratio = abs(tp - entry_price) / abs(sl - entry_price)
    
    # ✅ 로그 출력
    print(f"[SL/TP 계산 로그] symbol={symbol}, direction={direction}")
    print(f" - entry_price: {entry_price}")
    print(f" - support: {support}, resistance: {resistance}, buffer: {buffer}")
    print(f" - SL: {sl}, TP: {tp}, 손익비(r_ratio): {r_ratio:.2f}")
    return sl, tp, r_ratio

def get_buffer_by_symbol(symbol, atr=None):
    # 🟦 주식: 가격비례 pip(10*pip_value_for)는 ATR 대비 너무 작아짐
    #    (예: TSLA 10*0.03=0.30달러인데 ATR이 10달러면 노이즈에 바로 SL 털림)
    #    → ATR이 있으면 ATR*0.15를 버퍼로 사용, 없으면 기존 방식으로 폴백
    if is_stock_pair(symbol):
        try:
            atr_val = float(atr.iloc[-1]) if hasattr(atr, "iloc") else float(atr or 0)
        except Exception:
            atr_val = 0.0
        if atr_val > 0:
            return atr_val * ALPACA_SL_BUFFER_ATR_MULT
        return 10 * pip_value_for(symbol)

    # ===== 기존 FX 로직 (변경 없음, pip_value_for로 통합되어 있던 부분) =====
    return 10 * pip_value_for(symbol)

def get_multi_timeframe_context(pair):
    try:
        # 🟦 4h/5m 캔들 조회를 병렬로 실행 (순차 대기 시간 단축)
        with ThreadPoolExecutor(max_workers=2) as ex:
            fut_h4 = ex.submit(get_ohlcv, pair, interval='4h', limit=50)
            fut_m5 = ex.submit(get_ohlcv, pair, interval='5m', limit=30)
            df_h4 = fut_h4.result()
            df_m5 = fut_m5.result()

        h4_last = df_h4['close'].iloc[-1]

        h4_ema = ta.trend.ema_indicator(
            df_h4['close'],
            window=20
        ).iloc[-1]

        if pd.isna(h4_ema):
            h4_trend = "데이터부족"
        elif h4_last > h4_ema:
            h4_trend = "상승세(Bullish)"
        else:
            h4_trend = "하락세(Bearish)"

        m5_rsi = ta.momentum.rsi(
            df_m5['close'],
            window=14
        ).iloc[-1]

        if pd.isna(m5_rsi):
            print("[WARN] M5 RSI = NaN")
            m5_rsi_text = "N/A"
            m5_state = "데이터부족"
        else:
            m5_rsi_text = f"{m5_rsi:.2f}"

            if m5_rsi >= 70:
                m5_state = "과매수"
            elif m5_rsi <= 30:
                m5_state = "과매도"
            else:
                m5_state = "중립"

        return (
            f"[H4 추세]: {h4_trend} (EMA20 대비)\n"
            f"[M5 RSI]: {m5_rsi_text} ({m5_state})"
        )

    except Exception as e:
        print(f"[ERROR] get_multi_timeframe_context: {e}")
        return f"타임프레임 데이터 요약 실패: {e}"

def score_signal_with_filters(rsi, macd, macd_signal, stoch_rsi, prev_stoch_rsi, trend, prev_trend, signal, liquidity, pattern, pair, candles, atr, price, bollinger_upper, bollinger_lower, support, resistance, support_distance, resistance_distance, pip_size, macd_trend=None, expected_direction=None, strategy_name=None):
    signal_score = 0
    opportunity_score = 0  
    reasons = []

    
    score, base_reasons = must_capture_opportunity(rsi, stoch_rsi, macd, macd_signal, pattern, candles, trend, atr, price, bollinger_upper, bollinger_lower, support, resistance, support_distance, resistance_distance, pip_size, expected_direction=signal)
    extra_score, extra_reasons = additional_opportunity_score(rsi, stoch_rsi, macd, macd_signal, pattern, trend, signal)

    # ★ 통합 임계치 준비 (pip/ATR 기반)
    thr = dynamic_thresholds(pair, atr)
    pv = thr["pip_value"]           # pip 크기 (JPY=0.01, 그 외=0.0001)
    NEAR_PIPS = thr["near_pips"]    # 지지/저항 근접 금지 임계(pips)
    close = None
    try:
        if candles is not None and not candles.empty and "close" in candles.columns:
            close = float(candles["close"].iloc[-1])
    except Exception:
        close = None
    
    # price가 없으면 close로 대체, close가 없으면 price로 대체
    if price is None:
        price = close
    if close is None:
        close = price
    
    is_buy = expected_direction == "BUY"
    is_sell = expected_direction == "SELL"

    # RSI 중립 구간 (45~55) + 추세 중립 → 공통 감점
    if 45 <= rsi <= 55 and trend == "NEUTRAL":
        score -= 0.3
        reasons.append("⚠️ RSI 중립(45~55) + 트렌드 NEUTRAL → 진입 신호 약화 (-0.3)")
    
    # =========================
    # BUY 전용 감점 로직
    # =========================
    if is_buy:
        if (
            rsi > 40
            and stoch_rsi > 0.4
            and macd < macd_signal
            and trend != "UPTREND"
        ):
            score -= 1.0
            reasons.append(
                "📉 RSI & Stoch RSI 반등 중이나 MACD 약세 + 추세 불확실 → BUY 감점 (-1.0)"
            )
    
    # =========================
    # SELL 전용 감점 로직
    # =========================
    elif is_sell:
        if (
            rsi < 60
            and stoch_rsi < 0.6
            and macd > macd_signal
            and trend != "DOWNTREND"
        ):
            score -= 1.0
            reasons.append(
                "📈 RSI & Stoch RSI 하락 중이나 MACD 강세 + 추세 불확실 → SELL 감점 (-1.0)"
            )
    
    # === SL/TP 계산 및 손익비 조건 필터 ===
    entry_price = price
    direction = signal
    symbol = pair

    sl, tp, r_ratio = calculate_structured_sl_tp(entry_price, direction, symbol, support, resistance, pv, atr=atr)

    if r_ratio < 1.4:
        signal_score -= 4.0
        reasons.append("📉 손익비 낮음 (%.2f) → -4.0점 감점" % r_ratio)

    from datetime import datetime
    from zoneinfo import ZoneInfo
    
    now_atlanta = datetime.now(ZoneInfo("America/New_York"))
    atlanta_hour = now_atlanta.hour
    
    if 19 <= atlanta_hour < 23:
        signal_score -= 3
        reasons.append("🌙 애틀랜타 19~23시 거래 감점 (-3)")
        
    # ====================================
    if macd < -0.02 and trend != "DOWNTREND":
        score -= 1.5
        reasons.append("🔻 MACD 약세 + 추세 모호 → 신호 신뢰도 낮음 (감점 -1.5)")

    # RSI + Stoch RSI 과매수 상태에서 SELL 진입 위험
    if signal == "SELL" and rsi > 70 and stoch_rsi > 0.85:
        score -= 1.5
        reasons.append("🔻 RSI + Stoch RSI 과매수 → SELL 진입 위험 (감점 -1.5)")
    # (추세 일치 가점 바로 아래에 추가 추천)
    # ✅ NEUTRAL인데도 하락 전환/초기 하락 지속이면 SELL 기회 가점
    if signal == "SELL" and trend == "NEUTRAL":
        # 전환/지속의 “증거”를 지표로 강제: MACD 약세 + Stoch 상단권 + RSI 50+ (되돌림 후 하락 재개 자리)
        if (macd < 0) and (macd < macd_signal) and (stoch_rsi >= 0.6) and (rsi >= 50):
            signal_score += 1.5
            reasons.append("✅ NEUTRAL 구간이지만 MACD 약세 + 되돌림(고Stoch) → 하락 재개 SELL 가점 +1.5")
        
    # ⚠️ RSI + Stoch RSI 과매도 + 패턴 없음 or 애매한 추세 → 바닥 예측 위험
    if rsi < 30 and stoch_rsi < 0.15 and (pattern is None or trend == "NEUTRAL"):
        score -= 1.5
        reasons.append("⚠️ RSI + Stoch RSI 과매도 + 반등 근거 부족 → 진입 위험 (감점 -1.5)")

    if signal == "BUY" and stoch_rsi < 0.15 and prev_stoch_rsi > 0.3 and (macd < 0 or trend != "UPTREND"):
        score -= 1.5
        reasons.append("⚠️ Stoch RSI 급락 + MACD/추세 불확실 → 하락 지속 우려 (감점 -1.5)")
    # 장대 음봉 직후 + 반등 신호 없음 ➝ 위험
    if signal == "BUY" and candles["close"].iloc[-1] < candles["open"].iloc[-1] and \
       (candles["open"].iloc[-1] - candles["close"].iloc[-1]) > (candles["high"].iloc[-2] - candles["low"].iloc[-2]) * 0.9 and \
       pattern is None and trend != "UPTREND":
        score -= 1.5
        reasons.append("📉 장대 음봉 직후 + 반등 패턴 없음 + 추세 불확실 ➝ BUY 진입 위험 (감점 -1.5)")

    # 장대 양봉 직후 + 반전 신호 없음 ➝ 위험
    if signal == "SELL" and candles["close"].iloc[-1] > candles["open"].iloc[-1] and \
       (candles["close"].iloc[-1] - candles["open"].iloc[-1]) > (candles["high"].iloc[-2] - candles["low"].iloc[-2]) * 0.9 and \
       pattern is None and trend != "DOWNTREND":
        score -= 1.5
        reasons.append("📈 장대 양봉 직후 + 반전 패턴 없음 + 추세 불확실 ➝ SELL 진입 위험 (감점 -1.5)")

    # 🔻 최근 캔들 흐름이 진입 방향과 반대인 경우 경고 감점
    if signal == "BUY" and trend != "UPTREND":
    
        # 완전 약세 흐름만 제한
        if (
            candles["close"].iloc[-1] < candles["open"].iloc[-1] and
            candles["close"].iloc[-2] < candles["open"].iloc[-2] and
            rsi < 40
        ):
    
            score -= 0.5
    
            reasons.append(
                "⚠ 최근 약세 흐름 지속 → BUY continuation 약화 (-0.5)"
            )

    if signal == "SELL" and trend != "DOWNTREND":
    
        # 완전 강세 흐름만 제한
        if (
            candles["close"].iloc[-1] > candles["open"].iloc[-1] and
            candles["close"].iloc[-2] > candles["open"].iloc[-2] and
            rsi > 60
        ):
    
            score -= 0.5
    
            reasons.append(
                "⚠ 최근 강세 흐름 지속 → SELL continuation 약화 (-0.5)"
            )

    # 트렌드 전환 직후 경계 구간 감점
    if trend == "UPTREND" and prev_trend == "DOWNTREND" and signal == "BUY":
        score -= 0.5
        reasons.append("⚠️ 하락 추세 직후 상승 반전 → BUY 시그널 신뢰도 낮음 (감점 -0.5)")

    if trend == "DOWNTREND" and prev_trend == "UPTREND" and signal == "SELL":
        score -= 0.5
        reasons.append("⚠️ 상승 추세 직후 하락 반전 → SELL 시그널 신뢰도 낮음 (감점 -0.5)")

    # 🔄 추세 전환 직후 진입 위험
    if signal == "BUY" and trend == "UPTREND" and prev_trend == "DOWNTREND":
        score -= 1.0
        reasons.append("🔄 이전 추세가 DOWN → 추세 전환 직후 BUY → 조기 진입 경고 (감점 -1.0)")

    if signal == "SELL" and trend == "DOWNTREND" and prev_trend == "UPTREND":
        score -= 1.0
        reasons.append("🔄 이전 추세가 UP → 추세 전환 직후 SELL → 조기 진입 경고 (감점 -1.0)")
    

    
    signal_score += score + extra_score
    reasons.extend(base_reasons + extra_reasons)
    # ✅ 캔들 패턴과 추세 강한 일치 시 보너스 점수 부여
    if signal == "BUY" and trend == "UPTREND" and pattern in ["BULLISH_ENGULFING", "HAMMER", "PIERCING_LINE"]:
        signal_score += 1
        opportunity_score += 0.5  # ✅ 패턴-추세 일치 시 추가 점수
        reasons.append("✅ 강한 상승추세 + 매수 캔들 패턴 일치 → 보너스 + 기회 점수 강화 가점 +1.5")

    elif signal == "SELL" and trend == "DOWNTREND" and pattern in ["BEARISH_ENGULFING", "SHOOTING_STAR", "DARK_CLOUD_COVER"]:
        signal_score += 1
        opportunity_score += 0.5  # ✅ 패턴-추세 일치 시 추가 점수
        reasons.append("✅ 강한 하락추세 + 매도 캔들 패턴 일치 → 보너스 + 기회 점수 강화 가점 +1.5")
        
        # ✅ 거래 제한 시간 필터 (애틀랜타 기준)
        now_atlanta = datetime.now(ZoneInfo("America/New_York"))
        
        atlanta_hour = now_atlanta.hour
        atlanta_minute = now_atlanta.minute
        
        # ❌ 거래 금지 시간대 정의
        #is_restricted = (
        #    (3 <= atlanta_hour < 5) or  # 새벽 3~5시
        #    (atlanta_hour == 11) or  # 오전 11시부터 오후 2시
        #    (atlanta_hour == 12) or  # 
        #    (13 <= atlanta_hour < 14) or  # 
        #    (16 <= atlanta_hour < 19)  # 오후 4시부터 오후 7시
        #)
        
        #if is_restricted:
        #    print("❌ 현재 시간은 거래 제한 시간대입니다. GPT 호출 생략")
        #    return 0, reasons


    
    # ▼▼▼ 여기에 붙여넣기 ▼▼▼
    digits = int(abs(np.log10(pip_value_for(pair))))   # EURUSD=4, JPY계열=2
    pv = pip_value_for(pair)

    # 인자로 받은 값을 원시값으로 잡고, 표시는 반올림
    sup_raw = float(support)
    res_raw = float(resistance)

    sup = round(sup_raw, digits)
    res = round(res_raw, digits)

    # 거리는 반올림 전 원시값으로 계산(정확도 ↑)
    dist_to_res_pips = abs(res_raw - price) / pv
    dist_to_sup_pips = abs(price - sup_raw) / pv
    

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
            signal_score -= 1
            reasons.append("⚠️ 추세+패턴 충돌 + 보완 조건 미충족 → 감점-1")

        # === 저항/지지 근접 추격 진입 금지 규칙 ===
    # BUY: 저항 3pip 이내면 금지(FX). 주식은 3pip(가격비례, 예 TSLA $0.09)가 너무 타이트해 ATR*0.15로 대체.
    _near_atr_val = float(atr.iloc[-1]) if hasattr(atr, "iloc") else float(atr or 0)

    # 🟦 돌파 확정 버퍼: 주식은 2pip(예: TSLA $0.06)가 틱 노이즈 수준이라 ATR*0.05로 대체.
    #    너무 크게 잡으면(예 ATR*0.15=$1.5) 돌파 인식이 늦어지므로 0.05 비율 사용.
    if is_stock_pair(pair):
        _breakout_buf = _near_atr_val * 0.05
    else:
        # ===== 기존 FX 로직 (변경 없음) =====
        _breakout_buf = 2 * pip_value_for(pair)

    if signal == "BUY":
        dist_to_res_pips = pips_between(price, resistance, pair)
        if is_stock_pair(pair):
            near_res_block = (resistance is not None and price is not None
                               and abs(resistance - price) < (_near_atr_val * 0.15))
        else:
            # ===== 기존 FX 로직 (변경 없음) =====
            near_res_block = dist_to_res_pips < 3
        if near_res_block:
            signal_score -= 2
            reasons.append(f"📉 저항선 근접 → 신중 진입 필요 (감점-2) [dist={dist_to_res_pips:.1f}pip]")

        last2 = candles.tail(2)
        over1 = (last2.iloc[-1]['close'] > resistance + _breakout_buf) if not last2.empty else False
        over2 = (len(last2) > 1 and last2.iloc[-2]['close'] > resistance + _breakout_buf) if not last2.empty else False
        confirmed_breakout_up = over1 or (over1 and over2)


    # SELL: 지지 3pip 이내면 금지(FX). 주식은 동일하게 ATR*0.15로 대체.
    if signal == "SELL":
        dist_to_sup_pips = pips_between(price, support, pair)
        if is_stock_pair(pair):
            near_sup_block = (support is not None and price is not None
                               and abs(price - support) < (_near_atr_val * 0.15))
        else:
            # ===== 기존 FX 로직 (변경 없음) =====
            near_sup_block = dist_to_sup_pips < 3
        if near_sup_block:
            signal_score -= 1.5
            reasons.append(f"📉 지지선 근접 → 신중 진입 필요 (감점-1.5) [dist={dist_to_sup_pips:.1f}pip]")

        last2 = candles.tail(2)
        under1 = (last2.iloc[-1]['close'] < support - _breakout_buf) if not last2.empty else False
        under2 = (len(last2) > 1 and last2.iloc[-2]['close'] < support - _breakout_buf) if not last2.empty else False
        confirmed_breakdown = under1 or (under1 and under2)


        # ✅ RSI, MACD, Stoch RSI 모두 중립 + Trend도 NEUTRAL → 횡보장 진입 방어
    # ==================================================
    # 1️⃣ 완전 중립 횡보장 방어
    # ==================================================
    if trend == "NEUTRAL":
    
        # 진짜 chop만 약하게 제한
        if (
            47 <= rsi <= 53 and
            abs(macd) < 0.015 and
            0.4 <= stoch_rsi <= 0.6
        ):
    
            signal_score -= 0.5
            reasons.append(
                "⚠️ 완전 횡보(chop) 상태 → 약한 감점 (-0.5)"
            )
    
        # 🔥 애매한 전환/되돌림 구간
        else:
    
            signal_score -= 0.3
            reasons.append(
                "🟡 NEUTRAL 추세 → continuation 신뢰도 낮음 (-0.7)"
            )
    
    
    # ==================================================
    # 2️⃣ BUY 과열 진입 방어 (강력)
    # ==================================================
    if signal == "BUY" and rsi > 85 and stoch_rsi > 0.9:
        if macd < macd_signal:
            signal_score -= 1.0
            reasons.append("⛔ RSI/Stoch RSI 극단 과열 + MACD 약세 → BUY (감점 -1.0)")
        else:
            signal_score -= 0.5
            reasons.append("⚠️ RSI/Stoch 과열 → BUY 피로 구간 (감점 -0.5)")
    
    
        # ③ SELL 과매도 방어 (하락추세 예외 허용)
    # ==================================================
    if signal == "SELL" and rsi < 40:
    
        # ✅ [수정3 핵심] 강한 하락추세(DOWNTREND)에서는 '과매도'라도
        # 추세 지속 SELL이 자주 먹히므로, 과도한 차단을 완화한다.
        if trend == "DOWNTREND":
            # (선택) 너무 극단 과매도면 그래도 조심: rsi<30이면 가볍게만 패널티
            if rsi < 30:
                signal_score -= 0.5
                reasons.append("⚠️ DOWNTREND지만 RSI<30 극단 과매도 → 반등 리스크 경고 (감점 -0.5)")
            else:
                signal_score += 0.5
                reasons.append("📉 하락 추세 지속 + 과매도 → 추세 SELL 허용 (+0.5)")

        
    
        # ✅ NEUTRAL/UPTREND에서는 기존 방어 로직 유지
        else:
            if macd > macd_signal and 0.3 < stoch_rsi < 0.7:
                signal_score += 1
                reasons.append("✅ 과매도 SELL이나 MACD/Stoch 반등 → 예외적 진입 허용 (+1)")
            elif stoch_rsi > 0.3:
                signal_score -= 2
                reasons.append("⚠️ 과매도 SELL + 반등 가능성 → 신중 (감점 -2)")
            else:
                signal_score -= 1.5
                reasons.append("❌ 과매도 SELL + 반등 신호 부족 → 진입 위험 (감점 -1.5)")
    
    # ==================================================
    # 4️⃣ Stoch RSI 바닥 + 패턴 없음 방어
    # ==================================================
    if stoch_rsi < 0.1 and pattern is None:
        signal_score -= 1
        reasons.append("🔴 Stoch RSI 극단 과매도 + 반등 패턴 없음 → 반등 신뢰도 낮음 (감점 -1)")
    
    
    # ==================================================
    # 5️⃣ RSI < 30 구간 정리 (중복 제거)
    # ==================================================
    if rsi < 30:
    
        if pattern in ["HAMMER", "BULLISH_ENGULFING"]:
            signal_score += 2
            reasons.append("🟢 RSI < 30 + 반등 캔들 패턴 → 진입 강화 (+2)")
    
        elif (
            macd < macd_signal
            and trend == "DOWNTREND"
            and len(macd_trend) >= 3
            and macd_trend[-1] <= macd_trend[-2]
        ):
            signal_score -= 1.5
            reasons.append("🔴 RSI < 30 + MACD/추세 약세 지속 → 반등 기대 낮음 (감점 -1.5)")
    
        elif (
            len(macd_trend) >= 3
            and macd_trend[-1] > macd_trend[-2] > macd_trend[-3]
        ):
            signal_score += 1.0
            reasons.append("🟢 RSI 과매도 + MACD 회복 → 반등 기대 (+1.0)")
    
        else:
            signal_score -= 0.5
            reasons.append("⚠️ RSI < 30 but 반등 근거 부족 → 주의 (-0.5)")
    
    
    # ==================================================
    # 6️⃣ RSI > 70 과열 구간
    # ==================================================
    if rsi > 70 and pattern not in ["SHOOTING_STAR", "BEARISH_ENGULFING"]:
        if macd > macd_signal and macd > 0 and trend == "UPTREND":
            signal_score += 0.5
            reasons.append("📈 RSI > 70이나 MACD/UPTREND 유지 → 조건부 BUY 허용 (+0.5)")
        else:
            signal_score -= 1
            reasons.append("⚠️ RSI > 70 + 반전 패턴 없음 → 진입 위험 (감점 -2)")
    
    
    # ==================================================
    # 7️⃣ 눌림목 BUY 강화 (페어 공통)
    # ==================================================
    BOOST_BUY_PAIRS = {"EUR_USD", "GBP_USD", "USD_JPY"}
    
    if pair in BOOST_BUY_PAIRS and signal == "BUY":
    
        # ❌ 하락/횡보 추세에서는 눌림목 BUY 보너스 금지
        if trend != "UPTREND":
            reasons.append(f"{pair}: 하락/중립 추세 → 눌림목 BUY 보너스 제외")
    
        # ❌ 과열 late-entry 방지
        elif (
            rsi is not None and
            stoch_rsi is not None and
            rsi > 75 and
            stoch_rsi > 0.9
        ):
            reasons.append(
                f"{pair}: RSI/Stoch 과열 → late BUY 위험, 눌림목 BUY 보너스 제한"
            )
    
        else:
    
            # ✅ RSI 눌림목
            if 40 <= rsi <= 50:
                signal_score += 0.7
                reasons.append(f"{pair}: RSI 40~50 눌림목 영역 (+0.7)")
    
            # ✅ 초기 반등
            if 0.1 <= stoch_rsi <= 0.3:
                signal_score += 0.5
                reasons.append(f"{pair}: Stoch RSI 바닥 반등 초기 (+0.5)")
    
            # ✅ 캔들 패턴
            if pattern in ["HAMMER", "LONG_BODY_BULL"]:
                signal_score += 0.5
                reasons.append(f"{pair}: 매수 캔들 패턴 확인 (+0.5)")
    
            # ✅ MACD 확인 (보조 역할만)
            if macd > 0:
                signal_score += 0.3
                reasons.append(f"{pair}: MACD 양수 유지 (+0.3)")


        # 7️⃣-2 과매도 반등 BUY (DOWNTREND 허용, 단 조건 엄격)
    if signal == "BUY" and trend == "DOWNTREND":
        if rsi < 30 and stoch_rsi < 0.15 and macd > macd_signal:
            signal_score += 1.5
            reasons.append("🟢 하락추세 과매도 + MACD 반등 → 제한적 반등 BUY (+1.5)")
        else:
            signal_score -= 1
            reasons.append("❌ 하락추세 BUY → 반등 조건 미흡 (감점 -1)")
    
    # ==================================================
    # 8️⃣ 눌림목 조건 (모든 페어 공통)
    # ==================================================
    if signal == "BUY" and trend == "UPTREND":
        if 45 <= rsi <= 55 and 0.0 <= stoch_rsi <= 0.3 and macd > 0:
            signal_score += 1.5
            reasons.append("📈 눌림목 BUY 조건 충족 → 반등 기대 (+1.5)")
    
    if signal == "SELL" and trend == "DOWNTREND":
        if 45 <= rsi <= 55 and 0.7 <= stoch_rsi <= 1.0 and macd < 0:
            signal_score += 1.5
            reasons.append("📉 눌림목 SELL 조건 충족 → 반락 기대 (+1.5)")
    
    
    # ==================================================
    # 9️⃣ RSI 중립 BUY 보정 (과도 방지)
    # ==================================================
    if signal == "BUY" and trend == "UPTREND" and 50 <= rsi <= 60:
        signal_score += 0.5
        reasons.append("RSI 중립(50~60) + 상승추세 → 눌림목 반등 기대 (+0.5)")
    
    
    # ==================================================
    # 🔟 볼린저 밴드 위치
    # ==================================================
    if price >= bollinger_upper:
        reasons.append("🔴 볼린저 상단 → 과매수 경계 (참고)")
    elif price <= bollinger_lower:
        reasons.append("🟢 볼린저 하단 → 반등 관찰 구간 (가점 없음)")
    
    
    # ==================================================
    # 1️⃣1️⃣ 장대 바디 캔들 (과도 점수 축소)
    # ==================================================
    if pattern in ["LONG_BODY_BULL", "LONG_BODY_BEAR"]:
        signal_score += 1.5
        reasons.append(f"📊 장대 바디 캔들 → 추세 지속 가능성 (+1.5)")

    box_info = detect_box_breakout(candles, pair)
    
    high_low_flags = analyze_highs_lows(candles)
    if high_low_flags["new_high"]:
        reasons.append("📈 최근 고점 갱신 → 상승세 유지 가능성↑")
    if high_low_flags["new_low"]:
        reasons.append("📉 최근 저점 갱신 → 하락세 지속 가능성↑")

    if trend == "NEUTRAL" \
       and box_info.get("in_box") \
       and box_info.get("breakout") in ("UP", "DOWN") \
       and (high_low_flags.get("new_high") or high_low_flags.get("new_low")):

        # 신호 일치(+3) 블록과 중복 가점 방지
        aligns = ((box_info["breakout"] == "UP"   and signal == "BUY") or
              (box_info["breakout"] == "DOWN" and signal == "SELL"))

        if not aligns:
            signal_score += 1.5
            reasons.append("🟡 NEUTRAL 예외: 박스 이탈 + 고/저 갱신 → 기본 가점(+1.5)")

    
    if box_info["in_box"] and box_info["breakout"] == "UP" and signal == "BUY":
        signal_score += 3
        reasons.append("📦 박스권 상단 돌파 + 매수 신호 일치 (breakout 가점 강화 +3)")
    elif box_info["in_box"] and box_info["breakout"] == "DOWN" and signal == "SELL":
        signal_score += 3
        reasons.append("📦 박스권 하단 돌파 + 매도 신호 일치 가점+3")
    elif box_info["in_box"] and box_info["breakout"] is None:
        reasons.append("📦 박스권 유지 중 → 관망 경계")

    # SELL 점수 상한 제한
    if signal == "SELL" and signal_score > 5:
        reasons.append("⚠️ SELL 점수 상한 적용 (최대 5점)")
        signal_score = 5

        # --- MACD 교차 가점: 모든 페어 공통 (pip/ATR 스케일 적용) ---
    macd_diff = macd - macd_signal
    _macd_atr = float(atr.iloc[-1]) if hasattr(atr, "iloc") else float(atr or 0)
    if is_stock_pair(pair):
        # 🟦 주식: MACD 값 자체가 가격 스케일이 아니라 '변동성(ATR)' 스케일로 움직이므로
        #    가격 비례(pv) 대신 ATR 비례로 강/약 임계치를 산정. (TSLA처럼 MACD diff가 0.3~2.0대인 경우
        #    pv=price*0.0001 기준 임계치(예: 0.045)는 너무 작아 거의 항상 'strong'으로 잘못 판정되는 문제 수정)
        strong = max(_macd_atr * 0.20, pv * 1.5)
        weak = max(_macd_atr * 0.07, pv * 0.5)
    else:
        # FX는 기존 로직 그대로 유지 (JPY strong=0.015, 그 외 strong=0.00015와 동일)
        strong = 1.5 * pv
        weak = 0.5 * pv
    micro  = 2 * pv               # 미세변동(≈2 pip) 판단용

    if (macd_diff > strong) and trend == "UPTREND":
        signal_score += 3
        reasons.append("MACD 골든크로스(강) + 상승추세 일치 가점+3")
    elif (macd_diff < -strong) and trend == "DOWNTREND":
        signal_score += 3
        reasons.append("MACD 데드크로스(강) + 하락추세 일치 가점+3")
    elif abs(macd_diff) >= weak:
        signal_score += 1
        reasons.append("MACD 교차(약) → 초입 가점 +1")
    else:
        reasons.append("MACD 미세변동 → 가점 보류")
    if signal == "BUY" and len(macd_trend) >= 3:

        if (
            macd_trend[-1] > macd_trend[-2]
            and macd_trend[-2] > macd_trend[-3]
        ):
    
            if macd_trend[-1] < 0:
    
                signal_score += 0.7
    
                reasons.append(
                    "🟢 MACD 음수권 회복중 → 반등 가점 (+0.7)"
                )
    
            else:
    
                signal_score += 0.3
    
                reasons.append(
                    "🟢 MACD 상승 모멘텀 유지 (+0.3)"
                )

    # (선택) 히스토그램 보조 판단은 유지하되 임계도 pip화
    macd_hist = macd_diff
        # =========================
    # 개선1: MACD 방향(약화/반등) + Stoch 과열/과매도 추격 방지 (BUY/SELL 공통)
    if stoch_rsi is not None and macd is not None and macd_signal is not None:
    
        # 1) BUY 추격 방지 (과열 + MACD 약화)
        if signal == "BUY" and stoch_rsi > 0.8 and macd < macd_signal:
            signal_score -= 2.0
            reasons.append("⛔ BUY 차단: Stoch RSI 과열 + MACD 약화(macd<signal) → 추격 매수 위험 감점 -2")
    
        # 2) SELL 추격 방지 (과매도 + MACD 약화)  ✅ 여기서부터 보완이 핵심
        if signal == "SELL" and stoch_rsi < 0.2 and macd < macd_signal:
    
            # (A) 하락 추세면: 과매도라도 '추세형 하락'이 계속될 수 있으니 강차단 금지
            if trend == "DOWNTREND":
                signal_score -= 0.5
                reasons.append("🟡 DOWNTREND + 과매도(Stoch<0.2) + MACD 약화 → 추세형 하락 지속 가능(경고 -0.5)")
    
            # (B) NEUTRAL(전환/분배) 구간: RSI가 50 아래면 하락쪽 우세 가능 → 강차단 금지(중립 처리)
            elif trend == "NEUTRAL" and rsi is not None and rsi < 50:
                # 점수는 건드리지 않고 '중립 경고'만 남김
                reasons.append("🟡 NEUTRAL 전환 구간 + RSI<50 + 과매도(Stoch<0.2) → 추격 숏 단정 금지(중립)")
    
            # (C) 나머지(상승/횡보 성격): 과매도 숏은 반등에 말릴 확률 높음 → 기존처럼 강차단
            else:
                signal_score -= 2.0
                reasons.append("⛔ SELL 차단: 과매도(Stoch<0.2) + MACD 약화 + 추세 불리 → 추격 매도 위험 감점 -2")
   
    if stoch_rsi >= 0.95:
        if trend == "UPTREND" and macd > 0:
            signal_score -= 0.5
            reasons.append("🟡 Stoch RSI 과열이지만 상승추세 + MACD 양수 → 조건부 감점 -0.5")
        else:
            signal_score -= 1
            reasons.append("🔴 Stoch RSI 1.0 → 극단적 과매수 → 피로감 주의 감점 -1")
    
    pip = pv  # 🟦 고정 0.01(JPY 가정) 대신 자산군별 pip_value(pv)로 통일 (FX는 페어별, 주식은 가격비례)
    
    # 안전 처리
    if price is None:
        price = close
    if close is None:
        close = price
    
    atr_val = atr if atr is not None else 0.0
    res_val = resistance if resistance is not None else None
    
    near_resistance = False
    if res_val is not None and price is not None:
        near_resistance = (res_val - price) <= max(10 * pip, atr_val * 0.6)
    
    buffer = max(2 * pip, atr_val * 0.10)
    breakout_confirmed = False
    if res_val is not None and close is not None:
        breakout_confirmed = close >= (res_val + buffer)
    
    if stoch_rsi is not None and stoch_rsi > 0.8:
    
        if signal == "BUY" and trend == "UPTREND" and rsi < 70 and macd is not None and macd_signal is not None and macd >= macd_signal:
    
            if breakout_confirmed and not near_resistance:
                if pair == "USD_JPY":
                    signal_score += 2
                    reasons.append("USDJPY: Stoch RSI 과열 + 돌파확정 → 모멘텀 가점 +2")
                else:
                    signal_score += 1.5
                    reasons.append("Stoch RSI 과열 + 돌파확정 → 모멘텀 가점 +1.5")
            else:
                signal_score -= 2
                reasons.append("Stoch RSI 과열 + 저항 근접/돌파미확정 → 추격 BUY 위험 감점 -2")
    
        else:
            reasons.append("Stoch RSI 과열 → 고점 피로, 관망")
    
    elif stoch_rsi < 0.2:
        # BUY일 때만 과매도 처리
        if signal == "BUY":
    
            # 🔥 1순위 핵심 수정:
            # 극단 과매도 + MACD 약화(macd < macd_signal)이면
            # 반등 가점(+1) 주지 말고 "칼날"로 보고 감점
            if stoch_rsi < 0.05 and macd < macd_signal:
                signal_score -= 1.5
                reasons.append("🔴 Stoch RSI 극단 과매도(<0.05) + MACD<Signal → 하락 가속/전환 위험 (감점 -1.5)")
    
            else:
                # 기존 로직 유지
                if trend == "DOWNTREND":
                    signal_score += 0.5
                    reasons.append("Stoch RSI 과매도 + 하락추세 → 반등은 제한적(+0.5)")
                else:
                    # ✅ 방법1: Balance breakout에서는 과매도 반등 BUY(+1) 가점 제거
                    if (strategy_name or "").strip().lower() == "balance breakout":
                        reasons.append("ℹ Balance breakout: Stoch RSI 과매도 반등 BUY 가점 미적용")
                    else:
                        signal_score += 1
                        reasons.append("Stoch RSI 과매도 → BUY 반등 기대(+1)")
    
        else:
            # SELL은 기존대로 관망
            reasons.append("Stoch RSI 과매도 → SELL은 추격 위험, 관망")
    
    else:
        reasons.append("Stoch RSI 중립")

    if trend == "UPTREND" and signal == "BUY":
    
        # 🔥 과열 late-entry 방지
        if (
            stoch_rsi is not None and
            rsi is not None and
            stoch_rsi > 0.9 and
            rsi > 75
        ):
            reasons.append(
                "⚠️ RSI/Stoch 과열 → late BUY 위험, 추세 가점 제외"
            )
    
        # 🔥 칼날 방지
        elif stoch_rsi < 0.05 and macd < macd_signal:
            reasons.append(
                "⚠️ 표기상 UPTREND지만 Stoch 극단 과매도 + MACD 약화 → 추세 전환 의심(추세일치 가점 제외)"
            )
    
        else:
            signal_score += 0.5
            reasons.append("추세 상승 + 매수 일치 가점+0.5")
    
    
    elif trend == "DOWNTREND" and signal == "SELL":
    
        # 🔥 과매도 추격 SELL 방지
        if (
            stoch_rsi is not None and
            rsi is not None and
            stoch_rsi < 0.1 and
            rsi < 25
        ):
            reasons.append(
                "⚠️ RSI/Stoch 과매도 → late SELL 위험, 추세 가점 제외"
            )
    
        # 🔥 숏말림 방지
        elif stoch_rsi is not None and stoch_rsi >= 0.95:
            reasons.append(
                "⛔ Stoch RSI 과열(≥0.95) → 숏 말림 위험, 추세 매도 가점 미적용"
            )
    
        else:
            signal_score += 0.5
            reasons.append("추세 하락 + 매도 일치 가점+0.5")


    if liquidity == "좋음":
        reasons.append("🟡 유동성 양호 (참고)")
    last_3 = candles.tail(3)
    if (
        all(last_3["close"] < last_3["open"]) 
        and trend == "DOWNTREND" 
        and pattern in ["NEUTRAL", "SHOOTING_STAR", "LONG_BODY_BEAR"]
    ):
    
        # 🔥 과매도 추격 SELL 방지
        if (
            rsi is not None and
            stoch_rsi is not None and
            rsi < 25 and
            stoch_rsi < 0.1
        ):
            reasons.append(
                "⚠️ 3봉 연속 음봉이지만 RSI/Stoch 과매도 → late SELL 위험, 추가 가점 제외"
            )
    
        else:
            signal_score += 0.5
            reasons.append(
                "🔻 최근 3봉 연속 음봉 + 하락추세 → SELL continuation 가점+0.5"
            )

        # === 박스권 상단/하단 근접 진입 제한 ===
    recent = candles.tail(10)
    if not recent.empty:
        box_high = recent['high'].max()
        box_low  = recent['low'].min()

        # pip 단위 거리 계산(동적)
        near_top_pips = abs(box_high - price) / pv
        near_low_pips = abs(price - box_low) / pv

        # 돌파/이탈 확인을 위한 가격 버퍼(동적)
        buf_price = thr["breakout_buf_pips"] * pv  # 가격단위

        # 상단 근접 매수 금지 (확정 돌파 or 리테스트만 허용)
        if signal == "BUY" and box_info.get("in_box") and box_info.get("breakout") is None:
            confirmed_top_break = recent.iloc[-1]['close'] > (box_high + buf_price)
            retest_support = (recent.iloc[-1]['low'] > box_high - buf_price) and (near_top_pips <= NEAR_PIPS)
            if near_top_pips <= NEAR_PIPS and not (confirmed_top_break or retest_support):
                signal_score -= 1.5
                reasons.append("⚠️ 박스 상단 근접 매수 위험 (감점-1.5)")

        # 하단 근접 매도 금지 (확정 이탈 or 리테스트만 허용)
        if signal == "SELL" and box_info.get("in_box") and box_info.get("breakout") is None:
            confirmed_bottom_break = recent.iloc[-1]['close'] < (box_low - buf_price)
            retest_resist = (recent.iloc[-1]['high'] < box_low + buf_price) and (near_low_pips <= NEAR_PIPS)
            if near_low_pips <= NEAR_PIPS and not (confirmed_bottom_break or retest_resist):
                signal_score -= 1.5
                reasons.append("⚠️ 박스 하단 근접 매도 위험 (감점-1.5)")
                
    # 상승 연속 양봉 패턴 보정 BUY
    if (
        all(last_3["close"] > last_3["open"]) 
        and trend == "UPTREND" 
        and pattern in ["NEUTRAL", "LONG_BODY_BULL", "INVERTED_HAMMER"]
    ):
    
        # 🔥 과열 late-entry 방지
        if (
            rsi is not None and
            stoch_rsi is not None and
            rsi > 75 and
            stoch_rsi > 0.9
        ):
            reasons.append(
                "⚠️ 3봉 연속 양봉이지만 RSI/Stoch 과열 → late BUY 위험, 추가 가점 제외"
            )
    
        else:
            signal_score += 0.5
            reasons.append(
                "🟢 최근 3봉 연속 양봉 + 상승추세 → BUY continuation 가점+0.5"
            )

        # 1) 패턴 그룹 먼저 정의
    bullish_patterns = ["BULLISH_ENGULFING", "HAMMER", "MORNING_STAR"]
    bearish_patterns = ["SHOOTING_STAR", "BEARISH_ENGULFING", "HANGING_MAN", "EVENING_STAR"]
        # 2) 방향에 따라 가점/감점 다르게 적용
    if pattern in bullish_patterns:
        if is_buy:
            signal_score += 2
            reasons.append(f"🟢 강한 매수형 패턴 ({pattern}) ➜ BUY 근거 강화 (+2)")
        elif is_sell:
            signal_score -= 1.5
            reasons.append(f"⚠️ 매수 반전 패턴 ({pattern}) ➜ SELL 신뢰도 하락 (-1.5)")
    
    elif pattern in bearish_patterns:
        if is_sell:
            signal_score += 2
            reasons.append(f"🔴 강한 매도형 패턴 ({pattern}) ➜ SELL 근거 강화 (+2)")
        elif is_buy:
            signal_score -= 1.5
            reasons.append(f"⚠️ 매도 반전 패턴 ({pattern}) ➜ BUY 신뢰도 하락 (-1.5)")
    # 교과서적 기회 포착 보조 점수
    op_score, op_reasons = must_capture_opportunity(rsi, stoch_rsi, macd, macd_signal, pattern, candles, trend, atr, price, bollinger_upper, bollinger_lower, support, resistance, support_distance, resistance_distance, pip_size, expected_direction=None)
    if op_score > 0:
        signal_score += op_score
        reasons += op_reasons

    try:
        # 하락 추세 말기: 과매도 + 지지선 근접에서 SELL은 숏스퀴즈 위험 → 감점
        if trend == "DOWNTREND" and signal == "SELL":
        
            near_support = (
                support is not None and
                price is not None and
                atr is not None and
                abs(price - support) <= atr * 0.25
            )
        
            if (rsi is not None) and (rsi < 32) and near_support:
        
                signal_score -= 3.0
                reasons.append(
                    "🔴 과매도 + 지지선 매우 근접(ATR 기준) → late SELL / 숏스퀴즈 위험 (-3.0)"
                )
        
            elif (rsi is not None) and (rsi < 32):
        
                signal_score -= 1.0
                reasons.append(
                    "🟠 과매도 구간 SELL → 반등 위험 (-1.0)"
                )

        # 상승 추세 말기: 과매수 + 저항선 근접에서 BUY는 고점 물림 위험 → 감점
        if trend == "UPTREND" and signal == "BUY":
        
            near_resistance = (
                resistance is not None and
                price is not None and
                atr is not None and
                abs(resistance - price) <= atr * 0.25
            )
        
            if (rsi is not None) and (rsi > 68) and near_resistance:
        
                signal_score -= 3.0
                reasons.append(
                    "🔴 과매수 + 저항선 매우 근접(ATR 기준) → late BUY / 돌파 실패 위험 (-3.0)"
                )
        
            elif (rsi is not None) and (rsi > 68):
        
                signal_score -= 1.0
                reasons.append(
                    "🟠 과매수 구간 BUY → 조정 위험 (-1.0)"
                )

    except Exception as e:
        # 배포 중 예외로 전략이 멈추는 걸 방지 (안전장치)
        reasons.append(f"⚠️ 추세 말기 감점 필터 예외 발생(무시): {e}")
    

    return signal_score, reasons

app = FastAPI()

OANDA_API_KEY = os.getenv("OANDA_API_KEY")
ACCOUNT_ID = os.getenv("ACCOUNT_ID")
openai.api_key = os.getenv("OPENAI_API_KEY")

# ============================================================
# 🟦 Alpaca (미국 주식) 연동 설정
# ============================================================
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
# 기본값: Paper(모의투자). 실거래로 전환 시 환경변수 ALPACA_PAPER=false 설정
ALPACA_PAPER = os.getenv("ALPACA_PAPER", "true").strip().lower() != "false"
ALPACA_TRADE_BASE_URL = (
    "https://paper-api.alpaca.markets" if ALPACA_PAPER else "https://api.alpaca.markets"
)
ALPACA_DATA_BASE_URL = "https://data.alpaca.markets"
# 주문당 고정 매수 금액(달러). sizing_mode="fixed"일 때 또는 risk 계산 실패시 폴백으로 사용.
ALPACA_FIXED_NOTIONAL_USD = float(os.getenv("ALPACA_FIXED_NOTIONAL_USD", "1000"))

# 🟦 포지션 사이징 모드: "tiered"(가격대별 고정 수량표, 기본값) / "risk"(계좌 리스크 % 기반) / "fixed"(고정 금액)
ALPACA_SIZING_MODE = os.getenv("ALPACA_SIZING_MODE", "tiered").strip().lower()
# 1회 거래당 허용 리스크 = 계좌 equity의 이 비율(%). 예: 0.5 → 계좌 5만달러면 250달러 리스크.
ALPACA_RISK_PCT = float(os.getenv("ALPACA_RISK_PCT", "0.5"))
# SL이 너무 타이트해서 risk 계산상 수량이 과도하게 커지는 것을 막는 안전 캡(달러, notional 기준).
ALPACA_MAX_NOTIONAL_USD = float(os.getenv("ALPACA_MAX_NOTIONAL_USD", "5000"))
# 주식 SL 버퍼 = ATR * 이 배수 (get_buffer_by_symbol). 페이퍼 트레이딩하면서 0.15/0.20/0.25 A/B 테스트용.
ALPACA_SL_BUFFER_ATR_MULT = float(os.getenv("ALPACA_SL_BUFFER_ATR_MULT", "0.15"))
# 신호가 vs 주문 직전 실시간가 차이 허용 한도(%). 이걸 넘으면 신호를 신뢰할 수 없다고 보고 주문 스킵.
ALPACA_MAX_PRICE_GAP_PCT = float(os.getenv("ALPACA_MAX_PRICE_GAP_PCT", "1.5"))
# 🟦 주식 TP/SL ATR 배수. TradingView Pine 전략("BUY STOCK PORTFOLIO A2")의
#    tpATR(기본 0.8) / slATR(기본 1.0) 입력값과 반드시 동일하게 맞춰야 한다.
#    Pine에서 입력값을 바꾸면 여기 환경변수도 같이 바꿔야 정렬이 유지된다.
STOCK_TP_ATR_MULT = float(os.getenv("STOCK_TP_ATR_MULT", "0.8"))
STOCK_SL_ATR_MULT = float(os.getenv("STOCK_SL_ATR_MULT", "1.0"))

ALPACA_HEADERS = {
    "APCA-API-KEY-ID": ALPACA_API_KEY or "",
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY or "",
}

import re as _re
_STOCK_TICKER_RE = _re.compile(r"^[A-Z]{1,5}(\.[A-Z])?$")

# pair별 가장 최근에 들어온 가격을 캐시 (주식의 'pip 등가값' 계산에 사용)
_last_price_cache: dict[str, float] = {}


def is_stock_pair(pair: str) -> bool:
    """
    OANDA FX 페어는 'USD_JPY' 처럼 '_' 가 들어간다.
    Alpaca 주식 심볼은 'TSLA', 'AAPL' 처럼 '_' 없는 1~5자 영문 티커다.
    → '_'가 없고 티커 패턴에 맞으면 주식으로 판단.
    """
    if not pair:
        return False
    p = pair.upper().strip()
    if "_" in p or "/" in p:
        return False
    return bool(_STOCK_TICKER_RE.match(p))


def price_round_digits(pair: str) -> int:
    """주문 가격(TP/SL) 반올림 자릿수. 주식은 센트 단위(2자리)."""
    if is_stock_pair(pair):
        return 2
    return 3 if pair.endswith("JPY") else 5


def base_granularity_for(pair: str) -> str:
    """
    분석 기준 캔들 단위. 주식은 15분봉, FX는 기존과 동일하게 30분봉.
    (캔들 조회, 지지/저항, MTF 요약, GPT 프롬프트 안내문 전부 이 값을 따른다.)
    """
    return "M15" if is_stock_pair(pair) else "M30"


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

def summarize_recent_candle_flow(candles, window=20):
    highs = candles['high'].tail(window).dropna()
    lows = candles['low'].tail(window).dropna()
    closes = candles['close'].tail(window).dropna()

    if highs.empty or lows.empty or closes.empty:
        return "최근 캔들 데이터 부족"

    new_high = closes.iloc[-1] >= highs.max()
    new_low = closes.iloc[-1] <= lows.min()
    direction = "상승추세" if new_high else ("하락추세" if new_low else "횡보")

    up_count = (closes.diff() > 0).sum()
    down_count = (closes.diff() < 0).sum()

    return f"최근 {window}개 캔들 기준 {direction}, 상승:{up_count}개, 하락:{down_count}개"

@app.post("/webhook")
async def webhook(request: Request):
    """
    🟦 가벼운 async 래퍼. 실제 처리(process_webhook_sync)는 스레드 풀에서 돌려서,
       여러 알림이 거의 동시에 들어와도 이벤트 루프가 막히지 않고 동시에 처리되게 한다.
       (이전에는 전체 로직이 async def 안에서 동기 블로킹 호출을 그대로 실행해서,
        알림이 몰리면 한 줄로 순서대로 처리되며 뒤로 갈수록 지연/타임아웃이 생겼음)
    """
    raw = (await request.body()) or b""
    return await asyncio.to_thread(process_webhook_sync, raw)


def process_webhook_sync(raw: bytes):
    print("✅ STEP 1: 웹훅 진입")
    global _last_execution_time
    current_time = _t.time()
    if current_time - _last_execution_time < 600:  # 600초 = 10분
        print(f"⚠️ [차단] 10분 쿨다운 중입니다. (경과: {int(current_time - _last_execution_time)}초)")
        return JSONResponse(content={"status": "ignored", "reason": "cooldown_active"})
    try:
        data = json.loads(raw.decode("utf-8") or "{}")
    except Exception:
        return JSONResponse(
            content={"error": "invalid json body", "raw": raw[:200].decode("utf-8", "ignore")},
            status_code=400
        )
    # 🟦 TradingView 알림이 'pair' 대신 'symbol'로 보내는 경우도 허용
    #    (예: PineScript alert(... '{"symbol":"{{ticker}}", ...}' ...))
    pair = data.get("pair") or data.get("symbol")
    signal = data.get("signal")
    print(f"✅ STEP 2: 데이터 수신 완료 | pair: {pair}")

    _ = check_recent_opposite_signal(pair, signal)  # 소프트 OFF: 기록만, 차단 안 함
        
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

    # 🟦 주식의 'pip 등가값'/digits 계산에 사용할 최근가 캐시
    if pair:
        _last_price_cache[pair] = price

    # 🟦 FX 형식('_' 포함)도 아니고 주식 티커 패턴도 아니면(예: BTCUSD 등) 즉시 차단.
    #    그대로 두면 is_stock_pair=False → OANDA 경로로 새서 존재하지 않는 instrument로 주문 시도하게 됨.
    if not is_stock_pair(pair) and "_" not in (pair or "") and "/" not in (pair or ""):
        return JSONResponse(
            content={"error": f"지원하지 않는 심볼 형식입니다: {pair} (FX는 'USD_JPY', 주식은 'TSLA' 형식만 지원)"},
            status_code=400
        )

    alert_name = data.get("alert_name", "기본알림")

    candles = get_candles(pair, base_granularity_for(pair), 200)
    # ✅ 캔들 방어 로직 — ATR(14) 계산 가능한 최소 개수(14개)로 강화
    candle_count = len(candles) if candles is not None else 0
    print(f"📊 [{pair}] 캔들 수신: {candle_count}개")
    if candles is None or candles.empty or candle_count < 14:
        return JSONResponse(
            content={"error": f"캔들 데이터 부족: {pair} {candle_count}개 (ATR(14) 계산에 최소 14개 필요)"},
            status_code=400
        )
    print("✅ STEP 4: 캔들 데이터 수신")
    # 동적 지지/저항선 계산 (파동 기반)
    print("📉 candles.tail():\n", candles.tail())
    if candles is not None and not candles.empty and len(candles) >= 2:
        print("🧪 candles.iloc[-1]:", candles.iloc[-1])
        print("📌 columns:", candles.columns)
        current_price = candles.iloc[-1]['close']
    else:
        current_price = None

    # ✅ 방어 로직 추가 (607줄 기준)
    if current_price is None:
        return JSONResponse(
            content={"error": "current_price가 None (candles close missing)"},
            status_code=400
        )
    # ✅ ATR 먼저 계산 (Series)
    atr_series = calculate_atr(candles)
    last_atr = float(atr_series.dropna().iloc[-1]) if not atr_series.dropna().empty else None

    # ✅ ATR 계산 불가(캔들 부족 등)면 여기서 죽지 않고 깔끔하게 에러 응답
    if last_atr is None:
        print(f"❗ [{pair}] ATR 계산 불가 — 캔들 {len(candles)}개로는 ATR(14) 계산에 데이터 부족")
        return JSONResponse(
            content={
                "error": f"{pair}: ATR 계산 불가 (캔들 {len(candles)}개, ATR(14)에 최소 14개 필요)"
            },
            status_code=400
        )

    # ✅ 지지/저항 계산 - timeframe 키 "H1" 로, atr에는 Series 전달
    support, resistance = get_enhanced_support_resistance(
        candles, price=current_price, atr=last_atr, timeframe=base_granularity_for(pair), pair=pair
    )

    support_resistance = {"support": support, "resistance": resistance}
    support_distance = abs(price - support)
    resistance_distance = abs(resistance - price)

    # ✅ 현재가와 저항선 거리 계산 (pip 기준 거리 필터 적용을 위함)
    pip_size = pip_value_for(pair)  # 🟦 주식/JPY/그 외 FX를 모두 인식하는 통합 함수로 교체
    resistance_distance = abs(resistance - price)

    if candles is None or candles.empty:
        return JSONResponse(content={"error": "캔들 데이터를 불러올 수 없음"}, status_code=400)

    close = candles["close"]
    rsi = calculate_rsi(close)
    stoch_rsi_series = calculate_stoch_rsi(rsi)
    stoch_rsi = stoch_rsi_series.dropna().iloc[-1] if not stoch_rsi_series.dropna().empty else 0
    macd, macd_signal = calculate_macd(close)
    lookback = 14  # 최근 14봉 기준 추세 분석용
    # RSI 트렌드
    rsi_trend = list(rsi.iloc[-lookback:].round(2)) if not rsi.empty else []

    # MACD 트렌드
    macd_trend = list(macd.iloc[-lookback:].round(5)) if not macd.empty else []

    # MACD 시그널 트렌드
    macd_signal_trend = list(macd_signal.iloc[-lookback:].round(5)) if not macd_signal.empty else []

    # Stoch RSI 트렌드
    if not stoch_rsi_series.dropna().empty:
        stoch_rsi_trend = list(stoch_rsi_series.dropna().iloc[-lookback:].round(2))
    else:
        stoch_rsi_trend = []
    
    print(f"✅ STEP 5: 보조지표 계산 완료 | RSI: {rsi.iloc[-1]}")
    boll_up, boll_mid, boll_low = calculate_bollinger_bands(close)

    pattern = detect_candle_pattern(candles)
    trend = detect_trend(candles, rsi, boll_mid, pair=pair)
    prev_trend = detect_trend(candles.iloc[:-1], rsi.iloc[:-1], boll_mid.iloc[:-1], pair=pair)
    stoch_rsi_clean = stoch_rsi_series.dropna()
    prev_stoch_rsi = stoch_rsi_clean.iloc[-2] if len(stoch_rsi_clean) >= 2 else 0
    liquidity = estimate_liquidity(candles)
    news = fetch_forex_news()
    news_score, news_msg = news_risk_score(pair)
    high_low_analysis = analyze_highs_lows(candles)
    atr = float(atr_series.dropna().iloc[-1]) if not atr_series.dropna().empty else 0.0
    fibo_levels = calculate_fibonacci_levels(candles["high"].max(), candles["low"].min())
    # 📌 현재가 계산
    price = current_price
    price_digits = int(abs(np.log10(pip_value_for(pair))))  # EURUSD=4, JPY계열=2
    signal_score, reasons = score_signal_with_filters(
        rsi.iloc[-1],
        macd.iloc[-1],
        macd_signal.iloc[-1],
        stoch_rsi,
        prev_stoch_rsi,
        trend,
        prev_trend,
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
        pip_size,
        macd_trend
    )
    # ===== GPT 입력 업그레이드용 안전한 추가 정보 =====
    try:
        recent_ohlc = []
        for _, row in candles.tail(5).iterrows():
            recent_ohlc.append({
                "open": round(float(row["open"]), price_digits),
                "high": round(float(row["high"]), price_digits),
                "low": round(float(row["low"]), price_digits),
                "close": round(float(row["close"]), price_digits),
            })
    except Exception as e:
        print("❌ recent_ohlc 생성 실패:", e)
        recent_ohlc = []

    try:
        last_bar = candles.iloc[-1]
        last_open = float(last_bar["open"])
        last_high = float(last_bar["high"])
        last_low = float(last_bar["low"])
        last_close = float(last_bar["close"])

        last_range = max(last_high - last_low, pip_size)
        last_body = abs(last_close - last_open)
        upper_wick = last_high - max(last_open, last_close)
        lower_wick = min(last_open, last_close) - last_low

        candle_micro = {
            "last_body": round(last_body, price_digits),
            "last_range": round(last_range, price_digits),
            "last_body_ratio": round(last_body / last_range, 3),
            "upper_wick": round(max(upper_wick, 0.0), price_digits),
            "lower_wick": round(max(lower_wick, 0.0), price_digits),
        }
    except Exception as e:
        print("❌ candle_micro 생성 실패:", e)
        candle_micro = {}

    try:
        distance_to_support_pips = round(pips_between(price, support, pair), 1) if support is not None else None
        distance_to_resistance_pips = round(pips_between(price, resistance, pair), 1) if resistance is not None else None
    except Exception as e:
        print("❌ support/resistance 거리 계산 실패:", e)
        distance_to_support_pips = None
        distance_to_resistance_pips = None

    try:
        if len(candles) >= 4:
            recent_high_3 = float(candles["high"].iloc[-4:-1].max())
            recent_low_3 = float(candles["low"].iloc[-4:-1].min())
        else:
            recent_high_3 = float(candles["high"].tail(3).max())
            recent_low_3 = float(candles["low"].tail(3).min())

        breakout_context = {
            "above_recent_high_3": bool(price > recent_high_3),
            "below_recent_low_3": bool(price < recent_low_3),
            "breakout_margin_pips_up": round((price - recent_high_3) / pip_size, 1),
            "breakout_margin_pips_down": round((recent_low_3 - price) / pip_size, 1),
        }
    except Exception as e:
        print("❌ breakout_context 생성 실패:", e)
        breakout_context = {}

    try:
        recent10 = candles.tail(10)
        box_high = float(recent10["high"].max())
        box_low = float(recent10["low"].min())
        box_width = max(box_high - box_low, pip_size)

        structure_context = {
            "box_high": round(box_high, price_digits),
            "box_low": round(box_low, price_digits),
            "box_width_pips": round(box_width / pip_size, 1),
            "price_position_in_box": round((price - box_low) / box_width, 2),
        }
    except Exception as e:
        print("❌ structure_context 생성 실패:", e)
        structure_context = {}
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
        "support": round(support, price_digits),
        "resistance": round(resistance, price_digits),
        "recent_ohlc": recent_ohlc,
        "candle_micro": candle_micro,
        "distance_to_support_pips": distance_to_support_pips,
        "distance_to_resistance_pips": distance_to_resistance_pips,
        "breakout_context": breakout_context,
        "structure_context": structure_context,
        "news": f"{news} | {news_msg}",
        "new_high": bool(high_low_analysis["new_high"]),
        "new_low": bool(high_low_analysis["new_low"]),
        "atr": atr,
        "signal_score": signal_score,
        "score_components": reasons,
        "rsi_trend": rsi_trend[-3:],      # ✅ 최근 5개로 압축
        "macd_trend": macd_trend[-3:],
        "macd_signal_trend": macd_signal_trend[-3:],
        "stoch_rsi_trend": stoch_rsi_trend[-3:],
        "strategy_name": (
            data.get("strategy_name", "").strip()
            or data.get("alert_name", "").strip()
        ),
        "alert_name": data.get("alert_name", "").strip(),
        "alert_data": data.get("alert_data", {}),
    }




    # 🎯 뉴스 리스크 점수 추가 반영
    signal_score += news_score
    reasons.append(f"📰 뉴스 리스크: {news_msg} (점수 {news_score})")
            
    recent_trade_time = get_last_trade_time()
    time_since_last = datetime.utcnow() - recent_trade_time if recent_trade_time else timedelta(hours=999)
    allow_conditional_trade = time_since_last > timedelta(hours=2)

    strategy_thresholds = {
    "Balance breakout": 4.5,
    "BUY_ENTRY_BAR_CLOSE": -7.0,
    "SELL_ENTRY_BAR_CLOSE": -7.0,
    "기본알림": 3.0,
    "Test Alarm": 0.0,
    "BUY_STOCK_PORTFOLIO_A2": -1.0
    }

    alert_data = payload.get("alert_data", {})
    strategy_name = (
        alert_data.get("strategy_name")
        or alert_data.get("alert_name")
        or payload.get("strategy_name")
        or payload.get("alert_name")
        or payload.get("strategy")
        or "기본알림" 
        or ""
    ).strip()
    threshold = strategy_thresholds.get(strategy_name, 999)

    # 🟦 Pine 쪽 alert()는 안 건드리고, 주식 신호인데 strategy_name이 따로 안 와서
    #    "기본알림"(FX 기준 3.0)으로 떨어진 경우만 주식 전용 threshold로 바꿔준다.
    if is_stock_pair(pair) and strategy_name == "기본알림":
        threshold = strategy_thresholds.get("BUY_STOCK_PORTFOLIO_A2", -1.0)

    print(f"[DEBUG] strategy_name={strategy_name}, threshold={threshold}, score={signal_score}")
    gpt_feedback = "GPT 분석 생략: 점수 미달"
    decision, tp, sl = None, None, None  
    final_decision, final_tp, final_sl = None, None, None
    gpt_raw = None
    raw_text = ""  # ✅ 조건문 전에 미리 초기화
    if signal_score >= threshold:
        # 📸 [추가] 1. 사진 찍기
        # 🟦 주식은 차트 캡처를 스킵한다 (Playwright 미설치로 매번 실패할 뿐 아니라,
        #    GPT 분석 전 불필요한 지연(수 초)을 줄여서 알림→체결 시차를 최소화하기 위함).
        #    FX는 기존과 동일하게 캡처 시도.
        if is_stock_pair(pair):
            chart_path = None
        else:
            try:
                chart_path = capture_tradingview_chart(pair)
            except Exception as e:
                print(f"❌ 차트 캡처 실패, 이미지 없이 계속 진행: {e}")
                chart_path = None
    
        # 🖼 [추가] 2. 이미지를 GPT가 읽을 수 있는 문자열로 변환
        base64_image = encode_image(chart_path) if chart_path else None
    
        # 🤖 [수정] 3. GPT 분석 함수 호출 (base64_image 인자 추가)
        # ※ 주의: analyze_with_gpt 함수 정의 부분에도 image 인자를 받도록 수정해야 합니다.
        gpt_raw = None
        
        for attempt in range(3):
        
            try:
        
                gpt_raw = analyze_with_gpt(
                    payload,
                    price,
                    pair,
                    candles,
                    base64_image
                )
        
                if (
                    gpt_raw
                    and "GPT_ERROR" not in str(gpt_raw)
                ):
                    break
        
                print(
                    f"⚠ GPT 실패 → 재시도 {attempt+2}/3"
                )
        
                _t.sleep(2)
        
            except Exception as e:
        
                print(
                    f"⚠ GPT 호출 실패 {attempt+1}/3: {e}"
                )
        
                _t.sleep(2)
        
        if (
            not gpt_raw
            or "GPT_ERROR" in str(gpt_raw)
        ):
            print(
                "❌ GPT 3회 재시도 실패"
            )
            
        print("✅ STEP 6: GPT 응답 수신 완료 (이미지 분석 포함)")
        # ✅ 추가: 파싱 결과 강제 정규화 (대/소문자/공백/이상값 방지)
        raw_text = (
            gpt_raw if isinstance(gpt_raw, str)
            else json.dumps(gpt_raw, ensure_ascii=False)
            if isinstance(gpt_raw, dict) else str(gpt_raw)
        )
        print(f"📄 GPT Raw Response: {raw_text!r}")
        gpt_feedback = raw_text
        parsed_decision, tp, sl = parse_gpt_feedback(raw_text) if raw_text else ("WAIT", None, None)
        
        if parsed_decision in ["BUY", "SELL"]:
            final_decision = parsed_decision
            final_tp = tp
            final_sl = sl
        
            print(
                f"[✔️UPDATE] GPT 결정 적용: "
                f"{final_decision}, tp={final_tp}, sl={final_sl}"
            )
        else:
        
            final_decision = "WAIT"
            final_tp = None
            final_sl = None
        
            print(
                f"⚠️[WAIT] GPT 반환값 무효: "
                f"{parsed_decision}"
            )
       
    else:
        print("🚫 GPT 분석 생략: 점수 2.0점 미만")
        print("🔎 GPT 분석 상세 로그")
        print(f" - GPT Raw (일부): {raw_text[:150]}...")  # 응답 일부만 잘라서 표시
        print(f" - Parsed Decision: {decision}, TP: {tp}, SL: {sl}")
        print(f" - 최종 점수: {signal_score}")
        print(f" - 트리거 사유 목록: {reasons}")

        if final_decision is None:
            final_decision = "SKIPPED_BY_THRESHOLD"
            final_tp = None
            final_sl = None

    result = gpt_raw or ""

    # GPT 텍스트 추출(반환 키 다양성 대비)
    gpt_feedback = (
        gpt_raw.get("analysis_text")
        or gpt_raw.get("analysis")
        or gpt_raw.get("explanation")
        or gpt_raw.get("summary")
        or gpt_raw.get("reason")
        or gpt_raw.get("message")
        or json.dumps(gpt_raw, ensure_ascii=False)    # dict인데 위 키가 없으면 JSON 문자열로 기록
    ) if isinstance(gpt_raw, dict) else str(gpt_raw or "")
    

    if not gpt_feedback or not str(gpt_feedback).strip():
        gpt_feedback = "GPT 응답 없음"
    
    print(f"✅ STEP 7: GPT 해석 완료 | decision: {final_decision}, TP: {final_tp}, SL: {final_sl}")
   
    
    # 📌 outcome_analysis 및 suggestion 기본값 세팅
    outcome_analysis = "WAIT 또는 주문 미실행"
    adjustment_suggestion = ""
    price_movements = None
    gpt_feedback_dup = None
    filtered_movement = None


        
    print(f"✅ STEP 10: 전략 요약 저장 호출 | decision: {decision}, TP: {tp}, SL: {sl}")
    log_trade_result(
        pair=pair,
        signal=signal,
        decision=final_decision,
        score=signal_score,
        notes="\n".join(reasons) + f"\nATR: {round(atr or 0, 5)}",
        result=None,
        rsi=rsi.iloc[-1],
        macd=macd.iloc[-1],
        stoch_rsi=stoch_rsi,
        pattern=pattern,
        trend=trend,
        gpt_decision=decision,
        gpt_feedback=gpt_feedback,
        news=news,
        alert_name=alert_name,
        tp=final_tp,
        sl=final_sl,
        price=current_price,
        outcome_analysis=outcome_analysis,
        adjustment_suggestion=adjustment_suggestion,
        price_movements=price_movements,
        atr=atr,
        support=payload.get("support"),
        resistance=payload.get("resistance"),
        liquidity=payload.get("liquidity"),
        macd_signal=payload.get("macd_signal"),
        macd_trend=payload.get("macd_trend"),
        macd_signal_trend=payload.get("macd_signal_trend"),
        stoch_rsi_trend=payload.get("stoch_rsi_trend"),
        rsi_trend=payload.get("rsi_trend"),
        bollinger_upper=payload.get("bollinger_upper"),
        bollinger_lower=payload.get("bollinger_lower"),
        news_text=payload.get("news_text"),
        gpt_feedback_dup=gpt_feedback_dup,
        filtered_movement=filtered_movement,
    )
            
    #return JSONResponse(content={"status": "WAIT", "message": "GPT가 WAIT 판단"})
        
    #if is_recent_loss(pair) and recent_loss_within_cooldown(pair, window=60):
        #print(f"🚫 쿨다운 적용: 최근 {pair} 손실 후 반복 진입 차단")
        #return JSONResponse(content={"status": "COOLDOWN"})

    
    # ✅ TP/SL 값이 없을 경우 기본 설정 (15pip/10pip 기준)
    effective_decision = final_decision if final_decision in ["BUY", "SELL"] else signal
    if (final_tp is None or final_sl is None) and price is not None:
        print(f"[CHECK] TP/SL fallback 실행: final_decision={final_decision}, signal={signal}, 기존 tp={tp}, sl={sl}")
    
        pip_value = pip_value_for(pair)  # 🟦 주식/JPY/그 외 FX를 모두 인식하는 통합 함수로 교체

        tp, sl, atr_pips = calculate_realistic_tp_sl(
            price=price,
            atr=atr,
            pip_value=pip_value,
            risk_reward_ratio=1,
            min_pips=8
        )

        if final_decision == "SELL":
            # SELL이면 방향 반대로
            tp, sl = sl, tp

        gpt_feedback += f"\n⚠️ TP/SL 추출 실패 → 현실적 계산 적용 (ATR: {atr}, pips: {atr_pips})"
        final_tp, final_sl = adjust_tp_sl_for_structure(pair, price, tp, sl, support, resistance, atr)

    # 🟦 주식 신호는 TradingView Pine 전략("BUY STOCK PORTFOLIO A2")과 TP/SL을 강제로 일치시킨다.
    #    GPT가 무엇을 계산했든(또는 위 폴백이 무엇을 계산했든) 여기서 최종적으로 덮어써서,
    #    Pine: longTP = close + ATR*tpATR, longSL = close - ATR*slATR 와 100% 동일하게 만든다.
    if is_stock_pair(pair) and final_decision in ("BUY", "SELL") and price is not None and atr is not None:
        _stock_atr = float(atr.iloc[-1]) if hasattr(atr, "iloc") else float(atr)
        if _stock_atr and _stock_atr > 0:
            _digits = price_round_digits(pair)
            if final_decision == "BUY":
                final_tp = round(price + _stock_atr * STOCK_TP_ATR_MULT, _digits)
                final_sl = round(price - _stock_atr * STOCK_SL_ATR_MULT, _digits)
            else:  # SELL
                final_tp = round(price - _stock_atr * STOCK_TP_ATR_MULT, _digits)
                final_sl = round(price + _stock_atr * STOCK_SL_ATR_MULT, _digits)
            tp, sl = final_tp, final_sl  # 아래 검증 블록이 참조하는 tp/sl도 동기화
            gpt_feedback += (
                f"\n🟦 주식 TP/SL을 Pine 전략 공식으로 강제 재계산: "
                f"TP=entry±ATR*{STOCK_TP_ATR_MULT}, SL=entry∓ATR*{STOCK_SL_ATR_MULT} "
                f"(ATR={_stock_atr:.4f}) → TP={final_tp}, SL={final_sl}"
            )

    # ✅ 여기서부터 검증 블록 삽입 (FX는 기존과 동일하게 tp/sl 기준으로 계산)
    pip = pip_value_for(pair)
    min_pip = 5 * pip
    tp_sl_ratio = abs(tp - price) / max(1e-9, abs(price - sl))


    # ✅ ATR 조건 강화 (보완)
    # 🟦 절대값 0.0009는 FX(1.0~1.5 스케일) 기준이라 주식엔 적용하지 않음 (가격 스케일이 천차만별)
    last_atr = float(atr.iloc[-1]) if hasattr(atr, "iloc") else float(atr)
    if not is_stock_pair(pair) and last_atr < 0.0009:
        signal_score -= 1
        reasons.append("⚠️ ATR 낮음(0.0009↓) → 보수적 감점(-1)")

    
    result = {}
    price_movements = []
    pnl = None
    should_execute = False
    
    
    # 1️⃣ 기본 진입 조건
    # - GPT가 BUY/SELL
    # - 전략별 threshold (Balance=4.0 / Engulfing=2.5) 통과
    should_execute = (
        final_decision in ["BUY", "SELL"]
        and signal_score >= threshold
    )
    
    # 2️⃣ RSI 극단값 필터 (❗ 차단만 가능, True로 되살리지 않음)
    # 🟦 주식은 이 필터를 적용하지 않음 — Pine 전략(BUY STOCK PORTFOLIO)이 돌파/모멘텀
    #    지속(continuation) 전략이라 RSI>50만 요구하고 상한이 없음. RSI 과열을 "꼭지"로 보고
    #    차단하는 이 필터는 반전(reversal) 매매가 많은 FX용 안전장치라 주식 전략 의도와 안 맞음.
    #    FX는 기존 그대로 유지.
    if should_execute and not is_stock_pair(pair):
        if (
            (final_decision == "BUY" and rsi.iloc[-1] > 85)
            or (final_decision == "SELL" and rsi.iloc[-1] < 20)
        ):
            reasons.append(
                f"❌ RSI 극단값으로 진입 차단: {final_decision} @ RSI {rsi.iloc[-1]:.2f}"
            )
            should_execute = False
    
    # 3️⃣ (선택) ATR 보수 필터 – 이미 점수에 반영했으므로 여기선 추가 차단 안 함
    # if should_execute and last_atr < 0.0009:
    #     reasons.append("❌ ATR 너무 낮음 → 진입 차단")
    #     should_execute = False
    
    # 4️⃣ 디버그 로그 (강력 추천)
    print(
        f"[EXEC CHECK] decision={final_decision}, "
        f"score={signal_score:.2f}, threshold={threshold}, "
        f"execute={should_execute}"
    )
    if should_execute:
        pair_for_order = pair.replace("/", "_")
    
        # ✅ (추가) 이미 열린 트레이드가 있으면 신규 진입 스킵 (FIFO 방지)
        opened, cnt = has_open_trade(pair_for_order)
        if opened:
            print(f"[SKIP] {pair_for_order} openTrades={cnt} → FIFO 방지로 신규진입 스킵")
            should_execute = False
    
    if should_execute:
        if is_stock_pair(pair_for_order):
            # 🟦 주식: 실제 수량은 place_order_alpaca 내부에서 고정금액(ALPACA_FIXED_NOTIONAL_USD)
            #         ÷ 현재가로 산출되므로, 여기서는 매수/매도 방향만 표시
            units = 1 if final_decision == "BUY" else -1
            digits = price_round_digits(pair_for_order)
        else:
            units = 100000 if final_decision == "BUY" else -100000
            digits = 3 if pair.endswith("JPY") else 5
    
        print(f"[DEBUG] WILL PLACE ORDER → pair={pair}, side={final_decision}, units={units}, "
              f"price={price}, tp={final_tp}, sl={final_sl}, digits={digits}, score={signal_score}")
    
        result = place_order(pair_for_order, units, final_tp, final_sl, digits, price=price)
    else:
        print(f"[DEBUG] SKIP ORDER → should_execute={should_execute}, decision={final_decision}, score={signal_score}")
        result = {"status": "skipped"}
    
    executed_time = datetime.utcnow()
    candles_post = get_candles(pair, base_granularity_for(pair), 8)
    price_movements = candles_post[["high", "low"]].to_dict("records")

    if final_decision in ("BUY", "SELL") and isinstance(result, dict) and result.get("status") == "order_placed":

        print("[DEBUG] ORDER RESULT:", result)
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

    # 🟦 버그 수정: 이 함수가 끝까지 정상 처리됐을 때 명시적인 return이 없어서
    #    FastAPI가 암묵적으로 None을 받아 응답 바디가 그냥 "null"이 되고 있었음.
    #    (크래시는 아니었지만, 응답 내용이 비어있는 건 깔끔하지 않으므로 명확히 반환)
    #    signal_score 등이 numpy 타입(float64)일 수 있어 JSONResponse의 json.dumps가
    #    실패할 수 있으므로 안전하게 캐스팅.
    try:
        safe_score = float(signal_score) if signal_score is not None else None
    except Exception:
        safe_score = None
    return JSONResponse(content={
        "status": "processed",
        "pair": str(pair) if pair is not None else None,
        "decision": str(final_decision) if final_decision is not None else None,
        "score": safe_score,
    })


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
def get_multi_tf_scalping_data(pair):
    """
    단타 분석을 위한 MTF 캔들 + 보조지표 추세 리스트 수집.
    진입 타임프레임은 base_granularity_for(pair) — FX는 M30, 주식은 M15. H1(보조 흐름), H4(큰 흐름)는 공통.
    🟦 3개 타임프레임 캔들 조회를 순차 대신 병렬로 실행해서 대기 시간을 줄인다(네트워크 왕복 3번→1번 분량).
    """
    base_tf = base_granularity_for(pair)

    timeframes = {
        base_tf: 100,
        'H1': 100,
        'H4': 60
    }

    tf_data = {}

    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = {tf: ex.submit(get_candles, pair, tf, count) for tf, count in timeframes.items()}
        fetched = {tf: f.result() for tf, f in futures.items()}

    for tf, candles in fetched.items():
        if candles is None or candles.empty:
            continue

        df = candles.copy()
        try:
            # 보조지표 계산
            df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
            macd = ta.trend.MACD(close=df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['stoch_rsi'] = ta.momentum.StochRSIIndicator(close=df['close'], window=14).stochrsi()

            # 최근 14개 (H4는 10개) 보조지표 리스트 저장
            n = 14 if tf in [base_tf, 'H1'] else 10
            tf_data[tf] = {
                'rsi_trend': df['rsi'].dropna().iloc[-n:].tolist(),
                'macd_trend': df['macd'].dropna().iloc[-n:].tolist(),
                'macd_signal_trend': df['macd_signal'].dropna().iloc[-n:].tolist(),
                'stoch_rsi_trend': df['stoch_rsi'].dropna().iloc[-n:].tolist()
            }

        except Exception as e:
            print(f"[{tf}] 보조지표 계산 오류:", e)
            continue

    return tf_data
    
def summarize_mtf_indicators(mtf_data):
    summary = {}  # ✅ 문자열 리스트 → 딕셔너리로 변경

    for tf, data in mtf_data.items():
        if not data:
            continue

        summary[tf] = {
            "rsi_trend": data.get('rsi_trend', []),
            "macd_trend": data.get('macd_trend', []),
            "macd_signal_trend": data.get('macd_signal_trend', []),
            "stoch_rsi_trend": data.get('stoch_rsi_trend', [])
        }

    return summary  # ✅ 문자열이 아닌 JSON 딕셔너리 그대로 반환

_ALPACA_GRANULARITY_MAP = {
    "M1": "1Min",
    "M5": "5Min",
    "M15": "15Min",
    "M30": "30Min",
    "H1": "1Hour",
    "H4": "4Hour",
    "D": "1Day",
}


_ALPACA_BARS_PER_TRADING_DAY = {
    "1Min": 390, "5Min": 78, "15Min": 26, "30Min": 13, "1Hour": 7, "4Hour": 2, "1Day": 1
}


def get_alpaca_candles(symbol, granularity, count):
    """Alpaca Market Data API에서 주식 캔들(바)을 가져와 OANDA 캔들과 동일한 포맷의 DataFrame으로 반환."""
    timeframe = _ALPACA_GRANULARITY_MAP.get(granularity, "30Min")

    # 🟦 start를 안 주면 Alpaca가 충분히 과거로 안 거슬러가고 "오늘 일부만" 주는 경우가 있어서,
    #    count(예:200)를 채우기에 충분한 만큼 명시적으로 start를 과거로 잡아줌(데이터 없음 방지용 하한선).
    bars_per_day = _ALPACA_BARS_PER_TRADING_DAY.get(timeframe, 26)
    needed_trading_days = max(5, (count // max(1, bars_per_day)) + 5)
    # 주말/공휴일 버퍼로 1.6배 캘린더일로 환산
    start_dt = datetime.utcnow() - timedelta(days=int(needed_trading_days * 1.6))

    url = f"{ALPACA_DATA_BASE_URL}/v2/stocks/{symbol}/bars"
    params = {
        "timeframe": timeframe,
        "limit": count,
        "adjustment": "raw",
        "feed": "iex",  # 무료 플랜 기준. 유료(SIP) 사용 시 'sip'로 변경
        "start": start_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        # 🟦 핵심 수정: sort를 안 주면(기본 asc) start부터 "오래된 것부터" limit개를 채워서,
        #    조회 구간에 limit보다 많은 바가 있으면 최근 데이터가 통째로 잘려나간다
        #    (이번 GEV 사례: 19일치 중 200개를 과거부터 채우니 최근 6~7거래일이 누락되어,
        #     일주일 전 가격(975~990)이 "최신 캔들"로 둔갑함).
        #    desc로 최신 것부터 limit개를 받은 뒤 아래에서 시간순으로 다시 뒤집는다.
        "sort": "desc",
    }
    try:
        r = requests.get(url, headers=ALPACA_HEADERS, params=params, timeout=15)
        r.raise_for_status()
        bars = r.json().get("bars", [])
    except Exception as e:
        print(f"❗ [Alpaca] {symbol} 캔들 요청 실패: {e}")
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

    if not bars:
        print(f"❗ [Alpaca] {symbol} 캔들 데이터 없음 (start={params['start']})")
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

    # desc로 받았으니 시간 오름차순으로 뒤집어서, candles.iloc[-1]이 항상 "가장 최근" 캔들이 되게 한다.
    bars = list(reversed(bars))

    print(f"📊 [Alpaca] {symbol} {timeframe} 캔들 {len(bars)}개 수신 "
          f"(최근: {bars[-1].get('t')}, 가장 오래된: {bars[0].get('t')})")

    return pd.DataFrame([
        {
            "time": b.get("t"),
            "open": float(b["o"]),
            "high": float(b["h"]),
            "low": float(b["l"]),
            "close": float(b["c"]),
            "volume": b.get("v", 0),
        }
        for b in bars
    ])


def get_candles(pair, granularity, count):
    # 🟦 주식 심볼이면 Alpaca 데이터로 분기
    if is_stock_pair(pair):
        return get_alpaca_candles(pair, granularity, count)

    url = f"https://api-fxpractice.oanda.com/v3/instruments/{pair}/candles"
    headers = {"Authorization": f"Bearer {OANDA_API_KEY}"}
    params = {"granularity": granularity, "count": count, "price": "M"}
    
    try:
        r = requests.get(url, headers=headers, params=params)
        r.raise_for_status()
        candles = r.json().get("candles", [])
    except Exception as e:
        print(f"❗ 캔들 요청 실패: {e}")
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

    if not candles:
        print(f"❗ {pair} 캔들 데이터 없음")
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

def get_ohlcv(pair, interval="30m", limit=100):
    """
    get_multi_timeframe_context() 등에서 쓰기 위한 호환 래퍼.
    interval 문자열(예: 5m, 30m, 4h)을 OANDA granularity로 변환해서
    기존 get_candles()를 호출한다.
    """
    interval_map = {
        "5m": "M5",
        "15m": "M15",
        "30m": "M30",
        "1h": "H1",
        "4h": "H4",
        "1d": "D",
    }

    granularity = interval_map.get(str(interval).lower())
    if not granularity:
        raise ValueError(f"지원하지 않는 interval: {interval}")

    return get_candles(pair, granularity, limit)

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
    
def detect_box_breakout(candles, pair, box_window=10, box_threshold_pips=None):
    """
    박스권 돌파 감지 (통합/동적 임계치 버전)
    - box_threshold_pips가 None이면 ATR 기반으로 동적으로 결정
    - 🟦 주식은 pip 환산을 거치지 않고 '달러 단위'로 직접 비교 (가독성/정확도 개선).
      예: TSLA ATR=10 → box_threshold_pips=266.67pip(=$8) 같은 우회 계산 대신 바로 $8.0 사용.
    """
    if candles is None or candles.empty:
        return {"in_box": False, "breakout": None}

    # ATR 기반 임계치 계산
    atr_series = calculate_atr(candles)
    last_atr = float(atr_series.dropna().iloc[-1]) if not atr_series.dropna().empty else 0.0

    recent = candles.tail(box_window)
    high_max = recent["high"].max()
    low_min  = recent["low"].min()

    if is_stock_pair(pair) and box_threshold_pips is None:
        # 🟦 주식: 달러 단위로 직접 비교 (dynamic_thresholds의 box_threshold_pips(주식) * pip_value와 동일한 비율)
        box_range_dollars = high_max - low_min
        box_threshold_dollars = max(last_atr * 0.8, 0.12)  # 최소 12센트 하한(저ATR 종목 안전장치)
        if box_range_dollars > box_threshold_dollars:
            return {"in_box": False, "breakout": None}
    else:
        # ===== 기존 FX 로직 (변경 없음) =====
        thr = dynamic_thresholds(pair, last_atr)
        if box_threshold_pips is None:
            box_threshold_pips = thr["box_threshold_pips"]
        pv = thr["pip_value"]  # pip 크기(USDJPY=0.01, 그 외=0.0001)
        box_range_pips = (high_max - low_min) / pv
        if box_range_pips > box_threshold_pips:
            return {"in_box": False, "breakout": None}

    last_close = recent["close"].iloc[-1]

    if last_close > high_max:
        return {"in_box": True, "breakout": "UP"}
    elif last_close < low_min:
        return {"in_box": True, "breakout": "DOWN"}
    else:
        return {"in_box": True, "breakout": None}
# === 교체 끝 ===

def detect_trend(candles, rsi, mid_band, pair=None):
    close = candles["close"]
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()

    gap = abs(ema20.iloc[-1] - ema50.iloc[-1])

    # 🟦 주식: 고정 0.05달러는 가격대(예: TSLA $300)에서 의미가 없으므로 ATR 비례로 판정
    if pair and is_stock_pair(pair):
        try:
            atr_series = calculate_atr(candles)
            last_atr = float(atr_series.dropna().iloc[-1]) if not atr_series.dropna().empty else 0.0
        except Exception:
            last_atr = 0.0
        neutral_threshold = (last_atr * 0.10) if last_atr > 0 else 0.05
        if gap < neutral_threshold:
            return "NEUTRAL"
    else:
        # ===== 기존 FX 로직 (변경 없음, JPY 기준 튜닝값) =====
        if gap < 0.05:   # 필요시 0.03~0.08로 조정
            return "NEUTRAL"

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
            reasons.append("✅ 강한 장대양봉 → 매수 심리 강화 가점 +1")
        elif last['close'] < last['open'] and signal == "SELL":
            score += 1
            reasons.append("✅ 강한 장대음봉 → 매도 심리 강화 가점 +1")

    # ② 꼬리 비율 심리
    if lower_wick > 2 * body and signal == "BUY":
        score += 1
        reasons.append("✅ 아래꼬리 길다 → 매수 지지 심리 강화 가점+1")
    if upper_wick > 2 * body and signal == "SELL":
        score += 1
        reasons.append("✅ 위꼬리 길다 → 매도 압력 심리 강화 가점+1")

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
    # 🟦 주식은 "통화코드" 개념이 없어서(ForexFactory류 경제지표 뉴스는 FX 전용) 매칭 대상이 없음.
    #    pair.split("_")[1] 같은 FX 전용 파싱이 'NVDA'처럼 '_' 없는 티커에서 IndexError를 내던 부분 수정.
    if is_stock_pair(pair):
        return []

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

def has_open_position_alpaca(symbol: str) -> tuple[bool, int]:
    """
    Alpaca 계좌에 해당 심볼의 열린 포지션이 있는지 확인.
    return: (열려있음 여부, 1 또는 0 / 조회실패시 -1)
    """
    url = f"{ALPACA_TRADE_BASE_URL}/v2/positions/{symbol}"
    try:
        r = requests.get(url, headers=ALPACA_HEADERS, timeout=10)
        if r.status_code == 200:
            return True, 1
        if r.status_code == 404:
            return False, 0
        print(f"[Alpaca] 포지션 조회 status={r.status_code} body={r.text}")
        return True, -1  # 애매하면 보수적으로 진입 차단
    except Exception as e:
        print("[Alpaca] 포지션 조회 실패:", e)
        return True, -1


def has_open_trade(pair_for_order: str) -> tuple[bool, int]:
    """
    pair_for_order: FX는 'USD_JPY' 형태, 주식은 'TSLA' 형태
    return: (열려있음 여부, 해당 종목 open 포지션/트레이드 개수)
    """
    # 🟦 주식이면 Alpaca 포지션 조회로 분기
    if is_stock_pair(pair_for_order):
        return has_open_position_alpaca(pair_for_order)

    url = f"https://api-fxpractice.oanda.com/v3/accounts/{ACCOUNT_ID}/openTrades"
    headers = {
        "Authorization": f"Bearer {OANDA_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        r = requests.get(url, headers=headers, timeout=10)
        j = r.json() if r.ok else {}
        trades = j.get("trades", []) if isinstance(j, dict) else []

        cnt = 0
        for t in trades:
            if t.get("instrument") == pair_for_order:
                cnt += 1

        return (cnt > 0), cnt

    except Exception as e:
        # 조회 실패 시엔 보수적으로 "진입 막기"가 안전
        print("[OANDA] openTrades check failed:", e)
        return True, -1


def get_alpaca_account_equity():
    """Alpaca 계좌의 현재 equity(자산)를 조회. 실패 시 None."""
    url = f"{ALPACA_TRADE_BASE_URL}/v2/account"
    try:
        r = requests.get(url, headers=ALPACA_HEADERS, timeout=10)
        r.raise_for_status()
        j = r.json()
        return float(j["equity"])
    except Exception as e:
        print("[Alpaca] 계좌 조회 실패:", e)
        return None


def get_tiered_qty(price: float) -> int:
    """
    가격대별 고정 수량표.
    - $300 이상: 1주   (400불 이상도 동일하게 1주로 처리됨)
    - $200~300 미만: 2주
    - $100~200 미만: 3주
    - $100 미만: 5주
    필요하면 숫자만 바꿔서 쉽게 조정 가능.
    """
    if price >= 300:
        return 1
    elif price >= 200:
        return 2
    elif price >= 100:
        return 3
    else:
        return 5


def calc_alpaca_qty(ref_price: float, sl: float, notional_usd: float) -> int:
    """
    포지션 수량(qty) 산출. ALPACA_SIZING_MODE로 방식 선택:
    - "tiered": 가격대별 고정 수량표(get_tiered_qty) 사용. 가장 단순하고 예측 가능.
    - "risk": 계좌 equity * ALPACA_RISK_PCT(%) 만큼만 손실을 허용한다고 가정,
      SL까지의 거리(stop_distance)로 나눠 수량을 역산. (예: 계좌 5만달러, 리스크 0.5% → 250달러,
      SL 거리가 5달러면 qty=50주)
    - SL 거리가 너무 좁아 비정상적으로 큰 수량이 나오는 걸 막기 위해 ALPACA_MAX_NOTIONAL_USD로 캡.
    - equity 조회 실패/SL 거리 0 등 예외 상황에는 고정금액(ALPACA_FIXED_NOTIONAL_USD) 방식으로 폴백.
    """
    try:
        ref_price = float(ref_price)
    except Exception:
        return 1
    if ref_price <= 0:
        return 1

    max_qty_by_notional = max(1, int(ALPACA_MAX_NOTIONAL_USD // ref_price))

    if ALPACA_SIZING_MODE == "tiered":
        qty = get_tiered_qty(ref_price)
        print(f"[Alpaca][tiered-sizing] price={ref_price}, qty={qty}")
        return qty

    if ALPACA_SIZING_MODE == "risk":
        equity = get_alpaca_account_equity()
        try:
            stop_distance = abs(ref_price - float(sl))
        except Exception:
            stop_distance = 0.0

        if equity and stop_distance > 0:
            risk_dollars = equity * (ALPACA_RISK_PCT / 100.0)
            qty_by_risk = int(risk_dollars // stop_distance)
            print(f"[Alpaca][risk-sizing] equity={equity}, risk%={ALPACA_RISK_PCT}, "
                  f"risk$={risk_dollars:.2f}, stop_distance={stop_distance:.4f}, "
                  f"qty_by_risk={qty_by_risk}, cap_by_notional={max_qty_by_notional}")
            return max(1, min(qty_by_risk, max_qty_by_notional))

        print("[Alpaca][risk-sizing] equity 조회 실패 또는 stop_distance=0 → 고정금액 방식으로 폴백")

    # fixed 모드 또는 risk 계산 실패시 폴백
    qty_by_fixed = int(notional_usd // ref_price)
    return max(1, min(qty_by_fixed, max_qty_by_notional))


def get_alpaca_latest_price(symbol):
    """Alpaca 최신 체결가(latest trade) 조회. 실패 시 None."""
    url = f"{ALPACA_DATA_BASE_URL}/v2/stocks/{symbol}/trades/latest"
    params = {"feed": "iex"}
    try:
        r = requests.get(url, headers=ALPACA_HEADERS, params=params, timeout=10)
        r.raise_for_status()
        return float(r.json()["trade"]["p"])
    except Exception as e:
        print(f"[Alpaca] {symbol} 최신가 조회 실패: {e}")
        return None


def place_order_alpaca(symbol, side, notional_usd, ref_price, tp, sl, digits=2):
    """
    Alpaca Bracket Order로 시장가 진입 + TP/SL 동시 설정.
    수량(qty)은 calc_alpaca_qty()에서 산출 (기본: 계좌 리스크% 기반, ALPACA_SIZING_MODE로 전환 가능)
    side: "BUY" 또는 "SELL"

    🟦 알림 발사 시점 가격(ref_price)과 실제 주문 시점 가격 사이에 시차로 인한 괴리가 생기면
       (GPT 분석 등으로 수 초~수십 초 지연), TP/SL이 실시간가 기준으로 무효(예: BUY인데
       TP가 현재가보다 낮음)가 되어 Alpaca가 422로 거부하는 경우가 있었음.
       → 주문 직전 최신가를 다시 조회해서, TP/SL을 "원래 의도했던 거리"만큼 그대로 이동시켜
         항상 실시간가 기준으로 유효하게 만든다.
    """
    fresh_price = get_alpaca_latest_price(symbol)
    if fresh_price and ref_price:
        try:
            delta = fresh_price - float(ref_price)
            gap_pct = abs(delta) / float(ref_price) * 100 if float(ref_price) else 0.0
        except Exception:
            delta = 0.0
            gap_pct = 0.0

        # 🟦 신호가 vs 실시간가 차이가 비정상적으로 크면(예: 알림 자체가 묵혀있다가 늦게 도착한 경우),
        #    TP/SL을 억지로 끼워맞춰 체결시키는 대신 그냥 스킵한다 — 신호가 더 이상 신뢰할 수 없기 때문.
        if gap_pct > ALPACA_MAX_PRICE_GAP_PCT:
            print(f"⛔ [Alpaca] {symbol} 가격 갱신 폭({gap_pct:.2f}%)이 한도({ALPACA_MAX_PRICE_GAP_PCT}%) 초과 "
                  f"(신호가={ref_price} → 실시간가={fresh_price}) → 주문 스킵 (신호 신뢰 불가)")
            return {
                "status": "skipped",
                "reason": f"price_gap_{gap_pct:.2f}pct_exceeds_{ALPACA_MAX_PRICE_GAP_PCT}pct",
                "ref_price": ref_price,
                "fresh_price": fresh_price,
            }

        if delta:
            print(f"[Alpaca] {symbol} 가격 갱신: 신호가={ref_price} → 실시간가={fresh_price} "
                  f"(Δ{delta:+.4f}, {gap_pct:.2f}%) — TP/SL을 동일 거리만큼 이동")
            tp = tp + delta
            sl = sl + delta
            ref_price = fresh_price

    qty = calc_alpaca_qty(ref_price, sl, notional_usd)

    url = f"{ALPACA_TRADE_BASE_URL}/v2/orders"
    headers = {**ALPACA_HEADERS, "Content-Type": "application/json"}

    data = {
        "symbol": symbol,
        "qty": str(qty),
        "side": "buy" if side == "BUY" else "sell",
        "type": "market",
        "time_in_force": "day",
        "order_class": "bracket",
        "take_profit": {"limit_price": str(round(tp, digits))},
        "stop_loss": {"stop_price": str(round(sl, digits))},
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=15)
        try:
            j = response.json()
        except Exception:
            j = {"raw_text": response.text}

        print(f"[Alpaca] status_code={response.status_code}")
        print(f"[Alpaca] body={j}")

        if 200 <= response.status_code < 300:
            return {
                "status": "order_placed",
                "status_code": response.status_code,
                "raw": j,
                "qty": qty,
            }
        else:
            return {
                "status": "error",
                "status_code": response.status_code,
                "raw": j,
                "qty": qty,
            }

    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": str(e)}


def place_order(pair, units, tp, sl, digits, price=None):
    # 🟦 주식이면 Alpaca Bracket Order로 분기
    if is_stock_pair(pair):
        side = "BUY" if units > 0 else "SELL"
        # 🟦 전역 _last_price_cache는 동시에 들어오는 다른 요청이 같은 종목 캐시를 덮어쓸 수 있어서
        #    (예: GPT 분석 도는 몇 초 사이 같은 종목 알림이 또 들어오면 엉뚱한 가격이 섞임)
        #    이 요청 자체의 price를 우선 사용. price가 안 넘어온 경우에만 캐시로 폴백.
        ref_price = price if price is not None else (_last_price_cache.get(pair) or tp)
        return place_order_alpaca(
            pair, side, ALPACA_FIXED_NOTIONAL_USD, ref_price, tp, sl,
            digits=price_round_digits(pair)
        )

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
        response = requests.post(url, headers=headers, json=data, timeout=15)

        # ✅ 성공/실패와 무관하게 바디를 먼저 읽는다 (취소/거절 사유가 여기 들어있음)
        try:
            j = response.json()
        except Exception:
            j = {"raw_text": response.text}

        print(f"[OANDA] status_code={response.status_code}")
        print(f"[OANDA] body={j}")
    
        # ✅ (추가) 캔슬/리젝트 이유 요약 출력
        if isinstance(j, dict):
            cancel_tx = j.get("orderCancelTransaction") or {}
            reject_tx = j.get("orderRejectTransaction") or {}
            create_tx = j.get("orderCreateTransaction") or {}
        else:
            cancel_tx, reject_tx, create_tx = {}, {}, {}
    
        if cancel_tx:
            print(
                "[OANDA] cancel_reason =", cancel_tx.get("reason"),
                "| canceled_order_id =", cancel_tx.get("orderID"),
                "| cancel_id =", cancel_tx.get("id"),
            )
    
        if reject_tx:
            print(
                "[OANDA] reject_reason =", reject_tx.get("rejectReason"),
                "| rejected_order_id =", reject_tx.get("orderID"),
                "| reject_id =", reject_tx.get("id"),
            )
    
        if create_tx:
            print(
                "[OANDA] created_order_id =", create_tx.get("id"),
                "| instrument =", create_tx.get("instrument"),
                "| units =", create_tx.get("units"),
                "| timeInForce =", create_tx.get("timeInForce"),
            )

        # ✅ 성공 판단은 status_code로
        if 200 <= response.status_code < 300:
            return {
                "status": "order_placed",
                "status_code": response.status_code,
                "raw": j
            }
        else:
            # 실패여도 raw를 남겨야 reason 확인 가능
            return {
                "status": "error",
                "status_code": response.status_code,
                "raw": j
            }

    except requests.exceptions.RequestException as e:
        # 네트워크/타임아웃 등 진짜 요청 실패
        return {"status": "error", "message": str(e)}


def extract_json_block(text: str):
    """
    응답에서 '마지막 JSON 덩어리'를 안전하게 추출한다.
    - ```json/``` 같은 코드블록 표식 제거
    - 맨 마지막 { ... } 후보부터 파싱 시도
    - 실패하면 중괄호 스택으로 모든 덩어리 역순 시도
    """
    if not text:
        return None

    # 1) 코드블록/표식 제거
    cleaned = (
        str(text)
        .replace("```json", "")
        .replace("```JSON", "")
        .replace("```", "")
        .strip()
    )

    # 2) '마지막 { ... }' 구간 먼저 시도
    last_open = cleaned.rfind("{")
    last_close = cleaned.rfind("}")
    if last_open != -1 and last_close != -1 and last_close > last_open:
        candidate = cleaned[last_open:last_close + 1]
        try:
            return json.loads(candidate)
        except Exception as e:
            print(f"[WARN] JSON 파싱 실패(마지막 블록): {e} | cand[:200]={candidate[:200]}")

    # 3) 중괄호 매칭 스택으로 모든 후보 역순 시도
    stack = []
    spans = []
    for i, ch in enumerate(cleaned):
        if ch == "{":
            stack.append(i)
        elif ch == "}" and stack:
            start = stack.pop()
            spans.append((start, i + 1))

    for start, end in reversed(spans):
        s = cleaned[start:end]
        try:
            return json.loads(s)
        except Exception:
            continue

    return None


def parse_gpt_feedback(text):
    import re
    print(f"[DEBUG] 함수 진입 - 입력 텍스트:\n{text[:300]}") 
    final_decision = "WAIT"
    tp = None
    sl = None

    try:
        data = extract_json_block(text)
        print(f"[TRACE] Extracted JSON block: {data}")
        if isinstance(data, dict):  # ✅ dict인지 확인
            final_decision = str(data.get("decision", "WAIT")).upper()
            tp = safe_float(data.get("tp"))
            sl = safe_float(data.get("sl"))
            print(f"[DEBUG] JSON 추출 성공: decision={final_decision}, tp={tp}, sl={sl}")
            print(f"[TRACE] 최종 판단 결과: final_decision={final_decision}, tp={tp}, sl={sl}")  # ← 추가
            # ⛔️ 파싱 실패 시 강제 초기화
            if final_decision not in ["BUY", "SELL"]:
                final_decision = "WAIT"
                tp = None
                sl = None
            
            return final_decision, tp, sl

    except Exception as e:
        print(f"[WARN] JSON 파싱 실패: {e}, fallback 실행")
    
        # fallback 조건: 기존 판단이 없을 때만 덮어씀
        if final_decision != "WAIT" and (tp is not None and sl is not None):
            print("[INFO] fallback 진입했지만 기존 결정 BUY/SELL 유지함")
            return final_decision, tp, sl
        else:
            print("[INFO] fallback 조건 충족 → WAIT 처리")
            final_decision = "WAIT"
            tp = None
            sl = None
            return final_decision, tp, sl


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
    if final_decision == "WAIT":
        upper_text = text.upper()
        buy_score = upper_text.count("BUY")
        sell_score = upper_text.count("SELL")
    
        if buy_score > sell_score:
            final_decision = "BUY"
        elif sell_score > buy_score:
            final_decision = "SELL"

    # ✅ TP/SL 추출 (가장 마지막 숫자 사용)
    lines = text.splitlines()
    tp_line = next((ln for ln in reversed(lines) if re.search(r'(?i)\bTP\b|TP 제안 값|목표', ln)), "")
    sl_line = next((ln for ln in reversed(lines) if re.search(r'(?i)\bSL\b', ln) and re.search(r'\d+\.\d+', ln)), "")
    print(f"[DEBUG] TP 라인 추출: {tp_line}")
    print(f"[DEBUG] SL 라인 추출: {sl_line}")
    
    # 🛠️ 추가: SL/TP 라벨이 없지만, BUY/SELL 줄 바로 아래 숫자만 있는 경우 커버
    if not tp_line or not sl_line:
        for i, line in enumerate(lines):
            if re.search(r'\b(BUY|SELL)\b', line, re.I):
                # 다음 줄에 가격 숫자만 있을 경우 TP/SL로 추정
                if i+1 < len(lines) and re.search(r'\d+\.\d+', lines[i+1]):
                    price = lines[i+1]
                    if not tp_line:
                        tp_line = price
                    elif not sl_line:
                        sl_line = price

    
    if not sl_line:
        sl = None  # 결정은 유지
    # 아래처럼 결정 추출을 더 확실하게:
    m = re.search(r"진입판단\s*[:：]?\s*(BUY|SELL|WAIT)", text.upper())
    print(f"[TRACE] 정규식 보조 판단 결과: m={m}, decision={(m.group(1) if m else 'None')}")
    if m: 
        decision = m.group(1)
        final_decision = decision 
    print(f"[TRACE] ✅ 최종 결정 결과: final_decision={final_decision}, tp={tp}, sl={sl}")
    # TP/SL 숫자 인식도 유연화:
    def pick_price(line):
        nums = re.findall(r"\d{1,2}\.\d{3,5}", line)
        return float(nums[-1]) if nums else None


    def extract_last_price(line):
        nums = re.findall(r"\b\d{1,5}\.\d{1,5}\b", line)
        return float(nums[-1]) if nums else None


    return final_decision, tp, sl
    print(f"[DEBUG] 최종 결정 리턴: final_decision={final_decision}, tp={tp}, sl={sl}")





 # === TP/SL 구조·ATR 보정 ===
def adjust_tp_sl_for_structure(pair, entry, tp, sl, support, resistance, atr):
    if entry is None or tp is None or sl is None:
        return tp, sl
    pip = pip_value_for(pair)
    min_dist = 8 * pip  # 최소 8pip
    is_buy  = tp > entry and sl < entry
    is_sell = tp < entry and sl > entry

    # 구조 클램핑
    if is_buy:
        if resistance is not None:
            tp = min(tp, resistance + 5 * pip)
        if support is not None:
            sl = max(sl, support - 5 * pip)
    elif is_sell:
        if support is not None:
            tp = max(tp, support - 5 * pip)
        if resistance is not None:
            sl = min(sl, resistance + 5 * pip)

    # 최소 거리 확보
    if is_buy:
        tp = max(tp, entry + min_dist)
        sl = min(sl, entry - min_dist)
    elif is_sell:
        tp = min(tp, entry - min_dist)
        sl = max(sl, entry + min_dist)

    # RR ≥ 1.8 강제
    if is_buy and (entry - sl) > 0:
        desired_tp = entry + 1.8 * (entry - sl)
        tp = max(tp, desired_tp)
    if is_sell and (sl - entry) > 0:
        desired_tp = entry - 1.8 * (sl - entry)
        tp = min(tp, desired_tp)

    # ATR 과욕 방지(±1.5*ATR)
    if atr and float(atr) > 0:
        span = 1.5 * float(atr)
        if is_buy:
            tp = min(tp, entry + span)
            sl = max(sl, entry - span)
        elif is_sell:
            tp = max(tp, entry - span)
            sl = min(sl, entry + span)

    digits = price_round_digits(pair)
    return round(tp, digits), round(sl, digits)   
def analyze_with_gpt(payload, current_price, pair, candles, base64_image=None):
    try:
        mtf_info = get_multi_timeframe_context(pair)
    except Exception as e:
        print(f"❌ MTF 정보 생성 실패: {e}")
        mtf_info = "MTF 정보 없음"
    global _gpt_cooldown_until, _gpt_last_ts
    dbg("gpt.enter", t=int(_t.time()*1000))
    #✅ 거래 시간대 필터 추가
    from datetime import datetime
    from zoneinfo import ZoneInfo
    
    # ==========================================
    # 거래 제한 시간 필터 (Atlanta 기준)
    # ==========================================
    
    now_atlanta = datetime.now(ZoneInfo("America/New_York"))
    
    atlanta_hour = now_atlanta.hour
    weekday = now_atlanta.weekday()
    
    is_restricted = False
    restriction_reason = ""
    
    # ==========================================
    # 🔴 롤오버 시간
    # ==========================================
    
    if 17 <= atlanta_hour < 18:
    
        is_restricted = True
    
        restriction_reason = (
            "🔴 롤오버 시간 → 스프레드 확대 위험"
        )
    
    # ==========================================
    # 🔴 일요일 FX 오픈 직후
    # ==========================================
    
    elif weekday == 6 and atlanta_hour >= 17:
    
        is_restricted = True
    
        restriction_reason = (
            "🔴 일요일 FX 오픈 직후 → 갭 및 유동성 위험"
        )
    
    # ==========================================
    # 🔴 금요일 오후 (선택)
    # ==========================================
    
    elif weekday == 4 and atlanta_hour >= 15:
    
        is_restricted = True
    
        restriction_reason = (
            "🔴 금요일 오후 → 청산 및 변동성 위험"
        )
    
    # ==========================================
    # 거래 제한
    # ==========================================
    
    if is_restricted:
    
        print(f"⛔ 거래 제한: {restriction_reason}")
    
        return (
            f"⛔ 거래 제한: {restriction_reason}"
        )
        
    # ── 전역 쿨다운: 429 맞은 뒤 일정 시간은 호출 자체 스킵 ──
    global _gpt_cooldown_until
    now = _t.time()
    if now < _gpt_cooldown_until:
        dbg("gpt.skip.cooldown", wait=round(_gpt_cooldown_until - now, 2))
        return "GPT 응답 없음(쿨다운)"
    gpt_rate_gate()  # 3-b: 계정 단위 슬롯 대기
    headers = OPENAI_HEADERS
    score = payload.get("score", 0)
    signal_score = payload.get("signal_score", 0)
    recent_candle_summary = summarize_recent_candle_flow(candles)
    reasons = payload.get("reasons", [])
    recent_rsi_values = payload.get("recent_rsi_values", [])
    recent_macd_values = payload.get("recent_macd_values", [])
    recent_stoch_rsi_values = payload.get("recent_stoch_rsi_values", [])
    macd_signal = payload.get("macd_signal", None)
    rsi_trend = payload.get("rsi_trend", [])
    macd_trend = payload.get("macd_trend", [])
    stoch_rsi_trend = payload.get("stoch_rsi_trend", [])
    support     = payload.get("support", current_price)
    resistance  = payload.get("resistance", current_price)
    boll_up     = payload.get("bollinger_upper", current_price)
    boll_low    = payload.get("bollinger_lower", current_price)
    mtf_indicators = get_multi_tf_scalping_data(pair)
    mtf_summary_dict = summarize_mtf_indicators(mtf_indicators)
    mtf_summary = json.dumps(mtf_summary_dict, ensure_ascii=False, indent=2)
    print("✅ 테스트 출력: ", mtf_summary)
        
    # 1. GPT에게 보낼 콘텐츠 리스트 생성 (텍스트와 이미지를 분리해서 담기)
    user_content = [
        {
            "type": "input_text", 
            "text": f"데이터 분석 보고: {json.dumps(payload, ensure_ascii=False)}"
        }
    ]
    
    # 2. 사진(base64_image)이 있다면 리스트에 추가
    if base64_image:
        user_content.append({
            "type": "input_image",
            "image_url": {
                "url": f"data:image/png;base64,{base64_image}",
                "detail": "high"
            }
        })
    
    # 3. 전체 메시지 구조 구성
    messages = [
        {
            "role": "system",
            "content": (
                "너는 실전 FX 트레이딩 전략 조력자야.\n\n"
                "⚠️ [역할 정의 - 매우 중요]\n"
                "- 이미 이 신호는 사전 score / signal_score 필터를 통과했다.\n"
                "- 그러나 GPT는 필터 결과를 맹신하지 말고 현재 차트 구조를 독립적으로 검증해야 한다,\n"
                "  승률이 55% 미만으로 판단되면 WAIT을 선택할 수 있다.명백한 반대 시그널 뿐 아니라추세 부재, 모멘텀 부재, 박스권 상단/하단 정체도 WAIT 근거가 될 수 있다.\n"
                "- 애매함, 가능성, 추측만으로 WAIT을 선택해서는 안 된다.\n\n"
                
                f"📌 [{base_granularity_for(pair)} 알림 전용: 멀티 타임프레임 분석 지침 - 추가됨]\n"
                f"현재 알림은 {base_granularity_for(pair)}에서 발생했습니다. 아래 상위/하위 맥락을 반드시 참고하세요:\n"
                f"{mtf_info}\n"
                "- H4 추세가 진입 방향과 일치하면 강력한 가점 요소입니다.\n"
                "- M5 RSI가 극단적(80 이상/20 이하)일 때만 진입 타이밍 조절을 위해 WAIT을 검토하세요.\n\n"

                

                "📌 [판단 원칙]\n"
                "- 추세와 진입 방향이 일치하면 진입을 선호한다 그러나 NEUTRAL 추세에서는 모멘텀 증가가 확인되어야 한다 RSI, MACD, Stoch RSI가 모두 중립이면 기본 판단은 WAIT이다..\n"
                "- 실제로 가격이 SL을 먼저 터치할 명확한 근거가 없는 한 진입을 유지한다.\n"
                "- 결과 예측(사후적 반등/되돌림 가정)을 근거로 WAIT을 선택하지 마라.\n\n"
                
                "아래 JSON 테이블을 기반으로 전략 리포트를 작성해. `score_components` 리스트는 각 전략 요소가 신호 판단에 어떤 기여를 했는지를 설명해.\n"
                "- 너의 목표는 알림에서 울린 BUY 또는 SELL을 사전에 '고정'하지 않고, BUY 점수와 SELL 점수를 각각 산출한 뒤 더 높은 점수를 최종 판단으로 선택하는 것이야.\n"
                "- 판단할 때는 아래 고차원 전략 사고 프레임을 참고하라.\n"
                "  • GI = (O × C × P × S) / (A + B): 감정, 언급, 패턴, 종합을 강화하고 고정관념과 편향을 최소화하라.\n"
                "  • MDA = Σ(Di × Wi × Ii): 시간, 공간, 인과 등 다양한 차원에서 통찰과 영향을 조합하라.\n"
                "  • IL = (S × E × T) / (L × R): 직관도 논리/경험과 파악하고 전략과 경험 기반 도약도 반영하라.\n\n"

                "(2) 거래는 기본적으로 1~2시간 내 청산을 목표로 하는 단타 스캘핑 트레이딩이다.\n"
                "- 이 전략은 reversal 전략이 아니라 breakout/continuation scalp 전략이다.\n"
                "- resistance 근접, RSI 45~60, stoch 과열은 단독으로 WAIT 근거가 아니다 단, resistance/supply zone까지 3 pip 이하이고 Stoch RSI > 0.9 인 경우는 예외다 이 경우 breakout 확인 전 BUY 추격 진입은 높은 실패 확률로 간주한다.\n"
                "- recent_ohlc, candle_micro, breakout_context, structure_context를 우선 해석하라.\n"
                "- SL과 TP는 ATR 기준 가급적 최소 50% 이상 거리로 설정하되, 시간이 너무 오래 걸릴 것 같으면 무시해도 좋다.\n"
                "- 하지만 반드시 **현재가 기준으로 TP는 ATR기반으로 계산하되 과도한 목표 설정을 방지하기 위해, 계산식 TP distance는 max(ATRx1.2, 0.11) 이 공식을 항상 따라라**, SL distance는 max(ATRx1.1, 0.11)이 공식을 항상 따르되 SL은 항상 16pip을 초과하지 않도록 한다. 이내로 설정하게 해줘 어떻게 계산했는지도 보여줘. 예외는 없다 그렇지 않으면 시장 변동성 대비 손실 확률이 급격히 높아진다.\n"
                "  (※ 위 TP/SL 공식은 FX 전용이다. 아래 (3-1)에서 종목이 미국 주식인 경우 이 공식 대신 별도 규칙을 따른다.)\n"
                "- 최근 5개 캔들의 고점/저점을 참고해서 너가 설정한 TP/SL이 **REASONABLE한지 꼭 검토**해.\n"
                "- RSI가 60 이상이고 Stoch RSI가 0.8 이상이며, 가격이 볼린저밴드 상단에 근접한 경우에는 'BUY 피로감'으로 간주해 'SELL'을 좀 더 고려해라.\n"
                "- RSI가 40 이하이고 Stoch RSI가 0.1 이하이며, 가격이 볼린저밴드 하단에 근접한 경우에는 'SELL 피로감'으로 간주해'BUY'을 좀 더 고려해라.\n\n"

                "(3) 지지선(support), 저항선(resistance)은 최근 1시간봉 기준 마지막 6개 캔들의 고점/저점에서 계산되었고 이미 JSON에 포함되어 있다.\n"
                f"  • 현재가: {current_price}, 지지선: {support}, 저항선: {resistance}\n"
                "- BUY 결정일 경우 TP는 반드시 현재가보다 높은 가격(상방)에, SL은 반드시 현재가보다 낮은 가격(하방)에 설정해야 한다.\n"
                "- SELL 결정일 경우 TP는 반드시 현재가보다 낮은 가격(하방)에, SL은 반드시 현재가보다 높은 가격(상방)에 설정해야 한다.\n"
                "- 이 규칙은 예외 없이 무조건 지켜야 하며, 이를 위반하는 TP 또는 SL을 생성하는 것은 허용되지 않는다.\n"
                "- GPT는 BUY/SELL 방향을 기준으로 TP/SL의 방향을 항상 먼저 판단한 후 값(pip 거리)을 계산해야 한다.\n"
                "- USD/JPY는 pip 단위가 소수점 둘째 자리입니다. TP와 SL은 반드시 이 기준으로 계산하세요. 이 규칙을 어기면 거래가 취소되므로 반드시 지켜야 한다. 예를들면 sell 거래의 진입가가 155.015라면 TP는 154.915가 10pip차이이다 \n\n"
                + (
                    f"(3-1) ⚠️ 이번 종목({pair})은 미국 주식(Alpaca)이다. 위 (2)의 FX용 TP/SL 공식(ATRx1.2/1.1, pip, 16pip 캡)은 "
                    f"이 종목에는 적용하지 마라. 대신 TradingView Pine 전략과 동일한 아래 공식을 반드시 사용하라:\n"
                    f"  • BUY: TP = 현재가 + ATR×{STOCK_TP_ATR_MULT}, SL = 현재가 − ATR×{STOCK_SL_ATR_MULT}\n"
                    f"  • SELL: TP = 현재가 − ATR×{STOCK_TP_ATR_MULT}, SL = 현재가 + ATR×{STOCK_SL_ATR_MULT}\n"
                    f"  • 단위는 'pip'이 아니라 달러(센트, 소수점 둘째 자리)이다.\n"
                    f"  • (참고: 이 값은 서버에서 동일한 공식으로 다시 한번 강제 재계산되어 최종 주문에 사용되니, "
                    f"네가 계산한 값이 위 공식과 다르면 그건 서버 값으로 덮어써진다. 그래도 보고하는 값은 위 공식과 일치시켜라.)\n\n"
                    f"(3-2) ⚠️ [주식 전용 판단 규칙 — 반드시 지켜라]\n"
                    f"이 주식 알림들은 'breakout + continuation(지속)' 전략에서 나온다. 원본 Pine 진입 조건은 정확히 이렇다:\n"
                    f"  • 최근 3봉 고점 돌파 + 모멘텀 캔들(종가>시가, 종가>전봉고가) + RSI>50 + StochRSI K>20\n"
                    f"이 조건들은 이미 알림이 발사된 시점에 전부 충족된 상태다. 즉 너의 역할은 '진입할지 말지를 새로 정하는 것'이 아니라, "
                    f"'그 사이 추세가 꺾일 명백한 반대 증거가 있는지'만 확인하는 것이다.\n"
                    f"  ❌ 아래 항목은 절대로 '단독' WAIT 근거로 쓰지 마라 (이 전략에서는 경고가 아니라 돌파 확인 신호다):\n"
                    f"     - 볼린저밴드 상단 돌파/근접 (continuation 전략에서는 돌파가 강하다는 뜻)\n"
                    f"     - 저항선 근접 (저항을 뚫고 가는 게 이 전략의 핵심이다)\n"
                    f"     - Stoch RSI 과열(>0.8) 단독 (RSI/MACD가 같은 방향이면 과열은 모멘텀 강도일 뿐이다)\n"
                    f"     - RSI 60~80대 '과매수 경계' 단독 (이 전략은 RSI>50만 요구하며 상한이 없다)\n"
                    f"  ✅ WAIT은 아래처럼 '명백한 반대 증거'가 있을 때만 선택하라:\n"
                    f"     - MACD가 시그널선 아래로 새로 꺾이며(약세 교차) RSI도 같이 하락 중인 경우\n"
                    f"     - 최근 캔들이 분명한 약세 패턴(예: 강한 장대음봉, 갭다운)으로 돌파를 무효화한 경우\n"
                    f"     - RSI/MACD/StochRSI 셋 다 동시에 하락 방향으로 전환된 경우\n"
                    f"  위 '✅ WAIT 근거'에 해당하지 않는다면, 위 (3-1) 공식 그대로 BUY/SELL을 확정하라. "
                    f"애매하다고 보수적으로 WAIT을 고르지 마라 — 애매함은 BUY/SELL 유지 근거다.\n\n"
                    if is_stock_pair(pair) else ""
                )
                +
                "(4) 추세 판단 시 캔들 패턴뿐 아니라 보조지표(RSI, MACD, Stoch RSI, 볼린저밴드)의 **방향성과 강도**를 반드시 함께 고려하라.\n"
                "- 특히 보조지표의 최근 14봉 흐름 분석은 핵심 판단 자료다. 반드시 함께 고려해라\n"
                f"- 아래는 멀티타임프레임({base_granularity_for(pair)}, H1, H4) 기준 요약 정보이다. 각 시간대별 추세가 일치하면 강한 확신으로 간주하고, 상반된 경우 보수적으로 판단하라:\n"
                f"📌 시스템 스코어: {score}, 신호 스코어: {signal_score}\n"
                f"📎 점수 산정 근거 (reasons):\n" + "\n".join([f"- {r}" for r in reasons]) + "\n\n"
                f"🕯️ 최근 캔들 흐름 요약: {recent_candle_summary}\n\n" +
                "📊 MTF 요약:\n"
                f"{summarize_mtf_indicators(mtf_indicators)}\n\n" +
                f"📉 RSI: {rsi_trend}, 📈 MACD: {macd_trend}, 🔄 Stoch RSI: {stoch_rsi_trend}\n" +
                "📊 아래는 RSI, MACD, Stoch RSI의 최근 14개 수치야. 이를 기반으로 추세를 요약해줘.\n" +
                f"↪️ RSI: {recent_rsi_values}\n" +
                f"↪️ MACD: {recent_macd_values}\n" +
                f"↪️ Stoch RSI: {recent_stoch_rsi_values}\n" +
                "➡️ 위 수치를 기반으로 최근 추세 흐름이 '상승세', '하락세', 또는 '횡보세'인지 간단히 요약해줘. 강도나 방향성도 덧붙여 분석에 반영해.\n"
                "- 각 지표의 상승/하락 추세, 변화 속도, 과매수/과매도 여부, 꺾임 여부 등을 분석해\n"
                "- 가능하면 수치적인 기준 또는 '강세', '약세', '중립' 등의 판단 용어를 사용해 설명하라.\n\n"

                "(5) 전략 리포트는 자유롭게 작성하되 반드시 아래 4단계 형식을 따르라:\n"
                "1️⃣ 전략 요약 (BUY/SELL 이유 요약)\n"
                "2️⃣ 기술 지표 분석 요약\n"
                "3️⃣ TP/SL 설정 근거 및 리스크 관리\n"
                "4️⃣ 최종 판단 및 이유\n\n"

                "(6) 마지막에는 반드시 아래 JSON 의사결정 블록을 작성하라. 양식은 정확히 아래처럼!\n\n"
                "{\n"
                "  \"decision\": \"BUY\" | \"SELL\" | \"WAIT\",\n"
                "  \"tp\": <숫자>,       // 반드시 숫자(float). 따옴표 금지. 예: 1.1745\n"
                "  \"sl\": <숫자>,       // 반드시 숫자(float). 따옴표 금지.\n"
                "  \"reason\": \"<간단한 핵심 이유 하나만 간결하게>\"\n"
                "}\n\n"
                "‼️ 출력 시 유의사항:\n"
                "- 코드블럭(````json .... ````) 사용 금지. 마크다운 태그 금지.\n"
                "- JSON 외의 텍스트(리포트)는 위에 모두 쓰고, 마지막 줄에는 **JSON 하나만** 단독 출력해야 한다.\n"
            )
        },
        {
            "role": "user",
            "content": user_content # 텍스트 데이터 + 이미지 데이터가 포함된 리스트 전달
        }
    ]
        
    # 2-c) 요청 바이트 수 로깅 (선택)
    body = {
        "model": "gpt-4o-2024-11-20",  # 또는 "gpt-4o-2024-11-20"
        "input": messages,
        "temperature": 0.3,
        "max_output_tokens": 1000,
    }
    need_tokens = _approx_tokens(messages)
    _preflight_gate(need_tokens)   # 요청 직전 선대기

    try:
        _bytes = len(json.dumps(payload, ensure_ascii=False))
    except Exception:
        _bytes = -1
    
    dbg("gpt.body", bytes=_bytes, max_tokens=body.get("max_tokens"))
    print("🔍 FULL BODY DEBUG:", json.dumps(body, indent=2, ensure_ascii=False))


    # 2-d) 최소 스로틀: 같은 프로세스에서 1.2초(또는 네가 정한 값) 간격 보장
    with _gpt_lock:
        global _gpt_last_ts
        now = _t.time()
        gap = now - _gpt_last_ts
        min_gap = 12.0  
        if gap < min_gap:
            _t.sleep(min_gap - gap)
        _gpt_last_ts = _t.time()
    try:
        dbg("gpt.call")
        r = requests.post(
            OPENAI_URL,
            headers=OPENAI_HEADERS,
            json=body,
            timeout=90,
        )
        print("GPT STATUS:", r.status_code)
        r.raise_for_status()  # HTTP 에러 체크
        data = r.json()
        
        
        output_blocks = data.get("output", [])
        
        text = ""
        for block in output_blocks:
            # 1) assistant 메시지 찾기
            if block.get("role") == "assistant":
                # 2) 그 안에서 output_text 찾기
                for c in block.get("content", []):
                    if c.get("type") == "output_text":
                        text = c.get("text", "")
                        break
                if text:
                    break
        
        text = (text or "").strip()
        print(f"📩 GPT 원문 응답: {text[:500]}...")
        return text if text else "GPT 응답 없음"

    except requests.exceptions.Timeout:
        print("❌ GPT 응답 시간 초과")
        return "GPT_TIMEOUT"
    
    except Exception as e:
    
        print("\n========== GPT ERROR ==========")
    
        print("ERROR:", str(e))
    
        try:
            print("STATUS:", r.status_code)
        except:
            print("STATUS: UNKNOWN")
    
        try:
            print("BODY:")
            print(r.text)
        except:
            print("BODY: NONE")
    
        print("================================\n")
    
        return f"GPT_ERROR: {str(e)}"
    
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


def log_trade_result(
    pair, signal, decision, score, notes, result=None,
    rsi=None, macd=None, stoch_rsi=None,
    pattern=None, trend=None, fibo=None,
    gpt_decision=None, news=None, gpt_feedback=None,
    alert_name=None, tp=None, sl=None, entry=None,
    price=None, pnl=None,
    outcome_analysis=None, adjustment_suggestion=None,
    price_movements=None, atr=None,
    support=None, resistance=None,
    liquidity=None,
    macd_signal=None, macd_trend=None, macd_signal_trend=None,
    stoch_rsi_trend=None, rsi_trend=None,
    bollinger_upper=None, bollinger_lower=None,
    news_text=None,  # news 전문 별도 전달 시
    gpt_feedback_dup=None,
    filtered_movement=None
):
    
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name("/etc/secrets/google_credentials.json", scope)
    client = gspread.authorize(creds)
    sheet = client.open("민균 FX trading result").sheet1
    now_atlanta = datetime.now(ZoneInfo("America/New_York"))
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
    if len(filtered_movements) > 0:
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
    support_out = support
    resist_out  = resistance
    row = [
      
        str(now_atlanta),                 # timestamp
        pair,                             # symbol
        alert_name or "",                 # strategy
        signal,                           # signal_type
        decision,                         # decision
        score,                            # score
        safe_float(rsi),                  # rsi
        safe_float(macd),                 # macd
        safe_float(stoch_rsi),            # stoch_rsi

        trend or "",                      # trend
        pattern or "",                    # candle_trend (☜ 기존엔 pattern이 trend 앞/뒤 섞였음)

        support_out,                      # ✅ support (진짜 S/R)
        resist_out,                       # ✅ resistance

        gpt_decision or "",               # final_decision
        news or "",                       # news_summary
        notes,                            # reason
        json.dumps(result, ensure_ascii=False) if isinstance(result, dict) else (result or "미정"),
        gpt_feedback or "",               # order_json
        gpt_feedback or "GPT 응답 없음",   # gpt_feedback (필요 없으면 빈칸 유지)

        safe_float(price),                # price
        safe_float(tp),                   # tp
        safe_float(sl),                   # sl
        safe_float(pnl),                  # pnl

        is_new_high,                      # is_new_high
        is_new_low,                       # is_new_low
        safe_float(atr),                  # atr
        liquidity,
        macd_signal,
        macd_trend,
        macd_signal_trend,
        stoch_rsi_trend,
        rsi_trend,

        # ↓ 아래 필드들이 시트 헤더에 실제로 있다면 그대로 유지,
        #   없다면 이 아래 줄들만 지워도 무방 (헤더와 컬럼 수는 항상 동일해야 함)
        news,                             # (선택) news 원문
        outcome_analysis or "",           # (선택)
        adjustment_suggestion or "",      # (선택)
        gpt_feedback or "",               # (선택) gpt_feedback_dup
        filtered_movement_str or ""       # (선택)
        ]
    
    clean_row = []
    for v in row:
        if isinstance(v, (dict, list)):
            try:
                clean_row.append(json.dumps(v, ensure_ascii=False))
            except Exception as e:
                print(f"[❌ JSON 변환 실패 → {e}]")
                clean_row.append(str(v))  # fallback 처리
        elif isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            clean_row.append("")  # 빈 문자열로 처리
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


# ============================================================
# 🟦 결과 자동 추적 (백테스트 보조) — 1시간마다 미정 행들을 채워준다
# ============================================================

def _generate_outcome_note(outcome: str, reasons_text: str, decision_text: str, was_executed: bool) -> str:
    """
    GPT 호출 없이 규칙 기반으로 짧은 설명 생성.
    score_components/reason 텍스트에 특정 키워드가 있으면 그걸 결과와 엮어서 설명한다.
    """
    text = reasons_text or ""
    overheated = any(k in text for k in ["과열", "과매수", "과매도", "피로"])
    strong_momentum = any(k in text for k in ["골든크로스", "모멘텀 유지", "추세 상승"])
    exec_tag = "(실거래)" if was_executed else "(미실행/가정)"

    if outcome == "TP_HIT":
        if overheated:
            return f"✅ TP 적중 {exec_tag} — 과열 경고가 있었지만 모멘텀이 더 강하게 이어졌음"
        if strong_momentum:
            return f"✅ TP 적중 {exec_tag} — 모멘텀 신호와 결과가 일치함"
        return f"✅ TP 적중 {exec_tag}"
    elif outcome == "SL_HIT":
        if overheated:
            return f"❌ SL 적중 {exec_tag} — 과열 경고가 실제로 맞아떨어짐 (필터 강화 검토 필요)"
        return f"❌ SL 적중 {exec_tag} — 특별한 경고 신호 없었는데도 손절 도달"
    elif outcome == "TIMEOUT_NO_HIT":
        return f"⏳ 시간초과 {exec_tag} — TP/SL 둘 다 도달 못함 (박스권/모멘텀 부족 가능성)"
    return ""


def evaluate_pending_outcomes(max_window_minutes: int = 240, min_elapsed_minutes: int = 5):
    """
    구글시트에서 아직 결과가 안 채워진 행들을 찾아서,
    그 시점 이후 캔들을 다시 조회해 TP/SL 중 뭘 먼저 쳤는지 판정하고
    result / outcome_analysis 컬럼에 자동으로 채워넣는다.
    (1시간마다 백그라운드로 호출됨. 수동으로도 /run_outcome_tracker 로 트리거 가능)
    """
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name("/etc/secrets/google_credentials.json", scope)
        client = gspread.authorize(creds)
        sheet = client.open("민균 FX trading result").sheet1
        all_rows = sheet.get_all_values()
    except Exception as e:
        print(f"❌ [결과추적] 시트 읽기 실패: {e}")
        return {"checked": 0, "updated": 0, "error": str(e)}

    checked = 0
    updated = 0

    for i, row in enumerate(all_rows[1:], start=2):  # 1번째 줄은 헤더, 시트 row는 1-indexed
        try:
            timestamp_str = row[0] if len(row) > 0 else ""
            pair = row[1] if len(row) > 1 else ""
            signal_dir = row[3] if len(row) > 3 else ""   # 원래 알림 방향(BUY/SELL) — decision이 WAIT여도 이건 살아있음
            decision_text = row[4] if len(row) > 4 else ""
            reasons_text = row[15] if len(row) > 15 else ""
            result_col = row[16] if len(row) > 16 else ""
            price_s = row[19] if len(row) > 19 else ""
            tp_s = row[20] if len(row) > 20 else ""
            sl_s = row[21] if len(row) > 21 else ""
        except Exception:
            continue

        if signal_dir not in ("BUY", "SELL"):
            continue
        if result_col not in ("", "미정"):
            continue  # 이미 평가됨

        try:
            price_f = float(price_s)
            tp_f = float(tp_s)
            sl_f = float(sl_s)
        except Exception:
            continue  # TP/SL이 없는 행(WAIT인데 값 자체가 없는 경우 등)은 평가 불가 → 스킵

        try:
            entry_time = datetime.fromisoformat(timestamp_str)
        except Exception:
            continue

        now = datetime.now(entry_time.tzinfo) if entry_time.tzinfo else datetime.now()
        elapsed_minutes = (now - entry_time).total_seconds() / 60
        if elapsed_minutes < min_elapsed_minutes:
            continue  # 아직 너무 따끈따끈한 신호 → 다음 시간에 다시 체크

        checked += 1

        # 🟦 결과 "판정"은 알림 자체의 타임프레임(15분 등)과 무관하게 1분봉으로 본다.
        #    15분봉 하나엔 시가/고가/저가/종가만 있어서, 그 15분 안에서 SL을 먼저 쳤는지
        #    TP를 먼저 쳤는지 순서를 구분할 수 없다(둘 다 한 봉 안에 있으면 어느 게 먼저인지 모름).
        #    1분봉으로 보면 그 순서를 거의 다 구분할 수 있다.
        gran = "M1"
        _gran_minutes = 1
        # 🟦 버그 수정: 50개 고정이면, 평가가 늦어질수록(1시간마다 도니까 몇 시간 뒤일 수 있음)
        #    "가장 최근 N개"가 진입 시점보다 한참 뒤부터 시작돼서 진입 직후(SL 터치 가능 구간)를
        #    통째로 못 보고 누락 → 나중에 회복해서 TP쪽으로 간 부분만 보고 TP_HIT으로 오판하게 됨.
        #    경과 시간을 확실히 덮을 만큼 동적으로 더 많이 가져온다.
        bars_needed = int(elapsed_minutes / _gran_minutes) + 20  # 여유 버퍼 20개
        candles = get_candles(pair, gran, max(50, bars_needed))
        if candles is None or candles.empty:
            continue

        try:
            candles = candles.copy()
            candles["time_dt"] = pd.to_datetime(candles["time"], utc=True)
            entry_time_utc = entry_time.astimezone(ZoneInfo("UTC")) if entry_time.tzinfo else entry_time
            after = candles[candles["time_dt"] >= entry_time_utc]

            # 🟦 안전장치: 가져온 캔들의 시작점이 진입 시점보다 너무 늦으면(=진입 직후 구간이 누락됐으면)
            #    잘못된 판정(특히 거짓 TP_HIT)을 낼 수 있으니, 이번엔 건너뛰고 다음 시간에 재시도.
            earliest_fetched = candles["time_dt"].min()
            gap_minutes = (entry_time_utc - earliest_fetched).total_seconds() / 60
            if gap_minutes > _gran_minutes * 2:
                print(f"⚠️ [결과추적] {pair} 캔들이 진입시점을 충분히 못 덮음 "
                      f"(진입={entry_time_utc}, 가져온 캔들 시작={earliest_fetched}) → 이번엔 스킵")
                continue
        except Exception as e:
            print(f"❗ [결과추적] {pair} 캔들 시간 처리 실패: {e}")
            continue

        outcome = "PENDING"
        for _, c in after.iterrows():
            if signal_dir == "BUY":
                if c["low"] <= sl_f:
                    outcome = "SL_HIT"
                    break
                if c["high"] >= tp_f:
                    outcome = "TP_HIT"
                    break
            else:  # SELL
                if c["high"] >= sl_f:
                    outcome = "SL_HIT"
                    break
                if c["low"] <= tp_f:
                    outcome = "TP_HIT"
                    break

        if outcome == "PENDING":
            if elapsed_minutes > max_window_minutes:
                outcome = "TIMEOUT_NO_HIT"
            else:
                continue  # 아직 더 기다려야 함 (다음 시간에 재평가)

        was_executed = decision_text in ("BUY", "SELL")
        note = _generate_outcome_note(outcome, reasons_text, decision_text, was_executed)

        try:
            sheet.update_cell(i, 17, outcome)        # 'result' 컬럼 (1-indexed 17번째)
            sheet.update_cell(i, 34, note)            # 'outcome_analysis' 컬럼 (1-indexed 34번째)
            updated += 1
            print(f"✅ [결과추적] row {i} ({pair}, {signal_dir}) → {outcome}")
        except Exception as e:
            print(f"❌ [결과추적] row {i} 시트 업데이트 실패: {e}")

    print(f"📊 [결과추적] 체크 {checked}건 / 업데이트 {updated}건")
    return {"checked": checked, "updated": updated}


async def _hourly_outcome_tracker_loop():
    """1시간마다 evaluate_pending_outcomes()를 백그라운드 스레드에서 실행."""
    while True:
        try:
            await asyncio.to_thread(evaluate_pending_outcomes)
        except Exception as e:
            print(f"❌ [결과추적 루프] 오류: {e}")
        await asyncio.sleep(3600)  # 1시간


@app.on_event("startup")
async def _start_background_tasks():
    asyncio.create_task(_hourly_outcome_tracker_loop())


@app.post("/run_outcome_tracker")
async def run_outcome_tracker_endpoint():
    """수동으로 즉시 결과 추적을 돌리고 싶을 때 호출 (정기 1시간 루프와 별개)."""
    result = await asyncio.to_thread(evaluate_pending_outcomes)
    return JSONResponse(content=result)


def get_last_trade_time():
    try:
        with open("/tmp/last_trade_time.txt", "r") as f:
            return datetime.fromisoformat(f.read().strip())
    except:
        return None
