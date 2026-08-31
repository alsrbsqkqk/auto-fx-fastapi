# ⚠️ V2 업그레이드된 자동 트레이딩 스크립트 (학습 강화, 트렌드 보강, 시트 시간 보정 포함)
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from zoneinfo import ZoneInfo
import os
import requests
import json
import pandas as pd
from datetime import datetime, timedelta, timezone as _tz   # 🟥 [FIX-G10] _tz 를 전역으로
import openai
import numpy as np
import gspread
from collections import deque   # 🟥 [FIX-WJ1] 웹훅 도착 장부
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
import contextvars          # 🟥 [FIX-T1] 웹훅 요청별 시간축 보관용
# 🟥 [FIX-E1] API 키 전문을 stdout에 출력하던 줄을 제거.
#    Render/Docker 로그에 그대로 남아 유출 위험이 있었다.
#    키가 로드됐는지만 확인할 수 있게 마스킹해서 찍는다.
def _mask_secret(v: str | None) -> str:
    if not v:
        return "(없음)"
    return f"{v[:6]}…{v[-4:]} (len={len(v)})"


print("🔑 OPENAI KEY 로드:", _mask_secret(os.getenv("OPENAI_API_KEY")))
_gpt_lock = threading.Lock()
_gpt_last_ts = 0.0
_gpt_cooldown_until = 0.0
_gpt_rate_lock = threading.Lock()
# 🟦 같은 종목 반복신호 감지용 — 1시간 내 같은 종목에서 2번째 신호가 나오면,
#    그 신호까지는 허용하고 그 다음(3번째)부터는 그 종목만 1시간 쉬게 한다.
_symbol_signal_lock = threading.Lock()
# ============================================================
# 🟥 [FIX-E5] 심볼별 주문 락 + 알림 중복 제거
# ------------------------------------------------------------
#  기존엔 "포지션 확인 → 주문" 사이에 아무 락이 없어서, 같은 종목 알림이
#  거의 동시에 2건 들어오면 둘 다 "보유 없음"을 보고 둘 다 주문할 수 있었다
#  (웹훅이 스레드풀에서 병렬 처리되므로 실제로 가능한 시나리오다).
#  또 TradingView가 같은 봉에 대해 알림을 재전송해도 걸러낼 키가 없었다.
# ============================================================
_order_locks: dict[str, threading.Lock] = {}
_order_locks_guard = threading.Lock()
_recent_alert_keys: dict[str, float] = {}
_recent_alert_guard = threading.Lock()
# 같은 (종목·방향·봉시각) 알림이 이 시간 안에 다시 오면 중복으로 보고 버린다.
ALERT_DEDUP_SECONDS = int(os.getenv("ALERT_DEDUP_SECONDS", "60"))


def _get_order_lock(symbol: str) -> threading.Lock:
    """심볼별 주문 락을 가져온다(없으면 생성)."""
    key = (symbol or "").upper()
    with _order_locks_guard:
        lk = _order_locks.get(key)
        if lk is None:
            lk = threading.Lock()
            _order_locks[key] = lk
        return lk


def _is_duplicate_alert(symbol: str, signal: str, bar_time=None) -> bool:
    """
    같은 알림이 짧은 시간 안에 중복 도착했는지 판정.
    bar_time(봉 시각)이 오면 그것까지 키에 포함해 '같은 봉 재전송'을 정확히 잡는다.
    """
    if not symbol or not signal:
        return False
    key = f"{symbol.upper()}:{signal}:{bar_time or ''}"
    now = _t.time()
    with _recent_alert_guard:
        # 오래된 항목 정리
        for k in [k for k, v in _recent_alert_keys.items() if now - v > ALERT_DEDUP_SECONDS * 3]:
            _recent_alert_keys.pop(k, None)
        prev = _recent_alert_keys.get(key)
        if prev is not None and (now - prev) < ALERT_DEDUP_SECONDS:
            return True
        _recent_alert_keys[key] = now
    return False
_symbol_signal_history = {}   # symbol -> [datetime, ...] (최근 1시간 이내 신호만 유지)
_symbol_cooldown_until = {}   # symbol -> datetime (이 시각까지 신규진입 차단)
SYMBOL_REPEAT_WINDOW_MINUTES = int(os.getenv("SYMBOL_REPEAT_WINDOW_MINUTES", "60"))
SYMBOL_REPEAT_COOLDOWN_MINUTES = int(os.getenv("SYMBOL_REPEAT_COOLDOWN_MINUTES", "60"))
_gpt_next_slot = 0.0
_last_execution_time = 0.0  # 마지막 실행 시간을 저장할 변수
# 🟥 [FIX-E3] 전역(전 종목 공통) 쿨다운 초. 0이면 비활성(기본).
#    종목별 쿨다운은 SYMBOL_REPEAT_* 로 따로 관리한다.
GLOBAL_COOLDOWN_SECONDS = int(os.getenv("GLOBAL_COOLDOWN_SECONDS", "0"))
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
    # 🟥 [FIX-ROLE1] 지표 합이 음수라고 방향을 다시 뒤집는 건 셋업 재판단이다.
    #   게다가 이미 음수인 점수를 한 번 더 깎아 이중 처벌이 된다.
    if SETUP_RECHECK_ENABLED:
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

    # 2️⃣ 🟥 [FIX-B7] 주석과 코드가 정반대였다.
    #    주석은 "역방향이면 관망"인데 코드는 '같은 방향'일 때 return False(충돌 없음)였다.
    #    코드 쪽이 의도상 맞다(패턴이 없어도 신호와 추세가 같으면 충돌 아님).
    #    → 주석을 실제 동작에 맞게 고치고, 조기 return이 아래 3️⃣ 규칙을 건너뛰지 않도록
    #      정리한다(현재 조건상 겹치지는 않지만, 나중에 규칙을 추가할 때 함정이 된다).
    if pattern == "NEUTRAL":
        if (signal == "BUY" and trend == "UPTREND") or (signal == "SELL" and trend == "DOWNTREND"):
            return False   # 패턴은 없지만 신호와 추세가 일치 → 충돌 아님

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
    now = datetime.now(ZoneInfo("UTC"))   # 🟥 [FIX-E9] naive utcnow() → aware UTC (3.12+ deprecated)

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
    """
    🟥 [FIX-B8] 구조적(지지/저항 기반) SL/TP.

    기존 구현은 TP를 항상 `SL거리 × 1.8`로 만들었기 때문에 r_ratio가 수학적으로
    언제나 정확히 1.8이었다. 그런데 호출부에서는 `if r_ratio < 1.4: -4.0점` 감점을
    걸어놨다 — 절대 발동할 수 없는 죽은 감점이었다.
    → r_ratio를 "구조상 실제로 얻을 수 있는 손익비"로 계산하도록 바꾼다.
       즉 TP는 저항(BUY)/지지(SELL)라는 실제 구조 목표에 두고, 그 목표까지의 거리와
       SL 거리의 비율을 r_ratio로 본다. 구조 목표가 없으면 기존 1.8 폴백을 쓴다.
    """
    buffer = get_buffer_by_symbol(symbol, atr=atr)

    if direction == 'BUY':
        sl = support - buffer if support is not None else None
        structural_tp = resistance
    else:
        sl = resistance + buffer if resistance is not None else None
        structural_tp = support

    # SL을 못 구하면 (지지/저항 없음) 판정 불가 — 중립값 반환
    if sl is None or entry_price is None or abs(sl - entry_price) < 1e-12:
        print(f"[SL/TP 계산] {symbol} 구조 SL 산출 불가(support={support}, resistance={resistance}) → r_ratio 중립(1.8) 처리")
        return sl, None, 1.8

    risk = abs(entry_price - sl)

    if structural_tp is not None:
        reward = abs(structural_tp - entry_price)
        # 🟥 [FIX-B8b] "반대편에 있으면 폴백"이라고 써놓고 실제 방향 체크가 없었다.
        #    BUY인데 저항이 이미 진입가 아래(=돌파 후 낡은 값)면 tp가 진입가보다 낮아지고,
        #    abs() 때문에 r_ratio는 오히려 커져서 감점을 우회한다.
        wrong_side = (
            (direction == 'BUY' and structural_tp <= entry_price)
            or (direction != 'BUY' and structural_tp >= entry_price)
        )
        if wrong_side or reward < risk * 0.1:
            tp = entry_price + risk * 1.8 if direction == 'BUY' else entry_price - risk * 1.8
        else:
            tp = structural_tp
    else:
        tp = entry_price + risk * 1.8 if direction == 'BUY' else entry_price - risk * 1.8

    r_ratio = abs(tp - entry_price) / risk

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
        # 🟥 [FIX-ROLE1] 전략이 이미 RSI 하한을 걸어서 알람을 낸 것이다. 중복 감점.
        if SETUP_RECHECK_ENABLED:
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

    # 🟥 [FIX-B8] 이제 r_ratio가 실제 구조 손익비를 반영하므로 이 감점이 살아난다.
    #    다만 -4.0은 다른 항목(대부분 ±1~2)에 비해 과도해서 단독으로 점수를 지배한다.
    #    → -2.0으로 낮추고, 구간을 나눠 완만하게 적용한다.
    # 🟥 [FIX-ROLE1] 돌파 전략에서는 '저항까지 남은 거리' 로 감점하면 안 된다.
    #   진입 자리가 저항 바로 아래인 게 돌파 전략의 정의이기 때문이다.
    #   참고용으로 사유에는 남기되 점수는 깎지 않는다.
    if not SETUP_RECHECK_ENABLED:
        reasons.append("ℹ️ 구조 손익비 %.2f (참고) — 돌파 전략이라 감점하지 않음" % r_ratio)
    elif r_ratio < 1.0:
        signal_score -= 2.0
        reasons.append("📉 구조 손익비 매우 낮음 (%.2f < 1.0) → 감점 -2.0" % r_ratio)
    elif r_ratio < 1.4:
        signal_score -= 1.0
        reasons.append("📉 구조 손익비 낮음 (%.2f < 1.4) → 감점 -1.0" % r_ratio)

    from datetime import datetime
    from zoneinfo import ZoneInfo
    
    now_atlanta = datetime.now(ZoneInfo("America/New_York"))
    atlanta_hour = now_atlanta.hour
    
    # 🟥 [FIX-B9] 19~23시(ET) 감점은 FX 전용이다.
    #    미국 주식 정규장은 09:30~16:00 ET라 이 시간대에 주식 알림이 올 수 없고,
    #    프리/애프터 알림이 들어오면 -3점이 통째로 붙어버린다.
    #    주식엔 이미 별도의 시간대 게이트(점심/15:30/금요일)가 있으므로 여기선 FX만 적용한다.
    if (not is_stock_pair(pair)) and 19 <= atlanta_hour < 23:
        signal_score -= 3
        reasons.append("🌙 FX 19~23시(ET) 유동성 저하 구간 감점 (-3)")
        
    # ====================================
    # 🟦 -0.02는 FX 스케일(가격 1.0~1.5대) 전용 절대값이라, 주식(가격 수십~수천)에서는
    #    MACD가 조금만 음수여도 항상 걸려서 의미 없는 상시 감점이 됨. 주식은 ATR 비례로 교체.
    # 🟦 추가 수정: "MACD가 음수다" 자체와 "MACD가 음수인데 계속 나빠지고 있다"는 다르다.
    #    음수여도 최근 3개 값이 계속 좋아지고 있으면(회복 중) 이건 약세가 아니라 회복 신호라
    #    패널티를 절반으로 줄인다 (완전히 없애지는 않음 — 아직 양전환 전이라는 리스크는 남아있음).
    _macd_recovering = (
        macd_trend and len(macd_trend) >= 3
        and macd_trend[-1] > macd_trend[-2] > macd_trend[-3]
    )
    _macd_weak_thresh = -(atr * 0.02) if (is_stock_pair(pair) and atr) else -0.02
    if macd < _macd_weak_thresh and trend != "DOWNTREND":
        if _macd_recovering:
            score -= 0.75
            reasons.append("🔻 MACD 음수지만 회복 중 → 약세 판정 완화 (감점 -0.75, 기존 -1.5)")
        else:
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

    # 🟥 [FIX-B4] 추세 전환 직후 감점 — 중복 제거.
    #    기존엔 완전히 동일한 조건(UPTREND & prev DOWNTREND & BUY)에 -0.5와 -1.0이
    #    연달아 적용돼 합계 -1.5가 걸렸다. SELL 미러 조건도 마찬가지.
    #    조건이 같은데 블록만 두 개였던 것이라 하나로 합친다.
    if trend == "UPTREND" and prev_trend == "DOWNTREND" and signal == "BUY":
        score -= 1.0
        reasons.append("🔄 하락→상승 추세 전환 직후 BUY → 조기 진입 경고 (감점 -1.0)")

    if trend == "DOWNTREND" and prev_trend == "UPTREND" and signal == "SELL":
        score -= 1.0
        reasons.append("🔄 상승→하락 추세 전환 직후 SELL → 조기 진입 경고 (감점 -1.0)")
    

    
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
            # 🟦 주식은 Pine이 이미 "3봉 고점 돌파"를 확인한 뒤에야 알림을 보낸다.
            #    이 시점의 NEUTRAL은 추세가 진짜 없다는 뜻이 아니라, EMA 기반 추세 판정이
            #    막 시작된 돌파를 아직 못 따라잡은 지표 지연일 가능성이 높다.
            #    그래서 주식은 감점을 더 약하게(-0.15), FX는 기존 그대로(-0.3) 유지.
            #    (표시 문구가 "-0.7"로 돼있었는데 실제 감점은 -0.3이었던 불일치도 같이 수정)
            if is_stock_pair(pair):
                signal_score -= 0.15
                reasons.append("🟡 NEUTRAL 추세(돌파 초기 지표 지연 가능성) → 약한 감점 (-0.15)")
            else:
                signal_score -= 0.3
                reasons.append("🟡 NEUTRAL 추세 → continuation 신뢰도 낮음 (-0.3)")
    
    
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
    # 🟥 [FIX-B6b] 기존엔 방향을 안 봤다. FIX-B6로 LONG_BODY_* 라벨이 처음 생성되기
    #    시작하면, 큰 "음봉"이 BUY 신호에 +1.5를 주는 정반대 동작이 실제로 발생한다.
    #    → 캔들 방향과 신호 방향이 일치할 때만 가점하고, 역방향이면 감점한다.
    if pattern == "LONG_BODY_BULL":
        if signal == "BUY":
            signal_score += 1.5
            reasons.append("📊 장대 양봉 + BUY → 추세 지속 가능성 (+1.5)")
        elif signal == "SELL":
            signal_score -= 1.0
            reasons.append("⚠️ 장대 양봉인데 SELL → 역방향 진입 위험 (-1.0)")
    elif pattern == "LONG_BODY_BEAR":
        if signal == "SELL":
            signal_score += 1.5
            reasons.append("📊 장대 음봉 + SELL → 추세 지속 가능성 (+1.5)")
        elif signal == "BUY":
            signal_score -= 1.0
            reasons.append("⚠️ 장대 음봉인데 BUY → 역방향 진입 위험 (-1.0)")

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
                # 🟦 MACD 음수권에서 3봉 연속 상승 — 회복 속도를 판단해서 차등 적용
                #    데이터 분석(6/22~7/8, 48건): 이 패턴 전체 승률 46%, 손익 -$186
                #    원인: 빠른 수렴(진짜 전환)과 느린 수렴(노이즈)이 섞여있음
                #    해결: MACD와 시그널선 간 갭이 얼마나 빠르게 줄어드는지로 구분
                _gap_now  = abs(macd_trend[-1] - (macd_signal if not hasattr(macd_signal, 'iloc') else float(macd_signal.iloc[-1])))
                _gap_prev = abs(macd_trend[-2] - (macd_signal if not hasattr(macd_signal, 'iloc') else float(macd_signal.iloc[-2])))
                _recovery_speed = (_gap_prev - _gap_now) / max(_gap_prev, 1e-9)

                if _recovery_speed >= 0.15:
                    # 갭이 15% 이상 빠르게 줄어듦 → 진짜 전환 신호 → 가점 유지
                    signal_score += 0.7
                    reasons.append(
                        f"🟢 MACD 음수권 빠른 회복(수렴속도 {_recovery_speed*100:.0f}%) → 반등 가점 (+0.7)"
                    )
                else:
                    # 갭이 15% 미만으로 느리게 줄어듦 → 노이즈성 튀기 → 강한 감점
                    signal_score -= 1.5
                    reasons.append(
                        f"🔴 MACD 음수권 느린 회복(수렴속도 {_recovery_speed*100:.0f}%) → 노이즈 의심 강감점 (-1.5)"
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
            # 🟦 4번 수정: -2 → -3으로 강화
            #    데이터(14건): 이 필터 있어도 실거래 36% 승률 -$85 → 감점이 부족해서 threshold 통과 중
            #    -3으로 올리면 threshold -2.5 기준 대부분 차단됨
            signal_score -= 3.0
            # 🟥 [FIX-HB1] 감점이 아니라 진짜 차단
            reasons.append(f"{HARD_BLOCK_TAG}|BUY 차단: Stoch RSI 과열 + MACD 약화(macd<signal) → 추격 매수 위험")
    
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
                # 🟥 [FIX-HB1] 감점이 아니라 진짜 차단.
                #   ⚠️ 이 append 는 반드시 else 안에 있어야 한다. 한 칸 밖으로 나가면
                #      위 (A)DOWNTREND·(B)NEUTRAL 분기까지 전부 차단돼 버린다.
                reasons.append(f"{HARD_BLOCK_TAG}|SELL 차단: 과매도(Stoch<0.2) + MACD 약화 + 추세 불리 → 추격 매도 위험")
   
    # 🟦 breakout_confirmed/near_resistance를 stoch_rsi>=0.95 체크보다 먼저 계산해서,
    #    "추세 라벨(UPTREND)"이 아니라 "실제 돌파 확정 여부"로 과열 판정을 보완할 수 있게 함.
    _atr_val_early = atr if atr is not None else 0.0
    _res_val_early = resistance if resistance is not None else None
    _near_resistance_early = False
    if _res_val_early is not None and price is not None:
        # 🟥 [FIX-B2] 방향 조건(_res_val_early > price)이 빠져 있었다.
        #    가격이 저항을 뚫고 위로 올라가면 (저항 - 가격)이 음수라 이 식은 항상 True가 됐고,
        #    그 결과 아래 "돌파확정 + 저항 안 가까움" 조건이 수학적으로 성립 불가였다.
        #    → 모멘텀 가점(+2/+1.5)은 한 번도 안 나가고 항상 -2 감점만 적용됐다.
        _near_resistance_early = (
            _res_val_early > price
            and (_res_val_early - price) <= max(10 * pv, _atr_val_early * 0.6)
        )
    _buffer_early = max(2 * pv, _atr_val_early * 0.10)
    _breakout_confirmed_early = False
    if _res_val_early is not None and close is not None:
        _breakout_confirmed_early = close >= (_res_val_early + _buffer_early)

    if stoch_rsi >= 0.95:
        # 🟦 주식은 detect_trend의 NEUTRAL 판정이 막 시작된 돌파를 못 따라잡는 경우가 많아서,
        #    "UPTREND여야만 봐준다"는 조건 대신 실제 돌파 확정 여부(breakout_confirmed)도 같이 인정.
        _confirmed_momentum = (trend == "UPTREND" and macd is not None and macd > 0) or (
            is_stock_pair(pair) and _breakout_confirmed_early and not _near_resistance_early
            and macd is not None and macd > 0
        )
        if _confirmed_momentum:
            signal_score -= 0.5
            reasons.append("🟡 Stoch RSI 과열이지만 돌파확정/상승추세 + MACD 양수 → 조건부 감점 -0.5")
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
        # 🟥 [FIX-B2] 위와 동일한 부호 버그. 저항이 "위에 있을 때"만 근접으로 본다.
        near_resistance = (
            res_val > price
            and (res_val - price) <= max(10 * pip, atr_val * 0.6)
        )
    
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
                        # 🟦 데이터 분석(6/22~7/8): Stoch 과매도 BUY 11건, 승률 45%, 손익 -$25
                        #    가점을 줘도 승률이 오히려 낮아짐 → 가점 제거, 중립 처리
                        #    "과매도 = 반등"이라는 가정이 이 전략에서 성립하지 않음
                        reasons.append("🟡 Stoch RSI 과매도 → BUY 반등 기대 (데이터상 효과 미검증, 가점 0)")
    
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
                # 🟥 [FIX-ROLE1] 돌파 전략은 '박스 상단' 을 노리고 들어간다. 그걸 감점하면
                #   전략이 잡으라고 만든 자리를 전부 거부하게 된다.
                if SETUP_RECHECK_ENABLED:
                    signal_score -= 1.5
                    reasons.append("⚠️ 박스 상단 근접 매수 위험 (감점-1.5)")
                else:
                    reasons.append("ℹ️ 박스 상단 근접 (참고) — 돌파 전략의 진입 자리라 감점하지 않음")

        # 하단 근접 매도 금지 (확정 이탈 or 리테스트만 허용)
        if signal == "SELL" and box_info.get("in_box") and box_info.get("breakout") is None:
            confirmed_bottom_break = recent.iloc[-1]['close'] < (box_low - buf_price)
            retest_resist = (recent.iloc[-1]['high'] < box_low + buf_price) and (near_low_pips <= NEAR_PIPS)
            if near_low_pips <= NEAR_PIPS and not (confirmed_bottom_break or retest_resist):
                # 🟥 [FIX-ROLE1] BUY 쪽만 풀면 SELL 신호만 감점이 남아 방향 편향이 생긴다.
                if SETUP_RECHECK_ENABLED:
                    signal_score -= 1.5
                    reasons.append("⚠️ 박스 하단 근접 매도 위험 (감점-1.5)")
                else:
                    reasons.append("ℹ️ 박스 하단 근접 (참고) — 이탈 전략의 진입 자리라 감점하지 않음")
                
    # 상승 연속 양봉 패턴 보정 BUY
    #   🟥 [FIX-HB1] is_buy 가드 필수. 감점일 때는 다른 가점이 상쇄해서 티가 안 났지만,
    #      하드 차단이 된 뒤로는 이게 없으면 상승 3봉 구간의 SELL 신호까지 전부 죽는다.
    if (
        is_buy
        and all(last_3["close"] > last_3["open"]) 
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

        elif rsi is not None and rsi < 70:
            # 🟦 데이터 분석(6/22~7/8): RSI<70 + 3봉양봉 케이스 15건 승률 20%, 손익 -$215
            #    모멘텀이 약한 상태에서의 3봉 연속 양봉은 오히려 추격 진입 신호.
            #    스코어 감점으로는 해결 안 됨(다른 가점들이 상쇄) → 하드 차단
            # 🟥 [FIX-HB1] 실측 15건 승률 20% 짜리 패턴. 감점으로는 다른 가점에 상쇄된다.
            #    (8/28 TLT 2건이 정확히 이 조건에 걸렸고 둘 다 손절)
            reasons.append(
                f"{HARD_BLOCK_TAG}|3봉 연속 양봉이지만 RSI<70(모멘텀 부족) → 추격 진입 위험"
            )
            signal_score -= 3.0

        else:
            signal_score += 0.5
            reasons.append(
                "🟢 최근 3봉 연속 양봉 + 상승추세 → BUY continuation 가점+0.5"
            )

        # 1) 패턴 그룹 먼저 정의
    # 🟥 [FIX-B6d] detect_candle_pattern()이 실제로 반환하는 라벨과 목록을 일치시킨다.
    #    PIERCING_LINE / DARK_CLOUD_COVER가 빠져 있어서, 같은 성격의 반전 패턴인데
    #    BULLISH_ENGULFING만 ±2를 받고 이 둘은 0점이 되는 비대칭이 있었다.
    #    반대로 MORNING_STAR / EVENING_STAR / HANGING_MAN은 생성되지 않으므로 제거.
    bullish_patterns = ["BULLISH_ENGULFING", "HAMMER", "PIERCING_LINE"]
    bearish_patterns = ["SHOOTING_STAR", "BEARISH_ENGULFING", "DARK_CLOUD_COVER"]
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
    # 🟥 [FIX-B5] 이 블록은 그대로 두면 이중 계산이 된다.
    #    must_capture_opportunity()는 이미 함수 맨 위(L757)에서 expected_direction=signal로
    #    호출돼 score에 반영됐다. 여기서 또 부르면 같은 점수를 두 번 더하게 된다.
    #    (기존엔 expected_direction=None이라 op_score가 0 이하로만 나와서 `if op_score > 0`이
    #     절대 참이 되지 않았고, 그래서 이중 계산이 우연히 안 일어났을 뿐이다.
    #     B1 수정으로 방향이 제대로 전달되기 시작하면 이 블록이 실제 버그가 된다.)
    #    → 계산 자체를 제거한다.

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

            # 🟥 [FIX-B2] 여기도 방향 조건 추가. 이미 저항 위로 뚫고 올라간 상태는
            #    "저항 근접(돌파 실패 위험)"이 아니라 "돌파 성공"이다.
            near_resistance = (
                resistance is not None and
                price is not None and
                atr is not None and
                resistance > price and
                (resistance - price) <= atr * 0.25
            )

            if (rsi is not None) and (rsi > 68) and near_resistance:

                signal_score -= 3.0
                reasons.append(
                    "🔴 과매수 + 저항선 매우 근접(ATR 기준) → late BUY / 돌파 실패 위험 (-3.0)"
                )

            # 🟥 [FIX-B3] "과매수면 무조건 감점" 로직 제거.
            #    실거래 665건 분석 결과 RSI 구간별 승률이 정반대였다:
            #      RSI 50~60 → 42.6% / 70~80 → 53.0% / 80~100 → 53.6%
            #    이 봇은 돌파·모멘텀 지속 전략인데, 반전(reversal) 매매용 과열 페널티를
            #    붙여놓아서 가장 잘 맞는 구간을 스스로 깎아내리고 있었다.
            #    → 저항 근접(위 조건)일 때만 감점하고, 단순 과매수는 감점하지 않는다.
            elif (rsi is not None) and (rsi > 68):
                reasons.append(
                    "🟢 과매수 구간 BUY — 모멘텀 전략에서는 오히려 승률이 높은 구간 "
                    "(실거래 RSI 70~80: 53.0%, 80~100: 53.6%) → 감점 없음"
                )

    except Exception as e:
        # 배포 중 예외로 전략이 멈추는 걸 방지 (안전장치)
        reasons.append(f"⚠️ 추세 말기 감점 필터 예외 발생(무시): {e}")
    

    return signal_score, reasons

app = FastAPI()

OANDA_API_KEY = os.getenv("OANDA_API_KEY")
ACCOUNT_ID = os.getenv("ACCOUNT_ID")
# ============================================================
# 🟥 [FIX-E2] OANDA 엔드포인트 하드코딩 제거.
#  기존엔 세 곳 모두 api-fxpractice(데모)로 박혀 있어서, 환경변수로 실계좌를
#  가리켜도 항상 데모로 나갔다. 반대로 누가 무심코 이 문자열만 바꾸면
#  나머지 두 곳과 어긋나 계좌가 섞이는 사고가 난다.
#  기본값은 안전하게 데모(practice) 유지. 실계좌는 OANDA_LIVE=true 로만 전환.
# ============================================================
OANDA_LIVE = os.getenv("OANDA_LIVE", "false").strip().lower() == "true"
OANDA_BASE_URL = "https://api-fxtrade.oanda.com" if OANDA_LIVE else "https://api-fxpractice.oanda.com"
print(f"🏦 OANDA 엔드포인트: {OANDA_BASE_URL} ({'실계좌' if OANDA_LIVE else '데모'})")
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
# 주식 신규 진입 컷오프 시각(미국 동부시간, 24시간 기준). 이 시각 이후 알림은 진입 안 함.
STOCK_ENTRY_CUTOFF_HOUR = int(os.getenv("STOCK_ENTRY_CUTOFF_HOUR", "15"))

# 🟦 포트폴리오에서 제거된 종목 — Pine 백테스트 자체가 낮거나 실거래 일관 손실
# 환경변수로 관리하거나 아래 리스트를 직접 수정
# 🟥 [FIX-EX1] CEG 를 기본 차단 목록에서 뺀다(사용자 요청 — 거래 허용).
#    다시 막고 싶으면 Render 환경변수 EXCLUDED_SYMBOLS 로 언제든 되돌릴 수 있다.
_EXCLUDED_SYMBOLS_ENV = os.getenv("EXCLUDED_SYMBOLS", "ALAB,VRT")
EXCLUDED_SYMBOLS = set(s.strip().upper() for s in _EXCLUDED_SYMBOLS_ENV.split(",") if s.strip())
# 결과추적/거래내역/성과분석 탭들을 몇 분마다 갱신할지 (기본 30분)
OUTCOME_TRACKER_INTERVAL_MINUTES = int(os.getenv("OUTCOME_TRACKER_INTERVAL_MINUTES", "30"))
# ============================================================
# 🟥 [FIX-H1] 시간청산 기본값 90분 → 0(비활성)
# ------------------------------------------------------------
#  90분 컷오프는 TP=ATR×2.4 목표와 수학적으로 모순이었다.
#    · 실거래 데이터: TP=ATR×1.0 도달까지 중앙 21분
#    · TP 거리를 2.0배로 늘리면 필요 시간은 약 2.0² = 4배 → 중앙 약 85분
#      (2.4 배였을 때는 2.4² ≈ 5.8배 → 중앙 122분 이었다)
#    · 즉 90분(15분봉 6개) 안에 2.4 ATR을 달성하라는 요구가 되어,
#      이길 거래의 절반 이상을 TP 도달 전에 강제로 잘라낸다.
#  게다가 실측상 90분 초과 보유 구간이 유일한 흑자 구간이었다:
#      90분 이내 137건 승률 41.6% / -$579
#      90분 초과  29건 승률 55.2% / +$210   ← 여기를 잘라내고 있었다
#  → 시간청산은 끄고, 오버나이트 위험은 장마감 전 전량청산(15:50)으로 막는다.
#    자본 회전이 필요하면 240~360 정도로 설정할 것. 90은 쓰지 말 것.
# ============================================================
STOCK_TIME_EXIT_MINUTES = int(os.getenv("STOCK_TIME_EXIT_MINUTES", "0"))
# 🟥 [FIX-A3] 시간청산 전용 루프 주기(분). 기존엔 30분짜리 시트 동기화 체인의
#    3번째 순서로 묶여 있어서, 앞 단계(수백 행 gspread 업데이트)가 느리면
#    시간청산 차례가 아예 안 왔다. 독립 루프로 분리하고 주기도 짧게 가져간다.
TIME_EXIT_CHECK_MINUTES = int(os.getenv("TIME_EXIT_CHECK_MINUTES", "5"))
# 🟥 [FIX-A3] 장마감 전 전량청산. 실거래 최대 손실이 전부 오버나이트 갭에서 나왔다.
#    1550 = 15:50 ET. 0 또는 STOCK_EOD_FLATTEN_ENABLED=false 로 끌 수 있다.
STOCK_EOD_FLATTEN_ENABLED = os.getenv("STOCK_EOD_FLATTEN_ENABLED", "true").strip().lower() != "false"
STOCK_EOD_FLATTEN_HHMM = int(os.getenv("STOCK_EOD_FLATTEN_HHMM", "1550"))

# ============================================================
# 🟥 [FIX-A1/A2] TP/SL 구조 전면 개편 (2026-08 실거래 166건 분석 결과 반영)
# ------------------------------------------------------------
#  문제 진단:
#   · 기존 SL = ATR × 1.0 은 15분봉 ATR 중앙값(0.689%)의 1.07배에 불과해
#     "정상 노이즈" 안에 손절선이 들어가 있었다. 실제로 5분 이내 청산 26건의
#     승률이 23.1%(-$264)로 전체 손실의 72%를 만들었다.
#   · 기존 TP = ATR × 1.0 → 설계 손익비 1.0. 여기에 진입 슬리피지(거래의 74%가
#     불리 체결)와 청산 슬리피지가 겹쳐 실현 손익비가 0.804까지 떨어졌고,
#     손익분기 필요 승률이 55.4%가 됐다(실제 승률 44.0%).
#
#  수정 방향:
#   · SL 을 ATR × 1.5 로 넓혀 손절선을 노이즈 밖으로 뺀다.
#   · TP 는 더 이상 ATR 배수로 "독립" 지정하지 않는다. 반드시 SL 거리 × 손익비로
#     파생시켜, 어떤 파라미터를 만져도 손익비가 절대 무너지지 않게 한다.
#     (기존 구조는 TP/SL 배수를 따로 두어 손익비가 조용히 1.0으로 붕괴했다.)
#
#  기본값: SL 1.5×ATR, TP 2.0×ATR (RR 1.333). 손익분기 필요 승률 42.9%.
#  (2026-08-31 실측 156건 기준 실제 승률 50.0% — 여유 7.1%p)
#  ⚠️ TradingView Pine 전략의 tpATR/slATR 입력값도 같이 맞춰야 정렬이 유지된다.
# ============================================================
# 🟥 [FIX-FX3] FX 가상 TP/SL 계산용 (알림에 tp/sl 이 없을 때만 사용)
FX_SL_ATR_MULT = float(os.getenv("FX_SL_ATR_MULT", "1.5"))
FX_RR_RATIO = float(os.getenv("FX_RR_RATIO", "1.6"))

# 🟥 [FIX-TO1] 캔들 조회 타임아웃 (연결, 읽기) 초.
#   TradingView 알림은 몰릴 수 있으므로 넉넉하되 무한대는 절대 안 된다.
CANDLE_CONNECT_TIMEOUT = float(os.getenv("CANDLE_CONNECT_TIMEOUT", "5"))
CANDLE_READ_TIMEOUT = float(os.getenv("CANDLE_READ_TIMEOUT", "20"))

# ============================================================
# 🟥 [FIX-ROLE1] '셋업 판단' 과 '상황 판단' 을 분리한다
# ------------------------------------------------------------
#  ■ 무엇이 문제였나
#    Pine 전략이 이미 "이게 내 셋업이다" 라고 판단해서 알람을 울렸는데,
#    봇의 점수 엔진이 **같은 것을 반대 기준으로 다시 판단**하고 있었다.
#
#      A12 는 돌파 전략이다 — "최근 고점을 뚫을 때 산다"
#      그런데 점수 엔진은 "저항에 가까우면 감점" 을 한다
#      → 돌파하는 순간에는 저항이 코앞이다. 그게 돌파의 정의다.
#
#    실측: 알람 100건 중 구조 손익비가 1.4 를 넘은 적이 **0건**.
#          76번 측정해서 최댓값 1.34. 즉 조건부 필터가 아니라
#          거의 모든 신호에 -2.0 을 붙이는 **상수 오프셋**이었다.
#
#  ■ 어떻게 바꾸나
#    · 셋업을 다시 판단하는 항목은 끈다 (구조 손익비 / 박스 상단 근접 /
#      RSI 중립 감점 / opportunity_score 역행)
#    · 전략이 볼 수 없는 것만 남긴다:
#      뉴스, 상위 시간축 충돌, 진짜 극단(RSI 80+ 등), 변동성 이상,
#      유동성 나쁜 시간대, 캔들 소진 신호
#
#  ■ 되돌리려면
#    SETUP_RECHECK_ENABLED=true 로 두면 예전 동작으로 돌아간다.
# ============================================================
SETUP_RECHECK_ENABLED = os.getenv("SETUP_RECHECK_ENABLED", "false").strip().lower() == "true"

# ============================================================
# 🟥 [FIX-HB1] '차단' 이라고 써놓고 점수만 깎던 것을 진짜 차단으로
# ------------------------------------------------------------
#  ■ 무엇이 문제였나 (2026-08-28 실측)
#    코드에 이렇게 돼 있었다:
#        # 스코어 감점으로는 해결 안 됨(다른 가점들이 상쇄) → 하드 차단
#        reasons.append("⛔ ... 진입 차단")
#        signal_score -= 3.0        ← 실제로는 그냥 감점
#    주석은 '하드 차단' 인데 구현은 감점이고, 진짜 차단은 threshold 가
#    대신 해주기를 기대하고 있었다. 그런데 threshold 를 -99 로 내리면서
#    이 장치들이 통째로 죽었다.
#
#    8/28 결과: TLT(-$4.26) + MU(-$59.86) = -$64.12
#    그날 손실 -$116.08 의 55% 가 여기서 나왔다.
#    (그중 '3봉양봉 + RSI<70' 은 실측 15건 승률 20% 로 만든 장치였다)
#
#  ■ 어떻게 고쳤나
#    reasons 에 기계가 읽을 수 있는 표식을 넣고, 실행 판정에서 그것만 보고 막는다.
#    threshold 와 완전히 무관해진다 → threshold 는 -99 로 둬도 안전하다.
# ============================================================
# ============================================================
# 🟥 [FIX-MACRO1] 거시 뉴스는 한 종목의 뉴스가 아니다
# ------------------------------------------------------------
#  ■ 무엇이 문제였나 (2026-08-28)
#    SPY 뉴스에 이게 잡혔다:
#      "Fed Funds Futures Now Price A September Rate Hike As More Likely Than A Hold"
#    그런데 봇은 그걸 SPY 개별 뉴스로만 취급해 점수 0 으로 흘렸고,
#    같은 시각 EUR_USD·AUD_USD·TLT 에는 '영향 있는 뉴스 없음' 이라고 기록했다.
#
#    실제로는 그 하나가 전부를 움직였다:
#      달러 강세 → EUR_USD·AUD_USD 하락
#      금리 상승 → TLT 하락
#      주식 하락 → META·MU·SPY 하락
#    그날 8건 전부 손절. 8개의 실패가 아니라 **하나의 사건에 8번 부딪힌 것**이다.
#
#  ■ 어떻게 고쳤나
#    어느 종목에서든 거시 키워드가 잡히면 '시장 전체 경보' 를 켜고,
#    그 시간 동안은 모든 종목의 신규 진입을 막는다.
# ============================================================
# ============================================================
# 🟦 [FIX-COST1 — 폐기됨. 아래 FIX-COST4 로 대체] 비용을 'ATR 대비' 로 재던 시절의 기록.
#    측정치(8/28 슬리피지)는 여전히 유효하므로 남겨둔다. 판정 방식만 바뀌었다.
# ------------------------------------------------------------
#  ■ 8/28 실측 (알림가 → 실제 체결가)
#      SPY   슬립 $0.010 / ATR 1.32   =  0.8%  ✅
#      META  슬립 $0.135 / ATR 3.99   =  3.4%  ✅
#      MU    슬립 $0.503 / ATR 9.72   =  5.2%  ✅
#      TLT   슬립 $0.015 / ATR 0.092  = 16.3%  ❌
#      EWJ   슬립 $0.064 / ATR 0.246  = 26.0%  ❌
#
#    처음엔 'ATR/주가가 낮으면 문제' 라고 봤는데 틀렸다.
#    SPY 는 ATR/주가가 0.17% 로 낮은데도 가장 저렴하고,
#    EWJ 는 0.25% 로 더 큰데 26% 를 먹는다. **유동성이 결정한다.**
#    → 당시 결론: 이동폭을 ATR 로 나눠서 판정한다.
#    ⚠️ 이 결론은 2026-08-31 에 폐기됐다. 개장 직후 ATR 이 시간외 봉을 재는 바람에
#       ATR 기반 판정이 통째로 무의미해지는 게 확인됐기 때문이다. 지금은 이동폭을
#       '가격의 %' 로만 본다 (FIX-COST4).
# ============================================================
# ============================================================
# 🟥 [FIX-COST4] ATR 비율 게이트 삭제 — 절대 이동폭 하나만 남긴다 (2026-08-31)
# ------------------------------------------------------------
#  삭제 이유:
#    · 12% → 20% 로 올렸지만, 두 숫자 다 내가 데이터 없이 정한 값이었다.
#      "ATR 의 12%" 는 사람이 판단할 수 있는 단위가 아니다.
#    · 실측: 알람 124건 중 이 게이트가 막은 건 4건(3.2%), 그중 3건은
#      개장 직후 ATR stale 버그였다. 실질 효과가 거의 없었다.
#    · 결정적으로 — 개장 직후엔 ATR 자체를 못 믿는다. ATR 에서 파생된 어떤 기준도
#      그 시간대엔 무의미하다. 그래서 예외 창(09:30~10:15)까지 따로 만들어야 했다.
#      기준을 ATR 에서 떼면 그 특수 케이스가 통째로 사라진다.
#
#  남기는 것: '신호가 대비 몇 % 밀렸나' 하나.
#    단위가 명확하고(가격의 %), 개장이든 장중이든 똑같이 동작하며, 예외가 필요 없다.
#    실측 참고(2026-08-31 09:45): TSLA 0.185% / PANW 0.116% / PLTR 0.059%
#
#  ⚠️ 알려진 구멍: ATR 이 아주 작은 종목(예: 채권 ETF, ATR 0.17%)에서는
#     0.30% 이동이 ATR 을 통째로 넘는데도 통과한다. 지금 종목군(TSLA·PLTR·PANW·
#     ANET·NVDA·META·GEV·MU·AMAT·APP·CRWV·TZA)은 전부 변동성이 커서 문제가 안 되지만,
#     저변동성 종목을 편입하면 이 게이트를 다시 설계해야 한다.
#
#  ⚠️ 0.30 도 아직 근거 있는 값이 아니다. 그래서 이제 '차단 여부와 무관하게'
#     매 주문의 실제 이동폭을 로그에 남긴다. 분포가 쌓이면 그때 숫자로 정한다.
# ============================================================
COST_DRIFT_GATE_ENABLED = os.getenv("COST_DRIFT_GATE_ENABLED",
                                    os.getenv("COST_ATR_GATE_ENABLED", "true")).strip().lower() != "false"
COST_MAX_DRIFT_PCT = float(os.getenv("COST_MAX_DRIFT_PCT", "0.30"))

MACRO_NEWS_ENABLED = os.getenv("MACRO_NEWS_ENABLED", "true").strip().lower() != "false"
MACRO_BLOCK_MINUTES = int(os.getenv("MACRO_BLOCK_MINUTES", "90"))

#: 시장 전체를 움직이는 주제. 한 종목에서 잡혀도 전 종목에 적용한다.
_MACRO_KEYWORDS = [
    "fed ", "fomc", "federal reserve", "rate hike", "rate cut", "interest rate",
    "fed funds", "powell", "cpi", "inflation", "pce", "jobs report", "payroll",
    "nonfarm", "unemployment", "gdp", "recession", "tariff", "central bank",
    "ecb", "boj", "treasury yield", "연준", "금리", "물가", "고용지표", "관세",
]

#: {"until": epoch, "headline": str, "source": symbol}
_macro_alert = {"until": 0.0, "headline": "", "source": ""}


def _scan_macro_news(symbol: str, news_text: str):
    """뉴스에 거시 키워드가 있으면 시장 전체 경보를 켠다."""
    if not (MACRO_NEWS_ENABLED and news_text):
        return
    low = str(news_text).lower()
    # ⚠️ 단순 부분일치는 오탐이 난다: "fed " 는 "briefed ", "beefed " 에도 걸린다.
    #    짧고 모호한 단어는 단어경계(\b)로 검사한다. 한 번 오탐하면 90분간 전 종목이 막힌다.
    hit = None
    for k in _MACRO_KEYWORDS:
        kk = k.strip()
        if len(kk) <= 4 and kk.isascii():
            if _re.search(r"\b" + _re.escape(kk) + r"\b", low):
                hit = kk; break
        elif kk in low:
            hit = kk; break
    if not hit:
        return
    _macro_alert["until"] = _t.time() + MACRO_BLOCK_MINUTES * 60
    _macro_alert["headline"] = str(news_text)[:200]
    _macro_alert["source"] = symbol
    print("=" * 70)
    print(f"🌐 [거시경보] {symbol} 뉴스에서 '{hit.strip()}' 감지 → "
          f"앞으로 {MACRO_BLOCK_MINUTES}분간 **모든 종목** 신규 진입 차단")
    print(f"   {str(news_text)[:160]}")
    print("=" * 70)


def _macro_block_active() -> tuple[bool, str]:
    if not MACRO_NEWS_ENABLED:
        return False, ""
    left = _macro_alert["until"] - _t.time()
    if left <= 0:
        return False, ""
    return True, (f"거시 뉴스 경보({_macro_alert['source']} 발) — "
                  f"{left/60:.0f}분 남음: {_macro_alert['headline'][:110]}")


HARD_BLOCK_TAG = "⛔HARDBLOCK"
HARD_BLOCK_ENABLED = os.getenv("HARD_BLOCK_ENABLED", "true").strip().lower() != "false"


def _hard_blocked(reasons) -> tuple[bool, str]:
    """reasons 안에 하드 차단 표식이 있으면 (True, 사유)."""
    if not HARD_BLOCK_ENABLED:
        return False, ""
    for r in reasons or []:
        s = str(r)
        if s.startswith(HARD_BLOCK_TAG):
            return True, s.split("|", 1)[-1].strip()
    return False, ""

# ============================================================
# 🟥 [FIX-SLTP1] 어떤 전략은 손절을 'ATR' 이 아니라 '구조' 에서 잡는다
# ------------------------------------------------------------
#  개장 09:31 의 1분봉 ATR(14) 은 직전 14분 = 거의 거래 없는 장전 봉이다.
#  그 값으로 손절을 잡으면 폭이 비정상적으로 좁아 몇 초 만에 잘린다.
#  Gap-and-Go 교본이 ATR 대신 '첫 1분봉의 저가' 를 쓰는 이유가 이것이다.
#  → 아래 목록에 든 전략은 알림이 보낸 tp/sl 을 그대로 존중한다.
#    단, 값이 이상하면(방향 반대·0·비정상적으로 멀거나 좁음) ATR 로 되돌린다.
# ============================================================
ALERT_SLTP_STRATEGIES = set(
    s.strip().upper() for s in
    os.getenv("ALERT_SLTP_STRATEGIES", "GAP_AND_GO_G3").split(",") if s.strip()
)
#  ⚠️ 상한을 넉넉히 잡으면 안 된다. 기본 사이징(tiered)은 수량을 '가격'으로만 정하고
#     손절 거리를 보지 않는다. 즉 손절이 넓어져도 주식 수가 그대로라 **위험만 커진다.**
#     ($20 주식 60주에 8% 손절 = 1회 $95 손실. ATR 손절이었다면 $4 수준)
#     → G3 자체 상한(3%)과 같은 값으로 맞춘다.
ALERT_SL_MAX_PCT = float(os.getenv("ALERT_SL_MAX_PCT", "3"))
#  ⚠️ 하한도 G3 자체 하한(0.15%)과 맞춘다. 0.05% 는 $20 주식에서 1센트라 사실상 무의미.
ALERT_SL_MIN_PCT = float(os.getenv("ALERT_SL_MIN_PCT", "0.15"))
#  알림이 계산한 가격과 봇이 다시 받은 가격이 이보다 벌어지면 구조값을 못 믿는다
ALERT_SLTP_MAX_DRIFT_PCT = float(os.getenv("ALERT_SLTP_MAX_DRIFT_PCT", "0.5"))

# ============================================================
# 🟥 [FIX-FXSL1] FX 도 알림의 TP/SL 을 쓸 수 있게 한다.
# ------------------------------------------------------------
#  기존: 위 구조적 TP/SL 블록에 is_stock_pair(pair) 게이트가 걸려 있어서
#        FX 는 Pine 이 보낸 tp/sl 이 **주문에 단 한 번도 반영되지 않았다.**
#        FX 의 실제 TP/SL 은 (1) GPT 가 말한 값, 아니면
#        (2) calculate_realistic_tp_sl() + adjust_tp_sl_for_structure()
#            → FX_MIN_RR_RATIO(1.8) 강제 확대, 로 정해졌다.
#        즉 스캘핑처럼 'TP/SL 자체가 전략'인 경우 백테스트와 실전이 완전히 달라진다.
#  수정: 전략 이름이 ALERT_SLTP_STRATEGIES 에 있으면 FX 도 알림값을 존중한다.
#  ⚠️ 임계값은 주식과 같이 쓰면 안 된다. 주식 하한 0.15% 는 USD/JPY 159.9 에서
#     24pip 이라 스캘핑 손절(6~15pip)이 전부 거부된다. FX 전용 값을 따로 둔다.
ALERT_SL_FX_MIN_PCT = float(os.getenv("ALERT_SL_FX_MIN_PCT", "0.02"))   # 159.9 기준 약 3.2pip
ALERT_SL_FX_MAX_PCT = float(os.getenv("ALERT_SL_FX_MAX_PCT", "0.60"))   # 159.9 기준 약 96pip
#  드리프트 한도도 마찬가지. 0.5% = 80pip 은 FX 에서 사실상 무제한이라 의미가 없다.
ALERT_SLTP_FX_MAX_DRIFT_PCT = float(os.getenv("ALERT_SLTP_FX_MAX_DRIFT_PCT", "0.05"))  # 약 8pip

STOCK_SL_ATR_MULT = float(os.getenv("STOCK_SL_ATR_MULT", "1.5"))
# ============================================================
# 🟥 [FIX-TP1] TP 배수를 1차 손잡이로 승격 (2026-08-31)
# ------------------------------------------------------------
#  기존: RR(1.6) 이 기준이고 TP 배수는 파생값 → 사용자는 "TP = ATR×2.4" 로 생각하는데
#        손잡이는 1.6 이라 직관과 어긋났다. 이제 TP 배수를 직접 설정한다.
#  변경: TP = ATR×2.4 → ATR×2.0 (RR 1.6 → 1.333)
#
#  근거 (Alpaca 실거래 156건, 2026-08-31 실측):
#    · TP 도달 58건 / SL 도달 67건 / TIME_EXIT 31건, 승률 50.0%, PF 1.257
#    · TP 를 0.833배(2.4→2.0) 로 줄였을 때의 하한 시뮬레이션:
#        총 수익률합 +13.55% → +3.27%  (손절건의 TP 전환을 0으로 놓은 최악 가정)
#      본전이 되려면 손절 67건 중 6.3건(9%)만 TP 로 바뀌면 된다. 낮은 문턱이다.
#    · 손익분기 승률 38.5% → 42.9%. 실제 승률 50.0% 이므로 여전히 여유가 있다.
#  ⚠️ 단, MFE(손절 전 최대 유리 이동)를 아직 기록하지 않아 전환 건수를 정확히 셀 수 없다.
#     250건 시점에 재측정할 것. MFE 추적이 붙으면 이 값을 근거 있게 확정할 수 있다.
STOCK_TP_ATR_MULT = float(os.getenv("STOCK_TP_ATR_MULT", "2.0"))
#  하위호환: STOCK_RR_RATIO 를 환경변수로 직접 넣으면 그쪽이 이긴다.
_rr_env = os.getenv("STOCK_RR_RATIO", "").strip()
if _rr_env:
    STOCK_RR_RATIO = float(_rr_env)
    STOCK_TP_ATR_MULT = STOCK_SL_ATR_MULT * STOCK_RR_RATIO
else:
    STOCK_RR_RATIO = STOCK_TP_ATR_MULT / STOCK_SL_ATR_MULT if STOCK_SL_ATR_MULT else 1.6

# FX(OANDA)측 최소 손익비. adjust_tp_sl_for_structure 에서 강제한다.
FX_MIN_RR_RATIO = float(os.getenv("FX_MIN_RR_RATIO", "1.8"))

# ============================================================
# 🟥 [FIX-A4] 레버리지/인버스 ETF · 페니주 진입 차단
# ------------------------------------------------------------
#  TZA(3배 인버스 소형주 ETF)를 상승장에서 롱으로 26회 매수해 -$86 손실.
#  레버리지 ETF는 일간 리밸런싱 구조상 변동성 감쇠가 있어 이 전략의 대상이 아니다.
#  '오늘의 추천 후보' 스캐너가 거래량만 보고 SOXS·페니주를 공급하던 것도 같이 막는다.
# ============================================================
_LEVERAGED_ETF_DEFAULT = (
    "TZA,TNA,SOXL,SOXS,SPXL,SPXS,TQQQ,SQQQ,UPRO,SPXU,UDOW,SDOW,"
    "LABU,LABD,FAS,FAZ,YINN,YANG,NUGT,DUST,JNUG,JDST,BOIL,KOLD,"
    "UVXY,SVXY,VXX,VIXY,UVIX,SVIX,BITO,BITX,ETHU,TSLL,TSLQ,NVDL,NVD,"
    "AGQ,ZSL,UCO,SCO,ERX,ERY,DRN,DRV,CURE,WEBL,WEBS,BULZ,FNGD,FNGU"
)
_LEVERAGED_ETF_ENV = os.getenv("LEVERAGED_ETF_BLOCKLIST", _LEVERAGED_ETF_DEFAULT)
LEVERAGED_ETFS = set(s.strip().upper() for s in _LEVERAGED_ETF_ENV.split(",") if s.strip())
# 이 가격 미만 종목은 진입 금지(페니주 배제). 스프레드·슬리피지가 엣지를 삼킨다.
MIN_STOCK_PRICE = float(os.getenv("MIN_STOCK_PRICE", "5.0"))
# 진입 허용 최대 가격(0이면 무제한). 초고가주는 티어 수량 2주로 노출이 과도해진다.
MAX_STOCK_PRICE = float(os.getenv("MAX_STOCK_PRICE", "0"))


def is_blocked_instrument(pair: str, price: float | None = None) -> tuple[bool, str]:
    """
    [FIX-A4] 진입 자체를 막아야 하는 종목인지 판정.
    반환: (차단여부, 사유). FX는 항상 통과.
    """
    if not is_stock_pair(pair):
        return False, ""
    sym = (pair or "").upper().strip()
    if sym in EXCLUDED_SYMBOLS:
        return True, "EXCLUDED_SYMBOL"
    if sym in LEVERAGED_ETFS:
        return True, "LEVERAGED_ETF"
    try:
        p = float(price) if price is not None else None
    except (TypeError, ValueError):
        p = None
    if p is not None:
        if MIN_STOCK_PRICE > 0 and p < MIN_STOCK_PRICE:
            return True, f"PENNY_STOCK(<{MIN_STOCK_PRICE:g})"
        if MAX_STOCK_PRICE > 0 and p > MAX_STOCK_PRICE:
            return True, f"PRICE_TOO_HIGH(>{MAX_STOCK_PRICE:g})"
    return False, ""

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


# ============================================================
# 🟥 [FIX-S1] 알림 심볼 → 브로커가 이해하는 형식으로 정규화
# ------------------------------------------------------------
#  ■ 무엇이 문제였나
#    BUY_STOCK_PORTFOLIO_A10 은 알림에 "symbol": syminfo.ticker 를 보낸다.
#    주식 차트에서는 "PLTR" 이라 문제가 없었다.
#    그런데 OANDA 차트에 같은 스크립트를 걸면 syminfo.ticker 가
#        USDJPY / XAUUSD / WTICOUSD / SPX500USD
#    처럼 **밑줄 없는** 형태로 온다. OANDA API 는 USD_JPY 를 요구한다.
#    → /v3/instruments/USDJPY/candles 요청이 400 으로 죽고,
#      get_candles 가 빈 DataFrame 을 돌려주고,
#      current_price=None 으로 웹훅이 **시트에 한 줄도 안 남기고** 종료됐다.
#    거래도 안 되고 기록도 안 남는 이유가 이것이다. 로그에도 "캔들 요청 실패"
#    한 줄만 찍혀서 원인을 알기 어려웠다.
#
#  ■ 어떻게 고쳤나
#    끝 3글자가 통화코드면 밑줄을 넣어준다 (XAUUSD → XAU_USD).
#    주식 티커(1~5글자)는 건드리지 않는다 — 미국 주식은 5글자를 넘지 않으므로
#    6글자 이상만 변환 대상이 되어 안전하다.
#    그리고 OANDA 계좌의 실제 거래가능 목록과 대조해서, 없는 상품이면
#    조용히 죽지 않고 이유를 남긴다.
# ============================================================

#: OANDA 상품 이름의 끝에 올 수 있는 통화 코드
_QUOTE_CCY = {
    "USD", "EUR", "JPY", "GBP", "CHF", "CAD", "AUD", "NZD",
    "HKD", "SGD", "SEK", "NOK", "DKK", "MXN", "ZAR", "TRY",
    "PLN", "CZK", "HUF", "CNH", "THB", "INR", "SAR",
}

_OANDA_INSTR_CACHE = {"t": 0.0, "set": set()}
OANDA_INSTR_TTL = float(os.getenv("OANDA_INSTR_TTL", "86400"))


def get_oanda_instruments() -> set:
    """이 계좌에서 실제로 거래 가능한 OANDA 상품 이름 집합. 24시간 캐시.
    조회 실패 시 빈 집합 (그러면 검증을 건너뛰고 변환 결과를 그대로 쓴다)."""
    now = _t.time()
    if _OANDA_INSTR_CACHE["set"] and (now - _OANDA_INSTR_CACHE["t"]) < OANDA_INSTR_TTL:
        return _OANDA_INSTR_CACHE["set"]
    if not (OANDA_API_KEY and ACCOUNT_ID):
        return set()
    try:
        r = requests.get(f"{OANDA_BASE_URL}/v3/accounts/{ACCOUNT_ID}/instruments",
                         headers={"Authorization": f"Bearer {OANDA_API_KEY}"}, timeout=15)
        if r.ok:
            names = {i.get("name", "") for i in (r.json() or {}).get("instruments", [])}
            names.discard("")
            if names:
                _OANDA_INSTR_CACHE.update(t=now, set=names)
                print(f"📋 [OANDA] 거래가능 상품 {len(names)}개 로드")
                return names
        print(f"⚠️ [OANDA] 상품목록 조회 status={r.status_code}")
    except Exception as e:
        print(f"⚠️ [OANDA] 상품목록 조회 실패: {e}")
    return _OANDA_INSTR_CACHE["set"]


def normalize_symbol(raw: str) -> tuple[str, str]:
    """
    알림 심볼을 브로커 형식으로. return (정규화된 심볼, 설명)

    'PLTR'      → ('PLTR',      '주식')            그대로 Alpaca
    'USD_JPY'   → ('USD_JPY',   '이미 정상')
    'USDJPY'    → ('USD_JPY',   '밑줄 삽입')       ← A10 이 보내던 형태
    'XAUUSD'    → ('XAU_USD',   '밑줄 삽입')
    'SPX500USD' → ('SPX500_USD','밑줄 삽입')
    'ZZZZZZ'    → ('ZZZZZZ',    '변환 불가')       호출부에서 차단
    """
    s = (raw or "").strip().upper().replace("/", "_").replace("-", "_")
    if not s:
        return "", "빈 심볼"
    if is_stock_pair(s):
        return s, "주식"
    if "_" in s:
        return s, "이미 정상"
    # 끝 3글자가 통화코드면 나눠준다
    if len(s) > 3 and s[-3:] in _QUOTE_CCY:
        cand = f"{s[:-3]}_{s[-3:]}"
        known = get_oanda_instruments()
        if not known or cand in known:
            return cand, "밑줄 삽입"
        # 변환은 됐는데 계좌에서 거래 불가 — 비슷한 이름을 찾아 알려준다
        near = [k for k in known if k.replace("_", "") == s][:3]
        if near:
            return near[0], f"밑줄 삽입(계좌 목록에서 매칭: {near[0]})"
        return cand, f"⚠ {cand} 이(가) 계좌 거래가능 목록에 없음"
    known = get_oanda_instruments()
    near = [k for k in known if k.replace("_", "") == s][:1]
    if near:
        return near[0], "계좌 목록에서 매칭"
    return s, "변환 불가"


def price_round_digits(pair: str) -> int:
    """주문 가격(TP/SL) 반올림 자릿수. 주식은 센트 단위(2자리)."""
    if is_stock_pair(pair):
        return 2
    return 3 if pair.endswith("JPY") else 5


# ============================================================
# 🟥 [FIX-T1] 분석 시간축을 '알림이 온 차트'에 맞춘다
# ------------------------------------------------------------
#  ■ 무엇이 문제였나
#    이 함수는 주식이면 무조건 M15, FX면 무조건 M30 을 돌려줬다.
#    그런데 TP/SL 은 여기서 받은 캔들로 계산한 ATR 로 정해진다.
#    → 트레이딩뷰 차트를 1시간봉으로 바꿔서 알림을 걸면
#      신호는 1시간짜리인데 손절은 15분 ATR 로 잡힌다.
#      1시간 ATR 은 15분 ATR 의 약 2배다. 즉 **손절이 필요치의 절반**이 되어
#      정상적인 흔들림에도 계속 잘려나간다.
#    설정 어디에도 안 나타나고 로그도 정상으로 보인다. 조용히 지는 종류의 버그다.
#
#  ■ 어떻게 고쳤나
#    Pine 알림이 이미 "tf": timeframe.period 를 보내고 있었는데 봇이 안 읽고 있었다.
#    이제 그 값을 읽어 분석 시간축으로 쓴다.
#    → 차트를 1시간으로 바꾸면 봇도 자동으로 1시간 ATR 을 쓴다. 설정 불필요.
#    tf 가 없는 옛 알림은 아래 기본값으로 폴백한다.
# ============================================================
STOCK_BASE_GRANULARITY = os.getenv("STOCK_BASE_GRANULARITY", "M15")
FX_BASE_GRANULARITY = os.getenv("FX_BASE_GRANULARITY", "M30")

#: 이번 웹훅 요청의 시간축. 요청마다 독립적으로 유지된다.
_ALERT_TF = contextvars.ContextVar("alert_tf", default=None)

#: 🟥 [FIX-SLTP1] 이번 주문의 TP/SL 이 '구조적 레벨' 인가(=거리가 아니라 가격).
#   True 면 주문 직전 실시간가 보정에서 TP/SL 을 밀지 않는다.
_STRUCTURAL_SLTP = contextvars.ContextVar("structural_sltp", default=False)

#: TradingView timeframe.period → 봇 granularity
_TV_TF_MAP = {
    "1": "M1", "3": "M5", "5": "M5", "10": "M15", "15": "M15",
    "30": "M30", "45": "M30", "60": "H1", "120": "H1", "180": "H4",
    "240": "H4", "360": "H4", "480": "H4", "720": "H4",
    "1D": "D", "D": "D", "1W": "W", "W": "W",
}

#: Alpaca(주식)에서 실제로 조회 가능한 단위.
_STOCK_SUPPORTED = {"M1", "M5", "M15", "M30", "H1", "H4", "D"}


def normalize_alert_timeframe(raw, pair: str = "") -> str | None:
    """알림의 tf 문자열을 봇 granularity 로. 못 알아보면 None."""
    if raw is None:
        return None
    s = str(raw).strip().upper()
    if not s:
        return None
    gran = _TV_TF_MAP.get(s)
    if gran is None and s in ("M1", "M5", "M15", "M30", "H1", "H2", "H4", "D", "W"):
        gran = "H4" if s == "H2" else s      # H2 는 양쪽 다 애매해서 H4 로 올린다
    if gran is None:
        return None
    if pair and is_stock_pair(pair) and gran not in _STOCK_SUPPORTED:
        return None
    return gran


def base_granularity_for(pair: str) -> str:
    """
    분석 기준 캔들 단위.

    1순위: 이번 알림이 온 차트의 시간축 (Pine 이 보낸 "tf")
    2순위: 환경변수 STOCK_BASE_GRANULARITY / FX_BASE_GRANULARITY
    (캔들 조회, ATR→TP/SL, 지지/저항, MTF 요약, GPT 프롬프트가 전부 이 값을 따른다)
    """
    tf = _ALERT_TF.get()
    if tf:
        return tf
    return STOCK_BASE_GRANULARITY if is_stock_pair(pair) else FX_BASE_GRANULARITY


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

# ============================================================
# 🟥 [FIX-J1] TradingView "request took too long and timed out" 해결
# ------------------------------------------------------------
#  증상: TradingView 알림 로그에서 BUY 신호만 전부 빨간 실패,
#        KEEP_ALIVE 핑은 정상 성공.
#  원인: TradingView 웹훅은 응답을 몇 초(약 3~5초)만 기다린다.
#        그런데 기존 코드는 `await`로 처리 완료까지 붙잡고 있었고,
#        그 처리에는 캔들 200개 조회 → 지표 계산 → 뉴스 조회 → MTF 조회
#        → GPT 호출(최대 60초) → 주문 전송 → 구글시트 여러 번 쓰기가 들어 있다.
#        실제로 알림 09:45:11 → 시트 기록 09:45:40 으로 약 30초가 걸렸다.
#        KEEP_ALIVE만 성공한 건 그건 즉시 return하도록 따로 빼놨기 때문이다.
#
#  ⚠️ 중요: 이 "실패"는 주문이 안 나갔다는 뜻이 아니다.
#     TradingView가 기다리다 포기했을 뿐, 서버는 끝까지 처리해서 주문까지 넣었다.
#     (그래서 시트에는 EXECUTED_BUY와 TP_HIT이 정상적으로 남아 있었다)
#     다만 그대로 두면 ① TradingView가 재전송해 중복 주문 위험 ② 알림 로그가
#     전부 빨간색이라 진짜 장애를 구분할 수 없다.
#
#  해결: 접수 즉시 202를 돌려주고, 실제 처리는 백그라운드로 넘긴다.
#        응답 시간이 수십 ms로 떨어져 TradingView는 항상 성공으로 표시된다.
# ============================================================
_bg_tasks: set = set()
_bg_lock = threading.Lock()
_bg_running = 0


# ============================================================
# 🟥 [FIX-WJ1] 도착한 웹훅을 전부 기록해 두는 짧은 장부
# ------------------------------------------------------------
#  "트레이딩뷰에서 알람은 울렸는데 시트에 아무것도 없다" 를 지금까지는
#  Render 로그를 뒤져서 추측할 수밖에 없었다. 그런데 로그는 금방 흘러가고,
#  '요청이 아예 안 온 것' 과 '와서 조용히 죽은 것' 이 구분이 안 된다.
#  이 둘은 고치는 방법이 완전히 다르다(트레이딩뷰 설정 vs 봇 코드).
#  → 도착한 모든 요청을 메모리에 200건까지 남기고 /recent_webhooks 로 본다.
#    여기에 없으면 **요청이 안 온 것이다** — 봇 문제가 아니라는 뜻.
# ============================================================
_WEBHOOK_JOURNAL_MAX = int(os.getenv("WEBHOOK_JOURNAL_MAX", "200"))
_webhook_journal: deque = deque(maxlen=_WEBHOOK_JOURNAL_MAX)
_wj_lock = threading.Lock()


def _journal_webhook(raw: bytes, source: str = "/webhook"):
    """요청 도착 즉시 기록. 이후 결과는 _journal_outcome() 이 채운다."""
    try:
        body = (raw or b"")[:400].decode("utf-8", "ignore")
        try:
            sym = (json.loads(body or "{}") or {}).get("symbol") \
                  or (json.loads(body or "{}") or {}).get("pair") or "?"
        except Exception:
            sym = "?(JSON파싱실패)"
        rec = {
            "seq": len(_webhook_journal) + 1,
            "도착": datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M:%S ET"),
            "심볼": sym,
            "경로": source,
            "본문": body,
            "결과": "처리중",
        }
        with _wj_lock:
            _webhook_journal.append(rec)
        return rec
    except Exception as e:
        print(f"⚠️ [웹훅장부] 기록 실패(무시): {e}")
        return None


def _journal_outcome(rec, outcome: str):
    if rec is not None:
        try:
            rec["결과"] = str(outcome)[:200]
        except Exception:
            pass


def _run_webhook_bg(raw: bytes):
    """백그라운드 실행 래퍼 — 예외를 삼키지 않고 로그로 남긴다."""
    global _bg_running
    with _bg_lock:
        _bg_running += 1
        n = _bg_running
    if n > 10:
        print(f"⚠️ [웹훅] 동시 처리 {n}건 — 알림이 몰리고 있습니다(처리 지연 가능)")
    _t0 = _t.time()
    _rec = _journal_webhook(raw, "/webhook")
    try:
        _out = process_webhook_sync(raw)
        try:
            _body = getattr(_out, "body", b"") or b""
            _journal_outcome(_rec, _body[:180].decode("utf-8", "ignore") or "완료")
        except Exception:
            _journal_outcome(_rec, "완료")
        return _out
    except Exception as e:
        import traceback
        _journal_outcome(_rec, f"❌ 예외: {e}")
        print(f"❌ [웹훅 백그라운드] 처리 중 예외: {e}")
        traceback.print_exc()
    finally:
        _elapsed = _t.time() - _t0
        print(f"⏱️ [웹훅] 처리 완료 — {_elapsed:.1f}초 소요 "
              f"(TradingView 대기 한도는 3~5초라, 동기 처리였다면 여기서 타임아웃)")
        with _bg_lock:
            _bg_running -= 1


@app.post("/webhook")
async def webhook(request: Request):
    """
    🟥 [FIX-J1] 즉시 202를 반환하고 실제 처리는 백그라운드로 넘긴다.
    TradingView는 응답 본문을 쓰지 않으므로, 빨리 200/202를 주는 것이 유일하게 중요하다.
    """
    raw = (await request.body()) or b""
    task = asyncio.create_task(asyncio.to_thread(_run_webhook_bg, raw))
    # create_task 결과를 어디에도 안 붙들면 GC가 태스크를 회수할 수 있다 → 참조 유지
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)
    return JSONResponse(content={"status": "accepted"}, status_code=200)


@app.post("/webhook_sync")
async def webhook_sync(request: Request):
    """
    디버깅용 — 처리 완료까지 기다렸다가 결과를 그대로 돌려준다.
    (curl로 수동 테스트할 때 사용. TradingView에는 절대 이 주소를 쓰지 말 것)
    """
    raw = (await request.body()) or b""
    return await asyncio.to_thread(process_webhook_sync, raw)


def process_webhook_sync(raw: bytes):
    print("✅ STEP 1: 웹훅 진입")
    # 🟥 [FIX-E3] 전역 10분 쿨다운은 완전히 죽은 코드였다.
    #    _last_execution_time이 0.0으로 선언된 뒤 어디서도 갱신되지 않아서
    #    (current_time - 0)이 항상 600보다 커 한 번도 차단된 적이 없다.
    #    그리고 애초에 "전 종목 공통 10분 쿨다운"은 이 전략(여러 종목 동시 운용)과
    #    맞지 않는다. 종목별 쿨다운(check_symbol_repeat_cooldown)이 이미 있으므로
    #    전역 쿨다운은 기본 비활성으로 두되, 필요하면 env로 켤 수 있게 한다.
    global _last_execution_time
    current_time = _t.time()
    if GLOBAL_COOLDOWN_SECONDS > 0 and (current_time - _last_execution_time) < GLOBAL_COOLDOWN_SECONDS:
        print(f"⚠️ [차단] 전역 쿨다운 중 (경과: {int(current_time - _last_execution_time)}초 "
              f"/ 설정 {GLOBAL_COOLDOWN_SECONDS}초)")
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
    # 🟥 [FIX-S1] A10 이 OANDA 차트에서 보내는 'USDJPY' 형태를 'USD_JPY' 로 바로잡는다.
    #    이걸 안 하면 캔들 조회가 400 으로 죽고 시트에 한 줄도 안 남는다.
    if pair:
        _pair_before = str(pair).strip().upper()
        pair, _pair_note = normalize_symbol(pair)
        if pair != _pair_before:
            print(f"🔤 [심볼] {_pair_before} → {pair} ({_pair_note})")
        if _pair_note in ("변환 불가",) or _pair_note.startswith("⚠"):
            print(f"❌ [심볼] {_pair_before}: {_pair_note} — 거래 불가. 알림 심볼을 확인할 것.")
            _log_blocked_alert(pair, data.get("signal"), data.get("alert_name"),
                               f"SYMBOL_UNRESOLVED: {_pair_note}")
            return JSONResponse(content={
                "status": "blocked", "reason": "symbol_unresolved",
                "pair": _pair_before, "note": _pair_note}, status_code=200)
    signal = data.get("signal")
    print(f"✅ STEP 2: 데이터 수신 완료 | pair: {pair}")

    # 🟥 [FIX-G2] 헬스체크/keep-alive 핑을 매매 신호로 처리하지 않는다.
    #    로그에 계속 찍히던 것:
    #      캔들 요청 실패: 400 ... /v3/instruments/KEEP_ALIVE/candles ... → POST /webhook 400
    #    Render 슬립 방지용 핑이 OANDA 캔들 조회까지 타고 들어가서 매번 400을 냈다.
    #    (지표·GPT는 안 탔지만 로그를 더럽히고 불필요한 외부 호출을 발생시켰다)
    if str(pair or "").strip().upper() in ("KEEP_ALIVE", "KEEPALIVE", "PING", "HEALTH", "HEALTHCHECK"):
        return JSONResponse(content={"status": "alive", "ts": datetime.now(ZoneInfo("UTC")).isoformat()})

    # 🟥 [FIX-T1] 알림이 온 차트의 시간축을 이번 요청 내내 쓰도록 세팅한다.
    #    이게 없으면 1시간봉 차트에서 알림이 와도 봇은 15분 ATR 로 손절을 잡는다.
    _tf_raw = data.get("tf") or data.get("timeframe") or data.get("interval")
    _tf_norm = normalize_alert_timeframe(_tf_raw, pair)
    if _tf_norm:
        _ALERT_TF.set(_tf_norm)
        print(f"⏱️ [시간축] {pair} 알림 tf={_tf_raw} → 분석 {_tf_norm} (ATR·TP/SL 이 기준으로 계산됨)")
    else:
        _ALERT_TF.set(None)
        _fb = STOCK_BASE_GRANULARITY if is_stock_pair(pair) else FX_BASE_GRANULARITY
        print(f"⚠️ [시간축] {pair} 알림에 tf 없음/해석불가(tf={_tf_raw!r}) → 기본값 {_fb} 사용. "
              f"차트 시간축을 바꿨다면 Pine 알림에 tf 를 넣거나 환경변수를 맞출 것.")

    _ = check_recent_opposite_signal(pair, signal)  # 소프트 OFF: 기록만, 차단 안 함

    # ============================================================
    # 🟥 [FIX-D6] 진입 금지 종목은 여기서 바로 끊는다.
    # ------------------------------------------------------------
    #  기존엔 제외 종목 체크가 웹훅 거의 끝(실행 게이트)에 있어서,
    #  CEG 89건 + ALAB 50건 + VRT 12건 = 151건(전체의 14%)이
    #  캔들 조회 → 지표 계산 → 뉴스 조회 → GPT 호출까지 전부 태운 뒤에야 버려졌다.
    #  비용과 레이트리밋을 그대로 낭비한 것.
    #  (가격을 아직 모르므로 여기선 심볼 기반 차단만. 페니주 필터는 가격을 안 뒤 다시 본다.)
    # ============================================================
    _early_blocked, _early_reason = is_blocked_instrument(pair, None)
    if _early_blocked:
        print(f"🚫 [조기차단] {pair} — {_early_reason} (지표·GPT 호출 없이 즉시 종료)")
        # 🟥 [FIX-D6b] 조기 종료하더라도 감사 흔적은 남긴다.
        #    그냥 return하면 이 알림이 왔다는 사실 자체가 시트에서 사라져서
        #    "제외 종목에 알림이 몇 건이나 낭비되는지"를 나중에 셀 수 없다.
        _log_blocked_alert(pair, data.get("signal"), data.get("alert_name"), _early_reason)
        return JSONResponse(content={
            "status": "blocked", "reason": _early_reason, "pair": pair
        })

    # 🟥 [FIX-E5] 같은 봉에 대한 알림 재전송 차단.
    #    TradingView가 재시도하거나 여러 알림이 겹치면 같은 신호로 2번 진입할 수 있었다.
    # 🟥 [FIX-E5b] {{timenow}}는 "알림 발사 시각"이라 재전송마다 값이 달라진다.
    #    그걸 키에 넣으면 중복 판정이 절대 성립하지 않아 dedup이 무의미해진다.
    #    봉 시각(bar_time/time)만 쓰고, 없으면 심볼+방향만으로 판정한다.
    _bar_time = data.get("bar_time") or data.get("bar_close_time") or data.get("time") or ""
    if _is_duplicate_alert(pair, signal, _bar_time):
        print(f"🔁 [중복알림] {pair} {signal} (bar={_bar_time}) — {ALERT_DEDUP_SECONDS}초 내 재수신 → 무시")
        return JSONResponse(content={
            "status": "ignored", "reason": "duplicate_alert", "pair": pair
        })
        
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

    # 🟥 [FIX-B1] 전략명을 여기서 한 번만 확정해서 아래 전부(스코어 함수 인자 / threshold 조회)가
    #    같은 값을 보게 한다. 기존엔 threshold 조회용 strategy_name이 스코어 계산보다
    #    한참 뒤(웹훅 중반)에 따로 계산돼서, 스코어 함수는 전략명을 아예 못 받았다.
    _alert_data_raw = data.get("alert_data", {}) or {}
    strategy_name = (
        (_alert_data_raw.get("strategy_name") if isinstance(_alert_data_raw, dict) else None)
        or (_alert_data_raw.get("alert_name") if isinstance(_alert_data_raw, dict) else None)
        or data.get("strategy_name")
        or data.get("alert_name")
        or data.get("strategy")
        or "기본알림"
    )
    strategy_name = str(strategy_name).strip() or "기본알림"

    candles = get_candles(pair, base_granularity_for(pair), 200)
    # ✅ 캔들 방어 로직 — ATR(14) 계산 가능한 최소 개수(14개)로 강화
    candle_count = len(candles) if candles is not None else 0
    print(f"📊 [{pair}] 캔들 수신: {candle_count}개")
    if candles is None or candles.empty or candle_count < 14:
        # 🟥 [FIX-TO1] 예전엔 여기서 400 만 뱉고 끝나서 시트에 한 줄도 안 남았다.
        #   형님 눈에는 "알람은 울렸는데 아무 일도 없음" 으로 보이고,
        #   로그를 뒤지지 않으면 원인을 알 방법이 없었다. 이제 시트에 남긴다.
        _why = f"NO_CANDLES: {pair} {base_granularity_for(pair)} {candle_count}개 (ATR14에 최소 14개 필요)"
        print(f"❌ [캔들부족] {_why}")
        try:
            _log_blocked_alert(pair, data.get("signal"), strategy_name, _why)
        except Exception as _e:
            print(f"⚠️ [캔들부족] 시트 기록 실패: {_e}")
        return JSONResponse(
            content={"status": "blocked", "reason": "no_candles",
                     "pair": pair, "bars": candle_count,
                     "note": "브로커에서 캔들을 못 받았습니다. 잠시 뒤 자동 재시도되지 않으니 로그를 확인하세요."},
            status_code=200
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
        # 🟥 [FIX-S1b] 여기서 그냥 400 을 던지면 시트에 아무 흔적이 없어서
        #    "알림은 울렸는데 아무 일도 안 일어남" 으로만 보인다.
        #    실제로 USDJPY 등이 여기서 조용히 죽고 있었다. 이유를 남긴다.
        _venue = "Alpaca" if is_stock_pair(pair) else "OANDA"
        _reason = (f"NO_CANDLES: {_venue} 에서 {pair} {base_granularity_for(pair)} "
                   f"캔들을 못 받았다 (심볼 형식/거래가능 여부 확인 필요)")
        print(f"❌ [캔들] {_reason}")
        _log_blocked_alert(pair, data.get("signal"), data.get("alert_name"), _reason)
        return JSONResponse(
            content={"status": "blocked", "reason": "no_candles",
                     "pair": pair, "venue": _venue,
                     "granularity": base_granularity_for(pair)},
            status_code=200
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
    # 🟦 버그 수정: 예전엔 fetch_forex_news()(포렉스팩토리 홈페이지를 단순 스크래핑, 거의 항상
    #    고정값만 반환)를 모든 자산에 공통으로 썼고, 주식은 filter_relevant_news가 항상 []을 반환해서
    #    뉴스 체크가 사실상 아무 의미가 없었음(항상 "영향 적음"만 나옴).
    #    주식은 Alpaca News API로 그 종목의 실제 최근 뉴스를 확인하고, FX는 기존 경제캘린더 기반을 유지.
    if is_stock_pair(pair):
        news_score, news_msg, news_headlines = get_stock_news_risk(pair)
        if news_headlines:
            news_msg += " — " + " / ".join(news_headlines[:2])
        news = news_msg
        _macro_src = " ".join(str(x) for x in (news_headlines or []))
    else:
        news_score, news_msg, _fx_titles = news_risk_score(pair)
        news = news_msg
        _macro_src = " ".join(str(x) for x in (_fx_titles or []))

    # 🟥 [FIX-MACRO1] 거시 스캔은 **뉴스를 받은 직후 즉시** 해야 한다.
    #   예전엔 GPT 호출(최대 3회 재시도) 뒤에 했는데, 8/28 처럼 알림이 동시에 몰리면
    #   SPY 스레드가 GPT 를 끝내기 전에 다른 7종목이 이미 주문을 넣어버린다.
    #   경보는 '가장 먼저 뉴스를 본 스레드' 가 즉시 켜야 나머지를 막을 수 있다.
    #   ⚠️ 표시용 요약이 아니라 **헤드라인 원문**을 넘긴다(요약엔 2건만 들어간다).
    try:
        _scan_macro_news(pair, f"{news} {_macro_src}")
    except Exception as _e:
        print(f"⚠️ [거시경보] 스캔 실패(무시): {_e}")
    high_low_analysis = analyze_highs_lows(candles)
    atr = float(atr_series.dropna().iloc[-1]) if not atr_series.dropna().empty else 0.0
    fibo_levels = calculate_fibonacci_levels(candles["high"].max(), candles["low"].min())
    # 📌 현재가 계산
    price = current_price
    # 🟥 [FIX-E8] 주식에서 자릿수가 뭉개지던 버그.
    #    pip_value_for(주식) = max(0.01, 가격×0.0001)이라 $500짜리 주식은 pip=0.05가 되고
    #    log10(0.05)≈-1.3 → int(abs(...))=1 → GPT payload의 지지/저항·OHLC가
    #    소수 1자리로 반올림돼 정밀도가 통째로 날아갔다.
    #    → 주식은 항상 센트 단위(2자리)를 쓴다.
    if is_stock_pair(pair):
        price_digits = 2
    else:
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
        macd_trend,
        # 🟥 [FIX-B1] 여기가 이 파일 최대의 버그였다.
        #    기존 호출은 위치인자 22개까지만 넘기고 expected_direction / strategy_name 을
        #    빠뜨렸다. 그래서 함수 안의 is_buy / is_sell 이 항상 False가 되어
        #      · BUY/SELL 전용 감점 블록
        #      · 패턴 그룹 가·감점(+2 / -1.5)
        #      · balance breakout 전용 분기
        #      · 두 번째 must_capture_opportunity 블록
        #    이 전부 죽은 코드였다. 실거래 665건에서 점수-결과 상관이 r=0.039(p=0.31)로
        #    사실상 0이었던 직접 원인.
        expected_direction=signal,
        strategy_name=strategy_name,
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
        "news": news_msg,
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

    # 🟦 3번 수정: opportunity_score 역행 + 골든크로스 동시 발생 → 추가 감점
    #    데이터 분석(6/22~7/10, 44건): 역행+골든크로스 동시 발생 시 55% 승률 -$188
    #    역행만 있고 골든크로스 없을 때는 68% 승률 +$172로 오히려 좋음
    #    → 이 두 가지가 같이 오면 "추세와 반대 방향으로 강하게 올라온 것"을 의미
    #      골든크로스가 실제 추세 전환이 아닌 단기 과열 신호일 가능성이 높음
    # 🟥 [FIX-ROLE1] '역행' 사유는 SETUP_RECHECK_ENABLED 가 꺼지면 생성되지 않는다.
    #   그대로 두면 살아있는 것처럼 보이는 죽은 코드가 된다.
    _has_opp_reverse = (SETUP_RECHECK_ENABLED
                        and any("opportunity_score 역행" in r for r in reasons))
    _has_golden = any("골든크로스(강)" in r for r in reasons)
    if _has_opp_reverse and _has_golden:
        signal_score -= 1.5
        reasons.append("🔴 opportunity_score 역행 + MACD 골든크로스 동시 발생 → 과열 추격 위험 추가감점 (-1.5)")
            
    recent_trade_time = get_last_trade_time()
    # 🟥 [FIX-E9] naive/aware 혼용으로 TypeError가 날 수 있어 양쪽을 aware UTC로 맞춘다.
    _now_utc = datetime.now(ZoneInfo("UTC"))
    if recent_trade_time is not None and recent_trade_time.tzinfo is None:
        recent_trade_time = recent_trade_time.replace(tzinfo=ZoneInfo("UTC"))
    time_since_last = (_now_utc - recent_trade_time) if recent_trade_time else timedelta(hours=999)
    allow_conditional_trade = time_since_last > timedelta(hours=2)

    strategy_thresholds = {
    "Balance breakout": 4.5,
    "BUY_ENTRY_BAR_CLOSE": -7.0,
    "SELL_ENTRY_BAR_CLOSE": -7.0,
    "기본알림": 3.0,
    "Test Alarm": 0.0,
    "BUY_STOCK_PORTFOLIO_A2": -7.5,
    # 🟥 [FIX-F2] Pine 전략을 A5로 이름 바꿨으므로 같은 threshold로 등록.
    #    (등록 안 하면 '미등록 전략명' 경로로 빠진다 — 이제는 경고+기본값이지만, 명시가 낫다)
    "BUY_STOCK_PORTFOLIO_A5": -7.5,
    "BUY STOCK PORTFOLIO A5": -7.5,
    }

    # 🟥 [FIX-B1] strategy_name은 위(웹훅 앞부분)에서 이미 확정했다. 여기서 다시 계산하지 않는다.
    #    (기존엔 여기서만 계산해서 스코어 함수는 전략명을 못 받았고, 마지막 `or ""`는 도달 불가 코드였다)
    alert_data = payload.get("alert_data", {})
    # 🟥 [FIX-I1] 접두사 매칭으로 해석 — Pine 전략 이름이 A5→A10→A11로 바뀌어도 그대로 동작한다.
    threshold, _thr_how = resolve_strategy_threshold(strategy_name, strategy_thresholds, is_stock_pair(pair))
    if _thr_how.startswith("미등록"):
        print(f"⚠️ [threshold] 등록되지 않은 전략명 '{strategy_name}' → 기본값 {threshold} 적용")
        reasons.append(f"⚠️ 미등록 전략명 '{strategy_name}' → 기본 threshold {threshold} 사용")
    else:
        print(f"[threshold] '{strategy_name}' → {threshold} ({_thr_how})")

    # 🟦 주식인데 strategy_name이 아예 안 와서 "기본알림"(FX 기준 3.0)으로 떨어진 경우 보정
    # 🟥 [FIX-TH1] 전략명이 안 온 알림. 주식만 보정하면 FX 는 '기본알림'(3.0)에 걸려
    #    문턱을 내린 의미가 없어진다. 양쪽 다 상품 기준 기본값으로 떨어뜨린다.
    if strategy_name in ("기본알림", ""):
        threshold = STOCK_SCORE_THRESHOLD if is_stock_pair(pair) else FX_SCORE_THRESHOLD

    print(f"[DEBUG] strategy_name={strategy_name}, threshold={threshold}, score={signal_score}")
    gpt_feedback = "GPT 분석 생략: 점수 미달"
    decision, tp, sl = None, None, None
    # 🟥 [FIX-D4] GPT가 실제로 무엇을 판단했는지 별도 변수로 보관한다.
    #    기존엔 `decision`이 None으로 초기화된 뒤 한 번도 갱신되지 않는데
    #    그대로 시트 14번 열(final_decision)에 기록돼서 1,048행이 전부 공백이었다.
    gpt_parsed_decision = None
    wait_confidence = None
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
        
        # ============================================================
        # 🟥 [FIX-C1] GPT 실패 = 진입 차단
        # ------------------------------------------------------------
        #  기존 동작: GPT가 타임아웃/429/에러/시간제한으로 실패하면 그 문자열이
        #  parse_gpt_feedback()에서 "WAIT"으로 파싱되고, 바로 아래 "WAIT 확신도 부족"
        #  로직이 주식·JPY를 원래 알림 방향으로 강제 환원시켰다.
        #  → 결과적으로 GPT가 죽어 있어도 무검증으로 주문이 나갔고,
        #    GPT 내부의 롤오버/금요일 시간제한도 이 경로로 전부 무력화됐다.
        #  이제는 GPT가 유효한 판단을 못 주면 그 신호를 버린다.
        # ============================================================
        _gpt_failed = False
        _gpt_fail_reason = ""
        _raw_probe = str(gpt_raw) if gpt_raw is not None else ""
        if not gpt_raw:
            _gpt_failed, _gpt_fail_reason = True, "GPT_NO_RESPONSE"
        elif "GPT_ERROR" in _raw_probe:
            _gpt_failed, _gpt_fail_reason = True, "GPT_ERROR"
        elif "GPT_TIMEOUT" in _raw_probe or "타임아웃" in _raw_probe:
            _gpt_failed, _gpt_fail_reason = True, "GPT_TIMEOUT"
        elif "⛔ 거래 제한" in _raw_probe:
            # analyze_with_gpt() 내부의 롤오버/주말 시간제한 — 원래 의도대로 '차단'으로 처리
            _gpt_failed, _gpt_fail_reason = True, "GPT_TIME_RESTRICTED"
        elif "쿨다운" in _raw_probe or "429" in _raw_probe:
            _gpt_failed, _gpt_fail_reason = True, "GPT_RATE_LIMITED"

        if _gpt_failed:
            print(f"❌ GPT 검증 실패({_gpt_fail_reason}) → 이 신호는 진입하지 않는다 (무검증 진입 금지)")
            gpt_feedback = f"{_gpt_fail_reason}: {_raw_probe[:500]}"
            final_decision = f"BLOCKED_{_gpt_fail_reason}"
            final_tp, final_sl = None, None
            gpt_parsed_decision = _gpt_fail_reason
            _skip_gpt_parse = True
        else:
            _skip_gpt_parse = False

        print("✅ STEP 6: GPT 응답 수신 완료 (이미지 분석 포함)")
        # ✅ 추가: 파싱 결과 강제 정규화 (대/소문자/공백/이상값 방지)
        raw_text = (
            gpt_raw if isinstance(gpt_raw, str)
            else json.dumps(gpt_raw, ensure_ascii=False)
            if isinstance(gpt_raw, dict) else str(gpt_raw)
        )
        print(f"📄 GPT Raw Response: {raw_text!r}")
        if not _skip_gpt_parse:
            gpt_feedback = raw_text
            parsed_decision, tp, sl, wait_confidence = parse_gpt_feedback(raw_text) if raw_text else ("WAIT", None, None, None)
            gpt_parsed_decision = parsed_decision   # 🟥 [FIX-D4] 시트 14번 열에 기록될 값

            if parsed_decision in ["BUY", "SELL"]:
                # 🟥 [FIX-C1b] GPT가 알림과 반대 방향을 말하면 따라가지 않고 버린다.
                #    기존엔 GPT 방향을 그대로 채택해서, BUY 알림에 GPT가 SELL이라고 하면
                #    반대 포지션이 나갈 수 있었다.
                if signal in ("BUY", "SELL") and parsed_decision != signal:
                    print(f"⛔ GPT 판단({parsed_decision})이 알림 방향({signal})과 반대 → 진입 취소")
                    final_decision = "BLOCKED_GPT_DIRECTION_CONFLICT"
                    final_tp, final_sl = None, None
                else:
                    final_decision = parsed_decision
                    final_tp = tp
                    final_sl = sl
                    print(
                        f"[✔️UPDATE] GPT 결정 적용: "
                        f"{final_decision}, tp={final_tp}, sl={final_sl}"
                    )
            else:
                # 🟥 [FIX-C1] WAIT 강제 환원 로직 제거.
                #    기존엔 주식/JPY에서 wait_confidence가 80/95 미만이거나 아예 없으면
                #    GPT의 WAIT을 무시하고 원래 방향으로 되돌렸다. GPT가 wait_confidence를
                #    거의 안 돌려줬기 때문에(실거래 80건 중 기록 0건) 사실상 "WAIT은 항상 무시"였다.
                #    → GPT의 WAIT을 그대로 존중한다. 검증 레이어가 검증을 하려면 이래야 한다.
                final_decision = "WAIT"
                final_tp = None
                final_sl = None
                print(f"⏸️ [WAIT] GPT 관망 판단 존중 (wait_confidence={wait_confidence}) → 진입하지 않음")

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
        gpt_parsed_decision = "NOT_CALLED_BELOW_THRESHOLD"   # 🟥 [FIX-D4]

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
    # 🟥 [FIX-C1c] GPT 실패로 차단된 경우, 위의 재추출이 실패 사유를 덮어쓴다.
    #    시트에 왜 막혔는지가 남아야 하므로 사유를 다시 앞에 붙인다.
    if str(final_decision or "").startswith("BLOCKED_"):
        gpt_feedback = f"[{final_decision}] {str(gpt_feedback)[:800]}"
    
    print(f"✅ STEP 7: GPT 해석 완료 | decision: {final_decision}, TP: {final_tp}, SL: {final_sl}")
   
    
    # 📌 outcome_analysis 및 suggestion 기본값 세팅
    outcome_analysis = "WAIT 또는 주문 미실행"
    # 🟦 WAIT일 때 GPT가 보고한 wait_confidence를 같이 남겨둔다 (나중에 "GPT가 80 이상이라고 한
    #    WAIT들이 진짜로 맞았는지" 보정/검증 분석에 쓰임).
    # 🟥 [FIX-C5] wait_confidence가 None일 때도 기록한다.
    #    기존엔 `is not None` 조건 때문에 실거래 WAIT 80건 중 단 1건도 기록되지 않았고,
    #    "GPT의 관망 판단이 실제로 맞았는지"를 사후 검증할 방법이 아예 없었다.
    if final_decision == "WAIT":
        adjustment_suggestion = f"wait_confidence={wait_confidence if wait_confidence is not None else 'NONE'}"
    elif str(final_decision or "").startswith("BLOCKED_"):
        adjustment_suggestion = str(final_decision)
    else:
        adjustment_suggestion = ""
    price_movements = None
    gpt_feedback_dup = None
    filtered_movement = None


        
    print(f"✅ STEP 10: 전략 요약 저장 호출 | decision: {decision}, TP: {tp}, SL: {sl}")
    sheet_row_idx = log_trade_result(
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
        # 🟥 [FIX-D4] `decision`(항상 None) → `gpt_parsed_decision`으로 교체.
        gpt_decision=gpt_parsed_decision,
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
    #    🟦 WAIT인 경우에도 "원래 신호 방향대로 들어갔다면 TP/SL이 얼마였을지"를 계산해서 시트에는
    #       남긴다(실제 주문은 안 나간다 — 주문 실행 여부는 final_decision/should_execute로만 결정됨).
    #       이게 없으면 WAIT 행은 시트에 TP/SL이 항상 빈칸으로 남아서, 나중에 결과추적이
    #       "WAIT 했는데 실제로 TP/SL 중 뭐가 먼저 닿았을지"를 평가할 수가 없었다.
    _calc_direction = final_decision if final_decision in ("BUY", "SELL") else (signal if signal in ("BUY", "SELL") else None)

    # ============================================================
    # 🟥 [FIX-FX3] FX 도 TP/SL 을 시트에 남긴다.
    # ------------------------------------------------------------
    #  기존엔 아래 블록이 is_stock_pair(pair) 로 막혀 있어서, FX 는 GPT 가
    #  호출되지 않은 행(WAIT / SKIPPED_BY_THRESHOLD)에 TP/SL 이 영영 빈칸이었다.
    #  결과추적(evaluate_pending_outcomes)은 TP/SL 이 없으면 판정 자체를 못 하므로
    #  그 행들은 'summary = 미정' 에서 영원히 멈춘다. 주식은 채워지고 FX 만 안 채워진
    #  이유가 바로 이것이다.
    #
    #  FX 는 Pine 알림(A12)이 이미 tp/sl 을 실어 보내므로 그 값을 먼저 쓰고,
    #  없으면(구형 BUY_ENTRY_BAR_CLOSE 등) ATR 로 계산한다.
    # ============================================================
    if (not is_stock_pair(pair)) and _calc_direction and price is not None:
        def _fnum(v):
            # ⚠️ 아래쪽 _as_num() 은 이 지점보다 '뒤에' 정의돼 있어서 여기선 못 쓴다.
            try:
                f = float(str(v).strip())
                return f if f == f and abs(f) != float("inf") else None
            except (TypeError, ValueError):
                return None
        # ⚠️ 같은 함정 — 알림의 tp/sl 은 최상위 키다.
        def _fpick(k):
            v = alert_data.get(k) if isinstance(alert_data, dict) else None
            return v if v not in (None, "") else data.get(k)
        _fx_tp = _fnum(_fpick("tp"))
        _fx_sl = _fnum(_fpick("sl"))
        _fx_src = "알림값"
        if _fx_tp is None or _fx_sl is None:
            try:
                _fx_atr = float(atr.iloc[-1]) if hasattr(atr, "iloc") else float(atr or 0)
            except Exception:
                _fx_atr = 0.0
            if _fx_atr > 0:
                _d = price_round_digits(pair)
                _sd = _fx_atr * FX_SL_ATR_MULT
                _td = _sd * FX_RR_RATIO
                if _calc_direction == "BUY":
                    _fx_tp, _fx_sl = round(price + _td, _d), round(price - _sd, _d)
                else:
                    _fx_tp, _fx_sl = round(price - _td, _d), round(price + _sd, _d)
                _fx_src = f"ATR×{FX_SL_ATR_MULT}/RR{FX_RR_RATIO}"
        # ⚠️ 절대 실주문 값을 건드리지 않는다.
        #    final_tp/final_sl 은 place_order() 가 OANDA 로 보내는 값이고,
        #    final_sl 은 calc_fx_units() 가 수량을 정할 때 나누는 값이다.
        #    여기서 덮어쓰면 adjust_tp_sl_for_structure() 의 최소 RR·최소 pip·S/R 보정이
        #    통째로 무효가 되고, 알림이 tp=0 을 보내면 주문이 거부된다.
        #    이 블록의 목적은 '실행 안 된 행도 나중에 채점할 수 있게 시트에 남기는 것' 뿐이다.
        _fx_ok = (_fx_tp is not None and _fx_sl is not None
                  and _fx_tp > 0 and _fx_sl > 0
                  and (_fx_tp > price > _fx_sl if _calc_direction == "BUY"
                       else _fx_sl > price > _fx_tp))
        if _fx_ok and final_decision not in ("BUY", "SELL"):
            gpt_feedback += (f"\n🟦 [FX 평가용] {_calc_direction} 로 들어갔다면 "
                             f"TP={_fx_tp}, SL={_fx_sl} ({_fx_src}) — 실제 주문은 안 나감")
            try:
                correct_sheet_trade_prices(sheet_row_idx, price, _fx_tp, _fx_sl)
            except Exception as e:
                print(f"⚠️ [FX TP/SL 기록] 실패(무시): {e}")
        elif not _fx_ok and final_decision not in ("BUY", "SELL"):
            print(f"⚠️ [FX 평가용] {pair} TP/SL 값이 이상해서 기록 생략 "
                  f"(tp={_fx_tp}, sl={_fx_sl}, price={price}, dir={_calc_direction})")

    # 🟥 [FIX-SLTP1] 구조적 손절을 쓰는 전략은 알림이 보낸 값을 그대로 쓴다.
    _use_alert_sltp = False
    # 🟥 [FIX-FXSL1] is_stock_pair() 게이트 제거 — FX 도 통과시킨다.
    #    대신 아래에서 상품 유형별 임계값(_sl_min/_sl_max/_drift_max)을 갈라 쓴다.
    _sltp_is_stock = is_stock_pair(pair)
    _sl_min   = ALERT_SL_MIN_PCT if _sltp_is_stock else ALERT_SL_FX_MIN_PCT
    _sl_max   = ALERT_SL_MAX_PCT if _sltp_is_stock else ALERT_SL_FX_MAX_PCT
    _drift_max = ALERT_SLTP_MAX_DRIFT_PCT if _sltp_is_stock else ALERT_SLTP_FX_MAX_DRIFT_PCT
    if (_calc_direction and price is not None
            and _normalize_strategy_name(strategy_name) in ALERT_SLTP_STRATEGIES):
        def _n(v):
            try:
                f = float(str(v).strip())
                return f if f == f and abs(f) != float("inf") and f > 0 else None
            except (TypeError, ValueError):
                return None
        # ⚠️ Pine 알림은 tp/sl 을 **최상위 키**로 보낸다. alert_data 안에 있지 않다.
        #    alert_data 만 보면 영원히 None 이라 이 기능 전체가 죽은 코드가 된다.
        def _pick(k):
            v = alert_data.get(k) if isinstance(alert_data, dict) else None
            return v if v not in (None, "") else data.get(k)
        _a_tp = _n(_pick("tp"))
        _a_sl = _n(_pick("sl"))
        # ⚠️ 검증은 '알림이 계산할 때 쓴 가격' 기준으로 해야 한다.
        #    price 는 봇이 나중에 캔들로 다시 받은 값이라 몇 초~몇십 초 뒤 가격이다.
        #    그걸로 재면 손절이 코앞인 주문도 통과해버린다(즉시 손절).
        _alert_px = _n(data.get("price")) or price
        _drift = abs(price - _alert_px) / _alert_px * 100.0 if _alert_px else 999.0
        if _a_tp is not None and _a_sl is not None:
            _ok_dir = (_a_tp > _alert_px > _a_sl) if _calc_direction == "BUY" else (_a_sl > _alert_px > _a_tp)
            _risk_pct = abs(_alert_px - _a_sl) / _alert_px * 100.0 if _alert_px else 0.0
            if _drift > _drift_max:
                print(f"⚠️ [구조손절] {pair} 알림가 {_alert_px} → 현재가 {price} "
                      f"({_drift:.3f}% 이동, 한도 {_drift_max}%) → 구조값 못 믿음, ATR 로 계산")
            elif _ok_dir and _sl_min <= _risk_pct <= _sl_max:
                _use_alert_sltp = True
                _STRUCTURAL_SLTP.set(True)
                if final_decision in ("BUY", "SELL"):
                    final_tp, final_sl = _a_tp, _a_sl
                    tp, sl = final_tp, final_sl
                gpt_feedback += (f"\n🟦 [{strategy_name}] 구조적 TP/SL 사용 — "
                                 f"TP={_a_tp}, SL={_a_sl} (위험폭 {_risk_pct:.2f}%). ATR 재계산 안 함")
                print(f"🎯 [구조손절] {pair} {strategy_name} → TP={_a_tp} SL={_a_sl} "
                      f"(위험 {_risk_pct:.2f}%)")
                try:
                    correct_sheet_trade_prices(sheet_row_idx, price, _a_tp, _a_sl)
                except Exception as _e:
                    print(f"⚠️ [구조손절] 시트 기록 실패: {_e}")
            else:
                print(f"⚠️ [구조손절] {pair} 알림 TP/SL 거부 "
                      f"(tp={_a_tp}, sl={_a_sl}, 알림가={_alert_px}, 위험 {_risk_pct:.2f}%, "
                      f"허용 {_sl_min}~{_sl_max}%) → ATR 로 계산")
        else:
            print(f"⚠️ [구조손절] {pair} 알림에 tp/sl 이 없음 (tp={_a_tp}, sl={_a_sl}) → ATR 로 계산")

    if (not _use_alert_sltp) and is_stock_pair(pair) and _calc_direction and price is not None and atr is not None:
        _stock_atr = float(atr.iloc[-1]) if hasattr(atr, "iloc") else float(atr)
        if _stock_atr and _stock_atr > 0:
            _digits = price_round_digits(pair)
            # 🟥 [FIX-A2] SL 거리를 먼저 정하고, TP는 반드시 SL거리 × 손익비로 파생시킨다.
            #    (기존처럼 TP/SL 배수를 따로 두면 손익비가 조용히 1.0으로 무너진다 — 실제로 그랬다.)
            _sl_dist = _stock_atr * STOCK_SL_ATR_MULT
            _tp_dist = _sl_dist * STOCK_RR_RATIO
            if _calc_direction == "BUY":
                _hyp_tp = round(price + _tp_dist, _digits)
                _hyp_sl = round(price - _sl_dist, _digits)
            else:  # SELL
                _hyp_tp = round(price - _tp_dist, _digits)
                _hyp_sl = round(price + _sl_dist, _digits)

            if final_decision in ("BUY", "SELL"):
                # 실제 체결 방향 — 기존과 동일하게 final_tp/final_sl/tp/sl 전부 갱신
                final_tp, final_sl = _hyp_tp, _hyp_sl
                tp, sl = final_tp, final_sl  # 아래 검증 블록이 참조하는 tp/sl도 동기화
                gpt_feedback += (
                    f"\n🟦 주식 TP/SL 재계산: SL=ATR×{STOCK_SL_ATR_MULT}, "
                    f"TP=SL거리×RR{STOCK_RR_RATIO} (=ATR×{STOCK_TP_ATR_MULT:.2f}) "
                    f"(ATR={_stock_atr:.4f}) → TP={final_tp}, SL={final_sl}"
                )
                # 🟦 log_trade_result()가 이 재계산보다 먼저 호출돼서, GPT가 보고한 값이 공식과
                #    미묘하게 다른 드문 경우엔 시트에 그 (틀린) 값이 남을 수 있다. 사후 보정으로 확정.
                correct_sheet_trade_prices(sheet_row_idx, current_price, final_tp, final_sl)
            else:
                # WAIT — 실제 final_tp/final_sl(None)은 그대로 두고(주문 로직에 영향 없게),
                # 시트에는 사후보정으로 가상의 TP/SL을 채워넣는다.
                gpt_feedback += (
                    f"\n🟦 [평가용] WAIT이지만 원래 방향({_calc_direction})대로 들어갔다면: "
                    f"TP={_hyp_tp}, SL={_hyp_sl} (ATR={_stock_atr:.4f}) — 실제 주문은 안 나감"
                )
                # 🟦 log_trade_result()는 이미 위(line~2285)에서 이 값들 계산 전에 호출돼서
                #    시트에 price/tp/sl이 빈칸으로 박혀있다. 같은 행을 사후 보정해서 채워넣는다.
                #    (price는 로그 당시와 동일한 값을 그대로 다시 써서 다른 컬럼은 안 건드림)
                correct_sheet_trade_prices(sheet_row_idx, current_price, _hyp_tp, _hyp_sl)

    # ✅ 여기서부터 검증 블록 삽입 (FX는 기존과 동일하게 tp/sl 기준으로 계산)
    pip = pip_value_for(pair)
    # 🟥 [FIX-E4] tp/sl이 ""(safe_float 실패값)이나 None이면 아래 산술에서 TypeError로
    #    500이 났다. 계산 전에 숫자로 정규화한다.
    def _as_num(v):
        try:
            if v is None or v == "":
                return None
            return float(v)
        except (TypeError, ValueError):
            return None

    tp = _as_num(tp)
    sl = _as_num(sl)
    final_tp = _as_num(final_tp)
    final_sl = _as_num(final_sl)
    # min_pip / tp_sl_ratio는 계산만 하고 어디서도 쓰이지 않던 죽은 변수라 제거했다.


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
    _block_label = ""      # 🟥 [FIX-D2c] 진입이 막힌 구체적 사유(마지막에 시트로)
    _executed_units = None  # 🟥 [FIX-D9] 실제 주문 수량(주식=주, FX=units)
    
    
    # 1️⃣ 기본 진입 조건
    # - GPT가 BUY/SELL
    # - 전략별 threshold (Balance=4.0 / Engulfing=2.5) 통과
    should_execute = (
        final_decision in ["BUY", "SELL"]
        and signal_score >= threshold
    )

    # 🟥 [FIX-MACRO1] 다른 종목에서 켜진 경보라도 여기서 막는다
    _mb, _mb_why = _macro_block_active()
    if should_execute and _mb:
        should_execute = False
        _block_label = f"MACRO_NEWS: {_mb_why}"
        print(f"🌐 [거시차단] {pair} — {_mb_why}")
        reasons.append(f"🌐 거시 뉴스 경보로 진입 차단 — {_mb_why}")

    # 🟥 [FIX-HB1] threshold 와 무관한 하드 차단. 점수가 아무리 높아도 막는다.
    _hb, _hb_why = _hard_blocked(reasons)
    if should_execute and _hb:
        should_execute = False
        _block_label = f"HARD_BLOCK: {_hb_why}"
        print(f"⛔ [하드차단] {pair} — {_hb_why} (점수 {signal_score:.2f} 와 무관하게 차단)")
        reasons.append(f"⛔ 위 조건으로 진입을 실제로 차단했습니다 (점수 {signal_score:.2f})")
    
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
    
    # 1-1️⃣ 진입 금지 종목 차단
    #    🟥 [FIX-A4] 기존의 EXCLUDED_SYMBOLS 단독 체크를 is_blocked_instrument()로 통합.
    #    제외종목 + 레버리지/인버스 ETF(TZA·SOXS 등) + 페니주를 한 번에 막는다.
    _blocked, _instr_block_reason = is_blocked_instrument(pair, price)
    if _blocked:
        reasons.append(f"🚫 진입 금지 종목({pair}) → 차단 [{_instr_block_reason}]")
        should_execute = False
        # 🟥 [FIX-D2c] 사유를 로컬에 보관했다가 마지막 _finalize_sheet_row()에 넘긴다.
        #    여기서 시트에 바로 쓰면, 웹훅 끝의 finalize가 같은 칸(AH)을 일반 메시지로 덮어썼다.
        _block_label = _instr_block_reason

    # 2️⃣ 주식 전용: 요일별 거래 시간 제한
    #    ┌─────────────────────────────────────────────────────┐
    #    │  월~목: 12:00~12:59 차단 (점심)                       │
    #    │         13:00~15:29 허용                             │
    #    │         15:30 이후 차단 (장마감 임박)                  │
    #    │  금:    12:00 이후 전체 차단                           │
    #    └─────────────────────────────────────────────────────┘
    if should_execute and is_stock_pair(pair):
        _ny_now = datetime.now(ZoneInfo("America/New_York"))
        _ny_hour = _ny_now.hour
        _ny_minute = _ny_now.minute
        _ny_dow = _ny_now.weekday()   # 0=월 1=화 2=수 3=목 4=금
        _ny_hhmm = _ny_hour * 100 + _ny_minute  # 예: 15:30 → 1530

        _block_reason = None

        if _ny_dow == 4:
            # 금요일: 12:00 이후 전체 차단
            if _ny_hhmm >= 1200:
                _block_reason = f"❌ 금요일 12시 이후 거래 제한({_ny_hour}:{_ny_minute:02d} ET) → 신규 진입 차단"
        else:
            # 월~목
            if 1200 <= _ny_hhmm < 1300:
                # 12:00~12:59 점심 차단
                _block_reason = f"❌ 점심 구간 차단({_ny_hour}:{_ny_minute:02d} ET, 12:00~12:59) → 신규 진입 차단"
            elif _ny_hhmm >= 1530:
                # 15:30 이후 차단
                _block_reason = f"❌ 장마감 임박({_ny_hour}:{_ny_minute:02d} ET, 15:30 이후) → 신규 진입 차단"

        if _block_reason:
            reasons.append(_block_reason)
            should_execute = False
            # 🟥 [FIX-D2] 기존엔 미정의 함수 _get_sheet()를 호출해 NameError가 except에 삼켜지고
            #    차단 사유가 시트에 전혀 안 남았다. _mark_sheet_result()로 교체.
            _label = "TIME_BLOCKED_FRIDAY" if _ny_dow == 4 else (
                "TIME_BLOCKED_LUNCH" if 1200 <= _ny_hhmm < 1300 else "TIME_BLOCKED_CUTOFF"
            )
            _block_label = f"{_label}_{_ny_hour}h{_ny_minute:02d}m"   # 🟥 [FIX-D2c]


    # if should_execute and last_atr < 0.0009:
    #     reasons.append("❌ ATR 너무 낮음 → 진입 차단")
    #     should_execute = False
    
    # 4️⃣ 디버그 로그 (강력 추천)
    print(
        f"[EXEC CHECK] decision={final_decision}, "
        f"score={signal_score:.2f}, threshold={threshold}, "
        f"execute={should_execute}"
    )
    # 🟥 [FIX-E5] pair_for_order를 락 획득 전에 확정한다.
    pair_for_order = pair.replace("/", "_")
    # ============================================================
    # 🟥 [FIX-E5] "보유 확인 → 한도 확인 → 주문 전송"을 심볼별 락으로 원자화.
    #  기존엔 이 구간에 락이 없어서, 같은 종목 알림이 거의 동시에 2건 들어오면
    #  둘 다 "보유 없음"을 보고 둘 다 주문할 수 있었다(웹훅이 스레드풀에서 병렬 처리됨).
    #  락은 심볼 단위라 서로 다른 종목의 처리는 그대로 병렬로 돈다.
    # ============================================================
    with _get_order_lock(pair_for_order):
        if should_execute:

            if is_stock_pair(pair_for_order):
                # 🟦 같은 종목 반복신호 쿨다운 — 1시간 내 2번째 신호까지는 허용, 3번째부터 그 종목만 1시간 휴식.
                allowed, reason = check_symbol_repeat_cooldown(pair_for_order)
                if not allowed:
                    print(f"[SKIP] {reason}")
                    should_execute = False
                    _block_label = "SYMBOL_REPEAT_COOLDOWN"   # 🟥 [FIX-D2c]
                elif reason:
                    print(f"[INFO] {reason}")

            if should_execute and is_stock_pair(pair_for_order):
                # 🟦 주식: FX의 FIFO 완전차단 대신, "가격대별 정상 1회 거래수량의 2배"를
                #    누적 보유 한도로 둔다. 이미 그 한도까지 채워져 있으면 추가 진입 스킵.
                #    (FIFO 완전차단은 NFA 규정상 FX에만 강제되는 룰이라 주식에 그대로 가져올 필요는 없음.
                #     다만 한 종목에 무제한 집중되는 것은 막기 위해 한도를 둠.)
                existing_qty = get_alpaca_position_qty(pair_for_order)
                # 🟥 [FIX-E6b] 한도 계산도 실제 주문 수량 산출기(calc_alpaca_qty)와 같은 값을 써야 한다.
                #    기존엔 캡이 적용되지 않은 get_tiered_qty()를 쓰다 보니, 캡으로 수량이 줄어든
                #    고가주에서 한도가 실제 주문 4회분이 되어 의도(2회분)보다 느슨해졌다.
                # 🟥 [FIX-P1b/D4] 보유한도(정상수량×2)는 '축소되지 않은' 수량으로 계산해야 한다.
                #   상관 축소된 수량으로 한도를 잡으면, 그 종목을 많이 들수록 축소가 커지고
                #   한도도 같이 작아져서 멀쩡한 진입이 POSITION_LIMIT 으로 잘못 막힌다.
                #   (symbol 을 안 넘기면 축소가 걸리지 않는다)
                base_qty = calc_alpaca_qty(price, final_sl, ALPACA_FIXED_NOTIONAL_USD)
                intended_qty = calc_alpaca_qty(price, final_sl, ALPACA_FIXED_NOTIONAL_USD,
                                               side=final_decision, symbol=pair_for_order)
                # 🟥 [FIX-P1] 상관 한도로 0이 나오면 여기서 바로 끝낸다.
                #    (주문 직전까지 갔다가 취소하면 시트에 '진입'으로 남아 통계가 오염된다)
                if intended_qty <= 0:
                    print(f"[SKIP] {pair_for_order} 포트폴리오 상관 한도 소진 → 신규진입 스킵")
                    should_execute = False
                    _block_label = "PORTFOLIO_CORRELATION_LIMIT"
                max_total_qty = base_qty * 2
                if not should_execute:
                    pass   # 위에서 상관 한도로 이미 스킵됨
                elif existing_qty + intended_qty > max_total_qty:
                    print(f"[SKIP] {pair_for_order} 기존 보유 {existing_qty}주 + 신규 {intended_qty}주 "
                          f"= 한도({max_total_qty}주, 정상수량×2) 초과 → 신규진입 스킵")
                    should_execute = False
                    _block_label = "POSITION_LIMIT"   # 🟥 [FIX-D2c]
                else:
                    print(f"[OK] {pair_for_order} 기존 보유 {existing_qty}주 + 신규 {intended_qty}주 "
                          f"≤ 한도({max_total_qty}주) → 진입 허용")
            elif should_execute and not is_stock_pair(pair_for_order):
                # ✅ FX: 이미 열린 트레이드가 있으면 신규 진입 스킵 (FIFO 방지, NFA 규정 준수)
                opened, cnt = has_open_trade(pair_for_order)
                if opened:
                    print(f"[SKIP] {pair_for_order} openTrades={cnt} → FIFO 방지로 신규진입 스킵")
                    should_execute = False
                    _block_label = "FX_FIFO_OPEN_TRADE"   # 🟥 [FIX-D2c]
    
        if should_execute:
            # 🟥 [FIX-P1] 사이징 단계에서 진입이 취소될 수 있으므로 별도 플래그를 둔다.
            #    (기존 구조는 여기서 should_execute를 내려도 아래 place_order가 그대로 실행됐다)
            _abort_order = False
            if is_stock_pair(pair_for_order):
                # 🟦 주식: 실제 수량은 place_order_alpaca 내부에서 고정금액(ALPACA_FIXED_NOTIONAL_USD)
                #         ÷ 현재가로 산출되므로, 여기서는 매수/매도 방향만 표시
                units = 1 if final_decision == "BUY" else -1
                digits = price_round_digits(pair_for_order)
            else:
                # 🟥 [FIX-E7] FX 고정 100,000 units(1랩) 제거.
                #    계좌 규모·변동성과 무관한 고정 랏이라 리스크 관리가 사실상 없었다.
                #    (OANDA 데모 계좌 150건에서 거래손익 -$1,566 + 스왑 -$1,094가 나온 배경)
                #    → SL 거리 기준 리스크 사이징으로 바꾸고, FX_UNITS_FIXED로 옛 동작 복원 가능.
                units = calc_fx_units(pair, price, final_sl, final_decision)
                digits = 3 if pair.endswith("JPY") else 5
                # 🟥 [FIX-P1] 상관 한도 소진 → 주문 자체를 내지 않는다
                if units == 0:
                    print(f"[SKIP] {pair} 포트폴리오 상관 한도 소진 → 신규진입 스킵")
                    should_execute = False
                    _block_label = "PORTFOLIO_CORRELATION_LIMIT"
                    _abort_order = True

            if _abort_order:
                result = {"status": "skipped", "reason": "portfolio_correlation_limit"}
                print(f"[DEBUG] SKIP ORDER → 포트폴리오 상관 한도 (pair={pair})")
            else:
                print(f"[DEBUG] WILL PLACE ORDER → pair={pair}, side={final_decision}, units={units}, "
                      f"price={price}, tp={final_tp}, sl={final_sl}, digits={digits}, score={signal_score}")

                result = place_order(pair_for_order, units, final_tp, final_sl, digits, price=price, atr=atr)
            # 🟥 [FIX-E3b] 전역 쿨다운 타이머를 실제로 갱신한다.
            #    이 값이 한 번도 갱신되지 않아 GLOBAL_COOLDOWN_SECONDS 설정이 무의미했다.
            if isinstance(result, dict) and result.get("status") == "order_placed":
                _last_execution_time = _t.time()
                # 🟥 [FIX-D9] 실제 체결 수량 보관 (주식은 Alpaca가 산출한 qty, FX는 units)
                _executed_units = result.get("qty") or abs(units)

            # 🟦 주식이고 실제로 가격 재조정이 일어난 경우, 시트에 이미 적힌 옛날 price/tp/sl을
            #    실제 주문에 쓰인 최종값으로 다시 보정한다 (결과추적이 보는 기준값을 일치시키기 위함).
            if is_stock_pair(pair_for_order) and isinstance(result, dict) and "final_tp" in result:
                correct_sheet_trade_prices(
                    sheet_row_idx,
                    result.get("final_price", price),
                    result.get("final_tp"),
                    result.get("final_sl"),
                )
        else:
            print(f"[DEBUG] SKIP ORDER → should_execute={should_execute}, decision={final_decision}, score={signal_score}")
            result = {"status": "skipped"}
    
    executed_time = datetime.now(ZoneInfo("UTC"))   # 🟥 [FIX-E9]
    candles_post = get_candles(pair, base_granularity_for(pair), 8)
    price_movements = candles_post[["high", "low"]].to_dict("records")

    if final_decision in ("BUY", "SELL") and isinstance(result, dict) and result.get("status") == "order_placed":

        print("[DEBUG] ORDER RESULT:", result)
        if pnl is not None and None not in (tp, sl, price):   # 🟥 [FIX-E4] None 방어
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
    # 🟥 [FIX-E4] tp/sl/price가 None일 수 있으므로 산술 전에 방어한다.
    if outcome_analysis.startswith("실패") and None not in (tp, sl, price):
        if abs(sl - price) < abs(tp - price):
            adjustment_suggestion = "SL 터치 → SL 너무 타이트했을 수 있음, 다음 전략에서 완화 필요"
        elif abs(tp - price) < abs(sl - price):
            adjustment_suggestion = "TP 거의 닿았으나 실패 → TP 약간 보수적일 필요 있음"

    # ============================================================
    # 🟥 [FIX-D1] 실행 결과를 시트에 되써넣는다.
    # ------------------------------------------------------------
    #  구조상 log_trade_result()는 sheet_row_idx가 필요해서 실행 게이트보다
    #  먼저 호출될 수밖에 없다. 그래서 기존엔 이후 게이트(가격괴리·반복쿨다운·
    #  수량한도·시간대)에서 스킵돼도 시트에는 BUY/SELL로 남았고,
    #  실거래 1,048건 중 295건이 실제로는 체결되지 않았는데 시트만 보면
    #  구분할 수 없었다(메인 시트 승률 48.8% vs 실체결 44.0%).
    #  → 여기서 "실제로 무슨 일이 있었는지"를 확정해 덮어쓴다.
    # ============================================================
    _order_status = result.get("status") if isinstance(result, dict) else str(result)
    if str(final_decision or "").startswith("BLOCKED_"):
        _effective = final_decision
    elif final_decision == "SKIPPED_BY_THRESHOLD":
        _effective = "SKIPPED_BY_THRESHOLD"
    elif final_decision == "WAIT":
        _effective = "WAIT"
    elif _order_status == "order_placed":
        _effective = f"EXECUTED_{final_decision}"
    elif _order_status == "skipped":
        _reason = (result.get("reason") if isinstance(result, dict) else "") or _block_label
        _effective = f"SKIPPED_{_reason}" if _reason else "SKIPPED_BY_GATE"
    else:
        _effective = f"ORDER_FAILED_{_order_status}"
    # 🟥 [FIX-D2c] 게이트에서 막혔으면 그 사유가 decision 칸에도 드러나게 한다.
    if _block_label and not str(_effective).startswith("EXECUTED_"):
        _effective = f"SKIPPED_{_block_label}"

    _finalize_sheet_row(
        sheet_row_idx,
        effective_decision=_effective,
        gpt_decision=gpt_parsed_decision,
        # 🟥 [FIX-D2c] 구체적 차단 사유가 있으면 그것을 우선 기록한다.
        note=_block_label or adjustment_suggestion or outcome_analysis,
        quantity=_executed_units,
    )
    print(f"🧾 [최종] {pair} decision={final_decision} / 실제결과={_effective}")

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
    start_dt = datetime.now(ZoneInfo("UTC")) - timedelta(days=int(needed_trading_days * 1.6))   # 🟥 [FIX-E9]

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

    url = f"{OANDA_BASE_URL}/v3/instruments/{pair}/candles"   # 🟥 [FIX-E2] 하드코딩 제거
    headers = {"Authorization": f"Bearer {OANDA_API_KEY}"}
    params = {"granularity": granularity, "count": count, "price": "M"}

    # 🟥 [FIX-TO1] timeout 이 없으면 requests 는 **영원히** 기다린다.
    #   실제로 겪은 일: AUD_USD 알림이 STEP 3 까지 찍히고 그대로 멈췄다.
    #   시트 기록도, 거래도, 심지어 '처리 완료' 로그(= finally 블록)도 안 나왔다.
    #   OANDA 가 응답을 안 주면 이 스레드가 통째로 멈춰 서기 때문이다.
    #   더 나쁜 건 그 스레드가 영영 안 죽는다는 것 — 알림이 올 때마다 하나씩
    #   쌓여서 결국 스레드 풀이 마르고 **모든 웹훅이 멈춘다.**
    try:
        r = requests.get(url, headers=headers, params=params,
                         timeout=(CANDLE_CONNECT_TIMEOUT, CANDLE_READ_TIMEOUT))
        r.raise_for_status()
        candles = r.json().get("candles", [])
    except requests.exceptions.Timeout:
        print(f"❗ 캔들 요청 시간초과 ({CANDLE_READ_TIMEOUT}초): {pair} {granularity} "
              f"— OANDA 응답 없음. 이번 알림은 건너뜁니다.")
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
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
    """
    🟥 [FIX-B6] 캔들 패턴 인식 확장.

    기존 구현은 HAMMER / SHOOTING_STAR / NEUTRAL 세 가지만 반환했다.
    그런데 score_signal_with_filters()는
      BULLISH_ENGULFING, BEARISH_ENGULFING, PIERCING_LINE, DARK_CLOUD_COVER,
      LONG_BODY_BULL, LONG_BODY_BEAR
    같은 이름들을 참조하며 가·감점을 준다. 그 이름들이 절대 생성되지 않았으므로
    관련 로직 전체가 죽어 있었다(실거래 데이터에서도 pattern 값이 사실상
    NEUTRAL/HAMMER/SHOOTING_STAR뿐이었다).
    → 스코어 로직이 참조하는 패턴들을 실제로 판정하도록 구현한다.

    판정 우선순위: 2봉 패턴(신뢰도 높음) → 장대바디 → 1봉 꼬리 패턴 → NEUTRAL
    """
    if candles is None or candles.empty:
        return "NEUTRAL"

    last = candles.iloc[-1]
    for c in ("open", "high", "low", "close"):
        if c not in candles.columns or pd.isna(last[c]):
            return "NEUTRAL"

    o, h, l, c_ = float(last["open"]), float(last["high"]), float(last["low"]), float(last["close"])
    body = abs(c_ - o)
    rng = h - l
    if rng <= 0:
        return "NEUTRAL"

    upper_wick = h - max(c_, o)
    lower_wick = min(c_, o) - l
    bull = c_ > o
    bear = c_ < o

    # 최근 바디 평균(직전 10봉) — "장대"의 기준을 종목/변동성에 맞춰 상대화한다.
    try:
        prev = candles.iloc[-11:-1]
        avg_body = float((prev["close"] - prev["open"]).abs().mean())
    except Exception:
        avg_body = 0.0
    if not avg_body or pd.isna(avg_body) or avg_body <= 0:
        # 직전 봉들이 전부 도지라 평균 바디가 0인 경우 — 현재 바디를 기준으로 쓰면
        # (body >= body*1.8)이 절대 성립하지 않아 장대봉 판정이 영원히 막힌다.
        avg_body = rng * 0.3

    # ---------- 2봉 패턴 ----------
    if len(candles) >= 2:
        p = candles.iloc[-2]
        if not any(pd.isna(p[c]) for c in ("open", "high", "low", "close")):
            po, pc = float(p["open"]), float(p["close"])
            p_body = abs(pc - po)
            p_bull, p_bear = pc > po, pc < po
            p_mid = (po + pc) / 2.0

            # 상승 장악형: 직전 음봉을 현재 양봉이 완전히 감쌈
            if bull and p_bear and c_ >= po and o <= pc and body > p_body:
                return "BULLISH_ENGULFING"
            # 하락 장악형
            if bear and p_bull and c_ <= po and o >= pc and body > p_body:
                return "BEARISH_ENGULFING"
            # 관통형: 음봉 뒤 양봉이 직전 몸통 중간 이상까지 회복(단 완전 장악은 아님)
            if bull and p_bear and o < pc and c_ > p_mid and c_ < po:
                return "PIERCING_LINE"
            # 흑운형: 양봉 뒤 음봉이 직전 몸통 중간 아래까지 하락
            if bear and p_bull and o > pc and c_ < p_mid and c_ > po:
                return "DARK_CLOUD_COVER"

    # ---------- 장대 바디 ----------
    #  바디가 최근 평균의 1.8배 이상이고, 전체 범위의 60% 이상을 차지 = 방향성 강한 봉
    if body >= avg_body * 1.8 and body >= rng * 0.6:
        return "LONG_BODY_BULL" if bull else "LONG_BODY_BEAR"

    # ---------- 1봉 꼬리 패턴 ----------
    if body > 0:
        # 🟥 [FIX-B6c] 아래꼬리/위꼬리 패턴은 기존 동작(HAMMER / SHOOTING_STAR)을 그대로 유지한다.
        #    한때 "상승 흐름 뒤 아래꼬리 = HANGING_MAN(약세)"으로 세분화했으나,
        #    HANGING_MAN은 bearish_patterns에 들어 있어서 BUY 신호에 -1.5가 붙는다.
        #    즉 상승추세 중 망치형(원래 +2)이 갑자기 -1.5가 되는 3.5점짜리 역전이 생긴다.
        #    이 전략은 실거래에서 "과열/추세지속 구간이 더 잘 맞는" 것으로 확인됐으므로,
        #    검증되지 않은 반전 신호를 새로 도입하지 않는다.
        if lower_wick > 2 * body and upper_wick < body:
            return "HAMMER"
        if upper_wick > 2 * body and lower_wick < body:
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

def get_stock_news_risk(symbol, within_minutes=90):
    """
    Alpaca News API(GET /v1beta1/news)로 해당 종목의 최근 뉴스를 실제로 확인한다.
    (이전엔 주식은 뉴스 체크 자체를 안 하고 항상 '영향 없음'으로 고정돼있었음)
    return: (score, message, headlines)
    """
    try:
        end = datetime.now(ZoneInfo("UTC"))
        start = end - timedelta(minutes=within_minutes)
        url = "https://data.alpaca.markets/v1beta1/news"
        params = {
            "symbols": symbol,
            "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "limit": 10,
        }
        r = requests.get(url, headers=ALPACA_HEADERS, params=params, timeout=10)
        r.raise_for_status()
        articles = r.json().get("news", [])
    except Exception as e:
        print(f"❗ [뉴스] {symbol} Alpaca News API 조회 실패: {e}")
        return 0, "❓ 뉴스 확인 실패", []

    if not articles:
        return 0, f"🟢 최근 {within_minutes}분 내 뉴스 없음", []

    headlines = [a.get("headline", "") for a in articles[:3]]
    try:
        recent_time = datetime.fromisoformat(articles[0]["created_at"].replace("Z", "+00:00"))
        minutes_ago = (end - recent_time).total_seconds() / 60
    except Exception:
        minutes_ago = within_minutes

    # 🟦 뉴스 자체가 나쁜 건 아니다(진짜 호재 뉴스로 돌파가 나올 수도 있음) — 점수는 약하게만 반영하고,
    #    "이 돌파가 뉴스發일 수 있다"는 맥락을 GPT/로그에 보여주는 게 핵심 목적.
    if minutes_ago <= 15:
        return -1, f"⚠️ {symbol} 뉴스 직후({minutes_ago:.0f}분 전, {len(articles)}건) — 뉴스 주도 변동 가능성", headlines
    else:
        return 0, f"🟡 {symbol} 최근 {within_minutes}분 내 뉴스 {len(articles)}건", headlines


def filter_relevant_news(pair, within_minutes=90):
    # 🟦 주식은 "통화코드" 개념이 없어서(ForexFactory류 경제지표 뉴스는 FX 전용) 매칭 대상이 없음.
    #    pair.split("_")[1] 같은 FX 전용 파싱이 'NVDA'처럼 '_' 없는 티커에서 IndexError를 내던 부분 수정.
    if is_stock_pair(pair):
        return []

    currency = pair.split("_")[0] if pair.startswith("USD") else pair.split("_")[1]
    now_utc = datetime.now(ZoneInfo("UTC"))   # 🟥 [FIX-E9] utcnow()+replace 대신 직접 aware 생성
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
    """🟥 [FIX-MACRO1] 세 번째로 '실제 이벤트 제목 리스트' 를 같이 돌려준다.
    예전엔 고정 문구 4개만 반환해서, 거시 키워드 스캔이 FX 에서는 절대 걸리지 않았다."""
    relevant = filter_relevant_news(pair)
    if any("High" in title for title in relevant):
        return -2, "⚠️ 고위험 뉴스 임박", relevant
    elif any("Medium" in title for title in relevant):
        return -1, "⚠️ 중간위험 뉴스 임박", relevant
    elif relevant:
        return 0, "🟢 뉴스 있음 (낮은 영향)", relevant
    else:
        return 0, "🟢 영향 있는 뉴스 없음", relevant

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

def check_symbol_repeat_cooldown(pair: str) -> tuple[bool, str]:
    """
    같은 종목이 SYMBOL_REPEAT_WINDOW_MINUTES(기본 60분) 내에 2번째 신호를 내면,
    그 신호까지는 허용하고 그 다음(3번째)부터는 SYMBOL_REPEAT_COOLDOWN_MINUTES(기본 60분)
    동안 그 종목만 신규진입을 차단한다. (포트폴리오 전체가 아니라 그 종목만)
    return: (allowed: bool, reason: str)
    """
    now = datetime.now(ZoneInfo("UTC"))
    with _symbol_signal_lock:
        cd_until = _symbol_cooldown_until.get(pair)
        if cd_until and now < cd_until:
            remaining = (cd_until - now).total_seconds() / 60
            return False, f"{pair} 반복신호 쿨다운 중 (남은 시간 {remaining:.1f}분)"

        window_start = now - timedelta(minutes=SYMBOL_REPEAT_WINDOW_MINUTES)
        history = [t for t in _symbol_signal_history.get(pair, []) if t >= window_start]
        history.append(now)
        _symbol_signal_history[pair] = history

        if len(history) >= 2:
            _symbol_cooldown_until[pair] = now + timedelta(minutes=SYMBOL_REPEAT_COOLDOWN_MINUTES)
            return True, (f"{pair} {SYMBOL_REPEAT_WINDOW_MINUTES}분 내 {len(history)}번째 신호 → 이번엔 허용, "
                          f"이후 {SYMBOL_REPEAT_COOLDOWN_MINUTES}분간 이 종목만 쉬어감")

    return True, ""


def get_alpaca_position_qty(symbol: str) -> float:
    """
    Alpaca 계좌에 해당 심볼의 현재 보유 수량(절댓값)을 반환. 포지션 없으면 0.
    조회 실패 시 보수적으로 큰 값(99999)을 반환해서 신규 진입을 막는다(애매하면 차단).
    """
    url = f"{ALPACA_TRADE_BASE_URL}/v2/positions/{symbol}"
    try:
        r = requests.get(url, headers=ALPACA_HEADERS, timeout=10)
        if r.status_code == 404:
            return 0.0
        if r.status_code == 200:
            return abs(float(r.json().get("qty", 0)))
        print(f"[Alpaca] 포지션 수량 조회 status={r.status_code} body={r.text}")
        return 99999.0
    except Exception as e:
        print("[Alpaca] 포지션 수량 조회 실패:", e)
        return 99999.0


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

    url = f"{OANDA_BASE_URL}/v3/accounts/{ACCOUNT_ID}/openTrades"   # 🟥 [FIX-E2]
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


# ============================================================
# 🟥 [FIX-D2] 시트 접근 공통 헬퍼
# ------------------------------------------------------------
#  기존 코드는 _get_sheet()를 두 곳에서 호출했지만 정의가 어디에도 없었다.
#  NameError가 `except: pass`에 삼켜져서 EXCLUDED_SYMBOL / TIME_BLOCKED 표시가
#  시트에 조용히 안 찍혔고, 그래서 "왜 이 알림이 실행 안 됐는지"를 추적할 수 없었다.
#  여기서 정식으로 정의하고, 인증 객체는 캐싱해 매 호출마다 재인증하지 않게 한다.
# ============================================================
GOOGLE_SHEET_NAME = os.getenv("GOOGLE_SHEET_NAME", "민균 FX trading result")
GOOGLE_CREDS_PATH = os.getenv("GOOGLE_CREDS_PATH", "/etc/secrets/google_credentials.json")
_gs_client = None
_gs_client_lock = threading.Lock()


def _get_gspread_client():
    """gspread 클라이언트를 한 번만 인증해서 재사용. 실패 시 None."""
    global _gs_client
    if _gs_client is not None:
        return _gs_client
    with _gs_client_lock:
        if _gs_client is not None:
            return _gs_client
        try:
            scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
            creds = ServiceAccountCredentials.from_json_keyfile_name(GOOGLE_CREDS_PATH, scope)
            _gs_client = gspread.authorize(creds)
        except Exception as e:
            print(f"❌ [시트] 인증 실패: {e}")
            _gs_client = None
        return _gs_client


def _get_spreadsheet():
    """메인 스프레드시트 핸들. 실패 시 None."""
    c = _get_gspread_client()
    if c is None:
        return None
    try:
        return c.open(GOOGLE_SHEET_NAME)
    except Exception as e:
        print(f"❌ [시트] '{GOOGLE_SHEET_NAME}' 열기 실패: {e}")
        return None


def _get_sheet():
    """메인 탭(sheet1) 핸들. 실패 시 None. (기존 코드가 호출하던 이름 그대로 정의)"""
    ss = _get_spreadsheet()
    if ss is None:
        return None
    try:
        return ss.sheet1
    except Exception as e:
        print(f"❌ [시트] sheet1 접근 실패: {e}")
        return None


# 메인 시트 컬럼 번호(1-indexed) — 매직넘버를 한 곳에 모아둔다.
COL_DECISION = 5
COL_SCORE = 6
COL_FINAL_DECISION = 14
COL_RESULT = 17
COL_PRICE = 20
COL_TP = 21
COL_SL = 22
COL_PNL = 23
COL_QUANTITY = 24
COL_TOTAL_PNL = 25
COL_OUTCOME_ANALYSIS = 34


def _finalize_sheet_row(row_idx, effective_decision=None, gpt_decision=None, note=None, quantity=None):
    """
    🟥 [FIX-D1 / FIX-D4] 웹훅 처리가 끝난 뒤, 그 행에 "실제로 무슨 일이 있었는지"를 확정 기록.

    · COL_DECISION(5)      ← EXECUTED_BUY / SKIPPED_xxx / WAIT / BLOCKED_xxx
      기존엔 실행 게이트 이전 값(BUY/SELL)이 그대로 남아서 시트 승률이 실체결과 달랐다.
    · COL_FINAL_DECISION(14) ← GPT가 실제로 뭐라고 했는지
      기존엔 항상 None인 `decision` 변수를 써서 1,048행 전부 공백이었다.
    · COL_OUTCOME_ANALYSIS(34) ← 보조 메모

    셀 단위 3회 호출 대신 한 번의 batch_update로 처리해 API 호출과 경쟁 창을 줄인다.
    """
    if not row_idx:
        return
    try:
        sh = _get_sheet()
        if sh is None:
            return
        updates = []
        if effective_decision is not None:
            updates.append({"range": f"E{row_idx}", "values": [[str(effective_decision)]]})
        if gpt_decision is not None:
            updates.append({"range": f"N{row_idx}", "values": [[str(gpt_decision)]]})
        if note:
            updates.append({"range": f"AH{row_idx}", "values": [[str(note)[:400]]]})
        # 🟥 [FIX-D9] 실제 주문 수량(주식=주, FX=units)을 quantity(24열, X)에 남긴다.
        #    이게 없으면 결과추적이 FX 수량을 100,000으로 하드코딩할 수밖에 없고,
        #    리스크 기반 사이징(FIX-E7) 도입 후 total_pnl이 최대 100배 부풀려진다.
        if quantity:
            updates.append({"range": f"X{row_idx}", "values": [[abs(int(quantity))]]})
        if updates:
            _sheets_write_throttle()      # 🟥 [FIX-F1]
            sh.batch_update(updates)
    except Exception as e:
        print(f"⚠️ [시트] row {row_idx} 최종결과 기록 실패: {e}")


# ============================================================
# 🟥 [FIX-I1] 전략명 해석 — 버전이 바뀌어도 threshold가 따라오게
# ------------------------------------------------------------
#  문제: Pine 전략 이름을 A5 → A10 → A11로 올릴 때마다 여기 dict에 키를 추가하지 않으면
#        '미등록 전략명'으로 빠진다. 그래서 Pine alert에 이름을 A5로 하드코딩해뒀는데,
#        그 결과 시트의 strategy 컬럼이 항상 A5로 찍혀서 "어느 버전이 낸 신호인지"를
#        구분할 수 없게 됐다. A/B 테스트를 하려는 지금 이건 치명적이다.
#  해결: 이름을 정규화하고 접두사로 매칭한다. BUY_STOCK_PORTFOLIO_* 는 버전과 무관하게
#        같은 threshold를 쓰므로, Pine에서 실제 이름을 그대로 보내도 안전하다.
# ============================================================
# ============================================================
# 🟥 [FIX-TH1] 점수 문턱을 '측정 결과'에 맞춰 사실상 해제한다.
# ------------------------------------------------------------
#  ■ 왜 바꾸나 (2026-08-25, 실거래 53건 실측)
#      점수 ↔ 손익 상관  r = -0.296  (t = -2.21, n = 53, p ≈ 0.03)
#      → 점수가 '낮을수록' 성적이 좋다. 부호가 반대다.
#
#        점수 < -2 : 19건, 승률 57.9%, 총 +$225.55
#        점수 ≥ -2 : 34건, 승률 38.2%, 총 -$218.54
#
#      즉 문턱은 '나쁜 신호를 걸러내는 장치'가 아니라
#      **제일 좋은 신호를 골라서 버리는 장치**로 작동해 왔다.
#      실제로 '-3 미만' 구간에서 차단된 4건은 4건 모두 TP를 쳤다(4/4).
#
#  ■ 그래서 어떻게 하나
#      점수를 뒤집어서 쓰지는 않는다 — n=53 짜리 노이즈에 맞추는 짓이다.
#      대신 **문을 열어두고 점수는 계속 기록만** 한다. 표본이 쌓이면 그때 판단한다.
#      기본값 -99 = 사실상 전부 통과. 환경변수로 언제든 다시 조일 수 있다.
#
#  ■ 덤으로 고쳐지는 것
#      FX를 A12 스크립트로 옮기면서 전략명이 BUY_STOCK_PORTFOLIO_* 로 바뀌었고,
#      그 바람에 FX 문턱이 -7.0 → -2.5 로 아무도 모르게 올라가 있었다.
#      (GBP_USD -4.8, AUD_USD -4.1 이 그래서 차단됐다)
# ============================================================
STOCK_SCORE_THRESHOLD = float(os.getenv("STOCK_SCORE_THRESHOLD", "-99"))
FX_SCORE_THRESHOLD = float(os.getenv("FX_SCORE_THRESHOLD", "-99"))

STRATEGY_PREFIX_THRESHOLDS = [
    ("BUY_STOCK_PORTFOLIO", None),   # None = 상품 종류에 따라 위 환경변수로 결정
    ("BUY_ENTRY_BAR_CLOSE", None),
    ("SELL_ENTRY_BAR_CLOSE", None),
    ("BALANCE_BREAKOUT", 4.5),
]


def _normalize_strategy_name(name: str) -> str:
    """'BUY STOCK PORTFOLIO A10' / 'buy-stock-portfolio-a10' → 'BUY_STOCK_PORTFOLIO_A10'"""
    n = str(name or "").strip().upper()
    for ch in (" ", "-", ".", "/"):
        n = n.replace(ch, "_")
    while "__" in n:
        n = n.replace("__", "_")
    return n.strip("_")


def resolve_strategy_threshold(strategy_name: str, table: dict, is_stock: bool):
    """
    (threshold, 매칭방식) 반환. 등록된 이름이 없어도 접두사로 안전하게 떨어진다.
    """
    # 🟥 [FIX-TH1] 상품 종류로 결정되는 기본 문턱. 전략명 표보다 이게 우선한다.
    auto = STOCK_SCORE_THRESHOLD if is_stock else FX_SCORE_THRESHOLD

    raw = str(strategy_name or "").strip()
    norm = _normalize_strategy_name(raw)

    for prefix, thr in STRATEGY_PREFIX_THRESHOLDS:
        if norm.startswith(prefix):
            if thr is None:
                return auto, f"접두사({prefix}*) → {'주식' if is_stock else 'FX'} 기본 {auto}"
            return thr, f"접두사 매칭({prefix}*)"

    if raw in table:
        return table[raw], "정확히 일치"
    norm_table = {_normalize_strategy_name(k): v for k, v in table.items()}
    if norm in norm_table:
        return norm_table[norm], "정규화 후 일치"

    return auto, f"미등록 → {'주식' if is_stock else 'FX'} 기본 {auto}"


def _log_blocked_alert(pair, signal, alert_name, reason):
    """
    🟥 [FIX-D6b] 조기 차단된 알림을 메인 시트에 가볍게 한 줄 남긴다.
    지표·GPT를 전혀 태우지 않으므로 채울 수 있는 칸만 채운다.
    """
    try:
        sh = _get_sheet()
        if sh is None:
            return
        row = [""] * 37
        row[0] = str(datetime.now(ZoneInfo("America/New_York")))   # timestamp
        row[1] = pair or ""                                        # symbol
        row[2] = alert_name or ""                                  # strategy
        row[3] = signal or ""                                      # signal_type
        row[4] = f"BLOCKED_{reason}"                               # decision
        row[15] = f"진입 금지 종목 — {reason} (지표/GPT 호출 없이 조기 차단)"   # reason
        row[16] = "미정"                                            # result(가상평가 대상으로 남겨둠)
        sh.append_row(row, insert_data_option="INSERT_ROWS", table_range="A1")
    except Exception as e:
        print(f"⚠️ [시트] 조기차단 기록 실패({pair}/{reason}): {e}")


# ============================================================
# 🟥 [FIX-F1] Google Sheets 쓰기 배치화 + 레이트리밋 대응
# ------------------------------------------------------------
#  배포 실패의 직접 원인:
#    APIError [429] Quota exceeded ... 'Write requests per minute per user'
#  Google Sheets는 사용자당 분당 60회 쓰기가 한도인데,
#  evaluate_pending_outcomes()가 행 1개당 update_cell()을 최대 5번 호출했다.
#  미평가 행이 수백 개면 수천 번의 개별 쓰기가 발생해 즉시 한도를 넘긴다.
#  → 셀 단위 쓰기를 모두 모아 한 번의 batch_update로 보낸다(수천 회 → 수 회).
#    남는 호출에도 토큰버킷 스로틀을 걸어 한도 자체를 넘지 않게 한다.
# ============================================================
SHEETS_WRITES_PER_MIN = int(os.getenv("SHEETS_WRITES_PER_MIN", "50"))   # 한도 60에서 안전마진
_sheets_write_lock = threading.Lock()
_sheets_write_times: list = []


def _sheets_write_throttle():
    """분당 쓰기 횟수를 SHEETS_WRITES_PER_MIN 이하로 유지한다(필요하면 대기)."""
    while True:
        with _sheets_write_lock:
            now = _t.time()
            _sheets_write_times[:] = [t for t in _sheets_write_times if now - t < 60.0]
            if len(_sheets_write_times) < SHEETS_WRITES_PER_MIN:
                _sheets_write_times.append(now)
                return
            wait = 60.0 - (now - _sheets_write_times[0]) + 0.2
        print(f"⏳ [시트] 쓰기 한도 근접 → {wait:.1f}초 대기")
        _t.sleep(max(0.2, wait))


def _col_letter(idx: int) -> str:
    """1-indexed 컬럼 번호 → A1 표기 문자(1→A, 27→AA)."""
    out = ""
    while idx > 0:
        idx, rem = divmod(idx - 1, 26)
        out = chr(65 + rem) + out
    return out


def _flush_sheet_updates(sheet, updates, chunk=400, label="시트"):
    """
    모아둔 셀 업데이트를 batch_update로 한 번에 보낸다.
    updates: [(row, col, value), ...]
    반환: 실제로 반영된 셀 개수
    """
    if not sheet or not updates:
        return 0
    done = 0
    for start in range(0, len(updates), chunk):
        part = updates[start:start + chunk]
        body = [{"range": f"{_col_letter(c)}{r}", "values": [[v]]} for r, c, v in part]
        for attempt in range(4):
            try:
                _sheets_write_throttle()
                sheet.batch_update(body)
                done += len(part)
                break
            except Exception as e:
                msg = str(e)
                if "429" in msg or "Quota exceeded" in msg:
                    wait = 15 * (attempt + 1)
                    print(f"⏳ [{label}] 429 → {wait}초 후 재시도 ({attempt+1}/4)")
                    _t.sleep(wait)
                    continue
                print(f"❌ [{label}] batch_update 실패: {e}")
                break
    print(f"📝 [{label}] {done}/{len(updates)}개 셀 일괄 반영 완료 "
          f"(개별 쓰기였다면 API 호출 {len(updates)}회 → 실제 {max(1, (len(updates)+chunk-1)//chunk)}회)")
    return done


def _mark_sheet_result(row_idx, label):
    """
    [FIX-D2] 특정 행의 result 컬럼에 차단/스킵 사유를 기록한다.
    기존에 _get_sheet() 미정의로 실패하던 자리를 대체한다.
    """
    if not row_idx or not label:
        return
    try:
        sh = _get_sheet()
        if sh:
            # 🟥 [FIX-D2b] 차단 사유를 result(17열)에 쓰면 evaluate_pending_outcomes()가
            #    `result_col not in ("", "미정")` 조건으로 그 행을 영영 건너뛴다.
            #    그러면 "이 차단이 옳았는지"를 가상평가로 검증할 수 없다.
            #    → result는 미정으로 두고 outcome_analysis(34열)에 사유만 남긴다.
            _sheets_write_throttle()      # 🟥 [FIX-F1]
            sh.update_cell(row_idx, COL_OUTCOME_ANALYSIS, str(label))
            print(f"🏷️ [시트] row {row_idx} outcome_analysis ← {label}")
    except Exception as e:
        print(f"⚠️ [시트] row {row_idx} result 기록 실패({label}): {e}")


def correct_sheet_trade_prices(row_idx, price, tp, sl):
    """
    place_order_alpaca()가 주문 직전 실시간가로 TP/SL을 다시 맞춘 뒤에는,
    이미 log_trade_result()로 시트에 기록해둔 (옛날) price/tp/sl이 실제 주문값과 달라진다.
    이 함수가 해당 행의 price/tp/sl 컬럼을 실제 사용된 최종값으로 다시 덮어써서
    시트와 Alpaca가 항상 같은 숫자를 보게 한다. (결과추적도 이 보정된 값을 기준으로 판정하게 됨)

    🟥 [FIX-D3] 3번의 update_cell을 1번의 batch update로 합쳤다.
       기존엔 셀 단위로 3번 쓰면서 동시 웹훅 시 다른 행을 덮어쓸 창이 넓었고
       API 호출도 3배였다.
    """
    if row_idx is None:
        return
    try:
        sh = _get_sheet()
        if sh is None:
            return
        digits = 5
        _sheets_write_throttle()          # 🟥 [FIX-F1]
        sh.update(
            f"T{row_idx}:V{row_idx}",   # T=20(price), U=21(tp), V=22(sl)
            [[round(float(price), digits), round(float(tp), digits), round(float(sl), digits)]],
        )
        print(f"✅ [시트보정] row {row_idx} price/tp/sl을 실제 주문값으로 갱신 "
              f"(price={price}, tp={tp}, sl={sl})")
    except Exception as e:
        print(f"❌ [시트보정] row {row_idx} 업데이트 실패: {e}")


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


# ============================================================
# 🟥 [FIX-P1] 포트폴리오 상관 인식 사이징  ("1등 업데이트")
# ------------------------------------------------------------
#  ■ 무엇이 문제였나
#    알람이 여러 개 울리면 봇은 각 건을 **완전히 독립적으로** 계산했다.
#    현재 감시 목록(PLTR·META·CEG·GEV·ANET·MU·PANW·CRWV)은 8개처럼 보이지만
#    실제로는 전부 'AI 인프라' 한 가지 베팅이다.
#      · CEG(전력)·GEV(발전설비)는 전력회사처럼 보이지만 AI 데이터센터 수요로 움직인다
#      · 8건에 각각 0.5% 리스크를 걸면 분산이 아니라 AI에 4% 몰빵이다
#    AI 테마가 하루 무너지면 8개가 동시에 손절된다. 지금 구조는 그걸 못 본다.
#
#  ■ 무엇을 하는가
#    주문 직전에 세 가지를 본다.
#      (1) 지금 Alpaca + OANDA에 **뭘 들고 있는지** 전부 조회
#      (2) 새 종목이 어느 **그룹**에 속하고, 기존 노출과 **같은 방향**인지
#      (3) 그룹 한도를 넘는 만큼 수량을 **줄인다** (넘치면 0 = 스킵)
#
#  ■ 반대 방향은 줄이지 않는다
#    금 롱을 든 상태에서 은 숏이 들어오면 그건 집중이 아니라 헤지다.
#    부호를 살려서 순노출(net)로 계산하는 이유가 이것이다.
#
#  ■ FX는 'USD 방향'으로 묶는다
#    EUR_USD 롱과 GBP_USD 롱은 서로 다른 거래처럼 보이지만 둘 다 **USD 숏**이다.
#    반대로 USD_JPY 롱은 **USD 롱**이라 앞의 둘과 상쇄된다.
#    그래서 FX는 통화쌍이 아니라 USD 노출로 환산해 한 그룹에 넣는다.
#
#  ■ 실패 시 동작
#    포지션 조회가 실패하면 축소하지 않는다(factor=1.0). 지금과 같은 동작이라
#    API 한 번 삐끗했다고 거래가 전부 멈추지는 않는다. 대신 로그에 크게 남긴다.
# ============================================================

# 계좌 전체 자산. 비워두면 Alpaca equity + OANDA 잔고를 합쳐서 쓴다.
PORTFOLIO_EQUITY_USD = os.getenv("PORTFOLIO_EQUITY_USD", "").strip()

# 한 그룹이 가질 수 있는 최대 명목 노출 (자산 대비 %).
# 25%면 AI인프라에 자산의 1/4까지만 실린다. 그 뒤 신호는 자동으로 작아진다.
PORTFOLIO_GROUP_MAX_PCT = float(os.getenv("PORTFOLIO_GROUP_MAX_PCT", "25"))

# 전체 명목 노출 상한 (자산 대비 %).
PORTFOLIO_TOTAL_MAX_PCT = float(os.getenv("PORTFOLIO_TOTAL_MAX_PCT", "150"))

# 축소 후 이 비율보다 작아지면 아예 진입하지 않는다.
# 너무 작은 포지션은 수수료·스프레드만 내고 의미가 없다.
PORTFOLIO_MIN_FACTOR = float(os.getenv("PORTFOLIO_MIN_FACTOR", "0.30"))

# 상관 사이징 자체를 끄는 스위치 (문제 생겼을 때 즉시 원복용)
PORTFOLIO_SIZING_ENABLED = os.getenv("PORTFOLIO_SIZING_ENABLED", "true").lower() == "true"

# ------------------------------------------------------------
# 종목 → 분야 지도
#
#  ★★ 형님이 관리하실 필요 없습니다. ★★
#     실제 판단은 [FIX-P2] 의 '실측 상관계수'가 합니다. 트레이딩뷰에 종목을
#     추가하기만 하면 봇이 그 종목의 일봉을 받아서 기존 보유분과의 상관을
#     직접 계산합니다. 이 표에 없어도 정상 작동합니다.
#
#  이 표는 두 가지 용도로만 남아 있습니다:
#     1. 가격을 못 받아왔을 때의 비상용 (신규 상장·API 장애 등)
#     2. 시트에 '분야' 이름을 예쁘게 찍기 위한 라벨
#     둘 다 없어도 한도는 정상 작동합니다.
# ------------------------------------------------------------
SYMBOL_GROUP = {
    # ── AI 인프라 ────────────────────────────────────────────
    #  형님 현재 감시목록이 사실상 전부 여기다. 반도체·전력·클라우드·네트워크가
    #  다 하나의 'AI 데이터센터 투자' 사이클에 묶여 있다.
    "NVDA": "AI인프라", "AMD": "AI인프라", "MU": "AI인프라", "AVGO": "AI인프라",
    "MRVL": "AI인프라", "TSM": "AI인프라", "SMCI": "AI인프라", "DELL": "AI인프라",
    "ANET": "AI인프라", "VRT": "AI인프라", "CRWV": "AI인프라", "NBIS": "AI인프라",
    "CEG": "AI인프라",   # 원전 전력 — AI 데이터센터 전력 수요로 재평가된 종목
    "GEV": "AI인프라",   # 발전설비 — 같은 이유
    "VST": "AI인프라", "TLN": "AI인프라", "EQIX": "AI인프라", "DLR": "AI인프라",
    "PLTR": "AI인프라", "META": "AI인프라", "MSFT": "AI인프라", "GOOGL": "AI인프라",
    "GOOG": "AI인프라", "AMZN": "AI인프라", "ORCL": "AI인프라", "NOW": "AI인프라",
    "SNOW": "AI인프라", "APP": "AI인프라", "PANW": "AI인프라", "CRWD": "AI인프라",
    "ZS": "AI인프라", "NET": "AI인프라", "AAPL": "AI인프라",
    "NAS100_USD": "AI인프라",   # 나스닥100 = 위 종목들의 묶음. 따로 세면 안 된다.

    # ── 금융 ────────────────────────────────────────────────
    "JPM": "금융", "BAC": "금융", "GS": "금융", "MS": "금융", "WFC": "금융",
    "C": "금융", "SCHW": "금융", "BLK": "금융", "AXP": "금융", "V": "금융",
    "MA": "금융", "PYPL": "금융", "COF": "금융", "USB": "금융",
    "BRK.B": "금융", "BRK.A": "금융",   # 🟥 [FIX-P1b/D10] 추천 목록엔 있는데 지도에 없어서
                                        #   막상 담으면 '기타'로 빠져 금융 노출이 안 올라갔다

    # ── 헬스케어 ────────────────────────────────────────────
    "UNH": "헬스케어", "JNJ": "헬스케어", "LLY": "헬스케어", "MRK": "헬스케어",
    "PFE": "헬스케어", "ABBV": "헬스케어", "TMO": "헬스케어", "ABT": "헬스케어",
    "AMGN": "헬스케어", "ISRG": "헬스케어", "GILD": "헬스케어", "VRTX": "헬스케어",

    # ── 에너지(주식) ────────────────────────────────────────
    "XOM": "에너지", "CVX": "에너지", "COP": "에너지", "SLB": "에너지",
    "EOG": "에너지", "OXY": "에너지", "PSX": "에너지", "MPC": "에너지", "VLO": "에너지",

    # ── 필수소비재 ──────────────────────────────────────────
    "PG": "소비재", "KO": "소비재", "PEP": "소비재", "COST": "소비재",
    "WMT": "소비재", "MCD": "소비재", "NKE": "소비재", "HD": "소비재",
    "TGT": "소비재", "SBUX": "소비재", "MDLZ": "소비재", "CL": "소비재",

    # ── 산업재 ──────────────────────────────────────────────
    "CAT": "산업재", "DE": "산업재", "HON": "산업재", "GE": "산업재",
    "UNP": "산업재", "UPS": "산업재", "LMT": "산업재", "RTX": "산업재",
    "BA": "산업재", "NOC": "산업재", "ETN": "산업재", "EMR": "산업재",

    # ── 유틸리티 (AI 전력주는 위로 뺐다) ────────────────────
    "NEE": "유틸리티", "DUK": "유틸리티", "SO": "유틸리티",
    "AEP": "유틸리티", "D": "유틸리티", "EXC": "유틸리티",

    # ── 통신 ────────────────────────────────────────────────
    "T": "통신", "VZ": "통신", "TMUS": "통신",

    # ── 자동차/EV ───────────────────────────────────────────
    "TSLA": "자동차", "RIVN": "자동차", "F": "자동차", "GM": "자동차", "LCID": "자동차",

    # ── 크립토 연동 ─────────────────────────────────────────
    #  BTC 와 코인 관련주는 같이 움직인다. 그리고 위험자산이라
    #  AI인프라와도 상관이 높지만, 별개 그룹으로 두고 한도로 제어한다.
    "COIN": "크립토", "MSTR": "크립토", "MARA": "크립토", "RIOT": "크립토",
    "BTC": "크립토", "BTCUSD": "크립토", "ETH": "크립토", "ETHUSD": "크립토",

    # ── OANDA: 귀금속 ───────────────────────────────────────
    "XAU": "귀금속", "XAG": "귀금속", "XPT": "귀금속", "XPD": "귀금속",

    # ── OANDA: 에너지 원자재 ────────────────────────────────
    "WTICO": "원유", "BCO": "원유", "NATGAS": "천연가스",

    # ── OANDA: 지수 (나스닥은 AI인프라로 뺐다) ──────────────
    "SPX500": "미국지수", "US30": "미국지수", "US2000": "미국지수",
    "DE30": "유럽지수", "EU50": "유럽지수", "UK100": "유럽지수",
    "JP225": "아시아지수", "HK33": "아시아지수", "AU200": "아시아지수",

    # ── OANDA: 채권 ─────────────────────────────────────────
    "USB02Y": "채권", "USB05Y": "채권", "USB10Y": "채권", "USB30Y": "채권",
    "DE10YB": "채권", "UK10YB": "채권",

    # ── OANDA: 농산물/기타 ──────────────────────────────────
    "CORN": "농산물", "WHEAT": "농산물", "SOYBN": "농산물",
    "SUGAR": "농산물", "XCU": "산업금속",
}

# 🟥 [FIX-DV2] 분산 후보를 Alpaca ETF 로 바꿨으므로 분야 매핑도 같이 옮긴다.
#    안 하면 추천후보 탭의 '지금 어디에 쏠려 있나' 표가 전부 '기타' 로 나온다.
SYMBOL_GROUP.update({
    "GLD": "귀금속", "IAU": "귀금속", "SLV": "귀금속", "GDX": "귀금속",
    "XLE": "원유·에너지", "USO": "원유·에너지", "BNO": "원유·에너지", "UNG": "원유·에너지",
    "TLT": "채권", "IEF": "채권", "SHY": "채권", "AGG": "채권", "BND": "채권",
    "SPY": "미국지수", "IVV": "미국지수", "VOO": "미국지수",
    "IWM": "미국지수", "RSP": "미국지수", "QQQ": "미국지수", "DIA": "미국지수",
    "EWJ": "해외지수", "VGK": "해외지수", "EEM": "해외지수", "EFA": "해외지수", "VXUS": "해외지수",
    "CPER": "산업금속", "XME": "산업금속", "DBB": "산업금속",
})

#: FX 통화 코드. 둘 다 여기 있으면 '통화쌍'으로 보고 USD 노출로 환산한다.
_FX_CURRENCIES = {
    "USD", "EUR", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF",
    "SEK", "NOK", "DKK", "SGD", "HKD", "MXN", "ZAR", "TRY",
    "PLN", "CZK", "HUF", "CNH", "THB",
}

PORTFOLIO_GROUP_FX_USD = "USD통화"
PORTFOLIO_GROUP_OTHER = "기타"


def _split_instrument(symbol: str) -> tuple[str, str]:
    """'EUR_USD' → ('EUR','USD'),  'XAU_USD' → ('XAU','USD'),  'PLTR' → ('PLTR','')"""
    s = (symbol or "").upper().replace("/", "_").replace("-", "_")
    parts = [p for p in s.split("_") if p]
    if len(parts) >= 2:
        return parts[0], parts[-1]
    return (parts[0] if parts else ""), ""


def portfolio_group_for(symbol: str) -> str:
    """이 종목이 어느 상관 그룹에 속하는가."""
    sym = (symbol or "").upper().replace("/", "_").replace("-", "_")
    if sym in SYMBOL_GROUP:
        return SYMBOL_GROUP[sym]
    base, quote = _split_instrument(sym)
    # 통화 vs 통화 = FX. USD 노출 하나로 묶는다.
    if base in _FX_CURRENCIES and quote in _FX_CURRENCIES:
        # 🟥 [FIX-P1b/D6] USD가 낀 쌍만 'USD 노출'로 묶는다.
        #   EUR_GBP 같은 크로스를 USD통화에 넣으면 방향 부호를 만들어낼 수 없고,
        #   명목도 USD가 아니라 GBP 기준이라 진짜 USD 포지션을 상쇄해버린다.
        if base == "USD" or quote == "USD":
            return PORTFOLIO_GROUP_FX_USD
        return f"FX_{base}{quote}"
    if base in SYMBOL_GROUP:
        return SYMBOL_GROUP[base]
    return PORTFOLIO_GROUP_OTHER


def _signed_direction(symbol: str, side: str) -> int:
    """
    노출의 부호. 같은 그룹 안에서 같은 부호끼리는 더해지고 반대면 상쇄된다.

    ★ FX만 특별 취급한다.
      EUR_USD 롱 = EUR 사고 USD 팜 = **USD 숏** → -1
      USD_JPY 롱 = USD 사고 JPY 팜 = **USD 롱** → +1
      이렇게 해야 "EUR 롱 + USD_JPY 롱"이 서로 상쇄된다는 사실이 반영된다.
    """
    s = 1 if str(side).upper() in ("BUY", "LONG") else -1
    base, quote = _split_instrument(symbol)
    if base in _FX_CURRENCIES and quote in _FX_CURRENCIES:
        if base == "USD":
            return s          # USD_XXX 롱 = USD 롱
        if quote == "USD":
            return -s         # XXX_USD 롱 = USD 숏
    return s


def get_portfolio_equity_usd() -> float | None:
    """Alpaca equity + OANDA 잔고. 환경변수로 고정할 수도 있다."""
    if PORTFOLIO_EQUITY_USD:
        try:
            return float(PORTFOLIO_EQUITY_USD)
        except ValueError:
            pass
    total = 0.0
    got = False
    eq = get_alpaca_account_equity()
    if eq:
        total += float(eq); got = True
    bal = get_oanda_account_balance()
    if bal:
        total += float(bal); got = True
    return total if got and total > 0 else None


def get_portfolio_positions() -> list[dict] | None:
    """
    Alpaca + OANDA 의 열린 포지션 전부.
    return: [{symbol, side, notional_usd, group, signed_usd}, ...]
            조회가 하나라도 실패하면 None (그러면 축소를 걸지 않는다)
    """
    out: list[dict] = []
    # 🟥 [FIX-P1b/D2] 예전엔 둘 중 하나만 성공해도 ok=True 였다.
    #   Alpaca 조회가 실패하면 주식 노출이 0%로 보이고, 상관 한도가
    #   무력화된다 — 이 기능이 막으려던 바로 그 상황이 조용히 통과한다.
    alpaca_ok = False
    oanda_ok = None          # None = 시도 안 함(키 없음), False = 실패

    # ── Alpaca 주식 ──
    try:
        r = requests.get(f"{ALPACA_TRADE_BASE_URL}/v2/positions",
                         headers=ALPACA_HEADERS, timeout=10)
        if r.status_code == 200:
            alpaca_ok = True
            for p in r.json() or []:
                sym = p.get("symbol", "")
                qty = float(p.get("qty", 0) or 0)
                mv = abs(float(p.get("market_value", 0) or 0))
                if mv <= 0 and qty:
                    mv = abs(qty) * float(p.get("current_price", 0) or 0)
                side = "LONG" if qty >= 0 else "SHORT"
                g = portfolio_group_for(sym)
                out.append({"symbol": sym, "side": side, "notional_usd": mv,
                            "group": g, "signed_usd": mv * _signed_direction(sym, side)})
        else:
            print(f"⚠️ [포트폴리오] Alpaca 포지션 조회 status={r.status_code}")
    except Exception as e:
        print(f"⚠️ [포트폴리오] Alpaca 포지션 조회 실패: {e}")

    # ── OANDA (FX / 금 / 지수 등) ──
    if OANDA_API_KEY and ACCOUNT_ID:
        try:
            r = requests.get(f"{OANDA_BASE_URL}/v3/accounts/{ACCOUNT_ID}/openTrades",
                             headers={"Authorization": f"Bearer {OANDA_API_KEY}"}, timeout=10)
            if r.ok:
                oanda_ok = True
                for t in (r.json() or {}).get("trades", []):
                    inst = t.get("instrument", "")
                    units = float(t.get("currentUnits", 0) or 0)
                    px = float(t.get("price", 0) or 0)
                    if not inst or units == 0:
                        continue
                    base, quote = _split_instrument(inst)
                    # units 는 base 통화(또는 온스) 단위다. USD 명목으로 환산한다.
                    if base == "USD":
                        notional = abs(units)                 # USD_JPY: units 가 이미 USD
                    elif quote == "USD":
                        notional = abs(units) * px            # EUR_USD, XAU_USD
                    else:
                        notional = abs(units) * px            # 크로스는 근사
                    side = "LONG" if units > 0 else "SHORT"
                    g = portfolio_group_for(inst)
                    out.append({"symbol": inst, "side": side, "notional_usd": notional,
                                "group": g, "signed_usd": notional * _signed_direction(inst, side)})
            else:
                oanda_ok = False
                print(f"⚠️ [포트폴리오] OANDA openTrades status={r.status_code}")
        except Exception as e:
            oanda_ok = False
            print(f"⚠️ [포트폴리오] OANDA 포지션 조회 실패: {e}")

    if not alpaca_ok or oanda_ok is False:
        print(f"⚠️ [포트폴리오] 부분 조회 실패(alpaca={alpaca_ok}, oanda={oanda_ok}) "
              f"→ 반쪽 데이터로 한도를 계산하면 위험하므로 전체 무효 처리")
        return None
    return out


# ------------------------------------------------------------
# 🟥 [FIX-P2] 상관관계를 '표'가 아니라 '실제 가격'에서 잰다
#
#  ■ 왜 바꿨나
#    처음엔 종목→분야 표(SYMBOL_GROUP)를 손으로 관리하게 만들었다.
#    그건 잘못된 설계다. 트레이딩뷰에 종목 하나 추가할 때마다 파이썬 코드를
#    고쳐야 한다면, 언젠가 반드시 빠뜨리고, 빠뜨린 종목은 '기타'로 새서
#    한도가 걸리지 않는다. 즉 **제일 필요할 때 조용히 작동을 멈춘다.**
#
#  ■ 지금 방식
#    새 종목과 현재 보유 종목들의 **일봉 수익률 상관계수를 직접 계산**한다.
#    · PLTR vs META  → 실제로 높게 나온다 (같이 움직이니까)
#    · PLTR vs XAU   → 낮게 나온다
#    · EUR_USD vs GBP_USD → 높게 나온다 (둘 다 USD 반대편)
#    · EUR_USD vs USD_JPY → 음수로 나온다 (자동으로 상쇄 처리됨)
#    표를 손으로 관리할 필요가 없다. **트레이딩뷰에 종목만 추가하면 끝이다.**
#
#  ■ 표는 어떻게 되나
#    SYMBOL_GROUP 은 이제 '보조 수단'이다. 가격을 못 받아왔을 때만 쓴다.
#    (신규 상장, 데이터 부족, API 장애 등) 형님이 관리할 필요 없다.
#
#  ■ 비용
#    일봉 시세는 하루 한 번만 받으면 된다(24시간 캐시). 상관계수도 같이 캐시한다.
#    포지션이 5개면 하루에 6번 정도 캔들을 더 받는다. 그게 전부다.
# ------------------------------------------------------------

CORR_LOOKBACK_DAYS = int(os.getenv("CORR_LOOKBACK_DAYS", "120"))   # 상관 계산에 쓸 일봉 수
CORR_MIN_OVERLAP = int(os.getenv("CORR_MIN_OVERLAP", "40"))        # 겹치는 날이 이보다 적으면 포기
CORR_CACHE_TTL = float(os.getenv("CORR_CACHE_TTL", "86400"))       # 24시간
CORR_MAX_PEERS = int(os.getenv("CORR_MAX_PEERS", "20"))            # 비교할 보유종목 최대 개수

_series_cache: dict[str, tuple[float, dict]] = {}   # symbol -> (저장시각, {날짜: 종가})
_corr_cache: dict[tuple, tuple[float, float]] = {}  # (a,b) -> (저장시각, 상관계수)


def _daily_close_series(symbol: str) -> dict:
    """일봉 종가를 {'YYYY-MM-DD': 종가} 로. 실패하면 빈 dict."""
    now = _t.time()
    hit = _series_cache.get(symbol)
    if hit and (now - hit[0]) < CORR_CACHE_TTL:
        return hit[1]

    out: dict = {}
    try:
        df = get_candles(symbol, "D", CORR_LOOKBACK_DAYS)
        if df is not None and len(df) > 0:
            is_fx = not is_stock_pair(symbol)
            for _, row in df.iterrows():
                ts = str(row.get("time") or "")[:10]
                if not ts:
                    continue
                if is_fx:
                    # ★ OANDA 일봉은 뉴욕 17:00 시작 기준이라, 라벨이 'D일'인 캔들은
                    #   실제로는 D일 17:00 ~ D+1일 17:00 을 담는다. 주식 일봉(D+1일 장중)과
                    #   맞추려면 하루 밀어야 한다. 안 밀면 주식↔FX 상관이 엉뚱하게 낮게 나온다.
                    try:
                        ts = (datetime.fromisoformat(ts) + timedelta(days=1)).strftime("%Y-%m-%d")
                    except Exception:
                        pass
                out[ts] = float(row["close"])
    except Exception as e:
        print(f"⚠️ [상관] {symbol} 일봉 조회 실패: {e}")

    _series_cache[symbol] = (now, out)
    return out


def measured_correlation(a: str, b: str):
    """
    두 종목의 일봉 수익률 상관계수. 계산 불가면 None.

    ★ 종가가 아니라 '수익률'로 계산한다. 종가끼리 상관을 재면 둘 다 우상향이라는
      이유만으로 0.9 가 나온다(허위 상관). 실제로 같이 움직이는지는 수익률로 봐야 한다.
    """
    if a == b:
        return 1.0
    key = tuple(sorted((a, b)))
    now = _t.time()
    hit = _corr_cache.get(key)
    if hit and (now - hit[0]) < CORR_CACHE_TTL:
        return hit[1]

    sa, sb = _daily_close_series(a), _daily_close_series(b)
    common = sorted(set(sa) & set(sb))
    if len(common) < CORR_MIN_OVERLAP + 1:
        _corr_cache[key] = (now, None)
        return None

    ra, rb = [], []
    for i in range(1, len(common)):
        p0a, p1a = sa[common[i - 1]], sa[common[i]]
        p0b, p1b = sb[common[i - 1]], sb[common[i]]
        if p0a > 0 and p0b > 0:
            ra.append(p1a / p0a - 1.0)
            rb.append(p1b / p0b - 1.0)
    n = len(ra)
    if n < CORR_MIN_OVERLAP:
        _corr_cache[key] = (now, None)
        return None

    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra)
    vb = sum((y - mb) ** 2 for y in rb)
    if va <= 0 or vb <= 0:
        _corr_cache[key] = (now, None)
        return None
    rho = max(-1.0, min(1.0, cov / (va ** 0.5 * vb ** 0.5)))
    _corr_cache[key] = (now, rho)
    return rho


def _fallback_correlation(a: str, b: str) -> float:
    """가격을 못 받았을 때만 쓰는 보조 수단. 같은 분야면 0.8, 아니면 0."""
    ga, gb = portfolio_group_for(a), portfolio_group_for(b)
    if ga == gb and ga != PORTFOLIO_GROUP_OTHER:
        return 0.8
    return 0.0


def correlated_exposure(symbol: str, positions: list[dict]) -> tuple[float, list]:
    """
    '이 종목 방향에서 본' 현재 노출 합계.

        노출 = Σ ( 상관계수 × 그 포지션의 부호있는 명목 )

    같은 방향으로 강하게 상관된 포지션은 더해지고, 반대 방향이거나
    음의 상관이면 빼진다. 헤지가 자동으로 반영된다.

    return: (노출 합계, [(종목, 상관계수, 기여액), ...])
    """
    total = 0.0
    detail = []
    for p in positions[:CORR_MAX_PEERS]:
        peer = p["symbol"]
        rho = measured_correlation(symbol, peer)
        src = "실측"
        if rho is None:
            rho = _fallback_correlation(symbol, peer)
            src = "분야표"
        # 🟥 "LONG"/"BUY" 둘 다 매수로 받는다. 문자열 하나 어긋나면 부호가 통째로
        #    뒤집혀 '집중'을 '헤지'로 오판한다 — 조용히 반대로 도는 종류의 사고다.
        signed = p["notional_usd"] * (
            1 if str(p.get("side", "")).upper() in ("LONG", "BUY") else -1)
        contrib = rho * signed
        total += contrib
        detail.append((peer, round(rho, 2), round(contrib), src))
    return total, detail


# 🟥 [FIX-P1b/D5] 스냅샷 캐시.
#   상관 사이징은 한 신호에서 calc_alpaca_qty 가 2번 호출되며(사전체크+주문직전),
#   매번 Alpaca 계좌/포지션 + OANDA 잔고/트레이드 4개 요청을 냈다.
#   그 호출이 심볼 주문락 **안에서** 일어나 최악의 경우 100초까지 락을 잡는다.
#   같은 신호 처리 중에는 같은 스냅샷을 쓰면 충분하다.
_pf_cache = {"t": 0.0, "eq": None, "pos": None}
PORTFOLIO_CACHE_TTL = float(os.getenv("PORTFOLIO_CACHE_TTL", "10"))


def _pf_snapshot(force: bool = False):
    """(자산, 포지션목록). TTL 안에서는 재조회하지 않는다."""
    now = _t.time()
    if force or (now - _pf_cache["t"]) > PORTFOLIO_CACHE_TTL:
        _pf_cache["eq"] = get_portfolio_equity_usd()
        _pf_cache["pos"] = get_portfolio_positions()
        _pf_cache["t"] = now
    return _pf_cache["eq"], _pf_cache["pos"]


def summarize_portfolio_exposure() -> dict:
    """그룹별 노출 요약. 추천 후보 탭과 진단 엔드포인트에서 같이 쓴다."""
    equity, positions = _pf_snapshot(force=True)   # 진단·시트는 항상 최신으로
    if positions is None or not equity:
        return {"ok": False, "equity": equity, "positions": positions or [], "groups": {}}
    groups: dict[str, dict] = {}
    for p in positions:
        g = groups.setdefault(p["group"], {"gross": 0.0, "net": 0.0, "symbols": []})
        g["gross"] += p["notional_usd"]
        g["net"] += p["signed_usd"]
        g["symbols"].append(p["symbol"])
    for g in groups.values():
        g["gross_pct"] = round(g["gross"] / equity * 100, 2)
        g["net_pct"] = round(g["net"] / equity * 100, 2)
    return {"ok": True, "equity": equity, "positions": positions, "groups": groups,
            "total_gross_pct": round(sum(x["gross"] for x in groups.values()) / equity * 100, 2)}


def portfolio_size_factor(symbol: str, side: str, intended_notional_usd: float) -> tuple[float, str]:
    """
    이 진입을 얼마나 줄여야 하는가. 1.0 = 그대로, 0.0 = 스킵.

    🟥 [FIX-P2] 분야 표가 아니라 **실측 상관계수**로 계산한다.

        현재 노출 = Σ ( 상관계수(새종목, 보유종목) × 보유 명목 × 방향부호 )
        예상 노출 = | 현재 노출 + 이번 진입(부호 포함) |
        예산      = 자산 × PORTFOLIO_GROUP_MAX_PCT%

        예상 노출이 예산 이하  → 1.0
        넘으면                 → 남은 여유만큼만
        여유 없으면            → 0.0 (스킵)

    ★ 반대 방향이거나 음의 상관이면 노출이 줄어들어 자동으로 1.0 이 된다.
      헤지는 집중이 아니므로 막지 않는다.
    ★ 종목을 새로 추가할 때 코드를 고칠 필요가 없다. 가격만 있으면 계산된다.
    """
    if not PORTFOLIO_SIZING_ENABLED:
        return 1.0, "상관사이징 OFF"
    try:
        intended = abs(float(intended_notional_usd))
    except (TypeError, ValueError):
        return 1.0, "명목금액 불명 → 축소 없음"
    if intended <= 0:
        return 1.0, "명목금액 0"

    equity, positions = _pf_snapshot()
    if not equity or positions is None:
        print("⚠️ [상관사이징] 자산/포지션 조회 실패 → 축소 없이 진행(현행 유지). 원인 확인 필요.")
        return 1.0, "조회 실패 → 축소 없음"
    if not positions:
        return 1.0, "열린 포지션 없음"

    sign = 1 if str(side).upper() in ("BUY", "LONG") else -1
    exposure, detail = correlated_exposure(symbol, positions)

    total_gross = sum(p["notional_usd"] for p in positions)
    budget = equity * (PORTFOLIO_GROUP_MAX_PCT / 100.0)
    total_budget = equity * (PORTFOLIO_TOTAL_MAX_PCT / 100.0)

    projected = abs(exposure + sign * intended)
    if projected <= budget:
        f_group = 1.0
    else:
        room = budget - sign * exposure
        f_group = max(0.0, min(1.0, room / intended))

    room_total = total_budget - total_gross
    f_total = 1.0 if room_total >= intended else max(0.0, room_total / intended)
    factor = min(f_group, f_total)

    # 어떤 종목이 얼마나 겹쳐서 이렇게 됐는지 그대로 보여준다
    top = sorted(detail, key=lambda d: -abs(d[2]))[:3]
    peers = ", ".join(f"{s}(ρ{r:+.2f},{src})" for s, r, _, src in top) or "없음"
    detail_txt = (f"겹침={exposure/equity*100:+.1f}% 한도={PORTFOLIO_GROUP_MAX_PCT:.0f}% "
                  f"주요겹침[{peers}] → 계수 {factor:.2f}")

    if factor <= 0:
        return 0.0, f"{detail_txt} (한도 소진 — 스킵)"
    if factor < PORTFOLIO_MIN_FACTOR:
        return 0.0, (f"{detail_txt} (계수 {factor:.2f} < 최소 {PORTFOLIO_MIN_FACTOR:.2f} — "
                     f"너무 작아 비용만 나가므로 스킵)")
    return factor, detail_txt


def _apply_portfolio_factor_qty(qty: int, ref_price: float,
                                side: str | None, symbol: str | None) -> int:
    """산출된 주식 수량에 포트폴리오 상관 계수를 곱한다.

    symbol 이 없으면(옛 호출부) 축소하지 않는다 — 그룹을 알 수 없으면 판단할 수 없고,
    모르는 채로 줄이는 것보다 현행 유지가 안전하다.
    """
    # 🟥 [FIX-P1b/D8] 0은 '취소'라는 뜻이다. max(1,...)로 1주를 만들면 안 된다.
    if qty <= 0:
        return 0
    if not symbol:
        return int(qty)
    try:
        intended_notional = float(qty) * float(ref_price)
    except (TypeError, ValueError):
        return max(1, int(qty))
    factor, why = portfolio_size_factor(symbol, side or "BUY", intended_notional)
    if factor >= 0.999:
        print(f"[상관사이징] {symbol} 축소 없음 — {why}")
        return max(1, int(qty))
    if factor <= 0:
        print(f"🚫 [상관사이징] {symbol} 진입 취소 — {why}")
        return 0
    new_qty = int(qty * factor)
    if new_qty < 1:
        print(f"🚫 [상관사이징] {symbol} 축소 결과 1주 미만 → 취소 — {why}")
        return 0
    print(f"📉 [상관사이징] {symbol} {qty}주 → {new_qty}주 — {why}")
    return new_qty


def get_tiered_qty(price: float) -> int:
    """
    가격대별 고정 수량표 (목표: 한 거래당 약 $3,000 이하)
    $1000 이상   : 2주
    $500~999     : 4주
    $300~499     : 7주
    $200~299     : 10주
    $100~199     : 15주
    $50~99       : 30주
    $50 미만      : 60주
    """
    if price >= 1000:
        return 2
    elif price >= 500:
        return 4
    elif price >= 300:
        return 7
    elif price >= 200:
        return 10
    elif price >= 100:
        return 15
    elif price >= 50:
        return 30
    else:
        return 60


def calc_alpaca_qty(ref_price: float, sl: float, notional_usd: float,
                    side: str | None = None, symbol: str | None = None) -> int:
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
        # 🟥 [FIX-E6] tiered 모드만 notional 캡을 건너뛰고 조기 return 하고 있었다.
        #    그래서 $5,000짜리 주식 2주 = $10,000로 ALPACA_MAX_NOTIONAL_USD(5,000)를
        #    두 배 초과하는 노출이 나갈 수 있었다. 다른 모드와 동일하게 캡을 적용한다.
        capped = max(1, min(qty, max_qty_by_notional))
        if capped != qty:
            print(f"[Alpaca][tiered-sizing] price={ref_price}, 표준수량={qty}주 → "
                  f"notional 캡(${ALPACA_MAX_NOTIONAL_USD:g}) 적용 {capped}주로 축소")
        else:
            print(f"[Alpaca][tiered-sizing] price={ref_price}, qty={qty}")
        return _apply_portfolio_factor_qty(capped, ref_price, side, symbol)

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
            return _apply_portfolio_factor_qty(
                max(1, min(qty_by_risk, max_qty_by_notional)), ref_price, side, symbol)

        print("[Alpaca][risk-sizing] equity 조회 실패 또는 stop_distance=0 → 고정금액 방식으로 폴백")

    # fixed 모드 또는 risk 계산 실패시 폴백
    qty_by_fixed = int(notional_usd // ref_price)
    return _apply_portfolio_factor_qty(
        max(1, min(qty_by_fixed, max_qty_by_notional)), ref_price, side, symbol)


def calc_fx_units(pair: str, price, sl, direction: str) -> int:
    """
    🟥 [FIX-E7] FX 주문 수량(units) 산출.

    기존: `units = 100000 if BUY else -100000` — 계좌 크기·변동성과 무관한 고정 1랩.
          USD/JPY 1랩은 1pip ≈ $6.7이라, SL 20pip이면 한 번에 $134가 왔다 갔다 한다.
          데모 계좌 150건에서 스왑만 -$1,094가 쌓인 것도 이 크기 때문이다.
    수정: 계좌 잔고 × FX_RISK_PCT 를 SL 거리로 나눠 units를 역산한다.
          잔고 조회가 안 되면 FX_FALLBACK_UNITS(기본 10,000 = 0.1랩)로 보수적 폴백.
          FX_UNITS_FIXED에 값을 넣으면 그 값으로 고정(옛 동작 복원용).
    """
    sign = 1 if direction == "BUY" else -1

    fixed = os.getenv("FX_UNITS_FIXED", "").strip()
    if fixed:
        try:
            return sign * abs(int(float(fixed)))
        except ValueError:
            pass

    fallback = int(os.getenv("FX_FALLBACK_UNITS", "10000"))
    try:
        p = float(price)
        s = float(sl)
    except (TypeError, ValueError):
        print(f"[FX sizing] {pair} price/sl 불명 → 폴백 {fallback} units")
        return sign * fallback

    stop_dist = abs(p - s)
    if stop_dist <= 0:
        print(f"[FX sizing] {pair} SL 거리 0 → 폴백 {fallback} units")
        return sign * fallback

    balance = get_oanda_account_balance()
    if not balance or balance <= 0:
        print(f"[FX sizing] {pair} 잔고 조회 실패 → 폴백 {fallback} units")
        return sign * fallback

    risk_pct = float(os.getenv("FX_RISK_PCT", "0.5"))
    risk_amount = balance * (risk_pct / 100.0)

    # units당 손실 = SL 거리(가격 단위) × (계정통화 환산계수)
    # 계정통화가 USD이고 XXX_USD 페어면 손실 = stop_dist × units.
    # USD_JPY처럼 USD가 base면 손실(USD) = stop_dist × units / price.
    # 🟥 [FIX-E7b] 'EUR/USD'처럼 슬래시로 오는 경우 split("_")가 통째로 잡혀
    #    quote 판정이 틀린다. 구분자를 먼저 정규화한다.
    _pair_norm = (pair or "").upper().replace("/", "_").replace("-", "_")
    parts = [x for x in _pair_norm.split("_") if x]
    base = parts[0] if parts else ""
    quote = parts[-1] if len(parts) > 1 else ""
    if quote == "USD":
        # XXX_USD: 손익이 곧 USD (예: EUR_USD)
        loss_per_unit = stop_dist
    elif base == "USD":
        # USD_XXX: 손익(USD) = 거리 / 현재가 (예: USD_JPY)
        loss_per_unit = stop_dist / p if p else stop_dist
    else:
        # 🟥 크로스 페어(EUR_GBP 등)는 정확한 USD 환산에 제3의 환율이 필요하다.
        #    여기서 근사하면 리스크%가 틀어지므로, 안전하게 폴백 수량을 쓴다.
        print(f"[FX sizing] {pair} 크로스 페어(USD 미포함) → 정확한 환산 불가, 폴백 {fallback} units")
        return sign * fallback

    if loss_per_unit <= 0:
        return sign * fallback

    units = int(risk_amount / loss_per_unit)
    max_units = int(os.getenv("FX_MAX_UNITS", "100000"))
    units = max(1000, min(units, max_units))   # 최소 1,000 / 최대 FX_MAX_UNITS
    print(f"[FX sizing] {pair} 잔고={balance:.2f}, 리스크{risk_pct}%=${risk_amount:.2f}, "
          f"SL거리={stop_dist:.5f}, units={units} (기존 고정값 100000 → 리스크 기반)")

    # 🟥 [FIX-P1] 포트폴리오 상관 축소.
    #   EUR_USD 롱을 이미 들고 있는데 GBP_USD 롱이 들어오면 둘 다 'USD 숏'이라
    #   분산이 아니라 USD 한 방향에 두 배를 거는 것이다. 그만큼 줄인다.
    #   반대로 USD_JPY 롱(=USD 롱)이 들어오면 상쇄되므로 줄이지 않는다.
    _base, _quote = _split_instrument(pair)
    if _quote == "USD":
        _notional_usd = units * p          # EUR_USD, XAU_USD : units×가격
    elif _base == "USD":
        _notional_usd = float(units)       # USD_JPY : units 가 이미 USD
    else:
        _notional_usd = units * p
    _factor, _why = portfolio_size_factor(pair, direction, _notional_usd)
    if _factor <= 0:
        print(f"🚫 [상관사이징] {pair} 진입 취소 — {_why}")
        return 0
    if _factor < 0.999:
        # 🟥 [FIX-P1b/D7] max(1000, ...) 로 바닥을 깔면 축소 목표를 넘겨버린다.
        #   예: units 1200, 계수 0.35 → 목표 420 인데 1000 이 나가 2.4배 초과.
        #   최소 주문 단위에 못 미치면 늘리지 말고 그냥 건너뛴다.
        _new = int(units * _factor)
        if _new < 1000:
            print(f"🚫 [상관사이징] {pair} 축소 목표 {_new} units < 최소 1,000 → 진입 취소 — {_why}")
            return 0
        print(f"📉 [상관사이징] {pair} {units} → {_new} units — {_why}")
        units = _new
    else:
        print(f"[상관사이징] {pair} 축소 없음 — {_why}")

    return sign * units


def get_oanda_account_balance():
    """OANDA 계좌 잔고 조회. 실패 시 None. (🟥 [FIX-E7] FX 리스크 사이징용)"""
    if not (OANDA_API_KEY and ACCOUNT_ID):
        return None
    try:
        r = requests.get(
            f"{OANDA_BASE_URL}/v3/accounts/{ACCOUNT_ID}/summary",
            headers={"Authorization": f"Bearer {OANDA_API_KEY}"},
            timeout=10,
        )
        r.raise_for_status()
        return float(r.json()["account"]["balance"])
    except Exception as e:
        print(f"[OANDA] 잔고 조회 실패: {e}")
        return None


def get_alpaca_fill_status(symbol, after_iso):
    """
    Alpaca 주문 내역에서 해당 종목의 entry(시장가) 주문이 실제로 체결됐는지 확인.
    return: (filled: bool, filled_avg_price: float|None, filled_at: str|None, filled_qty: float|None)
    못 찾으면 (False, None, None, None) — 보수적으로 "아직 체결 안 됨"으로 취급.
    """
    url = f"{ALPACA_TRADE_BASE_URL}/v2/orders"
    params = {"symbols": symbol, "status": "all", "after": after_iso, "limit": 50, "direction": "asc"}
    try:
        r = requests.get(url, headers=ALPACA_HEADERS, params=params, timeout=10)
        r.raise_for_status()
        orders = r.json()
        for o in orders:
            # bracket의 진입(market) 주문만 본다. (TP/SL은 limit/stop이라 order_class로도 구분 가능)
            if o.get("type") == "market" and o.get("symbol") == symbol:
                status = o.get("status")
                if status == "filled":
                    return (
                        True,
                        float(o.get("filled_avg_price") or 0) or None,
                        o.get("filled_at"),
                        float(o.get("filled_qty") or 0) or None,
                    )
                else:
                    return False, None, None, None
        return False, None, None, None
    except Exception as e:
        print(f"❗ [Alpaca] {symbol} 주문 체결 상태 조회 실패: {e}")
        return False, None, None, None


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


def place_order_alpaca(symbol, side, notional_usd, ref_price, tp, sl, digits=2, atr=None):
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
        # 🟥 [FIX-COST1] 퍼센트가 아니라 ATR 대비로 본다.
        #   같은 0.02% 이동이라도 ATR 이 큰 종목엔 무시할 수준이고
        #   ATR 이 작은 종목(TLT·EWJ)엔 손익을 좌우한다.
        # ⚠️ 개장 직후 전략(G3)은 제외한다. 09:31 의 1분봉 ATR 은 장전 봉을 재서
        #    비정상적으로 좁고, 그 값으로 비율을 내면 정상 주문까지 전부 막힌다.
        #    그 전략은 애초에 ATR 을 안 쓰려고 구조적 손절을 쓰는 것이다.
        # 🟥 [FIX-COST4] 신호가 → 주문 시점 가격의 이동폭만 본다. ATR 은 쓰지 않는다.
        #   G3 처럼 구조적 손절을 쓰는 전략은 면제 (자체 손절이 이미 구조에서 나온다).
        _cost_exempt = _STRUCTURAL_SLTP.get()
        try:
            _drift_pct = abs(float(delta)) / float(ref_price) * 100.0 if float(ref_price) else 0.0
        except Exception:
            _drift_pct = 0.0
        # 차단 여부와 무관하게 항상 남긴다 — 나중에 분포를 보고 한도를 근거 있게 정하기 위해.
        print(f"📏 [체결지연] {symbol} 신호가 {ref_price} → 주문시점 {fresh_price} "
              f"({delta:+.4f}, {_drift_pct:.3f}%)")
        if COST_DRIFT_GATE_ENABLED and not _cost_exempt and _drift_pct > COST_MAX_DRIFT_PCT:
            print(f"⛔ [비용차단] {symbol} 이동 {_drift_pct:.3f}% > 한도 {COST_MAX_DRIFT_PCT}% "
                  f"→ 신호가 낡았다고 보고 주문 스킵")
            return {
                "status": "skipped",
                "reason": f"cost_drift_{_drift_pct:.3f}pct_exceeds_{COST_MAX_DRIFT_PCT}pct",
                "ref_price": ref_price, "fresh_price": fresh_price,
                "atr": float(atr) if atr else None,
            }

        if gap_pct > ALPACA_MAX_PRICE_GAP_PCT:
            print(f"⛔ [Alpaca] {symbol} 가격 갱신 폭({gap_pct:.2f}%)이 한도({ALPACA_MAX_PRICE_GAP_PCT}%) 초과 "
                  f"(신호가={ref_price} → 실시간가={fresh_price}) → 주문 스킵 (신호 신뢰 불가)")
            return {
                "status": "skipped",
                "reason": f"price_gap_{gap_pct:.2f}pct_exceeds_{ALPACA_MAX_PRICE_GAP_PCT}pct",
                "ref_price": ref_price,
                "fresh_price": fresh_price,
            }

        # 🟥 [FIX-SLTP1] 구조적 손절(첫 1분봉 저가 등)은 '거리'가 아니라 '가격 레벨'이다.
        #   delta 만큼 밀면 손절이 그 캔들 저가 '위' 로 올라가 의미가 사라진다.
        #   ATR 손절은 거리 개념이라 미는 게 맞다 — 그래서 종류에 따라 다르게 처리한다.
        structural_sltp = _STRUCTURAL_SLTP.get()
        if delta and structural_sltp:
            print(f"[Alpaca] {symbol} 가격 갱신 Δ{delta:+.4f} — 구조적 TP/SL 이라 "
                  f"이동하지 않고 원래 레벨 유지 (TP={tp}, SL={sl})")
            delta = 0.0
        if delta:
            print(f"[Alpaca] {symbol} 가격 갱신: 신호가={ref_price} → 실시간가={fresh_price} "
                  f"(Δ{delta:+.4f}, {gap_pct:.2f}%) — TP/SL을 동일 거리만큼 이동")
            tp = tp + delta
            sl = sl + delta
            ref_price = fresh_price

        # 🟦 2번 수정: TP까지 남은 거리가 너무 짧으면 SKIP
        #    - 알림→서버→주문 시차 동안 가격이 올라서 이미 TP 근처까지 와버린 경우,
        #      시장가로 들어가면 TP에 이미 닿았거나 거의 없어서 "TP_HIT인데 손익 마이너스" 케이스가 생김
        #    - ATR의 30% 미만 거리가 남았으면 진입 의미 없음 → 스킵
        try:
            _atr_now = float(atr) if atr else None
            if _atr_now and _atr_now > 0:
                _tp_remaining = abs(tp - ref_price)
                _min_tp_dist = _atr_now * 0.3
                if _tp_remaining < _min_tp_dist:
                    print(f"⛔ [Alpaca] {symbol} TP까지 남은 거리({_tp_remaining:.4f})가 "
                          f"ATR×0.3({_min_tp_dist:.4f}) 미만 → 이미 TP 근처 진입 의미 없음 → 스킵")
                    return {
                        "status": "skipped",
                        "reason": f"tp_too_close: remaining={_tp_remaining:.4f} < ATR×0.3={_min_tp_dist:.4f}",
                        "ref_price": ref_price,
                        "fresh_price": fresh_price,
                    }
        except Exception as e:
            print(f"[WARN] TP 근접 체크 실패(무시): {e}")

    # 🟥 [FIX-P1] side/symbol 을 넘겨야 상관 사이징이 그룹과 방향을 판단할 수 있다.
    qty = calc_alpaca_qty(ref_price, sl, notional_usd, side=side, symbol=symbol)

    # 🟥 [FIX-P1] 상관 한도 소진으로 수량이 0이 되면 주문을 내지 않는다.
    if qty <= 0:
        return {
            "status": "skipped",
            "reason": "portfolio_correlation_limit",
            "symbol": symbol,
            "ref_price": ref_price,
        }

    url = f"{ALPACA_TRADE_BASE_URL}/v2/orders"
    headers = {**ALPACA_HEADERS, "Content-Type": "application/json"}

    final_tp_rounded = round(tp, digits)
    final_sl_rounded = round(sl, digits)

    data = {
        "symbol": symbol,
        "qty": str(qty),
        "side": "buy" if side == "BUY" else "sell",
        "type": "market",
        # 🟦 day로 하면 TP/SL(자식 주문)도 같은 day로 적용돼서, 장마감까지 둘 다 안 닿으면
        #    보호 주문 자체가 사라지고 포지션이 무방비로 밤새 노출된다(Alpaca 공식 동작).
        #    GTC로 바꿔서, 당일에 못 닿아도 다음 거래일까지 TP/SL 보호가 계속 유지되게 한다.
        "time_in_force": "gtc",
        "order_class": "bracket",
        "take_profit": {"limit_price": str(final_tp_rounded)},
        "stop_loss": {"stop_price": str(final_sl_rounded)},
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=15)
        try:
            j = response.json()
        except Exception:
            j = {"raw_text": response.text}

        print(f"[Alpaca] status_code={response.status_code}")
        print(f"[Alpaca] body={j}")

        # 🟦 실제 주문에 쓰인 최종 가격들(실시간가로 보정된 값)을 항상 같이 반환.
        #    호출부에서 이 값으로 구글시트의 price/tp/sl을 다시 보정해서, 시트와 Alpaca가 항상 일치하게 한다.
        if 200 <= response.status_code < 300:
            # 🟥 [FIX-A3] 진입 시각을 메모리에 기록해둔다.
            #    기존 close_stale_positions()는 Alpaca 주문내역을 최근 20건만 뒤져서
            #    진입시각을 찾았는데, 같은 종목에 알림이 여러 번 오면 브래킷 자식주문들이
            #    그 20칸을 다 잡아먹어 진입 주문을 못 찾고 "이번엔 스킵"으로 빠졌다.
            #    → 90분 시간청산이 사실상 한 번도 실행되지 않은 직접 원인.
            _record_position_entry(symbol, "long" if side == "BUY" else "short")
            return {
                "status": "order_placed",
                "status_code": response.status_code,
                "raw": j,
                "qty": qty,
                "final_price": ref_price,
                "final_tp": final_tp_rounded,
                "final_sl": final_sl_rounded,
            }
        else:
            return {
                "status": "error",
                "status_code": response.status_code,
                "raw": j,
                "qty": qty,
                "final_price": ref_price,
                "final_tp": final_tp_rounded,
                "final_sl": final_sl_rounded,
            }

    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": str(e)}


def place_order(pair, units, tp, sl, digits, price=None, atr=None):
    # 🟦 주식이면 Alpaca Bracket Order로 분기
    if is_stock_pair(pair):
        side = "BUY" if units > 0 else "SELL"
        ref_price = price if price is not None else (_last_price_cache.get(pair) or tp)
        _atr_val = None
        try:
            if atr is not None:
                _atr_val = float(atr.iloc[-1]) if hasattr(atr, "iloc") else float(atr)
        except Exception:
            pass
        return place_order_alpaca(
            pair, side, ALPACA_FIXED_NOTIONAL_USD, ref_price, tp, sl,
            digits=price_round_digits(pair), atr=_atr_val
        )

    url = f"{OANDA_BASE_URL}/v3/accounts/{ACCOUNT_ID}/orders"   # 🟥 [FIX-E2]
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
    wait_confidence = None

    try:
        data = extract_json_block(text)
        print(f"[TRACE] Extracted JSON block: {data}")
        if isinstance(data, dict):  # ✅ dict인지 확인
            final_decision = str(data.get("decision", "WAIT")).upper()
            tp = safe_float(data.get("tp"))
            sl = safe_float(data.get("sl"))
            wait_confidence = safe_float(data.get("wait_confidence"))
            print(f"[DEBUG] JSON 추출 성공: decision={final_decision}, tp={tp}, sl={sl}, wait_confidence={wait_confidence}")
            print(f"[TRACE] 최종 판단 결과: final_decision={final_decision}, tp={tp}, sl={sl}")  # ← 추가
            # ⛔️ 파싱 실패 시 강제 초기화
            if final_decision not in ["BUY", "SELL"]:
                final_decision = "WAIT"
                tp = None
                sl = None
            
            return final_decision, tp, sl, wait_confidence

    except Exception as e:
        print(f"[WARN] JSON 파싱 실패: {e}, fallback 실행")
    
        # fallback 조건: 기존 판단이 없을 때만 덮어씀
        if final_decision != "WAIT" and (tp is not None and sl is not None):
            print("[INFO] fallback 진입했지만 기존 결정 BUY/SELL 유지함")
            return final_decision, tp, sl, wait_confidence
        else:
            print("[INFO] fallback 조건 충족 → WAIT 처리")
            final_decision = "WAIT"
            tp = None
            sl = None
            return final_decision, tp, sl, wait_confidence


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


    return final_decision, tp, sl, wait_confidence
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

    # 🟥 [FIX-E10] RR 하한을 하드코딩 1.8 → FX_MIN_RR_RATIO(env, 기본 1.8)로.
    #    기존엔 FX_MIN_RR_RATIO 상수를 선언만 하고 아무 데서도 안 썼다.
    if is_buy and (entry - sl) > 0:
        desired_tp = entry + FX_MIN_RR_RATIO * (entry - sl)
        tp = max(tp, desired_tp)
    if is_sell and (sl - entry) > 0:
        desired_tp = entry - FX_MIN_RR_RATIO * (sl - entry)
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
                    f"(3-3) ⚠️ [WAIT 선택 시 추가 규칙 — 둘 다 만족해야만 WAIT 가능]\n"
                    f"WAIT은 함부로 선택하면 안 된다. 아래 두 조건을 **모두** 만족해야만 WAIT을 선택할 수 있다:\n"
                    f"  1. 위 '✅ WAIT 근거' 중 최소 하나를 reason에 구체적으로(어떤 지표가 어떻게 꺾였는지) 명시해야 한다.\n"
                    f"     ('과매수라서', '저항 근접이라서' 같은 금지된 이유만 댄 WAIT은 무효다.)\n"
                    f"  2. 이 신호가 실패할 것이라는 확신도(wait_confidence, 0~100 정수)가 **80 이상**이어야 한다.\n"
                    f"     80 미만이면 WAIT을 선택할 수 없다 — 원래 알림 방향(BUY/SELL)을 그대로 확정하라.\n"
                    f"  위 둘 중 하나라도 못 만족하면 절대 WAIT을 출력하지 말고, 원래 신호 방향으로 decision을 내라.\n"
                    f"  JSON에 wait_confidence 필드를 추가하라 (WAIT이 아니면 0으로 채워라).\n\n"
                    if is_stock_pair(pair) else ""
                )
                +
                (
                    f"(3-FX) ⚠️ [USD/JPY 전용 판단 규칙 — 반드시 지켜라]\n"
                    f"USD/JPY는 M30 기준 추세 추종 전략이다. WAIT을 선택하면 거래 기회 자체가 사라지므로, "
                    f"WAIT 기준을 주식보다 훨씬 엄격하게 적용한다.\n"
                    f"  ❌ 아래 이유만으로는 절대 WAIT을 고르지 마라:\n"
                    f"     - RSI 70 이상 '과매수' — FX 추세 추종에서 RSI 70~85는 강한 추세의 증거다\n"
                    f"     - Stoch RSI 과열 — 주가 아닌 환율에서는 단기 과열이 바로 반전으로 이어지지 않는다\n"
                    f"     - NEUTRAL 추세 — 지표 지연으로 인해 방향 전환 초기에 NEUTRAL이 뜨는 게 정상이다\n"
                    f"     - 저항선/지지선 근접 — 돌파하면 강한 모멘텀이 생기므로 오히려 진입 근거다\n"
                    f"     - 박스권 상단/하단 — 박스권을 뚫고 가는 것이 이 전략의 진입 시그널이다\n"
                    f"  ✅ WAIT이 허용되는 유일한 조건 (아래 중 최소 2개가 동시에 충족될 때만):\n"
                    f"     - MACD가 시그널선을 방금 새로 하향 돌파했으며 RSI도 동반 하락 중인 경우\n"
                    f"     - 직전 봉이 신호 방향과 반대되는 강한 장대음봉/장대양봉인 경우\n"
                    f"     - 중앙은행(BOJ/FED) 긴급 개입 뉴스가 방금 나온 경우\n"
                    f"  조건이 충족되지 않으면 무조건 BUY/SELL로 거래를 진행하라.\n"
                    f"  wait_confidence는 95 이상일 때만 WAIT을 허용하며, 그 미만이면 BUY/SELL을 확정하라.\n\n"
                    if (not is_stock_pair(pair) and "JPY" in pair) else ""
                )
                +
                "- '🟢 최근 N분 내 뉴스 없음'이면 뉴스 요인은 무시해도 된다.\n"
                "- '⚠️ ... 뉴스 직후(...) — 뉴스 주도 변동 가능성'이면, 지금 이 돌파/움직임이 순수 기술적 돌파가 아니라 "
                "특정 뉴스(헤드라인이 같이 제공됨)에 의해 촉발된 것일 수 있다는 뜻이다. 이 경우:\n"
                "  · 뉴스가 진짜 호재/펀더멘털 변화라면 돌파에 더 신뢰를 줄 수 있다.\n"
                "  · 반대로 1회성 헤드라인 스파이크(예: 단순 소문, 루머, 과장된 헤드라인)로 보이면 "
                "되돌림(reversal) 위험이 더 크다고 보고 신중해야 한다.\n"
                "  · 리포트의 1️⃣ 전략 요약에서 뉴스 헤드라인 내용과 그게 이 신호에 어떤 영향을 주는지 반드시 한 줄 언급하라.\n"
                "- '🟡 ... 최근 N분 내 뉴스 M건'(뉴스가 있지만 막 나온 건 아님)이면, 그 뉴스가 이미 가격에 반영됐을 가능성이 높으니 "
                "참고만 하고 과도하게 비중을 두지 마라.\n\n"
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

                # 🟥 [FIX-ROLE1] 너의 역할을 분명히 한다.
                "\n"
                "════════ 너의 역할 (이게 제일 중요하다) ════════\n"
                "이 신호는 이미 검증된 매매 전략이 자기 규칙에 따라 낸 것이다.\n"
                "'진입 자리가 맞는지' 는 **전략이 이미 판단했다. 다시 판단하지 마라.**\n"
                "\n"
                "특히 아래는 감점 사유가 아니다 — 전략의 설계 그 자체다:\n"
                "  · 가격이 저항선/박스 상단에 가깝다  → 돌파 전략은 원래 거기서 산다\n"
                "  · 최근 고점 근처다                → 그게 돌파의 정의다\n"
                "  · RSI 가 중립이다                 → 전략이 이미 RSI 하한을 걸고 통과시켰다\n"
                "  · 추세가 NEUTRAL 이다             → 그것만으로 거부하지 마라\n"
                "이런 이유로 WAIT 을 내면 전략이 잡으라고 만든 자리를 전부 거부하게 된다.\n"
                "\n"
                "네가 봐야 할 것은 **전략이 볼 수 없는 것** 뿐이다:\n"
                "  ① 뉴스·이벤트 — 실적발표, 금리결정, 급등 원인이 일회성 뉴스인가\n"
                "  ② 상위 시간축과 정면 충돌 — H4/일봉이 강하게 반대 방향인가\n"
                "  ③ 진짜 극단 — RSI 80 이상 / 20 이하, Stoch 0.98 이상 같은 명백한 소진\n"
                "  ④ 변동성 이상 — ATR 이 평소 대비 비정상적으로 크거나(패닉) 작다(비용이 먹는다)\n"
                "  ⑤ 시간대 — FX 는 19~23시(ET) 유동성이 최악이라 스프레드가 벌어진다\n"
                "  ⑥ 캔들 소진 — 긴 윗꼬리 연속, 대량 거래 후 힘 빠짐 같은 그림\n"
                "\n"
                "WAIT 은 위 ①~⑥ 중 **구체적으로 짚을 수 있는 위험이 있을 때만** 내라.\n"
                "'확신이 부족하다', '모멘텀이 약하다' 같은 막연한 이유로는 WAIT 하지 마라.\n"
                "짚을 위험이 없으면 전략의 판단을 존중해서 그대로 BUY/SELL 을 내라.\n"
                "WAIT 을 낼 때는 reason 에 위 ①~⑥ 중 어느 것인지 번호를 반드시 밝혀라.\n"
                "══════════════════════════════════════════\n\n"

                "(5) 전략 리포트는 자유롭게 작성하되 반드시 아래 4단계 형식을 따르라:\n"
                "1️⃣ 이 신호가 왜 나왔나 (전략의 규칙 — 재판단 말고 요약만)\n"
                "2️⃣ 지금 시장 상황 (캔들 흐름·상위 시간축·변동성)\n"
                "3️⃣ 위 ①~⑥ 위험 점검 — 해당 없으면 '해당 없음' 이라고 명시\n"
                "4️⃣ 최종 판단 및 이유\n\n"

                "(6) 마지막에는 반드시 아래 JSON 의사결정 블록을 작성하라. 양식은 정확히 아래처럼!\n\n"
                "{\n"
                "  \"decision\": \"BUY\" | \"SELL\" | \"WAIT\",\n"
                "  \"tp\": <숫자>,       // 반드시 숫자(float). 따옴표 금지. 예: 1.1745\n"
                "  \"sl\": <숫자>,       // 반드시 숫자(float). 따옴표 금지.\n"
                "  \"wait_confidence\": <0~100 정수>,  // WAIT일 때만 의미 있음. WAIT이 아니면 0.\n"
                "  \"reason\": \"<핵심 이유 하나. WAIT 이면 위험 번호(①~⑥)를 반드시 포함>\"\n"
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
        "model": os.getenv("GPT_MODEL", "gpt-4o-2024-11-20"),
        "input": messages,
        "temperature": 0.3,
        # 🟥 [FIX-C4] 1000 → 1800.
        #    프롬프트가 4단계 리포트 + 마지막 줄 JSON을 요구하는데 1000토큰으로는
        #    긴 분석이 나올 때 JSON 블록이 잘린다. 잘리면 파싱 실패 → WAIT →
        #    (기존엔) 강제 환원으로 무검증 진입까지 이어졌다.
        "max_output_tokens": int(os.getenv("GPT_MAX_OUTPUT_TOKENS", "1800")),
    }
    need_tokens = _approx_tokens(messages)
    _preflight_gate(need_tokens)   # 요청 직전 선대기

    try:
        _bytes = len(json.dumps(payload, ensure_ascii=False))
    except Exception:
        _bytes = -1

    dbg("gpt.body", bytes=_bytes, max_tokens=body.get("max_output_tokens"))
    # 🟥 [FIX-E1b] 프롬프트 전문을 매번 stdout에 찍으면 로그가 비대해지고
    #    민감 정보가 남는다. 길이만 남긴다. (전문이 필요하면 GPT_DEBUG_BODY=true)
    if os.getenv("GPT_DEBUG_BODY", "false").strip().lower() == "true":
        print("🔍 FULL BODY DEBUG:", json.dumps(body, indent=2, ensure_ascii=False))
    else:
        print(f"🔍 GPT 요청 준비 완료 (model={body['model']}, prompt≈{need_tokens}tok, payload={_bytes}B)")

    # ============================================================
    # 🟥 [FIX-C2] 최소 스로틀 — 락을 잡은 채로 sleep 하지 않는다.
    # ------------------------------------------------------------
    #  기존 코드는 `with _gpt_lock:` 안에서 최대 12초를 sleep 했다.
    #  알림이 N건 동시에 오면 12×N초가 직렬로 쌓여서, 스캘핑 신호가
    #  체결 시점에는 이미 무의미해지는 상태였다(진입 슬리피지의 큰 원인).
    #  → 락 안에서는 "내가 언제 쏠지" 슬롯만 예약하고, 실제 대기는 락 밖에서 한다.
    #    이러면 대기가 병렬로 겹쳐서 총 지연이 12×N이 아니라 12초 수준으로 줄어든다.
    #  기본 간격도 12초 → 1.5초로 낮춘다(GPT_RPM 3000이면 12초는 과도한 보수 설정).
    # ============================================================
    _min_gap = float(os.getenv("GPT_MIN_INTERVAL_SEC", "1.5"))
    with _gpt_lock:
        # (global _gpt_last_ts 선언은 이 함수 상단에 이미 있다)
        _slot_at = max(_t.time(), _gpt_last_ts + _min_gap)
        _gpt_last_ts = _slot_at
    _wait = _slot_at - _t.time()
    if _wait > 0:
        dbg("gpt.throttle", wait=round(_wait, 2))
        _t.sleep(_wait)

    try:
        dbg("gpt.call")
        r = requests.post(
            OPENAI_URL,
            headers=OPENAI_HEADERS,
            json=body,
            timeout=int(os.getenv("GPT_TIMEOUT_SEC", "60")),
        )
        print("GPT STATUS:", r.status_code)
        # 🟥 [FIX-C3] 응답 헤더의 레이트리밋 정보를 실제로 저장한다.
        #    _save_rate_headers()는 정의만 되어 있고 호출하는 곳이 없어서
        #    _preflight_gate()가 항상 no-op이었고 429 백오프도 작동하지 않았다.
        try:
            _save_rate_headers(r.headers)
        except Exception as _e:
            dbg("gpt.rate_headers.fail", err=str(_e))
        if r.status_code == 429:
            # 429는 재시도해도 소용없으니 쿨다운을 걸고 즉시 실패로 반환
            # (global 선언은 이 함수 상단에 이미 있다)
            _retry_after = 30.0
            try:
                _retry_after = float(r.headers.get("retry-after") or 30.0)
            except Exception:
                pass
            _gpt_cooldown_until = _t.time() + _retry_after
            print(f"⛔ GPT 429 레이트리밋 → {_retry_after:.0f}초 쿨다운 설정")
            return f"GPT_ERROR: 429 rate limited, cooldown {_retry_after:.0f}s"
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
    
    # 🟥 [FIX-D3] 매번 재인증하던 것을 캐시된 헬퍼로 교체.
    sheet = _get_sheet()
    if sheet is None:
        print("❌ [기록] 시트 연결 실패 → 이번 행 기록 건너뜀")
        return None
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

        # 🟥 [FIX-D5] 이 두 칸(24·25열)은 evaluate_pending_outcomes()가 나중에
        #    quantity / total_pnl 숫자로 덮어쓰는 자리다. 그런데 여기서 신규 기록 시
        #    "신고점"/"신저점" 문자열을 넣고 있어서, 같은 열에 문자열과 숫자가 섞였다.
        #    그 결과 sync_score_bucket_analysis()의 float(row[24]) 캐스팅이 조용히
        #    실패하면서 SKIPPED 가상손익 집계가 통째로 누락됐다.
        #    → 신규 기록 시에는 빈칸으로 두고, 숫자 전용 열로 유지한다.
        #    (신고점/신저점 정보는 아래 reason/filtered_movement 쪽에 이미 남는다)
        "",                               # quantity (결과추적이 채움)
        "",                               # total_pnl (결과추적이 채움)
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

    # ============================================================
    # 🟥 [FIX-D3] 방금 추가한 행 번호를 append 응답에서 직접 읽는다.
    # ------------------------------------------------------------
    #  기존: append_row() 후 len(sheet.get_all_values())로 행 번호를 "추정"했다.
    #  웹훅 2건이 동시에 들어오면 두 스레드가 같은 길이를 읽어 같은 행 번호를 받고,
    #  이후 correct_sheet_trade_prices()가 다른 종목 행의 price/tp/sl을 덮어썼다.
    #  → Google Sheets API가 돌려주는 updatedRange("'시트명'!A123:AK123")를 파싱해
    #    실제로 내가 쓴 행 번호를 확정한다. 추정이 아니라 사실이다.
    # ============================================================
    try:
        # 🟥 [FIX-D3b] value_input_option은 기존 기본값(RAW)을 유지한다.
        #    USER_ENTERED로 바꾸면 Sheets가 문자열을 파싱해서 A열 timestamp가
        #    날짜형으로 강제 변환되고, 이후 datetime.fromisoformat(row[0])가
        #    전부 실패하며 결과추적/집계가 조용히 행을 건너뛴다.
        _sheets_write_throttle()          # 🟥 [FIX-F1]
        resp = sheet.append_row(clean_row, insert_data_option="INSERT_ROWS", table_range="A1")
        row_idx = None
        try:
            updated_range = (resp or {}).get("updates", {}).get("updatedRange", "")
            # 예: "'1. 알람 스프레드'!A1049:AK1049"
            m = _re.search(r"![A-Z]+(\d+)", updated_range or "")
            if m:
                row_idx = int(m.group(1))
        except Exception as e:
            print(f"⚠️ [기록] updatedRange 파싱 실패({e}) → 폴백 사용")
        if row_idx is None:
            # 폴백: 예전 방식(추정). 동시성 위험이 있으므로 경고를 남긴다.
            try:
                row_idx = len(sheet.get_all_values())
                print(f"⚠️ [기록] 행 번호를 길이로 추정함(row={row_idx}) — 동시 요청 시 부정확할 수 있음")
            except Exception:
                row_idx = None
        print(f"✅ [기록] row {row_idx} 저장 완료")
        return row_idx
    except Exception as e:
        print("❌ Google Sheet append_row 실패:", e)
        print("🧨 clean_row 전체 내용:\n", clean_row)
        return None


# ============================================================
# 🟦 결과 자동 추적 (백테스트 보조) — 1시간마다 미정 행들을 채워준다
# ============================================================

def _normalize_decision(decision_text: str) -> tuple[str, bool]:
    """
    🟥 [FIX-D8] 시트 decision 컬럼(E)의 값을 (방향, 실제체결여부)로 정규화한다.

    FIX-D1으로 이 컬럼이 "BUY" 대신 "EXECUTED_BUY" / "SKIPPED_price_gap_2.1pct" /
    "BLOCKED_GPT_TIMEOUT" 같은 값을 갖게 됐다. 그런데 evaluate_pending_outcomes()는
    여전히 `decision_text in ("BUY","SELL")`로 비교하고 있어서, 수정 후에는
    실제 체결된 거래가 전부 "미실행"으로 취급되는 심각한 회귀가 생긴다.
    (체결가 보정·수량 반영·NOT_FILLED 판정이 통째로 죽는다)
    → 옛 값과 새 값을 모두 이해하는 정규화 함수를 두고, 판정부는 이것만 쓴다.

    반환: (방향 "BUY"|"SELL"|"", 실제_주문전송_여부)
    """
    t = (decision_text or "").strip().upper()
    if not t:
        return "", False
    # 구버전 호환: 그냥 BUY / SELL
    if t in ("BUY", "SELL"):
        return t, True
    if t.startswith("EXECUTED_"):
        d = t[len("EXECUTED_"):]
        return (d if d in ("BUY", "SELL") else ""), True
    # SKIPPED_* / BLOCKED_* / ORDER_FAILED_* / WAIT → 주문 안 나감
    for direction in ("BUY", "SELL"):
        if direction in t:
            return direction, False
    return "", False


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


EOD_EVAL_BUFFER_MIN = int(os.getenv("EOD_EVAL_BUFFER_MIN", "15"))


def _eval_window_minutes(pair: str, entry_time, default_minutes: int) -> float:
    """
    🟥 [FIX-PN1] 가상 평가를 '언제까지' 볼지.

    기존엔 무조건 240분이었다. 그런데 봇은 주식을 15:50 전량청산까지 들고 간다.
    10:00 진입이면 실제 보유는 350분인데 240분만 보고 'TIMEOUT' 으로 끊어버렸다.
    그래서 시트의 total_pnl 과 실제 체결 손익의 **부호가 반대로** 나오는 일이 생겼다.
      · CEG  시트 +5.25  ↔ 실제 -26.80  (실제 SL 은 293분에 맞음)
      · GEV  시트 -2.60  ↔ 실제 +48.84  (실제 청산은 351분)
    → 주식은 그날 장마감 청산 시각까지로 창을 늘린다.
    """
    if not is_stock_pair(pair) or entry_time is None:
        return float(default_minutes)
    if not (STOCK_EOD_FLATTEN_ENABLED and STOCK_EOD_FLATTEN_HHMM > 0):
        return float(default_minutes)
    try:
        et = entry_time.astimezone(ZoneInfo("America/New_York")) if entry_time.tzinfo else entry_time
        eod = et.replace(hour=STOCK_EOD_FLATTEN_HHMM // 100,
                         minute=STOCK_EOD_FLATTEN_HHMM % 100, second=0, microsecond=0)
        # ⚠️ 청산 루프는 5분마다 돌고 체결도 몇 초 늦는다. 실제 청산은 15:50 이 아니라
        #    15:51~15:55 에 찍힌다. 창을 15:50 에 딱 맞추면 그 1~2분 때문에 또 놓친다.
        #    (실측: GEV 351.4분, PLTR 321.5분 — 둘 다 '정확히 15:50' 창 밖이었다)
        eod = eod + timedelta(minutes=EOD_EVAL_BUFFER_MIN)
        mins = (eod - et).total_seconds() / 60.0
        # ⚠️ max(기본값, mins) 로 하면 안 된다. 14:30 진입은 실제 보유가 80분인데
        #    240분 창이 살아남아 장마감 3시간 뒤(19:00)까지 평가하게 된다.
        #    주식은 무조건 '그날 청산 시각까지' 다. 너무 짧을 때만 최소치를 둔다.
        return float(default_minutes) if mins <= 0 else max(20.0, mins)
    except Exception:
        return float(default_minutes)


_ccy_rate_cache: dict = {}          # ccy -> (계수, 저장시각)  1시간 캐시


def _quote_ccy_to_usd(pair: str, price: float):
    """
    🟥 [FIX-PN2] FX 손익을 '호가통화' 에서 USD 로 환산하는 계수.

    XXX_YYY 의 가격은 'XXX 1단위 = YYY 얼마' 다. 그래서
        손익(YYY) = 가격차 × units
    이고, YYY 가 USD 가 아니면 그대로 USD 로 쓰면 안 된다.
    실제로 USD_JPY 손익이 시트에 -11,000 으로 찍혔는데 OANDA 실현손익은 -76.23 USD 였다.
    (-11,000 JPY ÷ 159 ≈ -69 USD — 자릿수가 통째로 틀렸던 것)

    환산 계수를 못 구하면 1.0 이 아니라 None 을 돌려준다.
    1.0 을 쓰면 엔화 손익이 150배로 부풀려진 채 조용히 기록된다 — 그게 원래 버그였다.
    못 구하면 total_pnl 을 아예 안 쓰는 게 맞다.
    """
    try:
        s = str(pair).upper().strip()
        if "_" not in s and len(s) == 6:
            s = f"{s[:3]}_{s[3:]}"          # 구형 기록 'USDJPY' 도 처리
        parts = s.split("_")
        if len(parts) != 2 or not all(parts):
            return None
        base, quote = parts
        if quote == "USD":
            return 1.0                      # EUR_USD, AUD_USD … 이미 USD
        if base == "USD":
            return (1.0 / float(price)) if price and float(price) > 0 else None

        # 크로스(EUR_GBP 등) — quote 통화의 USD 환산율. 1시간 캐시.
        hit = _ccy_rate_cache.get(quote)
        if hit and (_t.time() - hit[1]) < 3600:
            return hit[0]
        for sym, inv in ((f"{quote}_USD", False), (f"USD_{quote}", True)):
            px = _oanda_price_quick(sym)
            if px and float(px) > 0:
                val = (1.0 / float(px)) if inv else float(px)
                _ccy_rate_cache[quote] = (val, _t.time())
                return val
        _ccy_rate_cache[quote] = (None, _t.time())     # 실패도 캐시 — 매 행마다 재시도 방지
        return None
    except Exception as e:
        print(f"⚠️ [환산] {pair} 실패: {e}")
        return None


def evaluate_pending_outcomes(max_window_minutes: int = 240, min_elapsed_minutes: int = 5):
    """
    구글시트에서 아직 결과가 안 채워진 행들을 찾아서,
    그 시점 이후 캔들을 다시 조회해 TP/SL 중 뭘 먼저 쳤는지 판정하고
    result / outcome_analysis 컬럼에 자동으로 채워넣는다.
    (1시간마다 백그라운드로 호출됨. 수동으로도 /run_outcome_tracker 로 트리거 가능)
    """
    try:
        sheet = _get_sheet()                      # 🟥 [FIX-F1] 캐시된 클라이언트 재사용
        if sheet is None:
            return {"checked": 0, "updated": 0, "error": "sheet_unavailable"}
        all_rows = sheet.get_all_values()
        # 🟦 기존 is_new_high/is_new_low 컬럼을 quantity/total pnl로 재사용 — 헤더 라벨도 같이 갱신
        try:
            header_row = all_rows[0] if all_rows else []
            _hdr = []
            if len(header_row) > 23 and header_row[23] != "quantity":
                _hdr.append((1, COL_QUANTITY, "quantity"))
            if len(header_row) > 24 and header_row[24] != "total_pnl":
                _hdr.append((1, COL_TOTAL_PNL, "total_pnl"))
            if _hdr:
                _flush_sheet_updates(sheet, _hdr, label="결과추적:헤더")   # 🟥 [FIX-F1]
        except Exception as e:
            print(f"⚠️ [결과추적] 헤더 라벨 갱신 실패(무시): {e}")
    except Exception as e:
        print(f"❌ [결과추적] 시트 읽기 실패: {e}")
        return {"checked": 0, "updated": 0, "error": str(e)}

    checked = 0
    updated = 0
    # 🟥 [FIX-F1] 셀 쓰기를 즉시 보내지 않고 여기에 모았다가 마지막에 한 번에 flush 한다.
    #    (행마다 update_cell 5회 → 429 Quota exceeded 로 배포가 실패했다)
    pending: list = []

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

        # 🟦 decision이 SKIPPED_BY_THRESHOLD이거나 "미실행"으로 표시된 행은
        #    실제 거래가 안 들어간 것이라 TP/SL 판정 대상이 아님 (NOT_FILLED 오탐 방지).
        #    WAIT은 "가상 TP/SL"로 평가하도록 허용(의도된 설계).
        # 🟥 [FIX-D8] 새 decision 포맷(EXECUTED_/SKIPPED_/BLOCKED_)을 정규화해서 판정한다.
        _dir_norm, _was_sent = _normalize_decision(decision_text)
        # 🟥 [FIX-D10] SKIPPED_BY_THRESHOLD 행의 가상평가를 허용한다.
        #    기존엔 여기서 continue 해버려서 result/total_pnl이 영영 안 채워졌고,
        #    그 결과 '스코어대별 성과분석' 탭의 SKIPPED 컬럼(threshold 조정 근거)이
        #    구조적으로 항상 빈칸이었다. NOT_FILLED 오탐 우려는 이제 _was_sent 판정으로
        #    따로 막히므로(FIX-D8), 여기서 걸러낼 이유가 없다.
        #    (TP/SL이 없는 행은 아래 float 변환에서 자연스럽게 스킵된다)

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
        # 🟥 [FIX-PN1] 이 행에 적용할 평가 창 (주식은 그날 15:50 전량청산 시각까지)
        _win = _eval_window_minutes(pair, entry_time, max_window_minutes)

        # 🟦 거래 수량 — FX는 고정 100,000 units, 주식은 Alpaca 실제 체결 수량을 그대로 사용.
        #    이게 없으면 PNL이 "1주(또는 1단위) 기준" 가격차이로만 계산돼서 실제 손익과 안 맞는다.
        # 🟥 [FIX-D9] FX 수량을 100,000으로 하드코딩하지 않는다.
        #    FIX-E7로 FX가 리스크 기반 가변 units가 됐기 때문에, 시트에 기록된
        #    실제 수량(24열)을 우선 사용하고, 없을 때만 보수적 폴백을 쓴다.
        trade_qty = None
        if not is_stock_pair(pair):
            try:
                _q = float(row[23]) if len(row) > 23 and str(row[23]).strip() else 0.0
            except (TypeError, ValueError):
                _q = 0.0
            trade_qty = _q if _q > 0 else float(os.getenv("FX_FALLBACK_UNITS", "10000"))
        is_not_filled = False  # NOT_FILLED 여부 추적용

        # 🟦 주식은 평가 전에 "진짜 체결됐는지" 먼저 확인한다.
        #    market 주문이 장마감 직후/체결 지연 등으로 아직 'accepted' 상태일 수 있는데,
        #    이때 캔들 가격만 보고 TP_HIT/SL_HIT을 매기면 실제로는 포지션이 없는데 가짜 결과가 찍힌다.
        if is_stock_pair(pair) and _was_sent and _dir_norm in ("BUY", "SELL"):   # 🟥 [FIX-D8]
            entry_time_iso = entry_time.astimezone(ZoneInfo("UTC")).strftime("%Y-%m-%dT%H:%M:%SZ") if entry_time.tzinfo else entry_time.strftime("%Y-%m-%dT%H:%M:%SZ")
            filled, filled_price, filled_at, filled_qty = get_alpaca_fill_status(pair, entry_time_iso)
            if not filled:
                if elapsed_minutes > _win:
                    # 🟦 체결 안 됐지만 TP/SL 값이 있으면 가상 평가를 계속 진행한다.
                    #    실제 거래가 없었다는 사실만 표시하고, 캔들 기반으로 "만약 들어갔다면"을 평가해서
                    #    나중에 "14시 차단이 잘한 건지, 쿨다운이 너무 빡빡한지" 분석할 수 있게 한다.
                    if price_f and tp_f and sl_f:
                        is_not_filled = True  # 이후 캔들 평가는 계속 진행, 결과 앞에 NOT_FILLED_ 붙임
                        print(f"📊 [결과추적] {pair} NOT_FILLED이지만 TP/SL 있음 → 가상 평가 진행")
                    else:
                        try:
                            pending.append((i, COL_RESULT, "NOT_FILLED"))                      # 🟥 [FIX-F1]
                            pending.append((i, COL_OUTCOME_ANALYSIS, "⚠️ 주문 미체결 + TP/SL 없음 → 평가 불가"))
                            updated += 1
                        except Exception:
                            pass
                        continue
                else:
                    print(f"⏳ [결과추적] {pair} 아직 주문 미체결(accepted/held 등) → 이번엔 스킵, 다음 시간에 재확인")
                    continue
            elif filled_price:
                # 실제 체결가가 시트에 기록된 price와 다르면, 더 정확한 체결가로 보정해서 판정
                price_f = filled_price
            if filled_qty:
                trade_qty = filled_qty

        # 🟦 결과 "판정"은 알림 자체의 타임프레임(15분 등)과 무관하게 1분봉으로 본다.
        #    15분봉 하나엔 시가/고가/저가/종가만 있어서, 그 15분 안에서 SL을 먼저 쳤는지
        #    TP를 먼저 쳤는지 순서를 구분할 수 없다(둘 다 한 봉 안에 있으면 어느 게 먼저인지 모름).
        #    1분봉으로 보면 그 순서를 거의 다 구분할 수 있다.
        gran = "M1"
        _gran_minutes = 1
        # 🟦 경과 시간을 확실히 덮을 만큼 동적으로 더 많이 가져오되,
        #    OANDA/Alpaca 쪽 1회 요청 한도(보통 5000개 안팎)를 넘기면 400 에러가 나므로 안전하게 캡.
        bars_needed = int(elapsed_minutes / _gran_minutes) + 20  # 여유 버퍼 20개
        bars_capped = min(bars_needed, 4500)
        candles = get_candles(pair, gran, max(50, bars_capped))
        if candles is None or candles.empty:
            # 캔들 자체를 못 가져온 경우 — 그래도 4시간 넘었으면 더 기다릴 의미 없으니 시간초과로 정리
            if elapsed_minutes > _win:
                was_executed = _was_sent   # 🟥 [FIX-D8]
                note = _generate_outcome_note("TIMEOUT_NO_HIT", reasons_text, decision_text, was_executed)
                try:
                    pending.append((i, COL_RESULT, "TIMEOUT_NO_HIT"))                          # 🟥 [FIX-F1]
                    pending.append((i, COL_OUTCOME_ANALYSIS, note + " (캔들 조회 실패로 판정 불가)"))
                    updated += 1
                except Exception:
                    pass
            continue

        try:
            candles = candles.copy()
            candles["time_dt"] = pd.to_datetime(candles["time"], utc=True)
            entry_time_utc = entry_time.astimezone(ZoneInfo("UTC")) if entry_time.tzinfo else entry_time
            after = candles[candles["time_dt"] >= entry_time_utc]

            # 🟥 [FIX-PN1b] 창은 '언제 확정할지' 뿐 아니라 '어디까지 볼지' 도 정해야 한다.
            #    안 그러면 장마감(15:50 청산) 뒤의 시간외 봉까지 훑어서,
            #    이미 청산된 포지션이 시간외에 SL 을 스쳤다고 SL_HIT 을 찍는다.
            #    (IEX 분봉은 20:00 까지 있다 — 얇고 스프레드가 넓어 헛발이 잘 난다)
            _cut_dt = entry_time_utc + timedelta(minutes=float(_win))
            after = after[after["time_dt"] <= _cut_dt]

            # 🟦 안전장치: 가져온 캔들의 "가장 이른" 시점이 진입 시점보다 늦으면
            #    (=진입 직후 구간이 통째로 누락된 것) 잘못된 판정(특히 거짓 TP_HIT)을 낼 수 있다.
            #    ⚠️ 방향 주의: earliest_fetched가 entry_time보다 "나중"일 때만 문제다.
            #    (이전 버전엔 부호가 반대로 들어가서, 오히려 충분히 덮인 정상 케이스를 스킵시키던 버그가 있었음)
            earliest_fetched = candles["time_dt"].min()
            gap_minutes = (earliest_fetched - entry_time_utc).total_seconds() / 60
            if gap_minutes > _gran_minutes * 2:
                if elapsed_minutes > _win and bars_needed > bars_capped:
                    # 너무 오래된 신호라 캡에 걸려 영영 못 덮는 경우 → 시간초과로 정리하고 끝
                    was_executed = _was_sent   # 🟥 [FIX-D8]
                    note = _generate_outcome_note("TIMEOUT_NO_HIT", reasons_text, decision_text, was_executed)
                    pending.append((i, COL_RESULT, "TIMEOUT_NO_HIT"))                          # 🟥 [FIX-F1]
                    pending.append((i, COL_OUTCOME_ANALYSIS, note + " (데이터가 너무 오래돼 정밀 판정 불가)"))
                    updated += 1
                else:
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
            if elapsed_minutes > _win:
                outcome = "TIMEOUT_NO_HIT"
            else:
                continue  # 아직 더 기다려야 함 (다음 시간에 재평가)

        was_executed = _was_sent   # 🟥 [FIX-D8]
        note = _generate_outcome_note(outcome, reasons_text, decision_text, was_executed)

        # 🟦 실제 손익(가격 기준, 1주/1단위 기준) 계산 — 'pnl' 컬럼은 기존 그대로 유지
        if outcome == "TP_HIT":
            exit_price = tp_f
        elif outcome == "SL_HIT":
            exit_price = sl_f
        else:  # TIMEOUT_NO_HIT — 마지막으로 본 가격을 기준으로 평가손익 추정
            exit_price = float(after.iloc[-1]["close"]) if not after.empty else price_f
        pnl_value = (exit_price - price_f) if signal_dir == "BUY" else (price_f - exit_price)

        # 🟦 버그 수정: PNL이 1주/1단위 기준 가격차이로만 계산돼서 실제 수량을 반영 못 하고 있었음.
        #    수량(trade_qty)을 곱한 "실제 총손익"을 따로 계산해서 보여준다.
        #    주식인데 체결 수량을 못 가져온 경우(드묾)는 가격대별 고정수량표로 추정.
        if trade_qty is None:
            # 🟥 [FIX-D9] FX 폴백도 100,000 하드코딩 대신 FX_FALLBACK_UNITS를 따른다.
            trade_qty = (get_tiered_qty(price_f) if is_stock_pair(pair)
                         else float(os.getenv("FX_FALLBACK_UNITS", "10000")))
        # 🟥 [FIX-PN2] FX 는 호가통화 손익 → USD 환산. 안 하면 JPY 쌍이 150배 부풀려진다.
        _fx_mult = 1.0 if is_stock_pair(pair) else _quote_ccy_to_usd(pair, price_f)
        if _fx_mult is None:
            total_pnl_value = ""            # 환율을 못 구했으면 틀린 숫자를 쓰느니 비워둔다
            print(f"⚠️ [결과추적] {pair} USD 환산 실패 → total_pnl 기록 생략")
        else:
            total_pnl_value = round(pnl_value * trade_qty * _fx_mult, 2)

        # 🟦 NOT_FILLED 가상 평가: 실제 거래(TP_HIT/SL_HIT)와 구분하기 위해 앞에 NOT_FILLED_ 붙임.
        #    이렇게 하면 나중에 "필터로 막은 거래들이 실제로 어떻게 됐을지" 별도로 집계해서
        #    14시 차단, 쿨다운 등 각 필터의 효과를 데이터로 검증할 수 있다.
        display_outcome = f"NOT_FILLED_{outcome}" if is_not_filled else outcome
        display_note = f"[가상평가-미체결] {note}" if is_not_filled else note

        try:
            # 🟥 [FIX-F1] 5회 즉시 쓰기 → 배치 누적
            pending.append((i, COL_RESULT, display_outcome))
            pending.append((i, COL_PNL, round(pnl_value, 5)))
            pending.append((i, COL_QUANTITY, trade_qty))
            pending.append((i, COL_TOTAL_PNL, total_pnl_value))
            pending.append((i, COL_OUTCOME_ANALYSIS, display_note))
            updated += 1
            print(f"✅ [결과추적] row {i} ({pair}, {signal_dir}) → {display_outcome} "
                  f"(1단위pnl={pnl_value:.5f}, 수량={trade_qty}, 총손익={total_pnl_value})")
        except Exception as e:
            print(f"❌ [결과추적] row {i} 시트 업데이트 실패: {e}")

    # 🟥 [FIX-D1b] 알람 탭에도 날짜 구분선을 긋는다 (거래내역 탭들과 동일).
    try:
        _ss = _get_spreadsheet()
        if _ss is not None and all_rows:
            _ncols = max(len(r) for r in all_rows[:50]) if all_rows else 1
            _draw_date_separators(_ss, sheet, all_rows, date_col=0,
                                  ncols=_ncols, tag="알람탭구분선")
    except Exception as e:
        print(f"⚠️ [알람탭 구분선] 실패(무시): {e}")

    # 🟥 [FIX-F1] 루프 동안 모아둔 셀 업데이트를 여기서 한 번에 flush.
    #    행 300개 × 5셀 = 1,500번의 개별 쓰기가 batch_update 4회로 줄어든다.
    written = _flush_sheet_updates(sheet, pending, label="결과추적")
    print(f"📊 [결과추적] 체크 {checked}건 / 업데이트 {updated}건 / 반영 셀 {written}개")
    return {"checked": checked, "updated": updated, "cells_written": written}


def _build_score_lookup(main_rows):
    """메인 시트에서 종목별 (시각, 점수) 리스트를 만든다. 'Alpaca 거래내역'과 시각 매칭용."""
    lookup = {}
    for row in main_rows[1:]:
        if len(row) < 6 or not row[1]:
            continue
        try:
            ts = datetime.fromisoformat(row[0])
            score = float(row[5])
        except Exception:
            continue
        lookup.setdefault(row[1], []).append((ts, score))
    for sym in lookup:
        lookup[sym].sort(key=lambda x: x[0])
    return lookup


def _find_matching_score(lookup, symbol, target_time_str, tolerance_minutes=10):
    """주문의 entry_time과 가장 가까운(허용오차 내) 메인 시트 점수를 찾아 반환. 못 찾으면 None."""
    if symbol not in lookup or not target_time_str:
        return None
    try:
        target = datetime.fromisoformat(target_time_str.replace("Z", "+00:00"))
    except Exception:
        return None
    best, best_diff = None, None
    for ts, score in lookup[symbol]:
        ts_utc = ts.astimezone(ZoneInfo("UTC")) if ts.tzinfo else ts
        diff = abs((target - ts_utc).total_seconds())
        if diff <= tolerance_minutes * 60 and (best_diff is None or diff < best_diff):
            best, best_diff = score, diff
    return best


def _find_force_close_fill(symbol, entry_time_iso):
    """
    TP/SL 레그가 둘 다 취소된 채로 포지션이 닫혔을 때, 그 청산을 실행한
    별도의 시장가 주문(체결가/체결시각)을 찾아서 반환. 못 찾으면 (None, None).
    """
    if not entry_time_iso:
        return None, None
    url = f"{ALPACA_TRADE_BASE_URL}/v2/orders"
    params = {"symbols": symbol, "status": "closed", "after": entry_time_iso, "limit": 20, "direction": "asc"}
    try:
        r = requests.get(url, headers=ALPACA_HEADERS, params=params, timeout=10)
        r.raise_for_status()
        for o in r.json():
            # bracket의 자식(legs)이 아니라, 독립적으로 들어간 시장가 청산 주문만 찾는다.
            if o.get("type") == "market" and o.get("status") == "filled" and not o.get("legs"):
                return float(o.get("filled_avg_price") or 0), o.get("filled_at")
    except Exception as e:
        print(f"❗ [강제청산조회] {symbol} 청산주문 조회 실패: {e}")
    return None, None


# ============================================================
# 🟥 [FIX-A3] 진입 시각 메모리 레지스트리
# ------------------------------------------------------------
#  Alpaca 주문내역 조회에만 의존하면 진입시각을 놓치는 경우가 많다(아래 함수 주석 참조).
#  주문이 나갈 때 여기에 기록해두고, 시간청산이 이걸 1순위로 본다.
#  프로세스 재시작 시 비므로, 조회 폴백도 그대로 유지한다.
# ============================================================
_position_entry_times: dict[str, datetime] = {}
_position_entry_lock = threading.Lock()


def _record_position_entry(symbol: str, side: str):
    """주문 성공 직후 진입시각 기록. side: 'long' | 'short'"""
    if not symbol:
        return
    key = f"{symbol.upper()}:{side}"
    with _position_entry_lock:
        _position_entry_times[key] = datetime.now(ZoneInfo("UTC"))
    print(f"🕐 [진입기록] {key} @ {_position_entry_times[key].isoformat()}")


def _clear_position_entry(symbol: str, side: str | None = None):
    """포지션이 닫혔을 때 기록 제거."""
    if not symbol:
        return
    with _position_entry_lock:
        for s in (["long", "short"] if side is None else [side]):
            _position_entry_times.pop(f"{symbol.upper()}:{s}", None)


def _get_latest_entry_time_for_open_position(symbol, side):
    """
    현재 열려있는 포지션(symbol)의 진입 시각을 찾는다.

    🟥 [FIX-A3] 기존 구현의 치명적 결함:
       limit=20으로 최근 주문 20건만 조회했는데, 브래킷 주문은 진입 1건 + TP/SL 자식 2건을
       만들기 때문에 같은 종목에 알림이 6~7번만 와도 20칸이 다 차서 진짜 진입 주문이
       조회 범위 밖으로 밀려났다. 그러면 "진입시각을 못 찾아서 스킵"으로 빠지고
       포지션이 영원히 안 닫혔다 — 실거래에서 최장 4,189분(약 3일) 보유가 나온 이유.
       → ① 메모리 레지스트리 우선 조회 ② API 폴백은 limit=500으로 확대
         ③ 자식 주문(브래킷 leg) 제외하고 진입 market 주문만 선별
    """
    key = f"{(symbol or '').upper()}:{side}"
    with _position_entry_lock:
        cached = _position_entry_times.get(key)
    if cached:
        return cached

    try:
        url = f"{ALPACA_TRADE_BASE_URL}/v2/orders"
        # limit 20 → 500. Alpaca 1회 조회 상한이 500이다.
        params = {"symbols": symbol, "status": "all", "direction": "desc", "limit": 500,
                  "nested": "true"}
        r = requests.get(url, headers=ALPACA_HEADERS, params=params, timeout=15)
        r.raise_for_status()
        want_side = "buy" if side == "long" else "sell"
        for o in r.json():
            # 진입 주문만: 방향 일치 + 체결 완료 + market 타입 + 브래킷 부모(자식 leg 제외)
            if (o.get("side") == want_side
                    and o.get("status") == "filled"
                    and o.get("filled_at")
                    and o.get("type") == "market"
                    # (nested=true면 자식 leg는 최상위에 안 나오므로 별도 필터 불필요)
                    and not o.get("parent_id")):
                t = datetime.fromisoformat(o["filled_at"].replace("Z", "+00:00"))
                # 찾은 값을 캐시에 넣어 다음 루프부터는 API를 안 타게 한다
                with _position_entry_lock:
                    _position_entry_times[key] = t
                return t
    except Exception as e:
        print(f"❗ [강제청산] {symbol} 진입시각 조회 실패: {e}")
    return None


# ============================================================
# 🟥 [FIX-G1] 브래킷 예약주문 취소 → 그 다음 포지션 청산
# ------------------------------------------------------------
#  실제 로그에서 확인된 실패:
#    403 {"available":"0","existing_qty":"10","held_for_orders":"10",
#         "message":"insufficient qty available for order (requested:10, available:0)"}
#
#  원인: 브래킷 주문의 TP(limit) / SL(stop) 자식 주문이 보유 수량 전량을
#        'held_for_orders'로 예약(hold)하고 있다. 그래서 팔 수 있는 수량이 0이고,
#        DELETE /v2/positions/{symbol} 이 403으로 거부된다.
#        (기존 주석의 "DELETE 하면 TP/SL도 같이 정리된다"는 사실이 아니다 —
#         Alpaca는 단일 종목 청산 시 자식 주문을 자동 취소해주지 않는다.
#         cancel_orders 옵션은 '전체 청산'(DELETE /v2/positions)에만 있다.)
#
#  → 반드시 ① 해당 종목의 열린 주문 취소 ② 취소 반영 대기 ③ 포지션 청산 순서로 간다.
#  ⚠️ ①을 하고 ③이 실패하면 포지션이 보호장치 없이 남는다. 그래서 재시도를 넣고,
#     끝까지 실패하면 CRITICAL 로그를 남겨 즉시 눈에 띄게 한다.
# ============================================================
# 🟥 [FIX-G4] 같은 종목 청산이 계속 실패하면 5분마다 403 3연타 + CRITICAL이 무한 반복된다.
#    실패한 종목은 잠시 쉬었다가 다시 시도해 로그·API 폭주를 막는다.
_close_fail_until: dict = {}
_close_fail_count: dict = {}
CLOSE_RETRY_BACKOFF_MINUTES = int(os.getenv("CLOSE_RETRY_BACKOFF_MINUTES", "30"))

_ALPACA_OPEN_ORDER_STATES = {
    "new", "accepted", "held", "partially_filled",
    "pending_new", "accepted_for_bidding", "calculated", "replaced", "pending_replace",
}


def _cancel_open_orders_for_symbol(symbol: str) -> int:
    """
    해당 종목의 열려있는 주문(브래킷 TP/SL 자식 포함)을 전부 취소. 취소한 개수 반환.

    🟥 [FIX-G3] `nested=true` 가 이 함수를 무력화하고 있었다.
       · 브래킷 진입 주문(부모)은 체결되면 status='filled' 이 된다.
       · `status=open` 필터는 filled 부모를 결과에서 제외한다.
       · 그런데 `nested=true` 를 주면 TP/SL 자식이 부모 안에 중첩되어서만 나온다.
       → 부모가 제외되면 자식도 같이 사라진다. 조회 결과가 0건.
       실제 로그에서 "🧹 예약주문 N건 취소"가 한 번도 안 찍히고
       곧바로 403 insufficient qty 가 3번 반복된 것이 바로 이 증상이다.
       → nested 를 빼면 자식 leg 들이 최상위 주문으로 각각 조회된다.
    """
    if not symbol:
        return 0
    try:
        r = requests.get(
            f"{ALPACA_TRADE_BASE_URL}/v2/orders",
            headers=ALPACA_HEADERS,
            # nested 제거 — 자식 leg를 최상위로 받아야 취소할 수 있다
            params={"status": "open", "symbols": symbol, "limit": 500},
            timeout=15,
        )
        r.raise_for_status()
        orders = r.json() or []
    except Exception as e:
        print(f"❌ [강제청산] {symbol} 열린 주문 조회 실패: {e}")
        return 0

    ids = []
    for o in orders:
        if (o.get("symbol") or "").upper() != symbol.upper():
            continue
        if o.get("id"):
            ids.append(o["id"])

    if not ids:
        print(f"ℹ️ [강제청산] {symbol} 취소할 열린 주문이 없음 (조회 {len(orders)}건)")

    cancelled = 0
    for oid in dict.fromkeys(ids):          # 중복 제거, 순서 유지
        try:
            rr = requests.delete(f"{ALPACA_TRADE_BASE_URL}/v2/orders/{oid}",
                                 headers=ALPACA_HEADERS, timeout=15)
            if rr.status_code in (200, 204):
                cancelled += 1
            elif rr.status_code == 422:
                cancelled += 1          # 이미 체결/취소됨 — 목적은 달성된 것
            else:
                print(f"⚠️ [강제청산] {symbol} 주문 {oid} 취소 실패: {rr.status_code} {rr.text[:150]}")
        except Exception as e:
            print(f"⚠️ [강제청산] {symbol} 주문 {oid} 취소 예외: {e}")

    if cancelled:
        print(f"🧹 [강제청산] {symbol} 예약주문 {cancelled}건 취소 (held_for_orders 해제)")
    return cancelled


def _force_close_position(symbol: str, side: str | None = None) -> bool:
    """
    예약주문 취소 → 포지션 시장가 청산. 성공 여부 반환.
    403(insufficient qty)이 나면 취소가 아직 반영 안 된 것이므로 재시도한다.
    """
    key = (symbol or "").upper()
    now = _t.time()
    if _close_fail_until.get(key, 0) > now:
        wait_min = (_close_fail_until[key] - now) / 60
        print(f"⏸️ [강제청산] {symbol} 직전 실패로 대기 중 ({wait_min:.0f}분 남음) — 이번 회차 건너뜀")
        return False

    _cancel_open_orders_for_symbol(symbol)

    for attempt in range(3):
        if attempt:
            _t.sleep(1.5)               # 취소 반영 대기
        try:
            r = requests.delete(f"{ALPACA_TRADE_BASE_URL}/v2/positions/{symbol}",
                                headers=ALPACA_HEADERS, timeout=15)
        except Exception as e:
            print(f"❌ [강제청산] {symbol} 청산 요청 예외({attempt+1}/3): {e}")
            continue

        if r.status_code in (200, 207):
            print(f"✅ [강제청산] {symbol} 청산 완료")
            _clear_position_entry(symbol, side)
            _close_fail_until.pop(key, None)      # 🟥 [FIX-G4] 성공 시 백오프 해제
            _close_fail_count.pop(key, None)
            return True

        body = r.text[:300]
        print(f"[강제청산] {symbol} 결과({attempt+1}/3): {r.status_code} {body}")

        if r.status_code == 403 and "insufficient qty" in body:
            # 아직 hold가 안 풀렸다 → 한 번 더 취소 시도 후 재시도
            _cancel_open_orders_for_symbol(symbol)
            continue
        if r.status_code == 404:
            print(f"ℹ️ [강제청산] {symbol} 포지션 없음(이미 청산됨)")
            _clear_position_entry(symbol, side)
            _close_fail_until.pop(key, None)
            _close_fail_count.pop(key, None)
            return True
        break   # 그 외 에러는 재시도해도 소용없음

    # 🟥 [FIX-G4] 실패 백오프 설정 — 5분마다 무한 재시도하며 로그를 도배하지 않게 한다.
    _close_fail_count[key] = _close_fail_count.get(key, 0) + 1
    _close_fail_until[key] = _t.time() + CLOSE_RETRY_BACKOFF_MINUTES * 60
    print(f"🚨 [강제청산][CRITICAL] {symbol} 청산 실패({_close_fail_count[key]}회째). "
          f"{CLOSE_RETRY_BACKOFF_MINUTES}분 후 재시도합니다. "
          f"예약주문 취소 후에도 청산이 안 됐다면 포지션이 보호장치 없이 남아 있을 수 있으니 "
          f"Alpaca에서 직접 확인하세요.")
    return False


def _close_all_positions_with_orders() -> dict:
    """
    장마감 전 전량청산용. DELETE /v2/positions?cancel_orders=true 는
    '모든 예약주문 취소 + 모든 포지션 청산'을 한 번에 해준다(단일 종목 엔드포인트에는 없는 옵션).
    """
    try:
        r = requests.delete(f"{ALPACA_TRADE_BASE_URL}/v2/positions",
                            headers=ALPACA_HEADERS,
                            params={"cancel_orders": "true"}, timeout=30)
        ok = r.status_code in (200, 207)
        print(f"🌆 [전량청산] 결과: {r.status_code} {r.text[:300]}")
        if ok:
            with _position_entry_lock:
                _position_entry_times.clear()
        return {"ok": ok, "status": r.status_code}
    except Exception as e:
        print(f"❌ [전량청산] 실패: {e}")
        return {"ok": False, "error": str(e)}


def close_stale_positions(cutoff_minutes=None):
    """
    Alpaca의 '현재 실시간 보유 포지션'을 직접 조회해서(=구글시트 탭에 의존하지 않음),
    진입 후 cutoff_minutes(기본 STOCK_TIME_EXIT_MINUTES)가 지났는데도 안 닫힌 것들을
    시장가로 강제 청산한다. 'Alpaca 거래내역' 탭은 한 번에 최근 500건만 보여주는 한계가 있어서,
    오래된 미청산 포지션이 그 탭에서 빠질 수 있었음 — 그래서 탭이 아니라 Alpaca 포지션 API를 직접 본다.
    🟥 [FIX-G1] 청산은 반드시 _force_close_position()을 통해서 한다.
    (브래킷 TP/SL이 수량을 hold 하고 있어서, 예약주문을 먼저 취소하지 않으면 403이 난다)
    """
    # 🟥 [FIX-H1] 0 이하 = 시간청산 비활성. (기존 `or` 연산은 0을 falsy로 처리해
    #    STOCK_TIME_EXIT_MINUTES=0이면 cutoff=0이 되고, `held < 0`이 항상 거짓이라
    #    "모든 포지션 즉시 청산"이라는 정반대 동작이 됐다.)
    cutoff = STOCK_TIME_EXIT_MINUTES if cutoff_minutes is None else cutoff_minutes
    time_exit_on = cutoff is not None and cutoff > 0
    try:
        r = requests.get(f"{ALPACA_TRADE_BASE_URL}/v2/positions", headers=ALPACA_HEADERS, timeout=15)
        r.raise_for_status()
        positions = r.json()
    except Exception as e:
        print(f"❌ [강제청산] 포지션 조회 실패: {e}")
        return {"checked": 0, "closed": 0}

    now_utc = datetime.now(ZoneInfo("UTC"))
    now_ny = now_utc.astimezone(ZoneInfo("America/New_York"))
    ny_hhmm = now_ny.hour * 100 + now_ny.minute
    # 🟥 [FIX-A3] 장마감 전 전량 청산 구간인지 판정.
    #    실거래에서 최대 손실을 만든 건 SL 미달성이 아니라 "밤을 넘긴 포지션의 갭"이었다.
    #    (PLTR: -0.78% 손절 예정 → 오버나이트 갭으로 -5.17%, -$102)
    #    이 구간에서는 진입시각을 몰라도 무조건 닫는다.
    # 🟥 [FIX-A3b] HHMM=0을 "비활성"으로 문서화해놓고 코드는 0 <= ny_hhmm < 1600 이라
    #    하루 종일 전량청산이 도는 치명적 반대 동작이었다. 0/음수는 명시적으로 끈다.
    eod_flatten = (
        STOCK_EOD_FLATTEN_ENABLED
        and STOCK_EOD_FLATTEN_HHMM > 0
        and now_ny.weekday() < 5
        and STOCK_EOD_FLATTEN_HHMM <= ny_hhmm < 1600
    )
    if eod_flatten:
        print(f"🌆 [강제청산] 장마감 전 전량청산 구간({now_ny:%H:%M} ET ≥ {STOCK_EOD_FLATTEN_HHMM}) "
              f"→ 보유 포지션 전부 정리")
        # 🟥 [FIX-G1] 전량청산은 cancel_orders=true 옵션이 있는 전용 엔드포인트로 한 번에 처리한다.
        #    (종목별로 취소→청산을 반복하는 것보다 빠르고, 누락이 없다)
        if positions:
            _res = _close_all_positions_with_orders()
            if _res.get("ok"):
                print(f"📊 [강제청산] 전량청산 {len(positions)}건 처리 완료")
                return {"checked": len(positions), "closed": len(positions), "eod_flatten": True}
            print("⚠️ [강제청산] 전량청산 API 실패 → 종목별 개별 청산으로 폴백")

    # 🟥 [FIX-A3c] TP/SL로 정상 청산된 포지션은 _clear_position_entry()가 안 불린다.
    #    그 상태로 같은 종목에 새 포지션이 생기면 옛 진입시각을 읽어 즉시 강제청산될 수 있다.
    #    → 매 루프마다 "현재 실제로 열려있는 포지션" 집합과 캐시를 동기화한다.
    _open_keys = {
        f"{(p.get('symbol') or '').upper()}:{p.get('side')}"
        for p in positions if p.get("symbol")
    }
    # 🟥 [FIX-A3d] 방금 주문이 나간 건은 아직 Alpaca 포지션에 안 잡혀 있을 수 있다
    #    (market 주문이 accepted/held 상태). 유예 시간(기본 10분) 안의 기록은 지우지 않는다.
    _grace = timedelta(minutes=int(os.getenv("ENTRY_CACHE_GRACE_MINUTES", "10")))
    with _position_entry_lock:
        for _stale in [k for k, t in _position_entry_times.items()
                       if k not in _open_keys and (now_utc - t) > _grace]:
            _position_entry_times.pop(_stale, None)
            print(f"🧹 [진입기록] 청산 완료된 {_stale} 캐시 제거")

    checked, closed = 0, 0

    for pos in positions:
        symbol = pos.get("symbol")
        qty = abs(float(pos.get("qty", 0) or 0))
        side = pos.get("side")  # "long" or "short"
        if not symbol or qty <= 0:
            continue

        entry_t = _get_latest_entry_time_for_open_position(symbol, side)
        held_minutes = None
        if entry_t:
            held_minutes = (now_utc - entry_t).total_seconds() / 60

        # 🟥 [FIX-G5] 밤을 넘겨버린 포지션 정리.
        #    EOD 전량청산 창은 15:50~16:00 뿐이라, 그 10분에 배포/재시작/에러가 겹치면
        #    포지션이 그대로 밤을 넘긴다. 실제 로그에서 424~665분(7~11시간)짜리
        #    포지션이 남아 있던 게 이 경우다.
        #    → 진입일이 오늘(ET)이 아니면 정규장 시간에 즉시 정리한다.
        is_overnight = False
        entry_ny = None
        if entry_t is not None:
            entry_ny = entry_t.astimezone(ZoneInfo("America/New_York"))
            is_overnight = entry_ny.date() != now_ny.date()
        in_regular = 930 <= ny_hhmm < 1600 and now_ny.weekday() < 5

        if eod_flatten:
            why = f"장마감 전 전량청산({now_ny:%H:%M} ET)"
        elif is_overnight and in_regular:
            why = f"오버나이트 잔여 포지션(진입 {entry_ny:%m/%d %H:%M} ET) → 정규장 개시 후 즉시 정리"
        elif not time_exit_on and held_minutes is None:
            continue     # 🟥 [FIX-H1] 시간청산 꺼져 있으면 진입시각을 몰라도 문제없음
        elif held_minutes is None:
            # 🟥 [FIX-A3] 기존엔 여기서 무조건 continue라 포지션이 영원히 안 닫혔다.
            #    진입시각을 모르면 최소한 로그를 남기고, EOD 구간에서 반드시 정리되게 한다.
            print(f"⚠️ [강제청산] {symbol} 진입시각 미상 → 시간청산 보류 "
                  f"(장마감 {STOCK_EOD_FLATTEN_HHMM} ET에 전량청산 대상)")
            continue
        elif not in_regular:
            # 🟥 [FIX-G5] 정규장 밖에서는 시장가 주문이 체결되지 않는다.
            #    로그에서 21:10(장 종료 5시간 후)에 계속 청산을 시도하며 403을 뿜던 구간.
            continue
        elif not time_exit_on:
            continue     # 🟥 [FIX-H1] 시간청산 비활성 — EOD 전량청산만 담당
        elif held_minutes < cutoff:
            continue
        else:
            why = f"진입 후 {held_minutes:.1f}분 경과(컷오프 {cutoff}분)"

        checked += 1
        print(f"⏰ [강제청산] {symbol} {why} → 예약주문 취소 후 시장가 청산 시도")
        # 🟥 [FIX-G1] 예약주문(TP/SL) 취소 → 청산. 기존엔 바로 DELETE 해서 403이 무한 반복됐다.
        if _force_close_position(symbol, side):
            closed += 1

    print(f"📊 [강제청산] 체크 {checked}건 / 청산 {closed}건")
    return {"checked": checked, "closed": closed, "eod_flatten": eod_flatten}


# ============================================================
# 🟥 [FIX-MFE1] MFE / MAE 추적 (2026-08-31)
# ------------------------------------------------------------
#  왜 필요한가:
#    "TP 배수를 2.4 로 둘까 2.0 으로 둘까"를 지금까지 추정으로만 답했다.
#    진입가·청산가만 있으면 '손절난 거래가 잘리기 전에 어디까지 갔었나'를 알 수 없어서,
#    "TP 를 좁히면 손절 몇 건이 승리로 바뀌는가"를 셀 수가 없다.
#    MFE(최대 유리 이동)를 기록하면 그 질문에 정확히 답할 수 있다.
#    MAE(최대 불리 이동)는 반대로 "SL 을 넓혔으면 살았을 거래"를 세게 해준다.
#
#  ⚠️ 정확도 한계 — 반드시 알고 쓸 것:
#    무료(Basic) 플랜은 IEX 피드만 준다. IEX 는 전체 거래량의 2~10% 만 보므로
#    1분봉의 고가/저가가 실제 전체시장 극단값보다 **작게** 나온다.
#    즉 여기서 나오는 MFE 는 **하한**이다. 실제 MFE 는 이 값 이상이다.
#    → "MFE 가 X% 였다" 는 보수적으로 읽고, 방향 판단(대부분이 TP 근처까지 갔나)에만 쓴다.
#
#  비용 관리: 청산된 거래 1건당 API 1회. 이미 계산된 값은 시트에서 읽어 재사용하고,
#            한 번의 동기화에서 최대 MFE_MAX_FETCH_PER_SYNC 건만 새로 계산한다.
#            (나머지는 다음 동기화에서 채워진다 — 시간이 지나면 전부 채워짐)
# ============================================================
MFE_TRACK_ENABLED = os.getenv("MFE_TRACK_ENABLED", "true").strip().lower() != "false"
MFE_MAX_FETCH_PER_SYNC = int(os.getenv("MFE_MAX_FETCH_PER_SYNC", "40"))
#  방금 닫힌 거래는 데이터가 아직 안 채워졌을 수 있어 잠깐 미룬다(다음 동기화에서 계산).
MFE_MIN_AGE_MINUTES = int(os.getenv("MFE_MIN_AGE_MINUTES", "20"))
#  ⚠️ 건수 한도만으로는 부족하다. 알파카가 느려지면 40건 × 타임아웃 = 수 분간 물린다.
#     동기화 1회당 MFE 계산에 쓸 수 있는 총 시간을 벽시계로 못박는다.
MFE_TIME_BUDGET_SEC = int(os.getenv("MFE_TIME_BUDGET_SEC", "60"))
#  무료 플랜은 iex. 유료(SIP)면 환경변수로 sip 로 바꾸면 MFE 정확도가 크게 올라간다.
ALPACA_DATA_FEED = os.getenv("ALPACA_DATA_FEED", "iex").strip().lower()
#  ⚠️ 바 데이터가 아예 없는 거래(상장폐지·데이터 공백 등)는 영원히 재시도하며 API 를 먹는다.
#     너무 오래된 거래는 아예 시도하지 않아 그 누수를 막는다.
MFE_MAX_AGE_DAYS = int(os.getenv("MFE_MAX_AGE_DAYS", "45"))


def _fetch_excursion(symbol, side, entry_price, entry_iso, exit_iso):
    """보유 구간의 1분봉을 받아 (MFE%, MAE%) 를 돌려준다. 실패하면 (None, None).

    MFE = 보유 중 '가장 유리했던' 지점까지의 이동 (BUY 면 최고가, SELL 이면 최저가)
    MAE = 보유 중 '가장 불리했던' 지점까지의 이동 (부호는 음수)
    """
    if not (symbol and entry_price and entry_iso and exit_iso):
        return None, None
    try:
        t1 = datetime.fromisoformat(str(entry_iso).replace("Z", "+00:00"))
        t2 = datetime.fromisoformat(str(exit_iso).replace("Z", "+00:00"))
    except Exception:
        return None, None
    if t2 <= t1:
        return None, None
    # ⚠️ 방금 끝난 거래는 건너뛴다 — 바 데이터가 아직 안 올라와 MFE 가 0 으로 굳어버린다.
    _age = datetime.now(ZoneInfo("UTC")) - t2
    if _age.total_seconds() < MFE_MIN_AGE_MINUTES * 60:
        return None, None
    if _age.days > MFE_MAX_AGE_DAYS:
        return None, None

    # 진입/청산 봉 자체를 포함시키려고 앞뒤로 1분씩 여유를 준다.
    url = f"{ALPACA_DATA_BASE_URL}/v2/stocks/{symbol}/bars"
    params = {
        "timeframe": "1Min",
        "start": (t1 - timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end":   (t2 + timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "adjustment": "raw",
        "feed": ALPACA_DATA_FEED,
        "limit": 10000,
        "sort": "asc",
    }
    try:
        r = requests.get(url, headers=ALPACA_HEADERS, params=params, timeout=(5, 10))
        r.raise_for_status()
        bars = r.json().get("bars") or []
    except Exception as e:
        print(f"⚠️ [MFE] {symbol} 바 조회 실패(무시): {e}")
        return None, None
    if not bars:
        return None, None

    try:
        hi = max(float(b["h"]) for b in bars)
        lo = min(float(b["l"]) for b in bars)
        ep = float(entry_price)
        if ep <= 0:
            return None, None
        if (side or "").upper() == "BUY":
            mfe = (hi - ep) / ep * 100.0
            mae = (lo - ep) / ep * 100.0
        else:
            mfe = (ep - lo) / ep * 100.0
            mae = (ep - hi) / ep * 100.0
        # MFE 는 0 이상, MAE 는 0 이하로 정리 (진입가가 그 봉 범위 밖일 때의 부호 뒤집힘 방지)
        return round(max(mfe, 0.0), 3), round(min(mae, 0.0), 3)
    except Exception:
        return None, None


def _mfe_over_tp(r):
    """MFE 가 TP 거리의 몇 % 까지 갔는가. 100 이면 TP 도달, 80 이면 목표의 80% 까지 갔다는 뜻.
    이 값이 '손절난 거래' 에서 크게 나오면 → TP 를 그만큼 좁혔을 때 승리로 바뀌었을 거래다."""
    try:
        mfe = r.get("mfe")
        ep, tp = r.get("entry_price"), r.get("tp")
        if mfe is None or not ep or not tp:
            return ""
        tp_dist_pct = abs(float(tp) - float(ep)) / float(ep) * 100.0
        if tp_dist_pct <= 0:
            return ""
        return round(float(mfe) / tp_dist_pct * 100.0, 1)
    except Exception:
        return ""


def sync_alpaca_trade_log():
    """
    Alpaca 주문 내역(원본 데이터)을 직접 조회해서 'Alpaca 거래내역' 탭에 깔끔하게 정리.
    - 탭이 없으면 자동으로 만들고 헤더도 자동으로 씀 (사용자가 직접 만들 필요 없음).
    - 매번 전체를 다시 계산해서 덮어쓴다(상태 변화: 진행중→TP/SL청산 반영이 쉬워짐).
    - 메인 시트의 signal_score를 시각 매칭해서 같이 기록 → 나중에 threshold 백테스팅용.
    """
    HEADERS = [
        "주문ID", "진입시각", "종목", "방향", "점수", "수량", "진입가",
        "TP가", "SL가", "상태", "청산가", "청산시각", "보유시간(분)",
        "손익($)", "손익(%)", "누적손익($)",
        # 🟥 [FIX-MFE1] TP/SL 배수를 데이터로 정하기 위한 3개 컬럼
        "MFE(%)", "MAE(%)", "MFE/TP(%)"
    ]

    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name("/etc/secrets/google_credentials.json", scope)
        client = gspread.authorize(creds)
        spreadsheet = client.open("민균 FX trading result")
        score_lookup = _build_score_lookup(spreadsheet.sheet1.get_all_values())

        try:
            ws = spreadsheet.worksheet("Alpaca 거래내역")
        except gspread.exceptions.WorksheetNotFound:
            ws = spreadsheet.add_worksheet(title="Alpaca 거래내역", rows=1000, cols=len(HEADERS))
            print("✅ [Alpaca거래내역] 탭이 없어서 새로 생성했습니다.")

        # 🟥 [FIX-MFE1] ws.clear() 로 지우기 '전에' 기존 MFE/MAE 를 읽어 캐시한다.
        #   청산된 거래의 MFE 는 두 번 다시 안 바뀌므로 한 번만 계산하면 된다.
        #   이걸 안 하면 동기화할 때마다 157건 × API 1회를 다시 때려서 레이트리밋에 걸린다.
        _mfe_cache = {}
        try:
            _old_vals = ws.get_all_values()
            if _old_vals and len(_old_vals) > 1:
                _hdr = _old_vals[0]
                _i_id = _hdr.index("주문ID") if "주문ID" in _hdr else 0
                _i_mfe = _hdr.index("MFE(%)") if "MFE(%)" in _hdr else None
                _i_mae = _hdr.index("MAE(%)") if "MAE(%)" in _hdr else None
                if _i_mfe is not None:
                    for _row in _old_vals[1:]:
                        if len(_row) <= _i_mfe:
                            continue
                        _oid = (_row[_i_id] or "").strip()
                        _m = (_row[_i_mfe] or "").strip()
                        _a = (_row[_i_mae] or "").strip() if (_i_mae is not None and len(_row) > _i_mae) else ""
                        if _oid and _m != "":
                            try:
                                _mfe_cache[_oid] = (float(_m), float(_a) if _a != "" else None)
                            except ValueError:
                                pass
                print(f"🗂️ [MFE] 기존 값 {len(_mfe_cache)}건 재사용")
        except Exception as e:
            print(f"⚠️ [MFE] 기존 값 읽기 실패(무시): {e}")
    except Exception as e:
        print(f"❌ [Alpaca거래내역] 시트 연결 실패: {e}")
        return

    try:
        url = f"{ALPACA_TRADE_BASE_URL}/v2/orders"
        orders = []
        until_param = None
        # 🟦 Alpaca API는 한 번에 최대 500건만 주는데, 거래가 많이 쌓이면 오래된 미청산 포지션이
        #    아예 안 보이게 됨(이게 강제청산이 안 되던 진짜 원인이었음). 최대 5페이지(2500건)까지
        #    이어서 가져와서, 오래된 것도 이 탭에서 빠지지 않게 한다.
        for _ in range(5):
            params = {"status": "all", "nested": "true", "limit": 500, "direction": "desc"}
            if until_param:
                params["until"] = until_param
            r = requests.get(url, headers=ALPACA_HEADERS, params=params, timeout=15)
            r.raise_for_status()
            page = r.json()
            if not page:
                break
            orders.extend(page)
            if len(page) < 500:
                break  # 마지막 페이지
            until_param = page[-1].get("submitted_at")
    except Exception as e:
        print(f"❌ [Alpaca거래내역] 주문 내역 조회 실패: {e}")
        return

    rows = []
    for o in orders:
        if o.get("order_class") != "bracket":
            continue  # 우리가 직접 만든 bracket 진입 주문만 대상

        status = o.get("status")
        symbol = o.get("symbol")
        side = (o.get("side") or "").upper()
        qty = float(o.get("filled_qty") or o.get("qty") or 0)

        if status != "filled":
            # 진입 자체가 안 된 주문(취소/만료 등) — 참고용으로만 표시
            score = _find_matching_score(score_lookup, symbol, o.get("submitted_at"))
            rows.append({
                "order_id": o.get("id"), "entry_time": o.get("submitted_at"),
                "symbol": symbol, "side": side, "score": score, "qty": qty,
                "entry_price": None, "tp": None, "sl": None,
                "status_kr": f"미체결({status})", "exit_price": None, "exit_time": None,
                "pnl": None,
            })
            continue

        entry_price = float(o.get("filled_avg_price") or 0)
        entry_time = o.get("filled_at")

        tp_price, sl_price = None, None
        exit_price, exit_time, status_kr = None, None, "진행중"

        for leg in (o.get("legs") or []):
            leg_type = leg.get("type")
            if leg_type == "limit":
                tp_price = float(leg.get("limit_price") or 0) or tp_price
                if leg.get("status") == "filled":
                    exit_price = float(leg.get("filled_avg_price") or 0)
                    exit_time = leg.get("filled_at")
                    status_kr = "TP청산"
            elif leg_type in ("stop", "stop_limit"):
                sl_price = float(leg.get("stop_price") or 0) or sl_price
                if leg.get("status") == "filled":
                    exit_price = float(leg.get("filled_avg_price") or 0)
                    exit_time = leg.get("filled_at")
                    status_kr = "SL청산"

        # 🟦 TP/SL 둘 다 체결 안 됐는데 둘 다 "취소(canceled)" 상태면 → 우리 시간초과 강제청산
        #    (TIME_EXIT)으로 닫힌 경우다. 그 청산을 실행한 별도의 시장가 주문을 찾아서 채운다.
        legs = o.get("legs") or []
        if status_kr == "진행중" and legs and all(leg.get("status") == "canceled" for leg in legs):
            close_price, close_time = _find_force_close_fill(symbol, entry_time)
            if close_price is not None:
                exit_price, exit_time, status_kr = close_price, close_time, "TIME_EXIT"

        pnl = None
        if exit_price is not None and entry_price:
            direction = 1 if side == "BUY" else -1
            pnl = round((exit_price - entry_price) * qty * direction, 2)

        score = _find_matching_score(score_lookup, symbol, entry_time)

        rows.append({
            "order_id": o.get("id"), "entry_time": entry_time,
            "symbol": symbol, "side": side, "score": score, "qty": qty,
            "entry_price": entry_price, "tp": tp_price, "sl": sl_price,
            "status_kr": status_kr, "exit_price": exit_price, "exit_time": exit_time,
            "pnl": pnl,
        })

    # 진입시각 오름차순 정렬 (누적손익 계산을 위해)
    rows = [r for r in rows if r["entry_time"]]
    rows.sort(key=lambda r: r["entry_time"])

    def _to_et(iso_str):
        """UTC ISO 문자열 → ET(America/New_York) 표시 문자열. 사람이 읽기 편하게."""
        if not iso_str:
            return ""
        try:
            dt_utc = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
            dt_et = dt_utc.astimezone(ZoneInfo("America/New_York"))
            return dt_et.strftime("%Y-%m-%d %H:%M:%S ET")
        except Exception:
            return iso_str  # 변환 실패 시 원본 그대로

    # 🟥 [FIX-MFE1] 청산된 행에 MFE/MAE 를 채운다. 캐시에 있으면 그대로, 없으면 계산.
    _fetched = 0
    _mfe_t0 = time.monotonic()
    for r in rows:
        r["mfe"], r["mae"] = None, None
        if not MFE_TRACK_ENABLED:
            continue
        if r["status_kr"] in ("진행중",) or not r["exit_time"] or not r["entry_price"]:
            continue
        _oid = r.get("order_id") or ""
        if _oid in _mfe_cache:
            r["mfe"], r["mae"] = _mfe_cache[_oid]
            continue
        if _fetched >= MFE_MAX_FETCH_PER_SYNC:
            continue   # 이번 회차 건수 한도 초과 — 다음 동기화에서 채운다
        if time.monotonic() - _mfe_t0 > MFE_TIME_BUDGET_SEC:
            continue   # 시간 예산 초과 — 나머지는 다음 회차
        _m, _a = _fetch_excursion(r["symbol"], r["side"], r["entry_price"],
                                  r["entry_time"], r["exit_time"])
        if _m is not None:
            r["mfe"], r["mae"] = _m, _a
            _fetched += 1
    if MFE_TRACK_ENABLED:
        _todo = sum(1 for r in rows
                    if r["status_kr"] != "진행중" and r["exit_time"] and r.get("mfe") is None)
        print(f"📐 [MFE] 이번 회차 신규 {_fetched}건 계산 · 미계산 잔여 {_todo}건")

    sheet_rows = [HEADERS]
    cum_pnl = 0.0
    for r in rows:
        hold_minutes = ""
        if r["exit_time"] and r["entry_time"]:
            try:
                t1 = datetime.fromisoformat(r["entry_time"].replace("Z", "+00:00"))
                t2 = datetime.fromisoformat(r["exit_time"].replace("Z", "+00:00"))
                hold_minutes = round((t2 - t1).total_seconds() / 60, 1)
            except Exception:
                hold_minutes = ""

        pnl_pct = ""
        if r["pnl"] is not None and r["entry_price"]:
            pnl_pct = round(r["pnl"] / (r["entry_price"] * r["qty"]) * 100, 2) if r["qty"] else ""

        if r["pnl"] is not None:
            cum_pnl += r["pnl"]

        sheet_rows.append([
            r["order_id"],
            _to_et(r["entry_time"]),   # 🟦 UTC → ET 변환
            r["symbol"], r["side"], r["score"], r["qty"],
            r["entry_price"], r["tp"], r["sl"], r["status_kr"],
            r["exit_price"],
            _to_et(r["exit_time"]),    # 🟦 UTC → ET 변환
            hold_minutes,
            r["pnl"], pnl_pct, round(cum_pnl, 2) if r["pnl"] is not None else "",
            # 🟥 [FIX-MFE1]
            r.get("mfe") if r.get("mfe") is not None else "",
            r.get("mae") if r.get("mae") is not None else "",
            _mfe_over_tp(r),
        ])

    try:
        ws.clear()
        # 🟥 [FIX-D1] 탭은 1000행으로 만들어졌는데 주문이 그보다 쌓이면
        #    update 가 'exceeds grid limits' 로 터지면서 갱신이 아예 멈춘다.
        try:
            ws.resize(rows=max(len(sheet_rows) + 50, 1000), cols=len(HEADERS))
        except Exception as e:
            print(f"⚠️ [Alpaca거래내역] 크기 조정 실패: {e}")
        ws.update("A1", sheet_rows)
        print(f"✅ [Alpaca거래내역] {len(rows)}건 갱신 완료")
    except Exception as e:
        print(f"❌ [Alpaca거래내역] 시트 쓰기 실패: {e}")
        return

    # 🟥 [FIX-D1] 날짜가 바뀌는 곳마다 가로선
    _draw_date_separators(spreadsheet, ws, sheet_rows, date_col=1,
                          ncols=len(HEADERS), tag="Alpaca구분선")



# ============================================================
# 🟥 [FIX-D1] 날짜 구분선 — 날짜가 바뀌는 행 아래에 가로선을 긋는다
# ------------------------------------------------------------
#  ⚠️ ws.clear() 는 '값'만 지우고 '테두리'는 그대로 남긴다.
#     그래서 매번 먼저 전부 지우고 다시 긋는다. 안 그러면 갱신할 때마다
#     예전 선이 엉뚱한 행에 남아서 며칠 지나면 표가 걸레가 된다.
# ============================================================

def _draw_date_separators(spreadsheet, ws, sheet_rows, date_col, ncols, tag="구분선"):
    """
    sheet_rows[0] 은 헤더. sheet_rows[i][date_col] 의 앞 10글자(YYYY-MM-DD)를 날짜로 본다.
    날짜가 바뀌기 직전 행의 '아래'에 선을 긋는다.
    """
    try:
        sid = ws.id
    except Exception:
        return
    if not sheet_rows or ncols < 1:
        return

    # ⚠️ 구글은 '시트 격자 밖'의 범위를 주면 통째로 거부한다.
    #    ws.resize() 로 행을 줄여놨는데 그보다 넓게 지우려 들면 선이 하나도 안 그려진다.
    try:
        grid_rows = int(ws.row_count)
        grid_cols = int(ws.col_count)
    except Exception:
        grid_rows, grid_cols = len(sheet_rows), ncols
    ncols = max(1, min(int(ncols), grid_cols))
    wipe_to = max(1, min(max(len(sheet_rows), 1) + 500, grid_rows))
    reqs = [{
        "updateBorders": {
            "range": {"sheetId": sid, "startRowIndex": 0, "endRowIndex": wipe_to,
                      "startColumnIndex": 0, "endColumnIndex": ncols},
            "innerHorizontal": {"style": "NONE"},
            "bottom": {"style": "NONE"},
        }
    }]

    def _date_of(row):
        if date_col >= len(row):
            return ""
        return str(row[date_col] or "").strip()[:10]

    prev_date, last_idx = None, None
    for i in range(1, len(sheet_rows)):
        d = _date_of(sheet_rows[i])
        if not d:
            continue
        if prev_date is not None and d != prev_date and last_idx is not None:
            reqs.append({
                "updateBorders": {
                    "range": {"sheetId": sid,
                              "startRowIndex": last_idx, "endRowIndex": last_idx + 1,
                              "startColumnIndex": 0, "endColumnIndex": ncols},
                    "bottom": {"style": "SOLID_MEDIUM",
                               "color": {"red": 0.20, "green": 0.20, "blue": 0.20}},
                }
            })
        prev_date, last_idx = d, i

    try:
        spreadsheet.batch_update({"requests": reqs})
        print(f"✅ [{tag}] 날짜 경계 {len(reqs) - 1}곳에 선을 그었다")
    except Exception as e:
        # 선 긋기는 '있으면 좋은 것'이다. 실패해도 거래내역 자체는 이미 기록됐다.
        print(f"⚠️ [{tag}] 선 긋기 실패(내용은 정상 기록됨): {e}")


# ============================================================
# 🟥 [FIX-O1] 'OANDA 거래내역' 탭 — Alpaca 탭과 똑같은 모양으로
# ------------------------------------------------------------
#  왜 필요한가: 지금까지 FX/금 거래는 '어디에도 정리돼 있지 않았다'.
#  주식은 'Alpaca 거래내역'에서 승률·손익을 한눈에 보는데, OANDA 쪽은
#  오완다 웹사이트에 들어가야만 볼 수 있었다. 같은 표로 만들어야
#  주식 전략과 FX 전략을 나란히 비교할 수 있다.
#
#  Alpaca 탭과 다른 점 하나: '스왑($)' 칸이 있다.
#  FX는 포지션을 하루 넘겨 들고 있으면 이자(스왑)가 붙거나 빠진다.
#  주식엔 없는 비용이라 따로 뺐다. 진짜 손익 = 손익($) + 스왑($).
# ============================================================

def sync_oanda_trade_log():
    HEADERS = [
        "트레이드ID", "진입시각", "종목", "방향", "점수", "수량", "진입가",
        "TP가", "SL가", "상태", "청산가", "청산시각", "보유시간(분)",
        "손익($)", "손익(%)", "스왑($)", "누적손익($)"
    ]

    if not (OANDA_API_KEY and ACCOUNT_ID):
        print("⚠️ [OANDA거래내역] OANDA_API_KEY / ACCOUNT_ID 가 없어서 건너뜀")
        return

    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name(GOOGLE_CREDS_PATH, scope)
        client = gspread.authorize(creds)
        spreadsheet = client.open(GOOGLE_SHEET_NAME)
        score_lookup = _build_score_lookup(spreadsheet.sheet1.get_all_values())

        try:
            ws = spreadsheet.worksheet("OANDA 거래내역")
        except gspread.exceptions.WorksheetNotFound:
            ws = spreadsheet.add_worksheet(title="OANDA 거래내역", rows=1000, cols=len(HEADERS))
            print("✅ [OANDA거래내역] 탭이 없어서 새로 생성했습니다.")
    except Exception as e:
        print(f"❌ [OANDA거래내역] 시트 연결 실패: {e}")
        return

    # ── 1) 트레이드 내역 조회 (최대 5페이지 = 2500건) ──────────
    # ⚠️ 여기서 실패하면 '절대' 시트를 건드리면 안 된다.
    #    실패했는데 그냥 진행하면 ws.clear() 가 돌아서 그동안 쌓인 거래내역이
    #    통째로 날아간다. 그것도 30분마다 반복해서. 그래서 fetch_ok 로 막는다.
    headers = {"Authorization": f"Bearer {OANDA_API_KEY}"}
    trades = []
    seen_ids = set()
    before_id = None
    fetch_ok = False
    try:
        for _page_i in range(5):
            params = {"state": "ALL", "count": 500}
            if before_id:
                params["beforeID"] = before_id
            r = requests.get(f"{OANDA_BASE_URL}/v3/accounts/{ACCOUNT_ID}/trades",
                             headers=headers, params=params, timeout=20)
            if not r.ok:
                print(f"❌ [OANDA거래내역] 조회 실패 status={r.status_code} {r.text[:200]}")
                fetch_ok = False
                break
            page = (r.json() or {}).get("trades") or []
            fetch_ok = True
            if not page:
                break
            # 응답 정렬 순서를 믿지 않는다. ID 로 중복을 직접 걸러야
            # 누적손익이 두 배로 뻥튀기되는 사고가 안 난다.
            fresh = [t for t in page if str(t.get("id")) not in seen_ids]
            for t in fresh:
                seen_ids.add(str(t.get("id")))
            trades.extend(fresh)
            if len(page) < 500 or not fresh:
                break
            try:
                ids = [int(t["id"]) for t in page if str(t.get("id")).isdigit()]
                if not ids:
                    break
                nxt = min(ids) - 1     # beforeID 는 '이 ID 이하' → 1 빼야 중복이 없다
                if nxt <= 0 or (before_id and nxt >= int(before_id)):
                    break              # 진도가 안 나가면 무한루프 방지
                before_id = str(nxt)
            except Exception:
                break
    except Exception as e:
        print(f"❌ [OANDA거래내역] 조회 중 오류: {e}")
        return

    if not fetch_ok:
        print("⛔ [OANDA거래내역] 조회에 실패해서 시트는 건드리지 않고 그대로 둡니다.")
        return

    if not trades:
        print("ℹ️ [OANDA거래내역] 거래 기록이 아직 없습니다.")

    def _f(v, default=None):
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    def _to_et(iso_str):
        if not iso_str:
            return ""
        try:
            s = str(iso_str)
            # OANDA 는 나노초까지 준다: 2026-08-24T13:30:00.123456789Z → 파이썬이 못 읽는다
            if "." in s:
                head, tail = s.split(".", 1)
                frac = "".join(ch for ch in tail if ch.isdigit())[:6]
                s = f"{head}.{frac}+00:00"
            else:
                s = s.replace("Z", "+00:00")
            dt = datetime.fromisoformat(s)
            return dt.astimezone(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M:%S ET")
        except Exception:
            return str(iso_str)[:19]

    def _iso_utc(iso_str):
        """_find_matching_score 가 읽을 수 있는 형태로."""
        if not iso_str:
            return None
        s = str(iso_str)
        if "." in s:
            head, tail = s.split(".", 1)
            frac = "".join(ch for ch in tail if ch.isdigit())[:6]
            return f"{head}.{frac}+00:00"
        return s.replace("Z", "+00:00")

    rows = []
    for t in trades:
        inst = t.get("instrument") or ""
        units0 = _f(t.get("initialUnits"), 0.0) or 0.0
        side = "BUY" if units0 > 0 else "SELL"
        entry_price = _f(t.get("price"))
        entry_time = t.get("openTime")
        state = (t.get("state") or "").upper()

        tp_o = t.get("takeProfitOrder") or {}
        sl_o = t.get("stopLossOrder") or {}
        ts_o = t.get("trailingStopLossOrder") or {}
        tp_price = _f(tp_o.get("price"))
        sl_price = (_f(sl_o.get("price"))
                    or _f(ts_o.get("trailingStopValue"))
                    or _f(ts_o.get("price")))

        closed = (state == "CLOSED")
        # ⚠️ averageClosePrice 는 '일부만 줄인' 트레이드에도 채워진다.
        #    그걸 그대로 쓰면 '진행중인데 청산가가 찍혀 있는' 모순된 줄이 나온다.
        exit_price = _f(t.get("averageClosePrice")) if closed else None
        exit_time = t.get("closeTime") if closed else None
        part_closed = (not closed) and _f(t.get("averageClosePrice")) is not None

        if state == "CLOSE_WHEN_TRADEABLE":
            status_kr = "청산대기"
        elif state == "OPEN":
            status_kr = "일부청산·진행중" if part_closed else "진행중"
        elif str(tp_o.get("state") or "").upper() == "FILLED":
            status_kr = "TP청산"
        elif str(sl_o.get("state") or "").upper() == "FILLED":
            status_kr = "SL청산"
        elif str(ts_o.get("state") or "").upper() == "FILLED":
            status_kr = "트레일청산"
        elif closed:
            # ⚠️ OANDA 가 닫힌 트레이드에 TP/SL 주문 객체를 안 붙여주는 경우가 있다.
            #    그러면 전부 '수동청산'으로 보여서 TP 승률을 못 센다.
            #    그럴 땐 청산가가 TP선/SL선 중 어디에 붙었는지로 되짚는다.
            _ep = _f(t.get("averageClosePrice"))
            _tol = (abs(_ep) * 0.0005) if _ep else 0.0     # 0.05% 허용
            if _ep and tp_price and abs(_ep - tp_price) <= _tol:
                status_kr = "TP청산(추정)"
            elif _ep and sl_price and abs(_ep - sl_price) <= _tol:
                status_kr = "SL청산(추정)"
            else:
                status_kr = "수동/시간청산"
        else:
            status_kr = state or "?"

        realized = _f(t.get("realizedPL"), 0.0) or 0.0
        financing = _f(t.get("financing"), 0.0) or 0.0

        # 🟦 메인 시트 점수 매칭: OANDA 는 'USD_JPY', 트레이딩뷰 알람은 'USDJPY' 로 들어온다.
        #    어느 쪽으로 기록됐는지 모르니 둘 다 시도한다.
        score = None
        for cand in (inst, inst.replace("_", "")):
            score = _find_matching_score(score_lookup, cand, _iso_utc(entry_time))
            if score is not None:
                break

        rows.append({
            "id": t.get("id"), "entry_time": entry_time, "symbol": inst, "side": side,
            "score": score, "qty": abs(units0), "entry_price": entry_price,
            "tp": tp_price, "sl": sl_price, "status_kr": status_kr,
            "exit_price": exit_price, "exit_time": exit_time,
            "pnl": realized if state == "CLOSED" else None,
            "swap": financing,
        })

    rows = [r for r in rows if r["entry_time"]]
    rows.sort(key=lambda r: str(r["entry_time"]))

    sheet_rows = [HEADERS]
    cum_pnl = 0.0
    for r in rows:
        hold_minutes = ""
        if r["exit_time"] and r["entry_time"]:
            try:
                t1 = datetime.fromisoformat(_iso_utc(r["entry_time"]))
                t2 = datetime.fromisoformat(_iso_utc(r["exit_time"]))
                hold_minutes = round((t2 - t1).total_seconds() / 60, 1)
            except Exception:
                hold_minutes = ""

        # 🟦 손익(%)는 '가격이 몇 % 움직였나' 로 낸다. Alpaca 탭과 같은 의미다.
        #    FX 는 종목마다 계약 단위가 달라서 손익($)/명목 으로는 서로 비교가 안 된다.
        pnl_pct = ""
        if r["exit_price"] and r["entry_price"]:
            direction = 1 if r["side"] == "BUY" else -1
            pnl_pct = round((r["exit_price"] - r["entry_price"]) / r["entry_price"]
                            * 100 * direction, 3)

        if r["pnl"] is not None:
            cum_pnl += r["pnl"] + r["swap"]

        sheet_rows.append([
            r["id"], _to_et(r["entry_time"]), r["symbol"], r["side"], r["score"],
            round(r["qty"], 2), r["entry_price"], r["tp"], r["sl"], r["status_kr"],
            r["exit_price"], _to_et(r["exit_time"]), hold_minutes,
            round(r["pnl"], 2) if r["pnl"] is not None else "",
            pnl_pct,
            round(r["swap"], 2) if (r["pnl"] is not None and r["swap"]) else "",
            round(cum_pnl, 2) if r["pnl"] is not None else "",
        ])

    try:
        ws.clear()
        try:
            ws.resize(rows=max(len(sheet_rows) + 50, 200), cols=len(HEADERS))
        except Exception as e:
            print(f"⚠️ [OANDA거래내역] 크기 조정 실패: {e}")
        ws.update("A1", sheet_rows)
        print(f"✅ [OANDA거래내역] {len(rows)}건 갱신 완료")
    except Exception as e:
        print(f"❌ [OANDA거래내역] 시트 쓰기 실패: {e}")
        return

    # 🟥 [FIX-D1] 날짜 구분선
    _draw_date_separators(spreadsheet, ws, sheet_rows, date_col=1,
                          ncols=len(HEADERS), tag="OANDA구분선")


def sync_symbol_performance_summary():
    """
    'Alpaca 거래내역'(체결/손익 진실 데이터) + 메인 시트(전체 알림 빈도)를 합쳐서
    종목별 승률/손익/빈도를 정리한 '종목별 성과분석' 탭을 만든다.
    탭이 없으면 자동 생성, 매번 전체 재계산해서 덮어쓴다.
    승률·총손익 기준으로 정렬해서, 어떤 종목이 좋고 어떤 종목을 빼야 할지 한눈에 보이게 한다.
    """
    HEADERS = [
        "종목", "알림 빈도(전체)", "체결 건수", "체결비율(%)",
        "승(TP)", "패(SL)", "승률(%)", "총손익($)", "평균손익($)",
        "평균보유시간(분)", "평가"
    ]

    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name("/etc/secrets/google_credentials.json", scope)
        client = gspread.authorize(creds)
        spreadsheet = client.open("민균 FX trading result")

        try:
            trade_ws = spreadsheet.worksheet("Alpaca 거래내역")
            trade_rows = trade_ws.get_all_values()
        except gspread.exceptions.WorksheetNotFound:
            print("⚠️ [종목별성과] 'Alpaca 거래내역' 탭이 아직 없음 → sync_alpaca_trade_log()를 먼저 실행해야 함")
            return

        main_ws = spreadsheet.sheet1
        main_rows = main_ws.get_all_values()

        try:
            summary_ws = spreadsheet.worksheet("종목별 성과분석")
        except gspread.exceptions.WorksheetNotFound:
            summary_ws = spreadsheet.add_worksheet(title="종목별 성과분석", rows=200, cols=len(HEADERS))
            print("✅ [종목별성과] 탭이 없어서 새로 생성했습니다.")
    except Exception as e:
        print(f"❌ [종목별성과] 시트 연결 실패: {e}")
        return

    # 1) 메인 시트에서 종목별 전체 알림 빈도 집계 (실행 여부 무관, 그냥 알림이 몇 번 왔는지)
    freq = {}
    for row in main_rows[1:]:
        if len(row) > 1 and row[1]:
            freq[row[1]] = freq.get(row[1], 0) + 1

    # 2) 'Alpaca 거래내역' 탭에서 종목별 승/패/손익 집계
    #    헤더: 주문ID,진입시각,종목,방향,점수,수량,진입가,TP가,SL가,상태,청산가,청산시각,보유시간(분),손익($),손익(%),누적손익($)
    stats = {}  # symbol -> {tp, sl, pnl_list, hold_list}
    for row in trade_rows[1:]:
        if len(row) < 14:
            continue
        symbol = row[2]
        status_kr = row[9]
        pnl_str = row[13]
        hold_str = row[12]
        if not symbol or status_kr not in ("TP청산", "SL청산", "TIME_EXIT"):
            continue
        s = stats.setdefault(symbol, {"tp": 0, "sl": 0, "pnl_list": [], "hold_list": []})
        # 🟦 TIME_EXIT(시간초과 강제청산)은 TP/SL 어느 쪽도 아니라서, 실현손익 부호로 승/패를 나눈다.
        try:
            _pnl_for_winloss = float(pnl_str)
        except Exception:
            _pnl_for_winloss = None
        is_win = (status_kr == "TP청산") or (status_kr == "TIME_EXIT" and _pnl_for_winloss is not None and _pnl_for_winloss > 0)
        if is_win:
            s["tp"] += 1
        else:
            s["sl"] += 1
        try:
            s["pnl_list"].append(float(pnl_str))
        except Exception:
            pass
        try:
            s["hold_list"].append(float(hold_str))
        except Exception:
            pass

    # 🟦 포트폴리오에서 제거된 종목은 성과분석 탭에서도 자동으로 빠지도록:
    #    "현재 활성 종목" = 최근 30일 이내 메인 시트에 알림이 있는 종목만 포함.
    #    (제거된 종목의 과거 데이터는 'Alpaca 거래내역' 탭에 남아있지만, 이 탭에서는 안 보이게)
    cutoff_30d = datetime.now(_tz.utc) - timedelta(days=30)
    active_symbols = set()
    for row in main_rows[1:]:
        if len(row) < 2 or not row[1]:
            continue
        try:
            ts = datetime.fromisoformat(row[0])
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=_tz.utc)
            if ts >= cutoff_30d:
                active_symbols.add(row[1])
        except Exception:
            pass
    all_symbols = sorted(active_symbols | set(stats.keys()))
    summary_rows = [HEADERS]
    computed = []
    for sym in all_symbols:
        s = stats.get(sym, {"tp": 0, "sl": 0, "pnl_list": [], "hold_list": []})
        trades = s["tp"] + s["sl"]
        win_rate = round(s["tp"] / trades * 100, 1) if trades else ""
        total_pnl = round(sum(s["pnl_list"]), 2) if s["pnl_list"] else ""
        avg_pnl = round(sum(s["pnl_list"]) / len(s["pnl_list"]), 2) if s["pnl_list"] else ""
        avg_hold = round(sum(s["hold_list"]) / len(s["hold_list"]), 1) if s["hold_list"] else ""
        signal_count = freq.get(sym, 0)
        fill_rate = round(trades / signal_count * 100, 1) if signal_count else ""

        # 🟥 [FIX-D7] 자동 평가 코멘트 로직 수정.
        #    기존엔 total_pnl > 0 이기만 하면 "✅ 양호 — 유지 후보"였다. 그래서 18거래에
        #    총손익 $0.04인 PLTR이 "유지 후보"로 찍혔다 — 사실상 본전인데 양호로 오독된다.
        #    → 부호가 아니라 "거래당 평균 손익"과 "손익분기 필요 승률"을 기준으로 판정한다.
        _breakeven_wr = 100.0 / (1.0 + STOCK_RR_RATIO)   # 예: RR 1.6 → 38.5%
        _min_trades = int(os.getenv("SYMBOL_VERDICT_MIN_TRADES", "10"))
        _avg = avg_pnl if isinstance(avg_pnl, (int, float)) else None
        _wr = win_rate if isinstance(win_rate, (int, float)) else None
        if trades < _min_trades:
            verdict = f"표본 부족({trades}건/{_min_trades}건) — 판단 보류"
        elif _avg is None or _wr is None:
            verdict = "🟡 데이터 부족 — 추가 관찰"
        elif _avg > 1.0 and _wr >= _breakeven_wr:
            verdict = f"✅ 양호 — 유지 후보 (평균 ${_avg}/거래, 손익분기 {_breakeven_wr:.1f}%)"
        elif _avg < -1.0 or _wr < _breakeven_wr - 5:
            verdict = f"❌ 부진 — 제외 검토 (평균 ${_avg}/거래, 손익분기 {_breakeven_wr:.1f}%)"
        else:
            verdict = f"🟡 본전권 — 추가 관찰 (평균 ${_avg}/거래)"

        computed.append([
            sym, signal_count, trades, fill_rate,
            s["tp"], s["sl"], win_rate, total_pnl, avg_pnl,
            avg_hold, verdict
        ])

    # 총손익 내림차순 정렬 (숫자 아닌 건 맨 뒤로)
    computed.sort(key=lambda r: r[7] if isinstance(r[7], (int, float)) else -1e18, reverse=True)
    summary_rows.extend(computed)

    try:
        summary_ws.clear()
        summary_ws.update("A1", summary_rows)
        print(f"✅ [종목별성과] {len(computed)}개 종목 갱신 완료")
    except Exception as e:
        print(f"❌ [종목별성과] 시트 쓰기 실패: {e}")


# ============================================================
# 🟥 [FIX-G10] 아침 갭 스캐너 — '오늘 아침 스캔 결과' 탭
# ------------------------------------------------------------
#  장 시작 전에 5% 이상 갭상승한 종목을 찾아, 아래 5개 조건으로 걸러 상위 3개를 뽑는다.
#    ① 어제 고가 위
#    ② 200일 이동평균 위
#    ③ 장전(프리마켓) 고가 유지 — 고점에서 밀리지 않았나
#    ④ 당일 정규장 고가 유지 — (개장 전에는 해당 없음)
#    ⑤ 상승 추세 정렬: 현재가 > 50일선 > 200일선
#
#  ⚠️ 갭 종목은 페니주·저유동성 종목이 대부분이다. 예전에 이 봇의 스캐너가
#     SOXS·GPUS·ZBAO 같은 것들을 계속 공급했던 것과 같은 함정이다.
#     그래서 가격·거래량·레버리지ETF 필터를 먼저 통과시킨 뒤에 조건을 본다.
#
#  ⚠️ 조건 ③④ 는 '시각'에 따라 뜻이 달라진다.
#     개장 전에는 정규장 고가가 존재하지 않으므로 ④는 '해당없음' 으로 둔다.
#     그래서 스캔을 두 번 돌린다 — 개장 전(09:20)과 개장 후(10:00).
# ============================================================

GAP_MIN_PCT = float(os.getenv("GAP_MIN_PCT", "5.0"))          # 최소 갭 %
GAP_TOP_N = int(os.getenv("GAP_TOP_N", "3"))                  # 최종 추천 개수
GAP_UNIVERSE_TOP = int(os.getenv("GAP_UNIVERSE_TOP", "50"))   # 상승률 상위 몇 개를 볼지
GAP_MIN_PRICE = float(os.getenv("GAP_MIN_PRICE", "5.0"))      # 최소 주가
# 🟥 [G11] 무료(IEX) 피드는 IEX 한 거래소 체결분만 세므로 실제 거래량의 5~10% 로 나온다.
#    50만은 '전체 거래량' 기준 숫자였다 — 그대로 쓰면 멀쩡한 종목이 전부 잘린다.
#    (실측: SNAP 이 2.7M 으로 찍힘. 실제는 2천만 주 이상)
GAP_MIN_AVG_VOL = float(os.getenv("GAP_MIN_AVG_VOL", "50000"))   # IEX 기준 ≈ 전체 50만

# 🟥 [G12] '5개 전부 통과' 만 추천하면 추천이 0개인 날이 대부분이다.
#    조건 5개는 서로 독립이 아니다 — ②200일선과 ⑤추세정렬은 사실상 같은 것이고,
#    ③장전고가와 ④당일고가도 같은 것(고점에서 안 밀렸나)을 두 번 본다.
#    즉 실질적으로는 2~3개짜리 조건을 5개인 것처럼 세고 있었다.
#    → 전부 통과(1군)와 1개 미달(2군)을 나눠서 둘 다 보여주고,
#      관찰 기간 동안 '1군이 정말 2군보다 나은가' 를 데이터로 확인한다.
GAP_TIER2_ENABLED = os.getenv("GAP_TIER2_ENABLED", "true").strip().lower() != "false"
GAP_HIGH_TOL_PCT = float(os.getenv("GAP_HIGH_TOL_PCT", "0.5"))   # '고가 유지' 허용 오차 %
# 🟥 [G1/G2] 실행 시각을 개장 후로 옮겼다.
#   ① Alpaca movers 는 '개장 시점에 리셋' 된다 — 09:20 에 부르면 **어제 상승률 상위**가
#      돌아온다. 오늘의 갭 종목이 아니다.
#   ② 무료(Basic) 플랜은 '최근 15분' 데이터를 안 준다. 개장 직전/직후를 조회하면 빈 응답이다.
#   → movers 가 리셋되는 09:30 직후가 가장 이른 시각이다. 09:32 로 잡는다.
#     (2분은 첫 체결이 찍히고 movers 가 갱신될 여유)
#     09:45 / 10:15 에 다시 돌려 밀린 종목을 걸러낸다.
#   ⚠️ 09:30 '이전' 스캔은 무료(IEX) 로는 신뢰할 수 없다. IEX 는 프리마켓 체결이
#      거의 없어 갭을 제대로 못 잰다. 진짜 개장 전 스캔은 SIP(유료) 가 있어야 한다.
# 🟥 [G13] 개장 전 스캔을 기본에 넣는다.
#   09:32 에 목록을 받으면 알람을 거는 데 2~3분이 더 걸리고, 그때는 이미
#   갭 무브의 앞부분이 지나간 뒤다. 장전 고가는 개장 전에 이미 확정되므로
#   09:25 에 목록을 뽑아 **개장 전에 알람을 미리 걸어둘 수 있다.**
#     0915 = 미리보기(장전 고가가 아직 움직임)
#     0925 = ★ 실전 목록. 이걸 보고 알람을 건다
#     0935/1000 = 개장 후 재확인(④당일고가까지 포함한 전체 판정)
GAP_SCAN_TIMES = os.getenv("GAP_SCAN_TIMES", "0915,0925,0935,1000")   # ET 기준

#: 무료 플랜은 iex. 유료면 ALPACA_FEED=sip 로 바꿀 것 (프리마켓 데이터 품질이 크게 달라진다)
ALPACA_FEED = os.getenv("ALPACA_FEED", "iex")
# 🟥 [G9] '15분 지연' 을 추측하지 말고 **직접 재본다**.
#   자료가 엇갈린다:
#     · Alpaca 공식 FAQ — "end 는 15분 이전이어야 한다" 는 **SIP 에만** 적용
#     · 제3자 요약     — "무료는 REST 전체가 15분 지연"
#   추측으로 정하면 틀렸을 때 매일 조용히 0건이 나온다.
#   → 유동성 큰 종목의 최근 분봉을 실제로 받아보고 지연을 측정한다. 1시간 캐시.
ALPACA_DELAY_MIN = os.getenv("ALPACA_DELAY_MIN", "auto")   # "auto" 또는 분 단위 숫자
_delay_probe = {"t": 0.0, "minutes": None}
DELAY_PROBE_TTL = float(os.getenv("DELAY_PROBE_TTL", "3600"))


def measure_data_delay(symbol: str = "SPY") -> int:
    """이 계정의 실제 데이터 지연(분). 판정 불가면 보수적으로 16."""
    if str(ALPACA_DELAY_MIN).strip().isdigit():
        return int(ALPACA_DELAY_MIN)
    now = _t.time()
    if _delay_probe["minutes"] is not None and (now - _delay_probe["t"]) < DELAY_PROBE_TTL:
        return _delay_probe["minutes"]

    delay = 16
    try:
        start = (datetime.now(_tz.utc) - timedelta(minutes=90)).strftime("%Y-%m-%dT%H:%M:%SZ")
        r = requests.get(f"{ALPACA_DATA_BASE_URL}/v2/stocks/{symbol}/bars",
                         headers=ALPACA_HEADERS,
                         params={"timeframe": "1Min", "start": start,
                                 "feed": ALPACA_FEED, "limit": 200}, timeout=15)
        if r.ok:
            bars = (r.json() or {}).get("bars") or []
            if bars:
                last = datetime.fromisoformat(bars[-1]["t"].replace("Z", "+00:00"))
                gap_min = (datetime.now(_tz.utc) - last).total_seconds() / 60.0
                if gap_min <= 30:
                    delay = max(0, int(gap_min) - 1)   # 마지막 봉이 아직 안 닫힌 만큼 제외
                    print(f"📏 [데이터지연] {symbol} 마지막 봉 {gap_min:.1f}분 전 "
                          f"→ 지연 {delay}분 (feed={ALPACA_FEED})")
                else:
                    print(f"📏 [데이터지연] 장중이 아니라 측정 보류 "
                          f"(마지막 봉 {gap_min:.0f}분 전) → 16분 적용")
            else:
                print(f"📏 [데이터지연] {symbol} 분봉 비어있음 → 16분 적용")
        else:
            print(f"📏 [데이터지연] status={r.status_code} → 16분 적용")
    except Exception as e:
        print(f"📏 [데이터지연] 측정 실패({e}) → 16분 적용")

    _delay_probe.update(t=now, minutes=delay)
    return delay


def _data_end_iso() -> str:
    """조회 종료 시각. 측정된 지연만큼만 뒤로 물린다 (지연 0이면 지금까지)."""
    d = measure_data_delay()
    if d <= 0:
        return datetime.now(_tz.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return (datetime.now(_tz.utc) - timedelta(minutes=d)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _alpaca_movers(top: int = 50) -> list[dict]:
    """상승률 상위 종목. 개장 전에는 비어 있을 수 있어서 most-actives 로 보완한다."""
    out = []
    try:
        r = requests.get(f"{ALPACA_DATA_BASE_URL}/v1beta1/screener/stocks/movers",
                         headers=ALPACA_HEADERS, params={"top": min(top, 50)}, timeout=15)
        if r.ok:
            for g in (r.json() or {}).get("gainers", []):
                if g.get("symbol"):
                    out.append({"symbol": g["symbol"], "src": "상승률상위"})
        else:
            # 🟥 [G5] top 상한(50) 초과 등으로 400 이 나면 예전엔 조용히 0개가 됐다
            print(f"⚠️ [갭스캔] movers status={r.status_code} {r.text[:150]}")
    except Exception as e:
        print(f"⚠️ [갭스캔] movers 조회 실패: {e}")
    try:
        r = requests.get(f"{ALPACA_DATA_BASE_URL}/v1beta1/screener/stocks/most-actives",
                         headers=ALPACA_HEADERS, params={"by": "volume", "top": min(top, 100)}, timeout=15)
        if r.ok:
            j = r.json() or {}
            for a in (j.get("most_actives") or j.get("mostActives") or []):
                if a.get("symbol"):
                    out.append({"symbol": a["symbol"], "src": "거래량상위"})
        else:
            print(f"⚠️ [갭스캔] most-actives status={r.status_code} {r.text[:150]}")
    except Exception as e:
        print(f"⚠️ [갭스캔] most-actives 조회 실패: {e}")

    seen, uniq = set(), []
    for x in out:
        if x["symbol"] not in seen:
            seen.add(x["symbol"]); uniq.append(x)
    return uniq


def _alpaca_snapshots(symbols: list[str]) -> dict:
    """여러 종목의 현재가/오늘봉/어제봉을 한 번에. {symbol: snapshot}"""
    if not symbols:
        return {}
    out = {}
    for i in range(0, len(symbols), 100):          # 한 번에 100개씩
        chunk = symbols[i:i + 100]
        try:
            r = requests.get(f"{ALPACA_DATA_BASE_URL}/v2/stocks/snapshots",
                             headers=ALPACA_HEADERS,
                             params={"symbols": ",".join(chunk)}, timeout=20)
            if r.ok:
                out.update(r.json() or {})
            else:
                print(f"⚠️ [갭스캔] snapshots status={r.status_code}")
        except Exception as e:
            print(f"⚠️ [갭스캔] snapshots 실패: {e}")
    return out


def _alpaca_daily_bars(symbols: list[str], limit: int = 260) -> dict:
    """여러 종목의 일봉. {symbol: [{c,h,l,v}, ...]} — 오름차순"""
    if not symbols:
        return {}
    out: dict[str, list] = {}
    start = (datetime.now(ZoneInfo("America/New_York")) - timedelta(days=420)).strftime("%Y-%m-%d")
    for i in range(0, len(symbols), 40):
        chunk = symbols[i:i + 40]
        page = None
        try:
            while True:
                params = {"symbols": ",".join(chunk), "timeframe": "1Day",
                          "start": start, "limit": 10000, "adjustment": "split",
                          "feed": ALPACA_FEED, "end": _data_end_iso()}
                if page:
                    params["page_token"] = page
                r = requests.get(f"{ALPACA_DATA_BASE_URL}/v2/stocks/bars",
                                 headers=ALPACA_HEADERS, params=params, timeout=25)
                if not r.ok:
                    print(f"⚠️ [갭스캔] 일봉 status={r.status_code}")
                    break
                j = r.json() or {}
                for sym, bars in (j.get("bars") or {}).items():
                    # 🟥 [G4] 데이터 없는 심볼은 bars 가 None 이다.
                    #   extend(None) 은 TypeError → except 로 빠져 40개 청크가 통째로 사라졌다.
                    out.setdefault(sym, []).extend(bars or [])
                page = j.get("next_page_token")
                if not page:
                    break
        except Exception as e:
            print(f"⚠️ [갭스캔] 일봉 조회 실패: {e}")
    for sym in out:
        out[sym] = out[sym][-limit:]
    return out


def _premarket_and_regular_high(symbol: str) -> tuple[float | None, float | None]:
    """오늘의 (프리마켓 고가, 정규장 고가). 데이터 없으면 None."""
    ny = ZoneInfo("America/New_York")
    now_ny = datetime.now(ny)
    # 🟥 [G8] 04:00 ET 이전에 부르면 시작시각이 미래라 항상 빈 응답이 온다.
    #   그게 '조건 미달' 로 보이면 원인을 못 찾는다. 명시적으로 끊는다.
    if now_ny.hour < 4:
        print(f"⚠️ [갭스캔] {symbol} — 04:00 ET 이전이라 당일 분봉이 없다")
        return None, None
    day0 = now_ny.replace(hour=4, minute=0, second=0, microsecond=0)
    try:
        r = requests.get(f"{ALPACA_DATA_BASE_URL}/v2/stocks/{symbol}/bars",
                         headers=ALPACA_HEADERS,
                         params={"timeframe": "1Min",
                                 "start": day0.astimezone(_tz.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                                 "end": _data_end_iso(), "feed": ALPACA_FEED,
                                 "limit": 10000, "adjustment": "raw"}, timeout=20)
        if not r.ok:
            # 🟥 [G1] 예전엔 조용히 None 을 돌려줘서 '조건 미달' 로 보였다
            print(f"⚠️ [갭스캔] {symbol} 분봉 status={r.status_code} {r.text[:150]}")
            return None, None
        bars = (r.json() or {}).get("bars") or []
    except Exception as e:
        print(f"⚠️ [갭스캔] {symbol} 분봉 실패: {e}")
        return None, None

    pm_high = reg_high = None
    for b in bars:
        try:
            t = datetime.fromisoformat(b["t"].replace("Z", "+00:00")).astimezone(ny)
        except Exception:
            continue
        hhmm = t.hour * 100 + t.minute
        h = float(b["h"])
        if hhmm < 930:
            pm_high = h if pm_high is None else max(pm_high, h)
        elif hhmm < 1600:
            reg_high = h if reg_high is None else max(reg_high, h)
    return pm_high, reg_high


def _sma(values: list[float], n: int) -> float | None:
    return sum(values[-n:]) / n if len(values) >= n else None


def _gap_reason(symbol: str) -> str:
    """왜 올랐는지 — 최근 24시간 뉴스 제목. 갭은 대개 뉴스가 원인이다."""
    try:
        end = datetime.now(_tz.utc)
        r = requests.get("https://data.alpaca.markets/v1beta1/news",
                         headers=ALPACA_HEADERS,
                         params={"symbols": symbol,
                                 "start": (end - timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                                 "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                 "limit": 3, "sort": "desc"}, timeout=10)
        if not r.ok:
            return "뉴스 조회 실패"
        arts = (r.json() or {}).get("news", [])
        if not arts:
            return "⚠ 뉴스 없음 — 이유 불명 (수급만으로 오른 것일 수 있음)"
        return " / ".join((a.get("headline") or "")[:90] for a in arts[:2])
    except Exception:
        return "뉴스 조회 실패"


def scan_morning_gappers(top_n: int = None) -> dict:
    """
    갭 스캔 본체. '오늘 아침 스캔 결과' 탭을 통째로 새로 쓴다.
    return: 요약 dict (엔드포인트 응답용)
    """
    top_n = top_n or GAP_TOP_N
    ny = ZoneInfo("America/New_York")
    now_ny = datetime.now(ny)
    hhmm = now_ny.hour * 100 + now_ny.minute
    pre_open = hhmm < 930
    phase = "개장 전" if pre_open else "개장 후"

    print(f"🔎 [갭스캔] 시작 — {now_ny:%Y-%m-%d %H:%M} ET ({phase})")

    # ── 1) 후보 유니버스 ──────────────────────────────────
    universe = _alpaca_movers(GAP_UNIVERSE_TOP)
    if not universe:
        print("❌ [갭스캔] 후보를 하나도 못 받았다")
        return {"ok": False, "reason": "no_universe"}
    src_map = {u["symbol"]: u["src"] for u in universe}
    snaps = _alpaca_snapshots([u["symbol"] for u in universe])

    # ── 2) 갭 계산 + 1차 필터 ────────────────────────────
    cands = []
    skipped = {"블록": 0, "저가": 0, "갭부족": 0, "데이터없음": 0}
    for sym, sn in snaps.items():
        prev = (sn or {}).get("prevDailyBar") or {}
        last = (sn or {}).get("latestTrade") or {}
        day = (sn or {}).get("dailyBar") or {}
        # 🟥 [G3] dailyBar 는 당일 첫 체결이 찍히기 전까지 '어제 봉' 이다.
        #   그 상태에서 prevDailyBar 를 쓰면 이틀 전 종가와 비교하게 되어
        #   오늘 갭이 아니라 어제 움직임을 재게 된다. 날짜 도장을 확인한다.
        _today = now_ny.date().isoformat()
        _day_is_today = str(day.get("t") or "")[:10] == _today
        _ref = prev if _day_is_today else day
        prev_c = float(_ref.get("c") or 0)
        prev_h = float(_ref.get("h") or 0)
        if not _day_is_today:
            skipped.setdefault("당일봉없음", 0)
            skipped["당일봉없음"] += 1
        px = float(last.get("p") or day.get("c") or 0)
        if prev_c <= 0 or px <= 0:
            skipped["데이터없음"] += 1
            continue
        gap = (px - prev_c) / prev_c * 100.0
        if gap < GAP_MIN_PCT:
            skipped["갭부족"] += 1
            continue
        if px < GAP_MIN_PRICE:
            skipped["저가"] += 1
            continue
        blocked, why = is_blocked_instrument(sym, px)
        if blocked:
            skipped["블록"] += 1
            continue
        cands.append({"symbol": sym, "price": px, "gap": gap,
                      "prev_close": prev_c, "prev_high": prev_h,
                      "src": src_map.get(sym, "")})

    print(f"🔎 [갭스캔] 갭 {GAP_MIN_PCT}%+ 통과 {len(cands)}개 "
          f"(제외: {skipped})")
    if not cands:
        _write_gap_sheet([], now_ny, phase, skipped, 0, top_n)
        return {"ok": True, "candidates": 0, "phase": phase}

    cands.sort(key=lambda c: -c["gap"])
    cands = cands[:25]                       # 분봉·뉴스 호출을 아끼려 상위 25개만

    # ── 3) 일봉으로 이동평균·평균거래량 ──────────────────
    daily = _alpaca_daily_bars([c["symbol"] for c in cands])
    for c in cands:
        bars = daily.get(c["symbol"], [])
        closes = [float(b["c"]) for b in bars]
        vols = [float(b.get("v", 0)) for b in bars]
        c["sma50"] = _sma(closes, 50)
        c["sma200"] = _sma(closes, 200)
        c["avg_vol"] = _sma(vols, 20)
        c["bars"] = len(closes)

    # ── 4) 프리마켓/정규장 고가 ──────────────────────────
    for c in cands:
        pm, reg = _premarket_and_regular_high(c["symbol"])
        c["pm_high"] = pm
        c["reg_high"] = reg

    # ── 5) 5개 조건 판정 ─────────────────────────────────
    tol = 1.0 - GAP_HIGH_TOL_PCT / 100.0
    for c in cands:
        px = c["price"]
        c1 = c["prev_high"] > 0 and px > c["prev_high"]
        c2 = c["sma200"] is not None and px > c["sma200"]
        c3 = c["pm_high"] is not None and px >= c["pm_high"] * tol
        if pre_open:
            c4 = None                        # 정규장 고가가 아직 없음
        else:
            c4 = c["reg_high"] is not None and px >= c["reg_high"] * tol
        c5 = (c["sma50"] is not None and c["sma200"] is not None
              and px > c["sma50"] > c["sma200"])
        liq = c["avg_vol"] is not None and c["avg_vol"] >= GAP_MIN_AVG_VOL

        checks = [c1, c2, c3, c4, c5]
        passed = sum(1 for x in checks if x is True)
        applicable = sum(1 for x in checks if x is not None)
        c.update(c1=c1, c2=c2, c3=c3, c4=c4, c5=c5, liq=liq,
                 passed=passed, applicable=applicable,
                 all_ok=(passed == applicable and liq))

    # 전부 통과 → 갭 큰 순. 통과 개수 → 갭 순
    cands.sort(key=lambda c: (-c["passed"], -c["gap"]))
    picks = [c for c in cands if c["all_ok"]][:top_n]

    # ── 6) 오른 이유 (뉴스) — 상위 후보만 ─────────────────
    # 🟥 [G7] cands 는 passed 순 정렬이라, 유동성까지 통과한 종목이 10위 밖에
    #   있을 수 있다. 그러면 추천에 올라가고도 '오른 이유' 가 빈칸이 된다.
    #   추천 목록은 무조건 포함시킨다.
    # 🟥 [G12] 2군(조건 1개 미달 + 유동성 통과)도 시트에 실리므로 뉴스가 필요하다.
    #    cands 정렬 키에 liq 가 없어서, 유동성 미달 종목이 앞을 다 차지하면
    #    2군이 cands[:10] 밖으로 밀려 '오른 이유' 가 빈칸이 된다 (G7 과 같은 함정).
    _tier2_pre = [c for c in cands
                  if c.get("liq") and not c.get("all_ok")
                  and c.get("passed", 0) >= c.get("applicable", 5) - 1][:GAP_TOP_N]
    _need_reason = {c["symbol"]: c for c in (picks + _tier2_pre + cands[:10])}
    for c in _need_reason.values():
        c["reason"] = _gap_reason(c["symbol"])

    _write_gap_sheet(cands, now_ny, phase, skipped, len(picks), top_n)
    print(f"✅ [갭스캔] 완료 — 후보 {len(cands)}개, 전조건 통과 {len(picks)}개")
    return {"ok": True, "phase": phase, "candidates": len(cands),
            "picks": [p["symbol"] for p in picks]}


def _fmt_ck(v):
    return "—" if v is None else ("✅" if v else "❌")


def _write_gap_sheet(cands: list, now_ny, phase: str, skipped: dict, n_pick: int,
                     top_n: int = None):
    HEADERS = ["종목", "현재가", "갭 %", "돌파선(장전고가)", "오른 이유",
               "①어제고가", "②200일선", "③장전고가", "④당일고가", "⑤추세정렬",
               "유동성", "통과", "판정", "출처"]
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name(
            "/etc/secrets/google_credentials.json", scope)
        client = gspread.authorize(creds)
        ss = client.open(GOOGLE_SHEET_NAME)
        try:
            ws = ss.worksheet("오늘 아침 스캔 결과")
        except gspread.exceptions.WorksheetNotFound:
            ws = ss.add_worksheet(title="오늘 아침 스캔 결과", rows=200, cols=len(HEADERS))
            print("✅ [갭스캔] 탭 생성")
    except Exception as e:
        print(f"❌ [갭스캔] 시트 연결 실패: {e}")
        return

    out = [[f"오늘 아침 스캔 결과   {now_ny:%Y-%m-%d %H:%M} ET   ({phase})"]]
    out.append([f"갭 {GAP_MIN_PCT:g}% 이상 · 주가 ${GAP_MIN_PRICE:g} 이상 · "
                f"20일 평균거래량 {GAP_MIN_AVG_VOL:,.0f} 이상 · 레버리지ETF/페니주 제외"])
    out.append([f"제외된 것: {skipped}"])
    out.append([])

    # 🟥 [G6] 예전엔 여기서 GAP_TOP_N 으로 다시 잘라, ?top=5 로 불러도 시트는 3개만 썼다
    picks = [c for c in cands if c.get("all_ok")][:(top_n or GAP_TOP_N)]
    out.append([f"■ 1군 추천 {len(picks)}개 (5개 조건 전부 통과)"])
    if picks:
        out.append(HEADERS)
        for c in picks:
            out.append(_gap_row(c))
    else:
        out.append(["전부 통과한 종목이 없습니다. 아래 2군을 보세요."])
    out.append([])

    # 🟥 [G12] 2군 — 유동성은 통과했고 조건만 딱 1개 모자란 것들
    if GAP_TIER2_ENABLED:
        _picked = {c["symbol"] for c in picks}
        tier2 = [c for c in cands
                 if c.get("liq") and not c.get("all_ok")
                 and c.get("passed", 0) >= c.get("applicable", 5) - 1
                 and c["symbol"] not in _picked][:(top_n or GAP_TOP_N)]
        out.append([f"■ 2군 관찰 {len(tier2)}개 (유동성 통과 · 조건 1개 미달)"])
        if tier2:
            out.append(HEADERS)
            for c in tier2:
                out.append(_gap_row(c))
        else:
            out.append(["해당 없음"])
        out.append([])

    out.append(["■ 전체 후보 (통과 개수 순)"])
    out.append(HEADERS)
    for c in cands:
        out.append(_gap_row(c))

    out.append([])
    out.append(["※ ③장전고가/④당일고가 는 '고점에서 밀리지 않았나' 를 봅니다 "
                f"(허용오차 {GAP_HIGH_TOL_PCT:g}%). 개장 전에는 ④가 '—' 입니다."])
    if phase == "개장 전":
        out.append(["★ 개장 전 목록입니다. ④당일고가는 아직 존재하지 않아 4개 조건으로만 판정했습니다."])
        out.append(["★ '돌파선(장전고가)' 이 GAP AND GO G2 의 진입 트리거 가격입니다. "
                    "개장 전에 이미 확정된 값이라, 지금 알람을 걸어두면 09:35 에 바로 진입할 수 있습니다."])
        out.append(["★ 알람: 종목 차트 5분봉 → GAP AND GO G2 → alert() function calls only "
                    "→ Webhook URL 은 반드시 '/webhook'"])
    out.append(["※ ⑤추세정렬 = 현재가 > 50일선 > 200일선"])
    out.append(["※ 갭 종목은 뉴스로 움직입니다. '오른 이유' 가 비어 있으면 특히 조심하세요."])
    out.append(["※ 조건 5개는 서로 독립이 아닙니다. ②200일선↔⑤추세정렬, ③장전고가↔④당일고가 가 "
                "각각 거의 같은 것을 봅니다 — 실질 조건 수는 2~3개입니다."])
    out.append(["※ 관찰 기간에는 1군·2군을 모두 기록하세요. 둘의 성적 차이가 실제로 있는지는 "
                "표본이 쌓여야 알 수 있습니다. 지금 '몇 개 통과면 된다' 는 답은 아무도 모릅니다."])

    width = max(len(r) for r in out)
    out = [r + [""] * (width - len(r)) for r in out]
    try:
        ws.clear()
        try:
            ws.resize(rows=max(len(out) + 20, 60), cols=max(width, len(HEADERS)))
        except Exception as e:
            print(f"⚠️ [갭스캔] 시트 크기 조정 실패: {e}")
        ws.update("A1", out)
        print(f"✅ [갭스캔] 시트 기록 완료 — 후보 {len(cands)} / 추천 {len(picks)}")
    except Exception as e:
        print(f"❌ [갭스캔] 시트 쓰기 실패: {e}")


def _gap_row(c: dict) -> list:
    if c.get("all_ok"):
        verdict = "⭐ 전조건 통과"
    elif not c.get("liq"):
        verdict = "⛔ 유동성 부족"
    elif c.get("passed", 0) >= c.get("applicable", 5) - 1:
        verdict = "🟡 1개 미달"
    else:
        verdict = "❌ 조건 미달"
    _pmh = c.get("pm_high")
    return [
        c["symbol"],
        round(c["price"], 2),
        f"{c['gap']:.1f}%",
        # 🟥 [G13] G2 전략의 진입 트리거 가격. 개장 전에 이미 확정돼 있다.
        round(_pmh, 2) if _pmh else "—",
        c.get("reason", ""),
        _fmt_ck(c.get("c1")), _fmt_ck(c.get("c2")), _fmt_ck(c.get("c3")),
        _fmt_ck(c.get("c4")), _fmt_ck(c.get("c5")),
        f"{c.get('avg_vol') or 0:,.0f}",
        f"{c.get('passed', 0)}/{c.get('applicable', 5)}",
        verdict,
        c.get("src", ""),
    ]


async def _morning_gap_scan_loop():
    """GAP_SCAN_TIMES(ET) 마다 스캔. 기본 09:20(개장 전) / 10:00(개장 후)."""
    times = []
    for t in GAP_SCAN_TIMES.split(","):
        t = t.strip()
        if t.isdigit() and len(t) == 4:
            times.append((int(t[:2]), int(t[2:])))
    if not times:
        times = [(9, 20), (10, 0)]
    print(f"🕘 [갭스캔] 예약 시각(ET): {['%02d:%02d' % t for t in times]}")

    while True:
        ny = ZoneInfo("America/New_York")
        now = datetime.now(ny)
        nxt = None
        for h, m in times:
            cand = now.replace(hour=h, minute=m, second=0, microsecond=0)
            if cand <= now:
                cand += timedelta(days=1)
            if nxt is None or cand < nxt:
                nxt = cand
        await asyncio.sleep(max(30, (nxt - now).total_seconds()))
        try:
            # 주말은 건너뛴다
            if datetime.now(ZoneInfo("America/New_York")).weekday() < 5:
                await asyncio.to_thread(scan_morning_gappers)
        except Exception as e:
            print(f"❌ [갭스캔 루프] {e}")
        await asyncio.sleep(90)


def _oanda_price_quick(instrument: str):
    """OANDA 현재가(중간값). 추천 후보 표시용. 실패 시 None."""
    if not (OANDA_API_KEY and ACCOUNT_ID):
        return None
    try:
        r = requests.get(
            f"{OANDA_BASE_URL}/v3/accounts/{ACCOUNT_ID}/pricing",
            headers={"Authorization": f"Bearer {OANDA_API_KEY}"},
            params={"instruments": instrument}, timeout=10,
        )
        if not r.ok:
            return None
        p = (r.json() or {}).get("prices", [{}])[0]
        bids, asks = p.get("bids") or [], p.get("asks") or []
        if not bids or not asks:
            return None
        return round((float(bids[0]["price"]) + float(asks[0]["price"])) / 2, 5)
    except Exception:
        return None


# ------------------------------------------------------------
# 🟥 [FIX-P1] 분산 후보 유니버스
#   ★ '거래량 상위'는 분산에 아무 도움이 안 된다. 거래량 상위는 그날 화제인 종목이고,
#     화제인 종목은 대개 이미 들고 있는 것과 같은 테마다. (실제로 이 스캐너는
#     페니주와 3배 레버리지 ETF를 계속 공급했다)
#   → 그룹별로 '유동성 있고 그 분야를 대표하는' 종목을 미리 정해두고,
#     지금 노출이 적은 그룹부터 추천한다.
#   ★ OANDA 상품(금·원유·채권·지수·통화)이 여기 같이 들어간다.
#     AI 주식 8개를 아무리 늘려도 분산이 안 되지만, 원유 하나는 진짜 분산이 된다.
# ------------------------------------------------------------
# 🟥 [FIX-DV2] OANDA 계좌가 통화쌍만 거래 가능하다는 게 확인됐다(실측: 68개 전부 CURRENCY).
#    금·은·원유·천연가스·채권·지수 CFD 는 전부 주문이 불가능하므로, 추천해봐야 BLOCKED 만 쌓인다.
#    → 같은 노출을 Alpaca 에서 살 수 있는 ETF 로 전부 갈아끼운다. 통화쌍만 OANDA 로 남긴다.
#    ⚠️ ETF 는 미국 장중에만 거래된다. 24시간 상품이 아니므로 주식 경로(시간필터·장마감청산)를 탄다.
DIVERSIFIER_UNIVERSE = {
    "귀금속": [
        ("GLD", "Alpaca", "금 ETF. 주식과 상관이 낮고 위기 때 반대로 움직인다. 금 노출의 표준"),
        ("IAU", "Alpaca", "금 ETF(보수 더 저렴). GLD 와 사실상 같은 것 — 둘 중 하나만"),
        ("SLV", "Alpaca", "은 ETF. 금과 같은 방향이지만 변동성이 더 크다"),
    ],
    "원유·에너지": [
        ("XLE", "Alpaca", "에너지 섹터 ETF. 유가와 연동되면서 USO 같은 롤오버 손실이 없다 — 원유 노출은 이걸 우선"),
        ("USO", "Alpaca", "WTI 원유 선물 ETF. 유가를 직접 따라가지만 콘탱고일 때 장기보유 손실이 난다"),
        ("BNO", "Alpaca", "브렌트유 ETF. USO 와 거의 같이 움직이므로 둘 중 하나만"),
    ],
    "채권": [
        ("TLT", "Alpaca", "미국 20년+ 국채 ETF. 주식이 무너질 때 반대로 가는 대표 자산, 유동성 최상위"),
        ("IEF", "Alpaca", "미국 7~10년 국채 ETF. TLT 보다 변동성이 작다"),
        ("SHY", "Alpaca", "미국 1~3년 국채 ETF. 사실상 현금 대용, 금리 정책에 민감"),
    ],
    "USD통화": [
        ("USD_JPY", "OANDA", "달러/엔. 유동성 최상위, 스프레드 최저. 단 개입 리스크 주의"),
        ("EUR_USD", "OANDA", "유로/달러. 세계 최대 거래량, 비용이 가장 싸다"),
        ("AUD_USD", "OANDA", "호주달러. 원자재·중국 경기와 연동돼 EUR 와는 다르게 움직인다"),
        ("USD_CAD", "OANDA", "캐나다달러. 유가와 연동. 원유 대신 쓸 수도 있다"),
    ],
    "미국지수": [
        ("SPY", "Alpaca", "S&P500 ETF. 나스닥보다 넓어 AI 편중이 덜하다. 세계에서 가장 유동성 높은 종목"),
        ("IWM", "Alpaca", "러셀2000 소형주 ETF. 대형 테크와 다르게 움직이는 구간이 있다"),
        ("RSP", "Alpaca", "S&P500 동일가중 ETF. 시총가중 SPY 와 달리 대형 테크 쏠림이 없다"),
    ],
    "해외지수": [
        ("EWJ", "Alpaca", "일본 주식 ETF. 엔화 방향과 미국장 양쪽에 반응, 구성이 완전히 다르다"),
        ("VGK", "Alpaca", "유럽 주식 ETF. 산업재·금융 비중이 높아 미국 테크지수와 구성이 다르다"),
        ("EEM", "Alpaca", "신흥국 ETF. 달러 방향과 반대로 움직이는 경향"),
    ],
    "산업금속": [
        ("CPER", "Alpaca", "구리 ETF. 제조업 경기 지표, 금과도 테크와도 다르게 움직인다. 거래량은 얇은 편"),
        ("XME", "Alpaca", "금속·광업 ETF. CPER 보다 유동성이 낫지만 주식이라 증시와 상관이 좀 더 높다"),
    ],
    "금융": [
        ("JPM", "Alpaca", "JP모건. 금리 상승 국면에서 테크와 반대로 움직이는 대표 종목"),
        ("GS", "Alpaca", "골드만삭스. 금융 대표주, 유동성 충분"),
        ("BRK.B", "Alpaca", "버크셔. 가치주 성향이라 AI 사이클과 상관이 낮다"),
    ],
    "헬스케어": [
        ("LLY", "Alpaca", "일라이릴리. 헬스케어 대장주, 테크 사이클과 무관"),
        ("UNH", "Alpaca", "유나이티드헬스. 경기방어 성격"),
        ("JNJ", "Alpaca", "존슨앤드존슨. 변동성 낮고 방어적"),
    ],
    "에너지": [
        ("XOM", "Alpaca", "엑슨모빌. 유가 연동, AI와 상관 거의 없음"),
        ("CVX", "Alpaca", "셰브론. XOM과 같이 움직이므로 둘 중 하나만"),
    ],
    "소비재": [
        ("COST", "Alpaca", "코스트코. 경기방어 + 꾸준한 추세"),
        ("WMT", "Alpaca", "월마트. 방어적이면서 추세가 잘 이어진다"),
        ("KO", "Alpaca", "코카콜라. 변동성 최저 수준, 포트폴리오 안정용"),
    ],
    "산업재": [
        ("CAT", "Alpaca", "캐터필러. 실물경기 대표주"),
        ("RTX", "Alpaca", "RTX. 방산 — 지정학 이벤트에 테크와 반대로 반응"),
        ("LMT", "Alpaca", "록히드마틴. 방산, 경기와 무관한 수주 기반"),
    ],
    "자동차": [
        ("TSLA", "Alpaca", "테슬라. 변동성 크고 자체 사이클이 있어 AI 그룹과 완전히 겹치지는 않는다"),
    ],
    "통신": [
        ("TMUS", "Alpaca", "T-모바일. 방어적, 배당 성향"),
    ],
}


def sync_top_active_candidates(top_n: int = 12):
    """
    🟥 [FIX-P1] '오늘의 추천 후보' 탭 전면 개편.

    ■ 예전 방식의 문제
      Alpaca '거래량 상위'만 보고 뽑았다. 그런데 거래량 상위는
        · 그날 화제인 종목 = 대개 이미 들고 있는 것과 같은 테마
        · 주식 수가 많은 페니주 (거래량이 많은 건 유동성이 아니라 주식 수 때문)
        · 3배 레버리지 ETF
      셋 중 하나다. 분산에 아무 도움이 안 됐다.

    ■ 새 방식
      (1) 지금 Alpaca+OANDA에 뭘 얼마나 들고 있는지 **그룹별로 집계**한다
      (2) 노출이 **적은 그룹부터** 후보를 추천한다
      (3) 금·원유·채권·통화·해외지수(OANDA)도 후보에 포함한다
          — AI 주식을 아무리 늘려도 분산이 안 되지만 원유 하나는 진짜 분산이 된다
      (4) 왜 추천하는지, 지금 그 분야 노출이 얼마인지, 이미 들고 있는지를 같이 적는다

    탭은 매번 통째로 새로 쓴다(append 아님).
    """
    HEADERS = ["분야", "종목", "거래소", "현재가", "추천 이유",
               "내 포트폴리오와 겹침", "가장 많이 겹치는 보유종목",
               "보유 여부", "분산 점수", "판정"]

    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name(
            "/etc/secrets/google_credentials.json", scope)
        client = gspread.authorize(creds)
        spreadsheet = client.open(GOOGLE_SHEET_NAME)
        try:
            ws = spreadsheet.worksheet("오늘의 추천 후보")
        except gspread.exceptions.WorksheetNotFound:
            ws = spreadsheet.add_worksheet(title="오늘의 추천 후보", rows=400, cols=len(HEADERS))
            print("✅ [추천후보] 탭이 없어서 새로 생성했습니다.")
    except Exception as e:
        print(f"❌ [추천후보] 시트 연결 실패: {e}")
        return

    today_str = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M ET")

    # ── (1) 현재 노출 집계 ──────────────────────────────────
    snap = summarize_portfolio_exposure()
    groups = snap.get("groups", {})
    equity = snap.get("equity")
    held_symbols = {p["symbol"].upper() for p in snap.get("positions", [])}

    positions = snap.get("positions", [])

    def overlap_pct(sym: str) -> tuple[float, str]:
        """이 종목이 지금 내 포트폴리오와 얼마나 겹치는가 (자산 대비 %).

        🟥 [FIX-P2] 분야 표가 아니라 실측 상관계수로 계산한다.
        같이 나오는 문자열은 '가장 많이 겹치는 보유 종목'이다.
        """
        if not positions or not equity:
            return 0.0, "-"
        expo, detail = correlated_exposure(sym, positions)
        if not detail:
            return 0.0, "-"
        top = max(detail, key=lambda d: abs(d[2]))
        return abs(expo) / equity * 100.0, f"{top[0]} (ρ{top[1]:+.2f})"

    # ── 후보 점수화 ─────────────────────────────────────────
    #  분산 점수 = 100 - (내 포트폴리오와 겹치는 정도 %) × 3
    #  이미 보유 중이면 -40. 한도를 이미 넘긴 종목은 0점.
    # ============================================================
    # 🟥 [FIX-WATCH1] '보유 여부' 가 항상 '없음' 으로 나오던 문제.
    #   이 칸은 지금까지 '열린 포지션이 있나' 만 봤다. 그래서 트레이딩뷰에 종목을
    #   넣고 알람까지 걸어놨어도, 그 순간 포지션이 없으면 전부 '없음' 이었다.
    #   봇은 트레이딩뷰 감시목록을 볼 수 없다 — 하지만 **알람이 왔었는지**는 안다.
    #   최근 N일 안에 알람이 한 번이라도 온 종목은 '감시 중' 으로 표시한다.
    #   (그게 사실상 "내가 트레이딩뷰에 넣어둔 종목" 과 같은 뜻이다)
    # ============================================================
    WATCH_LOOKBACK_DAYS = int(os.getenv("WATCH_LOOKBACK_DAYS", "14"))
    watching: set = set()
    try:
        _sh = _get_sheet()
        if _sh is not None:
            _cut = (datetime.now(ZoneInfo("America/New_York"))
                    - timedelta(days=WATCH_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
            for _r in _sh.get_all_values()[1:]:
                if len(_r) > 1 and _r[0] and str(_r[0])[:10] >= _cut and _r[1]:
                    watching.add(str(_r[1]).strip().upper())
        print(f"👀 [추천후보] 최근 {WATCH_LOOKBACK_DAYS}일 알람 온 종목 {len(watching)}개")
    except Exception as e:
        print(f"⚠️ [추천후보] 감시목록 조회 실패(무시): {e}")

    rows: list[list] = []
    for grp, items in DIVERSIFIER_UNIVERSE.items():
        for sym, venue, reason in items:
            ov, top_peer = overlap_pct(sym)
            held = sym.upper() in held_symbols
            score = 100.0 - ov * 3.0
            if held:
                score -= 40.0
            if ov >= PORTFOLIO_GROUP_MAX_PCT:
                score = 0.0
            score = max(0.0, round(score, 1))

            px = _oanda_price_quick(sym) if venue == "OANDA" else get_alpaca_latest_price(sym)

            # 페니주·레버리지 ETF 방어는 그대로 유지
            if venue == "Alpaca":
                blocked, why = is_blocked_instrument(sym, px)
                if blocked:
                    print(f"🚫 [추천후보] {sym} 제외 — {why}")
                    continue

            if ov >= PORTFOLIO_GROUP_MAX_PCT:
                verdict = f"⛔ 한도 초과 — 이미 {ov:.1f}% 겹침 (상한 {PORTFOLIO_GROUP_MAX_PCT:.0f}%)"
            elif held:
                verdict = "🔁 이미 보유 — 추가하면 같은 베팅이 커진다"
            elif sym.upper() in watching:
                verdict = "👀 이미 감시 중 — 알람은 걸려 있고 지금 포지션만 없다"
            elif ov <= 2.0:
                verdict = "⭐ 최우선 — 지금 포트폴리오와 거의 안 겹친다"
            elif ov < PORTFOLIO_GROUP_MAX_PCT / 2:
                verdict = "✅ 추천 — 아직 여유 있음"
            else:
                verdict = "🟡 여유 적음 — 소량만"

            rows.append([
                grp, sym, venue,
                px if px is not None else "조회실패",
                reason,
                f"{ov:.1f}%",
                top_peer,
                ("보유 중" if held else ("👀 감시 중" if sym.upper() in watching else "없음")),
                score,
                verdict,
            ])

    rows.sort(key=lambda r: r[8], reverse=True)
    rows = rows[:max(top_n, 1)]

    # ── (3) 현재 포트폴리오 요약 블록 ───────────────────────
    out: list[list] = []
    out.append([f"오늘의 추천 후보  ({today_str})"])
    out.append([])
    out.append(["■ 지금 내 포지션이 어디에 쏠려 있나"])
    if not snap.get("ok"):
        out.append(["포지션/자산 조회 실패 — 아래 추천은 노출 0% 가정으로 계산된 값입니다"])
    else:
        out.append(["분야", "보유 종목", "총 노출", "순 노출(방향 반영)", "상태"])
        if not groups:
            out.append(["(열린 포지션 없음)", "", "", "", ""])
        for g, info in sorted(groups.items(), key=lambda kv: -abs(kv[1]["net_pct"])):
            _net_abs = abs(info["net_pct"])
            if _net_abs >= PORTFOLIO_GROUP_MAX_PCT:
                state = f"⛔ 한도 초과 (상한 {PORTFOLIO_GROUP_MAX_PCT:.0f}%)"
            elif _net_abs >= PORTFOLIO_GROUP_MAX_PCT * 0.7:
                state = "🟡 한도 근접"
            else:
                state = "✅ 여유"
            out.append([g, ", ".join(sorted(set(info["symbols"])))[:200],
                        f"{info['gross_pct']:.1f}%", f"{info['net_pct']:+.1f}%", state])
        out.append(["합계", "", f"{snap.get('total_gross_pct', 0):.1f}%",
                    f"자산 ${equity:,.0f}" if equity else "", ""])

        # 🟥 [FIX-P2] "내 10개가 사실은 몇 개인가" — 실측 상관계수로 직접 보여준다.
        #   분야 이름표가 아니라 실제 가격이 답한다.
        held = [p["symbol"] for p in positions]
        if len(held) >= 2:
            pairs = []
            for i in range(len(held)):
                for j in range(i + 1, len(held)):
                    rho = measured_correlation(held[i], held[j])
                    if rho is not None:
                        pairs.append((abs(rho), held[i], held[j], rho))
            pairs.sort(reverse=True)
            if pairs:
                out.append([])
                out.append(["■ 지금 보유 종목끼리 실제로 얼마나 같이 움직이나 (일봉 수익률 상관)"])
                out.append(["종목 A", "종목 B", "상관계수", "해석", ""])
                for _, a, b, rho in pairs[:12]:
                    if rho >= 0.7:
                        note = "사실상 같은 베팅"
                    elif rho >= 0.4:
                        note = "상당히 겹침"
                    elif rho <= -0.4:
                        note = "서로 상쇄 (헤지 효과)"
                    else:
                        note = "거의 독립 — 분산에 도움"
                    out.append([a, b, f"{rho:+.2f}", note, ""])
                strong = sum(1 for p in pairs if p[0] >= 0.7)
                out.append([f"※ 상관 0.7 이상인 쌍이 {strong}개 / 전체 {len(pairs)}쌍. "
                            f"많을수록 '종목 수'보다 실제 분산이 훨씬 적다는 뜻입니다."])

    out.append([])
    out.append(["■ 분산에 도움이 되는 후보 (내 포트폴리오와 덜 겹치는 순)"])
    out.append(HEADERS)
    out.extend(rows)
    out.append([])
    out.append(["※ '겹침'은 분야 이름이 아니라 **실제 일봉 상관계수**로 계산합니다. "
                "새 종목을 트레이딩뷰에 추가만 하면 자동으로 반영됩니다 (코드 수정 불필요)."])
    out.append(["※ '분산 점수'가 높을수록 지금 포트폴리오와 안 겹치는 종목입니다."])
    out.append([f"※ '보유 여부' — 보유 중: 지금 포지션이 있음 / 👀 감시 중: 최근 "
                f"{WATCH_LOOKBACK_DAYS}일 안에 알람이 온 적 있음(=트레이딩뷰에 넣어둔 종목) / "
                f"없음: 둘 다 아님. 봇은 트레이딩뷰 감시목록을 직접 볼 수 없어서 알람 이력으로 판단합니다."])
    out.append([f"※ 한 분야 상한 {PORTFOLIO_GROUP_MAX_PCT:.0f}% / 전체 상한 {PORTFOLIO_TOTAL_MAX_PCT:.0f}% "
                f"(환경변수 PORTFOLIO_GROUP_MAX_PCT / PORTFOLIO_TOTAL_MAX_PCT 로 조정)"])
    out.append(["※ 같은 분야에 이미 차 있으면 알람이 울려도 봇이 자동으로 수량을 줄이거나 건너뜁니다."])

    # 열 개수를 맞춰야 gspread 가 거부하지 않는다
    width = max(len(r) for r in out)
    out = [r + [""] * (width - len(r)) for r in out]

    try:
        ws.clear()
        # 🟥 [FIX-P1b/D3] 기존 탭은 5열로 만들어져 있다(옛 헤더 5개).
        #   clear() 는 격자 크기를 안 바꾸고 update() 도 자동 확장하지 않아서
        #   "exceeds grid limits. Max columns: 5" 로 실패한다 — 그런데 except 에
        #   먹혀서 '조용히 갱신 안 됨' 이 된다. 쓰기 전에 격자를 늘린다.
        try:
            ws.resize(rows=max(len(out) + 20, 60), cols=max(width, len(HEADERS)))
        except Exception as _e:
            print(f"⚠️ [추천후보] 시트 크기 조정 실패(계속 진행): {_e}")
        ws.update("A1", out)
        print(f"✅ [추천후보] 전면 갱신 완료 — 후보 {len(rows)}건, 그룹 {len(groups)}개 ({today_str})")
    except Exception as e:
        print(f"❌ [추천후보] 시트 쓰기 실패: {e}")


async def _daily_top_movers_loop():
    """매일 미국 동부시간 오전 10시에 sync_top_active_candidates()를 1번 실행."""
    while True:
        now_ny = datetime.now(ZoneInfo("America/New_York"))
        target = now_ny.replace(hour=10, minute=0, second=0, microsecond=0)
        if now_ny >= target:
            target = target + timedelta(days=1)
        wait_seconds = (target - now_ny).total_seconds()
        await asyncio.sleep(wait_seconds)
        try:
            await asyncio.to_thread(sync_top_active_candidates)
        except Exception as e:
            print(f"❌ [추천후보 루프] 오류: {e}")
        await asyncio.sleep(60)  # 같은 분에 중복 실행 방지용 약간의 여유


def sync_score_bucket_analysis():
    """
    'Alpaca 거래내역'의 점수 컬럼을 구간별로 나눠서 승률/손익을 분석.
    "threshold를 X로 올리면/내리면 승률·손익이 어떻게 바뀌는지"를 보기 위한 용도.
    탭이 없으면 자동 생성, 매번 전체 재계산해서 덮어쓴다.
    🟦 SKIPPED 가상 손익도 별도 컬럼으로 추가:
       "threshold 안쪽 신호들이 실제로 어떻게 됐을지"를 같이 보여줘서
       threshold 조정 근거를 데이터로 명확히 판단할 수 있게 함.
    """
    HEADERS = [
        "점수구간",
        "거래건수", "승(TP)", "패(SL)", "승률(%)", "총손익($)", "평균손익($)",
        "SKIPPED건수", "SKIPPED_TP", "SKIPPED_SL", "SKIPPED승률(%)", "SKIPPED가상손익($)"
    ]
    BUCKETS = [
        (-999, -3, "-3 미만"), (-3, -2, "-3~-2"), (-2, -1, "-2~-1"), (-1, 0, "-1~0"),
        (0, 1, "0~1"), (1, 2, "1~2"), (2, 3, "2~3"), (3, 4, "3~4"), (4, 999, "4 이상"),
    ]

    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name("/etc/secrets/google_credentials.json", scope)
        client = gspread.authorize(creds)
        spreadsheet = client.open("민균 FX trading result")

        try:
            trade_ws = spreadsheet.worksheet("Alpaca 거래내역")
            trade_rows = trade_ws.get_all_values()
        except gspread.exceptions.WorksheetNotFound:
            print("⚠️ [점수구간분석] 'Alpaca 거래내역' 탭이 아직 없음 → sync_alpaca_trade_log()를 먼저 실행해야 함")
            return

        # 🟦 메인 시트에서 SKIPPED 가상 손익 가져오기
        try:
            main_rows = spreadsheet.sheet1.get_all_values()
        except Exception:
            main_rows = []

        try:
            ws = spreadsheet.worksheet("스코어대별 성과분석")
        except gspread.exceptions.WorksheetNotFound:
            ws = spreadsheet.add_worksheet(title="스코어대별 성과분석", rows=50, cols=len(HEADERS))
            print("✅ [점수구간분석] 탭이 없어서 새로 생성했습니다.")
    except Exception as e:
        print(f"❌ [점수구간분석] 시트 연결 실패: {e}")
        return

    # 실제 거래 집계
    bucket_stats = {b[2]: {"tp": 0, "sl": 0, "pnl_list": []} for b in BUCKETS}
    for row in trade_rows[1:]:
        if len(row) < 14:
            continue
        score_str, status_kr, pnl_str = row[4], row[9], row[13]
        if status_kr not in ("TP청산", "SL청산", "TIME_EXIT"):
            continue
        try:
            score = float(score_str)
        except Exception:
            continue

        for lo, hi, label in BUCKETS:
            if lo <= score < hi:
                b = bucket_stats[label]
                try:
                    _pnl_for_winloss = float(pnl_str)
                except Exception:
                    _pnl_for_winloss = None
                is_win = (status_kr == "TP청산") or (status_kr == "TIME_EXIT" and _pnl_for_winloss is not None and _pnl_for_winloss > 0)
                if is_win:
                    b["tp"] += 1
                else:
                    b["sl"] += 1
                try:
                    b["pnl_list"].append(float(pnl_str))
                except Exception:
                    pass
                break

    # 🟦 SKIPPED 가상 손익 집계 (메인 시트의 SKIPPED_BY_THRESHOLD 행들)
    # 헤더: timestamp(0), symbol(1), strategy(2), signal_type(3), decision(4), score(5), ..., summary(16), ..., total_pnl(24)
    skip_stats = {b[2]: {"tp": 0, "sl": 0, "pnl_list": []} for b in BUCKETS}
    for row in main_rows[1:]:
        if len(row) < 25:
            continue
        if row[4] != "SKIPPED_BY_THRESHOLD":
            continue
        summary_val = row[16] if len(row) > 16 else ""
        if summary_val not in ("TP_HIT", "SL_HIT"):
            continue
        try:
            score = float(row[5])
        except Exception:
            continue
        try:
            pnl_val = float(row[24])
        except Exception:
            continue
        for lo, hi, label in BUCKETS:
            if lo <= score < hi:
                s = skip_stats[label]
                if summary_val == "TP_HIT":
                    s["tp"] += 1
                else:
                    s["sl"] += 1
                s["pnl_list"].append(pnl_val)
                break

    summary_rows = [HEADERS]
    for lo, hi, label in BUCKETS:
        b = bucket_stats[label]
        trades = b["tp"] + b["sl"]
        win_rate = round(b["tp"] / trades * 100, 1) if trades else ""
        total_pnl = round(sum(b["pnl_list"]), 2) if b["pnl_list"] else ""
        avg_pnl = round(sum(b["pnl_list"]) / len(b["pnl_list"]), 2) if b["pnl_list"] else ""

        s = skip_stats[label]
        skip_trades = s["tp"] + s["sl"]
        skip_wr = round(s["tp"] / skip_trades * 100, 1) if skip_trades else ""
        skip_pnl = round(sum(s["pnl_list"]), 2) if s["pnl_list"] else ""

        summary_rows.append([
            label,
            trades, b["tp"], b["sl"], win_rate, total_pnl, avg_pnl,
            skip_trades, s["tp"], s["sl"], skip_wr, skip_pnl
        ])

    try:
        ws.clear()
        ws.update("A1", summary_rows)
        print("✅ [점수구간분석] 갱신 완료 (SKIPPED 가상손익 포함)")
    except Exception as e:
        print(f"❌ [점수구간분석] 시트 쓰기 실패: {e}")


def _aggregate_trade_stats(trade_rows, start=None, end=None):
    """
    'Alpaca 거래내역' 행들을 [start, end) 구간으로 필터링해서 통계 집계.
    start/end가 둘 다 None이면 전체 기간(누적) 집계.
    헤더: 주문ID,진입시각,종목,방향,점수,수량,진입가,TP가,SL가,상태,청산가,청산시각,보유시간(분),손익($),손익(%),누적손익($)
    """
    hour_stats, dow_stats = {}, {}
    hold_times, risk_amounts, pnl_list = [], [], []
    total_trades = 0
    total_pnl = 0.0
    intervals = []  # (entry_dt, exit_dt) — 동시노출 계산용

    for row in trade_rows[1:]:
        if len(row) < 14 or row[9] not in ("TP청산", "SL청산", "TIME_EXIT"):
            continue
        try:
            raw_ts = str(row[1]).strip()
            # 🟦 진입시각이 "2026-07-01 09:45:30 ET" 형식(ET 변환 후 저장)이거나
            #    "2026-06-22T13:15:29.978106Z" 같은 UTC ISO 형식일 수 있음.
            if raw_ts.endswith(" ET"):
                # ET 형식 → ET로 직접 파싱
                dt_naive = datetime.strptime(raw_ts[:-3], "%Y-%m-%d %H:%M:%S")
                t = dt_naive.replace(tzinfo=ZoneInfo("America/New_York"))
            else:
                t = datetime.fromisoformat(raw_ts.replace("Z", "+00:00")).astimezone(ZoneInfo("America/New_York"))
        except Exception:
            continue
        if start and t < start:
            continue
        if end and t >= end:
            continue
        total_trades += 1
        h = hour_stats.setdefault(t.hour, {"tp": 0, "sl": 0})
        d = dow_stats.setdefault(t.weekday(), {"tp": 0, "sl": 0})
        try:
            _pnl_for_winloss = float(row[13])
        except Exception:
            _pnl_for_winloss = None
        is_win = (row[9] == "TP청산") or (row[9] == "TIME_EXIT" and _pnl_for_winloss is not None and _pnl_for_winloss > 0)
        if is_win:
            h["tp"] += 1
            d["tp"] += 1
        else:
            h["sl"] += 1
            d["sl"] += 1
        try:
            hold_times.append(float(row[12]))
        except Exception:
            pass
        try:
            pnl = float(row[13])
            pnl_list.append(pnl)
            total_pnl += pnl
        except Exception:
            pass
        # R-멀티플 계산용 리스크 금액 = |진입가-SL가| × 수량
        try:
            entry_p, sl_p, qty = float(row[6]), float(row[8]), float(row[5])
            risk_amounts.append(abs(entry_p - sl_p) * qty)
        except Exception:
            pass
        # 동시노출 계산용 (진입~청산 구간)
        try:
            raw_exit = str(row[11]).strip()
            if raw_exit.endswith(" ET"):
                dt_naive = datetime.strptime(raw_exit[:-3], "%Y-%m-%d %H:%M:%S")
                exit_t = dt_naive.replace(tzinfo=ZoneInfo("America/New_York"))
            else:
                exit_t = datetime.fromisoformat(raw_exit.replace("Z", "+00:00")).astimezone(ZoneInfo("America/New_York"))
            intervals.append((t, exit_t))
        except Exception:
            pass

    hour_table, dow_table = [], []
    for h in sorted(hour_stats):
        s = hour_stats[h]
        t = s["tp"] + s["sl"]
        hour_table.append(f"{h}시: {t}건, 승률 {round(s['tp']/t*100,1)}%" if t else f"{h}시: 0건")
    dow_names = ["월", "화", "수", "목", "금", "토", "일"]
    for d in sorted(dow_stats):
        s = dow_stats[d]
        t = s["tp"] + s["sl"]
        dow_table.append(f"{dow_names[d]}: {t}건, 승률 {round(s['tp']/t*100,1)}%" if t else f"{dow_names[d]}: 0건")

    avg_hold = round(sum(hold_times) / len(hold_times), 1) if hold_times else None
    win_rate = None
    tp_total = sum(s["tp"] for s in hour_stats.values())
    sl_total = sum(s["sl"] for s in hour_stats.values())
    if tp_total + sl_total > 0:
        win_rate = round(tp_total / (tp_total + sl_total) * 100, 1)

    # 🟦 R-멀티플(기대값): 각 거래의 pnl을 그 거래의 리스크금액(R)으로 나눈 평균.
    #    "승률"만으로는 못 보이는 손익비 효율을 같이 보기 위함.
    avg_risk = sum(risk_amounts) / len(risk_amounts) if risk_amounts else None
    expectancy_r = None
    if avg_risk and avg_risk > 0 and pnl_list:
        expectancy_r = round(sum(p / avg_risk for p in pnl_list) / len(pnl_list), 3)

    # 🟦 동시노출: 같은 시각에 동시에 열려있던 포지션 수의 최댓값 (스윕 라인 방식)
    max_concurrent = 0
    if intervals:
        events = []
        for s, e in intervals:
            events.append((s, 1))
            events.append((e, -1))
        events.sort()
        cur = 0
        for _, delta in events:
            cur += delta
            max_concurrent = max(max_concurrent, cur)

    return {
        "total_trades": total_trades,
        "total_pnl": round(total_pnl, 2),
        "avg_hold": avg_hold,
        "hour_table": hour_table,
        "dow_table": dow_table,
        "win_rate": win_rate,
        "expectancy_r": expectancy_r,
        "max_concurrent": max_concurrent,
    }


def _aggregate_wait_calibration(main_rows, start=None, end=None):
    """
    메인 시트에서 WAIT 행들의 실제 결과(TP_HIT=놓친기회 / SL_HIT=방어성공)와,
    GPT가 보고한 wait_confidence(adjustment_suggestion 컬럼에 'wait_confidence=NN' 형식으로 저장됨)를
    같이 봐서 "GPT가 80 이상이라고 한 WAIT들이 실제로 맞았는지" 보정 정확도를 계산.
    """
    wait_tp, wait_sl = 0, 0
    conf_high_correct, conf_high_wrong = 0, 0  # confidence>=80인데 실제로 맞았는지/틀렸는지
    for row in main_rows[1:]:
        if len(row) < 35:
            continue
        try:
            t = datetime.fromisoformat(row[0])
        except Exception:
            continue
        if start and t < start:
            continue
        if end and t >= end:
            continue
        if row[4] != "WAIT" and not row[4].startswith("SKIPPED"):
            continue
        result = row[16] if len(row) > 16 else ""
        if result == "TP_HIT":
            wait_tp += 1
        elif result == "SL_HIT":
            wait_sl += 1
        else:
            continue

        adj = row[34] if len(row) > 34 else ""
        if adj.startswith("wait_confidence="):
            try:
                conf = float(adj.split("=")[1])
            except Exception:
                conf = None
            if conf is not None and conf >= 80:
                if result == "SL_HIT":  # WAIT이 맞았음(실패를 예측했고 실제로 실패함)
                    conf_high_correct += 1
                else:  # TP_HIT인데도 WAIT함 → 확신도는 높았지만 틀림(기회를 놓침)
                    conf_high_wrong += 1

    return {
        "wait_tp": wait_tp, "wait_sl": wait_sl,
        "conf_high_correct": conf_high_correct, "conf_high_wrong": conf_high_wrong,
    }


def _build_stats_text(label, trade_stats, wait_stats, symbol_summary, score_summary):
    breakeven_wr = round(1 / (1 + STOCK_TP_ATR_MULT / STOCK_SL_ATR_MULT) * 100, 1)
    wait_total = wait_stats["wait_tp"] + wait_stats["wait_sl"]
    conf_total = wait_stats["conf_high_correct"] + wait_stats["conf_high_wrong"]
    conf_acc = round(wait_stats["conf_high_correct"] / conf_total * 100, 1) if conf_total else None

    return f"""
[{label} 거래 통계 — 전부 코드로 정확히 집계된 숫자, 환각 없음]
총 체결 거래: {trade_stats['total_trades']}건
총손익: ${trade_stats['total_pnl']}
평균 보유시간: {trade_stats['avg_hold']}분
전체 승률: {trade_stats['win_rate']}% (TP/SL 비율 0.8:1.0 기준 손익분기 승률 {breakeven_wr}%)
기대값(R-멀티플, 거래당 평균): {trade_stats['expectancy_r']}  (0보다 크면 장기적으로 이익 구조)
최대 동시노출 포지션 수: {trade_stats['max_concurrent']}건

시간대별 승률:
{chr(10).join(trade_stats['hour_table']) if trade_stats['hour_table'] else "데이터 없음"}

요일별 승률:
{chr(10).join(trade_stats['dow_table']) if trade_stats['dow_table'] else "데이터 없음"}

WAIT/필터된 신호 중 실제 결과 ({wait_total}건 평가됨):
- 놓친 기회(WAIT했는데 TP_HIT): {wait_stats['wait_tp']}건
- 방어 성공(WAIT했는데 SL_HIT): {wait_stats['wait_sl']}건
- GPT가 wait_confidence 80 이상이라고 보고한 것 중 실제 정확도: {conf_acc}% ({conf_total}건 중 {wait_stats['conf_high_correct']}건 맞음)

종목별 성과분석 탭(상위 20행):
{symbol_summary}

점수구간별 성과분석 탭:
{score_summary}
"""


def _ask_gpt_for_report(stats_text, period_label):
    prompt = f"""너는 퀀트 트레이딩 시스템 분석가다. 아래는 {period_label} 자동매매 시스템의 실제 거래 통계다.
이 숫자들(이미 정확히 집계된 값, 네가 새로 계산하지 마라)을 바탕으로 한국어 리포트를 작성하라.

{stats_text}

리포트에 반드시 포함할 것:
1. 전체 요약 (한 줄)
2. 시간대별/요일별 패턴에서 발견된 것 — 특정 시간/요일이 유난히 안 좋으면 짚어라
3. TP/SL 비율(0.8:1.0)이 손익분기 승률과 실제 승률 대비 합리적인지, 기대값(R-멀티플)도 같이 평가
4. WAIT 판단이 합리적이었는지 — 놓친 기회 vs 방어 성공 비율 + GPT 확신도 보정 정확도로 판단
   (확신도 보정 정확도가 낮으면 "GPT가 자신감만 높고 실제로는 못 맞춘다"는 뜻이니 명확히 짚어라)
5. 최대 동시노출 포지션 수가 리스크 관리 관점에서 괜찮은지
6. 점수구간별 성과를 보고 threshold를 올리거나 내려야 할지 구체적 제안
7. 제외를 검토할 만한 종목과 그 이유
8. 마지막에 명시: "이 리포트는 통계 기반이며, 코드 레벨 버그나 로직 오류 진단은 Claude와 직접 데이터를 보며 논의하는 것을 권장함"

너무 길게 쓰지 말고, 핵심만 명확하게. 마크다운 헤더(##) 써도 된다."""
    try:
        body = {
            "model": "gpt-4o-2024-11-20",
            "input": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_output_tokens": 1800,
        }
        r = requests.post(OPENAI_URL, headers=OPENAI_HEADERS, json=body, timeout=60)
        r.raise_for_status()
        resp = r.json()
        report_text = ""
        for item in resp.get("output", []):
            for c in item.get("content", []):
                if c.get("type") == "output_text":
                    report_text += c.get("text", "")
        return report_text or stats_text
    except Exception as e:
        print(f"❌ [리포트] GPT 호출 실패: {e}")
        return stats_text


def generate_weekly_report():
    """
    매주 토요일 오전, "이번 주(월~금)" 데이터 + "전체 누적" 데이터를 종합 분석해서
    '주간 리포트' 탭에 한 행(이번 주 분석 | 누적 분석)으로 남긴다.
    - 시간대별/요일별 승률, TP/SL 합리성(손익분기 대비), 기대값(R-멀티플), WAIT 놓친기회/방어성공 비율,
      WAIT 확신도 보정 정확도, 최대 동시노출, 점수구간/종목별 성과 — 전부 코드로 정확히 집계.
    - 그 집계 결과를 GPT에게 줘서 자연어 리포트로 작성하게 함.
    - 코드 레벨 버그 진단까지는 이 자동 리포트로 한계가 있다는 점은 리포트 안에도 명시함.
    """
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name("/etc/secrets/google_credentials.json", scope)
        client = gspread.authorize(creds)
        spreadsheet = client.open("민균 FX trading result")
        main_rows = spreadsheet.sheet1.get_all_values()
        try:
            trade_rows = spreadsheet.worksheet("Alpaca 거래내역").get_all_values()
        except gspread.exceptions.WorksheetNotFound:
            trade_rows = []
        try:
            report_ws = spreadsheet.worksheet("주간 리포트")
            header = report_ws.row_values(1)
            if header[:3] != ["작성일", "이번 주 분석", "누적 분석"]:
                report_ws.update_cell(1, 1, "작성일")
                report_ws.update_cell(1, 2, "이번 주 분석")
                report_ws.update_cell(1, 3, "누적 분석")
        except gspread.exceptions.WorksheetNotFound:
            report_ws = spreadsheet.add_worksheet(title="주간 리포트", rows=2000, cols=3)
            report_ws.append_row(["작성일", "이번 주 분석", "누적 분석"])
            print("✅ [주간리포트] 탭이 없어서 새로 생성했습니다.")
    except Exception as e:
        print(f"❌ [주간리포트] 시트 연결 실패: {e}")
        return

    now_ny = datetime.now(ZoneInfo("America/New_York"))
    # 🟦 "이번 주"를 롤링 7일이 아니라 이번 주의 월요일 00:00 ~ 금요일 24:00(=토요일 00:00 직전)로 한정.
    monday = (now_ny - timedelta(days=now_ny.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    week_start, week_end = monday, monday + timedelta(days=5)

    try:
        symbol_rows = spreadsheet.worksheet("종목별 성과분석").get_all_values()
        symbol_summary = "\n".join([",".join(r) for r in symbol_rows[:20]])
    except Exception:
        symbol_summary = "데이터 없음"
    try:
        score_rows = spreadsheet.worksheet("스코어대별 성과분석").get_all_values()
        score_summary = "\n".join([",".join(r) for r in score_rows])
    except Exception:
        score_summary = "데이터 없음"

    # 1) 이번 주(월~금) 분석
    week_trade_stats = _aggregate_trade_stats(trade_rows, week_start, week_end)
    week_wait_stats = _aggregate_wait_calibration(main_rows, week_start, week_end)
    week_stats_text = _build_stats_text("이번 주(월~금)", week_trade_stats, week_wait_stats, symbol_summary, score_summary)
    week_report = _ask_gpt_for_report(week_stats_text, f"{week_start.strftime('%Y-%m-%d')}~{(week_end-timedelta(days=1)).strftime('%Y-%m-%d')}(월~금)")

    # 2) 전체 누적 분석 (기간 제한 없음)
    cum_trade_stats = _aggregate_trade_stats(trade_rows, None, None)
    cum_wait_stats = _aggregate_wait_calibration(main_rows, None, None)
    cum_stats_text = _build_stats_text("전체 누적", cum_trade_stats, cum_wait_stats, symbol_summary, score_summary)
    cum_report = _ask_gpt_for_report(cum_stats_text, "데이터 수집 시작 이후 전체 누적")

    try:
        report_ws.append_row([now_ny.strftime("%Y-%m-%d"), week_report, cum_report])
        print(f"✅ [주간리포트] {now_ny.strftime('%Y-%m-%d')} 리포트 작성 완료")
    except Exception as e:
        print(f"❌ [주간리포트] 시트 쓰기 실패: {e}")


async def _weekly_report_loop():
    """매주 토요일 오전 9시(ET)에 generate_weekly_report()를 1번 실행."""
    while True:
        now_ny = datetime.now(ZoneInfo("America/New_York"))
        days_until_sat = (5 - now_ny.weekday()) % 7  # weekday(): 월=0 ... 토=5
        target = (now_ny + timedelta(days=days_until_sat)).replace(hour=9, minute=0, second=0, microsecond=0)
        if target <= now_ny:
            target += timedelta(days=7)
        wait_seconds = (target - now_ny).total_seconds()
        await asyncio.sleep(wait_seconds)
        try:
            await asyncio.to_thread(generate_weekly_report)
        except Exception as e:
            print(f"❌ [주간리포트 루프] 오류: {e}")
        await asyncio.sleep(60)


async def _hourly_outcome_tracker_loop():
    """OUTCOME_TRACKER_INTERVAL_MINUTES(기본 30분)마다 evaluate_pending_outcomes(), sync_alpaca_trade_log(),
    sync_symbol_performance_summary(), sync_score_bucket_analysis()를 순서대로 백그라운드 스레드에서 실행."""
    while True:
        try:
            await asyncio.to_thread(evaluate_pending_outcomes)
        except Exception as e:
            print(f"❌ [결과추적 루프] 오류: {e}")
        try:
            await asyncio.to_thread(sync_alpaca_trade_log)
        except Exception as e:
            print(f"❌ [Alpaca거래내역 루프] 오류: {e}")
        try:
            await asyncio.to_thread(sync_oanda_trade_log)   # 🟥 [FIX-O1]
        except Exception as e:
            print(f"❌ [OANDA거래내역 루프] 오류: {e}")
        # 🟥 [FIX-A3] close_stale_positions()는 여기서 제거하고 _time_exit_loop()로 독립시켰다.
        #    이 체인에 묶여 있으면 앞 단계가 느릴 때 시간청산 차례가 오지 않는다.
        try:
            await asyncio.to_thread(sync_symbol_performance_summary)
        except Exception as e:
            print(f"❌ [종목별성과 루프] 오류: {e}")
        try:
            await asyncio.to_thread(sync_score_bucket_analysis)
        except Exception as e:
            print(f"❌ [점수구간분석 루프] 오류: {e}")
        await asyncio.sleep(OUTCOME_TRACKER_INTERVAL_MINUTES * 60)


async def _time_exit_loop():
    """
    🟥 [FIX-A3] 시간청산 전용 루프.
    TIME_EXIT_CHECK_MINUTES(기본 5분)마다 close_stale_positions()를 돌린다.
    시트 동기화 체인과 완전히 분리돼 있어서, 시트가 느리거나 실패해도
    포지션 정리는 항상 제시간에 돈다.
    """
    while True:
        try:
            await asyncio.to_thread(close_stale_positions)
        except Exception as e:
            print(f"❌ [시간청산 루프] 오류: {e}")
        await asyncio.sleep(max(60, TIME_EXIT_CHECK_MINUTES * 60))


@app.on_event("startup")
async def _start_background_tasks():
    asyncio.create_task(_hourly_outcome_tracker_loop())
    asyncio.create_task(_time_exit_loop())          # 🟥 [FIX-A3] 신규
    asyncio.create_task(_daily_top_movers_loop())
    asyncio.create_task(_morning_gap_scan_loop())   # 🟥 [FIX-G10] 아침 갭 스캔
    asyncio.create_task(_weekly_report_loop())


@app.post("/run_outcome_tracker")
@app.get("/run_outcome_tracker")
async def run_outcome_tracker_endpoint():
    """수동으로 즉시 결과 추적을 돌리고 싶을 때 호출 (정기 1시간 루프와 별개).
    GET도 받게 해놔서 브라우저 주소창에 URL만 붙여넣어도 바로 실행됨."""
    result = await asyncio.to_thread(evaluate_pending_outcomes)
    return JSONResponse(content=result)


@app.post("/sync_alpaca_trade_log")
@app.get("/sync_alpaca_trade_log")
async def sync_alpaca_trade_log_endpoint():
    """'Alpaca 거래내역' 탭을 지금 바로 갱신하고 싶을 때 호출 (정기 1시간 루프와 별개)."""
    await asyncio.to_thread(sync_alpaca_trade_log)
    return JSONResponse(content={"status": "done"})


@app.post("/sync_oanda_trade_log")
@app.get("/sync_oanda_trade_log")
async def sync_oanda_trade_log_endpoint():
    """🟥 [FIX-O1] 'OANDA 거래내역' 탭을 지금 바로 갱신."""
    await asyncio.to_thread(sync_oanda_trade_log)
    return {"status": "ok", "message": "'OANDA 거래내역' 탭 갱신 완료"}


@app.post("/close_stale_positions")
@app.get("/close_stale_positions")
async def close_stale_positions_endpoint():
    """STOCK_TIME_EXIT_MINUTES(기본 90분)가 지난 미청산 포지션들을 지금 바로 강제 청산하고 싶을 때 호출
    (정기 루프와 별개). 'Alpaca 거래내역'이 먼저 최신 상태여야 정확하다."""
    result = await asyncio.to_thread(close_stale_positions)
    return JSONResponse(content=result)


@app.post("/sync_symbol_performance")
@app.get("/sync_symbol_performance")
async def sync_symbol_performance_endpoint():
    """'종목별 성과분석' 탭을 지금 바로 갱신하고 싶을 때 호출 (정기 1시간 루프와 별개).
    'Alpaca 거래내역' 탭이 먼저 갱신돼 있어야 의미 있는 데이터가 나온다."""
    await asyncio.to_thread(sync_symbol_performance_summary)
    return JSONResponse(content={"status": "done"})


@app.post("/oanda_instruments")
@app.get("/oanda_instruments")
async def oanda_instruments_endpoint(q: str = "", refresh: int = 1):
    """🟥 [FIX-S2] 이 OANDA 계좌가 '실제로' 거래할 수 있는 상품 전체 목록.

    BLOCKED_SYMBOL_UNRESOLVED 가 떴을 때 추측하지 말고 이걸로 확인한다.
    q=XAG 처럼 검색어를 주면 그 글자가 들어간 상품만 골라 보여준다.
    """
    def _go():
        if refresh:
            _OANDA_INSTR_CACHE["t"] = 0.0          # 24시간 캐시 무시하고 새로 받는다
        if not (OANDA_API_KEY and ACCOUNT_ID):
            return {"오류": "OANDA_API_KEY 또는 ACCOUNT_ID 가 설정되지 않았습니다"}
        try:
            r = requests.get(f"{OANDA_BASE_URL}/v3/accounts/{ACCOUNT_ID}/instruments",
                             headers={"Authorization": f"Bearer {OANDA_API_KEY}"}, timeout=20)
        except Exception as e:
            return {"오류": f"조회 실패: {e}"}
        if not r.ok:
            return {"오류": f"status={r.status_code}", "본문": r.text[:300]}

        items = (r.json() or {}).get("instruments", []) or []
        names = {str(i.get("name") or "") for i in items}
        names.discard("")
        if names:                                   # 캐시도 같이 갱신해 둔다
            _OANDA_INSTR_CACHE.update(t=_t.time(), set=names)

        by_type = {}
        for i in items:
            by_type.setdefault(str(i.get("type") or "?"), []).append(str(i.get("name") or ""))
        for k in by_type:
            by_type[k].sort()

        watch = ["XAU_USD", "XAG_USD", "SPX500_USD", "NAS100_USD", "WTICO_USD",
                 "USD_JPY", "EUR_USD", "GBP_USD", "AUD_USD"]
        out = {
            "계좌": ACCOUNT_ID,
            "엔드포인트": OANDA_BASE_URL,
            "실계좌여부": OANDA_LIVE,
            "총_상품수": len(items),
            "종류별_개수": {k: len(v) for k, v in sorted(by_type.items())},
            "주요종목_거래가능": {w: (w in names) for w in watch},
        }
        if q:
            key = q.strip().upper().replace("_", "")
            out["검색결과"] = sorted(n for n in names if key in n.replace("_", ""))
        else:
            out["전체목록"] = {k: v for k, v in sorted(by_type.items())}
        return out

    return JSONResponse(content=await asyncio.to_thread(_go))


@app.post("/check_data_delay")
@app.get("/check_data_delay")
async def check_data_delay_endpoint(request: Request, symbol: str = "SPY"):
    """🟥 [G9] 내 계정의 실제 데이터 지연이 몇 분인지 측정한다.

    장중(09:30~16:00 ET)에 호출해야 의미가 있다.
    0~2분이면 개장 직후 스캔이 가능하고, 15분 이상이면 그만큼 미뤄야 한다.
    """
    if await _warn_if_alert_body(request, "/check_data_delay"):
        return JSONResponse(content={
            "status": "wrong_url",
            "오류": "매매 알림이 잘못된 주소로 왔습니다.",
            "고칠_것": "트레이딩뷰 알람 → Notifications → Webhook URL 을 '/webhook' 으로 바꾸세요.",
        }, status_code=200)
    _delay_probe["minutes"] = None            # 캐시 무효화 후 재측정
    d = await asyncio.to_thread(measure_data_delay, symbol)
    ny = datetime.now(ZoneInfo("America/New_York"))
    hhmm = ny.hour * 100 + ny.minute
    intraday = 930 <= hhmm < 1600
    return {
        "측정_종목": symbol,
        "피드": ALPACA_FEED,
        "측정된_지연_분": d,
        "현재_ET": ny.strftime("%H:%M"),
        "장중_여부": intraday,
        "권장_스캔시각": "0932,0945,1015" if d <= 3 else f"지연 {d}분 → 개장 후 {d + 2}분 뒤부터",
        "메모": ("장중이 아니면 측정이 부정확합니다. 09:30~16:00 ET 에 다시 호출하세요."
                 if not intraday else "정상 측정"),
    }


@app.post("/scan_morning_gappers")
@app.get("/scan_morning_gappers")
async def scan_morning_gappers_endpoint(request: Request, top: int = 0):
    """🟥 [FIX-G10] 아침 갭 스캔을 지금 바로 실행.

    자동 실행은 GAP_SCAN_TIMES(기본 09:20 / 10:00 ET). 이건 수동 트리거다.
    """
    if await _warn_if_alert_body(request, "/scan_morning_gappers"):
        return {"status": "wrong_url",
                "고칠_것": "트레이딩뷰 알람의 Webhook URL 을 '/webhook' 으로 바꾸세요."}
    res = await asyncio.to_thread(scan_morning_gappers, top or None)
    return res


# ============================================================
# 🟥 [FIX-WJ2] 알람이 '엉뚱한 주소' 로 갔을 때 조용히 넘어가지 않는다.
# ------------------------------------------------------------
#  실제로 겪은 일: 트레이딩뷰 알람의 Webhook URL 이 /webhook 이 아니라
#  /check_data_delay 로 되어 있었다. 그 엔드포인트는 본문을 읽지도 않고
#  200 OK 를 돌려준다 → 트레이딩뷰는 '전송 성공' 으로 표시하고,
#  봇은 아무것도 하지 않고, 시트에는 한 줄도 안 남는다.
#  어디를 봐도 정상인데 거래만 안 되는, 제일 찾기 어려운 종류의 고장이다.
#
#  ⚠️ 미들웨어로 모든 POST 본문을 가로채는 방법은 쓰지 않는다.
#     Starlette 미들웨어에서 body 를 읽으면 아래 핸들러가 본문을 다시 못 읽어
#     **웹훅 전체가 멈출 수** 있다. 오타로 붙일 만한 주소에만 직접 검사를 단다.
# ============================================================
_ALERT_LIKE_KEYS = (b'"signal"', b'"strategy_name"', b'"alert_name"', b'"tf"')


async def _warn_if_alert_body(request, path: str):
    """이 주소로 매매 알림이 잘못 온 거면 로그와 장부에 남기고 True 를 돌려준다."""
    try:
        if request is None or request.method != "POST":
            return False
        body = await request.body()
        if not body or not any(k in body for k in _ALERT_LIKE_KEYS):
            return False
        print("=" * 70)
        print(f"🚨 [주소오류] 매매 알림이 '{path}' 로 왔습니다. 여기는 알림 처리 주소가 아닙니다.")
        print(f"   트레이딩뷰 알람의 Webhook URL 을 '/webhook' 으로 고쳐야 합니다.")
        print(f"   본문: {body[:200].decode('utf-8', 'ignore')}")
        print("=" * 70)
        _journal_outcome(
            _journal_webhook(body, f"⚠️ 잘못된 주소 {path}"),
            f"❌ 처리 안 됨 — Webhook URL 이 '{path}' 입니다. '/webhook' 으로 고치세요.")
        return True
    except Exception as e:
        print(f"⚠️ [주소오류 감지] 건너뜀: {e}")
        return False


@app.post("/recent_webhooks")
@app.get("/recent_webhooks")
async def recent_webhooks_endpoint(n: int = 30, s: str = ""):
    """🟥 [FIX-WJ1] 최근 도착한 웹훅 목록.

    "알람은 울렸는데 아무 일도 안 일어났다" 를 판정하는 용도다.
        /recent_webhooks          최근 30건
        /recent_webhooks?s=SPY    SPY 만
        /recent_webhooks?n=100    최근 100건
    여기에 안 보이면 요청이 봇에 도착하지 않은 것 → 트레이딩뷰 쪽 문제다.
    """
    with _wj_lock:
        items = list(_webhook_journal)
    if s:
        key = s.strip().upper()
        items = [r for r in items if key in str(r.get("심볼", "")).upper()
                 or key in str(r.get("본문", "")).upper()]
    items = items[-max(1, min(n, _WEBHOOK_JOURNAL_MAX)):]
    items.reverse()                     # 최신이 위로
    return {
        "총_보관중": len(_webhook_journal),
        "보관_한도": _WEBHOOK_JOURNAL_MAX,
        "주의": "재배포/재시작하면 이 목록은 초기화됩니다(메모리 보관).",
        "목록": items,
    }


@app.post("/check_symbol")
@app.get("/check_symbol")
async def check_symbol_endpoint(s: str = ""):
    """🟥 [FIX-S1] 심볼 하나가 어디로 어떻게 라우팅되는지 즉시 확인.

    알림을 기다리지 않고 브라우저에서 바로 볼 수 있다.
        /check_symbol?s=USDJPY
        /check_symbol?s=PLTR
    """
    if not s:
        return {"error": "?s=심볼 을 붙일 것 (예: /check_symbol?s=USDJPY)"}

    raw = s.strip().upper()
    norm, note = await asyncio.to_thread(normalize_symbol, raw)
    is_stock = is_stock_pair(norm)
    venue = "Alpaca (주식)" if is_stock else "OANDA (FX/원자재/지수)"
    gran = STOCK_BASE_GRANULARITY if is_stock else FX_BASE_GRANULARITY

    # 실제로 캔들이 오는지까지 확인한다 — 이게 최종 판정이다
    try:
        df = await asyncio.to_thread(get_candles, norm, gran, 30)
        bars = 0 if df is None else len(df)
    except Exception as e:
        bars = 0
        note += f" / 캔들조회 예외: {e}"

    tradable = None
    if not is_stock:
        known = await asyncio.to_thread(get_oanda_instruments)
        tradable = (norm in known) if known else None

    ok = bars > 0
    return {
        "입력": raw,
        "정규화": norm,
        "변환메모": note,
        "라우팅": venue,
        "기본_시간축": gran,
        "캔들_수신": bars,
        "OANDA_거래가능": tradable,
        "판정": "✅ 정상 — 알림이 오면 처리됨" if ok else
                "❌ 이 심볼로는 캔들을 못 받는다 — 알림이 와도 거래·기록이 안 된다",
        "그룹": portfolio_group_for(norm),
    }


@app.post("/portfolio_exposure")
@app.get("/portfolio_exposure")
async def portfolio_exposure_endpoint():
    """🟥 [FIX-P1] 지금 봇이 보는 '분야별 쏠림'을 그대로 돌려준다.

    알람을 기다리지 않고 바로 확인할 수 있다. 상관 사이징이 왜 수량을 줄였는지
    알고 싶을 때 여기부터 보면 된다.
    """
    snap = await asyncio.to_thread(summarize_portfolio_exposure)
    return {
        "ok": snap.get("ok"),
        "equity_usd": snap.get("equity"),
        "total_gross_pct": snap.get("total_gross_pct"),
        "group_max_pct": PORTFOLIO_GROUP_MAX_PCT,
        "total_max_pct": PORTFOLIO_TOTAL_MAX_PCT,
        "sizing_enabled": PORTFOLIO_SIZING_ENABLED,
        "groups": snap.get("groups"),
        "positions": snap.get("positions"),
    }


@app.post("/sync_top_active_candidates")
@app.get("/sync_top_active_candidates")
async def sync_top_active_candidates_endpoint():
    """'오늘의 추천 후보' 탭을 지금 바로 갱신하고 싶을 때 호출 (정기 매일 오전 10시 자동 실행과 별개)."""
    await asyncio.to_thread(sync_top_active_candidates)
    return JSONResponse(content={"status": "done"})


@app.post("/sync_score_bucket_analysis")
@app.get("/sync_score_bucket_analysis")
async def sync_score_bucket_analysis_endpoint():
    """'스코어대별 성과분석' 탭을 지금 바로 갱신하고 싶을 때 호출 (정기 1시간 루프와 별개).
    'Alpaca 거래내역'이 먼저 갱신돼 있어야 의미 있는 데이터가 나온다."""
    await asyncio.to_thread(sync_score_bucket_analysis)
    return JSONResponse(content={"status": "done"})


@app.post("/generate_weekly_report")
@app.get("/generate_weekly_report")
async def generate_weekly_report_endpoint():
    """'주간 리포트' 탭에 지금 바로 리포트를 1건 작성하고 싶을 때 호출 (정기 토요일 오전 9시 자동 실행과 별개).
    'Alpaca 거래내역'/'종목별 성과분석'/'스코어대별 성과분석'이 먼저 갱신돼 있어야 의미 있는 리포트가 나온다."""
    await asyncio.to_thread(generate_weekly_report)
    return JSONResponse(content={"status": "done"})


def get_last_trade_time():
    try:
        with open("/tmp/last_trade_time.txt", "r") as f:
            return datetime.fromisoformat(f.read().strip())
    except:
        return None
