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
import math # math 모듈 추가 임포트
import re   # re 모듈 추가 임포트 (parse_gpt_feedback에서 사용)

app = FastAPI()

OANDA_API_KEY = os.getenv("OANDA_API_KEY")
ACCOUNT_ID = os.getenv("ACCOUNT_ID")
openai.api_key = os.getenv("OPENAI_API_KEY")

# Google Sheet 설정 (환경 변수 또는 직접 정의)
GOOGLE_SHEET_NAME = os.getenv("GOOGLE_SHEET_NAME", "민균 FX trading result")
STRATEGY_SETTINGS_SHEET_NAME = os.getenv("STRATEGY_SETTINGS_SHEET_NAME", "StrategySettings") # 새로운 설정 시트 이름

# Google Sheet 인증 정보
def get_google_sheet_client():
    """Google Sheet API 클라이언트를 인증하고 반환합니다."""
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    # Render 환경 변수를 통해 인증 정보 파일 경로 설정
    creds = ServiceAccountCredentials.from_json_keyfile_name("/etc/secrets/google_credentials.json", scope)
    client = gspread.authorize(creds)
    return client

def get_strategy_settings():
    """Google Sheet에서 전략 설정을 읽어옵니다."""
    client = get_google_sheet_client()
    try:
        # 스프레드시트와 특정 시트를 엽니다.
        settings_sheet = client.open(GOOGLE_SHEET_NAME).worksheet(STRATEGY_SETTINGS_SHEET_NAME)
        # B1 셀에서 최소 시그널 점수를 읽어온다고 가정 (A1: MIN_SIGNAL_SCORE, B1: 실제 값)
        min_signal_score_str = settings_sheet.acell('B1').value 
        print(f"✅ 설정 시트에서 MIN_SIGNAL_SCORE 값 읽음: {min_signal_score_str}")
        
        try:
            min_signal_score = int(min_signal_score_str)
        except (ValueError, TypeError):
            print(f"⚠️ MIN_SIGNAL_SCORE 값 '{min_signal_score_str}'이 숫자가 아닙니다. 기본값 3을 사용합니다.")
            min_signal_score = 3 # 유효하지 않은 값일 경우 기본값
            
        return {"min_signal_score": min_signal_score}
    except gspread.exceptions.SpreadsheetNotFound:
        print(f"⚠️ Google Sheet '{GOOGLE_SHEET_NAME}'를 찾을 수 없습니다. 기본 설정을 사용합니다.")
        return {"min_signal_score": 3}
    except gspread.exceptions.WorksheetNotFound:
        print(f"⚠️ 설정 시트 '{STRATEGY_SETTINGS_SHEET_NAME}'를 찾을 수 없습니다. 기본 설정을 사용합니다.")
        return {"min_signal_score": 3}
    except Exception as e:
        print(f"❌ 전략 설정 로딩 중 오류 발생: {e}. 기본 설정을 사용합니다.")
        return {"min_signal_score": 3}


def analyze_highs_lows(candles, window=20):
    highs = candles['high'].tail(window).dropna()
    lows = candles['low'].tail(window).dropna()

    if highs.empty or lows.empty:
        return {"new_high": False, "new_low": False}

    # 현재 캔들의 고점/저점을 이전 캔들들과 비교
    new_high = highs.iloc[-1] > highs.iloc[:-1].max() if len(highs) > 1 else False
    new_low = lows.iloc[-1] < lows.iloc[:-1].min() if len(lows) > 1 else False
    return {
        "new_high": new_high,
        "new_low": new_low
    }

@app.post("/webhook")
async def webhook(request: Request):
    print("✅ STEP 1: 웹훅 진입")
    try:
        raw_body = await request.body()
        print(f"DEBUG: 수신된 웹훅 Raw Body: {raw_body.decode('utf-8')}") # 웹훅 원본 내용 로그
        data = json.loads(raw_body)
    except json.JSONDecodeError as e:
        print(f"❌ JSON 파싱 실패: {e} | Raw Body 내용: {raw_body.decode('utf-8')}")
        return JSONResponse(
            content={"error": f"유효하지 않은 JSON 페이로드: {e}"},
            status_code=400
        )
    except Exception as e:
        print(f"❌ 웹훅 요청 처리 중 예상치 못한 초기 오류 발생: {e}")
        return JSONResponse(
            content={"error": f"웹훅 처리 중 예상치 못한 오류: {e}"},
            status_code=400
        )
    print(f"✅ STEP 2: 데이터 수신 완료 | pair: {pair}")

    price_raw = data.get("price")
    try:
        price = float(price_raw)
    except (TypeError, ValueError):
        import re
        numeric_match = re.search(r"\d+\.?\d*", str(price_raw))
        price = float(numeric_match.group()) if numeric_match else None
    print(f"✅ STEP 3: 가격 파싱 완료 | price: {price}")

    if price is None:
        print("❌ price 필드를 float으로 변환할 수 없습니다. 요청 중단.")
        return JSONResponse(
            content={"error": "price 필드를 float으로 변환할 수 없습니다"},
            status_code=400
        )

    signal = data.get("signal")
    alert_name = data.get("alert_name", "기본알림")

    candles = get_candles(pair, "M30", 250)
    print("✅ STEP 4: 캔들 데이터 수신")
    
    if candles is None or candles.empty:
        print("❌ 캔들 데이터를 불러올 수 없습니다. 요청 중단.")
        return JSONResponse(content={"error": "캔들 데이터를 불러올 수 없음"}, status_code=400)

    # ✅ 최근 10봉 기준으로 지지선/저항선 다시 설정 (중복 제거)
    candles_recent = candles.tail(10)
    support_resistance = {
        "support": candles_recent["low"].min(),
        "resistance": candles_recent["high"].max()
    }
    
    close = candles["close"]

    # 지표 계산 변수 초기화
    rsi = pd.Series([np.nan])
    stoch_rsi_series = pd.Series([np.nan])
    stoch_rsi = 0
    macd = pd.Series([np.nan])
    macd_signal = pd.Series([np.nan])
    boll_up = pd.Series([np.nan])
    boll_mid = pd.Series([np.nan])
    boll_low = pd.Series([np.nan])
    atr = np.nan # ATR 초기화

    # ✅ 들여쓰기 수정 및 NaN/데이터 부족 처리 강화
    if len(close.dropna()) < 20: # 최소 20봉의 유효한 close 데이터 필요
        print("❌ close 데이터 부족 (20봉 미만) → 지표 계산 실패 예상. NaN 또는 0으로 설정.")
    else:
        rsi = calculate_rsi(close)
        # RSI가 NaN일 경우 Stoch RSI 계산 방지
        if not rsi.dropna().empty:
            stoch_rsi_series = calculate_stoch_rsi(rsi)
            stoch_rsi = stoch_rsi_series.dropna().iloc[-1] if not stoch_rsi_series.dropna().empty else 0
        else:
            print("❌ RSI 계산 실패로 Stoch RSI 계산 건너뜀.")

        macd, macd_signal = calculate_macd(close)
        boll_up, boll_mid, boll_low = calculate_bollinger_bands(close)
        atr = calculate_atr(candles).iloc[-1] if not calculate_atr(candles).dropna().empty else np.nan

    print(f"✅ STEP 5: 보조지표 계산 완료 | RSI: {safe_float(rsi.iloc[-1])}")
    
    pattern = detect_candle_pattern(candles)
    # Boll_mid가 NaN일 수 있으므로 유효성 검사 추가
    current_boll_mid = boll_mid.iloc[-1] if not boll_mid.empty and not np.isnan(boll_mid.iloc[-1]) else np.nan
    trend = detect_trend(candles, rsi, boll_mid if not boll_mid.empty else pd.Series([np.nan])) # 유효하지 않은 경우 np.nan 포함 Series 전달
    liquidity = estimate_liquidity(candles)
    news = fetch_forex_news()
    
    high_low_analysis = analyze_highs_lows(candles)
    fibo_levels = calculate_fibonacci_levels(candles["high"].max(), candles["low"].min())

    payload = {
        "pair": pair,
        "price": price,
        "signal": signal,
        "rsi": safe_float(rsi.iloc[-1]),
        "macd": safe_float(macd.iloc[-1]),
        "macd_signal": safe_float(macd_signal.iloc[-1]),
        "stoch_rsi": safe_float(stoch_rsi),
        "bollinger_upper": safe_float(boll_up.iloc[-1]),
        "bollinger_lower": safe_float(boll_low.iloc[-1]),
        "pattern": pattern,
        "trend": trend,
        "liquidity": liquidity,
        "support": safe_float(support_resistance["support"]),
        "resistance": safe_float(support_resistance["resistance"]),
        "news": news,
        "new_high": bool(high_low_analysis["new_high"]),
        "new_low": bool(high_low_analysis["new_low"]),
        "atr": safe_float(atr)
    }
    
    signal_score = 0
    reasons = []

    # BUY 신호 점수 계산
    if signal == "BUY":
        if not rsi.empty and not np.isnan(rsi.iloc[-1]) and rsi.iloc[-1] < 45:
            signal_score += 2
            reasons.append("RSI < 45")
        else:
            reasons.append("RSI 조건 미달 또는 계산 실패")

        if not macd.empty and not macd_signal.empty and not np.isnan(macd.iloc[-1]) and not np.isnan(macd_signal.iloc[-1]) and macd.iloc[-1] > macd_signal.iloc[-1]:
            signal_score += 2
            reasons.append("MACD 골든크로스")
        else:
            reasons.append("MACD 조건 미달 또는 계산 실패")

        if not stoch_rsi_series.dropna().empty:
            stoch_last = stoch_rsi_series.dropna().iloc[-1]
            if stoch_last > 0.5:
                signal_score += 1
                reasons.append("Stoch RSI 상승 모멘텀")
            else:
                reasons.append("Stoch RSI 상승 모멘텀 아님")
        else:
            reasons.append("Stoch RSI 값 부족 → 점수 제외")

        if trend == "UPTREND":
            signal_score += 1
            reasons.append("상승 추세")
        else:
            reasons.append("상승 추세 아님")

    # SELL 신호 점수 계산
    elif signal == "SELL":
        if not rsi.empty and not np.isnan(rsi.iloc[-1]) and rsi.iloc[-1] > 55:
            signal_score += 2
            reasons.append("RSI > 55")
        else:
            reasons.append("RSI 조건 미달 또는 계산 실패")

        if not macd.empty and not macd_signal.empty and not np.isnan(macd.iloc[-1]) and not np.isnan(macd_signal.iloc[-1]) and macd.iloc[-1] < macd_signal.iloc[-1]:
            signal_score += 2
            reasons.append("MACD 데드크로스")
            
        else:
            reasons.append("MACD 조건 미달 또는 계산 실패")

        if not stoch_rsi_series.dropna().empty:
            stoch_last = stoch_rsi_series.dropna().iloc[-1]
            if stoch_last < 0.5:
                signal_score += 1
                reasons.append("Stoch RSI 하락 모멘텀")
            else:
                reasons.append("Stoch RSI 하락 모멘텀 아님")
        else:
            reasons.append("Stoch RSI 값 부족 → 점수 제외")

        if trend == "DOWNTREND":
            signal_score += 1
            reasons.append("하락 추세")
        else:
            reasons.append("하락 추세 아님")

    
    # ✅ 동적으로 가져온 최소 시그널 점수 사용
    strategy_settings = get_strategy_settings()
    min_signal_score_threshold = strategy_settings.get("min_signal_score", 3) # 기본값 3
    print(f"✅ MIN_SIGNAL_SCORE (설정 시트에서 불러옴): {min_signal_score_threshold}")

    gpt_feedback = "GPT 분석 생략: 점수 미달"
    decision, tp, sl = "WAIT", None, None
    gpt_decision = "WAIT" # gpt_decision 초기화

    if signal_score >= min_signal_score_threshold: # ✅ 동적으로 조정된 임계값 적용
        gpt_feedback = analyze_with_gpt(payload)
        print("✅ STEP 6: GPT 응답 수신 완료")
        gpt_decision, _, _ = parse_gpt_feedback(gpt_feedback) # GPT의 raw decision 저장
        decision = gpt_decision # 시스템의 최종 결정도 GPT 판단과 일치시킴
        
        # 📌 TP/SL은 무조건 고정값으로 설정
        # price가 None이 아닐 경우에만 계산 (price None 체크는 이미 위에서 함)
        if price is not None:
            pip_value = 0.01 if "JPY" in pair else 0.0001
            # TP/SL 계산 후 반올림
            raw_tp = price + pip_value * 15 if decision == "BUY" else price - pip_value * 15
            raw_sl = price - pip_value * 10 if decision == "BUY" else price + pip_value * 10
            
            # JPY 통화쌍은 소수점 0자리, 그 외는 5자리
            if "JPY" in pair:
                tp = round(raw_tp)
                sl = round(raw_sl)
            else:
                tp = round(raw_tp, 5)
                sl = round(raw_sl, 5)
        else:
            print("⚠️ 가격(price)이 유효하지 않아 TP/SL 고정값 설정 불가능.")

        gpt_feedback += "\n⚠️ TP/SL은 GPT 무시, 고정값 적용 (15pip / 10pip)"
        
    else:
        print(f"🚫 GPT 분석 생략: 점수 {signal_score}점 (최소 {min_signal_score_threshold}점 미만)")
    
    print(f"✅ STEP 7: GPT 해석 완료 | decision: {decision}, TP: {tp}, SL: {sl}")
    
    # ❌ GPT가 WAIT이면 주문하지 않음
    if decision == "WAIT":
        print("🚫 GPT 판단: WAIT → 주문 실행하지 않음")
        outcome_analysis = "WAIT 또는 주문 미실행"
        adjustment_suggestion = ""
        
        log_trade_result(
            pair, signal, decision, signal_score,
            "\n".join(reasons) + f"\nATR: {safe_float(atr)}",
            {}, safe_float(rsi.iloc[-1]), 
            safe_float(macd.iloc[-1]), 
            safe_float(stoch_rsi),
            pattern, trend, fibo_levels, gpt_decision, news, gpt_feedback, # gpt_decision 추가
            alert_name, tp, sl, price, None, # pnl은 여전히 None
            outcome_analysis, adjustment_suggestion, [],
            atr
        )
        return JSONResponse(content={"status": "WAIT", "message": "GPT가 WAIT 판단"})

    # --- 주문 실행 로직 통합 및 정리 ---
    should_execute = False
    allow_conditional_trade = False # ✅ 이 변수를 명시적으로 정의 (현재는 비활성 상태)
    # 이 부분에 last_trade_time을 가져와 allow_conditional_trade를 True로 설정하는 로직 추가 가능

    # 1️⃣ 기본 진입 조건: GPT가 BUY/SELL 판단 + 점수 (동적 임계값) 이상
    if decision in ["BUY", "SELL"] and signal_score >= min_signal_score_threshold: # ✅ 동적 임계값 적용
        should_execute = True
    # 2️⃣ 조건부 진입: 최근 2시간 거래 없으면 점수 4점 미만이어도 진입 허용 (allow_conditional_trade가 True일 때)
    # 현재 allow_conditional_trade = False 이므로 이 블록은 실행되지 않음
    elif allow_conditional_trade and decision in ["BUY", "SELL"]: 
        gpt_feedback += "\n⚠️ 조건부 진입: 최근 2시간 거래 없음 → 점수 기준 완화"
        should_execute = True

    result = {} # result 초기화
    price_movements = []
    pnl = None # PnL 조회 로직 추가 필요

    if should_execute and tp is not None and sl is not None: # TP/SL이 유효할 때만 주문 시도
        units = 100000 if decision == "BUY" else -100000
        # digits는 OANDA API 요청에 필요하지만, TP/SL은 이미 위에서 반올림됨
        digits = 3 if pair.endswith("JPY") else 5 # 이 값은 OANDA에 전달되는 값의 소수점 자릿수 결정 (여기서는 라운딩에 사용)

        print(f"[DEBUG] 조건 충족 → 실제 주문 실행: {pair}, units={units}, tp={tp}, sl={sl}, digits={digits}")
        result = place_order(pair, units, tp, sl, digits)
        print("✅ STEP 9: 주문 결과 확인 |", result)

        # 여기에 실제 PnL을 조회하는 로직 (예: OANDA 포지션 조회 API) 추가 필요
        # pnl = get_current_pnl(ACCOUNT_ID, pair)

        executed_time = datetime.utcnow() # 이 변수는 현재 사용되지 않음
        candles_post = get_candles(pair, "M30", 8) # 거래 후 캔들 데이터 수집
        if candles_post is not None: # 캔들 데이터가 있을 경우에만 처리
            price_movements = candles_post[["high", "low"]].to_dict("records")
    else:
        print("🚫 최종 판단: 주문 미실행 (조건 불충족 또는 TP/SL 미설정)")
        result = {"status": "order_skipped", "message": "주문 조건 불충족 또는 TP/SL 없음"} # 결과에 스킵 정보 추가

    # PnL이 계산되거나, 주문이 실행되었을 때 outcome_analysis 설정
    if should_execute and isinstance(result, dict) and "order_placed" in result.get("status", ""):
        # PnL 로직이 추가되었다면 이 부분을 활성화
        # if pnl is not None:
        #     if pnl > 0:
        #         outcome_analysis = "성공: 수익 실현"
        #     elif pnl < 0:
        #         outcome_analysis = "실패: 손실 발생"
        #     else:
        #         outcome_analysis = "보류: 실현손익 0"
        # else:
        #     outcome_analysis = "보류: 실현손익 미확정"
        outcome_analysis = "주문 실행됨 (PnL 미확정)" # PnL 로직이 없으므로 임시 설정
    else:
        outcome_analysis = "WAIT 또는 주문 미실행" # GPT WAIT 또는 should_execute가 False일 경우

    adjustment_suggestion = ""
    if outcome_analysis.startswith("실패"):
        if sl is not None and tp is not None and price is not None: # 유효한 값일 때만 비교
            if abs(sl - price) < abs(tp - price):
                adjustment_suggestion = "SL 터치 → SL 너무 타이트했을 수 있음, 다음 전략에서 완화 필요"
            elif abs(tp - price) < abs(sl - price):
                adjustment_suggestion = "TP 거의 닿았으나 실패 → TP 약간 보수적일 필요 있음"
        else:
            adjustment_suggestion = "손절/익절 분석 불가 (TP/SL/Price 미정)"
            
    print(f"✅ STEP 10: 전략 요약 저장 호출 | decision: {decision}, TP: {tp}, SL: {sl}")
    log_trade_result(
        pair, signal, decision, signal_score,
        "\n".join(reasons) + f"\nATR: {safe_float(atr)}",
        result, 
        safe_float(rsi.iloc[-1]), 
        safe_float(macd.iloc[-1]), 
        safe_float(stoch_rsi),
        pattern, trend, fibo_levels, gpt_decision, news, gpt_feedback, # gpt_decision 추가
        alert_name, tp, sl, price, pnl, # pnl은 None으로 전달
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
        print(f"⚠️ {pair} 캔들 데이터 없음.")
        # 빈 DataFrame 대신 None을 반환하여 상위 호출자에서 처리하도록 함
        return None 
    
    df = pd.DataFrame([
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
    # 시간 필드를 datetime 객체로 변환 (필요시)
    df['time'] = pd.to_datetime(df['time'])
    return df

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = -delta.clip(upper=0).rolling(window=period).mean()
    
    # RSI 계산 시 0으로 나누기 방지
    rs = gain / loss
    # loss가 0인 경우 rs가 무한대(inf)가 될 수 있으므로 처리
    rs.replace([np.inf, -np.inf], np.nan, inplace=True) 
    
    rsi = 100 - (100 / (1 + rs))
    
    # 만약 gain이 0이고 loss도 0인 경우 rsi는 50으로 간주 (변동 없을 때)
    # gain 또는 loss가 모두 NaN인 경우 (데이터 부족)에도 NaN 유지
    if gain.isnull().all() and loss.isnull().all():
        rsi = pd.Series([np.nan] * len(series), index=series.index) # 데이터 부족 시 np.nan
    elif gain.isnull().all(): # loss만 있을 때 (즉, 계속 하락만 한 경우)
        rsi = pd.Series([0.0] * len(series), index=series.index)
    elif loss.isnull().all(): # gain만 있을 때 (즉, 계속 상승만 한 경우)
        rsi = pd.Series([100.0] * len(series), index=series.index)

    print("✅ RSI tail:", rsi.tail(5))
    return rsi

def calculate_macd(series):
    ema12 = series.ewm(span=12, adjust=False).mean() # adjust=False for classic EMA
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def calculate_stoch_rsi(rsi, period=14):
    print("✅ [입력된 RSI tail]", rsi.tail(5))
    
    # RSI 시리즈에 NaN이 많으면 min/max 계산 불가. dropna() 후 충분한 데이터가 있는지 확인
    if rsi.dropna().empty or len(rsi.dropna()) < period:
        print(f"⚠️ Stoch RSI 계산을 위한 RSI 데이터 부족: 유효 데이터 {len(rsi.dropna())}개 (최소 {period}개 필요)")
        return pd.Series([np.nan] * len(rsi), index=rsi.index)

    min_rsi = rsi.rolling(window=period).min()
    max_rsi = rsi.rolling(window=period).max()
    
    # 분모가 0이 되는 경우 방지 (max_rsi == min_rsi)
    denominator = (max_rsi - min_rsi)
    stoch_rsi = (rsi - min_rsi) / denominator
    stoch_rsi.replace([np.inf, -np.inf], np.nan, inplace=True) # 무한대 값 제거
    stoch_rsi.fillna(0.5, inplace=True) # 분모 0으로 인한 NaN은 0.5로 대체 (중립)

    print("✅ [Stoch RSI 계산 결과 tail]", stoch_rsi.tail(5))
    return stoch_rsi

def calculate_bollinger_bands(series, window=20):
    if len(series.dropna()) < window:
        print(f"⚠️ 볼린저 밴드 계산을 위한 데이터 부족: 유효 데이터 {len(series.dropna())}개 (최소 {window}개 필요)")
        return pd.Series([np.nan]*len(series)), pd.Series([np.nan]*len(series)), pd.Series([np.nan]*len(series))
        
    mid = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper = mid + 2 * std
    lower = mid - 2 * std
    return upper, mid, lower

def detect_trend(candles, rsi, mid_band):
    close = candles["close"]
    # EMA 계산 시에도 데이터 부족 고려
    if len(close.dropna()) < 50: # EMA50 필요
        return "NEUTRAL"

    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()

    # 지표 값들이 유효한지 확인
    # mid_band가 Series이고 비어있을 수 있으므로 .empty 체크 추가
    if np.isnan(ema20.iloc[-1]) or np.isnan(ema50.iloc[-1]) or np.isnan(close.iloc[-1]) or (mid_band is not None and not mid_band.empty and np.isnan(mid_band.iloc[-1])):
        return "NEUTRAL"

    if ema20.iloc[-1] > ema50.iloc[-1] and close.iloc[-1] > mid_band.iloc[-1]:
        return "UPTREND"
    elif ema20.iloc[-1] < ema50.iloc[-1] and close.iloc[-1] < mid_band.iloc[-1]:
        return "DOWNTREND"
    return "NEUTRAL"

def detect_candle_pattern(candles):
    # 실제 캔들 패턴 분석 로직을 여기에 추가
    # 현재는 항상 "NEUTRAL"을 반환
    return "NEUTRAL"

def estimate_liquidity(candles):
    if candles.empty or "volume" not in candles.columns:
        return "확인불가"
    # 최근 10봉의 volume 데이터가 충분한지 확인
    recent_volumes = candles["volume"].tail(10).dropna()
    if recent_volumes.empty:
        return "낮음" # 데이터 없으면 유동성 낮다고 판단
    return "좋음" if recent_volumes.mean() > 100 else "낮음"

def fetch_forex_news():
    try:
        response = requests.get("https://www.forexfactory.com/", timeout=5)
        # 응답 상태 코드 확인
        response.raise_for_status() 
        if "High Impact Expected" in response.text:
            return "⚠️ 고위험 뉴스 존재"
        return "🟢 뉴스 영향 적음"
    except requests.exceptions.RequestException as e:
        print(f"❗ 뉴스 확인 실패: {e}")
        return "❓ 뉴스 확인 실패"

def place_order(pair, units, tp, sl, digits):
    url = f"https://api-fxpractice.oanda.com/v3/accounts/{ACCOUNT_ID}/orders"
    headers = {
        "Authorization": f"Bearer {OANDA_API_KEY}",
        "Content-Type": "application/json"
    }

    # TP/SL이 None이거나 유효하지 않은 값일 경우 None으로 설정하여 주문 요청에 포함시키지 않음
    if tp is None or math.isnan(tp) or math.isinf(tp):
        take_profit_order_details = None
    else:
        take_profit_order_details = {"price": str(tp)}

    if sl is None or math.isnan(sl) or math.isinf(sl):
        stop_loss_order_details = None
    else:
        stop_loss_order_details = {"price": str(sl)}

    data = {
        "order": {
            "instrument": pair,
            "units": str(units),
            "type": "MARKET",
            "positionFill": "DEFAULT"
        }
    }

    if take_profit_order_details:
        data["order"]["takeProfitOnFill"] = take_profit_order_details
    if stop_loss_order_details:
        data["order"]["stopLossOnFill"] = stop_loss_order_details

    try:
        response = requests.post(url, headers=headers, json=data, timeout=10)
        response.raise_for_status()  # 200 이외의 상태 코드를 받으면 예외 발생
        print(f"OANDA 응답: {response.json()}")
        return {"status": "order_placed", "details": response.json()}
    except requests.exceptions.RequestException as e:
        print(f"❌ OANDA 주문 실패: {e}")
        if response is not None:
            print(f"OANDA 에러 응답: {response.text}")
        return {"status": "order_failed", "error": str(e), "response": response.text if response is not None else "No response"}
    except Exception as e:
        print(f"❌ 예외 발생: {e}")
        return {"status": "order_failed", "error": str(e)}

def analyze_with_gpt(payload):
    prompt_messages = [
        {"role": "system", "content": """You are an expert forex trader AI. Analyze the provided market data and indicators to provide a trading decision (BUY, SELL, or WAIT) for the given currency pair.
        Always start your response with [DECISION]: BUY/SELL/WAIT.
        Then, provide a detailed analysis explaining your decision based on the provided indicators and market context.
        If you decide BUY or SELL, suggest a Take Profit (TP) and Stop Loss (SL) level in pips.
        Consider these factors:
        - Signal: The initial signal (BUY/SELL) from TradingView.
        - Price: Current market price.
        - RSI: Overbought/oversold conditions.
        - MACD: Momentum and trend changes (crosses, divergence).
        - Stoch RSI: Confirmation of momentum, particularly overbought/oversold.
        - Bollinger Bands: Volatility and potential reversals (price relative to bands).
        - Pattern: Candle patterns (e.g., NEUTRAL, bullish/bearish patterns).
        - Trend: Overall trend (UPTREND, DOWNTREND, NEUTRAL).
        - Liquidity: Market liquidity (e.g., '좋음', '낮음', '확인불가').
        - Support/Resistance: Key price levels.
        - News: Impact of upcoming news (e.g., '고위험 뉴스 존재', '뉴스 영향 적음', '뉴스 확인 실패').
        - New High/Low: Whether the current price is a new high or low in the recent window.
        - ATR: Average True Range, for volatility and potential TP/SL sizing.
        
        Example Output:
        [DECISION]: BUY
        Analysis: The market shows a strong uptrend (UPTREND) with RSI at 30 (oversold, suggesting potential bounce). MACD has just crossed above its signal line (MACD Golden Cross), indicating bullish momentum. Price is near the lower Bollinger Band, hinting at a rebound. No high-impact news. New low indicates potential reversal. ATR is 0.0010.
        TP_PIPS: 20
        SL_PIPS: 10
        """},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
    ]

    try:
        response = openai.chat.completions.create(
            model="gpt-4o", # 또는 "gpt-4", "gpt-3.5-turbo" 등 사용 가능한 모델
            messages=prompt_messages,
            temperature=0.7,
            max_tokens=500,
            timeout=30 # 30초 타임아웃 설정
        )
        gpt_response_content = response.choices[0].message.content
        print(f"✅ GPT 원본 응답: {gpt_response_content}")
        return gpt_response_content
    except openai.APITimeoutError:
        print("❌ OpenAI API Timeout Error: 요청 시간 초과")
        return "[DECISION]: WAIT\nAnalysis: OpenAI API 요청 시간 초과."
    except openai.APIConnectionError as e:
        print(f"❌ OpenAI API Connection Error: {e}")
        return f"[DECISION]: WAIT\nAnalysis: OpenAI API 연결 오류: {e}"
    except openai.APIStatusError as e:
        print(f"❌ OpenAI API Status Error: {e.status_code} - {e.response}")
        return f"[DECISION]: WAIT\nAnalysis: OpenAI API 상태 오류: {e.status_code}"
    except Exception as e:
        print(f"❌ GPT 분석 중 예외 발생: {e}")
        return "[DECISION]: WAIT\nAnalysis: GPT 분석 중 알 수 없는 오류 발생."

def parse_gpt_feedback(gpt_response):
    decision = "WAIT"
    tp_pips = None
    sl_pips = None
    
    # DECISION 파싱
    decision_match = re.search(r"\[DECISION\]:\s*(BUY|SELL|WAIT)", gpt_response)
    if decision_match:
        decision = decision_match.group(1)

    # TP_PIPS 파싱
    tp_match = re.search(r"TP_PIPS:\s*(\d+)", gpt_response)
    if tp_match:
        tp_pips = int(tp_match.group(1))

    # SL_PIPS 파싱
    sl_match = re.search(r"SL_PIPS:\s*(\d+)", gpt_response)
    if sl_match:
        sl_pips = int(sl_match.group(1))

    return decision, tp_pips, sl_pips

# NaN, Inf 값을 안전하게 처리하여 문자열로 반환하는 헬퍼 함수
def safe_float(value):
    if isinstance(value, (float, np.float64)):
        if math.isnan(value) or math.isinf(value):
            return ""
        return round(value, 5) # 기본 5자리 반올림
    elif isinstance(value, pd.Series) and not value.empty:
        return safe_float(value.iloc[-1]) # Series의 마지막 값을 처리
    return value

def log_trade_result(
    pair, signal, decision, score, reasons, result, rsi, macd, stoch_rsi, 
    pattern, trend, fibo, gpt_original_decision, news, gpt_feedback, 
    alert_name, tp, sl, price, pnl, 
    outcome_analysis, adjustment_suggestion, price_movements, atr_value
):
    """
    거래 결과를 Google Sheet에 기록합니다.
    Google Sheet 컬럼 순서에 맞게 데이터를 매핑합니다 (총 30개 컬럼).
    """
    client = get_google_sheet_client()
    sheet = client.open(GOOGLE_SHEET_NAME).sheet1 # 첫 번째 시트 (기본 시트)

    now_atlanta = datetime.now() + timedelta(hours=-4) # GMT-4 (애틀랜타 시간)

    # is_new_high, is_new_low는 이미 high_low_analysis에서 bool 값으로 가져왔을 것이므로
    # 직접 문자열로 변환합니다. (analyze_highs_lows 함수의 반환값)
    # 다만 이 함수에 직접 해당 bool 값을 전달받지 않으므로 임시로 빈 문자열 처리
    # 실제 호출 시 인자로 is_new_high_str, is_new_low_str을 받도록 변경 필요
    # 현재는 이 함수 호출 시 해당 인자가 없으므로 "" 처리
    is_new_high_str = ""
    is_new_low_str = ""

    # price_movements 리스트를 시트에 기록할 문자열로 변환
    filtered_movement_str_for_sheet = ""
    try:
        # 최근 8개 캔들만 고려
        filtered_movements_last_8 = price_movements[-8:] 
        
        movement_parts = []
        for p in filtered_movements_last_8:
            if isinstance(p, dict) and "high" in p and "low" in p:
                high_val = p['high']
                low_val = p['low']
                
                # float 또는 int 타입이 아니거나 NaN/Inf인 경우 건너뛰기
                if not isinstance(high_val, (float, int)) or not isinstance(low_val, (float, int)):
                    continue
                if math.isnan(high_val) or math.isinf(high_val) or math.isnan(low_val) or math.isinf(low_val):
                    continue
                
                movement_parts.append(f"H: {round(high_val, 5)} / L: {round(low_val, 5)}")
        
        filtered_movement_str_for_sheet = ", ".join(movement_parts)
    except Exception as e:
        print(f"❌ price_movements 변환 실패: {e}")
        filtered_movement_str_for_sheet = "error_in_conversion"

    # Google Sheet 컬럼 순서 (30개)에 맞춰 데이터 준비
    # ⚠️ 컬럼 개수 및 매핑 정확히 확인 필요
    row = [
        str(now_atlanta),                              # 1. 타임스탬프
        pair,                                          # 2. 종목
        alert_name or "",                              # 3. 알림명
        signal,                                        # 4. 신호
        decision,                                      # 5. GPT 최종 결정 (WAIT/BUY/SELL)
        score,                                         # 6. 점수 (signal_score)
        safe_float(rsi),                               # 7. RSI
        safe_float(macd),                              # 8. MACD
        safe_float(stoch_rsi),                         # 9. Stoch RSI
        pattern or "",                                 # 10. 캔들 패턴 (현재는 "NEUTRAL")
        trend or "",                                   # 11. 추세 (UPTREND/DOWNTREND/NEUTRAL)
        safe_float(fibo.get("0.382", "")),             # 12. FIBO 0.382
        safe_float(fibo.get("0.618", "")),             # 13. FIBO 0.618
        gpt_original_decision or "",                   # 14. GPT 원본 판단 (GPT가 직접 리턴한 BUY/SELL/WAIT)
        news or "",                                    # 15. 뉴스 요약 (fetch_forex_news 결과)
        reasons or "",                                 # 16. 조건 요약 (signal_score 이유)
        json.dumps(result, ensure_ascii=False) if isinstance(result, dict) else (result or "미정"), # 17. OANDA 주문 결과
        gpt_feedback or "",                            # 18. GPT 상세 분석 (GPT가 제공하는 전체 분석 리포트 내용)
        safe_float(price),                             # 19. 진입가
        safe_float(tp),                                # 20. Take Profit
        safe_float(sl),                                # 21. Stop Loss
        safe_float(pnl),                               # 22. 실현 손익 (현재는 None, PnL 구현 필요)
        # high_low_analysis["new_high"]와 high_low_analysis["new_low"]는 이 함수에 직접 인자로 전달되지 않으므로,
        # 웹훅 함수에서 이 함수를 호출할 때 인자로 추가해야 합니다.
        # 현재는 인자로 받지 않으므로 임시로 빈 문자열 처리합니다.
        # (웹훅 호출 시점에 is_new_high_str, is_new_low_str 변수가 정의되어 있어야 함)
        "", # 23. 신고점 (웹훅 함수에서 인자로 받아야 함)
        "", # 24. 신저점 (웹훅 함수에서 인자로 받아야 함)
        safe_float(atr_value),                         # 25. ATR (인자명 atr_value로 통일)
        outcome_analysis or "",                        # 26. 거래 성과 분석
        adjustment_suggestion or "",                   # 27. 전략 조정 제안
        gpt_feedback or "",                            # 28. GPT 리포트 전문 (18번과 동일)
        filtered_movement_str_for_sheet,               # 29. 최근 8봉 가격 흐름
        ""                                             # 30. 미사용/비고
    ]

    # Google Sheets에 dict, list 타입이 직접 들어가지 않도록 문자열화
    clean_row = []
    for v in row:
        if isinstance(v, (dict, list)):
            clean_row.append(json.dumps(v, ensure_ascii=False))
        elif isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            clean_row.append("") # NaN 또는 Inf는 빈 문자열로
        else:
            clean_row.append(v)
    
    # 디버깅을 위해 최종 clean_row 내용 출력
    print("✅ STEP 8: 시트 저장 직전 (clean_row):", clean_row)
    print(f"🧪 최종 clean_row 길이: {len(clean_row)}")

    try:
        sheet.append_row(clean_row)
        print("✅ STEP 8: Google Sheet에 성공적으로 기록됨.")
    except Exception as e:
        print(f"❌ Google Sheet 기록 실패: {e}")
        # 실패 시에도 에러를 리턴하지 않고 계속 진행 (웹훅은 완료되어야 함)
