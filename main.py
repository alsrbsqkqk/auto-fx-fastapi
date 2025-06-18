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
import math # math 모듈 추가

app = FastAPI()

OANDA_API_KEY = os.getenv("OANDA_API_KEY")
ACCOUNT_ID = os.getenv("ACCOUNT_ID")
openai.api_key = os.getenv("OPENAI_API_KEY")

# Google Sheets API 설정
# SERVICE_ACCOUNT_FILE_PATH 환경 변수로부터 서비스 계정 파일 경로를 가져옵니다.
SERVICE_ACCOUNT_FILE_PATH = os.getenv("SERVICE_ACCOUNT_FILE_PATH")

scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive"
]

creds = None
if SERVICE_ACCOUNT_FILE_PATH and os.path.exists(SERVICE_ACCOUNT_FILE_PATH):
    try:
        creds = ServiceAccountCredentials.from_json_keyfile_name(SERVICE_ACCOUNT_FILE_PATH, scope)
        client = gspread.authorize(creds)
        
        # SPREADSHEET_NAME 환경 변수로부터 스프레드시트 이름을 가져옵니다.
        SPREADSHEET_NAME = os.getenv("SPREADSHEET_NAME")
        if SPREADSHEET_NAME:
            sheet = client.open(SPREADSHEET_NAME).sheet1
            print(f"✅ Google 스프레드시트 '{SPREADSHEET_NAME}' 연결 성공!")
        else:
            print("❌ SPREADSHEET_NAME 환경 변수가 설정되지 않았습니다.")
            sheet = None
    except Exception as e:
        print(f"❌ Google 스프레드시트 연결 실패: {e}")
        sheet = None
else:
    print("❌ SERVICE_ACCOUNT_FILE_PATH 환경 변수가 설정되지 않았거나 파일이 존재하지 않습니다.")
    sheet = None


# --- 캔들 데이터 가져오기 함수 (수정됨) ---
def get_candles(pair, granularity, count):
    url = f"https://api-fxpractice.oanda.com/v3/instruments/{pair}/candles"
    headers = {"Authorization": f"Bearer {OANDA_API_KEY}"}
    
    # price 파라미터를 "B" (Bid)로 변경하여 시도해봅니다.
    # 만약 "B"도 안되면 "A" (Ask)로 변경하여 시도해보세요.
    params = {"granularity": granularity, "count": count, "price": "B"} # <-- M -> B로 변경
    
    print(f"DEBUG: OANDA API 요청 URL: {url}")
    print(f"DEBUG: OANDA API 요청 헤더: {{'Authorization': 'Bearer <숨김>', 'Content-Type': 'application/json'}}")
    print(f"DEBUG: OANDA API 요청 파라미터: {params}")
    
    try:
        r = requests.get(url, headers=headers, params=params, timeout=15) # 타임아웃 추가
        r.raise_for_status() # HTTP 오류가 발생하면 예외를 발생시킵니다.
        
        print(f"DEBUG: OANDA API 응답 상태 코드: {r.status_code}")
        full_response_json = r.json()
        print(f"DEBUG: OANDA API 응답 전체 JSON: {json.dumps(full_response_json, indent=2, ensure_ascii=False)}") # 전체 JSON 출력
        
        candles_data = full_response_json.get("candles", [])
        if not candles_data:
            print(f"WARNING: OANDA API에서 {pair} 캔들 데이터가 없습니다. 빈 DataFrame 반환.")
            return pd.DataFrame() # 빈 DataFrame 반환
        
        df = pd.DataFrame(candles_data)
        df["time"] = pd.to_datetime(df["time"])
        
        # 'price' 파라미터를 "B"로 설정했으므로, bid 가격을 사용합니다.
        df["open"] = df["bid"].apply(lambda x: float(x["o"])) # <-- bid로 변경
        df["high"] = df["bid"].apply(lambda x: float(x["h"])) # <-- bid로 변경
        df["low"] = df["bid"].apply(lambda x: float(x["l"]))  # <-- bid로 변경
        df["close"] = df["bid"].apply(lambda x: float(x["c"])) # <-- bid로 변경
        
        df["volume"] = df["volume"].astype(int)
        df = df[["time", "open", "high", "low", "close", "volume"]]
        return df
    except requests.exceptions.RequestException as e:
        print(f"ERROR: OANDA API 요청 중 오류 발생: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"ERROR: OANDA API 응답: {e.response.text}")
        return None # 오류 발생 시 None 반환
    except json.JSONDecodeError as e:
        print(f"ERROR: OANDA API 응답 JSON 디코딩 실패: {e} | 응답 텍스트: {r.text[:500]}...")
        return None
    except Exception as e:
        print(f"ERROR: get_candles 함수에서 알 수 없는 오류 발생: {e}")
        return None

# --- 뉴스 가져오기 함수 (임시 비활성화됨) ---
def fetch_forex_news():
    print("DEBUG: fetch_forex_news 함수 임시 비활성화됨.")
    # 실제 뉴스 API 호출 로직은 이 함수에서 제거되거나 주석 처리됩니다.
    # 테스트를 위해 항상 고정된 값을 반환합니다.
    return "뉴스 기능 임시 비활성화" # 스프레드시트에 기록될 메시지


# --- 지표 계산 함수 (변동 없음) ---
def calculate_rsi(candles, window=14):
    delta = candles['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(candles, fast=12, slow=26, signal=9):
    exp1 = candles['close'].ewm(span=fast, adjust=False).mean()
    exp2 = candles['close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_stoch_rsi(rsi, k_window=3, d_window=3, rsi_window=14):
    min_rsi = rsi.rolling(window=rsi_window).min()
    max_rsi = rsi.rolling(window=rsi_window).max()
    stoch_rsi = ((rsi - min_rsi) / (max_rsi - min_rsi)) * 100
    k_line = stoch_rsi.rolling(window=k_window).mean()
    d_line = k_line.rolling(window=d_window).mean()
    return k_line, d_line

def calculate_bollinger_bands(candles, window=20, num_std_dev=2):
    rolling_mean = candles['close'].rolling(window=window).mean()
    rolling_std = candles['close'].rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std_dev)
    lower_band = rolling_mean - (rolling_std * num_std_dev)
    return rolling_mean, upper_band, lower_band

def analyze_highs_lows(candles, window=20):
    if candles.empty:
        return {"new_high": False, "new_low": False}
    
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

def analyze_trend(candles, short_window=10, long_window=50):
    if candles.empty:
        return "N/A"
    short_ma = candles['close'].rolling(window=short_window).mean()
    long_ma = candles['close'].rolling(window=long_window).mean()

    if short_ma.iloc[-1] > long_ma.iloc[-1]:
        return "상승"
    elif short_ma.iloc[-1] < long_ma.iloc[-1]:
        return "하락"
    else:
        return "횡보"

def analyze_pattern(candles):
    if candles.empty or len(candles) < 3:
        return "N/A"
    
    last_three = candles.tail(3)
    if len(last_three) < 3:
        return "N/A"

    open1, close1 = last_three['open'].iloc[-3], last_three['close'].iloc[-3]
    open2, close2 = last_three['open'].iloc[-2], last_three['close'].iloc[-2]
    open3, close3 = last_three['open'].iloc[-1], last_three['close'].iloc[-1]

    # 상승 (양봉), 하락 (음봉)
    is_bullish1 = close1 > open1
    is_bullish2 = close2 > open2
    is_bullish3 = close3 > open3

    # 망치형 (Hammer)
    if (close3 > open3 and # 양봉
        (open3 - last_three['low'].iloc[-1]) > 2 * (close3 - open3) and # 긴 아래 꼬리
        (last_three['high'].iloc[-1] - close3) < (close3 - open3)): # 짧은 윗 꼬리
        return "망치형 (Hammer)"
    
    if (open3 > close3 and # 음봉
        (close3 - last_three['low'].iloc[-1]) > 2 * (open3 - close3) and # 긴 아래 꼬리
        (last_three['high'].iloc[-1] - open3) < (open3 - close3)): # 짧은 윗 꼬리
        return "행잉맨 (Hanging Man)"

    # 도지 (Doji)
    if abs(open3 - close3) < (last_three['high'].iloc[-1] - last_three['low'].iloc[-1]) * 0.1:
        return "도지 (Doji)"

    # 상승 반전 패턴: Morning Star (샛별형)
    if (open1 > close1 and # 1: 음봉
        abs(open2 - close2) < (last_three['high'].iloc[-2] - last_three['low'].iloc[-2]) * 0.2 and # 2: 작은 몸통 (도지 또는 작은 캔들)
        close3 > open3 and # 3: 양봉
        close3 > open1): # 3: 1번 캔들의 몸통 안으로 들어감 (강력한 반전 신호)
        return "샛별형 (Morning Star)"

    # 하락 반전 패턴: Evening Star (석별형)
    if (close1 > open1 and # 1: 양봉
        abs(open2 - close2) < (last_three['high'].iloc[-2] - last_three['low'].iloc[-2]) * 0.2 and # 2: 작은 몸통
        open3 > close3 and # 3: 음봉
        close3 < open1): # 3: 1번 캔들의 몸통 안으로 들어감
        return "석별형 (Evening Star)"

    # 상승 잉태형 (Harami Bullish)
    if (open1 > close1 and # 1: 큰 음봉
        close2 > open2 and # 2: 작은 양봉
        close2 < open1 and open2 > close1): # 2번 캔들이 1번 캔들 몸통 안에 포함
        return "상승 잉태형 (Harami Bullish)"

    # 하락 잉태형 (Harami Bearish)
    if (close1 > open1 and # 1: 큰 양봉
        open2 > close2 and # 2: 작은 음봉
        open2 < close1 and close2 > open1): # 2번 캔들이 1번 캔들 몸통 안에 포함
        return "하락 잉태형 (Harami Bearish)"

    # 역망치형 (Inverted Hammer)
    if (close3 > open3 and # 양봉
        (last_three['high'].iloc[-1] - close3) > 2 * (close3 - open3) and # 긴 윗 꼬리
        (open3 - last_three['low'].iloc[-1]) < (close3 - open3)): # 짧은 아래 꼬리
        return "역망치형 (Inverted Hammer)"

    # 슈팅스타 (Shooting Star)
    if (open3 > close3 and # 음봉
        (last_three['high'].iloc[-1] - open3) > 2 * (open3 - close3) and # 긴 윗 꼬리
        (close3 - last_three['low'].iloc[-1]) < (open3 - close3)): # 짧은 아래 꼬리
        return "슈팅스타 (Shooting Star)"

    # 삼백병 (Three White Soldiers)
    if (is_bullish1 and is_bullish2 and is_bullish3 and
        close2 > close1 and close3 > close2 and
        open2 > open1 and open3 > open2):
        return "삼백병 (Three White Soldiers)"

    # 삼봉우리 (Three Black Crows)
    if (not is_bullish1 and not is_bullish2 and not is_bullish3 and
        close2 < close1 and close3 < close2 and
        open2 < open1 and open3 < open2):
        return "삼봉우리 (Three Black Crows)"

    # 일반적인 상승/하락
    if close3 > close2 and close2 > close1:
        return "지속 상승"
    if close3 < close2 and close2 < close1:
        return "지속 하락"

    return "특별 패턴 없음"


def analyze_liquidity(volume):
    if volume > 1000:
        return "매우 높음"
    elif volume > 500:
        return "높음"
    elif volume > 100:
        return "보통"
    else:
        return "낮음"

def analyze_with_gpt(payload):
    messages = [
        {"role": "system", "content": "You are a professional forex trading assistant. Analyze the given trading data and provide a concise recommendation (BUY, SELL, or WAIT) and a brief analysis. All responses MUST be in Korean. Provide the decision first in Korean, then the analysis."},
        {"role": "user", "content": f"""
        Analyze the following forex trading data and provide a concise recommendation (BUY, SELL, or WAIT) and a brief analysis.
        All responses MUST be in Korean.
        
        [TRADING DATA]
        Pair: {payload['pair']}
        Current Price: {payload['price']}
        TradingView Signal: {payload['signal']}
        RSI: {payload['rsi']:.2f}
        MACD: {payload['macd']:.2f}
        MACD Signal: {payload['macd_signal']:.2f}
        Stoch RSI: {payload['stoch_rsi']:.2f}
        Bollinger Upper: {payload['bollinger_upper']:.5f}
        Bollinger Lower: {payload['bollinger_lower']:.5f}
        Pattern: {payload['pattern']}
        Trend: {payload['trend']}
        Liquidity: {payload['liquidity']}
        Support: {payload['support']:.5f}
        Resistance: {payload['resistance']:.5f}
        News Impact: {payload['news']}
        New High in Window: {payload['new_high']}
        New Low in Window: {payload['new_low']}
        ATR: {payload['atr']:.5f}

        [RESPONSE FORMAT]
        You MUST start your response with "[DECISION]: [YOUR_DECISION_IN_KOREAN]" followed by "Analysis: [YOUR_ANALYSIS_IN_KOREAN]".
        Example: [DECISION]: 매수
        Analysis: 시장이 강한 상승 모멘텀을 보이고 있습니다...
        """}
    ]
    try:
        response = openai.chat.completions.create(
            model="gpt-4o", # gpt-4o 또는 gpt-3.5-turbo
            messages=messages,
            max_tokens=500
        )
        content = response.choices[0].message.content
        return content
    except Exception as e:
        print(f"❌ GPT 분석 요청 실패: {e}")
        return "[DECISION]: WAIT\nAnalysis: GPT 분석 요청 실패로 인해 대기합니다."

def parse_gpt_feedback(feedback):
    decision_line = ""
    analysis_line = ""
    
    lines = feedback.split('\n')
    for line in lines:
        if line.startswith("[DECISION]:"):
            decision_line = line.replace("[DECISION]:", "").strip()
        elif line.startswith("Analysis:"):
            analysis_line = line.replace("Analysis:", "").strip()
    
    # decision_line이 비어있다면, 전체를 analysis로 간주하고 WAIT 처리
    if not decision_line:
        decision_line = "WAIT"
        analysis_line = "GPT 응답 파싱 실패 또는 예상치 못한 형식: " + feedback[:100] + "..."

    return decision_line, analysis_line

def calculate_atr(candles, window=14):
    if candles.empty or len(candles) < window:
        return None
    
    high_low = candles['high'] - candles['low']
    high_close = np.abs(candles['high'] - candles['close'].shift())
    low_close = np.abs(candles['low'] - candles['close'].shift())
    
    tr = pd.DataFrame({'hl': high_low, 'hc': high_close, 'lc': low_close}).max(axis=1)
    atr = tr.ewm(span=window, adjust=False).mean()
    return atr.iloc[-1]

# --- log_trade_result 함수 (수정됨) ---
def log_trade_result(
    trade_time, pair, tradingview_signal, decision, signal_score,
    reasons, result, rsi, macd, stoch_rsi,
    pattern, trend, fibo_levels, final_decision_score, news, gpt_feedback,
    alert_name, tp, sl, price, pnl, notes, # notes 파라미터 추가
    outcome_analysis, adjustment_suggestion, price_movements,
    atr
):
    if sheet is None:
        print("❌ Google 스프레드시트에 연결되지 않아 기록할 수 없습니다.")
        return

    try:
        trade_time_str = trade_time.strftime("%Y-%m-%d %H:%M:%S")

        # pnl (손익) 값 유효성 검사 및 float 변환
        if pnl is not None:
            try:
                pnl = float(pnl)
            except ValueError:
                pnl = None # float 변환 실패 시 None

        # price_movements를 문자열로 변환 (dict/list가 직접 들어가지 않도록)
        filtered_movement_str = json.dumps(price_movements, ensure_ascii=False) if price_movements else ""

        # 안전하게 float 값 처리
        def safe_float(value):
            try:
                if isinstance(value, (int, float)):
                    return value
                return float(value) if value is not None else ""
            except (ValueError, TypeError):
                return ""

        row = [
            trade_time_str, pair, tradingview_signal, decision, signal_score,
            f"RSI: {round(rsi, 2) if not math.isnan(rsi) else 'N/A'}",
            f"MACD: {round(macd, 5) if not math.isnan(macd) else 'N/A'}",
            f"StochRSI: {round(stoch_rsi, 5) if not math.isnan(stoch_rsi) else 'N/A'}",
            # 볼린저밴드 추가 (함수 인자에 추가되지 않아 일단 임시로 N/A)
            "N/A", # Boll_Upper
            "N/A", # Boll_Lower
            pattern, trend, 
            "N/A", # Liquidity (함수 인자에 없는 필드)
            safe_float(fibo_levels.get('support', '')), safe_float(fibo_levels.get('resistance', '')), news,
            
            # --- 이 부분을 아래 한 줄로 변경합니다 ---
            json.dumps(result, ensure_ascii=False) if isinstance(result, dict) else str(result or ""), # result가 dict면 JSON으로, 아니면 문자열로 변환 (None이면 빈 문자열)
            # --- 여기까지 변경 ---
            
            gpt_feedback or "", # GPT 분석 결과를 여기에 한 번만 저장
            alert_name, 
            safe_float(tp), safe_float(sl), safe_float(price), safe_float(pnl),
            notes or "", # Notes 필드
            outcome_analysis or "",
            adjustment_suggestion or "",
            filtered_movement_str, # 필터링된 price_movements
            safe_float(atr)
        ]

        # 모든 요소를 스프레드시트가 인식할 수 있는 문자열 또는 숫자 형식으로 변환
        clean_row = []
        for v in row:
            if isinstance(v, (dict, list)):
                clean_row.append(json.dumps(v, ensure_ascii=False))
            elif isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                clean_row.append("")
            elif v is None:
                clean_row.append("")
            else:
                clean_row.append(str(v)) # 모든 값을 최종적으로 문자열로 강제 변환

        print("✅ STEP 8: 시트 저장 직전", clean_row)
        for idx, val in enumerate(clean_row):
            if isinstance(val, (dict, list)):
                print(f"❌ [오류] clean_row[{idx}]에 dict 또는 list가 남아 있음 → {val}")
        
        sheet.append_row(clean_row)
        print("✅ STEP 9: 스프레드시트에 결과 기록 성공!")

    except Exception as e:
        print(f"❌ 스프레드시트 기록 중 오류 발생: {e}")


# --- 웹훅 함수 (@app.post("/webhook")) (일부 수정됨) ---
@app.post("/webhook")
async def webhook(request: Request):
    print("✅ STEP 1: 웹훅 진입")
    data = json.loads(await request.body())
    pair = data.get("pair")

    # TradingView에서 EURUSD로 보낸다면 OANDA 형식으로 변경
    if pair is not None:
        if "_" not in pair and len(pair) == 6: # EURUSD처럼 6글자인데 _가 없으면
            pair = pair[:3] + "_" + pair[3:]
            print(f"DEBUG: pair 값 OANDA 형식으로 변환됨: {pair}")
    
    print(f"✅ STEP 2: 데이터 수신 완료 | pair: {pair}")

    price_raw = data.get("price")
    try:
        price = float(price_raw)
        print(f"✅ STEP 3: 가격 파싱 완료 | price: {price}")
    except (ValueError, TypeError):
        print(f"❌ 가격 파싱 실패: {price_raw}. 유효한 가격이 아닙니다.")
        return JSONResponse(content={"status": "error", "message": "유효하지 않은 가격"})

    alert_name = data.get("alert_name", "N/A")
    signal = data.get("signal")
    
    trade_time = datetime.utcnow()

    print("✅ STEP 4: 캔들 데이터 수신")
    candles = get_candles(pair, "M30", 250) # 30분봉 250개
    
    if candles is None or candles.empty:
        print("❌ 캔들 데이터를 불러올 수 없습니다. 요청 중단.")
        # 캔들 데이터가 없어도 스프레드시트에 기록하도록 변경 (정보 부족 알림)
        log_trade_result(
            trade_time, pair, signal, "WAIT", 0,
            ["캔들 데이터 없음"], {}, float('nan'), float('nan'), float('nan'),
            "N/A", "N/A", {}, "WAIT", "캔들 데이터 없음",
            "[DECISION]: WAIT\nAnalysis: 캔들 데이터를 불러올 수 없어 분석을 진행할 수 없습니다.",
            alert_name, float('nan'), float('nan'), price, None, "캔들 데이터 부족",
            "캔들 데이터 없음", "캔들 데이터 부족으로 인한 오류", [], float('nan')
        )
        return JSONResponse(content={"status": "error", "message": "캔들 데이터를 불러올 수 없습니다."})

    # 지표 계산
    rsi = calculate_rsi(candles)
    macd, macd_signal = calculate_macd(candles)
    stoch_k, stoch_d = calculate_stoch_rsi(rsi)
    
    # 최신 값 추출
    latest_rsi = rsi.iloc[-1] if not rsi.empty and not rsi.isnull().all() else float('nan')
    latest_macd = macd.iloc[-1] if not macd.empty and not macd.isnull().all() else float('nan')
    latest_macd_signal = macd_signal.iloc[-1] if not macd_signal.empty and not macd_signal.isnull().all() else float('nan')
    latest_stoch_k = stoch_k.iloc[-1] if not stoch_k.empty and not stoch_k.isnull().all() else float('nan')
    
    # 볼린저 밴드 계산 및 최신 값 추출
    _, bollinger_upper, bollinger_lower = calculate_bollinger_bands(candles)
    latest_bollinger_upper = bollinger_upper.iloc[-1] if not bollinger_upper.empty and not bollinger_upper.isnull().all() else float('nan')
    latest_bollinger_lower = bollinger_lower.iloc[-1] if not bollinger_lower.empty and not bollinger_lower.isnull().all() else float('nan')

    pattern = analyze_pattern(candles)
    trend = analyze_trend(candles)
    
    # ATR 계산
    atr = calculate_atr(candles)
    atr_val = atr if atr is not None and not math.isnan(atr) else float('nan')


    # 피보나치 레벨 (예시, 실제 계산 로직은 추가 필요)
    fibo_levels = {"support": 1.14054, "resistance": 1.1414} # 임시 값, 실제 계산 로직 필요

    # 뉴스 확인
    news_impact = fetch_forex_news() # 임시 비활성화 상태

    # 신호 점수화
    signal_score = 0
    reasons = []

    # 지표 기반 판단 로직 (GPT 호출을 위한 데이터 준비)
    payload = {
        "pair": pair,
        "price": price,
        "signal": signal,
        "rsi": latest_rsi,
        "macd": latest_macd,
        "macd_signal": latest_macd_signal,
        "stoch_rsi": latest_stoch_k, # Stoch RSI의 K-line 사용
        "bollinger_upper": latest_bollinger_upper,
        "bollinger_lower": latest_bollinger_lower,
        "pattern": pattern,
        "trend": trend,
        "liquidity": "Good", # 임시 값
        "support": fibo_levels.get("support", float('nan')),
        "resistance": fibo_levels.get("resistance", float('nan')),
        "news": news_impact,
        "new_high": analyze_highs_lows(candles)["new_high"],
        "new_low": analyze_highs_lows(candles)["new_low"],
        "atr": atr_val
    }
    
    print("✅ STEP 5: GPT 분석 요청")
    gpt_feedback = analyze_with_gpt(payload)
    decision, analysis = parse_gpt_feedback(gpt_feedback)
    print(f"✅ STEP 6: GPT 판단: {decision}")
    print(f"✅ STEP 7: GPT 분석: {analysis}")

    # --- TP/SL 계산 (수정됨) ---
    tp = float('nan')
    sl = float('nan')
    pnl = None
    outcome_analysis = ""
    adjustment_suggestion = ""
    result = {} # result 딕셔너리 초기화
    
    # 고정 TP/SL (예: 15pip TP / 10pip SL)
    TP_PIPS = 15
    SL_PIPS = 10
    
    if decision == "BUY":
        tp = price + (TP_PIPS * 0.0001)
        sl = price - (SL_PIPS * 0.0001)
    elif decision == "SELL":
        tp = price - (TP_PIPS * 0.0001)
        sl = price + (SL_PIPS * 0.0001)
    else: # WAIT 이거나 알 수 없는 결정
        tp = float('nan')
        sl = float('nan')

    print(f"⚠️ TP/SL은 GPT 무시, 고정값 적용 ({TP_PIPS}pip / {SL_PIPS}pip)")
    print(f"계산된 TP: {tp}, SL: {sl}")

    # 실제 주문 실행 (이 부분은 OANDA API 연동 필요)
    # 현재는 주문 실행 로직이 없으므로, 항상 WAIT 또는 주문 미실행으로 기록
    order_placed = False # 실제 주문 API 호출 결과에 따라 변경될 변수

    if decision in ["BUY", "SELL"]:
        # 여기에 실제 OANDA 주문 실행 로직이 들어갑니다.
        # 예:
        # order_response = place_oanda_order(pair, decision, units=1000, tp=tp, sl=sl)
        # if order_response.get("order_id"):
        #     order_placed = True
        #     result = {"status": "order_placed", "order_id": order_response.get("order_id")}
        # else:
        #     result = {"status": "order_failed", "message": order_response.get("errorMessage")}

        # 현재는 주문 실행 로직이 없으므로 항상 미실행으로 가정
        result = {"status": "주문 미실행", "message": "주문 실행 로직 미구현"}
        outcome_analysis = "주문 미실행 (코드 미구현)"
        print("🚫 주문 실행 로직이 구현되지 않았습니다. 주문 미실행.")
    else: # decision == "WAIT"
        outcome_analysis = "대기 또는 주문 미실행" # <-- 한국어로 변경

    # --- log_trade_result 호출 (수정됨) ---
    # log_trade_result 함수 호출 전 필요한 변수 준비
    price_movements = [] # 기본값 설정
    candles_post_execution = get_candles(pair, "M30", 8) # 주문 실행 후 캔들 데이터 재확인 (8개)
    if candles_post_execution is not None and not candles_post_execution.empty:
        price_movements = candles_post_execution[["high", "low"]].to_dict("records")
    
    # price_movements_str 변수를 추가하여 문자열로 변환합니다.
    price_movements_str = json.dumps(price_movements, ensure_ascii=False) if price_movements else ""

    print(f"✅ STEP 10: 전략 요약 저장 호출 | decision: {decision}, TP: {tp}, SL: {sl}")
    log_trade_result(
        trade_time, pair, signal, decision, signal_score,
        "\n".join(reasons) + f"\nATR: {round(atr_val, 5) if not math.isnan(atr_val) else 'N/A'}", # ATR 값 사용
        result, # result는 dict이므로 log_trade_result에서 처리
        latest_rsi, latest_macd, latest_stoch_k, # 최신 지표 값 사용
        pattern, trend, fibo_levels, decision, news_impact, gpt_feedback, # news_impact 사용
        alert_name, tp, sl, price, pnl, "알림 완료", # notes 필드 추가
        outcome_analysis, adjustment_suggestion, price_movements_str, # price_movements_str 사용
        atr_val # atr_val 사용
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
    # OANDA_API_KEY와 ACCOUNT_ID는 환경 변수에서 가져오므로, 파일 상단에 import os 가 있는지 확인해주세요.
    # 만약 없다면, 파일 맨 위쪽에 'import os' 줄을 추가해주세요.
    headers = {"Authorization": f"Bearer {os.getenv('OANDA_API_KEY')}"}
    params = {"granularity": granularity, "count": count, "price": "M"}
    
    print(f"DEBUG: OANDA API 요청 URL: {url}")
    print(f"DEBUG: OANDA API 요청 헤더: {{'Authorization': 'Bearer <숨김>', 'Content-Type': 'application/json'}}")
    print(f"DEBUG: OANDA API 요청 파라미터: {params}")
    
    try:
        r = requests.get(url, headers=headers, params=params)
        r.raise_for_status() # HTTP 오류가 발생하면 예외를 발생시킵니다.
        
        print(f"DEBUG: OANDA API 응답 상태 코드: {r.status_code}")
        print(f"DEBUG: OANDA API 응답 본문: {r.text[:500]}...") # 응답 내용 일부만 출력
        
        candles_data = r.json().get("candles", [])
        if not candles_data:
            print(f"WARNING: OANDA API에서 {pair} 캔들 데이터가 없습니다. 빈 DataFrame 반환.")
            return pd.DataFrame() # 빈 DataFrame 반환
        
        df = pd.DataFrame(candles_data)
        df["time"] = pd.to_datetime(df["time"])
        df["open"] = df["mid"].apply(lambda x: float(x["o"]))
        df["high"] = df["mid"].apply(lambda x: float(x["h"]))
        df["low"] = df["mid"].apply(lambda x: float(x["l"]))
        df["close"] = df["mid"].apply(lambda x: float(x["c"]))
        df["volume"] = df["volume"].astype(int)
        df = df[["time", "open", "high", "low", "close", "volume"]]
        return df
    except requests.exceptions.RequestException as e:
        print(f"ERROR: OANDA API 요청 중 오류 발생: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"ERROR: OANDA API 응답: {e.response.text}")
        return None # 오류 발생 시 None 반환
    except json.JSONDecodeError as e:
        print(f"ERROR: OANDA API 응답 JSON 디코딩 실패: {e} | 응답 텍스트: {r.text[:500]}...")
        return None
    except Exception as e:
        print(f"ERROR: get_candles 함수에서 알 수 없는 오류 발생: {e}")
        return None

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
        If you decide BUY or SELL, suggest a Take Profit (TP) and Stop Loss (SL) level in pips. 너의 모든 분석은 한글로 답변해.
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
