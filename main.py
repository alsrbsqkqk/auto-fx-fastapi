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

# --- 환경 변수 설정 ---
# OANDA API 키와 계정 ID
OANDA_API_KEY = os.getenv("OANDA_API_KEY")
ACCOUNT_ID = os.getenv("ACCOUNT_ID")

# OpenAI API 키
openai.api_key = os.getenv("OPENAI_API_KEY")

# Google Sheets API 설정
SERVICE_ACCOUNT_FILE_PATH = os.getenv("SERVICE_ACCOUNT_FILE_PATH")
scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive"
]

creds = None
sheet = None # sheet 초기화
if SERVICE_ACCOUNT_FILE_PATH and os.path.exists(SERVICE_ACCOUNT_FILE_PATH):
    try:
        creds = ServiceAccountCredentials.from_json_keyfile_name(SERVICE_ACCOUNT_FILE_PATH, scope)
        client = gspread.authorize(creds)
        
        SPREADSHEET_NAME = os.getenv("SPREADSHEET_NAME")
        if SPREADSHEET_NAME:
            sheet = client.open(SPREADSHEET_NAME).sheet1
            print(f"✅ Google 스프레드시트 '{SPREADSHEET_NAME}' 연결 성공!")
        else:
            print("❌ SPREADSHEET_NAME 환경 변수가 설정되지 않았습니다.")
    except Exception as e:
        print(f"❌ Google 스프레드시트 연결 실패: {e}")
else:
    print("❌ SERVICE_ACCOUNT_FILE_PATH 환경 변수가 설정되지 않았거나 파일이 존재하지 않습니다.")


# --- 캔들 데이터 가져오기 함수 ---
def get_candles(pair, granularity, count):
    url = f"https://api-fxpractice.oanda.com/v3/instruments/{pair}/candles"
    headers = {"Authorization": f"Bearer {OANDA_API_KEY}"}
    
    # price 파라미터를 "B" (Bid)로 변경하여 시도 (이전 에러 해결 목적)
    params = {"granularity": granularity, "count": count, "price": "B"}
    
    print(f"DEBUG: OANDA API 요청 URL: {url}")
    print(f"DEBUG: OANDA API 요청 헤더: {{'Authorization': 'Bearer <숨김>', 'Content-Type': 'application/json'}}")
    print(f"DEBUG: OANDA API 요청 파라미터: {params}")
    
    try:
        r = requests.get(url, headers=headers, params=params, timeout=15)
        r.raise_for_status() # HTTP 오류 발생 시 예외 발생
        
        print(f"DEBUG: OANDA API 응답 상태 코드: {r.status_code}")
        full_response_json = r.json()
        # API 응답 전체 JSON을 로그에 출력하여 디버깅 용이 (캔들 데이터 없음 원인 파악용)
        print(f"DEBUG: OANDA API 응답 전체 JSON: {json.dumps(full_response_json, indent=2, ensure_ascii=False)}")
        
        candles_data = full_response_json.get("candles", [])
        if not candles_data:
            print(f"WARNING: OANDA API에서 {pair} 캔들 데이터가 없습니다. 빈 DataFrame 반환.")
            return pd.DataFrame()
        
        df = pd.DataFrame(candles_data)
        df["time"] = pd.to_datetime(df["time"])
        
        # 'price' 파라미터를 "B"로 설정했으므로, bid 가격을 사용
        df["open"] = df["bid"].apply(lambda x: float(x["o"]))
        df["high"] = df["bid"].apply(lambda x: float(x["h"]))
        df["low"] = df["bid"].apply(lambda x: float(x["l"]))
        df["close"] = df["bid"].apply(lambda x: float(x["c"]))
        
        df["volume"] = df["volume"].astype(int)
        df = df[["time", "open", "high", "low", "close", "volume"]]
        return df
    except requests.exceptions.RequestException as e:
        print(f"ERROR: OANDA API 요청 중 오류 발생: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"ERROR: OANDA API 응답: {e.response.text}")
        return None
    except json.JSONDecodeError as e:
        print(f"ERROR: OANDA API 응답 JSON 디코딩 실패: {e} | 응답 텍스트: {r.text[:500]}...")
        return None
    except Exception as e:
        print(f"ERROR: get_candles 함수에서 알 수 없는 오류 발생: {e}")
        return None

# --- 뉴스 가져오기 함수 (임시 비활성화) ---
def fetch_forex_news():
    print("DEBUG: fetch_forex_news 함수 임시 비활성화됨.")
    return "뉴스 기능 임시 비활성화" # 스프레드시트에 기록될 메시지


# --- 지표 계산 함수들 ---
def calculate_rsi(candles, window=14):
    if candles.empty: return pd.Series([np.nan])
    delta = candles['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(candles, fast=12, slow=26, signal=9):
    if candles.empty: return pd.Series([np.nan]), pd.Series([np.nan])
    exp1 = candles['close'].ewm(span=fast, adjust=False).mean()
    exp2 = candles['close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_stoch_rsi(rsi, k_window=3, d_window=3, rsi_window=14):
    if rsi.empty: return pd.Series([np.nan]), pd.Series([np.nan])
    min_rsi = rsi.rolling(window=rsi_window).min()
    max_rsi = rsi.rolling(window=rsi_window).max()
    stoch_rsi = ((rsi - min_rsi) / (max_rsi - min_rsi)) * 100
    k_line = stoch_rsi.rolling(window=k_window).mean()
    d_line = k_line.rolling(window=d_window).mean()
    return k_line, d_line

def calculate_bollinger_bands(candles, window=20, num_std_dev=2):
    if candles.empty: return pd.Series([np.nan]), pd.Series([np.nan]), pd.Series([np.nan])
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
    if candles.empty or len(candles) < max(short_window, long_window):
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

# --- GPT 분석 요청 함수 ---
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
    
    # 결정 라인이 비어있다면, 전체를 analysis로 간주하고 WAIT 처리 (안전장치)
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

# --- 스프레드시트 기록 함수 (최종 수정 버전) ---
def log_trade_result(
    trade_time, pair, tradingview_signal, decision, signal_score,
    result, # 'reasons' 인자 제거됨
    rsi, macd, stoch_rsi,
    bollinger_upper, bollinger_lower, # Boll Bands 인자
    pattern, trend, liquidity, support, resistance, news,
    gpt_feedback, # gpt_feedback 위치 조정
    alert_name, tp, sl, price, pnl, notes, # notes 인자
    outcome_analysis, adjustment_suggestion, price_movements_str, # price_movements_str 인자
    atr
):
    if sheet is None:
        print("❌ Google 스프레드시트에 연결되지 않아 기록할 수 없습니다.")
        return

    try:
        trade_time_str = trade_time.strftime("%Y-%m-%d %H:%M:%S")

        if pnl is not None:
            try:
                pnl = float(pnl)
            except ValueError:
                pnl = None

        # 안전하게 float 값 처리 (NaN, None, 비숫자 문자열 등 처리)
        def safe_float(value):
            try:
                if isinstance(value, (int, float)):
                    return value
                if isinstance(value, str) and value.replace('.', '', 1).isdigit(): # 소수점 포함 숫자 문자열
                    return float(value)
                return float('nan') # 변환 불가능한 경우 NaN 반환
            except (ValueError, TypeError):
                return float('nan') # 에러 발생 시 NaN 반환

        # 지표 값을 문자열로 포맷팅하는 헬퍼 함수
        def format_indicator(value, decimal_places=2):
            if isinstance(value, (float, int)) and not math.isnan(value) and not math.isinf(value):
                return str(round(value, decimal_places))
            return "N/A" # NaN, Inf, None 등 유효하지 않은 값 처리

        # 스프레드시트에 기록될 행 데이터 구성 (순서 매우 중요)
        row = [
            trade_time_str,                                    # 0: 거래 시간
            pair,                                              # 1: 통화쌍
            tradingview_signal,                                # 2: TradingView 신호
            decision,                                          # 3: 최종 결정 (BUY/SELL/WAIT)
            signal_score,                                      # 4: 신호 점수
            f"RSI: {format_indicator(rsi, 2)}",                # 5: RSI
            f"MACD: {format_indicator(macd, 5)}",              # 6: MACD
            f"StochRSI: {format_indicator(stoch_rsi, 5)}",     # 7: StochRSI
            f"BOLL U: {format_indicator(bollinger_upper, 5)}", # 8: Boll_Upper
            f"BOLL L: {format_indicator(bollinger_lower, 5)}", # 9: Boll_Lower
            pattern,                                           # 10: 패턴
            trend,                                             # 11: 트렌드
            liquidity,                                         # 12: 유동성
            format_indicator(safe_float(support), 5),          # 13: 지지선
            format_indicator(safe_float(resistance), 5),       # 14: 저항선
            news,                                              # 15: 뉴스 영향
            
            json.dumps(result, ensure_ascii=False) if isinstance(result, dict) else str(result or ""), # 16: 주문 결과 (JSON 문자열 또는 문자열)
            
            gpt_feedback or "",                                # 17: GPT 분석 결과 (GPT가 제공한 원본 피드백)
            alert_name,                                        # 18: 알림 이름
            format_indicator(safe_float(tp), 5),               # 19: TP
            format_indicator(safe_float(sl), 5),               # 20: SL
            format_indicator(safe_float(price), 5),            # 21: 현재 가격
            format_indicator(safe_float(pnl), 2),              # 22: 실현 손익 (PNL)
            notes or "",                                       # 23: 비고 (Notes)
            outcome_analysis or "",                            # 24: 결과 분석 (내부 로직)
            adjustment_suggestion or "",                       # 25: 조정 제안
            price_movements_str,                               # 26: 가격 변동 (JSON 문자열)
            format_indicator(safe_float(atr), 5)               # 27: ATR
        ]

        # 모든 요소를 스프레드시트가 인식할 수 있는 문자열 또는 숫자 형식으로 변환 (강제 변환)
        clean_row = []
        for v in row:
            if isinstance(v, (dict, list)):
                clean_row.append(json.dumps(v, ensure_ascii=False))
            elif isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                clean_row.append("") # NaN 또는 Inf는 빈 문자열로 처리
            elif v is None:
                clean_row.append("") # None은 빈 문자열로 처리
            else:
                clean_row.append(str(v)) # 그 외 모든 값은 문자열로 강제 변환

        print("✅ STEP 8: 시트 저장 직전", clean_row)
        for idx, val in enumerate(clean_row):
            if isinstance(val, (dict, list)):
                print(f"❌ [오류] clean_row[{idx}]에 dict 또는 list가 남아 있음 → {val}")
        print(f"🧪 최종 clean_row 길이: {len(clean_row)}")

        sheet.append_row(clean_row)
        print("✅ STEP 9: 스프레드시트에 결과 기록 성공!")

    except Exception as e:
        print(f"❌ 스프레드시트 기록 중 오류 발생: {e}")


# --- 웹훅 수신 및 처리 함수 (@app.post("/webhook")) ---
@app.post("/webhook")
async def webhook(request: Request):
    print("✅ STEP 1: 웹훅 진입")
    data = json.loads(await request.body())
    pair = data.get("pair")

    # TradingView에서 EURUSD로 보낸다면 OANDA 형식 EUR_USD로 변경
    if pair is not None:
        if "_" not in pair and len(pair) == 6:
            pair = pair[:3] + "_" + pair[3:]
            print(f"DEBUG: pair 값 OANDA 형식으로 변환됨: {pair}")
    
    print(f"✅ STEP 2: 데이터 수신 완료 | pair: {pair}")

    price_raw = data.get("price")
    try:
        price = float(price_raw)
        print(f"✅ STEP 3: 가격 파싱 완료 | price: {price}")
    except (ValueError, TypeError):
        print(f"❌ 가격 파싱 실패: {price_raw}. 유효한 가격이 아닙니다.")
        # 가격 파싱 실패 시에도 스프레드시트에 기록하고 종료
        trade_time = datetime.utcnow()
        # 모든 지표 관련 인자를 NaN 또는 기본값으로 넘깁니다.
        log_trade_result(
            trade_time, pair, data.get("signal", "N/A"), "WAIT", 0,
            {}, # result (빈 딕셔너리)
            float('nan'), float('nan'), float('nan'), # rsi, macd, stoch_rsi
            float('nan'), float('nan'), # bollinger_upper, bollinger_lower
            "N/A", "N/A", "N/A", float('nan'), float('nan'), "가격 파싱 실패", # pattern, trend, liquidity, support, resistance, news
            "[DECISION]: WAIT\nAnalysis: 가격 파싱 실패로 분석 불가.", # gpt_feedback
            data.get("alert_name", "N/A"), float('nan'), float('nan'), float('nan'), None, "가격 파싱 오류", # alert_name, tp, sl, price, pnl, notes
            "가격 파싱 오류", "가격 데이터 확인 필요", "", float('nan') # outcome_analysis, adjustment_suggestion, price_movements_str, atr
        )
        return JSONResponse(content={"status": "error", "message": "유효하지 않은 가격"})

    alert_name = data.get("alert_name", "N/A")
    signal = data.get("signal")
    
    trade_time = datetime.utcnow()

    print("✅ STEP 4: 캔들 데이터 수신")
    candles = get_candles(pair, "M30", 250) # 30분봉 250개

    # 지표 값 초기화 (캔들 데이터가 없거나 비어있을 경우 대비)
    latest_rsi = float('nan')
    latest_macd = float('nan')
    latest_macd_signal = float('nan')
    latest_stoch_k = float('nan')
    latest_bollinger_upper = float('nan')
    latest_bollinger_lower = float('nan')
    pattern = "N/A"
    trend = "N/A"
    liquidity = "N/A" # 기본 유동성
    atr_val = float('nan')
    fibo_levels = {"support": float('nan'), "resistance": float('nan')} # 기본 피보나치 레벨 (계산 로직 필요)
    news_impact = fetch_forex_news() # 뉴스 함수 호출 (현재 임시 비활성화)
    
    # 캔들 데이터가 유효할 경우에만 지표 계산
    if candles is not None and not candles.empty:
        # 지표 계산
        rsi_series = calculate_rsi(candles)
        macd_series, macd_signal_series = calculate_macd(candles)
        stoch_k_series, _ = calculate_stoch_rsi(rsi_series)
        _, bollinger_upper_series, bollinger_lower_series = calculate_bollinger_bands(candles)
        
        # 최신 값 추출 (NaN/None 체크)
        latest_rsi = rsi_series.iloc[-1] if not rsi_series.empty and not rsi_series.isnull().all() else float('nan')
        latest_macd = macd_series.iloc[-1] if not macd_series.empty and not macd_series.isnull().all() else float('nan')
        latest_macd_signal = macd_signal_series.iloc[-1] if not macd_signal_series.empty and not macd_signal_series.isnull().all() else float('nan')
        latest_stoch_k = stoch_k_series.iloc[-1] if not stoch_k_series.empty and not stoch_k_series.isnull().all() else float('nan')
        latest_bollinger_upper = bollinger_upper_series.iloc[-1] if not bollinger_upper_series.empty and not bollinger_upper_series.isnull().all() else float('nan')
        latest_bollinger_lower = bollinger_lower_series.iloc[-1] if not bollinger_lower_series.empty and not bollinger_lower_series.isnull().all() else float('nan')

        pattern = analyze_pattern(candles)
        trend = analyze_trend(candles)
        liquidity = analyze_liquidity(candles['volume'].iloc[-1]) if not candles.empty and 'volume' in candles.columns else "N/A"
        atr_val = calculate_atr(candles)
        atr_val = atr_val if atr_val is not None and not math.isnan(atr_val) else float('nan')
        
        # 피보나치 레벨 (예시, 실제 계산 로직은 여기에 추가해야 함)
        fibo_levels = {"support": 1.14054, "resistance": 1.1414}
    else:
        print("❌ 캔들 데이터를 불러올 수 없거나 비어 있습니다. 지표 계산 및 분석을 건너뛰고 WAIT 처리합니다.")
        # 캔들 데이터가 없어서 처리 중단 시에도 스프레드시트에 기록
        log_trade_result(
            trade_time, pair, signal, "WAIT", 0,
            {}, # result (빈 딕셔너리)
            latest_rsi, latest_macd, latest_stoch_k, # rsi, macd, stoch_rsi
            latest_bollinger_upper, latest_bollinger_lower, # NaN 값으로 전달
            pattern, trend, liquidity, fibo_levels.get("support", float('nan')), fibo_levels.get("resistance", float('nan')), news_impact,
            "[DECISION]: WAIT\nAnalysis: 캔들 데이터를 불러올 수 없어 분석을 진행할 수 없습니다.",
            alert_name, float('nan'), float('nan'), price, None, "캔들 데이터 부족",
            "캔들 데이터 없음", "캔들 데이터 부족으로 인한 오류", "", float('nan') # price_movements_str (빈 문자열), atr
        )
        return JSONResponse(content={"status": "error", "message": "캔들 데이터를 불러올 수 없습니다."})


    # GPT 페이로드 준비
    payload = {
        "pair": pair,
        "price": price,
        "signal": signal,
        "rsi": latest_rsi,
        "macd": latest_macd,
        "macd_signal": latest_macd_signal,
        "stoch_rsi": latest_stoch_k,
        "bollinger_upper": latest_bollinger_upper,
        "bollinger_lower": latest_bollinger_lower,
        "pattern": pattern,
        "trend": trend,
        "liquidity": liquidity,
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

    # TP/SL 및 주문 관련 초기화
    tp = float('nan')
    sl = float('nan')
    pnl = None
    outcome_analysis = ""
    adjustment_suggestion = ""
    result = {} # 주문 실행 결과 저장용 딕셔너리
    
    TP_PIPS = 15
    SL_PIPS = 10
    
    if decision == "BUY":
        tp = price + (TP_PIPS * 0.0001)
        sl = price - (SL_PIPS * 0.0001)
    elif decision == "SELL":
        tp = price - (TP_PIPS * 0.0001)
        sl = price + (SL_PIPS * 0.0001)
    else: # WAIT 또는 알 수 없는 결정
        tp = float('nan')
        sl = float('nan')

    print(f"⚠️ TP/SL은 GPT 무시, 고정값 적용 ({TP_PIPS}pip / {SL_PIPS}pip)")
    print(f"계산된 TP: {tp}, SL: {sl}")

    # 실제 주문 실행 로직 (현재는 목업)
    if decision in ["BUY", "SELL"]:
        # 여기에 실제 OANDA 주문 실행 API 호출 로직이 들어갑니다.
        # 예시:
        # order_response = place_oanda_order(pair, decision, units=1000, tp=tp, sl=sl)
        # if order_response.get("order_id"):
        #     order_placed = True
        #     result = {"status": "order_placed", "order_id": order_response.get("order_id")}
        # else:
        #     result = {"status": "order_failed", "message": order_response.get("errorMessage")}

        # 현재는 주문 실행 로직이 없으므로 항상 "주문 미실행"으로 가정
        result = {"status": "주문 미실행", "message": "주문 실행 로직 미구현"}
        outcome_analysis = "주문 미실행 (코드 미구현)"
        print("🚫 주문 실행 로직이 구현되지 않았습니다. 주문 미실행.")
    else: # decision == "WAIT"
        outcome_analysis = "대기 또는 주문 미실행"

    # log_trade_result 함수 호출 전 price_movements_str 준비
    price_movements = [] 
    # 주문 후 캔들 데이터 재확인 (8개), PNL 계산 등에 사용될 수 있으나 현재는 기록용
    candles_post_execution = get_candles(pair, "M30", 8) 
    if candles_post_execution is not None and not candles_post_execution.empty:
        price_movements = candles_post_execution[["high", "low"]].to_dict("records")
    
    price_movements_str = json.dumps(price_movements, ensure_ascii=False) if price_movements else ""

    print(f"✅ STEP 10: 전략 요약 저장 호출 | decision: {decision}, TP: {tp}, SL: {sl}")
    log_trade_result(
        trade_time,                       # trade_time
        pair,                             # pair
        signal,                           # tradingview_signal
        decision,                         # decision (from GPT)
        0,                                # signal_score (임시 0, TradingView 신호 점수 필요시 추가)
        result,                           # result (order execution info)
        latest_rsi,                       # rsi
        latest_macd,                      # macd
        latest_stoch_k,                   # stoch_rsi
        latest_bollinger_upper,           # bollinger_upper
        latest_bollinger_lower,           # bollinger_lower
        pattern,                          # pattern
        trend,                            # trend
        liquidity,                        # liquidity
        fibo_levels.get("support", float('nan')), # support
        fibo_levels.get("resistance", float('nan')), # resistance
        news_impact,                      # news
        gpt_feedback,                     # gpt_feedback
        alert_name,                       # alert_name
        tp,                               # tp
        sl,                               # sl
        price,                            # price (current)
        pnl,                              # pnl
        "알림 완료",                     # notes ("알림 완료" 기본값)
        outcome_analysis,                 # outcome_analysis
        adjustment_suggestion,            # adjustment_suggestion
        price_movements_str,              # price_movements_str
        atr_val                           # atr
    )
    
    return JSONResponse(content={"status": "completed", "decision": decision})


