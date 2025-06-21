import requests
import pandas as pd
from datetime import datetime, timedelta
import time

# 본인의 OANDA API 정보 입력
OANDA_API_KEY = "a2d997ea9e6476005739c1e0d11ccc24-ca1acd820317ee8e47cfd263d64c0c7d"
ACCOUNT_ID = "101-001-30264285-002"

# 설정값
PAIR = "EUR_USD"  # EURUSD 백테스트용
GRANULARITY = "M30"  # 30분봉
DAYS_BACK = 90  # 최근 90일치 다운로드

BASE_URL = "https://api-fxpractice.oanda.com/v3"
headers = {
    "Authorization": f"Bearer {OANDA_API_KEY}"
}

def fetch_candles(pair, granularity, start, end):
    url = f"{BASE_URL}/instruments/{pair}/candles"
    params = {
        "granularity": granularity,
        "from": start.isoformat() + "Z",
        "to": end.isoformat() + "Z",
        "price": "M"
    }
    r = requests.get(url, headers=headers, params=params)
    candles = r.json().get("candles", [])
    data = []
    for c in candles:
        data.append({
            "time": c["time"],
            "open": float(c["mid"]["o"]),
            "high": float(c["mid"]["h"]),
            "low": float(c["mid"]["l"]),
            "close": float(c["mid"]["c"]),
            "volume": c.get("volume", 0)
        })
    return data

def download_full_history():
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=DAYS_BACK)
    all_candles = []

    # OANDA API는 너무 긴 기간 한번에 요청 못함 → 7일씩 쪼개서 반복
    while start_time < end_time:
        chunk_end = min(start_time + timedelta(days=7), end_time)
        print(f"다운로드: {start_time} ~ {chunk_end}")
        candles = fetch_candles(PAIR, GRANULARITY, start_time, chunk_end)
        all_candles.extend(candles)
        start_time = chunk_end
        time.sleep(1)  # API 제한 때문에 살짝 쉬기

    df = pd.DataFrame(all_candles)
    df.to_csv("backtest_data.csv", index=False)
    print("✅ 다운로드 완료. 파일: backtest_data.csv")

if __name__ == "__main__":
    download_full_history()
