import requests
import pandas as pd
from datetime import datetime, timedelta
import time

# 본인의 OANDA API 정보 입력
OANDA_API_KEY = "여기에_본인_API_KEY_입력"
ACCOUNT_ID = "여기에_본인_ACCOUNT_ID_입력"

# 설정값
PAIR = "EUR_USD"
GRANULARITY = "M30"
DAYS_BACK = 90  # 최근 90일치 다운로드

# OANDA API endpoint
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
        "price": "M",
        "count": 5000  # 최대치로 한 번에 요청
    }
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    return response.json().get("candles", [])

# 전체 데이터 수집 루프
all_candles = []
end_time = datetime.utcnow()
start_time = end_time - timedelta(days=DAYS_BACK)

# OANDA는 한 번에 너무 많은 데이터를 못 받으므로 7일 단위로 쪼개서 요청
chunk = timedelta(days=7)

while start_time < end_time:
    chunk_end = min(start_time + chunk, end_time)
    print(f"Fetching: {start_time} ~ {chunk_end}")
    candles = fetch_candles(PAIR, GRANULARITY, start_time, chunk_end)
    all_candles.extend(candles)
    start_time += chunk
    time.sleep(0.5)  # 잠깐 쉬어주기 (API 제한 방지)

# 데이터 변환
records = []
for c in all_candles:
    record = {
        "time": c["time"],
        "open": float(c["mid"]["o"]),
        "high": float(c["mid"]["h"]),
        "low": float(c["mid"]["l"]),
        "close": float(c["mid"]["c"]),
        "volume": c.get("volume", 0)
    }
    records.append(record)

df = pd.DataFrame(records)
df.to_csv("EURUSD_M30_history.csv", index=False)

print("✅ 데이터 다운로드 완료: EURUSD_M30_history.csv")

