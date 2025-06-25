import requests
import pandas as pd
from datetime import datetime, timedelta
import time

OANDA_API_KEY = "bb1207dc608f5a09b8b3bcf64fb04d1a-c3191973284dded434e45b62c74474fe"
ACCOUNT_ID = "101-001-30264285-002"

BASE_URL = "https://api-fxpractice.oanda.com/v3"
headers = { "Authorization": f"Bearer {OANDA_API_KEY}" }

PAIRS = ["EUR_USD", "GBP_USD", "USD_JPY"]
GRANULARITY = "M30"
DAYS_BACK = 90

def fetch_candles(pair, start, end):
    url = f"{BASE_URL}/instruments/{pair}/candles"
    params = {
        "granularity": GRANULARITY,
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

def download(pair):
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=DAYS_BACK)
    all_candles = []
    while start_time < end_time:
        chunk_end = min(start_time + timedelta(days=7), end_time)
        print(f"{pair} 다운로드: {start_time} ~ {chunk_end}")
        candles = fetch_candles(pair, start_time, chunk_end)
        all_candles.extend(candles)
        start_time = chunk_end
        time.sleep(1)
    df = pd.DataFrame(all_candles)
    pair_file = pair.replace("_", "") + ".csv"
    df.to_csv(pair_file, index=False)
    print(f"✅ {pair_file} 저장 완료")

if __name__ == "__main__":
    for p in PAIRS:
        download(p)