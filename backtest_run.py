import pandas as pd
import numpy as np

# ✅ 원본 데이터 불러오기 (조금 수정)
df = pd.read_csv("backtest_data.csv", parse_dates=["time"])
df = df.dropna().reset_index(drop=True)

# ✅ 보조지표 계산
def calculate_indicators(df):
    close = df["close"]

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window=14).mean()
    loss = -delta.clip(upper=0).rolling(window=14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()

    # Stoch RSI
    min_rsi = df["rsi"].rolling(window=14).min()
    max_rsi = df["rsi"].rolling(window=14).max()
    df["stoch_rsi"] = (df["rsi"] - min_rsi) / (max_rsi - min_rsi)

    return df

df = calculate_indicators(df)

# ✅ 심플한 백테스트 전략: 실전 로직 최대한 반영
results = []
position = None
entry_price = 0

for i in range(50, len(df)):
    row = df.iloc[i]

    signal_score = 0
    reasons = []

    # RSI 조건
    if row["rsi"] < 30:
        signal_score += 2
    if row["rsi"] > 70:
        signal_score += 2

    # MACD
    if (row["macd"] - row["macd_signal"]) > 0.001:
        signal_score += 2
    elif (row["macd_signal"] - row["macd"]) > 0.001:
        signal_score += 2

    # Stoch RSI
    if row["stoch_rsi"] > 0.8:
        signal_score += 1
    elif row["stoch_rsi"] < 0.2:
        signal_score += 1

    # 실제 진입 판단
    if position is None:
        if signal_score >= 5 and row["rsi"] < 30:
            position = "BUY"
            entry_price = row["close"]
            entry_time = row["time"]
        elif signal_score >= 5 and row["rsi"] > 70:
            position = "SELL"
            entry_price = row["close"]
            entry_time = row["time"]
    else:
        # 간단한 TP SL 로직 (예: 10pip)
        pip = 0.0001
        tp = entry_price + 10 * pip if position == "BUY" else entry_price - 10 * pip
        sl = entry_price - 7 * pip if position == "BUY" else entry_price + 7 * pip

        if position == "BUY":
            if row["high"] >= tp:
                results.append({"entry": entry_time, "exit": row["time"], "side": "BUY", "pl": tp - entry_price})
                position = None
            elif row["low"] <= sl:
                results.append({"entry": entry_time, "exit": row["time"], "side": "BUY", "pl": sl - entry_price})
                position = None
        if position == "SELL":
            if row["low"] <= tp:
                results.append({"entry": entry_time, "exit": row["time"], "side": "SELL", "pl": entry_price - tp})
                position = None
            elif row["high"] >= sl:
                results.append({"entry": entry_time, "exit": row["time"], "side": "SELL", "pl": entry_price - sl})
                position = None

# ✅ 결과 출력
result_df = pd.DataFrame(results)
print(result_df)
print("총 거래 수:", len(result_df))
print("총 수익:", result_df["pl"].sum())

# 결과 CSV 저장
result_df.to_csv("backtest_result.csv", index=False)
