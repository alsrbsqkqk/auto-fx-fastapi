from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

# OANDA 계정 정보 입력 (아래 두 줄 수정!)
OANDA_API_KEY = "058d089b726bf6ea58abef413f963bb4-1c70dafae38e324b65d6eafe8aafac7f"
OANDA_ACCOUNT_ID = "101-001-30264285-002"
OANDA_API_URL = f"https://api-fxpractice.oanda.com/v3/accounts/{OANDA_ACCOUNT_ID}/orders"

# 주문 전송 함수
def send_order_to_oanda(instrument, units, side, price, tp_offset, sl_offset):
    headers = {
        "Authorization": f"Bearer {OANDA_API_KEY}",
        "Content-Type": "application/json"
    }

    price = float(price)
    tp = price + tp_offset if side == "buy" else price - tp_offset
    sl = price - sl_offset if side == "buy" else price + sl_offset

    data = {
        "order": {
            "instrument": instrument,
            "units": str(units if side == "buy" else -units),
            "type": "MARKET",
            "positionFill": "DEFAULT",
            "takeProfitOnFill": {"price": f"{tp:.5f}"},
            "stopLossOnFill": {"price": f"{sl:.5f}"}
        }
    }

    response = requests.post(OANDA_API_URL, headers=headers, json=data)
    print("📤 OANDA 응답:", response.status_code, response.text)
    return response

# Webhook 처리 엔드포인트
@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.get_json()

    if data and 'signal' in data:
        signal = data['signal'].upper()
        pair = data.get('pair', 'EURUSD').replace('/', '_').upper()
        price = float(data.get('price', '0'))
        strategy = data.get('strategy', 'NO-STRATEGY')

        print(f"📩 [{strategy}] {pair} @ {price} → {signal}")

        # 전략별 기본 설정 (이후 전략별로 따로 설정 가능)
        units = 10000
        tp_offset = 0.0020  # 20 pip
        sl_offset = 0.0040  # 40 pip

        if signal in ['BUY', 'SELL']:
            send_order_to_oanda(pair, units, signal.lower(), price, tp_offset, sl_offset)

        return jsonify({'status': 'order_sent'}), 200
    else:
        return jsonify({'error': 'Invalid format'}), 400

# 서버 시작
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)