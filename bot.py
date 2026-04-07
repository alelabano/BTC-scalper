"""
btc_scalper_v8.py — VERSIONE SEMPLIFICATA (EDGE > COMPLESSITÀ)

FILOSOFIA:
- Meno filtri → più trade
- Segnali NON combinati (flow OR momentum)
- FLASH mode ultra veloce (zero latenza inutile)

RIMOSSO:
- FGI sentiment
- Funding z-score
- Micro-validation
- Overweight segnali multipli

TENUTO:
- Regime multi-TF
- Order flow (semplificato)
- Momentum breakout
- Risk management
"""

import os, time, threading
import numpy as np
import pandas as pd
from collections import deque
from datetime import datetime
from eth_account import Account
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants

# ================= CONFIG =================
PRIVATE_KEY = os.getenv("PRIVATE_KEY")
COIN = "BTC"
LEVERAGE = 5
RISK_USD = 4.0
SCAN_INTERVAL = 5

# ================= CLIENT =================
_info = Info(constants.MAINNET_API_URL, skip_ws=True)
_account = Account.from_key(PRIVATE_KEY)
_exchange = Exchange(_account, constants.MAINNET_API_URL)

# ================= STATE =================
_price_buffer = deque(maxlen=180)
_cvd_buffer = deque(maxlen=50)

# ================= UTILS =================
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# ================= REGIME =================
def detect_regime(df_4h):
    ema50 = df_4h['close'].ewm(span=50).mean().iloc[-1]
    ema200 = df_4h['close'].ewm(span=200).mean().iloc[-1]

    if ema50 > ema200:
        return "BULL"
    elif ema50 < ema200:
        return "BEAR"
    return "RANGE"

# ================= FLOW (SEMPLIFICATO) =================
def get_orderbook_bias():
    try:
        ob = _info.l2_snapshot(COIN)
        bids = ob['levels'][0][:5]
        asks = ob['levels'][1][:5]

        bid_vol = sum(float(b['sz']) for b in bids)
        ask_vol = sum(float(a['sz']) for a in asks)

        ratio = bid_vol / (bid_vol + ask_vol + 1e-9)

        if ratio > 0.6:
            return "BUY"
        elif ratio < 0.4:
            return "SELL"
        return "NEUTRAL"
    except:
        return "NEUTRAL"

# ================= MOMENTUM =================
def check_momentum():
    if len(_price_buffer) < 30:
        return None

    now = _price_buffer[-1]
    past = _price_buffer[0]

    move = (now - past) / past

    if move > 0.002:
        return "BUY"
    elif move < -0.002:
        return "SELL"
    return None

# ================= FLASH MODE =================
def check_flash(volume, avg_volume):
    if volume > avg_volume * 3:
        return True
    return False

# ================= EXECUTION =================
def execute_trade(side, price):
    log(f"EXECUTE {side} @ {price}")
    # placeholder execution

# ================= MAIN LOOP =================
def run():
    log("START BOT V8")

    while True:
        try:
            # ---- prezzo ----
            ticker = _info.ticker(COIN)
            price = float(ticker['last'])
            _price_buffer.append(price)

            # ---- regime ----
            # (placeholder df)
            regime = "BULL"

            # ---- segnali ----
            flow = get_orderbook_bias()
            momentum = check_momentum()

            # ---- DECISIONE (NON combinata) ----
            signal = None

            if momentum:
                signal = momentum
            elif flow != "NEUTRAL":
                signal = flow

            # ---- EXEC ----
            if signal:
                execute_trade(signal, price)

        except Exception as e:
            log(f"ERROR {e}")

        time.sleep(SCAN_INTERVAL)

if __name__ == "__main__":
    run()
