"""
BTC SCALPER V9 — COMPLETO, FUNZIONANTE, LIVE READY (Hyperliquid)

CARATTERISTICHE:
- Price feed corretto (cached)
- Segnali: MOMENTUM OR FLOW
- 1 posizione alla volta
- Market execution reale
- SL / TP base
"""

import os, time
from collections import deque
from datetime import datetime
from eth_account import Account
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants

# ================= CONFIG =================
PRIVATE_KEY = os.getenv("PRIVATE_KEY")
COIN = "BTC"
SIZE = 0.001
SCAN_INTERVAL = 2
TP_PCT = 0.002
SL_PCT = 0.001

# ================= CLIENT =================
_info = Info(constants.MAINNET_API_URL, skip_ws=True)
_account = Account.from_key(PRIVATE_KEY)
_exchange = Exchange(_account, constants.MAINNET_API_URL)

# ================= STATE =================
_price_buffer = deque(maxlen=120)
_last_price_cache = {"ts": 0, "price": None}
_last_ob_cache = {"ts": 0, "bias": "NEUTRAL"}
_position = None  # {side, entry}

# ================= UTILS =================
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# ================= PRICE =================
def get_price():
    now = time.time()
    if now - _last_price_cache["ts"] < 2:
        return _last_price_cache["price"]

    try:
        ctx = _info.meta_and_asset_ctxs()
        for a, c in zip(ctx[0]["universe"], ctx[1]):
            if a["name"] == COIN:
                price = float(c["midPx"])
                _last_price_cache.update({"ts": now, "price": price})
                return price
    except Exception as e:
        log(f"PRICE ERROR {e}")

    return None

# ================= FLOW =================
def get_orderbook_bias():
    now = time.time()
    if now - _last_ob_cache["ts"] < 2:
        return _last_ob_cache["bias"]

    try:
        ob = _info.l2_snapshot(COIN)
        bids = ob['levels'][0][:5]
        asks = ob['levels'][1][:5]

        bid_vol = sum(float(b['sz']) for b in bids)
        ask_vol = sum(float(a['sz']) for a in asks)

        ratio = bid_vol / (bid_vol + ask_vol + 1e-9)

        if ratio > 0.6:
            bias = "BUY"
        elif ratio < 0.4:
            bias = "SELL"
        else:
            bias = "NEUTRAL"

        _last_ob_cache.update({"ts": now, "bias": bias})
        return bias

    except Exception as e:
        log(f"OB ERROR {e}")
        return "NEUTRAL"

# ================= MOMENTUM =================
def check_momentum():
    if len(_price_buffer) < 30:
        return None

    move = (_price_buffer[-1] - _price_buffer[0]) / _price_buffer[0]

    if move > 0.002:
        return "BUY"
    elif move < -0.002:
        return "SELL"

    return None

# ================= EXECUTION =================
def open_position(side, price):
    global _position
    try:
        is_buy = side == "BUY"
        _exchange.market_open(COIN, is_buy, SIZE)
        _position = {"side": side, "entry": price}
        log(f"OPEN {side} @ {price}")
    except Exception as e:
        log(f"OPEN ERROR {e}")


def close_position(price):
    global _position
    try:
        is_buy = _position["side"] == "SELL"
        _exchange.market_open(COIN, is_buy, SIZE)
        log(f"CLOSE @ {price}")
        _position = None
    except Exception as e:
        log(f"CLOSE ERROR {e}")

# ================= RISK =================
def check_exit(price):
    if not _position:
        return False

    entry = _position["entry"]
    side = _position["side"]

    if side == "BUY":
        pnl = (price - entry) / entry
    else:
        pnl = (entry - price) / entry

    if pnl >= TP_PCT or pnl <= -SL_PCT:
        return True

    return False

# ================= MAIN =================
def run():
    log("START BOT V9")

    while True:
        try:
            price = get_price()
            if not price:
                time.sleep(1)
                continue

            _price_buffer.append(price)

            # gestione posizione
            if _position:
                if check_exit(price):
                    close_position(price)
                time.sleep(SCAN_INTERVAL)
                continue

            momentum = check_momentum()
            flow = get_orderbook_bias()

            signal = None
            if momentum:
                signal = momentum
            elif flow != "NEUTRAL":
                signal = flow

            if signal:
                open_position(signal, price)

        except Exception as e:
            log(f"ERROR {e}")

        time.sleep(SCAN_INTERVAL)

if __name__ == "__main__":
    run()
