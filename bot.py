"""
btc_scalper_v8_fixed.py — VERSIONE CORRETTA E OTTIMIZZATA
FIX: Sostituito _info.ticker (inesistente) con _info.all_mids()
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
RISK_USD = 10.0  # Budget per trade
SCAN_INTERVAL = 2 # Più veloce per V8

# ================= CLIENT =================
# Nota: Assicurati che la chiave privata sia caricata correttamente
try:
    _account = Account.from_key(PRIVATE_KEY)
    _info = Info(constants.MAINNET_API_URL, skip_ws=True)
    _exchange = Exchange(_account, constants.MAINNET_API_URL)
except Exception as e:
    print(f"ERRORE CONFIGURAZIONE: {e}")
    exit()

# ================= STATE =================
_price_buffer = deque(maxlen=180)
_is_position_open = False

# ================= UTILS =================
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def get_current_price():
    """
    FIX: Metodo corretto per ottenere il prezzo su Hyperliquid SDK
    """
    try:
        all_mids = _info.all_mids()
        if COIN in all_mids:
            return float(all_mids[COIN])
    except Exception as e:
        log(f"Errore recupero prezzo: {e}")
    return None

# ================= ANALISI =================
def get_orderbook_bias():
    """Analisi semplificata dell'Order Book"""
    try:
        l2_data = _info.l2_snapshot(COIN)
        levels = l2_data['levels']
        
        # Somma volumi prime 5 entries
        bid_vol = sum(float(l['sz']) for l in levels[0][:5])
        ask_vol = sum(float(l['sz']) for l in levels[1][:5])
        
        if bid_vol > ask_vol * 1.5: return "BUY"
        if ask_vol > bid_vol * 1.5: return "SELL"
    except:
        pass
    return "NEUTRAL"

def check_momentum():
    if len(_price_buffer) < 30:
        return None
    
    # Calcolo variazione percentuale negli ultimi N cicli
    move = (_price_buffer[-1] - _price_buffer[0]) / _price_buffer[0]
    
    if move > 0.0015: return "BUY"  # +0.15%
    if move < -0.0015: return "SELL" # -0.15%
    return None

# ================= EXECUTION =================
def execute_trade(side, price):
    global _is_position_open
    if _is_position_open:
        return

    try:
        log(f"🚀 ESECUZIONE {side} @ {price}")
        
        # Calcolo size in base a RISK_USD e Leva
        sz = round(RISK_USD * LEVERAGE / price, 4)
        is_buy = (side == "BUY")
        
        # Ordine Market (Market-with-Protection su HL)
        order_result = _exchange.market_open(COIN, is_buy, sz, slippage=0.01)
        
        if order_result['status'] == 'ok':
            log(f"✅ Ordine eseguito: {order_result}")
            _is_position_open = True
            # Qui andrebbe aggiunto un thread per gestire la chiusura (TP/SL)
        else:
            log(f"❌ Errore ordine: {order_result}")
            
    except Exception as e:
        log(f"CRITICAL EXECUTION ERROR: {e}")

# ================= MAIN LOOP =================
def run():
    log("=== START BOT V8 (FIXED) ===")
    log(f"Monitoraggio su: {COIN} | Rischio: ${RISK_USD}")

    while True:
        try:
            # 1. Recupero Prezzo (Corretto)
            price = get_current_price()
            if not price:
                time.sleep(1)
                continue
                
            _price_buffer.append(price)

            # 2. Logica Segnali (Logica "OR" della V8)
            flow = get_orderbook_bias()
            momentum = check_momentum()
            
            signal = None
            if momentum:
                signal = momentum
                log(f"Segnale MOMENTUM rilevato: {signal}")
            elif flow != "NEUTRAL":
                signal = flow
                log(f"Segnale ORDERFLOW rilevato: {signal}")

            # 3. Esecuzione
            if signal and not _is_position_open:
                execute_trade(signal, price)
                
            # Log di stato ogni 10 cicli
            if len(_price_buffer) % 10 == 0:
                log(f"Price: {price} | Flow: {flow} | Open: {_is_position_open}")

        except Exception as e:
            log(f"ERROR LOOP: {e}")

        time.sleep(SCAN_INTERVAL)

if __name__ == "__main__":
    run()
