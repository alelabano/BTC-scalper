"""
btc_scalper_v6.py — Bot BTC-only scalping su 5m.

DESIGN:
  Solo BTC. Niente scanner, niente ranking, niente AI gatekeeper.
  Segnali puramente meccanici su 3 TF (4h trend, 1h setup, 5m entry).
  Entry frequenti (8+/giorno), risk $5 per trade, SL/TP meccanici.

LOGICA:
  4h → regime (BULL/BEAR/RANGE) + trend direction
  1h → setup condition (momentum o reversal)
  5m → entry trigger + SL/TP

BULL:  solo LONG  — breakout + pullback continuation
BEAR:  solo SHORT — breakdown + rally fade
RANGE: entrambi   — reversal agli estremi
"""

import os, sys, time, json, threading, requests
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from datetime import datetime, timezone
from eth_account import Account
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants

# ================================================================
# CONFIG
# ================================================================
PRIVATE_KEY = os.getenv("PRIVATE_KEY")
TG_TOKEN    = os.getenv("TELEGRAM_TOKEN", "")
TG_CHAT_ID  = os.getenv("TELEGRAM_CHAT_ID", "")
REDIS_URL   = os.getenv("UPSTASH_REDIS_REST_URL", "").rstrip("/")
REDIS_TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN", "")

if not PRIVATE_KEY:
    print("❌ PRIVATE_KEY mancante"); sys.exit(1)

COIN = "BTC"
LEVERAGE = 20
RISK_USD = 5.0                 # $ rischiati per trade (SL hit = -$5)
MAX_POSITIONS = 1              # 1 posizione alla volta
COOLDOWN_SEC = 180             # 3 min tra trade

# Timing
SCAN_INTERVAL = 30             # check ogni 30s per entry veloci
REGIME_INTERVAL = 300          # ricalcola regime ogni 5 min

# SL/TP — Dynamic Scalping Modes
# Mode auto-detected ogni ciclo basato su volatilità, ADX, volume

# MODE 1: RANGE SCALPING (Mean Reversion)
# Quando: ADX < 20, prezzo tra BB, volatilità bassa
# Strategia: compra supporto, vendi resistenza, TP fisso stretto
RANGE_SL_ATR     = 1.5
RANGE_TP_PCT     = 0.004       # TP fisso 0.4%
RANGE_SL_MIN     = 0.002       # SL min 0.2%
RANGE_SL_MAX     = 0.006       # SL max 0.6%
RANGE_TRAILING   = False       # no trailing, TP fisso

# MODE 2: TREND SCALPING (Momentum Pullback)
# Quando: ADX > 25, regime direzionale, candele con volume
# Strategia: entra sui pullback, trailing lascia correre
TREND_SL_ATR     = 2.0
TREND_TP_RR      = 2.0         # R:R 1:2 (lascia correre)
TREND_SL_MIN     = 0.003       # SL min 0.3%
TREND_SL_MAX     = 0.012       # SL max 1.2%
TREND_TRAILING   = True        # trailing attivo
TREND_TRAIL_ATR  = 1.0
TREND_PARTIAL    = 0.6         # partial close al 60% TP

# MODE 3: FLASH SCALPING (Volatility Expansion)
# Quando: volume spike >3x, funding z-score estremo, ATR spike
# Strategia: breakout puro, TP fulmineo, ordini IoC market
FLASH_SL_ATR     = 1.0         # SL stretto (momentum forte)
FLASH_TP_PCT     = 0.003       # TP 0.3% — esci subito
FLASH_SL_MIN     = 0.0015      # SL min 0.15%
FLASH_SL_MAX     = 0.005       # SL max 0.5%
FLASH_TRAILING   = False       # no trailing, esci veloce
FLASH_USE_IOC    = True        # forza IoC, no GTC (velocità)

# Defaults (usati dal backtest e come fallback)
SL_ATR_MULT = TREND_SL_ATR     # default = TREND
TP_RR = TREND_TP_RR
SL_MIN_PCT = TREND_SL_MIN
SL_MAX_PCT = TREND_SL_MAX
TRAILING_ACTIVATE = 0.5
TRAILING_ATR = TREND_TRAIL_ATR
PARTIAL_CLOSE_PCT = TREND_PARTIAL

FUNDING_BLOCK_THRESH = 0.0003

# Order execution
SLIPPAGE = 0.001               # 0.1% slippage base per limit orders
GTC_TIMEOUT = 6                # secondi max attesa per GTC fill
DRIFT_MAX_FAVORABLE = 0.004    # +0.4% max drift nella direzione del trade
DRIFT_MAX_ADVERSE = 0.008      # -0.8% max drift contro → segnale invalidato
PENDING_ORDER_TTL = 120        # 2 min max per ordini pendenti

# Circuit breaker
MAX_DAILY_LOSS = 20.0          # stop dopo -$20 giornaliero
MAX_CONSEC_LOSS = 3            # pausa 30min dopo 3 loss consecutive
DAILY_REPORT_HOUR = 0          # report giornaliero a mezzanotte UTC

# ================================================================
# CLIENTS
# ================================================================
_info = Info(constants.MAINNET_API_URL, skip_ws=True)
_account = Account.from_key(PRIVATE_KEY)
_exchange = Exchange(_account, constants.MAINNET_API_URL)
_pool = ThreadPoolExecutor(max_workers=3)

# ================================================================
# UTILS
# ================================================================
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def tg(msg, silent=False):
    if not TG_TOKEN: return
    try:
        requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json={"chat_id": TG_CHAT_ID, "text": msg, "parse_mode": "HTML",
                  "disable_notification": silent}, timeout=5)
    except: pass

def call(fn, *a, timeout=15, **kw):
    f = _pool.submit(fn, *a)
    try: return f.result(timeout=timeout)
    except FuturesTimeout: raise TimeoutError(f"Timeout {fn.__name__}")

def round_to_precision(value, precision):
    """Arrotonda al numero esatto di decimali accettati dall'exchange.
    Usa format string come V4 — più preciso di round() per float."""
    if precision == 0:
        return int(round(float(value)))
    try:
        return float(f"{float(value):.{int(precision)}f}")
    except:
        return round(float(value), 2)

# Alias breve per uso interno
rpx = round_to_precision

def min_decimals_for_price(px):
    """
    Calcola i decimali minimi perché round(px) mantenga significatività.
    Es: px=0.048 con dec=2 → 0.05 (errore 4%) → serve dec=3.
    Per BTC ($73692) → dec=0 è sufficiente.
    """
    if px <= 0: return 6
    for d in range(0, 10):
        rounded = round(px, d)
        if rounded > 0 and abs(rounded - px) / px < 0.005:
            return d
    return 6

def effective_price_dec(px, sl, tp, base_px_dec):
    """
    Calcola i decimali effettivi per garantire che entry, SL e TP
    siano DISTINTI dopo arrotondamento. Se SL==entry dopo round → alza decimali.
    """
    dec = max(base_px_dec, min_decimals_for_price(px))
    e = rpx(px, dec)
    s = rpx(sl, dec)
    t = rpx(tp, dec)
    if s == e or t == e:
        dec = min(dec + 2, 8)
    return dec

# ================================================================
# FUNDING Z-SCORE & OI TRACKING (from V4)
# ================================================================
_funding_history = []   # list of BTC funding rates (last 42 readings)
_oi_cache = 0.0
_oi_prev = 0.0
FUNDING_HISTORY_LEN = 42  # ~14h di readings ogni 20min

def update_funding_oi():
    """Fetch BTC funding rate e OI, aggiorna storico per z-score."""
    global _funding_history, _oi_cache, _oi_prev
    try:
        ctx = call(_info.meta_and_asset_ctxs, timeout=15)
        for a, c in zip(ctx[0]["universe"], ctx[1]):
            if a["name"] == COIN:
                funding = float(c.get("funding", 0) or 0)
                _funding_history.append(funding)
                if len(_funding_history) > FUNDING_HISTORY_LEN:
                    _funding_history.pop(0)

                _oi_prev = _oi_cache
                _oi_cache = float(c.get("openInterest", 0) or 0)
                break
    except Exception as e:
        log(f"update_funding_oi: {e}")

def get_funding_z():
    """Z-score del funding BTC: quanto è estremo rispetto alla storia recente."""
    if len(_funding_history) < 3:
        return 0.0
    arr = np.array(_funding_history)
    std = arr.std()
    if std < 1e-10:
        return 0.0
    return float((_funding_history[-1] - arr.mean()) / std)

def get_oi_change():
    """Variazione % dell'OI BTC rispetto al reading precedente."""
    if _oi_prev == 0:
        return 0.0
    return (_oi_cache - _oi_prev) / (_oi_prev + 1e-10)

# ================================================================
# REDIS STATE
# ================================================================
def _rget(key):
    if not REDIS_URL: return None
    try:
        r = requests.get(f"{REDIS_URL}/get/{key}",
            headers={"Authorization": f"Bearer {REDIS_TOKEN}"}, timeout=5)
        v = r.json().get("result")
        return json.loads(v) if v else None
    except: return None

def _rset(key, val):
    if not REDIS_URL: return
    try:
        requests.post(f"{REDIS_URL}/set/{key}/{requests.utils.quote(json.dumps(val))}",
            headers={"Authorization": f"Bearer {REDIS_TOKEN}"}, timeout=5)
    except: pass

_trades_today = []
_last_trade_ts = 0
_consec_losses = 0
_params = {}   # adaptive parameters from Redis
_is_trading = False  # lock: blocca nuovi segnali mentre un ordine è in esecuzione

def load_state():
    global _trades_today, _consec_losses, _params, _pending_order
    _trades_today = _rget("btc6:trades") or []
    now = time.time()
    _trades_today = [t for t in _trades_today if now - t.get("ts_close", t.get("ts", 0)) < 86400]
    _consec_losses = 0
    for t in reversed(_trades_today):
        if t.get("pnl", 0) < 0: _consec_losses += 1
        else: break
    _params = _rget("btc6:params") or {}
    # Load pending order
    _pending_order = _rget("btc6:pending") or {}
    if _pending_order:
        age = now - _pending_order.get("placed_at", 0)
        if age > PENDING_ORDER_TTL:
            log(f"🧹 Pending order scaduto in Redis ({age:.0f}s) — clearing")
            _pending_order = {}
            _rset("btc6:pending", None)
        else:
            log(f"⏳ Pending order restored from Redis (age:{age:.0f}s)")
    log(f"State: {len(_trades_today)} trades, {_consec_losses} losses")

def save_trade(pnl, direction, entry, exit_px, sig_type="", sl_dist=0, regime="",
               extra=None):
    """
    Salva trade completo su Redis con tutti i dettagli:
    entry, exit, SL/TP originali, trailing history, partial close, durata.
    """
    global _trades_today, _consec_losses
    ex = extra or {}
    t = {
        "pnl":          round(pnl, 3),
        "pnl_pct":      round(ex.get("pnl_pct", 0), 3),
        "dir":          direction,
        "entry":        entry,
        "exit":         exit_px,
        "type":         sig_type,
        "regime":       regime,
        "sl_original":  round(ex.get("sl_original", 0), 2),
        "tp_original":  round(ex.get("tp_original", 0), 2),
        "sl_dist_pct":  round(sl_dist/entry*100, 3) if entry > 0 else 0,
        "trailing_activated_at": round(ex.get("trailing_activated_at", 0), 2),
        "trailing_final_sl":    round(ex.get("trailing_final_sl", 0), 2),
        "trailing_moves":       ex.get("trailing_moves", 0),
        "partial_close_px":     round(ex.get("partial_close_px", 0), 2),
        "partial_pnl_pct":      round(ex.get("partial_pnl_pct", 0), 3),
        "ai_confidence":        ex.get("ai_confidence", 0),
        "setup_score":          ex.get("setup_score", 0),
        "close_reason":         ex.get("close_reason", ""),
        "duration_s":           round(ex.get("duration_s", 0), 0),
        "ts_open":      ex.get("ts_open", 0),
        "ts_close":     time.time(),
    }
    _trades_today.append(t)
    _rset("btc6:trades", _trades_today[-200:])
    if pnl < 0: _consec_losses += 1
    else: _consec_losses = 0
    _update_adaptive_params()

# ================================================================
# ACTIVE POSITION PERSISTENCE (Redis)
# ================================================================
def save_pos_state(pos_state):
    """Persist active position state to Redis — survives restart."""
    if pos_state:
        _rset("btc6:active_pos", pos_state)
    else:
        _rset("btc6:active_pos", None)

def load_pos_state():
    """Load active position state from Redis."""
    return _rget("btc6:active_pos")

def _update_adaptive_params():
    """
    Calcola stats rolling e adatta parametri in base alla performance.
    Salvato in Redis → persiste tra restart.
    """
    global _params
    now = time.time()
    recent = [t for t in _trades_today if now - t["ts"] < 86400]
    if len(recent) < 3:
        return

    wins = [t for t in recent if t["pnl"] > 0]
    losses = [t for t in recent if t["pnl"] < 0]
    wr = len(wins) / len(recent) if recent else 0
    avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
    avg_loss = np.mean([abs(t["pnl"]) for t in losses]) if losses else 1
    pf = avg_win * len(wins) / (avg_loss * len(losses)) if losses else 10
    total_pnl = sum(t["pnl"] for t in recent)

    # WR per tipo di segnale
    by_type = {}
    for t in recent:
        tp = t.get("type", "?")
        if tp not in by_type: by_type[tp] = {"w": 0, "l": 0}
        if t["pnl"] > 0: by_type[tp]["w"] += 1
        else: by_type[tp]["l"] += 1

    # WR per regime
    by_regime = {}
    for t in recent:
        rg = t.get("regime", "?")
        if rg not in by_regime: by_regime[rg] = {"w": 0, "l": 0}
        if t["pnl"] > 0: by_regime[rg]["w"] += 1
        else: by_regime[rg]["l"] += 1

    # Adaptive: se WR < 35% → allarga SL (troppo stretto), restringi TP (troppo ambizioso)
    # Se WR > 55% → stringi SL (cattura più profitto), allarga TP
    sl_adj = 1.0
    tp_adj = 1.0
    if len(recent) >= 5:
        if wr < 0.35:
            sl_adj = 1.15    # +15% SL (più spazio)
            tp_adj = 0.85    # -15% TP (più realistico)
        elif wr > 0.55:
            sl_adj = 0.9     # -10% SL
            tp_adj = 1.15    # +15% TP

    _params = {
        "wr": round(wr, 3), "pf": round(pf, 2), "total_pnl": round(total_pnl, 2),
        "n_trades": len(recent), "avg_win": round(avg_win, 2), "avg_loss": round(avg_loss, 2),
        "sl_adj": sl_adj, "tp_adj": tp_adj,
        "by_type": {k: round(v["w"]/(v["w"]+v["l"]), 2) for k,v in by_type.items() if v["w"]+v["l"]>0},
        "by_regime": {k: round(v["w"]/(v["w"]+v["l"]), 2) for k,v in by_regime.items() if v["w"]+v["l"]>0},
        "ts": time.time()
    }
    _rset("btc6:params", _params)
    log(f"📊 Stats: WR:{wr:.0%} PF:{pf:.2f} PnL:${total_pnl:.2f} SLadj:{sl_adj} TPadj:{tp_adj}")

def get_sl_tp_adjustments():
    """Returns (sl_mult, tp_mult) from adaptive params."""
    return _params.get("sl_adj", 1.0), _params.get("tp_adj", 1.0)

def check_circuit_breaker():
    now = time.time()
    daily_pnl = sum(t["pnl"] for t in _trades_today if now - t["ts"] < 86400)
    if daily_pnl <= -MAX_DAILY_LOSS:
        return True, f"daily loss ${daily_pnl:.1f}"
    if _consec_losses >= MAX_CONSEC_LOSS:
        # Check if 30min pause has passed since last loss
        last_loss_ts = max((t["ts"] for t in _trades_today if t["pnl"] < 0), default=0)
        if now - last_loss_ts < 1800:
            return True, f"{_consec_losses} consec losses, pause {int((1800-(now-last_loss_ts))/60)}min"
    return False, ""

# ================================================================
# FEATURES
# ================================================================
def build(df):
    c = df['close']
    df['ema9']   = c.ewm(span=9, min_periods=9).mean()
    df['ema21']  = c.ewm(span=21, min_periods=21).mean()
    df['ema50']  = c.ewm(span=50, min_periods=50).mean()
    df['ema200'] = c.ewm(span=200, min_periods=200).mean()
    df['ema_slope'] = df['ema9'].pct_change(3)

    delta = c.diff()
    up  = delta.clip(lower=0).ewm(com=13, min_periods=13).mean()
    dn  = (-delta.clip(upper=0)).ewm(com=13, min_periods=13).mean()
    df['rsi'] = 100 - (100 / (1 + up / (dn + 1e-10)))

    tr = pd.concat([df['high']-df['low'],
                     abs(df['high']-c.shift()),
                     abs(df['low']-c.shift())], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    df['vol_rel'] = df['volume'] / (df['volume'].rolling(50, min_periods=20).mean() + 1e-10)

    ema12 = c.ewm(span=12).mean()
    ema26 = c.ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_sig'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_sig']

    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    df['bb_upper'] = bb_mid + 2*bb_std
    df['bb_lower'] = bb_mid - 2*bb_std
    df['bb_pos']   = (c - bb_mid) / (2*bb_std + 1e-10)
    df['bb_width'] = (4*bb_std) / (bb_mid + 1e-10)

    df['hh'] = (df['high'] > df['high'].rolling(10).max().shift(1)).astype(int)
    df['ll'] = (df['low']  < df['low'].rolling(10).min().shift(1)).astype(int)

    # ADX — Average Directional Index (trend strength)
    plus_dm = df['high'].diff()
    minus_dm = -df['low'].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    atr14 = tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / (atr14 + 1e-10))
    minus_di = 100 * (minus_dm.rolling(14).mean() / (atr14 + 1e-10))
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    df['adx'] = dx.rolling(14).mean()

    return df.dropna(subset=['rsi','atr','ema200'])

def fetch_df(tf, days):
    now = int(time.time() * 1000)
    for _ in range(3):
        try:
            c = call(_info.candles_snapshot, COIN, tf, now - 86400000*days, now, timeout=15)
            if c and len(c) > 50:
                df = pd.DataFrame(c)
                df.columns = ['t','T','s','i','o','c','h','l','v','n']
                df[['o','c','h','l','v']] = df[['o','c','h','l','v']].astype(float)
                df.rename(columns={'o':'open','c':'close','h':'high','l':'low','v':'volume'}, inplace=True)
                return build(df)
        except: pass
        time.sleep(2)
    return None

# ================================================================
# REGIME (4h)
# ================================================================
_regime = "RANGE"
_regime_ts = 0

def update_regime():
    global _regime, _regime_ts
    if time.time() - _regime_ts < REGIME_INTERVAL:
        return _regime

    df = fetch_df("4h", 90)
    if df is None or len(df) < 200:
        return _regime

    r = df.iloc[-1]
    px = float(r['close'])
    e50 = float(r['ema50']); e200 = float(r['ema200'])
    rsi = float(r['rsi']); macd = float(r['macd_hist'])

    if px > e50 > e200 and rsi > 50:
        _regime = "BULL"
    elif px < e50 < e200 and rsi < 50:
        _regime = "BEAR"
    else:
        _regime = "RANGE"

    _regime_ts = time.time()
    return _regime

# ================================================================
# BACKTEST — testa i segnali esatti del bot su storico 5m
# ================================================================
_bt_results = {}   # {"PULLBACK_LONG": {"pf":1.3, "wr":0.45, "n":80}, ...}
_bt_ts = 0

def run_backtest():
    """
    Backtest su 14 giorni di candele 5m BTC.
    Testa ogni tipo di segnale: PULLBACK, BREAKOUT, REVERSAL × LONG/SHORT.
    Salva risultati in _bt_results — usato per bloccare segnali non profittevoli.
    """
    global _bt_results, _bt_ts
    if time.time() - _bt_ts < 3600 and _bt_results:
        return _bt_results  # cache 1 ora

    log("📊 Running backtest...")
    df_1h = fetch_df("1h", 60)
    df_5m = fetch_df("15m", 30)   # 30 giorni di 15m = ~2880 candele
    if df_1h is None or df_5m is None or len(df_5m) < 500 or len(df_1h) < 200:
        log("📊 Backtest: insufficient data")
        return _bt_results

    # Allinea 1h con 5m per timestamp
    df_1h['t_hour'] = pd.to_datetime(df_1h['t'], unit='ms').dt.floor('h')
    df_5m['t_hour'] = pd.to_datetime(df_5m['t'], unit='ms').dt.floor('h')

    # Merge 1h indicators onto 5m (forward fill)
    h_cols = df_1h[['t_hour','rsi','macd_hist','ema9','ema21','ema_slope','ema50','ema200']].copy()
    h_cols.columns = ['t_hour','rsi_1h','macd_1h','ema9_1h','ema21_1h','slope_1h','ema50_1h','ema200_1h']
    merged = pd.merge_asof(
        df_5m.sort_values('t_hour'), h_cols.sort_values('t_hour'),
        on='t_hour', direction='backward'
    ).sort_index()

    if len(merged) < 500:
        log("📊 Backtest: merge failed")
        return _bt_results

    c = merged['close'].values.astype(np.float64)
    h = merged['high'].values.astype(np.float64)
    l = merged['low'].values.astype(np.float64)
    rsi5 = merged['rsi'].values.astype(np.float64)
    macd5 = merged['macd_hist'].values.astype(np.float64)
    slope5 = merged['ema_slope'].values.astype(np.float64)
    vol5 = merged['vol_rel'].values.astype(np.float64)
    atr5 = merged['atr'].values.astype(np.float64)
    hh = merged['hh'].values.astype(np.float64)
    ll = merged['ll'].values.astype(np.float64)
    bb = merged['bb_pos'].values.astype(np.float64)

    rsi1h = merged['rsi_1h'].values.astype(np.float64)
    macd1h = merged['macd_1h'].values.astype(np.float64)
    ema9_1h = merged['ema9_1h'].values.astype(np.float64)
    ema21_1h = merged['ema21_1h'].values.astype(np.float64)
    slope1h = merged['slope_1h'].values.astype(np.float64)
    ema50_1h = merged['ema50_1h'].values.astype(np.float64)
    ema200_1h = merged['ema200_1h'].values.astype(np.float64)

    n = len(c)
    fwd = 12  # 12 candele 15m = 3h max hold
    results = {}

    for sig_name, sig_dir, sig_cond in [
        # PULLBACK LONG: BULL setup + RSI pullback + MACD accelera
        ("PULLBACK_LONG", "LONG", lambda i: (
            ema9_1h[i] > ema21_1h[i] and
            35 <= rsi5[i] <= 62 and macd5[i] > macd5[i-1] and slope5[i] > 0 and vol5[i] >= 0.3
        )),
        # BREAKOUT LONG: BULL setup + HH + volume
        ("BREAKOUT_LONG", "LONG", lambda i: (
            ema9_1h[i] > ema21_1h[i] and
            rsi5[i] > 50 and rsi5[i] < 80 and macd5[i] > 0 and hh[i] > 0 and vol5[i] >= 0.8
        )),
        # PULLBACK SHORT: BEAR setup + RSI rally + MACD decelera
        ("PULLBACK_SHORT", "SHORT", lambda i: (
            ema9_1h[i] < ema21_1h[i] and
            38 <= rsi5[i] <= 65 and macd5[i] < macd5[i-1] and slope5[i] < 0 and vol5[i] >= 0.3
        )),
        # BREAKDOWN SHORT: BEAR setup + LL + volume
        ("BREAKDOWN_SHORT", "SHORT", lambda i: (
            ema9_1h[i] < ema21_1h[i] and
            rsi5[i] < 50 and rsi5[i] > 20 and macd5[i] < 0 and ll[i] > 0 and vol5[i] >= 0.8
        )),
        # REVERSAL LONG: RSI 1h basso + 5m inversione
        ("REVERSAL_LONG", "LONG", lambda i: (
            rsi1h[i] < 38 and rsi5[i] < 35 and macd5[i] > macd5[i-1] and slope5[i] > -0.001
        )),
        # REVERSAL SHORT: RSI 1h alto + 5m inversione
        ("REVERSAL_SHORT", "SHORT", lambda i: (
            rsi1h[i] > 62 and rsi5[i] > 65 and macd5[i] < macd5[i-1] and slope5[i] < 0.001
        )),
    ]:
        trades = []
        i = 200
        while i < n - fwd:
            try:
                if not sig_cond(i):
                    i += 1; continue
            except:
                i += 1; continue

            px_i = c[i]; atr_i = atr5[i]
            if atr_i <= 0: i += 1; continue

            sl_d = max(atr_i * SL_ATR_MULT, px_i * SL_MIN_PCT)
            sl_d = min(sl_d, px_i * SL_MAX_PCT)
            tp_d = sl_d * TP_RR

            if sig_dir == "LONG":
                tp_hits = np.where(h[i+1:i+fwd] >= px_i + tp_d)[0]
                sl_hits = np.where(l[i+1:i+fwd] <= px_i - sl_d)[0]
            else:
                tp_hits = np.where(l[i+1:i+fwd] <= px_i - tp_d)[0]
                sl_hits = np.where(h[i+1:i+fwd] >= px_i + sl_d)[0]

            tp_f = tp_hits[0]+1 if len(tp_hits) else fwd+1
            sl_f = sl_hits[0]+1 if len(sl_hits) else fwd+1

            if tp_f == sl_f == fwd+1:
                i += fwd; continue  # skip forward to avoid overlap

            win = 1 if tp_f < sl_f and tp_f != sl_f else 0
            ret = (tp_d/px_i) if win else (-sl_d/px_i)
            trades.append((win, ret, tp_d if win else 0, sl_d if not win else 0))
            i += max(tp_f, sl_f, 3)  # skip past this trade

        nt = len(trades)
        if nt >= 5:
            wins = sum(t[0] for t in trades)
            gw = sum(t[2] for t in trades)
            gl = sum(t[3] for t in trades)
            wr = round(wins/nt, 3)
            pf = round(min(gw/(gl+1e-10), 10), 2)
            avg = round(sum(t[1] for t in trades)/nt, 4)

            # Ultimi 30 trade (regime recente)
            rec = trades[-30:]
            rw = sum(t[0] for t in rec)
            rpf = round(min(sum(t[2] for t in rec)/(sum(t[3] for t in rec)+1e-10), 10), 2)

            results[sig_name] = {"pf": pf, "pf_recent": rpf, "wr": wr, "n": nt, "avg_ret": avg}
            emoji = "✅" if pf >= 1.0 else "❌"
            log(f"  {emoji} {sig_name:<20} PF:{pf:.2f} WR:{wr:.0%} N:{nt} Recent:{rpf:.2f}")
        else:
            results[sig_name] = {"pf": 0, "pf_recent": 0, "wr": 0, "n": nt, "avg_ret": 0}
            log(f"  ⚠️ {sig_name:<20} only {nt} trades — insufficient")

    _bt_results = results
    _bt_ts = time.time()
    _rset("btc6:backtest", results)
    log(f"📊 Backtest done: {len(results)} signal types tested")
    return results

def is_signal_allowed(sig_type, direction):
    """Check if this signal type has positive edge in backtest."""
    key = f"{sig_type}_{direction}"
    bt = _bt_results.get(key, {})
    pf = bt.get("pf", 0)
    n = bt.get("n", 0)
    if n < 10:
        return True  # not enough data — allow
    if pf < 0.8:
        return False  # negative edge — block
    return True
def check_signal():
    """
    Returns: (direction, signal_type, sl, tp, entry_px, atr, details) or None
    Fetch 1h e 5m in PARALLELO per ridurre latenza.
    """
    regime = update_regime()

    # Fetch 1h (setup) e 15m (entry) in parallelo
    f_1h  = _pool.submit(fetch_df, "1h", 30)
    f_15m = _pool.submit(fetch_df, "15m", 14)
    try:
        df_1h  = f_1h.result(timeout=20)
        df_15m = f_15m.result(timeout=20)
    except:
        return None

    if df_1h is None or len(df_1h) < 50: return None
    if df_15m is None or len(df_15m) < 50: return None

    h = df_1h.iloc[-1]     # 1h setup
    r = df_15m.iloc[-1]    # 15m entry
    r2 = df_15m.iloc[-2]   # 15m precedente

    px    = float(r['close'])
    atr5  = float(r['atr'])
    rsi5  = float(r['rsi'])
    macd5 = float(r['macd_hist'])
    macd5_prev = float(r2['macd_hist'])
    slope5 = float(r['ema_slope'])
    vol5  = float(r['vol_rel'])
    bb5   = float(r['bb_pos'])
    adx15 = float(r.get('adx', 20))

    rsi1h  = float(h['rsi'])
    macd1h = float(h['macd_hist'])
    ema9_1h = float(h['ema9'])
    ema21_1h = float(h['ema21'])
    slope1h = float(h['ema_slope'])
    adx1h  = float(h.get('adx', 20))

    # ══════════════════════════════════════════════════════════
    # DETECT SCALP MODE
    # ══════════════════════════════════════════════════════════
    fz = get_funding_z()

    # MODE 3: FLASH — volume spike >3x O funding z-score estremo (>2.5)
    if vol5 >= 3.0 or abs(fz) > 2.5:
        scalp_mode = "FLASH"
    # MODE 2: TREND — ADX > 25 e regime direzionale
    elif adx1h > 25 and regime in ("BULL", "BEAR"):
        scalp_mode = "TREND"
    # MODE 1: RANGE — ADX < 20, prezzo tra le BB
    elif adx1h < 20 and -0.8 < bb5 < 0.8:
        scalp_mode = "RANGE"
    # Fallback: TREND se ADX > 20, RANGE altrimenti
    elif adx1h >= 20:
        scalp_mode = "TREND"
    else:
        scalp_mode = "RANGE"

    rsi1h  = float(h['rsi'])
    macd1h = float(h['macd_hist'])
    ema9_1h = float(h['ema9'])
    ema21_1h = float(h['ema21'])
    slope1h = float(h['ema_slope'])

    direction = None
    sig_type = None
    details = ""

    if regime == "BULL":
        # ── LONG ONLY ──
        setup_ok = ema9_1h > ema21_1h  # removed macd1h > 0 requirement

        if setup_ok:
            pullback = (35 <= rsi5 <= 62 and
                       macd5 > macd5_prev and
                       slope5 > 0 and
                       vol5 >= 0.3)

            breakout = (rsi5 > 50 and rsi5 < 80 and
                       macd5 > 0 and
                       float(r['hh']) > 0 and
                       vol5 >= 0.8)

            if pullback and is_signal_allowed("PULLBACK", "LONG"):
                direction = "LONG"; sig_type = "PULLBACK"
                details = f"RSI5:{rsi5:.0f} MACD↑ slope:{slope5:.4f}"
            elif breakout and is_signal_allowed("BREAKOUT", "LONG"):
                direction = "LONG"; sig_type = "BREAKOUT"
                details = f"RSI5:{rsi5:.0f} HH vol:{vol5:.1f}x"

    elif regime == "BEAR":
        # ── SHORT ONLY ──
        setup_ok = ema9_1h < ema21_1h and macd1h < 0

        if setup_ok:
            pullback = (38 <= rsi5 <= 65 and
                       macd5 < macd5_prev and
                       slope5 < 0 and
                       vol5 >= 0.3)

            breakdown = (rsi5 < 50 and rsi5 > 20 and
                        macd5 < 0 and
                        float(r['ll']) > 0 and
                        vol5 >= 0.8)

            if pullback and is_signal_allowed("PULLBACK", "SHORT"):
                direction = "SHORT"; sig_type = "PULLBACK"
                details = f"RSI5:{rsi5:.0f} MACD↓ slope:{slope5:.4f}"
            elif breakdown and is_signal_allowed("BREAKDOWN", "SHORT"):
                direction = "SHORT"; sig_type = "BREAKDOWN"
                details = f"RSI5:{rsi5:.0f} LL vol:{vol5:.1f}x"

    else:  # RANGE
        # ── REVERSAL ──
        if rsi1h < 38:
            if (rsi5 < 35 and macd5 > macd5_prev and slope5 > -0.001):
                if is_signal_allowed("REVERSAL", "LONG"):
                    direction = "LONG"; sig_type = "REVERSAL"
                    details = f"RSI1h:{rsi1h:.0f} RSI5:{rsi5:.0f} MACD turning"

        if rsi1h > 62 and direction is None:
            if (rsi5 > 65 and macd5 < macd5_prev and slope5 < 0.001):
                if is_signal_allowed("REVERSAL", "SHORT"):
                    direction = "SHORT"; sig_type = "REVERSAL"
                    details = f"RSI1h:{rsi1h:.0f} RSI5:{rsi5:.0f} MACD turning"

    if direction is None:
        return None

    # ── AI CONTEXT ANALYSIS (async — non blocca il calcolo SL/TP) ──
    ai_future = None
    try:
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if api_key:
            ohlcv_str = ""
            for _, row in df_15m.iloc[-6:][['close','high','low','volume']].iterrows():
                ohlcv_str += f"C:{row['close']:.1f} H:{row['high']:.1f} L:{row['low']:.1f} V:{row['volume']:.0f}\n"

            prompt = (f"BTC {direction} {sig_type} signal. Regime:{regime}. "
                     f"RSI15m:{rsi5:.0f} RSI1h:{rsi1h:.0f} MACD15m:{macd5:.1f} Vol:{vol5:.1f}x BB:{bb5:.2f}\n"
                     f"Last 6 candles 15m:\n{ohlcv_str}"
                     f"Rate confidence 1-10 and suggest SL adjustment (tight/normal/wide).\n"
                     f"JSON only: {{\"confidence\":<1-10>,\"sl\":\"tight|normal|wide\",\"note\":\"<5 words>\"}}")

            def _ai_call():
                r = requests.post("https://api.anthropic.com/v1/messages",
                    headers={"Content-Type": "application/json", "x-api-key": api_key,
                             "anthropic-version": "2023-06-01"},
                    json={"model": "claude-haiku-4-5-20251001", "max_tokens": 150,
                          "messages": [{"role": "user", "content": prompt}]}, timeout=10)
                return r
            ai_future = _pool.submit(_ai_call)
    except: pass

    # ── SL / TP (calcolati MENTRE l'AI risponde) ──
    ai_confidence = 7
    ai_sl_adj = 1.0
    ai_note = ""

    # Collect AI result (max 8s wait — se non risponde, defaults)
    if ai_future:
        try:
            resp = ai_future.result(timeout=8)
            if resp.status_code == 200:
                txt = resp.json()["content"][0]["text"]
                start = txt.find("{"); end = txt.rfind("}")+1
                if start >= 0 and end > start:
                    aj = json.loads(txt[start:end])
                    ai_confidence = min(10, max(1, int(aj.get("confidence", 7))))
                    sl_sug = aj.get("sl", "normal")
                    if sl_sug == "tight": ai_sl_adj = 0.85
                    elif sl_sug == "wide": ai_sl_adj = 1.2
                    ai_note = aj.get("note", "")
        except: pass

    details += f" | AI:{ai_confidence}/10 {ai_note}"

    # ── Funding z-score e OI (already called for mode detect) ──
    update_funding_oi()
    oi_chg = get_oi_change()
    details += f" FZ:{fz:+.1f} OI:{oi_chg:+.2%} [{scalp_mode}]"

    # ── SL / TP — MODE ADAPTIVE + lev_scale ──
    actual_lev = get_effective_lev()
    lev_scale = LEVERAGE / max(actual_lev, 1)
    if lev_scale != 1.0:
        log(f"⚠️ Lev {actual_lev}x vs {LEVERAGE}x → SL ×{lev_scale:.2f}")

    sl_adj_redis, tp_adj_redis = get_sl_tp_adjustments()
    buffer = px * 0.002 * lev_scale

    if scalp_mode == "FLASH":
        # FLASH: SL stretto, TP fisso 0.3%, no trailing
        sl_atr_m = FLASH_SL_ATR * lev_scale * sl_adj_redis * ai_sl_adj
        sl_dist = atr5 * sl_atr_m
        sl_dist = max(sl_dist, px * FLASH_SL_MIN * lev_scale)
        sl_dist = min(sl_dist, px * FLASH_SL_MAX * lev_scale)
        tp_dist = px * FLASH_TP_PCT * tp_adj_redis

    elif scalp_mode == "RANGE":
        # RANGE: SL da ATR, TP fisso 0.4%
        sl_atr_m = RANGE_SL_ATR * lev_scale * sl_adj_redis * ai_sl_adj
        sl_dist = atr5 * sl_atr_m
        sl_dist = max(sl_dist, px * RANGE_SL_MIN * lev_scale)
        sl_dist = min(sl_dist, px * RANGE_SL_MAX * lev_scale)
        tp_dist = px * RANGE_TP_PCT * tp_adj_redis

    else:  # TREND
        # TREND: SL da ATR, TP da R:R (trailing farà il resto)
        sl_atr_m = TREND_SL_ATR * lev_scale * sl_adj_redis * ai_sl_adj
        sl_dist = atr5 * sl_atr_m
        sl_dist = max(sl_dist, px * TREND_SL_MIN * lev_scale)
        sl_dist = min(sl_dist, px * TREND_SL_MAX * lev_scale)
        tp_dist = sl_dist * TREND_TP_RR * tp_adj_redis / lev_scale

    # TP floor (non scalano con leva)
    tp_dist = max(tp_dist, px * 0.002)  # min 0.2%

    # Swing SL override (se migliore dell'ATR)
    if direction == "LONG":
        swing_low = float(df_15m['low'].iloc[-10:].min())
        swing_sl = px - swing_low + buffer
        sl_min = px * (FLASH_SL_MIN if scalp_mode=="FLASH" else RANGE_SL_MIN if scalp_mode=="RANGE" else TREND_SL_MIN)
        sl_max = px * (FLASH_SL_MAX if scalp_mode=="FLASH" else RANGE_SL_MAX if scalp_mode=="RANGE" else TREND_SL_MAX) * lev_scale
        if sl_min < swing_sl < sl_max:
            sl_dist = swing_sl
        sl = px - sl_dist
        tp = px + tp_dist
    else:
        swing_high = float(df_15m['high'].iloc[-10:].max())
        swing_sl = swing_high - px + buffer
        sl_min = px * (FLASH_SL_MIN if scalp_mode=="FLASH" else RANGE_SL_MIN if scalp_mode=="RANGE" else TREND_SL_MIN)
        sl_max = px * (FLASH_SL_MAX if scalp_mode=="FLASH" else RANGE_SL_MAX if scalp_mode=="RANGE" else TREND_SL_MAX) * lev_scale
        if sl_min < swing_sl < sl_max:
            sl_dist = swing_sl
        sl = px + sl_dist
        tp = px - tp_dist

    sl_pct = sl_dist / px * 100
    tp_pct = tp_dist / px * 100
    rr = tp_dist / sl_dist if sl_dist > 0 else 0
    log(f"[{scalp_mode}] SL:{sl_pct:.2f}% TP:{tp_pct:.2f}% R:R=1:{rr:.1f} ADX:{adx1h:.0f} lev:{actual_lev}x")

    # Confidence sizing: AI confidence 1-10 → size multiplier 0.5-1.0
    size_mult = 0.5 + (ai_confidence - 1) * 0.055  # 1→0.5, 5→0.72, 10→1.0
    size_mult = max(0.5, min(1.0, size_mult))

    # ── LIQUIDITY CHECK + SETUP SCORE ──
    liq = fetch_liquidity()
    if liq.get("spread") is not None and liq["spread"] > 0.001:
        log(f"⚠️ Spread troppo alto: {liq['spread']:.4%} — skip")
        return None

    ema50_1h = float(h.get('ema50', px))
    ema200_1h = float(h.get('ema200', px))
    setup = compute_setup_score(direction, px, ema50_1h, ema200_1h,
                                rsi5, ai_confidence, fz, liq)
    if setup < 40:
        log(f"⚠️ Setup score {setup}/100 — troppo basso")
        return None

    details += f" score:{setup}"
    liq_str = ""
    if liq.get("spread") is not None:
        liq_str = f" spread:{liq['spread']:.3%} imb:{liq['imbalance']:.0%}"
        details += liq_str

    return (direction, sig_type, sl, tp, px, atr5, details, sl_dist, size_mult, regime, setup, scalp_mode)

# ================================================================
# EXECUTION
# ================================================================
_coin_meta = {}

def get_meta():
    """
    Recupera szDecimals e maxLeverage da Hyperliquid.
    priceDecimals non esiste nell'API — calcolato dal tick size.
    """
    global _coin_meta
    for attempt in range(3):
        try:
            m = call(_info.meta, timeout=15)
            for a in m["universe"]:
                if a["name"] == COIN:
                    sz_dec = int(a["szDecimals"])
                    max_lev = int(a.get("maxLeverage", 50))

                    # Price decimals: Hyperliquid non ha questo campo.
                    # BTC usa 1 decimale ($73692.0), ma per SL/TP precisi
                    # calcoliamo dal prezzo corrente.
                    px_dec = 1  # default BTC
                    try:
                        mid = float(call(_info.all_mids, timeout=10).get(COIN, 0))
                        if mid > 0:
                            # Conta decimali significativi dal prezzo
                            px_str = f"{mid:.10f}".rstrip('0')
                            if '.' in px_str:
                                px_dec = len(px_str.split('.')[1])
                                px_dec = max(1, min(px_dec, 6))  # clamp 1-6
                    except: pass

                    _coin_meta = {
                        "sz_dec":  sz_dec,
                        "px_dec":  px_dec,
                        "max_lev": max_lev,
                    }
                    log(f"✅ Meta {COIN}: szDec={sz_dec} pxDec={px_dec} maxLev={max_lev}x")

                    if LEVERAGE > max_lev:
                        log(f"⚠️ LEVERAGE {LEVERAGE}x > max {max_lev}x")
                        tg(f"⚠️ Leva {LEVERAGE}x > max {max_lev}x per {COIN}")

                    return sz_dec, px_dec
        except Exception as e:
            if attempt < 2: time.sleep(3 * (attempt + 1))
            else: log(f"get_meta failed: {e}")
    return 5, 1  # BTC defaults

def get_max_leverage():
    return _coin_meta.get("max_lev", 50)

def get_position():
    try:
        s = call(_info.user_state, _account.address, timeout=15)
        for p in s.get("assetPositions", []):
            pp = p["position"]
            if pp["coin"] == COIN and float(pp["szi"]) != 0:
                return {
                    "szi": float(pp["szi"]),
                    "entry": float(pp.get("entryPx", 0)),
                    "lev": int(pp.get("leverage", {}).get("value", LEVERAGE))
                            if isinstance(pp.get("leverage"), dict) else LEVERAGE
                }
    except: pass
    return None

def get_effective_lev():
    """
    Legge la leva effettiva da Hyperliquid.
    Hyperliquid può applicare leva diversa da quella richiesta.
    """
    try:
        s = call(_info.user_state, _account.address, timeout=10)
        for p in s.get("assetPositions", []):
            pp = p.get("position", {})
            if pp.get("coin") == COIN:
                lev = pp.get("leverage", {})
                if isinstance(lev, dict):
                    return int(lev.get("value", LEVERAGE))
                if isinstance(lev, (int, float)):
                    return int(lev)
        return LEVERAGE
    except:
        return LEVERAGE
    try:
        s = call(_info.user_state, _account.address, timeout=10)
        return float(s["marginSummary"]["accountValue"])
    except: return 0

def get_mid():
    try:
        mids = call(_info.all_mids, timeout=10)
        return float(mids.get(COIN, 0))
    except: return 0

def get_funding():
    """Returns current BTC funding rate (raw, not bps)."""
    try:
        ctx = call(_info.meta_and_asset_ctxs, timeout=15)
        for a, c in zip(ctx[0]["universe"], ctx[1]):
            if a["name"] == COIN:
                return float(c.get("funding", 0) or 0)
    except: pass
    return 0

def get_open_orders():
    try:
        return call(_info.open_orders, _account.address, timeout=10)
    except: return []

def check_margin(size, entry_px):
    """
    Verifica che ci sia margine sufficiente prima di inviare l'ordine.
    Calcola il margine richiesto e lo confronta con il margine disponibile.
    Evita errori API superflui e ordini rifiutati.
    """
    try:
        s = call(_info.user_state, _account.address, timeout=10)
        margin_summary = s.get("marginSummary", {})
        account_value = float(margin_summary.get("accountValue", 0) or 0)
        total_margin  = float(margin_summary.get("totalMarginUsed", 0) or 0)
        available = account_value - total_margin

        # Margine richiesto per questo trade
        notional = size * entry_px
        margin_required = notional / LEVERAGE
        # Buffer 20%: serve margine extra per SL/TP trigger orders e fluttuazioni
        margin_with_buffer = margin_required * 1.2

        if available < margin_with_buffer:
            log(f"⚠️ Margine insufficiente: need ${margin_with_buffer:.2f} "
                f"have ${available:.2f} (used:${total_margin:.2f} total:${account_value:.2f})")
            return False

        # Check anche che l'account value sia sopra una soglia minima
        if account_value < RISK_USD * 2:
            log(f"⚠️ Account value troppo basso: ${account_value:.2f}")
            return False

        return True
    except Exception as e:
        log(f"check_margin error: {e}")
        return True

# ================================================================
# ORDERBOOK & LIQUIDITY (from V4)
# ================================================================
def fetch_liquidity():
    """
    Orderbook L2 per BTC: spread, imbalance, cluster liquidazione.
    Usato per setup_score e per decidere se il mercato è tradabile.
    """
    result = {"spread": None, "imbalance": 0.5, "aggressive": False,
              "cluster_dist": None, "cluster_side": None}
    try:
        ob = call(_info.l2_snapshot, COIN, timeout=10)
        if not ob: return result

        levels = ob.get("levels", [])
        bids = levels[0] if len(levels) > 0 else []
        asks = levels[1] if len(levels) > 1 else []
        if not bids or not asks: return result

        best_bid = float(bids[0]["px"])
        best_ask = float(asks[0]["px"])
        mid = (best_bid + best_ask) / 2
        result["spread"] = (best_ask - best_bid) / (mid + 1e-10)

        # Imbalance: sum top 10 levels
        bid_sz = sum(float(b["sz"]) for b in bids[:10])
        ask_sz = sum(float(a["sz"]) for a in asks[:10])
        result["imbalance"] = bid_sz / (bid_sz + ask_sz + 1e-10)
        result["aggressive"] = result["imbalance"] > 0.65

        # Liquidation clusters: level con size massima
        ask_levels = [(float(a["px"]), float(a["sz"])) for a in asks[:20]]
        bid_levels = [(float(b["px"]), float(b["sz"])) for b in bids[:20]]

        if ask_levels:
            mx = max(ask_levels, key=lambda x: x[1])
            result["cluster_dist"] = (mx[0] - mid) / (mid + 1e-10)
            result["cluster_side"] = "above"
        if bid_levels:
            mx = max(bid_levels, key=lambda x: x[1])
            d = (mid - mx[0]) / (mid + 1e-10)
            if result["cluster_dist"] is None or d < result["cluster_dist"]:
                result["cluster_dist"] = d
                result["cluster_side"] = "below"
    except Exception as e:
        log(f"fetch_liquidity: {e}")
    return result

# ================================================================
# SETUP SCORE (from V4 — adapted for BTC-only)
# ================================================================
def compute_setup_score(direction, px, ema50, ema200, rsi, ai_score,
                        funding_z, liq=None):
    """
    Score composito 0-100. Combina trend, RSI, AI, funding z-score, liquidità.
    """
    score = 0

    # Trend Confluence (30%)
    if direction == "LONG":
        if px > ema200 and ema50 > ema200: score += 30
        elif px > ema200 or ema50 > ema200: score += 15
    else:
        if px < ema200 and ema50 < ema200: score += 30
        elif px < ema200 or ema50 < ema200: score += 15

    # RSI zone (20%)
    if direction == "LONG":
        if 40 <= rsi <= 60: score += 20
        elif rsi > 60: score += 10
    else:
        if 40 <= rsi <= 60: score += 20
        elif rsi < 40: score += 10

    # AI confidence (25%)
    score += int((ai_score / 10) * 25)

    # Funding z-score (10%)
    if direction == "LONG":
        if funding_z < -1.0: score += 10    # ti pagano per long
        elif funding_z < 0: score += 5
        elif funding_z > 2.0: score -= 10   # crowded long = pericolo
    else:
        if funding_z > 1.0: score += 10
        elif funding_z > 0: score += 5
        elif funding_z < -2.0: score -= 10

    # Liquidity (15% + bonus/penalty)
    if liq and liq.get("spread") is not None:
        sp = liq["spread"]
        if sp < 0.0003: score += 15       # < 0.03% tight
        elif sp < 0.0008: score += 10     # < 0.08%
        elif sp > 0.001: score -= 40      # > 0.1% danger

        imb = liq.get("imbalance", 0.5)
        if direction == "LONG" and imb > 0.60: score += 15
        elif direction == "SHORT" and imb < 0.40: score += 15

        cd = liq.get("cluster_dist")
        cs = liq.get("cluster_side")
        if cd is not None and cd < 0.03:
            if (direction == "LONG" and cs == "above") or \
               (direction == "SHORT" and cs == "below"):
                score += 30    # liquidation magnet

        if liq.get("aggressive"):
            if direction == "LONG": score += 10
    else:
        score += 8  # no data = neutral

    return max(0, min(100, score))

# ================================================================
# REAL EXIT PRICE (from V4 — precise PnL from fills)
# ================================================================
def get_recent_fills(since_ts):
    """Recupera fill recenti BTC da API."""
    try:
        fills = call(_info.user_fills_by_time, _account.address,
                     int(since_ts * 1000), timeout=15)
        return [f for f in (fills or []) if f.get("coin") == COIN]
    except: return []

def compute_real_exit(direction, entry_px, ts_open):
    """
    Calcola exit price reale dalla media ponderata dei fill di chiusura.
    Fallback a 0 se fill non disponibili.
    """
    try:
        fills = get_recent_fills(ts_open)
        close_side = "B" if direction == "SHORT" else "A"
        close_fills = [f for f in fills
                       if f.get("side", "")[0:1].upper() == close_side
                       and f.get("dir", "") == "Close"]
        if not close_fills:
            close_fills = [f for f in fills
                           if f.get("side", "")[0:1].upper() == close_side]
        if close_fills:
            total_sz = sum(float(f.get("sz", 0)) for f in close_fills)
            if total_sz > 0:
                exit_px = sum(float(f.get("px", 0)) * float(f.get("sz", 0))
                              for f in close_fills) / total_sz
                pnl_pct = ((exit_px - entry_px) / entry_px if direction == "LONG"
                           else (entry_px - exit_px) / entry_px)
                return exit_px, pnl_pct
    except Exception as e:
        log(f"compute_real_exit: {e}")
    return 0, 0

# ================================================================
# TRIGGER ORDERS & CLEANUP (from V4)
# ================================================================
def get_trigger_orders():
    """Returns SL/TP trigger orders for BTC."""
    try:
        orders = call(_info.open_orders, _account.address, timeout=10)
        return [o for o in (orders or [])
                if o.get("coin") == COIN
                and o.get("orderType") in ("Stop Market", "Take Profit Market")]
    except: return []

def cancel_trigger_order(oid):
    try:
        res = call(_exchange.cancel, COIN, oid, timeout=10)
        return res and res.get("status") == "ok"
    except: return False

def cleanup_orphan_orders():
    """
    Cancella ordini trigger orfani — SL/TP senza posizione aperta.
    Chiamato al startup e periodicamente.
    """
    pos = get_position()
    if pos is not None:
        return  # posizione aperta — ordini sono legittimi

    triggers = get_trigger_orders()
    if triggers:
        log(f"🧹 {len(triggers)} ordini orfani trovati — cancello")
        for o in triggers:
            oid = o.get("oid")
            if oid:
                cancel_trigger_order(oid)
                log(f"  Cancelled {o.get('orderType')} #{oid}")
        tg(f"🧹 Cancellati {len(triggers)} ordini BTC orfani", silent=True)

# ================================================================
# PENDING ORDERS TRACKING (from V4)
# ================================================================
_pending_order = {}  # {"oid": X, "placed_at": T, "sl": S, "tp": T, "direction": D, "sl_dist": SD}

def set_pending(oid, sl, tp, direction, sl_dist, sig_type="", regime=""):
    """Registra un ordine GTC che non è stato fillato immediatamente."""
    global _pending_order
    _pending_order = {
        "oid": oid, "placed_at": time.time(),
        "sl": sl, "tp": tp, "direction": direction,
        "sl_dist": sl_dist, "type": sig_type, "regime": regime
    }
    _rset("btc6:pending", _pending_order)
    log(f"⏳ Ordine pendente registrato (oid={oid})")

def clear_pending():
    global _pending_order
    _pending_order = {}
    _rset("btc6:pending", None)

def check_pending(sz_dec, px_dec):
    """
    Controlla se un ordine pendente è stato fillato o è scaduto.
    Se fillato → piazza SL/TP.
    Se scaduto → cancella.
    """
    global _pending_order
    if not _pending_order:
        return None

    oid = _pending_order.get("oid")
    placed_at = _pending_order.get("placed_at", 0)
    now = time.time()

    # Check se è stato fillato (posizione aperta)
    pos = get_position()
    if pos is not None:
        direction = _pending_order["direction"]
        is_buy = direction == "LONG"
        sl_px = rpx(_pending_order["sl"], px_dec)
        tp_px = rpx(_pending_order["tp"], px_dec)
        actual_size = rpx(abs(pos["szi"]), sz_dec)

        log(f"✅ Pendente fillato → piazzo SL/TP")
        try:
            call(_exchange.order, COIN, not is_buy, actual_size, sl_px,
                 {"trigger": {"triggerPx": sl_px, "isMarket": True, "tpsl": "sl"}},
                 True, timeout=15)
            time.sleep(0.3)
            call(_exchange.order, COIN, not is_buy, actual_size, tp_px,
                 {"trigger": {"triggerPx": tp_px, "isMarket": True, "tpsl": "tp"}},
                 True, timeout=15)

            entry = pos["entry"]
            sl_pct = abs(entry - sl_px) / entry * 100
            tp_pct = abs(tp_px - entry) / entry * 100
            log(f"🔒 Pendente protetto: SL:{sl_px}({sl_pct:.1f}%) TP:{tp_px}({tp_pct:.1f}%)")
            tg(f"🔒 <b>BTC</b> pendente fillato | SL:{sl_px} TP:{tp_px}", silent=True)
        except Exception as e:
            log(f"🚨 SL/TP pendente error: {e}")
            tg(f"🚨 BTC pendente SL/TP ERROR!")

        result = _pending_order.copy()
        result["entry"] = pos["entry"]
        result["szi"] = pos["szi"]
        clear_pending()
        return result

    # Check se è scaduto
    if now - placed_at > PENDING_ORDER_TTL:
        log(f"⏱ Pendente scaduto dopo {PENDING_ORDER_TTL}s — cancello")
        try:
            if oid:
                call(_exchange.cancel, COIN, oid, timeout=10)
                log(f"  Cancelled GTC #{oid}")
        except Exception as e:
            log(f"  Cancel error: {e}")
        tg(f"⏱ BTC ordine scaduto — cancellato", silent=True)
        clear_pending()
        return None

    # Ancora in attesa
    elapsed = int(now - placed_at)
    if elapsed % 30 == 0:  # log ogni 30s
        log(f"⏳ Pendente in attesa... {elapsed}s/{PENDING_ORDER_TTL}s")
    return None

# ================================================================
# STARTUP RECOVERY
# ================================================================
def recover_position(sz_dec, px_dec):
    """
    Al restart: se c'è una posizione aperta senza SL, piazza SL di emergenza.
    Ritorna il pos_state per tracking.
    """
    pos = get_position()
    if pos is None:
        log("🔍 No open position — clean start")
        return None

    entry = pos["entry"]
    szi = pos["szi"]
    d = "LONG" if szi > 0 else "SHORT"
    mid = get_mid()
    pnl_pct = ((mid-entry)/entry if d == "LONG" else (entry-mid)/entry) * 100

    log(f"🔍 Found open position: {d} @ {entry} size:{abs(szi)} PnL:{pnl_pct:+.1f}%")

    # Check if SL exists
    orders = get_open_orders()
    has_sl = any(o.get("coin") == COIN and o.get("orderType") == "Stop Market" for o in orders)
    has_tp = any(o.get("coin") == COIN and o.get("orderType") == "Take Profit Market" for o in orders)

    if not has_sl:
        # Emergency SL at 3× SL_MAX_PCT (wide, just protection)
        emergency_dist = entry * SL_MAX_PCT * 2
        if d == "LONG":
            sl_px = rpx(entry - emergency_dist, px_dec)
        else:
            sl_px = rpx(entry + emergency_dist, px_dec)

        try:
            size_abs = rpx(abs(szi), sz_dec)
            is_buy = d == "LONG"
            call(_exchange.order, COIN, not is_buy, size_abs, sl_px,
                 {"trigger": {"triggerPx": sl_px, "isMarket": True, "tpsl": "sl"}},
                 True, timeout=15)
            log(f"🚨 EMERGENCY SL placed @ {sl_px}")
            tg(f"🚨 <b>BTC</b> restart — emergency SL @ {sl_px}")
        except Exception as e:
            log(f"🚨 EMERGENCY SL FAILED: {e}")
            tg(f"🚨🚨 BTC NO SL — MANUAL CHECK!")

    if not has_tp:
        log(f"⚠️ No TP order — trade will be managed by AI only")

    pos_state = pos.copy()
    pos_state["type"] = "RECOVERED"
    pos_state["regime"] = _regime
    pos_state["sl_dist"] = entry * SL_MAX_PCT
    pos_state["last_mgmt"] = 0
    pos_state["partial_done"] = False
    pos_state["trailing_active"] = False
    pos_state["atr"] = 0
    return pos_state

# ================================================================
# TRAILING STOP MECCANICO
# ================================================================
def update_trailing(pos_state, mid, atr, sz_dec, px_dec):
    """
    Trailing stop meccanico:
    1. Attiva dopo che il prezzo raggiunge TRAILING_ACTIVATE × TP distance
    2. Trail a TRAILING_ATR × ATR dal prezzo corrente
    3. Solo in direzione favorevole (mai allontana lo SL)
    """
    if atr <= 0:
        return

    entry = pos_state["entry"]
    szi = pos_state["szi"]
    d = "LONG" if szi > 0 else "SHORT"
    sl_dist_orig = pos_state.get("sl_dist", entry * SL_MAX_PCT)
    tp_dist = sl_dist_orig * TP_RR

    # Quanto profitto percentuale
    if d == "LONG":
        profit_dist = mid - entry
    else:
        profit_dist = entry - mid

    # Attivazione: profitto > TRAILING_ACTIVATE × tp_dist
    if profit_dist < tp_dist * TRAILING_ACTIVATE:
        return  # non ancora

    if not pos_state.get("trailing_active"):
        pos_state["trailing_active"] = True
        pos_state["trailing_activated_at"] = mid
        pos_state["trailing_moves"] = 0
        log(f"📈 Trailing ACTIVATED at {mid:.1f} (profit:{profit_dist/entry*100:.2f}%)")
        save_pos_state(pos_state)

    # Calcola nuovo trailing SL
    trail_dist = atr * TRAILING_ATR
    if d == "LONG":
        new_ts = mid - trail_dist
        # Mai sotto l'entry (almeno breakeven)
        new_ts = max(new_ts, entry * 1.0005)
    else:
        new_ts = mid + trail_dist
        new_ts = min(new_ts, entry * 0.9995)

    new_ts = rpx(new_ts, px_dec)
    old_ts = pos_state.get("current_ts", 0)

    # Solo muovi in direzione favorevole
    if d == "LONG" and new_ts <= old_ts:
        return
    if d == "SHORT" and new_ts >= old_ts and old_ts > 0:
        return

    # Aggiorna SL su exchange
    try:
        orders = get_open_orders()
        for o in orders:
            if o.get("coin") == COIN and o.get("orderType") == "Stop Market":
                call(_exchange.cancel, COIN, o["oid"], timeout=10)
        time.sleep(0.3)

        size_abs = rpx(abs(szi), sz_dec)
        is_buy = d == "LONG"
        call(_exchange.order, COIN, not is_buy, size_abs, new_ts,
             {"trigger": {"triggerPx": new_ts, "isMarket": True, "tpsl": "sl"}},
             True, timeout=15)

        pos_state["current_ts"] = new_ts
        pos_state["trailing_moves"] = pos_state.get("trailing_moves", 0) + 1
        trail_pct = abs(new_ts - entry) / entry * 100
        log(f"📈 Trailing SL → {new_ts} ({'+' if (d=='LONG' and new_ts>entry) or (d=='SHORT' and new_ts<entry) else ''}{trail_pct:.2f}% from entry) [move #{pos_state['trailing_moves']}]")
        save_pos_state(pos_state)
    except Exception as e:
        log(f"Trailing update error: {e}")

# ================================================================
# PARTIAL CLOSE
# ================================================================
def check_partial_close(pos_state, mid, sz_dec, px_dec):
    """
    Chiudi 50% della posizione quando il prezzo raggiunge PARTIAL_CLOSE_PCT del TP.
    Prende profitto parziale e lascia correre il resto con trailing.
    """
    if pos_state.get("partial_done"):
        return

    entry = pos_state["entry"]
    szi = pos_state["szi"]
    d = "LONG" if szi > 0 else "SHORT"
    sl_dist_orig = pos_state.get("sl_dist", entry * SL_MAX_PCT)
    tp_dist = sl_dist_orig * TP_RR

    if d == "LONG":
        profit_dist = mid - entry
    else:
        profit_dist = entry - mid

    if profit_dist < tp_dist * PARTIAL_CLOSE_PCT:
        return

    # Chiudi 50%
    close_size = rpx(abs(szi) * 0.5, sz_dec)
    if close_size <= 0:
        return

    try:
        is_buy = d == "LONG"
        close_px = rpx(mid * (0.998 if is_buy else 1.002), px_dec)
        call(_exchange.order, COIN, not is_buy, close_size, close_px,
             {"limit": {"tif": "Ioc"}}, False, timeout=15)

        pos_state["partial_done"] = True
        pos_state["partial_close_px"] = mid
        pos_state["partial_pnl_pct"] = profit_dist / entry * 100
        partial_pnl = profit_dist / entry * 100
        log(f"💰 PARTIAL CLOSE 50% @ {mid:.1f} PnL:{partial_pnl:+.2f}%")
        tg(f"💰 BTC partial close 50% PnL:{partial_pnl:+.1f}%", silent=True)
        save_pos_state(pos_state)

        # Aggiorna TP order per la size rimanente
        time.sleep(1)
        remaining = rpx(abs(szi) - close_size, sz_dec)
        if remaining > 0:
            orders = get_open_orders()
            for o in orders:
                if o.get("coin") == COIN and o.get("orderType") == "Take Profit Market":
                    call(_exchange.cancel, COIN, o["oid"], timeout=10)
            # Non ripiazzare TP — lascia correre con trailing
            log(f"📈 Remaining {remaining} runs with trailing stop")
    except Exception as e:
        log(f"Partial close error: {e}")

# ================================================================
# FUNDING CHECK
# ================================================================
def is_funding_ok(direction):
    """
    Blocca entry se funding contro la posizione.
    Usa sia il raw funding rate che il z-score.
    """
    funding = get_funding()
    fz = get_funding_z()

    # Raw funding check
    if direction == "LONG" and funding > FUNDING_BLOCK_THRESH:
        log(f"⚠️ Funding {funding*100:.3f}% contro LONG — skip")
        return False
    if direction == "SHORT" and funding < -FUNDING_BLOCK_THRESH:
        log(f"⚠️ Funding {funding*100:.3f}% contro SHORT — skip")
        return False

    # Z-score check: funding estremo = crowded = rischio squeeze
    if direction == "LONG" and fz > 2.5:
        log(f"⚠️ Funding z-score {fz:+.1f} — crowded long, skip")
        return False
    if direction == "SHORT" and fz < -2.5:
        log(f"⚠️ Funding z-score {fz:+.1f} — crowded short, skip")
        return False

    return True

# ================================================================
# DAILY REPORT
# ================================================================
_last_report_ts = 0

def maybe_send_daily_report():
    global _last_report_ts
    now = time.time()
    if now - _last_report_ts < 82800:  # max once per 23h
        return

    utc_hour = datetime.now(timezone.utc).hour
    if utc_hour != DAILY_REPORT_HOUR:
        return

    _last_report_ts = now
    trades = [t for t in _trades_today if now - t["ts"] < 86400]
    if not trades:
        tg("📊 <b>Daily Report</b>\nNo trades today")
        return

    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    total = sum(t["pnl"] for t in trades)
    wr = len(wins)/len(trades)*100 if trades else 0
    avg_w = np.mean([t["pnl"] for t in wins]) if wins else 0
    avg_l = np.mean([t["pnl"] for t in losses]) if losses else 0
    best = max(trades, key=lambda t: t["pnl"])
    worst = min(trades, key=lambda t: t["pnl"])

    by_type = {}
    for t in trades:
        tp = t.get("type", "?")
        if tp not in by_type: by_type[tp] = []
        by_type[tp].append(t["pnl"])

    type_str = "\n".join(f"  {k}: {len(v)} trades ${sum(v):+.2f}"
                         for k,v in by_type.items())

    bal = get_balance()
    report = (
        f"📊 <b>BTC Scalper Daily Report</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"💰 PnL: <b>${total:+.2f}</b>\n"
        f"📈 Trades: {len(trades)} (W:{len(wins)} L:{len(losses)})\n"
        f"🎯 Win Rate: {wr:.0f}%\n"
        f"📊 Avg Win: ${avg_w:+.2f} | Avg Loss: ${avg_l:+.2f}\n"
        f"🏆 Best: ${best['pnl']:+.2f} ({best.get('type','?')})\n"
        f"💀 Worst: ${worst['pnl']:+.2f} ({worst.get('type','?')})\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"By signal type:\n{type_str}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"💼 Balance: ${bal:.2f}"
    )
    tg(report)
    log(f"📊 Daily report sent: {len(trades)} trades ${total:+.2f}")

def open_trade(direction, sl, tp, entry_px, sl_dist, sz_dec, px_dec, size_mult=1.0, scalp_mode="TREND"):
    global _last_trade_ts, _is_trading

    # Lock: previeni race condition (ordini doppi)
    if _is_trading:
        log("⚠️ Trade already in progress — skipping")
        return False
    _is_trading = True

    try:
        return _execute_trade(direction, sl, tp, entry_px, sl_dist, sz_dec, px_dec, size_mult, scalp_mode)
    finally:
        _is_trading = False

def _execute_trade(direction, sl, tp, entry_px, sl_dist, sz_dec, px_dec, size_mult=1.0, scalp_mode="TREND"):
    global _last_trade_ts
    is_buy = direction == "LONG"

    # Size basato su risk: RISK_USD / sl_dist_pct / lev_effective
    # Se SL = 0.3% e risk = $5, size = $5 / 0.003 = $1666 notional
    # Con leva 20x, margine = $83
    sl_pct = sl_dist / entry_px
    notional = (RISK_USD * size_mult) / sl_pct
    # Cap al balance × leverage
    bal = get_balance()
    max_notional = bal * LEVERAGE * 0.9  # 90% del max
    notional = min(notional, max_notional)
    size = rpx(notional / entry_px, sz_dec)

    if size <= 0:
        log(f"Size zero: notional=${notional:.0f} px={entry_px}")
        return False

    # ── MARGIN CHECK: verifica prima di inviare ──
    if not check_margin(size, entry_px):
        return False

    # Set leverage
    # Set leverage (clamped to max allowed)
    effective_lev = min(LEVERAGE, get_max_leverage())
    try: call(_exchange.update_leverage, effective_lev, COIN, timeout=10)
    except: pass
    time.sleep(0.3)

    # Verify leverage
    pos_check = get_position()
    if pos_check is None:
        # No existing position, leverage should be set
        pass

    # Calcola decimali effettivi — garantisce SL ≠ entry ≠ TP dopo round
    eff_dec = effective_price_dec(entry, sl, tp, px_dec)

    sl_px = rpx(sl, eff_dec)
    tp_px = rpx(tp, eff_dec)
    entry_rounded = rpx(entry, eff_dec)

    # Verifica finale: SL e TP devono essere distinti da entry
    if (is_buy and (sl_px >= entry_rounded or tp_px <= entry_rounded)) or \
       (not is_buy and (sl_px <= entry_rounded or tp_px >= entry_rounded)):
        eff_dec = min(eff_dec + 2, 8)
        sl_px = rpx(sl, eff_dec)
        tp_px = rpx(tp, eff_dec)
        log(f"⚠️ px_dec alzato a {eff_dec} per distinguere SL/TP")

    tp_rr_display = tp_dist / sl_dist if sl_dist > 0 else 0
    log(f"{'🟢' if is_buy else '🔴'} ORDER {COIN} {direction} [{scalp_mode}] @ {entry} size:{size} "
        f"SL:{sl_px}({sl_dist/entry*100:.2f}%) TP:{tp_px}({tp_dist/entry*100:.2f}%) R:R=1:{tp_rr_display:.1f} "
        f"risk:${RISK_USD*size_mult:.1f} notional:${notional:.0f}")

    # ══════════════════════════════════════════════════════════════
    # EXECUTION — mode-adaptive
    # FLASH: IoC diretto (velocità, no GTC)
    # RANGE/TREND: GTC Maker → IoC fallback
    # ══════════════════════════════════════════════════════════════

    filled = False

    # ── STEP 1: GTC MAKER (skip in FLASH mode) ──
    if scalp_mode == "FLASH":
        log(f"⚡ FLASH mode — skip GTC, direct IoC")
    else:
        try:
            fresh_mid = get_mid()
            if fresh_mid <= 0: return False

            drift = (fresh_mid - entry_px) / entry_px
            if (is_buy and drift > DRIFT_MAX_FAVORABLE) or (not is_buy and drift < -DRIFT_MAX_FAVORABLE):
                log(f"Drift {drift:+.3%} — too late"); return False
            if (is_buy and drift < -DRIFT_MAX_ADVERSE) or (not is_buy and drift > DRIFT_MAX_ADVERSE):
                log(f"Drift {drift:+.3%} — invalidated"); return False

            if abs(drift) > 0.001:
                if is_buy:
                    sl_px = rpx(fresh_mid - sl_dist, eff_dec)
                    tp_px = rpx(fresh_mid + tp_dist, eff_dec)
                else:
                    sl_px = rpx(fresh_mid + sl_dist, eff_dec)
                    tp_px = rpx(fresh_mid - tp_dist, eff_dec)

            gtc_px = rpx(fresh_mid * (1 + SLIPPAGE) if is_buy else fresh_mid * (1 - SLIPPAGE), eff_dec)
            res = call(_exchange.order, COIN, is_buy, size, gtc_px,
                       {"limit": {"tif": "Gtc"}}, False, timeout=15)
            log(f"GTC Maker @ {gtc_px}: {res}")

            oid_gtc = None
            if res and res.get("status") == "ok":
                for s in res.get("response",{}).get("data",{}).get("statuses",[]):
                    if "filled" in s:
                        filled = True
                        log(f"✅ GTC instant fill (maker)")
                        break
                    if "resting" in s:
                        oid_gtc = s["resting"]["oid"]

                if not filled and oid_gtc:
                    for tick in range(GTC_TIMEOUT * 2):
                        time.sleep(0.5)
                        p = get_position()
                        if p and p.get("szi", 0) != 0:
                            filled = True
                            log(f"✅ GTC filled in {(tick+1)*0.5:.1f}s (maker)")
                            break
                    if not filled:
                        try: call(_exchange.cancel, COIN, oid_gtc, timeout=10)
                        except: pass
                        log(f"GTC not filled in {GTC_TIMEOUT}s")
        except Exception as e:
            log(f"GTC error: {e}")

    # ── STEP 2: IoC TAKER (fallback) ──
    if not filled:
        try:
            fresh_mid = get_mid()
            if fresh_mid <= 0: return False

            drift2 = (fresh_mid - entry_px) / entry_px
            if abs(drift2) > DRIFT_MAX_FAVORABLE * 1.5:
                log(f"Post-GTC drift {drift2:+.3%} — abort"); return False

            # Slippage dinamico: base SLIPPAGE × 2, scalato con volatilità
            atr_pct = sl_dist / fresh_mid
            dyn_slip = max(SLIPPAGE * 2, min(atr_pct * 0.5, 0.005))

            ioc_px = rpx(fresh_mid * (1 + dyn_slip) if is_buy else fresh_mid * (1 - dyn_slip), px_dec)
            res = call(_exchange.order, COIN, is_buy, size, ioc_px,
                       {"limit": {"tif": "Ioc"}}, False, timeout=15)
            log(f"IoC Taker @ {ioc_px} (slip:{dyn_slip:.2%}): {res}")

            if res and res.get("status") == "ok":
                for s in res.get("response",{}).get("data",{}).get("statuses",[]):
                    if "filled" in s:
                        filled = True
                        avg = float(s["filled"].get("avgPx", 0))
                        real_slip = abs(avg - fresh_mid) / fresh_mid if avg > 0 else 0
                        log(f"✅ IoC filled @ {avg} (slip:{real_slip:.3%})")
                        break
        except Exception as e:
            log(f"IoC error: {e}")

    if not filled:
        log(f"GTC + IoC failed — no fill"); return False

    # ── Conferma posizione ──
    pos = get_position()
    if not pos:
        for _ in range(6):
            time.sleep(1)
            pos = get_position()
            if pos: break
    if not pos:
        log(f"Filled but position not found"); return False

    actual_size = rpx(abs(pos["szi"]), sz_dec)
    actual_entry = pos["entry"]
    actual_lev = pos.get("lev", LEVERAGE)

    # Recalc SL/TP from actual entry if leverage differs
    if actual_lev != LEVERAGE:
        ls = LEVERAGE / max(actual_lev, 1)
        new_sl_dist = sl_dist * ls
        new_sl_dist = max(new_sl_dist, actual_entry * SL_MIN_PCT)
        new_sl_dist = min(new_sl_dist, actual_entry * SL_MAX_PCT)
        if is_buy:
            sl_px = rpx(actual_entry - new_sl_dist, px_dec)
            tp_px = rpx(actual_entry + new_sl_dist * TP_RR, px_dec)
        else:
            sl_px = rpx(actual_entry + new_sl_dist, px_dec)
            tp_px = rpx(actual_entry - new_sl_dist * TP_RR, px_dec)
        log(f"⚠️ Lev {actual_lev}x → SL/TP ricalcolati")

    # Place SL + TP
    try:
        call(_exchange.order, COIN, not is_buy, actual_size, sl_px,
             {"trigger": {"triggerPx": sl_px, "isMarket": True, "tpsl": "sl"}},
             True, timeout=15)
        time.sleep(0.3)
        call(_exchange.order, COIN, not is_buy, actual_size, tp_px,
             {"trigger": {"triggerPx": tp_px, "isMarket": True, "tpsl": "tp"}},
             True, timeout=15)
    except Exception as e:
        log(f"🚨 SL/TP ERROR: {e}")
        tg(f"🚨 BTC SL/TP ERROR — check!")
        return False

    _last_trade_ts = time.time()
    filled_px = actual_entry or entry
    sl_pct_real = abs(filled_px - sl_px) / filled_px * 100
    tp_pct_real = abs(tp_px - filled_px) / filled_px * 100

    log(f"✅ FILLED @ {filled_px} size:{actual_size} lev:{actual_lev}x "
        f"SL:{sl_px}({sl_pct_real:.2f}%) TP:{tp_px}({tp_pct_real:.2f}%)")
    tg(f"{'🟢' if is_buy else '🔴'} <b>BTC</b> {direction} @ {filled_px}\n"
       f"SL:{sl_px} ({sl_pct_real:.1f}%) | TP:{tp_px} ({tp_pct_real:.1f}%)\n"
       f"Size:{actual_size} | Lev:{actual_lev}x | Risk:${RISK_USD*size_mult:.1f}")
    return True

# ================================================================
# MAIN LOOP
# ================================================================
def main():
    global _last_trade_ts
    log(f"🚀 BTC SCALPER V6")
    log(f"Risk:${RISK_USD} Lev:{LEVERAGE}x Modes:RANGE/TREND/FLASH")

    load_state()

    sz_dec, px_dec = get_meta()
    log(f"Meta: szDec={sz_dec} pxDec={px_dec}")

    # Run backtest all'avvio
    run_backtest()

    # Recovery: load pos_state da Redis o detect posizione orfana
    last_pos_state = load_pos_state()
    if last_pos_state:
        pos_check = get_position()
        if pos_check:
            log(f"🔍 Restored pos_state from Redis: {last_pos_state.get('type','?')} "
                f"entry:{last_pos_state.get('entry',0)} trailing:{last_pos_state.get('trailing_active',False)}")
        else:
            log(f"🔍 Redis had pos_state but no position — clearing")
            last_pos_state = None
            save_pos_state(None)
    else:
        last_pos_state = recover_position(sz_dec, px_dec)

    # Cleanup ordini orfani
    cleanup_orphan_orders()

    cycle = 0

    while True:
        try:
            cycle += 1
            pos = get_position()
            mid = get_mid()
            regime = update_regime()

            # ── Check pending orders ──
            if _pending_order:
                pend_result = check_pending(sz_dec, px_dec)
                if pend_result:
                    # Pending fillato → setup pos_state
                    last_pos_state = {
                        "szi": pend_result["szi"], "entry": pend_result["entry"],
                        "type": pend_result.get("type", ""), "regime": pend_result.get("regime", regime),
                        "sl_dist": pend_result.get("sl_dist", 0),
                        "sl_original": pend_result.get("sl", 0),
                        "tp_original": pend_result.get("tp", 0),
                        "last_mgmt": 0, "partial_done": False, "partial_close_px": 0,
                        "partial_pnl_pct": 0, "trailing_active": False,
                        "trailing_activated_at": 0, "trailing_moves": 0,
                        "current_ts": 0, "atr": 0, "open_ts": time.time(),
                        "close_reason": "", "setup_score": 0, "ai_confidence": 7,
                    }
                    save_pos_state(last_pos_state)
                time.sleep(SCAN_INTERVAL)
                continue

            # ── Detect trade close ──
            if last_pos_state is not None and pos is None:
                entry = last_pos_state["entry"]
                szi = last_pos_state["szi"]
                d = "LONG" if szi > 0 else "SHORT"
                size_abs = abs(szi)

                # Real exit price from fills API (precise)
                open_ts = last_pos_state.get("open_ts", time.time() - 3600)
                real_exit, real_pnl = compute_real_exit(d, entry, open_ts)

                if real_exit > 0 and real_pnl != 0:
                    exit_px = real_exit
                    pnl_pct = real_pnl * 100
                    pnl_usd = real_pnl * size_abs * entry
                else:
                    # Fallback: mid price
                    exit_px = mid
                    pnl_pct = ((mid - entry)/entry if d == "LONG" else (entry - mid)/entry) * 100
                    pnl_usd = pnl_pct/100 * size_abs * entry

                # Determine close reason
                close_reason = last_pos_state.get("close_reason", "")
                if not close_reason:
                    # Deduce from PnL and SL/TP distances
                    sl_dist = last_pos_state.get("sl_dist", 0)
                    if sl_dist > 0:
                        tp_dist = sl_dist * TP_RR
                        if d == "LONG":
                            hit_sl = mid <= entry - sl_dist * 0.9
                            hit_tp = mid >= entry + tp_dist * 0.9
                        else:
                            hit_sl = mid >= entry + sl_dist * 0.9
                            hit_tp = mid <= entry - tp_dist * 0.9

                        if hit_tp:
                            close_reason = "🎯 Take Profit"
                        elif hit_sl:
                            close_reason = "🛑 Stop Loss"
                        elif last_pos_state.get("trailing_active"):
                            close_reason = "📈 Trailing Stop"
                        elif last_pos_state.get("partial_done"):
                            close_reason = "💰 Partial + Trail"
                        else:
                            close_reason = "❓ Unknown"
                    else:
                        close_reason = "❓ Unknown"

                emoji = "✅" if pnl_usd > 0 else "❌"
                sig_type = last_pos_state.get("type", "?")
                trade_regime = last_pos_state.get("regime", "?")

                # Durata trade
                open_ts = last_pos_state.get("open_ts", time.time())
                duration_s = time.time() - open_ts
                if duration_s < 60:
                    dur_str = f"{duration_s:.0f}s"
                elif duration_s < 3600:
                    dur_str = f"{duration_s/60:.0f}min"
                else:
                    dur_str = f"{duration_s/3600:.1f}h"

                # Daily stats
                now = time.time()
                daily_trades = [t for t in _trades_today if now - t.get("ts_close", t.get("ts", 0)) < 86400]
                daily_pnl = sum(t["pnl"] for t in daily_trades) + pnl_usd
                daily_n = len(daily_trades) + 1
                daily_wins = sum(1 for t in daily_trades if t["pnl"] > 0) + (1 if pnl_usd > 0 else 0)
                daily_wr = daily_wins / daily_n * 100 if daily_n > 0 else 0

                # Save completo con tutti i dettagli
                save_trade(pnl_usd, d, entry, exit_px,
                           sig_type=sig_type,
                           sl_dist=last_pos_state.get("sl_dist", 0),
                           regime=trade_regime,
                           extra={
                               "pnl_pct":              pnl_pct,
                               "sl_original":           last_pos_state.get("sl_original", 0),
                               "tp_original":           last_pos_state.get("tp_original", 0),
                               "trailing_activated_at":  last_pos_state.get("trailing_activated_at", 0),
                               "trailing_final_sl":      last_pos_state.get("current_ts", 0),
                               "trailing_moves":         last_pos_state.get("trailing_moves", 0),
                               "partial_close_px":       last_pos_state.get("partial_close_px", 0),
                               "partial_pnl_pct":        last_pos_state.get("partial_pnl_pct", 0),
                               "ai_confidence":          last_pos_state.get("ai_confidence", 0),
                               "setup_score":            last_pos_state.get("setup_score", 0),
                               "close_reason":           close_reason,
                               "duration_s":             duration_s,
                               "ts_open":                open_ts,
                           })

                # Clear pos_state da Redis
                save_pos_state(None)

                # Console log dettagliato
                trail_info = ""
                if last_pos_state.get("trailing_active"):
                    trail_info = (f" | Trail: activated@{last_pos_state.get('trailing_activated_at',0):.1f}"
                                  f" final_SL:{last_pos_state.get('current_ts',0):.1f}"
                                  f" moves:{last_pos_state.get('trailing_moves',0)}")
                partial_info = ""
                if last_pos_state.get("partial_done"):
                    partial_info = f" | Partial@{last_pos_state.get('partial_close_px',0):.1f}"

                log(f"{emoji} CLOSED {d} {sig_type} | {close_reason}")
                log(f"   Entry: {entry:.1f} → Exit: {exit_px:.1f} | PnL: ${pnl_usd:+.2f} ({pnl_pct:+.2f}%)")
                log(f"   Duration: {dur_str} | Regime: {trade_regime}{trail_info}{partial_info}")
                log(f"   Day: {daily_n} trades ${daily_pnl:+.2f} WR:{daily_wr:.0f}%")

                # Telegram dettagliato
                tg_trail = ""
                if last_pos_state.get("trailing_active"):
                    tg_trail = (f"\n📈 Trail: {last_pos_state.get('trailing_moves',0)} moves, "
                                f"SL {last_pos_state.get('current_ts',0):.1f}")
                tg_partial = ""
                if last_pos_state.get("partial_done"):
                    tg_partial = f"\n💰 Partial: 50% @ {last_pos_state.get('partial_close_px',0):.1f}"

                tg_msg = (
                    f"{emoji} <b>BTC {d} CLOSED</b>\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n"
                    f"📊 <b>PnL: ${pnl_usd:+.2f} ({pnl_pct:+.2f}%)</b>\n"
                    f"🏷 Reason: {close_reason}\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n"
                    f"📍 Entry: {entry:.1f} → Exit: {exit_px:.1f}\n"
                    f"🛡 SL: {last_pos_state.get('sl_original',0):.1f} | TP: {last_pos_state.get('tp_original',0):.1f}"
                    f"{tg_trail}{tg_partial}\n"
                    f"📐 Type: {sig_type} | Regime: {trade_regime}\n"
                    f"⏱ Duration: {dur_str}\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n"
                    f"📅 Today: {daily_n} trades | WR: {daily_wr:.0f}%\n"
                    f"💰 Daily PnL: <b>${daily_pnl:+.2f}</b>"
                )
                tg(tg_msg)

            last_pos_state = pos

            # ── Status log ──
            # ── Status log ──
            if cycle % 10 == 1:  # ogni 10 × 30s = 5 min
                bal = get_balance()
                daily_pnl = sum(t["pnl"] for t in _trades_today if time.time()-t["ts"]<86400)
                n_today = len([t for t in _trades_today if time.time()-t["ts"]<86400])
                pos_str = f"{'LONG' if pos['szi']>0 else 'SHORT'} @ {pos['entry']}" if pos else "flat"
                log(f"#{cycle} ${bal:.2f} | {regime} | {pos_str} | "
                    f"today: {n_today} trades ${daily_pnl:+.2f} | BTC ${mid:,.0f}")

            # ── Refresh backtest every hour ──
            if cycle % 120 == 0:  # 120 × 30s = 1h
                run_backtest()

            # ── Cleanup ordini orfani ogni 10 min ──
            if cycle % 20 == 0 and pos is None:  # 20 × 30s = 10min
                cleanup_orphan_orders()

            # ── Daily report ──
            maybe_send_daily_report()

            # ── TRADE MANAGEMENT (quando in posizione) ──
            if pos is not None:
                entry = pos["entry"]
                szi = pos["szi"]
                d = "LONG" if szi > 0 else "SHORT"
                pnl_pct = ((mid - entry)/entry if d == "LONG" else (entry - mid)/entry) * 100
                atr_now = last_pos_state.get("atr", 0) if last_pos_state else 0

                # Init pos_state se necessario (recovery)
                if last_pos_state is None:
                    last_pos_state = pos.copy()
                    last_pos_state.update({"type":"?","regime":regime,"sl_dist":entry*SL_MAX_PCT,
                                           "last_mgmt":0,"partial_done":False,"trailing_active":False,"atr":0,"current_ts":0})

                # Fetch ATR fresco
                if cycle % 4 == 0:  # ogni 2min
                    df_mgmt = fetch_df("5m", 1)
                    if df_mgmt is not None and len(df_mgmt) >= 5:
                        atr_now = float(df_mgmt.iloc[-1]['atr'])
                        last_pos_state["atr"] = atr_now

                # ── Trailing stop meccanico (solo TREND mode) ──
                trade_mode = last_pos_state.get("scalp_mode", "TREND")
                if atr_now > 0 and trade_mode == "TREND":
                    update_trailing(last_pos_state, mid, atr_now, sz_dec, px_dec)

                # ── Partial close (solo TREND mode) ──
                if trade_mode == "TREND":
                    check_partial_close(last_pos_state, mid, sz_dec, px_dec)

                # ── AI management ogni 2 minuti ──
                mgmt_interval = 120
                last_mgmt = last_pos_state.get("last_mgmt", 0)

                if time.time() - last_mgmt >= mgmt_interval:
                    last_pos_state["last_mgmt"] = time.time()

                    # Fetch 5m fresco per contesto
                    df_mgmt = fetch_df("5m", 1)
                    mgmt_ctx = ""
                    if df_mgmt is not None and len(df_mgmt) >= 5:
                        rm = df_mgmt.iloc[-1]
                        atr_now = float(rm['atr'])
                        last_pos_state["atr"] = atr_now
                        mgmt_ctx = (f"RSI:{rm['rsi']:.0f} MACD:{rm['macd_hist']:.1f} "
                                   f"slope:{rm['ema_slope']:.4f} vol:{rm['vol_rel']:.1f}x")

                    try:
                        api_key = os.getenv("ANTHROPIC_API_KEY", "")
                        if api_key and mgmt_ctx:
                            prompt = (
                                f"BTC {d} position. Entry:{entry:.1f} Current:{mid:.1f} PnL:{pnl_pct:+.1f}%\n"
                                f"5m indicators: {mgmt_ctx}\n"
                                f"Regime: {regime}\n"
                                f"Decide: HOLD (keep SL/TP), TIGHTEN (move SL to breakeven+0.1% if profitable), "
                                f"or EXIT (close now if momentum reversed).\n"
                                f"JSON only: {{\"action\":\"HOLD|TIGHTEN|EXIT\",\"reason\":\"<5 words>\"}}"
                            )
                            resp = requests.post("https://api.anthropic.com/v1/messages",
                                headers={"Content-Type": "application/json",
                                         "x-api-key": api_key,
                                         "anthropic-version": "2023-06-01"},
                                json={"model": "claude-haiku-4-5-20251001", "max_tokens": 100,
                                      "messages": [{"role": "user", "content": prompt}]}, timeout=10)

                            if resp.status_code == 200:
                                txt = resp.json()["content"][0]["text"]
                                start = txt.find("{"); end = txt.rfind("}")+1
                                if start >= 0 and end > start:
                                    mj = json.loads(txt[start:end])
                                    action = mj.get("action", "HOLD").upper()
                                    reason = mj.get("reason", "")

                                    if action == "TIGHTEN" and pnl_pct > 0.1:
                                        # Move SL to breakeven + 0.1%
                                        new_sl = entry * (1.001 if d == "LONG" else 0.999)
                                        new_sl = rpx(new_sl, px_dec)
                                        try:
                                            # Cancel old SL, place new
                                            opens = call(_info.open_orders, _account.address, timeout=10)
                                            for o in opens:
                                                if o.get("coin") == COIN and o.get("orderType") == "Stop Market":
                                                    call(_exchange.cancel, COIN, o["oid"], timeout=10)
                                            time.sleep(0.3)
                                            size_abs = rpx(abs(szi), sz_dec)
                                            is_buy = d == "LONG"
                                            call(_exchange.order, COIN, not is_buy, size_abs, new_sl,
                                                 {"trigger": {"triggerPx": new_sl, "isMarket": True, "tpsl": "sl"}},
                                                 True, timeout=15)
                                            log(f"🔒 TIGHTEN SL → {new_sl} (BE+0.1%) | {reason}")
                                            tg(f"🔒 BTC SL → breakeven {new_sl} | {reason}", silent=True)
                                        except Exception as e:
                                            log(f"Tighten error: {e}")

                                    elif action == "EXIT":
                                        # Close position at market
                                        last_pos_state["close_reason"] = f"🤖 AI Exit: {reason}"
                                        try:
                                            is_buy = d == "LONG"
                                            size_abs = rpx(abs(szi), sz_dec)
                                            close_px = rpx(mid * (0.997 if is_buy else 1.003), px_dec)
                                            call(_exchange.order, COIN, not is_buy, size_abs, close_px,
                                                 {"limit": {"tif": "Ioc"}}, False, timeout=15)
                                            log(f"🚪 AI EXIT @ {mid:.1f} PnL:{pnl_pct:+.1f}% | {reason}")
                                            tg(f"🚪 BTC AI EXIT PnL:{pnl_pct:+.1f}% | {reason}")
                                            # Cancel remaining orders
                                            time.sleep(1)
                                            try:
                                                opens = call(_info.open_orders, _account.address, timeout=10)
                                                for o in opens:
                                                    if o.get("coin") == COIN:
                                                        call(_exchange.cancel, COIN, o["oid"], timeout=10)
                                            except: pass
                                        except Exception as e:
                                            log(f"Exit error: {e}")

                                    else:
                                        if cycle % 8 == 0:
                                            log(f"📊 HOLD PnL:{pnl_pct:+.1f}% | {reason}")

                    except Exception as e:
                        log(f"AI mgmt error: {e}")

                time.sleep(SCAN_INTERVAL)
                continue

            # ── Circuit breaker ──
            cb, reason = check_circuit_breaker()
            if cb:
                if cycle % 20 == 1:
                    log(f"🛑 {reason}")
                time.sleep(SCAN_INTERVAL)
                continue

            # ── Cooldown ──
            if time.time() - _last_trade_ts < COOLDOWN_SEC:
                time.sleep(SCAN_INTERVAL)
                continue

            # ── Trade lock: skip se un ordine è in esecuzione ──
            if _is_trading:
                time.sleep(SCAN_INTERVAL)
                continue

            # ── Check signal ──
            sig = check_signal()
            if sig is None:
                time.sleep(SCAN_INTERVAL)
                continue

            direction, sig_type, sl, tp, entry_px, atr, details, sl_dist, size_mult, sig_regime, setup, scalp_mode = sig
            log(f"📡 SIGNAL: {direction} {sig_type} [{scalp_mode}] | {regime} | {details}")

            # ── Funding check ──
            if not is_funding_ok(direction):
                time.sleep(SCAN_INTERVAL)
                continue

            # ── Execute ──
            success = open_trade(direction, sl, tp, entry_px, sl_dist, sz_dec, px_dec, size_mult, scalp_mode)
            if success:
                p = get_position()
                if p:
                    last_pos_state = p
                    last_pos_state["type"] = sig_type
                    last_pos_state["sl_dist"] = sl_dist
                    last_pos_state["sl_original"] = sl
                    last_pos_state["tp_original"] = tp
                    last_pos_state["regime"] = sig_regime
                    last_pos_state["scalp_mode"] = scalp_mode
                    last_pos_state["setup_score"] = setup
                    last_pos_state["ai_confidence"] = ai_confidence if 'ai_confidence' in dir() else 7
                    last_pos_state["last_mgmt"] = 0
                    last_pos_state["partial_done"] = False
                    last_pos_state["partial_close_px"] = 0
                    last_pos_state["partial_pnl_pct"] = 0
                    last_pos_state["trailing_active"] = False
                    last_pos_state["trailing_activated_at"] = 0
                    last_pos_state["trailing_moves"] = 0
                    last_pos_state["current_ts"] = 0
                    last_pos_state["atr"] = 0
                    last_pos_state["open_ts"] = time.time()
                    last_pos_state["close_reason"] = ""
                    save_pos_state(last_pos_state)
            else:
                log(f"Trade failed — cooldown {COOLDOWN_SEC}s")
                _last_trade_ts = time.time()  # prevent spam

        except KeyboardInterrupt:
            log("🛑 Stopped"); break
        except Exception as e:
            log(f"Loop error: {e}")

        # Heartbeat OGNI ciclo — Railway killa senza output
        print(f".", end="", flush=True)
        if cycle % 10 == 0:
            print(f" [cycle {cycle}]", flush=True)  # newline ogni 5min

        time.sleep(SCAN_INTERVAL)

if __name__ == "__main__":
    main()
