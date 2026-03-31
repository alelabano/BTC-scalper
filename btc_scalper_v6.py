"""
unified_bot.py — BTC Scalper V7 + Altcoin Processor in un singolo container.

ARCHITETTURA (4 thread):
  Thread A — BTC_SCANNER:  regime 4h + backtest + V7 analytics ogni 5 min
  Thread B — ALT_SCANNER:  filtra 229 coin → top 25 ogni 20 min
  Thread C — PROCESSOR:    BTC signal check ogni 10s + alt processor ogni 5 min
  Thread D — EXECUTOR:     esegue trade BTC + ALT, gestisce posizioni

V7 MODULES (shared):
  ✦ Sentiment Analysis — FGI + CoinGecko + Reddit + CryptoCompare
  ✦ Order Flow — CVD, OB delta, OI momentum, delta divergence
  ✦ Machine Learning — Online SGD classifier, 16 features

BTC MODES: RANGE (ADX<20) | TREND (ADX>25) | FLASH (vol>3x)
ALT MODES: Mean Reversion + Trend Following + Scalping

Fleet system: INTERNO (no più Redis cross-worker).
BTC engine genera regime/bias → ALT engine lo legge in-memory.

Variabili d'ambiente:
  PRIVATE_KEY, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
  UPSTASH_REDIS_REST_URL, UPSTASH_REDIS_REST_TOKEN
  ANTHROPIC_API_KEY, CRYPTOCOMPARE_API_KEY
  ACCOUNT_CAPITAL_USD (default 1000)
  COINGECKO_API_KEY (opzionale)
"""

import os, sys, time, json, hashlib, threading
import pandas as pd
import numpy as np
import requests
from collections import deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from datetime import datetime, timezone
from eth_account import Account
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants

# ================================================================
# CONFIGURAZIONE GLOBALE
# ================================================================
PRIVATE_KEY = os.getenv("PRIVATE_KEY")
TG_TOKEN    = os.getenv("TELEGRAM_TOKEN", "")
TG_CHAT_ID  = os.getenv("TELEGRAM_CHAT_ID", "")
REDIS_URL   = os.getenv("UPSTASH_REDIS_REST_URL", "").rstrip("/")
REDIS_TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN", "")

if not PRIVATE_KEY:
    print("❌ PRIVATE_KEY mancante"); sys.exit(1)

# ── BTC Scalper Config ───────────────────────────────────────────
BTC_COIN = "BTC"
BTC_LEVERAGE = 5
BTC_RISK_USD = 2.0
BTC_MAX_POSITIONS = 2
BTC_COOLDOWN_SEC = 180
BTC_COOLDOWN_AFTER_LOSS = 300  # 5 min dopo un loss
MAX_TRADES_PER_HOUR = 2
BTC_SCAN_INTERVAL = 5
BTC_SIGNAL_MAX_AGE = 30
BTC_REGIME_INTERVAL = 300

# ROE-based SL/TP — SL 
ROE_TP = 0.10; ROE_SL = 0.05  
TP_PRICE_PCT = ROE_TP / BTC_LEVERAGE  # 0.003 = 0.3%
SL_PRICE_PCT = ROE_SL / BTC_LEVERAGE  # 0.003 = 0.3%

RANGE_SL_ATR = 1.2; RANGE_TP_PCT = TP_PRICE_PCT
RANGE_SL_MIN = 0.003; RANGE_SL_MAX = 0.006  # 0.3% - 0.6%
TREND_SL_ATR = 1.2; TREND_TP_RR = 1.5  # R:R 1:1.5
TREND_SL_MIN = 0.003; TREND_SL_MAX = 0.006
TREND_TRAIL_ATR = 0.7; TREND_PARTIAL = 0.4
FLASH_SL_ATR = 1.0; FLASH_TP_PCT = TP_PRICE_PCT
FLASH_SL_MIN = 0.002; FLASH_SL_MAX = 0.004  # 0.2% - 0.4% (flash = più stretto)
FLASH_TRAILING = False; FLASH_USE_IOC = True

SL_ATR_MULT = TREND_SL_ATR; TP_RR = TREND_TP_RR
SL_MIN_PCT = TREND_SL_MIN; SL_MAX_PCT = TREND_SL_MAX
TRAILING_ACTIVATE = 0.3; TRAILING_ATR = TREND_TRAIL_ATR
PARTIAL_CLOSE_PCT = 0.4
TRAILING_STOP_INTERVAL = 15  # check trailing ogni 20s
FUNDING_BLOCK_THRESH = 0.0003
SLIPPAGE = 0.001; GTC_TIMEOUT = 6
DRIFT_MAX_FAVORABLE = 0.004; DRIFT_MAX_ADVERSE = 0.008
PENDING_ORDER_TTL = 120
MAX_DAILY_LOSS = 20.0; MAX_CONSEC_LOSS = 2
DAILY_REPORT_HOUR = 0

# ── Altcoin Processor Config ────────────────────────────────────
TIMEFRAME_TREND = "1h"; TIMEFRAME_SETUP = "15m"; TIMEFRAME_ENTRY = "5m"
LOOKBACK_DAYS_TREND = 60; LOOKBACK_DAYS_SETUP = 14; LOOKBACK_DAYS_ENTRY = 5
MIN_CANDLES_TREND = 200; MIN_CANDLES_SETUP = 200; MIN_CANDLES_ENTRY = 200
MIN_VOLUME_24H_USD = 200_000; MIN_VOLUME_TREND_MULT = 1.8
FORWARD_WINDOW = 36
MIN_PRECISION = 0.35; MIN_PRECISION_FLOOR = 0.28
MIN_PROFIT_FACTOR = 1.2; MIN_PROFIT_FACTOR_FLOOR = 1.0
MIN_BACKTEST_TRADES = 20; VOLATILITY_MIN = 0.0015
DEFAULT_CAPITAL = float(os.getenv("ACCOUNT_CAPITAL_USD", "1000"))
API_TIMEOUT_SEC = 30; PROCESSOR_INTERVAL = 2 * 60
ALT_FUNDING_HISTORY_LEN = 42

# ── Unified Executor Config ─────────────────────────────────────
ALT_MAX_CONCURRENT = 2         # max altcoin positions
ALT_TRADE_SIZE_USD = 2.5  # overridden by balance % in execute
ALT_LEVERAGE = 5
ALT_CHECK_INTERVAL = 15
ALT_SIGNAL_MAX_AGE = 3 * 60
META_REFRESH_CYCLES = 20
ENTRY_POLL_ATTEMPTS = 8; ENTRY_POLL_INTERVAL = 0.5
COIN_COOLDOWN_MR = 3600; COIN_COOLDOWN_TREND = 14400; COIN_COOLDOWN = 3600
SETUP_SCORE_MIN = 20
TRAILING_STOP_INTERVAL = 5 * 60
SCANNER_INTERVAL = 10 * 60; SCANNER_MAX_UNIVERSE = 229
PROCESSOR_MAX_COINS = 30
CORRELATION_THRESHOLD = 0.55
# ── Missing aliases (ALT engine compatibility) ───────────────────
TRADE_SIZE_USD = ALT_TRADE_SIZE_USD
LEVERAGE = ALT_LEVERAGE
CHECK_INTERVAL = ALT_CHECK_INTERVAL
MAX_CONCURRENT_TRADES = ALT_MAX_CONCURRENT
FUNDING_HISTORY_LEN = ALT_FUNDING_HISTORY_LEN
MAX_DAILY_LOSS_PCT = 15.0
MAX_CONSECUTIVE_LOSSES = 4

def log_err(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️  {msg}", flush=True)
def log_p(msg): log("ALT-P", msg)
def log_e(msg): log("ALT-E", msg)
def round_to_decimals(val, decimals):
    if decimals == 0: return int(round(float(val)))
    try: return float(f"{float(val):.{int(decimals)}f}")
    except: return round(float(val), 2)


# Order Flow globals
_last_oi = 0.0

# ── Correlation & Categories ────────────────────────────────────
_correlation_matrix = {}
_correlation_lock = threading.Lock()
COIN_CATEGORIES = {
    "highcap": {"BTC", "ETH"},
    "midcap":  {"SOL", "AVAX", "LINK", "DOT", "ADA", "NEAR", "ARB", "OP"},
    "meme":    {"PEPE", "BONK", "WIF", "DOGE", "SHIB", "FARTCOIN", "FLOKI", "TURBO", "NEIRO", "POPCAT"},
}

def get_coin_category(coin):
    for cat, coins in COIN_CATEGORIES.items():
        if coin in coins: return cat
    return "other"

def are_coins_correlated(coin_a, coin_b):
    with _correlation_lock:
        key = tuple(sorted([coin_a, coin_b]))
        return _correlation_matrix.get(key, 0) > CORRELATION_THRESHOLD

def is_correlated_with_active(coin, active_coins):
    for ac in active_coins:
        if ac == coin: continue
        if are_coins_correlated(coin, ac):
            return True, ac
    return False, ""

# ================================================================
# HYPERLIQUID CLIENTS (shared)
# ================================================================
account = Account.from_key(PRIVATE_KEY)
_account = account  # alias for BTC engine compatibility
_info = Info(constants.MAINNET_API_URL, skip_ws=True)
_exchange = Exchange(account, constants.MAINNET_API_URL)
_pool = ThreadPoolExecutor(max_workers=5)

# ================================================================
# UTILITIES (shared)
# ================================================================
def log(prefix, msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [{prefix}] {msg}", flush=True)

def log_btc(msg):  log("BTC", msg)
def log_alt(msg):  log("ALT", msg)
def log_exec(msg): log("EXEC", msg)

def tg(msg, silent=False):
    if not TG_TOKEN: return
    try:
        requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json={"chat_id": TG_CHAT_ID, "text": msg, "parse_mode": "HTML",
                  "disable_notification": silent}, timeout=5)
    except: pass

_api_call_times = []
_api_lock = threading.Lock()
API_MAX_CALLS_PER_MIN = 40  # Aggressive limit — stop 429s

def call(fn, *a, timeout=20, label='', **kw):
    with _api_lock:
        now = time.time()
        _api_call_times[:] = [t for t in _api_call_times if now - t < 60]
        if len(_api_call_times) >= API_MAX_CALLS_PER_MIN:
            wait = 61 - (now - _api_call_times[0])
            if wait > 0:
                time.sleep(wait)  # full wait, no cap
        _api_call_times.append(time.time())
    f = _pool.submit(fn, *a, **kw)
    try: return f.result(timeout=timeout)
    except FuturesTimeout:
        raise TimeoutError(f"Timeout {label or fn.__name__}")

def round_px(value, precision):
    if precision == 0: return int(round(float(value)))
    try: return float(f"{float(value):.{int(precision)}f}")
    except: return round(float(value), 2)

rpx = round_px  # alias

def min_decimals_for_price(px):
    if px <= 0: return 6
    for d in range(0, 10):
        rounded = round(px, d)
        if rounded > 0 and abs(rounded - px) / px < 0.005: return d
    return 6

def effective_price_dec(px, sl, tp, base_px_dec):
    dec = max(base_px_dec, min_decimals_for_price(px))
    e = rpx(px, dec); s = rpx(sl, dec); t = rpx(tp, dec)
    if s == e or t == e: dec = min(dec + 2, 8)
    return dec

# ================================================================
# REDIS (shared)
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

def _rdel(key):
    if not REDIS_URL: return
    try:
        requests.get(f"{REDIS_URL}/del/{key}",
            headers={"Authorization": f"Bearer {REDIS_TOKEN}"}, timeout=5)
    except: pass

# ================================================================
# FLEET SYSTEM — INTERNO (BTC → ALT in-memory)
# ================================================================
# BTC engine scrive, ALT engine legge. No più Redis cross-worker.
_fleet = {
    "bias": "NEUTRAL", "bias_reason": "", "bias_ts": 0,
    "btc_regime": "RANGE", "regime_ts": 0,
    "btc_pos": None, "pos_ts": 0,
    "kill_switch": False, "kill_reason": "",
}
_fleet_lock = threading.Lock()

def fleet_set_bias(bias, reason=""):
    with _fleet_lock:
        _fleet["bias"] = bias
        _fleet["bias_reason"] = reason
        _fleet["bias_ts"] = time.time()
    # Also publish to Redis for external monitoring
    _rset("fleet:bias", {"bias": bias, "reason": reason, "ts": time.time()})

def fleet_set_btc_regime(regime):
    with _fleet_lock:
        _fleet["btc_regime"] = regime
        _fleet["regime_ts"] = time.time()
    _rset("fleet:btc_regime", {"regime": regime, "ts": time.time()})

def fleet_set_btc_position(pos):
    with _fleet_lock:
        _fleet["btc_pos"] = pos
        _fleet["pos_ts"] = time.time()

def fleet_get_bias():
    with _fleet_lock:
        if time.time() - _fleet["bias_ts"] < 7200:
            return _fleet["bias"], _fleet["bias_reason"]
    return "NEUTRAL", "no bias data"

def fleet_get_btc_regime():
    with _fleet_lock:
        if time.time() - _fleet["regime_ts"] < 600:
            return _fleet["btc_regime"]
    return "RANGE"

def fleet_get_btc_position():
    with _fleet_lock:
        pos = _fleet["btc_pos"]
        if pos and time.time() - _fleet["pos_ts"] < 300:
            return pos.get("direction"), pos.get("size", 0)
    return None, 0

def fleet_check_kill_switch():
    with _fleet_lock:
        return _fleet["kill_switch"], _fleet["kill_reason"]

def fleet_set_kill_switch(active, reason=""):
    with _fleet_lock:
        _fleet["kill_switch"] = active
        _fleet["kill_reason"] = reason
    if active:
        _rset("fleet:kill_switch", {"active": True, "reason": reason, "ts": time.time()})
        log_btc("FLEET", f"🚨 KILL SWITCH: {reason}")
        tg(f"🚨 <b>KILL SWITCH</b>: {reason}")

# ================================================================
# MODULE 1: SENTIMENT ANALYSIS (V7.2 — Free APIs)
# ================================================================
# 4 fonti con pesi calibrati per BTC scalping:
#   A. Fear & Greed Index          (30%) — macro sentiment, contrarian
#   B. CoinGecko + Reddit Social   (30%) — social real-time (free)
#   C. Funding z-score interno     (25%) — posizionamento derivati
#   D. CryptoCompare IntoTheBlock  (15%) — on-chain bull/bear signals
#
# ENV VARS necessarie:
#   CRYPTOCOMPARE_API_KEY  — free tier (~100k calls/mese)
#   COINGECKO_API_KEY      — opzionale (free v3 funziona senza key)
#
# Tutte le API sono gratuite. Zero costi operativi.
# ================================================================

_sentiment_cache = {"score": 50, "components": {}, "ts": 0}
SENTIMENT_CACHE_TTL = 300  # 5 min cache — ~288 calls/day per API

# Sub-caches per API con TTL indipendenti (evita burst se una è lenta)
_social_cache = {"ts": 0, "score": 50, "components": {}}
_cryptocompare_cache = {"ts": 0, "bull_pct": 50, "signals": {}}
SOCIAL_CACHE_TTL = 300      # 5 min
CRYPTOCOMPARE_CACHE_TTL = 300

# ── Keyword sentiment scoring per Reddit titles ──
_BULLISH_WORDS = {
    "bull", "bullish", "moon", "pump", "rally", "breakout", "surge", "ath",
    "buy", "buying", "long", "green", "recovery", "bounce", "accumulate",
    "institutional", "adoption", "etf", "halving", "uptrend", "support"
}
_BEARISH_WORDS = {
    "bear", "bearish", "dump", "crash", "sell", "selling", "short", "red",
    "fear", "panic", "liquidat", "capitulat", "reject", "resistance",
    "bubble", "scam", "fraud", "ban", "regulation", "downtrend", "broke"
}


def _score_title(title):
    """
    Score un titolo Reddit: +1 per word bullish, -1 per bearish.
    Returns: float in [-1, +1] normalizzato.
    """
    words = set(title.lower().split())
    bull = sum(1 for w in words if any(bw in w for bw in _BULLISH_WORDS))
    bear = sum(1 for w in words if any(bw in w for bw in _BEARISH_WORDS))
    total = bull + bear
    if total == 0:
        return 0.0
    return (bull - bear) / total  # [-1, +1]


def _fetch_social_sentiment():
    """
    Sostituto gratuito di LunarCrush — combina:
    
    A. CoinGecko /coins/bitcoin — sentiment_votes + community_data
       - sentiment_votes_up_percentage (0-100)
       - community_data: reddit_subscribers, reddit_posts_48h, reddit_comments_48h
       - Free Demo: 30 calls/min, 10k/mese (no key needed per v3 public)
    
    B. Reddit r/bitcoin/hot.json — keyword sentiment sui titoli top 25 post
       - No auth needed (public JSON endpoint)
       - Rate limit: ~10 req/min senza auth
       - Analisi keyword: bullish/bearish word scoring
    
    Returns: aggiorna _social_cache con score composito e componenti.
    """
    global _social_cache
    if time.time() - _social_cache["ts"] < SOCIAL_CACHE_TTL:
        return _social_cache

    components = {}
    scores = []

    # ── A. CoinGecko — sentiment votes + community ──
    cg_score = 50  # default
    try:
        # v3 public API: no key needed, 30 calls/min
        url = ("https://api.coingecko.com/api/v3/coins/bitcoin"
               "?localization=false&tickers=false&market_data=false"
               "&community_data=true&developer_data=false&sparkline=false")
        headers = {"accept": "application/json"}
        # Se hai una key CoinGecko (opzionale), usala
        cg_key = os.getenv("COINGECKO_API_KEY", "")
        if cg_key:
            # Pro API usa un diverso base URL
            url = ("https://pro-api.coingecko.com/api/v3/coins/bitcoin"
                   "?localization=false&tickers=false&market_data=false"
                   "&community_data=true&developer_data=false&sparkline=false")
            headers["x-cg-pro-api-key"] = cg_key

        r = requests.get(url, headers=headers, timeout=10)

        if r.status_code == 200:
            data = r.json()

            # Sentiment votes (CoinGecko user voting)
            sent_up = float(data.get("sentiment_votes_up_percentage", 50) or 50)
            sent_down = float(data.get("sentiment_votes_down_percentage", 50) or 50)
            components["cg_sent_up"] = round(sent_up, 1)
            components["cg_sent_down"] = round(sent_down, 1)

            # Community data
            community = data.get("community_data", {})
            reddit_subs = int(community.get("reddit_subscribers", 0) or 0)
            reddit_posts = float(community.get("reddit_average_posts_48h", 0) or 0)
            reddit_comments = float(community.get("reddit_average_comments_48h", 0) or 0)
            reddit_active = int(community.get("reddit_accounts_active_48h", 0) or 0)

            components["cg_reddit_subs"] = reddit_subs
            components["cg_reddit_posts_48h"] = round(reddit_posts, 1)
            components["cg_reddit_comments_48h"] = round(reddit_comments, 1)
            components["cg_reddit_active_48h"] = reddit_active

            # Score: sentiment_up% è già un buon indicatore (50 = neutral)
            cg_score = sent_up  # 0-100, 50 = neutral
            scores.append(("coingecko", cg_score))
            log_btc(f"[SENT] CoinGecko OK: SentUp:{sent_up:.0f}% Posts48h:{reddit_posts:.0f} "
                f"Comments48h:{reddit_comments:.0f} Active:{reddit_active}")
        elif r.status_code == 429:
            log_btc("[SENT] CoinGecko rate limited — using cache")
        else:
            log_btc(f"[SENT] CoinGecko HTTP {r.status_code}")

    except Exception as e:
        log_btc(f"[SENT] CoinGecko error: {e}")

    # Reddit: rimosso — bloccato su IP datacenter (Railway/render/etc).
    # CoinGecko community_data fornisce già reddit_posts_48h e reddit_comments_48h.

    # ── Composite social score ──
    if scores:
        # CoinGecko peso 55% (structured data), Reddit 45% (real-time sentiment)
        weights_map = {"coingecko": 0.55, "reddit": 0.45}
        total_w = sum(weights_map.get(name, 0.5) for name, _ in scores)
        social_composite = sum(s * weights_map.get(name, 0.5) for name, s in scores) / total_w
    else:
        social_composite = 50

    social_composite = max(5, min(95, round(social_composite, 1)))
    components["social_composite"] = social_composite
    components["sources"] = [name for name, _ in scores]

    _social_cache = {"ts": time.time(), "score": social_composite, "components": components}
    log_btc(f"[SENT] Social composite: {social_composite:.0f} ({len(scores)} sources: {components['sources']})")
    return _social_cache


def _fetch_cryptocompare():
    """
    Fetch IntoTheBlock trading signals da CryptoCompare.
    Ritorna bull/bear percentages basate su:
    - Large Transactions (whale activity)
    - In/Out of the Money (profit/loss distribution)
    - Concentration (holder distribution)
    - Net Network Growth
    
    Endpoint: /data/tradingsignals/intotheblock/latest
    API key passa nell'header (metodo raccomandato) o come query param.
    """
    global _cryptocompare_cache
    if time.time() - _cryptocompare_cache["ts"] < CRYPTOCOMPARE_CACHE_TTL:
        return _cryptocompare_cache

    cc_key = os.getenv("CRYPTOCOMPARE_API_KEY", "")
    if not cc_key:
        return _cryptocompare_cache

    try:
        # Metodo raccomandato: key nell'header Authorization
        url = "https://min-api.cryptocompare.com/data/tradingsignals/intotheblock/latest?fsym=BTC"
        headers = {"authorization": f"Apikey {cc_key}"}
        r = requests.get(url, headers=headers, timeout=10)

        if r.status_code == 200:
            resp = r.json()
            # Check per errori API nella risposta JSON
            if resp.get("Response") == "Error":
                log_btc(f"[SENT] CryptoCompare API error: {resp.get('Message', 'unknown')}")
                return _cryptocompare_cache

            data = resp.get("Data", {})

            if not data:
                log_btc(f"[SENT] CryptoCompare: Data vuoto. Keys risposta: {list(resp.keys())}")
                return _cryptocompare_cache

            # Estrai i segnali principali
            signals = {}
            bull_scores = []

            # Tutti i possibili signal keys di IntoTheBlock
            signal_keys = [
                "largetxsTransaction", "addressesNetGrowth", "inOutVar",
                "concentrationVar", "largetxsVar", "breakdownVar"
            ]

            for key in signal_keys:
                sig = data.get(key, {})
                if sig and isinstance(sig, dict):
                    bull = float(sig.get("bullish", 0) or 0)
                    bear = float(sig.get("bearish", 0) or 0)
                    neutral = float(sig.get("neutral", 0) or 0)
                    sentiment_val = sig.get("sentiment", "")
                    signals[key] = {"bull": bull, "bear": bear, "sentiment": sentiment_val}
                    if bull + bear > 0:
                        bull_scores.append(bull / (bull + bear) * 100)
                    elif sentiment_val:
                        # Fallback: usa il campo "sentiment" testuale
                        if sentiment_val.lower() == "bullish":
                            bull_scores.append(75)
                        elif sentiment_val.lower() == "bearish":
                            bull_scores.append(25)
                        else:
                            bull_scores.append(50)

            if not signals:
                # Log delle keys effettive per debug
                log_btc(f"[SENT] CryptoCompare: nessun signal trovato. Data keys: {list(data.keys())[:15]}")

            # Media delle bull percentages
            if bull_scores:
                avg_bull = sum(bull_scores) / len(bull_scores)
            else:
                avg_bull = 50

            _cryptocompare_cache = {
                "ts": time.time(),
                "bull_pct": round(avg_bull, 1),
                "signals": signals,
                "n_signals": len(bull_scores),
            }
            log_btc(f"[SENT] CryptoCompare OK: Bull:{avg_bull:.0f}% ({len(bull_scores)} signals)")
        elif r.status_code == 429:
            log_btc(f"[SENT] CryptoCompare rate limited — using cache")
        else:
            log_btc(f"[SENT] CryptoCompare HTTP {r.status_code}: {r.text[:200]}")

    except Exception as e:
        log_btc(f"[SENT] CryptoCompare error: {e}")

    return _cryptocompare_cache


def get_sentiment_score():
    """
    Composite sentiment score 0-100 (V7.2 — Dual API).

    Pesi:
      A. Fear & Greed Index          30%  (macro contrarian)
      B. CoinGecko + Reddit Social  30%  (social real-time, free)
      C. Funding z-score             25%  (derivatives positioning)
      D. CryptoCompare IntoTheBlock  15%  (on-chain signals)

    Fallback: se un'API manca, il suo peso viene redistribuito
    proporzionalmente sulle altre fonti.
    """
    global _sentiment_cache
    if time.time() - _sentiment_cache["ts"] < SENTIMENT_CACHE_TTL:
        return _sentiment_cache["score"]

    components = {}
    sources = {}  # {name: (score, weight)}

    # ── A. Fear & Greed Index (alternative.me — gratis, no key) ──
    fgi_score = 50
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1&format=json", timeout=8)
        if r.status_code == 200:
            data = r.json().get("data", [])
            if data:
                fgi_score = int(data[0].get("value", 50))
                components["fgi"] = fgi_score
                components["fgi_label"] = data[0].get("value_classification", "Neutral")
    except Exception as e:
        log_btc(f"[SENT] FGI fetch error: {e}")
    components.setdefault("fgi", fgi_score)
    sources["fgi"] = (fgi_score, 0.30)

    # ── B. CoinGecko + Reddit Social Sentiment (free, replaces LunarCrush) ──
    social = _fetch_social_sentiment()
    social_available = social["ts"] > 0

    if social_available:
        social_score = social["score"]
        components["social_score"] = round(social_score, 1)
        components["social_sources"] = social.get("components", {}).get("sources", [])
        # Passa i sub-componenti per il log dettagliato
        sc = social.get("components", {})
        components["cg_sent_up"] = sc.get("cg_sent_up", "off")
        components["reddit_bullish"] = sc.get("reddit_bullish", "off")
        components["reddit_bearish"] = sc.get("reddit_bearish", "off")
        sources["social"] = (social_score, 0.30)
    else:
        components["social_score"] = None

    # ── C. Funding z-score interno (derivati) ──
    fz = get_funding_z()
    # Map z-score [-3, +3] → [10, 90]
    funding_sent = max(10, min(90, 50 + fz * 13.3))
    components["funding_z"] = round(fz, 2)
    components["funding_sent"] = round(funding_sent, 1)
    sources["funding"] = (funding_sent, 0.25)

    # ── D. CryptoCompare IntoTheBlock (on-chain) ──
    cc = _fetch_cryptocompare()
    cc_available = cc["ts"] > 0

    if cc_available:
        cc_score = max(5, min(95, cc["bull_pct"]))
        components["cc_bull_pct"] = cc["bull_pct"]
        components["cc_n_signals"] = cc.get("n_signals", 0)
        components["cc_score"] = round(cc_score, 1)
        sources["cryptocompare"] = (cc_score, 0.15)
    else:
        components["cc_score"] = None

    # ── Composite con redistribuzione pesi se API mancanti ──
    total_weight = sum(w for _, w in sources.values())
    if total_weight <= 0:
        composite = 50  # fallback totale
    else:
        composite = sum(score * (weight / total_weight)
                        for score, weight in sources.values())
    composite = max(0, min(100, round(composite)))
    components["composite"] = composite

    # Dettaglio fonti attive
    active = [k for k in sources]
    missing = []
    if not social_available:
        missing.append("Social")
    if not cc_available:
        missing.append("CC")
    components["active_sources"] = active
    components["missing_sources"] = missing

    _sentiment_cache = {"score": composite, "components": components, "ts": time.time()}

    # Log dettagliato
    soc_str = f"Social:{social_score:.0f}" if social_available else "Social:off"
    cc_str = f"CC:{cc_score:.0f}" if cc_available else "CC:off"
    log_btc(f"[SENT] Score:{composite} | FGI:{fgi_score} {soc_str} "
        f"Fund:{funding_sent:.0f} {cc_str} | {len(sources)} sources")
    return composite


def get_sentiment_detail():
    """Ritorna i componenti dettagliati dell'ultimo calcolo sentiment."""
    return _sentiment_cache.get("components", {})


def get_sentiment_bias():
    """
    Returns sentiment-based directional bias:
    - score < 20: EXTREME_FEAR → contrarian LONG bias
    - score < 35: FEAR → slight LONG bias
    - score > 80: EXTREME_GREED → contrarian SHORT bias
    - score > 65: GREED → slight SHORT bias
    - else: NEUTRAL

    V7.2: conferma anche con Reddit sentiment per extreme levels.
    """
    s = get_sentiment_score()
    social = _social_cache

    # Reddit bearish > 60% dei post rinforza fear
    reddit_extreme = False
    sc = social.get("components", {})
    r_bull = sc.get("reddit_bullish", 0)
    r_bear = sc.get("reddit_bearish", 0)
    if r_bull + r_bear > 5:  # almeno 5 post classificati
        if r_bear / (r_bull + r_bear) > 0.6:
            reddit_extreme = True  # extreme bearish on Reddit

    if s < 20:
        return "EXTREME_FEAR", s
    if s < 35:
        if reddit_extreme:
            return "EXTREME_FEAR", s
        return "FEAR", s
    if s > 80:
        return "EXTREME_GREED", s
    if s > 65:
        return "GREED", s
    return "NEUTRAL", s


def get_sentiment_adjustment():
    """
    Returns (size_mult_adj, sl_adj, confidence_adj) based on sentiment.
    V7.2: modulato anche da Reddit activity spike.
    """
    s = get_sentiment_score()
    sc = _social_cache.get("components", {})

    # Reddit activity spike: se i commenti 48h sono molto alti → mercato rumoroso
    reddit_comments = sc.get("cg_reddit_comments_48h", 0)
    vol_penalty = 1.0
    if reddit_comments > 5000:  # spike di attività
        vol_penalty = 0.85
        log_btc(f"[SENT] Reddit activity spike: {reddit_comments} comments/48h → size ×0.85")

    if s < 15:
        return 0.7 * vol_penalty, 1.3, 2
    elif s < 30:
        return 0.85 * vol_penalty, 1.15, 1
    elif s > 85:
        return 0.7 * vol_penalty, 1.3, 2
    elif s > 70:
        return 0.85 * vol_penalty, 1.15, 1
    return 1.0 * vol_penalty, 1.0, 0


# ================================================================
# MODULE 2: ORDER FLOW ANALYSIS
# ================================================================
# CVD (Cumulative Volume Delta), OI Momentum, Liquidation Proximity,
# Delta Divergence detection.
# Uses Hyperliquid L2 orderbook + trade data for real-time flow signals.
# ================================================================

_cvd_buffer = deque(maxlen=120)   # 120 readings (one per scan = ~20min window)
_oi_momentum = deque(maxlen=30)   # 30 OI readings for momentum
_flow_cache = {"ts": 0, "data": {}}
FLOW_CACHE_TTL = 15  # aggiorna ogni 15s

# ── Shared API caches (reduce Hyperliquid rate limit pressure) ──
_mid_cache = {"ts": 0, "value": 0}
_pos_cache = {"ts": 0, "value": None}
_bal_cache = {"ts": 0, "value": 0}
API_CACHE_TTL = 3  # 3s cache for mid/pos/bal — fresh enough for scalping


def get_btc_open_interest():
    """Returns current BTC open interest in USD notional."""
    try:
        ctx = call(_info.meta_and_asset_ctxs, timeout=15)
        for a, c in zip(ctx[0]["universe"], ctx[1]):
            if a["name"] == BTC_COIN:
                return float(c.get("openInterest", 0) or 0)
    except Exception as e:
        log_btc(f"[FLOW] OI fetch error: {e}")
    return _btc_oi_cache


def compute_orderbook_delta():
    """
    Computes bid/ask pressure delta from L2 orderbook.
    Returns: {delta, imbalance_ratio, aggressive_side, depth_bid, depth_ask}
    """
    result = {"delta": 0, "imbalance_ratio": 0.5, "aggressive_side": "NEUTRAL",
              "depth_bid": 0, "depth_ask": 0, "wall_level": 0, "wall_side": ""}
    try:
        ob = call(_info.l2_snapshot, BTC_COIN, timeout=10)
        if not ob:
            return result

        levels = ob.get("levels", [])
        bids = levels[0] if len(levels) > 0 else []
        asks = levels[1] if len(levels) > 1 else []
        if not bids or not asks:
            return result

        best_bid = float(bids[0]["px"])
        best_ask = float(asks[0]["px"])
        mid = (best_bid + best_ask) / 2

        # Depth a 5 livelli (tight) e 20 livelli (deep)
        bid_tight = sum(float(b["sz"]) for b in bids[:5])
        ask_tight = sum(float(a["sz"]) for a in asks[:5])
        bid_deep = sum(float(b["sz"]) for b in bids[:20])
        ask_deep = sum(float(a["sz"]) for a in asks[:20])

        result["depth_bid"] = bid_deep
        result["depth_ask"] = ask_deep

        # Delta = bid_size - ask_size (positivo = pressione buy)
        result["delta"] = bid_tight - ask_tight
        total = bid_tight + ask_tight
        result["imbalance_ratio"] = bid_tight / total if total > 0 else 0.5

        # Aggressive side detection (chi sta mangiando il book)
        if result["imbalance_ratio"] > 0.62:
            result["aggressive_side"] = "BUY"
        elif result["imbalance_ratio"] < 0.38:
            result["aggressive_side"] = "SELL"
        else:
            result["aggressive_side"] = "NEUTRAL"

        # Wall detection: livello con size > 3x la media
        all_levels = [(float(b["px"]), float(b["sz"]), "BID") for b in bids[:20]] + \
                     [(float(a["px"]), float(a["sz"]), "ASK") for a in asks[:20]]
        avg_sz = np.mean([l[1] for l in all_levels]) if all_levels else 0
        walls = [l for l in all_levels if l[1] > avg_sz * 3]
        if walls:
            nearest = min(walls, key=lambda w: abs(w[0] - mid))
            result["wall_level"] = nearest[0]
            result["wall_side"] = nearest[2]
            result["wall_dist_pct"] = (nearest[0] - mid) / mid * 100

    except Exception as e:
        log_btc(f"[FLOW] OB delta error: {e}")
    return result


def update_order_flow():
    """
    Aggiorna buffer CVD e OI momentum. Chiamato ogni ciclo di scan.
    """
    global _flow_cache
    if time.time() - _flow_cache["ts"] < FLOW_CACHE_TTL:
        return _flow_cache["data"]

    ob_delta = compute_orderbook_delta()
    _cvd_buffer.append({
        "ts": time.time(),
        "delta": ob_delta["delta"],
        "imbalance": ob_delta["imbalance_ratio"]
    })

    # OI momentum tracking
    current_oi = get_btc_open_interest()
    _oi_momentum.append({"ts": time.time(), "oi": current_oi})

    # Calcola metriche rolling
    flow_data = {
        "ob_delta": ob_delta,
        "cvd_trend": _compute_cvd_trend(),
        "oi_momentum": _compute_oi_momentum(),
        "delta_divergence": _detect_delta_divergence(),
    }

    _flow_cache = {"ts": time.time(), "data": flow_data}
    return flow_data


def _compute_cvd_trend():
    """
    CVD Trend: somma cumulativa dei delta nel buffer.
    Rising CVD = pressione compratori, Falling = venditori.
    Soglia dinamica basata su std dei delta recenti — filtra rumore.
    Returns: {"cvd": float, "cvd_slope": float, "threshold": float, "signal": "BUY"|"SELL"|"NEUTRAL"}
    """
    if len(_cvd_buffer) < 5:
        return {"cvd": 0, "cvd_slope": 0, "threshold": 0, "signal": "NEUTRAL"}

    deltas = [d["delta"] for d in _cvd_buffer]
    cvd = sum(deltas)

    # Slope: regressione lineare sui delta recenti (ultimi 20)
    recent = deltas[-20:] if len(deltas) >= 20 else deltas
    x = np.arange(len(recent))
    if len(recent) >= 3:
        slope = np.polyfit(x, recent, 1)[0]
    else:
        slope = 0

    # Soglia dinamica: CVD deve superare 0.7× std dei delta recenti
    # In mercato rumoroso la soglia sale, in mercato calmo scende
    threshold = float(np.std(recent)) * 0.3 if len(recent) > 5 else 0

    signal = "NEUTRAL"
    if cvd > threshold and slope > 0:
        signal = "BUY"
    elif cvd < -threshold and slope < 0:
        signal = "SELL"

    return {"cvd": round(cvd, 2), "cvd_slope": round(slope, 4),
            "threshold": round(threshold, 2), "signal": signal}


def _compute_oi_momentum():
    """
    OI Momentum: variazione % dell'OI sugli ultimi N readings.
    Rising OI = nuovi contratti = conviction. Falling = chiusure.
    """
    if len(_oi_momentum) < 3:
        return {"oi_change_pct": 0, "oi_accel": 0, "signal": "NEUTRAL"}

    ois = [d["oi"] for d in _oi_momentum]
    latest = ois[-1]
    first = ois[0]

    if first == 0:
        return {"oi_change_pct": 0, "oi_accel": 0, "signal": "NEUTRAL"}

    oi_change = (latest - first) / first
    # Accelerazione: diff della diff
    if len(ois) >= 5:
        diffs = np.diff(ois[-5:])
        accel = diffs[-1] - diffs[0] if len(diffs) >= 2 else 0
    else:
        accel = 0

    signal = "NEUTRAL"
    if oi_change > 0.02 and accel > 0:
        signal = "CONVICTION"     # nuovi contratti in accelerazione
    elif oi_change < -0.02:
        signal = "UNWIND"          # chiusura posizioni
    elif oi_change > 0.01:
        signal = "BUILDING"

    return {
        "oi_change_pct": round(oi_change * 100, 2),
        "oi_accel": round(accel, 2),
        "signal": signal
    }


def _detect_delta_divergence():
    """
    Delta Divergence: prezzo sale ma CVD scende (o viceversa).
    Segnale di esaurimento del trend — high-probability reversal.
    """
    if len(_cvd_buffer) < 10:
        return {"divergence": False, "type": "", "strength": 0}

    # Prezzo degli ultimi 10 readings vs CVD
    recent_deltas = [d["delta"] for d in list(_cvd_buffer)[-10:]]
    cvd_trend = sum(recent_deltas[-5:]) - sum(recent_deltas[:5])

    try:
        mid = get_mid()
        # Approssima il trend del prezzo dall'imbalance trend
        price_momentum = sum(d["imbalance"] - 0.5 for d in list(_cvd_buffer)[-5:])

        # Bearish divergence: price up (imbalance > 0) ma CVD down
        if price_momentum > 0.1 and cvd_trend < -0.5:
            return {"divergence": True, "type": "BEARISH", "strength": abs(cvd_trend)}
        # Bullish divergence: price down ma CVD up
        if price_momentum < -0.1 and cvd_trend > 0.5:
            return {"divergence": True, "type": "BULLISH", "strength": abs(cvd_trend)}
    except:
        pass

    return {"divergence": False, "type": "", "strength": 0}


def get_flow_signal():
    """
    Restituisce un segnale composito dall'Order Flow:
    {"bias": "BUY"|"SELL"|"NEUTRAL", "confidence": 0-10, "details": str}
    """
    flow = update_order_flow()
    if not flow:
        return {"bias": "NEUTRAL", "confidence": 0, "details": "no flow data"}

    score = 0  # -10 (strong sell) to +10 (strong buy)

    # CVD trend (weight: 3)
    cvd = flow.get("cvd_trend", {})
    if cvd.get("signal") == "BUY":
        score += 3
    elif cvd.get("signal") == "SELL":
        score -= 3

    # OB delta (weight: 2)
    ob = flow.get("ob_delta", {})
    if ob.get("aggressive_side") == "BUY":
        score += 2
    elif ob.get("aggressive_side") == "SELL":
        score -= 2

    # OI momentum (weight: 2)
    oi_m = flow.get("oi_momentum", {})
    if oi_m.get("signal") == "CONVICTION":
        score += 2
    elif oi_m.get("signal") == "UNWIND":
        score -= 1

    # Delta divergence (weight: 3 — reversal signal)
    div = flow.get("delta_divergence", {})
    if div.get("divergence"):
        if div["type"] == "BEARISH":
            score -= 3
        elif div["type"] == "BULLISH":
            score += 3

    # Map to bias
    if score >= 3:
        bias = "BUY"
    elif score <= -3:
        bias = "SELL"
    else:
        bias = "NEUTRAL"

    confidence = min(10, abs(score))
    details = (f"CVD:{cvd.get('signal','?')} OB:{ob.get('aggressive_side','?')} "
               f"OI:{oi_m.get('signal','?')} Div:{div.get('type','none')}")

    return {"bias": bias, "confidence": confidence, "details": details}


# ================================================================
# MODULE 3: MACHINE LEARNING — Online Signal Quality Predictor
# ================================================================
# Stochastic Gradient Descent classifier (online learning).
# NO sklearn necessario — implementato a mano per Railway/lightweight.
# Prevede probabilità di win per ogni segnale in base a features storiche.
# Si aggiorna ad ogni trade chiuso (online learning).
# ================================================================

class OnlineGBClassifier:
    """
    Mini gradient boosting binario online — pesi aggiornati ad ogni sample.
    Features: RSI, MACD, ADX, vol, funding_z, sentiment, OB_imbalance, CVD,
              regime_encoded, scalp_mode_encoded, setup_score, oi_change.
    Target: 1 = trade vincente, 0 = perdente.
    """
    FEATURE_NAMES = [
        "rsi_15m", "rsi_1h", "macd_hist", "adx_1h", "vol_rel",
        "funding_z", "sentiment", "ob_imbalance", "cvd_slope",
        "regime_enc", "mode_enc", "setup_score", "oi_change_pct",
        "bb_pos", "ema_slope", "spread"
    ]
    N_FEATURES = len(FEATURE_NAMES)

    def __init__(self):
        # Weights: linear model con learning rate decay
        self.weights = np.zeros(self.N_FEATURES)
        self.bias = 0.0
        self.lr = 0.05
        self.n_samples = 0
        self.feature_means = np.zeros(self.N_FEATURES)
        self.feature_vars = np.ones(self.N_FEATURES)
        # Performance tracking
        self.predictions = deque(maxlen=200)  # (predicted_prob, actual_outcome)
        self.accuracy_window = deque(maxlen=50)

    def _sigmoid(self, z):
        z = np.clip(z, -20, 20)
        return 1.0 / (1.0 + np.exp(-z))

    def _normalize(self, x):
        """Running normalization usando media e varianza online."""
        return (x - self.feature_means) / (np.sqrt(self.feature_vars) + 1e-8)

    def _update_stats(self, x):
        """Welford's online algorithm per media e varianza."""
        self.n_samples += 1
        n = self.n_samples
        delta = x - self.feature_means
        self.feature_means += delta / n
        delta2 = x - self.feature_means
        self.feature_vars = ((n - 1) * self.feature_vars + delta * delta2) / n

    def predict_proba(self, features):
        """
        Predict P(win) per il segnale corrente.
        Returns: float 0.0-1.0
        """
        if self.n_samples < 10:
            return 0.5  # non abbastanza dati — default neutral

        x = np.array(features, dtype=np.float64)
        x_norm = self._normalize(x)
        z = np.dot(self.weights, x_norm) + self.bias
        return float(self._sigmoid(z))

    def update(self, features, outcome):
        """
        Online gradient descent step.
        features: array di N_FEATURES float
        outcome: 1 (win) o 0 (loss)
        """
        x = np.array(features, dtype=np.float64)
        self._update_stats(x)
        x_norm = self._normalize(x)

        # Forward pass
        z = np.dot(self.weights, x_norm) + self.bias
        pred = self._sigmoid(z)

        # Binary cross-entropy gradient
        error = pred - outcome

        # Learning rate con decay
        lr = self.lr / (1 + self.n_samples * 0.001)

        # L2 regularization (weight decay)
        reg = 0.001

        # Update
        self.weights -= lr * (error * x_norm + reg * self.weights)
        self.bias -= lr * error

        # Track accuracy
        predicted_class = 1 if pred >= 0.5 else 0
        correct = 1 if predicted_class == outcome else 0
        self.accuracy_window.append(correct)
        self.predictions.append((pred, outcome))

        # Log periodico
        if self.n_samples % 10 == 0:
            acc = np.mean(self.accuracy_window) if self.accuracy_window else 0
            log_btc(f"[ML] Update #{self.n_samples} | Acc:{acc:.0%} | pred:{pred:.2f} actual:{outcome}")

    def get_accuracy(self):
        if not self.accuracy_window:
            return 0.5
        return float(np.mean(self.accuracy_window))

    def get_feature_importance(self):
        """Returns feature importance (absolute weight magnitude)."""
        importance = np.abs(self.weights)
        total = importance.sum()
        if total == 0:
            return {}
        normalized = importance / total
        return {name: round(float(val), 3)
                for name, val in zip(self.FEATURE_NAMES, normalized)
                if val > 0.01}

    def to_dict(self):
        return {
            "weights": self.weights.tolist(),
            "bias": self.bias,
            "n_samples": self.n_samples,
            "feature_means": self.feature_means.tolist(),
            "feature_vars": self.feature_vars.tolist(),
            "lr": self.lr
        }

    def from_dict(self, d):
        if not d:
            return
        try:
            self.weights = np.array(d["weights"])
            self.bias = d["bias"]
            self.n_samples = d["n_samples"]
            self.feature_means = np.array(d["feature_means"])
            self.feature_vars = np.array(d["feature_vars"])
            self.lr = d.get("lr", 0.05)
        except Exception as e:
            log_btc(f"[ML] Load error: {e}")


# Singleton ML model
_ml_model = OnlineGBClassifier()

def encode_regime(regime):
    return {"BULL": 1.0, "BEAR": -1.0, "RANGE": 0.0}.get(regime, 0.0)

def encode_mode(mode):
    return {"TREND": 1.0, "RANGE": 0.0, "FLASH": 2.0}.get(mode, 0.0)


def build_ml_features(rsi_15m, rsi_1h, macd_hist, adx_1h, vol_rel,
                       funding_z, sentiment, ob_imbalance, cvd_slope,
                       regime, scalp_mode, setup_score, oi_change_pct,
                       bb_pos=0, ema_slope=0, spread=0):
    """
    Costruisce il vettore di features per il modello ML.
    Ordine DEVE corrispondere a FEATURE_NAMES.
    """
    return [
        float(rsi_15m), float(rsi_1h), float(macd_hist), float(adx_1h),
        float(vol_rel), float(funding_z), float(sentiment),
        float(ob_imbalance), float(cvd_slope),
        encode_regime(regime), encode_mode(scalp_mode),
        float(setup_score), float(oi_change_pct),
        float(bb_pos), float(ema_slope), float(spread)
    ]


def ml_predict_signal(features):
    """
    ML filter:
    - < 50 samples: OSSERVA (non blocca)
    - >= 50 samples + accuracy > 55%: BLOCCA se P(win) < 40%
    """
    prob = _ml_model.predict_proba(features)
    acc = _ml_model.get_accuracy()
    n = _ml_model.n_samples

    info = {
        "ml_active": False,
        "prob": round(prob, 3),
        "acc": round(acc, 3),
        "n_samples": n,
        "action": "OBSERVE",
        "block": False,
    }

    if n >= 50 and acc >= 0.55:
        info["ml_active"] = True
        if prob < 0.40:
            info["action"] = "BLOCK"
            info["block"] = True
            info["action"] = "SOFT_REDUCE"
        elif prob < 0.45:
            # Sotto media → riduci size del 10%
            info["size_mult"] = 0.90
            info["action"] = "SLIGHT_REDUCE"
        else:
            # Probabilità OK o alta → nessuna modifica (mai boosta)
            info["action"] = "NEUTRAL"
            info["size_mult"] = 1.0

    return prob, info


def ml_record_outcome(features, won):
    """Record trade outcome per aggiornare il modello online."""
    _ml_model.update(features, 1 if won else 0)
    if _ml_model.n_samples % 5 == 0:
        _rset("btc7:ml_model", _ml_model.to_dict())
        log_btc(f"[ML] Saved ({_ml_model.n_samples} samples, acc:{_ml_model.get_accuracy():.0%})")


def ml_load_model():
    """Carica modello ML da Redis (persistenza tra restart)."""
    d = _rget("btc7:ml_model")
    if d:
        _ml_model.from_dict(d)
        log_btc(f"[ML] Model loaded: {_ml_model.n_samples} samples, weights:{np.abs(_ml_model.weights).sum():.2f}")
    else:
        log_btc("[ML] No saved model — starting fresh")


def compute_hourly_bias():
    """
    Fleet bias basato su momentum BTC + trade in corso.
    Se BTC ha un trade LONG in loss → il mercato scende → SHORT_BIAS per ALT.
    """
    global _last_oi
    try:
        df_15m = fetch_df("15m", 3)
        df_1h = fetch_df("1h", 2)
    except:
        fleet_set_bias("NEUTRAL", "data error")
        return "NEUTRAL"

    mom_15m = 0
    mom_1h = 0
    if df_15m is not None and len(df_15m) >= 3:
        mom_15m = (float(df_15m['close'].iloc[-1]) / float(df_15m['close'].iloc[-3])) - 1
    if df_1h is not None and len(df_1h) >= 2:
        mom_1h = (float(df_1h['close'].iloc[-1]) / float(df_1h['close'].iloc[-2])) - 1

    # ── Trade in loss = segnale direzionale forte ──
    pos = get_position()
    if pos:
        entry = pos["entry"]
        mid = get_mid()
        d = "LONG" if pos["szi"] > 0 else "SHORT"
        pnl_pct = ((mid - entry) / entry if d == "LONG" else (entry - mid) / entry)
        if pnl_pct < -0.003:  # in loss > -0.3%
            # Trade LONG in loss → mercato scende → SHORT per ALT
            if d == "LONG":
                bias = "SHORT_BIAS"; reason = f"BTC LONG in loss {pnl_pct:+.2%} → mercato scende"
            else:
                bias = "LONG_BIAS"; reason = f"BTC SHORT in loss {pnl_pct:+.2%} → mercato sale"
            fleet_set_bias(bias, reason)
            log_btc(f"Fleet: {bias} | {reason}")
            return bias

    # ── Momentum standard ──
    if mom_15m > 0.0015:
        bias = "LONG_BIAS"; reason = f"BTC +{mom_15m:.2%}/30m"
    elif mom_15m < -0.0015:
        bias = "SHORT_BIAS"; reason = f"BTC {mom_15m:.2%}/30m"
    elif mom_1h > 0.004:
        bias = "LONG_BIAS"; reason = f"BTC +{mom_1h:.2%}/1h"
    elif mom_1h < -0.004:
        bias = "SHORT_BIAS"; reason = f"BTC {mom_1h:.2%}/1h"
    else:
        bias = "NEUTRAL"; reason = f"BTC flat ({mom_15m:+.2%}/30m)"

    fleet_set_bias(bias, reason)
    log_btc(f"Fleet: {bias} | {reason}")
    return bias


# ================================================================
# BTC STATE MANAGEMENT
# ================================================================
_btc_trades_today = []
_btc_last_trade_ts = 0
_btc_consec_losses = 0
_btc_params = {}
_btc_is_trading = False
_btc_start_balance = None
_btc_kill_switch = False
_btc_pending_order = {}
_btc_regime = "RANGE"
_btc_regime_ts = 0
_btc_bt_results = {}
_btc_bt_ts = 0
_btc_current_signal = None
_btc_sl_oid = None
_btc_tp_oid = None
_btc_coin_meta = {}
_btc_funding_history = []
_btc_oi_cache = 0.0
_btc_oi_prev = 0.0
BTC_FUNDING_HISTORY_LEN = 42
_btc_last_report_ts = 0

def update_funding_oi():
    """Fetch BTC funding rate e OI, aggiorna storico per z-score."""
    global _btc_funding_history, _btc_oi_cache, _btc_oi_prev
    try:
        ctx = call(_info.meta_and_asset_ctxs, timeout=15)
        for a, c in zip(ctx[0]["universe"], ctx[1]):
            if a["name"] == BTC_COIN:
                funding = float(c.get("funding", 0) or 0)
                _btc_funding_history.append(funding)
                if len(_btc_funding_history) > FUNDING_HISTORY_LEN:
                    _btc_funding_history.pop(0)

                _btc_oi_prev = _btc_oi_cache
                _btc_oi_cache = float(c.get("openInterest", 0) or 0)
                break
    except Exception as e:
        log_btc(f"update_funding_oi: {e}")

def get_funding_z():
    """Z-score del funding BTC: quanto è estremo rispetto alla storia recente."""
    if len(_btc_funding_history) < 3:
        return 0.0
    arr = np.array(_btc_funding_history)
    std = arr.std()
    if std < 1e-10:
        return 0.0
    return float((_btc_funding_history[-1] - arr.mean()) / std)

def get_oi_change():
    """Variazione % dell'OI BTC rispetto al reading precedente."""
    if _btc_oi_prev == 0:
        return 0.0
    return (_btc_oi_cache - _btc_oi_prev) / (_btc_oi_prev + 1e-10)

# BTC state load/save
def load_state():
    global _btc_trades_today, _btc_consec_losses, _btc_params, _btc_pending_order
    _btc_trades_today = _rget("btc6:trades") or []
    now = time.time()
    _btc_trades_today = [t for t in _btc_trades_today if now - t.get("ts_close", t.get("ts", 0)) < 86400]
    _btc_consec_losses = 0
    for t in reversed(_btc_trades_today):
        if t.get("pnl", 0) < 0: _btc_consec_losses += 1
        else: break
    _btc_params = _rget("btc6:params") or {}
    # Load pending order
    _btc_pending_order = _rget("btc6:pending") or {}
    if _btc_pending_order:
        age = now - _btc_pending_order.get("placed_at", 0)
        if age > PENDING_ORDER_TTL:
            log_btc(f"🧹 Pending order scaduto in Redis ({age:.0f}s) — clearing")
            _btc_pending_order = {}
            _rset("btc6:pending", None)
        else:
            log_btc(f"⏳ Pending order restored from Redis (age:{age:.0f}s)")
    log_btc(f"State: {len(_btc_trades_today)} trades, {_btc_consec_losses} losses")

def save_trade(pnl, direction, entry, exit_px, sig_type="", sl_dist=0, regime="",
               extra=None):
    """
    Salva trade completo su Redis con tutti i dettagli:
    entry, exit, SL/TP originali, trailing history, partial close, durata.
    """
    global _btc_trades_today, _btc_consec_losses
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
    _btc_trades_today.append(t)
    _rset("btc6:trades", _btc_trades_today[-200:])
    if pnl < 0: _btc_consec_losses += 1
    else: _btc_consec_losses = 0
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
    global _btc_params
    now = time.time()
    recent = [t for t in _btc_trades_today if now - t.get("ts_close", t.get("ts", 0)) < 86400]
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

    _btc_params = {
        "wr": round(wr, 3), "pf": round(pf, 2), "total_pnl": round(total_pnl, 2),
        "n_trades": len(recent), "avg_win": round(avg_win, 2), "avg_loss": round(avg_loss, 2),
        "sl_adj": sl_adj, "tp_adj": tp_adj,
        "by_type": {k: round(v["w"]/(v["w"]+v["l"]), 2) for k,v in by_type.items() if v["w"]+v["l"]>0},
        "by_regime": {k: round(v["w"]/(v["w"]+v["l"]), 2) for k,v in by_regime.items() if v["w"]+v["l"]>0},
        "ts": time.time()
    }
    _rset("btc6:params", _btc_params)
    log_btc(f"📊 Stats: WR:{wr:.0%} PF:{pf:.2f} PnL:${total_pnl:.2f} SLadj:{sl_adj} TPadj:{tp_adj}")

def get_sl_tp_adjustments():
    """Returns (sl_mult, tp_mult) from adaptive params."""
    return _btc_params.get("sl_adj", 1.0), _btc_params.get("tp_adj", 1.0)

def check_circuit_breaker():
    now = time.time()
    daily_pnl = sum(t["pnl"] for t in _btc_trades_today if now - t.get("ts_close", t.get("ts", 0)) < 86400)
    if daily_pnl <= -MAX_DAILY_LOSS:
        return True, f"daily loss ${daily_pnl:.1f}"
    if _btc_consec_losses >= MAX_CONSEC_LOSS:
        last_loss_ts = max((t.get("ts_close", t.get("ts", 0)) for t in _btc_trades_today if t["pnl"] < 0), default=0)
        if now - last_loss_ts < 1800:
            return True, f"{_btc_consec_losses} consec losses, pause {int((1800-(now-last_loss_ts))/60)}min"
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
            c = call(_info.candles_snapshot, BTC_COIN, tf, now - 86400000*days, now, timeout=15)
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
_btc_regime = "RANGE"
_btc_regime_ts = 0

def update_regime():
    """
    4-state regime detection basato su ADX + ATR relativo.
    
    RANGE_LOW_VOL: ADX < 18 + ATR sotto media → NO TRADE (spread mangia TP)
    RANGE:         ADX < 22 → mean reversion, entrambe direzioni
    TREND:         ADX >= 22 → momentum, segui EMA direction
    TREND_STRONG:  ADX >= 22 + ATR sopra media → trend forte, size piena
    """
    global _btc_regime, _btc_regime_ts
    if time.time() - _btc_regime_ts < BTC_REGIME_INTERVAL:
        return _btc_regime

    df = fetch_df("4h", 90)
    if df is None or len(df) < 60:
        return _btc_regime

    r = df.iloc[-1]
    adx = float(r.get('adx', 20))
    atr = float(r.get('atr', 0))

    # ATR medio su 50 candele (12.5 giorni su 4h)
    atr_series = df['atr'].astype(float)
    atr_mean = float(atr_series.rolling(50, min_periods=20).mean().iloc[-1])

    low_vol = atr < atr_mean * 0.8  if atr_mean > 0 else False
    high_vol = atr > atr_mean * 1.2 if atr_mean > 0 else False

    if adx < 18 and low_vol:
        _btc_regime = "RANGE_LOW_VOL"
    elif adx < 22:
        _btc_regime = "RANGE"
    elif adx >= 22 and high_vol:
        _btc_regime = "TREND_STRONG"
    else:
        _btc_regime = "TREND"

    _btc_regime_ts = time.time()
    log_btc(f"Regime: {_btc_regime} (ADX:{adx:.0f} ATR:{atr:.0f} mean:{atr_mean:.0f} "
            f"{'LOW_VOL' if low_vol else 'HIGH_VOL' if high_vol else 'NORMAL'})")
    return _btc_regime

# ================================================================
# BACKTEST — testa i segnali esatti del bot su storico 5m
# ================================================================
_btc_bt_results = {}   # {"PULLBACK_LONG": {"pf":1.3, "wr":0.45, "n":80}, ...}
_btc_bt_ts = 0

def run_backtest():
    """
    Backtest su 14 giorni di candele 5m BTC.
    Testa ogni tipo di segnale: PULLBACK, BREAKOUT, REVERSAL × LONG/SHORT.
    Salva risultati in _btc_bt_results — usato per bloccare segnali non profittevoli.
    """
    global _btc_bt_results, _btc_bt_ts
    if time.time() - _btc_bt_ts < 3600 and _btc_bt_results:
        return _btc_bt_results  # cache 1 ora

    log_btc("📊 Running backtest...")
    df_1h = fetch_df("1h", 60)
    df_5m = fetch_df("15m", 30)   # 30 giorni di 15m = ~2880 candele
    if df_1h is None or df_5m is None or len(df_5m) < 500 or len(df_1h) < 200:
        log_btc("📊 Backtest: insufficient data")
        return _btc_bt_results

    # Allinea 1h con 5m per timestamp
    df_1h['t_hour'] = pd.to_datetime(df_1h['t'], unit='ms').dt.floor('h')
    df_5m['t_hour'] = pd.to_datetime(df_5m['t'], unit='ms').dt.floor('h')

    # Merge 1h indicators onto 15m (forward fill)
    h_cols = df_1h[['t_hour','rsi','macd_hist','ema9','ema21','ema_slope','ema50','ema200','adx']].copy()
    h_cols.columns = ['t_hour','rsi_1h','macd_1h','ema9_1h','ema21_1h','slope_1h','ema50_1h','ema200_1h','adx_1h']
    merged = pd.merge_asof(
        df_5m.sort_values('t_hour'), h_cols.sort_values('t_hour'),
        on='t_hour', direction='backward'
    ).sort_index()

    if len(merged) < 500:
        log_btc("📊 Backtest: merge failed")
        return _btc_bt_results

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
    adx1h = merged['adx_1h'].values.astype(np.float64)

    n = len(c)
    fwd = 12  # 12 candele 15m = 3h — TP 0.15% si raggiunge in minuti
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
        while i < n - fwd - 5:  # -5 per spazio delay
            try:
                if not sig_cond(i):
                    i += 1; continue
            except:
                i += 1; continue

            atr_i = atr5[i]
            if atr_i <= 0: i += 1; continue

            # ══════════════════════════════════════════════════════
            # REALISTIC EXECUTION MODEL
            # ══════════════════════════════════════════════════════

            # 1. ENTRY DELAY: 2-3 barre dopo il segnale (tempo di elaborazione)
            #    Il bot vede il segnale alla chiusura di candela i,
            #    processa per ~30s, entra alla candela i+2 o i+3.
            #    Usiamo un delay random tra 2 e 3 barre.
            delay = 2 + (i % 2)  # alternato 2 e 3 per simulare variabilità
            entry_bar = i + delay
            if entry_bar >= n - fwd:
                i += fwd; continue

            # 2. ENTRY = OPEN della candela di entry (non midpoint)
            #    Nella realtà: il bot vede il segnale alla chiusura di candela i,
            #    piazza ordine, viene fillato all'apertura della candela i+delay.
            entry_open = float(merged['open'].values[entry_bar]) if 'open' in merged.columns else c[entry_bar]

            # 3. SLIPPAGE DINAMICO: basato su range della candela precedente
            #    (la volatilità al momento dell'entry)
            prev_range = h[entry_bar - 1] - l[entry_bar - 1]
            slippage_pct = min(prev_range / (entry_open + 1e-10) * 0.3, 0.003)  # max 0.3%
            # Slippage + fee sempre sfavorevoli
            fee = 0.0005  # 0.05% per lato
            if sig_dir == "LONG":
                px_entry = entry_open * (1 + slippage_pct + fee)
            else:
                px_entry = entry_open * (1 - slippage_pct - fee)

            # SL/TP: ATR-based (come il live)
            sl_d = atr_i * 1.2
            tp_d = atr_i * 1.8
            sl_d = max(sl_d, px_entry * 0.003)
            tp_d = max(tp_d, px_entry * 0.003)

            # 4. TP/SL CHECK: da entry_bar+1 in poi (non dalla candela segnale)
            check_start = entry_bar + 1
            check_end = min(entry_bar + fwd, n)

            if check_start >= check_end:
                i += fwd; continue

            if sig_dir == "LONG":
                tp_hits = np.where(h[check_start:check_end] >= px_entry + tp_d)[0]
                sl_hits = np.where(l[check_start:check_end] <= px_entry - sl_d)[0]
            else:
                tp_hits = np.where(l[check_start:check_end] <= px_entry - tp_d)[0]
                sl_hits = np.where(h[check_start:check_end] >= px_entry + sl_d)[0]

            tp_f = tp_hits[0]+1 if len(tp_hits) else fwd+1
            sl_f = sl_hits[0]+1 if len(sl_hits) else fwd+1

            if tp_f == sl_f == fwd+1:
                i += fwd; continue

            # 5. EXIT SLIPPAGE: SL exit ha slippage extra (market order sotto pressione)
            win = 1 if tp_f < sl_f else 0
            if win:
                # TP hit — slippage minima (prezzo favorevole)
                net_gain = tp_d - (px_entry * fee)  # TP minus exit fee
                ret = net_gain / px_entry
            else:
                # SL hit — slippage peggiore (market order in panico)
                net_loss = sl_d + (px_entry * fee)  # SL plus exit fee
                ret = -net_loss / px_entry

            trades.append((win, ret, tp_d * 0.95 if win else 0, sl_d * 1.10 if not win else 0))
            i += max(entry_bar - i + max(tp_f, sl_f), 4)  # skip past trade

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
            log_btc(f"  {emoji} {sig_name:<20} PF:{pf:.2f} WR:{wr:.0%} N:{nt} Recent:{rpf:.2f}")
        else:
            results[sig_name] = {"pf": 0, "pf_recent": 0, "wr": 0, "n": nt, "avg_ret": 0}
            log_btc(f"  ⚠️ {sig_name:<20} only {nt} trades — insufficient")

    _btc_bt_results = results
    _btc_bt_ts = time.time()
    _rset("btc6:backtest", results)
    log_btc(f"📊 Backtest done: {len(results)} signal types tested")
    return results

def is_signal_allowed(sig_type, direction):
    """Backtest gate — recent PF >= 0.9 passa sempre, PF < 0.6 su entrambi blocca."""
    key = f"{sig_type}_{direction}"
    bt = _btc_bt_results.get(key, {})
    pf = bt.get("pf", 0)
    pf_recent = bt.get("pf_recent", 0)
    n = bt.get("n", 0)
    if n < 10:
        return True
    if pf_recent >= 0.9:
        return True
    if pf < 0.6 and pf_recent < 0.6:
        return False
    return True
    # Edge debole — permetti ma il setup score + ML scaleranno la size
    if pf >= 0.6 or pf_recent >= 0.6:
        return True
    return True  # nel dubbio, permetti — il mercato cambia
def flow_trigger(flow_data, regime, allow_long=True, allow_short=True):
    """
    Flow trigger a 2 livelli.
    Usa CVD sopra soglia dinamica (std-based) + slope positivo.
    CVD sopra threshold = pressione reale, non rumore.
    Slope > 0 = la pressione sta accelerando.
    """
    cvd_data = flow_data.get("cvd_trend", {})
    ob_data = flow_data.get("ob_delta", {})
    oi_data = flow_data.get("oi_momentum", {})

    cvd = cvd_data.get("cvd", 0)
    slope = cvd_data.get("cvd_slope", 0)
    threshold = cvd_data.get("threshold", 0)
    ob = ob_data.get("imbalance_ratio", 0.5)
    oi = oi_data.get("oi_change_pct", 0)
    oi_signal = oi_data.get("signal", "NEUTRAL")

    direction = None
    sig_type = None

    # ── FLOW_STRONG: CVD sopra soglia + slope + OB forte + OI cresce ──
    if (allow_long and cvd > threshold and slope > 0 and
        ob > 0.58 and oi_signal != "UNWIND" and oi > 0):
        direction = "LONG"; sig_type = "FLOW_STRONG"
    elif (allow_short and cvd < -threshold and slope < 0 and
          ob < 0.42 and oi_signal != "UNWIND" and oi > 0):
        direction = "SHORT"; sig_type = "FLOW_STRONG"

    # ── FLOW_WEAK: CVD sopra soglia + slope + OB conferma ──
    elif (allow_long and cvd > threshold and slope > 0 and ob > 0.52 and
          oi_signal in ("BUILDING", "CONVICTION")):
        direction = "LONG"; sig_type = "FLOW_WEAK"
    elif (allow_short and cvd < -threshold and slope < 0 and ob < 0.48 and
          oi_signal in ("BUILDING", "CONVICTION")):
        direction = "SHORT"; sig_type = "FLOW_WEAK"

    if direction:
        details = f"CVD:{cvd:.1f}(thr:{threshold:.1f}) slope:{slope:.3f} OI:{oi:+.1f}%({oi_signal}) OB:{ob:.0%} [{regime}]"
        log_btc(f"⚡ {sig_type} {'BUY' if direction=='LONG' else 'SELL'} — {details}")
        return {"direction": direction, "sig_type": sig_type, "details": details}

    return None


_mid_price_cache = []  # [(timestamp, price), ...]

def momentum_trigger(df_15m, allow_long=True, allow_short=True):
    """
    Momentum a 2 velocità:
    1. INSTANT: confronta mid attuale con 2-3 minuti fa (veloce, cattura inizio move)
    2. CANDLE: conferma con candele 15m (robusto, filtra noise)
    """
    global _mid_price_cache

    # ── INSTANT MOMENTUM: mid price vs 2-3 min fa ──
    mid_now = get_mid()
    now = time.time()
    if mid_now > 0:
        _mid_price_cache.append((now, mid_now))
        # Tieni solo ultimi 5 minuti
        _mid_price_cache = [(t, p) for t, p in _mid_price_cache if now - t < 300]

    # Confronta con 2 min fa
    instant_sig = None
    old_prices = [(t, p) for t, p in _mid_price_cache if 90 < now - t < 180]
    if old_prices and mid_now > 0:
        old_px = old_prices[0][1]
        instant_move = (mid_now - old_px) / old_px

        if allow_long and instant_move > 0.002:  # +0.2% in 2min
            instant_sig = {"direction": "LONG", "type": "MOMENTUM",
                          "details": f"move:+{instant_move:.2%}/2m (instant)"}
        elif allow_short and instant_move < -0.002:
            instant_sig = {"direction": "SHORT", "type": "MOMENTUM",
                          "details": f"move:{instant_move:.2%}/2m (instant)"}

    if instant_sig:
        return instant_sig

    # ── CANDLE MOMENTUM: fallback su candele 15m ──
    if df_15m is None or len(df_15m) < 5:
        return None

    c_now = float(df_15m.iloc[-1]['close'])
    c_1ago = float(df_15m.iloc[-2]['close'])
    c_2ago = float(df_15m.iloc[-3]['close'])
    vol = float(df_15m.iloc[-1]['vol_rel'])

    move_30m = (c_now - c_2ago) / c_2ago
    move_last = (c_now - c_1ago) / c_1ago

    if allow_long and move_30m > 0.002 and move_last > 0 and vol >= 0.5:
        return {"direction": "LONG", "type": "MOMENTUM",
                "details": f"move:+{move_30m:.2%}/30m last:+{move_last:.2%} vol:{vol:.1f}x"}

    if allow_short and move_30m < -0.002 and move_last < 0 and vol >= 0.5:
        return {"direction": "SHORT", "type": "MOMENTUM",
                "details": f"move:{move_30m:.2%}/30m last:{move_last:.2%} vol:{vol:.1f}x"}

    return None


def technical_trigger(r, r_prev, h_1h, allow_long=True, allow_short=True):
    """
    Technical trigger — pullback CON trend direction check.
    Non genera SHORT se il trend 1h è UP (e viceversa).
    Non entra se il move è già esaurito (RSI troppo lontano da estremi).
    """
    rsi = float(r['rsi'])
    macd = float(r['macd_hist'])
    macd_prev = float(r_prev['macd_hist'])
    vol = float(r['vol_rel'])
    slope = float(r['ema_slope'])

    # Trend 1h direction
    ema9 = float(h_1h['ema9'])
    ema21 = float(h_1h['ema21'])
    trend_up = ema9 > ema21

    if rsi < 20:
        if allow_long and macd > macd_prev:
            return {
                "direction": "LONG", 
                "type": "MEAN_REVERSION",
                "details": f"🔥 RSI:{rsi:.1f} ESTREMO - Rimbalzo tecnico cercato (MACD↑)"
            }
        return None # Blocca lo Short se RSI < 20 anche se le altre condizioni Short sono vere

    # --- LOGICA PULLBACK STANDARD (ESISTENTE) ---

    # Pullback BUY: RSI oversold + MACD turning + trend 1h UP (non contro-trend)
    if (allow_long and rsi < 40 and macd > macd_prev and vol >= 0.3
        and trend_up and slope > -0.001):
        return {"direction": "LONG", "type": "PULLBACK",
                "details": f"RSI:{rsi:.0f} MACD↑ vol:{vol:.1f}x trend:UP"}

    # Pullback SELL: RSI overbought + MACD turning + trend 1h DOWN
    if (allow_short and rsi > 60 and macd < macd_prev and vol >= 0.3
        and not trend_up and slope < 0.001):
        return {"direction": "SHORT", "type": "PULLBACK",
                "details": f"RSI:{rsi:.0f} MACD↓ vol:{vol:.1f}x trend:DOWN"}

    return None


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

    direction = None
    sig_type = None
    details = ""

    # RANGE_LOW_VOL: non tradare — spread mangia il TP
    if regime == "RANGE_LOW_VOL":
        return None

    # ── ENVIRONMENT CHECK: blocca se tutto è neutro ──
    # Se flow NEUTRAL + sentiment 40-60 + ADX < 20 → mercato morto, non tradare
    flow_sig_check = get_flow_signal()
    flow_neutral = flow_sig_check.get("bias", "NEUTRAL") == "NEUTRAL"
    sent = get_sentiment_score()
    sent_neutral = 40 <= sent <= 60
    adx_low = adx1h < 20

    if flow_neutral and sent_neutral and adx_low:
        return None  # mercato morto — nessun edge

    # ATR troppo basso → volatilità insufficiente per raggiungere TP
    atr_pct = atr5 / px if px > 0 else 0
    if atr_pct < 0.0015:
        return None

    allow_long = True
    allow_short = True

    # ══════════════════════════════════════════════════════════
    # TRIPLE TRIGGER: Flow (master) → Tech (fallback) → Momentum (catch-all)
    # + RANGE TRIGGER: Bollinger Band mean reversion
    # ══════════════════════════════════════════════════════════

    flow_sig = flow_trigger(_flow_cache.get("data", {}), regime, allow_long, allow_short)
    tech_sig = technical_trigger(r, r2, h, allow_long, allow_short)
    mom_sig = momentum_trigger(df_15m, allow_long, allow_short)

    # ── RANGE TRIGGER: mean reversion alle Bollinger Bands ──
    # In RANGE: compra quando tocca BB bassa, vendi quando tocca BB alta
    range_sig = None
    if scalp_mode == "RANGE":
        rsi_15m = float(r['rsi'])
        bb_pos_now = float(r['bb_pos'])  # -1 = sotto BB bassa, +1 = sopra BB alta

        # LONG: prezzo alla BB bassa + RSI oversold
        if allow_long and bb_pos_now < -0.7 and rsi_15m < 35:
            range_sig = {"direction": "LONG", "type": "RANGE_REV",
                        "details": f"BB:{bb_pos_now:.2f} RSI:{rsi_15m:.0f} (mean reversion)"}
        # SHORT: prezzo alla BB alta + RSI overbought
        elif allow_short and bb_pos_now > 0.7 and rsi_15m > 65:
            range_sig = {"direction": "SHORT", "type": "RANGE_REV",
                        "details": f"BB:{bb_pos_now:.2f} RSI:{rsi_15m:.0f} (mean reversion)"}

    if flow_sig:
        direction = flow_sig["direction"]
        sig_type = flow_sig["sig_type"]
        details = flow_sig["details"]
        if tech_sig and tech_sig["direction"] == direction:
            sig_type += "+TECH"
            details += f" +{tech_sig['type']}"
        if mom_sig and mom_sig["direction"] == direction:
            sig_type += "+MOM"
            details += f" +{mom_sig['details']}"
    elif range_sig:
        # RANGE mean reversion ha priorità su tech/momentum in RANGE mode
        direction = range_sig["direction"]
        sig_type = range_sig["type"]
        details = range_sig["details"]
    elif tech_sig:
        direction = tech_sig["direction"]
        sig_type = tech_sig["type"]
        details = tech_sig["details"]
        if mom_sig and mom_sig["direction"] == direction:
            sig_type += "+MOM"
            details += f" +{mom_sig['details']}"
    elif mom_sig:
        direction = mom_sig["direction"]
        sig_type = mom_sig["type"]
        details = mom_sig["details"]
    else:
        return None

    # ══════════════════════════════════════════════════════════
    # SIZE SCALING per signal quality
    # FLOW_STRONG+TECH+MOM = 1.0x (massima conviction)
    # FLOW_STRONG          = 1.0x
    # FLOW_WEAK+TECH       = 0.8x
    # FLOW_WEAK            = 0.7x
    # PULLBACK (tech)      = 0.5x
    # MOMENTUM (solo)      = 0.5x (no flow)
    # ══════════════════════════════════════════════════════════

    # Size fissa $5 — con questo capitale non si scala
    size_mult = 1.0

    # Spread check
    liq = fetch_liquidity()
    if liq.get("spread") is not None and liq["spread"] > 0.0008:
        log_btc(f"⚠️ Spread {liq['spread']:.4%} > 0.08% — skip")
        return None

    # ML: log only, no block (0 samples)
    ml_features = []
    details += f" sent:{get_sentiment_score()}"

    log_btc(f"[SIGNAL] {direction} {sig_type} | {details}")

    # ══════════════════════════════════════════════════════════
    # EXPECTATION: il regime crea l'aspettativa, la strategia SL/TP si adatta
    #
    # Prezzo scende da giorni (TREND down) → aspetto inversione LONG
    #   → SHORT nel frattempo: TP stretto, SL veloce (prendi profitto rapido)
    #   → LONG quando arriva: TP largo, trailing attivo (lascia correre)
    #
    # Prezzo sale da giorni (TREND up) → aspetto continuazione o inversione SHORT
    #   → LONG nel frattempo: TP stretto (potrebbe invertire)
    #   → SHORT se inverte: TP largo (il move ribassista è potente)
    # ══════════════════════════════════════════════════════════

    trend_dir = "UP" if ema9_1h > ema21_1h else "DOWN"
    trade_aligns_with_trend = (
        (trend_dir == "UP" and direction == "LONG") or
        (trend_dir == "DOWN" and direction == "SHORT")
    )

    # Aspettativa = inversione del trend attuale
    # Se il trade è CONTRO il trend → è il trade che segue il trend in corso
    #   = TP stretto (il trend sta per finire)
    # Se il trade è CON il trend atteso dell'inversione → TP largo
    #   = ovvero: controtendenza = il trade che aspettavamo

    if trade_aligns_with_trend:
        # Segui il trend in corso — TP stretto, SL stretto, prendi profitto rapido
        strategy = "FOLLOW"
        tp_mult = 0.7    # TP 30% più stretto
        sl_mult = 0.8    # SL 20% più stretto (esci veloce se gira)
        details += f" [FOLLOW trend:{trend_dir}]"
    else:
        # Controtendenza = possibile inversione — TP largo, trailing, lascia correre
        strategy = "REVERSAL"
        tp_mult = 1.5    # TP 50% più largo
        sl_mult = 1.0    # SL normale (dai spazio all'inversione)
        scalp_mode = "TREND"  # forza trailing attivo
        details += f" [REVERSAL vs trend:{trend_dir}]"

    # ── SL / TP — ATR-based, pulito ──
    update_funding_oi()
    oi_chg = get_oi_change()
    details += f" FZ:{fz:+.1f} OI:{oi_chg:+.2%} [{scalp_mode}]"

    # SL = 1.2× ATR
    # SL = ATR×1.2 — respira col mercato
    # TP1 = ATR×0.8 → chiudi 50% (profitto veloce e sicuro)
    # TP2 = ATR×1.2 → chiudi resto (R:R 1:1)
    # Se il trend continua → il bot rientra al prossimo segnale
    sl_dist = atr5 * 1.2
    tp1_dist = atr5 * 0.8
    tp2_dist = atr5 * 1.2

    # Floor
    sl_dist = max(sl_dist, px * 0.003)
    tp1_dist = max(tp1_dist, px * 0.003)
    tp2_dist = max(tp2_dist, px * 0.005)

    # Expectation strategy
    sl_dist *= sl_mult
    tp1_dist *= tp_mult
    tp2_dist *= tp_mult

    # R:R check: TP2 deve essere almeno 1.5× SL dopo fee
    fee_cost = px * 0.001  # 0.05% × 2 lati
    effective_rr = (tp2_dist - fee_cost) / (sl_dist + fee_cost)
    if effective_rr < 0.8:
        log_btc(f"❌ R:R {effective_rr:.2f} < 1.3 — skip"); return None

    if direction == "LONG":
        sl = px - sl_dist
        tp = px + tp2_dist
        tp1 = px + tp1_dist
    else:
        sl = px + sl_dist
        tp = px - tp2_dist
        tp1 = px - tp1_dist

    sl_pct = sl_dist / px * 100
    tp1_pct = tp1_dist / px * 100
    tp2_pct = tp2_dist / px * 100
    log_btc(f"SL:{sl_pct:.2f}% TP1:{tp1_pct:.2f}%(50%) TP2:{tp2_pct:.2f}%(rest) R:R={effective_rr:.1f} [{strategy}]")

    return (direction, sig_type, sl, tp, px, atr5, details, sl_dist,
            size_mult, regime, 50, scalp_mode, ml_features, tp1)

# ================================================================
# EXECUTION
# ================================================================
_btc_coin_meta = {}

def get_meta():
    """
    Recupera szDecimals e maxLeverage da Hyperliquid.
    priceDecimals non esiste nell'API — calcolato dal tick size.
    """
    global _btc_coin_meta
    for attempt in range(3):
        try:
            m = call(_info.meta, timeout=15)
            for a in m["universe"]:
                if a["name"] == BTC_COIN:
                    sz_dec = int(a["szDecimals"])
                    max_lev = int(a.get("maxLeverage", 50))

                    # Price decimals: BTC su Hyperliquid ha tick size $1 = 0 decimali
                    px_dec = 0  # BTC = numeri interi

                    _btc_coin_meta = {
                        "sz_dec":  sz_dec,
                        "px_dec":  px_dec,
                        "max_lev": max_lev,
                    }
                    log_btc(f"✅ Meta {BTC_COIN}: szDec={sz_dec} pxDec={px_dec} maxLev={max_lev}x")

                    if BTC_LEVERAGE > max_lev:
                        log_btc(f"⚠️ BTC_LEVERAGE {BTC_LEVERAGE}x > max {max_lev}x")
                        tg(f"⚠️ Leva {BTC_LEVERAGE}x > max {max_lev}x per {BTC_COIN}")

                    return sz_dec, px_dec
        except Exception as e:
            if attempt < 2: time.sleep(3 * (attempt + 1))
            else: log_btc(f"get_meta failed: {e}")
    return 5, 0  # BTC defaults: szDec=5, pxDec=0

def get_max_leverage():
    return _btc_coin_meta.get("max_lev", 50)

def get_position():
    global _pos_cache
    if time.time() - _pos_cache["ts"] < API_CACHE_TTL:
        return _pos_cache["value"]
    try:
        s = call(_info.user_state, _account.address, timeout=15)
        for p in s.get("assetPositions", []):
            pp = p["position"]
            if pp["coin"] == BTC_COIN and float(pp["szi"]) != 0:
                result = {
                    "szi": float(pp["szi"]),
                    "entry": float(pp.get("entryPx", 0)),
                    "lev": int(pp.get("leverage", {}).get("value", BTC_LEVERAGE))
                            if isinstance(pp.get("leverage"), dict) else BTC_LEVERAGE
                }
                _pos_cache = {"ts": time.time(), "value": result}
                return result
        _pos_cache = {"ts": time.time(), "value": None}
    except: pass
    return _pos_cache.get("value")

def get_effective_lev():
    try:
        s = call(_info.user_state, _account.address, timeout=10)
        for p in s.get("assetPositions", []):
            pp = p.get("position", {})
            if pp.get("coin") == BTC_COIN:
                lev = pp.get("leverage", {})
                if isinstance(lev, dict):
                    return int(lev.get("value", BTC_LEVERAGE))
                if isinstance(lev, (int, float)):
                    return int(lev)
        return BTC_LEVERAGE
    except:
        return BTC_LEVERAGE

def get_balance():
    global _bal_cache
    if time.time() - _bal_cache["ts"] < API_CACHE_TTL:
        return _bal_cache["value"]
    try:
        s = call(_info.user_state, _account.address, timeout=10)
        v = float(s["marginSummary"]["accountValue"])
        _bal_cache = {"ts": time.time(), "value": v}
        return v
    except: return _bal_cache.get("value", 0)

def get_mid():
    global _mid_cache
    if time.time() - _mid_cache["ts"] < API_CACHE_TTL and _mid_cache["value"] > 0:
        return _mid_cache["value"]
    try:
        mids = call(_info.all_mids, timeout=10)
        v = float(mids.get(BTC_COIN, 0))
        if v > 0:
            _mid_cache = {"ts": time.time(), "value": v}
        return v
    except: return _mid_cache.get("value", 0)

def get_funding():
    """Returns current BTC funding rate (raw, not bps)."""
    try:
        ctx = call(_info.meta_and_asset_ctxs, timeout=15)
        for a, c in zip(ctx[0]["universe"], ctx[1]):
            if a["name"] == BTC_COIN:
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
        margin_required = notional / BTC_LEVERAGE
        # Buffer 5%: margine minimo extra
        margin_with_buffer = margin_required * 1.05

        if available < margin_with_buffer:
            # Se siamo vicini, riduci la size automaticamente
            if available > margin_required * 0.8:
                log_btc(f"⚠️ Margine stretto: need ${margin_with_buffer:.2f} have ${available:.2f} — procedo")
                return True
            log_btc(f"⚠️ Margine insufficiente: need ${margin_with_buffer:.2f} "
                f"have ${available:.2f} (used:${total_margin:.2f} total:${account_value:.2f})")
            return False

        if account_value < 3:
            log_btc(f"⚠️ Account value troppo basso: ${account_value:.2f}")
            return False

        return True
    except Exception as e:
        log_btc(f"check_margin error: {e}")
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
        ob = call(_info.l2_snapshot, BTC_COIN, timeout=10)
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
        log_btc(f"fetch_liquidity: {e}")
    return result

# ================================================================
# SETUP SCORE (from V4 — adapted for BTC-only)
# ================================================================
def btc_compute_setup_score(direction, px, ema50, ema200, rsi, ai_score,
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
def btc_get_recent_fills(since_ts):
    """Recupera fill recenti BTC da API."""
    try:
        fills = call(_info.user_fills_by_time, _account.address,
                     int(since_ts * 1000), timeout=15)
        return [f for f in (fills or []) if f.get("coin") == BTC_COIN]
    except: return []

def check_btc_exit(pos_state):
    """
    Check se la posizione è stata chiusa (SL/TP hit o liquidazione).
    Cerca fill recenti sul lato opposto della posizione.
    Returns: (exited: bool, exit_px: float or None)
    """
    try:
        side = pos_state.get("side", "LONG" if pos_state.get("szi", 0) > 0 else "SHORT")
        ts_open = pos_state.get("open_ts", pos_state.get("entry_time", time.time() - 3600))
        fills = btc_get_recent_fills(ts_open)
        for f in fills:
            fill_side = f.get("side", "")[0:1].upper()
            # Se il fill è sul lato opposto → posizione chiusa
            if (side == "LONG" and fill_side == "A") or \
               (side == "SHORT" and fill_side == "B"):
                exit_px = float(f.get("px", 0))
                if exit_px > 0:
                    return True, exit_px
        return False, None
    except Exception as e:
        log_btc(f"check_exit: {e}")
        return False, None


def btc_compute_real_exit(direction, entry_px, ts_open):
    """Wrapper per compatibilità — usa check_btc_exit internamente."""
    try:
        fills = btc_get_recent_fills(ts_open)
        close_side = "B" if direction == "SHORT" else "A"
        close_fills = [f for f in fills if f.get("side", "")[0:1].upper() == close_side]
        if close_fills:
            total_sz = sum(float(f.get("sz", 0)) for f in close_fills)
            if total_sz > 0:
                exit_px = sum(float(f.get("px", 0)) * float(f.get("sz", 0))
                              for f in close_fills) / total_sz
                pnl_pct = ((exit_px - entry_px) / entry_px if direction == "LONG"
                           else (entry_px - exit_px) / entry_px)
                return exit_px, pnl_pct
    except Exception as e:
        log_btc(f"compute_real_exit: {e}")
    return 0, 0

# ================================================================
# TRIGGER ORDERS & CLEANUP (from V4)
# ================================================================
def get_trigger_orders():
    """Returns SL/TP trigger orders for BTC."""
    try:
        orders = call(_info.open_orders, _account.address, timeout=10)
        return [o for o in (orders or [])
                if o.get("coin") == BTC_COIN
                and o.get("orderType") in ("Stop Market", "Take Profit Market")]
    except: return []

def cancel_trigger_order(oid):
    try:
        res = call(_exchange.cancel, BTC_COIN, oid, timeout=10)
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
        log_btc(f"🧹 {len(triggers)} ordini orfani trovati — cancello")
        for o in triggers:
            oid = o.get("oid")
            if oid:
                cancel_trigger_order(oid)
                log_btc(f"  Cancelled {o.get('orderType')} #{oid}")
        tg(f"🧹 Cancellati {len(triggers)} ordini BTC orfani", silent=True)

# ================================================================
# PENDING ORDERS TRACKING (from V4)
# ================================================================
_btc_pending_order = {}  # {"oid": X, "placed_at": T, "sl": S, "tp": T, "direction": D, "sl_dist": SD}

def set_pending(oid, sl, tp, direction, sl_dist, sig_type="", regime=""):
    """Registra un ordine GTC che non è stato fillato immediatamente."""
    global _btc_pending_order
    _btc_pending_order = {
        "oid": oid, "placed_at": time.time(),
        "sl": sl, "tp": tp, "direction": direction,
        "sl_dist": sl_dist, "type": sig_type, "regime": regime
    }
    _rset("btc6:pending", _btc_pending_order)
    log_btc(f"⏳ Ordine pendente registrato (oid={oid})")

def clear_pending():
    global _btc_pending_order
    _btc_pending_order = {}
    _rset("btc6:pending", None)

def check_pending(sz_dec, px_dec):
    """
    Controlla se un ordine pendente è stato fillato o è scaduto.
    Se fillato → piazza SL/TP.
    Se scaduto → cancella.
    """
    global _btc_pending_order
    if not _btc_pending_order:
        return None

    oid = _btc_pending_order.get("oid")
    placed_at = _btc_pending_order.get("placed_at", 0)
    now = time.time()

    # Check se è stato fillato (posizione aperta)
    pos = get_position()
    if pos is not None:
        direction = _btc_pending_order["direction"]
        is_buy = direction == "LONG"
        sl_px = rpx(_btc_pending_order["sl"], px_dec)
        tp_px = rpx(_btc_pending_order["tp"], px_dec)
        actual_size = rpx(abs(pos["szi"]), sz_dec)

        log_btc(f"✅ Pendente fillato → piazzo SL/TP")
        try:
            call(_exchange.order, BTC_COIN, not is_buy, actual_size, sl_px,
                 {"trigger": {"triggerPx": sl_px, "isMarket": True, "tpsl": "sl"}},
                 True, timeout=15)
            time.sleep(0.3)
            call(_exchange.order, BTC_COIN, not is_buy, actual_size, tp_px,
                 {"trigger": {"triggerPx": tp_px, "isMarket": True, "tpsl": "tp"}},
                 True, timeout=15)

            entry = pos["entry"]
            sl_pct = abs(entry - sl_px) / entry * 100
            tp_pct = abs(tp_px - entry) / entry * 100
            log_btc(f"🔒 Pendente protetto: SL:{sl_px}({sl_pct:.1f}%) TP:{tp_px}({tp_pct:.1f}%)")
            tg(f"🔒 <b>BTC</b> pendente fillato | SL:{sl_px} TP:{tp_px}", silent=True)
        except Exception as e:
            log_btc(f"🚨 SL/TP pendente error: {e}")
            tg(f"🚨 BTC pendente SL/TP ERROR!")

        result = _btc_pending_order.copy()
        result["entry"] = pos["entry"]
        result["szi"] = pos["szi"]
        clear_pending()
        return result

    # Check se è scaduto
    if now - placed_at > PENDING_ORDER_TTL:
        log_btc(f"⏱ Pendente scaduto dopo {PENDING_ORDER_TTL}s — cancello")
        try:
            if oid:
                call(_exchange.cancel, BTC_COIN, oid, timeout=10)
                log_btc(f"  Cancelled GTC #{oid}")
        except Exception as e:
            log_btc(f"  Cancel error: {e}")
        tg(f"⏱ BTC ordine scaduto — cancellato", silent=True)
        clear_pending()
        return None

    # Ancora in attesa
    elapsed = int(now - placed_at)
    if elapsed % 30 == 0:  # log ogni 30s
        log_btc(f"⏳ Pendente in attesa... {elapsed}s/{PENDING_ORDER_TTL}s")
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
        log_btc("🔍 No open position — clean start")
        return None

    entry = pos["entry"]
    szi = pos["szi"]
    d = "LONG" if szi > 0 else "SHORT"
    mid = get_mid()
    pnl_pct = ((mid-entry)/entry if d == "LONG" else (entry-mid)/entry) * 100

    log_btc(f"🔍 Found open position: {d} @ {entry} size:{abs(szi)} PnL:{pnl_pct:+.1f}%")

    # Check if SL exists
    orders = get_open_orders()
    has_sl = any(o.get("coin") == BTC_COIN and o.get("orderType") == "Stop Market" for o in orders)
    has_tp = any(o.get("coin") == BTC_COIN and o.get("orderType") == "Take Profit Market" for o in orders)

    if not has_sl:
        # Emergency SL at 3× SL_MAX_PCT (wide, just protection)
        emergency_dist = entry * 0.01  # 1%
        if d == "LONG":
            sl_px = rpx(entry - emergency_dist, px_dec)
        else:
            sl_px = rpx(entry + emergency_dist, px_dec)

        try:
            size_abs = rpx(abs(szi), sz_dec)
            is_buy = d == "LONG"
            call(_exchange.order, BTC_COIN, not is_buy, size_abs, sl_px,
                 {"trigger": {"triggerPx": sl_px, "isMarket": True, "tpsl": "sl"}},
                 True, timeout=15)
            log_btc(f"🚨 EMERGENCY SL placed @ {sl_px}")
            tg(f"🚨 <b>BTC</b> restart — emergency SL @ {sl_px}")
        except Exception as e:
            log_btc(f"🚨 EMERGENCY SL FAILED: {e}")
            tg(f"🚨🚨 BTC NO SL — MANUAL CHECK!")

    if not has_tp:
        # Piazza TP a 2×ATR (o 2% se ATR non disponibile)
        try:
            df_r = fetch_df("15m", 1)
            atr_r = float(df_r.iloc[-1]['atr']) if df_r is not None and len(df_r) >= 5 else entry * 0.02
        except:
            atr_r = entry * 0.02
        tp_dist = atr_r * 2.0
        size_abs = rpx(abs(szi), sz_dec)
        if d == "LONG":
            tp_px = rpx(entry + tp_dist, px_dec)
        else:
            tp_px = rpx(entry - tp_dist, px_dec)
        try:
            call(_exchange.order, BTC_COIN, d != "LONG", size_abs, tp_px,
                 {"trigger": {"triggerPx": tp_px, "isMarket": True, "tpsl": "tp"}},
                 True, timeout=15)
            log_btc(f"🎯 Recovery TP placed @ {tp_px}")
        except Exception as e:
            log_btc(f"⚠️ Recovery TP failed: {e}")
        tp1_px = rpx(entry + atr_r * 1.2 if d == "LONG" else entry - atr_r * 1.2, px_dec)
    else:
        tp1_px = 0

    pos_state = {
        "coin": BTC_COIN,
        "side": d,
        "entry_px": entry,
        "entry": entry,
        "size": abs(szi),
        "szi": szi,
        "open_ts": time.time(),
        "entry_time": time.time(),
        "sl_px": sl_px if not has_sl else 0,
        "tp_px": tp_px if not has_tp else 0,
        "tp1_px": tp1_px,
        "sl_dist": entry * 0.012,
        "sl_oid": None,
        "tp_oid": None,
        "partial_done": False,
        "max_favorable_px": entry,
        "mode": "TREND",
        "scalp_mode": "TREND",
        "type": "RECOVERED",
        "regime": _btc_regime,
        "trailing_active": False,
        "trailing_moves": 0,
        "current_ts": 0,
        "atr": 0,
        "post_validated": True,
        "close_reason": "",
        "ml_features": [],
        "last_trail_check": 0,
        "lev": pos.get("lev", BTC_LEVERAGE),
    }
    return pos_state

# ================================================================
# TRAILING STOP MECCANICO
# ================================================================
def update_trailing(pos_state, mid, atr, sz_dec, px_dec):
    """
    Trailing stop pulito:
    - Traccia max_favorable_px
    - Attiva a +0.3% di profitto
    - Trail a 0.7×ATR dal massimo
    - Cancel vecchio SL per OID, piazza nuovo
    """
    global _btc_sl_oid
    if atr <= 0:
        return

    is_long = pos_state.get("side", "LONG" if pos_state.get("szi", 0) > 0 else "SHORT") == "LONG"
    entry = pos_state.get("entry_px", pos_state.get("entry", 0))

    # Aggiorna max favorevole
    if is_long:
        pos_state["max_favorable_px"] = max(pos_state.get("max_favorable_px", entry), mid)
    else:
        pos_state["max_favorable_px"] = min(pos_state.get("max_favorable_px", entry), mid)

    # Profitto corrente
    move = (mid - entry) / entry if is_long else (entry - mid) / entry

    # Attiva trailing solo dopo +0.3%
    if move < 0.003:
        return

    if not pos_state.get("trailing_active"):
        pos_state["trailing_active"] = True
        pos_state["trailing_moves"] = 0
        log_btc(f"📈 Trailing ON at {mid:.0f} (+{move:.2%})")

    # Nuovo SL dal max favorevole
    max_px = pos_state["max_favorable_px"]
    if is_long:
        new_sl = rpx(max_px - atr * 0.7, px_dec)
    else:
        new_sl = rpx(max_px + atr * 0.7, px_dec)

    # Mai peggiorare lo SL
    old_sl = pos_state.get("sl_px", 0)
    if is_long and new_sl <= old_sl:
        return
    if not is_long and (new_sl >= old_sl and old_sl > 0):
        return

    # Cancel TUTTI gli SL esistenti prima di piazzarne uno nuovo
    try:
        # Prima: cancel per OID se disponibile
        cancelled = False
        if _btc_sl_oid:
            try:
                call(_exchange.cancel, BTC_COIN, _btc_sl_oid, timeout=10)
                cancelled = True
            except:
                pass

        # Poi: scan e cancel tutti gli Stop Market rimasti (cleanup)
        try:
            orders = get_open_orders()
            for o in (orders or []):
                if o.get("coin") == BTC_COIN and o.get("orderType") == "Stop Market":
                    try:
                        call(_exchange.cancel, BTC_COIN, o["oid"], timeout=10)
                        cancelled = True
                    except:
                        pass
        except:
            pass

        if not cancelled:
            log_btc(f"⚠️ Trailing: nessun SL cancellato — NON piazzo nuovo")
            return

        time.sleep(0.3)

        size_abs = rpx(pos_state.get("size", abs(pos_state.get("szi", 0))), sz_dec)
        sl_res = call(_exchange.order, BTC_COIN, not is_long, size_abs, new_sl,
                     {"trigger": {"triggerPx": new_sl, "isMarket": True, "tpsl": "sl"}},
                     True, timeout=15)
        if sl_res and sl_res.get("status") == "ok":
            for s in sl_res.get("response", {}).get("data", {}).get("statuses", []):
                if "resting" in s:
                    _btc_sl_oid = s["resting"]["oid"]

        pos_state["sl_px"] = new_sl
        pos_state["sl_oid"] = _btc_sl_oid
        pos_state["trailing_moves"] = pos_state.get("trailing_moves", 0) + 1
        log_btc(f"📈 Trailing SL → {new_sl} (max:{max_px:.0f} trail:{atr*0.7:.0f}) [#{pos_state['trailing_moves']}]")
        save_pos_state(pos_state)
    except Exception as e:
        log_btc(f"Trailing error: {e}")

# ================================================================
# PARTIAL CLOSE
# ================================================================
def btc_check_partial_close(pos_state, mid, sz_dec, px_dec):
    """
    TP1 = ATR×1.0 → chiudi 50%. Il resto corre fino a TP2 (exchange) o trailing.
    """
    if pos_state.get("partial_done"):
        return

    is_long = pos_state.get("side", "LONG" if pos_state.get("szi", 0) > 0 else "SHORT") == "LONG"
    tp1 = pos_state.get("tp1_px", 0)

    if tp1 <= 0:
        return

    # TP1 raggiunto?
    hit = (is_long and mid >= tp1) or (not is_long and mid <= tp1)
    if not hit:
        return

    size = pos_state.get("size", abs(pos_state.get("szi", 0)))
    close_size = rpx(size * 0.5, sz_dec)
    if close_size <= 0:
        return

    entry = pos_state.get("entry_px", pos_state.get("entry", 0))
    pnl = (mid - entry) / entry if is_long else (entry - mid) / entry

    try:
        btc_market_close(pos_state.get("side", "LONG"), close_size, mid, sz_dec, px_dec)
        pos_state["partial_done"] = True
        pos_state["size"] = rpx(size * 0.5, sz_dec)
        log_btc(f"💰 TP1 HIT — 50% closed @ {mid:.0f} (+{pnl:.2%}) — rest runs to TP2")
        tg(f"💰 <b>BTC TP1</b> 50% +{pnl:.2%}", silent=True)
        save_pos_state(pos_state)
    except Exception as e:
        log_btc(f"TP1 close error: {e}")

# ================================================================
# FUNDING CHECK
# ================================================================
def is_funding_ok(direction):
    """
    Blocca entry se funding contro la posizione.
    Usa sia il raw funding rate che il z-score.
    Usa sia il raw funding rate che il z-score.
    """
    funding = get_funding()
    fz = get_funding_z()

    # Raw funding check
    if direction == "LONG" and funding > FUNDING_BLOCK_THRESH:
        log_btc(f"⚠️ Funding {funding*100:.3f}% contro LONG — skip")
        return False
    if direction == "SHORT" and funding < -FUNDING_BLOCK_THRESH:
        log_btc(f"⚠️ Funding {funding*100:.3f}% contro SHORT — skip")
        return False

    # Z-score check: funding estremo = crowded = rischio squeeze
    if direction == "LONG" and fz > 2.5:
        log_btc(f"⚠️ Funding z-score {fz:+.1f} — crowded long, skip")
        return False
    if direction == "SHORT" and fz < -2.5:
        log_btc(f"⚠️ Funding z-score {fz:+.1f} — crowded short, skip")
        return False

    return True

# ================================================================
# DAILY REPORT
# ================================================================
_btc_last_report_ts = 0

def maybe_send_daily_report():
    global _btc_last_report_ts
    now = time.time()
    if now - _btc_last_report_ts < 82800:  # max once per 23h
        return

    utc_hour = datetime.now(timezone.utc).hour
    if utc_hour != DAILY_REPORT_HOUR:
        return

    _btc_last_report_ts = now
    trades = [t for t in _btc_trades_today if now - t.get("ts_close", t.get("ts", 0)) < 86400]
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
    # V7.2: Sentiment + ML stats for report
    ml_acc = _ml_model.get_accuracy()
    ml_n = _ml_model.n_samples
    sent = get_sentiment_score()
    sd = get_sentiment_detail()
    lc_info = f"CG_Up:{sd.get('cg_sent_up','off')}% Reddit:+{sd.get('reddit_bullish','?')}/-{sd.get('reddit_bearish','?')}"
    cc_info = f"Bull:{sd.get('cc_bull_pct','off')}%"
    report = (
        f"📊 <b>BTC Scalper V7.2 Daily Report</b>\n"
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
        f"🤖 <b>V7.2 Analytics</b>\n"
        f"  ML: {ml_n} samples, acc:{ml_acc:.0%}\n"
        f"  Sentiment: {sent}/100\n"
        f"  🌐 Social: {lc_info}\n"
        f"  📊 CryptoCompare: {cc_info}\n"
        f"  FGI: {sd.get('fgi','?')} | Funding: {sd.get('funding_sent','?')}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"💼 Balance: ${bal:.2f}"
    )
    tg(report)
    log_btc(f"📊 Daily report sent: {len(trades)} trades ${total:+.2f}")

def btc_open_trade(direction, sl, tp, entry_px, sl_dist, sz_dec, px_dec, size_mult=1.0, scalp_mode="TREND"):
    """
    Entry pulita: prezzo aggressivo → wait 2s → SL/TP da ATR → return pos_state o False
    """
    global _btc_last_trade_ts, _btc_is_trading, _btc_sl_oid, _btc_tp_oid

    if _btc_is_trading:
        log_btc("⚠️ Trade already in progress")
        return False
    _btc_is_trading = True

    try:
        is_long = direction == "LONG"

        # ── SIZE: fisso $5 notional ──
        BTC_MARGIN = 5.0
        notional = BTC_MARGIN * BTC_LEVERAGE  # $5 margin × 5x = $25 notional
        
        bal = get_balance()
        if notional / BTC_LEVERAGE > bal * 0.9:
            log_btc(f"❌ Balance ${bal:.2f} insufficiente per ${notional} notional")
            return False
        size = rpx(notional / entry_px, sz_dec)
        if size <= 0:
            log_btc(f"Size zero: notional=${notional:.0f}"); return False

        # ── LEVERAGE ──
        try: call(_exchange.update_leverage, min(BTC_LEVERAGE, get_max_leverage()), BTC_COIN, is_cross=False, timeout=10)
        except: pass

        # ── ENTRY: prezzo aggressivo ──
        mid = get_mid()
        if mid <= 0:
            log_btc("❌ get_mid() failed"); return False
        px = rpx(mid * (1.0003 if is_long else 0.9997), px_dec)

        log_btc(f"{'🟢' if is_long else '🔴'} ORDER {direction} [{scalp_mode}] @ {px} size:{size}")

        # ── PLACE ORDER + WAIT 2s ──
        res = call(_exchange.order, BTC_COIN, is_long, size, px,
                   {"limit": {"tif": "Gtc"}}, False, timeout=15)

        oid = None
        filled = False
        if res and res.get("status") == "ok":
            for s in res.get("response", {}).get("data", {}).get("statuses", []):
                if "filled" in s:
                    filled = True; break
                if "resting" in s:
                    oid = s["resting"]["oid"]

        if not filled and oid:
            for _ in range(4):
                time.sleep(0.5)
                p = get_position()
                if p and abs(p.get("szi", 0)) > 0:
                    filled = True; break
            if not filled:
                try: call(_exchange.cancel, BTC_COIN, oid, timeout=10)
                except: pass
                log_btc(f"❌ Not filled in 2s — cancelled"); return False

        if not filled:
            log_btc(f"❌ Order failed — res: {res}"); return False

        # ── GET FILL PRICE ──
        pos = get_position()
        if not pos:
            time.sleep(0.5)
            pos = get_position()
        if not pos:
            log_btc("❌ Position not found after fill"); return False

        entry_real = pos["entry"]
        size_real = rpx(abs(pos["szi"]), sz_dec)
        atr = sl_dist / 1.2  # reverse ATR from sl_dist

        # ── SL/TP da ATR sul prezzo reale di fill ──
        if is_long:
            sl_px = rpx(entry_real - atr * 1.2, px_dec)
            tp_px = rpx(entry_real + atr * 1.8, px_dec)
        else:
            sl_px = rpx(entry_real + atr * 1.2, px_dec)
            tp_px = rpx(entry_real - atr * 1.8, px_dec)

        # ── PLACE SL ──
        _btc_sl_oid = None
        _btc_tp_oid = None
        try:
            # SL sull'exchange (safety net)
            sl_res = call(_exchange.order, BTC_COIN, not is_long, size_real, sl_px,
                         {"trigger": {"triggerPx": sl_px, "isMarket": True, "tpsl": "sl"}},
                         True, timeout=15)
            if sl_res and sl_res.get("status") == "ok":
                for s in sl_res.get("response", {}).get("data", {}).get("statuses", []):
                    if "resting" in s: _btc_sl_oid = s["resting"]["oid"]
            time.sleep(0.2)
            # TP sull'exchange (prende profitto automatico)
            tp_res = call(_exchange.order, BTC_COIN, not is_long, size_real, tp_px,
                         {"trigger": {"triggerPx": tp_px, "isMarket": True, "tpsl": "tp"}},
                         True, timeout=15)
            if tp_res and tp_res.get("status") == "ok":
                for s in tp_res.get("response", {}).get("data", {}).get("statuses", []):
                    if "resting" in s: _btc_tp_oid = s["resting"]["oid"]
        except Exception as e:
            log_btc(f"🚨 SL/TP ERROR: {e}")

        _btc_last_trade_ts = time.time()
        sl_pct = abs(entry_real - sl_px) / entry_real * 100
        tp_pct = abs(tp_px - entry_real) / entry_real * 100
        log_btc(f"✅ FILLED @ {entry_real} size:{size_real} SL:{sl_px}({sl_pct:.2f}%) TP:{tp_px}({tp_pct:.2f}%)")
        tg(f"{'🟢' if is_long else '🔴'} <b>BTC {direction}</b> @ {entry_real}\n"
           f"SL:{sl_px} ({sl_pct:.1f}%) | TP:{tp_px} ({tp_pct:.1f}%)\nSize:{size_real}")
        return True

    except Exception as e:
        log_btc(f"Execute error: {e}")
        import traceback; traceback.print_exc()
        return False
    finally:
        _btc_is_trading = False


def btc_market_close(side, size, mid, sz_dec, px_dec):
    """
    Chiudi posizione con IoC aggressivo (simula market order).
    Prezzo 0.5% sfavorevole = fill garantito.
    """
    is_long = side == "LONG"
    # Prezzo aggressivo: vendi basso, compra alto
    close_px = rpx(mid * (0.995 if is_long else 1.005), px_dec)
    size_abs = rpx(size, sz_dec)
    try:
        res = call(_exchange.order, BTC_COIN, not is_long, size_abs, close_px,
                   {"limit": {"tif": "Ioc"}}, False, timeout=15)
        return res and res.get("status") == "ok"
    except Exception as e:
        log_btc(f"Market close error: {e}")
        return False


# ================================================================
# THREAD A — SCANNER (regime + backtest ogni 5 min)
# ================================================================
_btc_scanner_ready = threading.Event()

def scanner_thread():
    log_btc("[SCAN] BTC Scanner avviato")
    ml_load_model()
    while True:
        try:
            regime = update_regime()
            update_funding_oi()
            fz = get_funding_z()
            oi = get_oi_change()
            run_backtest()
            mid = get_mid()
            bal = get_balance()

            flow_data = update_order_flow()
            sent_score = get_sentiment_score()
            flow_sig = get_flow_signal()

            adx_val = 20
            try:
                df_1h = fetch_df("1h", 5)
                if df_1h is not None and len(df_1h) > 5 and 'adx' in df_1h.columns:
                    adx_val = float(df_1h.iloc[-1]['adx'])
            except: pass
            mode = "FLASH" if abs(fz) > 2.5 else "TREND" if adx_val > 25 and regime in ("BULL","BEAR") else "RANGE"

            log_btc(f"════ Regime:{regime} ADX:{adx_val:.0f}→{mode} BTC ${mid:,.0f} ════")
            for k, v in _btc_bt_results.items():
                pf = v.get("pf", 0); wr = v.get("wr", 0); n = v.get("n", 0)
                st = "✅" if pf >= 1.0 else "⚠️" if pf >= 0.8 else "❌"
                log_btc(f"  {st} {k:<18} PF:{pf:.2f} WR:{wr:.0%} N:{n}")
            log_btc(f"  FZ:{fz:+.1f} OI:{oi:+.2%} Bal:${bal:.2f}")
            sent_detail = get_sentiment_detail()
            log_btc(f"  Sent:{sent_score} Flow:{flow_sig['bias']}({flow_sig['confidence']}) ML:{_ml_model.n_samples}samples")

            # Fleet: publish regime + bias for ALT engine (internal)
            fleet_set_btc_regime(regime)
            bias = compute_hourly_bias()
            log_btc(f"  Fleet bias: {bias} | Regime: {regime}")

        except Exception as e:
            log_btc(f"Scanner error: {e}")
            import traceback; traceback.print_exc()

        if not _btc_scanner_ready.is_set():
            _btc_scanner_ready.set()
            log_btc("Scanner ready")

        for i in range(6):
            time.sleep(10)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] [BTC] wait {(i+1)*10}s/60s", flush=True)


# ================================================================
# BTC THREAD B — PROCESSOR (check signal ogni 10s)
# ================================================================
def processor_thread(sz_dec, px_dec):
    """
    BTC Processor V7.3 — DETECT + EXECUTE in un unico ciclo.
    
    Quando trova un segnale, lo esegue IMMEDIATAMENTE (no publish → wait → read).
    Latenza: check_signal (~8s) → execute (~3s) = ~11s totale (era ~40s).
    
    Il position management (trailing, partial, AI) resta nel btc_executor_loop.
    """
    global _btc_current_signal, _btc_last_trade_ts, _btc_is_trading

    log_btc("Processor avviato — attendo Scanner...")
    _btc_scanner_ready.wait()
    log_btc("Scanner pronto — avvio (detect+execute mode)")

    while True:
        try:
            pos = get_position()
            mid = get_mid()

            # Se in posizione → il btc_executor_loop gestisce trailing/partial
            if pos is not None:
                time.sleep(BTC_SCAN_INTERVAL)
                continue

            # Kill switch / circuit breaker check
            ks, ks_reason = fleet_check_kill_switch()
            if ks:
                time.sleep(BTC_SCAN_INTERVAL)
                continue

            cb, cb_reason = check_circuit_breaker()
            if cb:
                time.sleep(BTC_SCAN_INTERVAL)
                continue

            if time.time() - _btc_last_trade_ts < BTC_COOLDOWN_SEC:
                time.sleep(BTC_SCAN_INTERVAL)
                continue

            if _btc_is_trading:
                time.sleep(BTC_SCAN_INTERVAL)
                continue

            # ── KILL SWITCH: daily loss limit ──
            global _btc_start_balance, _btc_kill_switch
            bal = get_balance()
            if _btc_start_balance is None:
                _btc_start_balance = bal
                log_btc(f"📊 Start balance: ${bal:.2f}")

            if bal > 0 and _btc_start_balance > 0:
                daily_loss_pct = (1 - bal / _btc_start_balance) * 100
                if daily_loss_pct >= MAX_DAILY_LOSS_PCT:
                    if not _btc_kill_switch:
                        _btc_kill_switch = True
                        log_btc(f"🛑 KILL SWITCH — daily loss {daily_loss_pct:.1f}% >= {MAX_DAILY_LOSS_PCT}%")
                        tg(f"🛑 <b>KILL SWITCH</b>\nDaily loss: {daily_loss_pct:.1f}%\nStart: ${_btc_start_balance:.2f} → Now: ${bal:.2f}")

            if _btc_kill_switch:
                time.sleep(300)  # check ogni 5 min se il giorno è cambiato
                # Reset a mezzanotte UTC
                if datetime.now(timezone.utc).hour == 0 and datetime.now(timezone.utc).minute < 6:
                    _btc_kill_switch = False
                    _btc_start_balance = bal
                    log_btc(f"📊 Kill switch reset — new day, balance: ${bal:.2f}")
                continue

            # ── DETECT ──
            sig_ts = time.time()
            sig = check_signal()
            if sig is None:
                log_btc(f"no signal | {_btc_regime} | BTC ${mid:,.0f}")
                time.sleep(BTC_SCAN_INTERVAL)
                continue

            direction, sig_type, sl, tp, entry_px, atr, details, sl_dist, size_mult, sig_regime, setup, scalp_mode, ml_features, tp1 = sig

            # Blocca entry vecchie — se check_signal ha impiegato >5s il prezzo è stale
            if time.time() - sig_ts > 25:
                log_btc(f"⚠️ Signal stale ({time.time()-sig_ts:.1f}s) — skip")
                continue

            if not is_funding_ok(direction):
                time.sleep(BTC_SCAN_INTERVAL)
                continue

            # ── PRICE CONFIRMATION: aspetta 15s, verifica che il prezzo confermi ──
            price_at_signal = get_mid()
            log_btc(f"📡 Signal {direction} {sig_type} @ {price_at_signal:,.0f} — waiting 8s for confirmation...")
            time.sleep(8)
            price_after = get_mid()

            if price_after <= 0:
                log_btc(f"⚠️ Confirmation failed — no price")
                continue

            if direction == "LONG" and price_after < price_at_signal:
                log_btc(f"❌ LONG not confirmed — price dropped {price_at_signal:,.0f}→{price_after:,.0f}")
                continue
            elif direction == "SHORT" and price_after > price_at_signal:
                log_btc(f"❌ SHORT not confirmed — price rose {price_at_signal:,.0f}→{price_after:,.0f}")
                continue

            log_btc(f"✅ Confirmed: {direction} price {price_at_signal:,.0f}→{price_after:,.0f}")

            # ── FLOW CONTRADICTION CHECK: non entrare contro il flow ──
            flow_now = get_flow_signal()
            if direction == "LONG" and flow_now.get("bias") == "SELL":
                log_btc(f"❌ LONG vs Flow SELL — skip")
                continue
            if direction == "SHORT" and flow_now.get("bias") == "BUY":
                log_btc(f"❌ SHORT vs Flow BUY — skip")
                continue

            # ── SENTIMENT CONTRARIAN CHECK ──
            sent_score_now = get_sentiment_score()
            if direction == "LONG" and sent_score_now > 70:
                log_btc(f"❌ LONG ma sentiment {sent_score_now} > 70 (troppo greed) — skip")
                continue
            if direction == "SHORT" and sent_score_now < 30:
                log_btc(f"❌ SHORT ma sentiment {sent_score_now} < 30 (troppo fear) — skip")
                continue

            # ── ML FILTER: blocca se ha dati sufficienti e P(win) bassa ──
            if ml_features:
                ml_result = ml_predict_signal(ml_features)
                if ml_result.get("block"):
                    log_btc(f"🤖 ML BLOCK: P(win)={ml_result['prob']:.0%} < 40% ({ml_result['n_samples']} samples, acc:{ml_result['acc']:.0%})")
                    continue

            # ── MAX TRADES PER HOUR ──
            now = time.time()
            trades_last_hour = len([t for t in _btc_trades_today
                                    if now - t.get("ts_close", t.get("ts", 0)) < 3600])
            if trades_last_hour >= MAX_TRADES_PER_HOUR:
                log_btc(f"⏸️ {trades_last_hour} trades nell'ultima ora (max {MAX_TRADES_PER_HOUR}) — skip")
                continue

            # ── EXECUTE ──
            log_btc(f"⚡ DETECT+EXEC {direction} {sig_type} [{scalp_mode}] @ {entry_px:,.0f}")

            success = btc_open_trade(direction, sl, tp, entry_px,
                                     sl_dist, sz_dec, px_dec, size_mult, scalp_mode)
            if success:
                p = get_position()
                if p:
                    pos_state = {
                        "coin": BTC_COIN,
                        "side": direction,
                        "entry_px": p["entry"],
                        "size": abs(p["szi"]),
                        "szi": p["szi"],
                        "entry": p["entry"],
                        "entry_time": time.time(),
                        "open_ts": time.time(),
                        "sl_px": sl,
                        "tp_px": tp,       # TP2 = ATR×1.8 (exchange)
                        "tp1_px": tp1,     # TP1 = ATR×1.0 (close 50%)
                        "sl_dist": sl_dist,
                        "sl_oid": _btc_sl_oid,
                        "tp_oid": _btc_tp_oid,
                        "partial_done": False,
                        "max_favorable_px": p["entry"],
                        "mode": scalp_mode,
                        "scalp_mode": scalp_mode,
                        "type": sig_type,
                        "regime": sig_regime,
                        "trailing_active": False,
                        "trailing_moves": 0,
                        "current_ts": 0,
                        "atr": 0,
                        "post_validated": False,
                        "close_reason": "",
                        "ml_features": ml_features,
                        "last_trail_check": 0,
                        "lev": p.get("lev", BTC_LEVERAGE),
                    }
                    save_pos_state(pos_state)
                    _btc_current_signal = {"filled": True, "pos_state": pos_state}
                    sl_pct = sl_dist / entry_px * 100
                    log_btc(f"✅ FILLED SL:{sl:,.0f}({sl_pct:.2f}%) TP:{tp:,.0f}")
            else:
                _btc_last_trade_ts = time.time()
                log_btc(f"❌ Trade fallito — cooldown {BTC_COOLDOWN_SEC}s")

        except Exception as e:
            log_btc(f"Processor error: {e}")
            import traceback; traceback.print_exc()

        time.sleep(BTC_SCAN_INTERVAL)


# ================================================================
# ALTCOIN ENGINE (from combined_bot.py)
# ================================================================

# ALT global stores
_state_lock = threading.Lock()
_scanner_lock = threading.Lock()
_alt_scanner_ready = threading.Event()
COIN_BLACKLIST = {"kPEPE", "kBONK", "kSHIB", "kFLOKI", "NEIRO", "LUCE", "GOAT"}  # coins to never trade
pending_orders = {}  # {coin: {oid, placed_at, signal, ...}}
last_trade_time = {}  # {coin: timestamp}
BTC_ETH_COINS = {"BTC", "ETH"}  # high-cap coins with different signal_max_age
_signals_store = {}
_candidates_store = {}
_cooldown_store = {}
_all_mids_cache = {}
_active_positions_cache = {}
_scanner_candidates = []
_alt_funding_data = {}  # {coin: {"rate": float, "history": [...], "oi": float, "oi_prev": float}}
_funding_history = {}   # {coin: [rate1, rate2, ...]}
_funding_cache = {}     # {coin: current_rate}
_oi_cache = {}      # {coin: current_oi}
_oi_prev_cache = {}     # {coin: previous_oi}
_alt_trade_history = []
_alt_daily_pnl = 0.0
_alt_consec_losses = 0
ALT_SIGNAL_MAX_AGE = 3 * 60
SIGNAL_MAX_AGE = ALT_SIGNAL_MAX_AGE  # alias for ALT engine

# ── Carica stato da Redis all'avvio ─────────────────────────────

def load_state_from_redis():
    global _signals_store, _candidates_store, _cooldown_store
    now = time.time()

    signals = _rget("state:signals") or {}
    # Filtra segnali scaduti
    signals = {k: v for k, v in signals.items()
               if (now - v.get("ts", 0)) < v.get("signal_max_age", SIGNAL_MAX_AGE)}

    candidates = _rget("state:candidates") or {}
    # Filtra candidati scaduti (TTL: 3 cicli processor = 15 min)
    candidates = {k: v for k, v in candidates.items()
                  if (now - v.get("ts", 0)) < PROCESSOR_INTERVAL * 3}

    cooldowns_raw = _rget("state:cooldowns") or {}
    # Compatibilità: converti vecchio formato {coin: timestamp} al nuovo {coin: {ts, strategy}}
    cooldowns = {}
    for k, v in cooldowns_raw.items():
        if isinstance(v, dict):
            cooldowns[k] = v
        else:
            cooldowns[k] = {"ts": float(v), "strategy": "MEAN_REV"}

    with _state_lock:
        _signals_store    = signals
        _candidates_store = candidates
        _cooldown_store   = cooldowns

    log("MAIN", f"Stato caricato da Redis: {len(signals)} segnali, {len(candidates)} candidati, {len(cooldowns)} cooldown")

def _persist_signals():
    with _state_lock:
        data = dict(_signals_store)
    _rset("state:signals", data)

def _persist_candidates():
    with _state_lock:
        data = dict(_candidates_store)
    _rset("state:candidates", data)

def _persist_cooldowns():
    with _state_lock:
        data = dict(_cooldown_store)
    _rset("state:cooldowns", data)

# ── API pubblica (in-memory + persist) ──────────────────────────

def get_all_signals() -> dict:
    with _state_lock:
        return dict(_signals_store)

def delete_signal(coin: str):
    with _state_lock:
        _signals_store.pop(coin, None)
    _persist_signals()

def publish_signal(coin: str, data: dict):
    with _state_lock:
        _signals_store[coin] = data
    _persist_signals()

def get_candidate(coin: str) -> dict:
    with _state_lock:
        return dict(_candidates_store.get(coin, {}))

def set_candidate(coin: str, data: dict):
    with _state_lock:
        _candidates_store[coin] = data
    _persist_candidates()

def clear_candidate(coin: str):
    with _state_lock:
        _candidates_store.pop(coin, None)
    _persist_candidates()

def is_cooldown_ok(coin: str, strategy: str = "MEAN_REV") -> bool:
    with _state_lock:
        entry = _cooldown_store.get(coin, {})
    if not entry:
        return True
    cooldown = COIN_COOLDOWN_TREND if entry.get("strategy") == "TREND" else COIN_COOLDOWN_MR
    return (time.time() - entry.get("ts", 0)) > cooldown

def set_cooldown(coin: str, strategy: str = "MEAN_REV"):
    with _state_lock:
        _cooldown_store[coin] = {"ts": time.time(), "strategy": strategy}
    _persist_cooldowns()


# ── Trade History (buffer rotante max 50 trade) ──────────────────

TRADE_HISTORY_MAX = 50

def load_trade_history() -> list:
    return _rget("state:trade_history") or []

def save_trade_outcome(trade: dict):
    """
    Aggiunge trade completato alla history.
    Campi: coin, direction, strategy, regime, rsi, bb_pos, funding_z,
           ai_score, setup_score, entry_px, exit_px, sl, tp,
           outcome (win/loss/unknown), pnl_pct, ts_open, ts_close
    """
    try:
        history = load_trade_history()
        history.append(trade)
        if len(history) > TRADE_HISTORY_MAX:
            history = history[-TRADE_HISTORY_MAX:]
        _rset("state:trade_history", history)
    except Exception as e:
        log_err(f"save_trade_outcome: {e}")


def is_circuit_breaker_active() -> tuple[bool, str]:
    """
    Circuit breaker: blocca nuovi trade se:
    1. Le perdite nelle ultime 24h superano MAX_DAILY_LOSS_PCT del capitale
    2. Ci sono MAX_CONSECUTIVE_LOSSES stop loss consecutivi nelle ultime 6h
       (trade più vecchi di 6h non contano — evita blocco da storia stale)

    Ritorna (active: bool, reason: str)
    """
    try:
        history = load_trade_history()
        if not history:
            return False, ""

        now = time.time()

        # 1. Perdite giornaliere (ultime 24h)
        today_trades = [t for t in history if t.get("ts_close", 0) > now - 86400]
        if today_trades:
            daily_pnl = sum(t.get("pnl_pct", 0) for t in today_trades)
            if daily_pnl <= -MAX_DAILY_LOSS_PCT:
                return True, f"daily loss {daily_pnl:.1f}% > max {MAX_DAILY_LOSS_PCT}%"

        # 2. Stop loss consecutivi — SOLO trade delle ultime 6h
        recent_6h = [t for t in history if t.get("ts_close", 0) > now - 21600]
        if len(recent_6h) >= MAX_CONSECUTIVE_LOSSES:
            last_n = recent_6h[-MAX_CONSECUTIVE_LOSSES:]
            if all(t.get("outcome") == "loss" for t in last_n):
                # Pausa 30 minuti dall'ultimo loss
                last_loss_ts = max(t.get("ts_close", 0) for t in last_n)
                if now - last_loss_ts < 1800:
                    mins_left = int((1800 - (now - last_loss_ts)) / 60)
                    return True, f"{MAX_CONSECUTIVE_LOSSES} stop loss consecutivi, pausa {mins_left}min"

        return False, ""
    except Exception as e:
        log_err(f"circuit_breaker check: {e}")
        return False, ""


# ================================================================
# PROCESSOR — FUNDING + OI
# ================================================================

def fetch_funding_and_oi(ctxs=None, meta_assets=None):
    global _funding_cache, _oi_cache, _oi_prev_cache
    try:
        if ctxs is None or meta_assets is None:
            ctx_data    = call(_info.meta_and_asset_ctxs,
                               timeout=API_TIMEOUT_SEC, label='meta_ctxs')
            meta_assets = ctx_data[0]["universe"]
            ctxs        = ctx_data[1]
        _funding_cache = {}
        _oi_prev_cache = _oi_cache.copy()
        _oi_cache      = {}
        for asset, ctx in zip(meta_assets, ctxs):
            try:
                name    = asset["name"]
                funding = float(ctx.get("funding", 0) or 0)
                _funding_cache[name] = funding
                _oi_cache[name]      = float(ctx.get("openInterest", 0) or 0)
                if name not in _funding_history:
                    _funding_history[name] = []
                _funding_history[name].append(funding)
                if len(_funding_history[name]) > FUNDING_HISTORY_LEN:
                    _funding_history[name].pop(0)
            except Exception:
                _funding_cache[asset["name"]] = 0.0
                _oi_cache[asset["name"]]      = 0.0
        log_alt(f"Funding+OI: {len(_funding_cache)} coin")
    except Exception as e:
        log_err(f"fetch_funding_and_oi: {e}")

def alt_get_oi_change(coin):
    curr = _oi_cache.get(coin, 0.0)
    prev = _oi_prev_cache.get(coin, 0.0)
    if prev == 0.0:
        return 0.0
    return (curr - prev) / (prev + 1e-10)

def alt_get_funding_z(coin):
    history = _funding_history.get(coin, [])
    if len(history) < 3:
        return 0.0
    arr = np.array(history)
    std = arr.std()
    if std < 1e-10:
        return 0.0
    return float((history[-1] - arr.mean()) / std)

def alt_inject_funding(df, coin):
    df = df.copy()
    df['funding_z'] = alt_get_funding_z(coin)
    df['oi_change'] = alt_get_oi_change(coin)
    return df


# ================================================================
# PROCESSOR — FEATURE ENGINE
# ================================================================

def build_features(df_input: pd.DataFrame) -> pd.DataFrame:
    df    = df_input.copy()
    close = df['close']

    df['ema20']       = close.ewm(span=20,  min_periods=20).mean()
    df['ema50']       = close.ewm(span=50,  min_periods=50).mean()
    df['ema200']      = close.ewm(span=200, min_periods=200).mean()
    df['ema20_slope'] = df['ema20'].pct_change(5)

    delta = close.diff()
    up    = delta.clip(lower=0).ewm(com=13, min_periods=13).mean()
    down  = -delta.clip(upper=0).ewm(com=13, min_periods=13).mean()
    df['rsi'] = 100 - (100 / (1 + (up / (down + 1e-10))))

    tr = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - close.shift()),
        abs(df['low']  - close.shift())
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14, min_periods=14).mean()

    df['vol_rel']     = df['volume'] / (df['volume'].rolling(96, min_periods=48).mean() + 1e-10)
    df['vol_trend']   = df['volume'].rolling(5).mean() / (df['volume'].rolling(20).mean() + 1e-10)
    df['momentum_1']  = close.pct_change(1)
    df['momentum_4']  = close.pct_change(4)
    df['momentum_12'] = close.pct_change(12)
    df['dist_ema20']  = (close - df['ema20'])  / (df['ema20']  + 1e-10)
    df['dist_ema50']  = (close - df['ema50'])  / (df['ema50']  + 1e-10)
    df['dist_ema200'] = (close - df['ema200']) / (df['ema200'] + 1e-10)
    df['volatility']  = df['atr'] / (close + 1e-10)

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df['bb_width']   = (bb_std * 2) / (bb_mid + 1e-10)
    df['bb_pos']     = (close - bb_mid) / (bb_std * 2 + 1e-10)
    df['body_ratio'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-10)
    df['hh']         = (df['high'] > df['high'].rolling(5).max().shift(1)).astype(int)
    df['ll']         = (df['low']  < df['low'].rolling(5).min().shift(1)).astype(int)
    df['funding_z']  = 0.0
    df['oi_change']  = 0.0

    # MACD: differenza tra EMA12 e EMA26, signal line EMA9 del MACD
    ema12           = close.ewm(span=12, min_periods=12).mean()
    ema26           = close.ewm(span=26, min_periods=26).mean()
    df['macd']      = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, min_periods=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # ADX — Average Directional Index (trend strength, from V7)
    plus_dm = df['high'].diff()
    minus_dm = -df['low'].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    atr14 = tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / (atr14 + 1e-10))
    minus_di = 100 * (minus_dm.rolling(14).mean() / (atr14 + 1e-10))
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    df['adx'] = dx.rolling(14).mean()

    return df.dropna(subset=['rsi', 'vol_rel', 'atr', 'volatility', 'bb_pos', 'ema200', 'ema20_slope'])


# ================================================================
# PROCESSOR — SCORE + SL/TP + REGIME
# ================================================================

def detect_trading_mode(df: pd.DataFrame, regime: str, vol_rel: float) -> str:
    """
    Decide la modalità operativa in base a regime + BB width + volume.

    SCALPING  → mercato laterale/compresso, movimenti 3-5%
    SWING     → mercato in trend, movimenti 20-30%
    """
    if df is None or len(df) < 20:
        return "SCALPING"

    bb_width    = float(df['bb_width'].iloc[-1])
    bb_width_ma = float(df['bb_width'].rolling(20).mean().iloc[-1])
    expanding   = bb_width > bb_width_ma * 1.2   # BB in espansione = breakout in corso
    compressed  = bb_width < bb_width_ma * 0.8   # BB compressa = ranging

    if regime in ("BULL", "BEAR") and expanding and vol_rel >= 2.0:
        return "SWING"
    if regime == "RANGE" or compressed:
        return "SCALPING"
    # Default: SCALPING se incerto
    return "SCALPING"


def dynamic_sl_tp(px, atr, volatility, direction, df: pd.DataFrame = None,
                  coin: str = "", ai_score: int = 5, mode: str = "SCALPING",
                  effective_leverage: int = LEVERAGE):
    """
    SL basato su swing low/high reali (ultimi 5-10 periodi) + buffer 0.2%.
    Fallback su ATR se swing non trovato entro distanza ragionevole.

    APPROCCIO:
    1. SL scala con leva piena (protegge dal noise)
    2. TP calcolato come R:R rispetto allo SL (non moltiplicatore ATR indipendente)
       - SCALPING: R:R 1:1.2 → TP = SL × 1.2 (target realistico, alta WR)
       - SWING:    R:R 1:2.5 → TP = SL × 2.5 (target ambizioso, bassa WR)
    3. Cap massimo TP in % del prezzo per evitare target irrealistici
       - SCALPING: max 1.5% (raggiungibile in 2-4 candele 15m)
       - SWING:    max 5.0%
    """
    is_btc_eth = coin in BTC_ETH_COINS

    # Fattore di scala leva: applicato SOLO allo SL
    lev_scale = LEVERAGE / max(effective_leverage, 1)
    if lev_scale != 1.0:
        log_alt(f"[{coin}] Leva {effective_leverage}x vs attesa {LEVERAGE}x → SL scalato ×{lev_scale:.1f}")

    buffer = px * 0.002 * lev_scale

    # ── SL: swing low/high reali ─────────────────────────────────
    sl_dist = None
    if df is not None and len(df) >= 10:
        if direction == "LONG":
            swing = float(df['low'].iloc[-10:].min())
            dist  = px - swing
            if 0 < dist <= atr * 3 * lev_scale:
                sl_dist = dist + buffer
        else:
            swing = float(df['high'].iloc[-10:].max())
            dist  = swing - px
            if 0 < dist <= atr * 3 * lev_scale:
                sl_dist = dist + buffer

    # Fallback ATR
    if sl_dist is None or sl_dist <= 0:
        sl_mult = (1.5 if mode == "SCALPING" else 2.0) * lev_scale
        sl_dist = atr * sl_mult
        log_err(f"[{coin}] SL fallback ATR×{sl_mult:.1f}")

    # ── TP: basato su R:R rispetto allo SL + cap % prezzo ────────
    # Il TP NON scala con lev_scale — deve restare raggiungibile.
    # Invece usa un rapporto fisso rispetto allo SL calcolato.
    if mode == "SWING":
        rr_target = 3.0 if ai_score >= 8 else 2.0
        tp_cap_pct = 0.08 if is_btc_eth else 0.05   # max 5% (8% BTC)
    elif mode == "HYBRID":
        rr_target = 1.8 if ai_score >= 8 else 1.5
        tp_cap_pct = 0.04 if is_btc_eth else 0.03   # max 3%
    else:  # SCALPING
        rr_target = 1.5 if ai_score >= 8 else 1.1
        tp_cap_pct = 0.025 if is_btc_eth else 0.015  # max 1.5%

    tp_dist_rr  = sl_dist * rr_target          # TP dal rapporto R:R
    tp_dist_cap = px * tp_cap_pct              # TP cap in % del prezzo
    tp_dist     = min(tp_dist_rr, tp_dist_cap) # usa il più conservativo

    # Floor: TP almeno 0.3% del prezzo (altrimenti non copre nemmeno lo spread)
    tp_floor = px * 0.003
    tp_dist  = max(tp_dist, tp_floor)

    # Log per debug
    sl_pct = sl_dist / px * 100
    tp_pct = tp_dist / px * 100
    rr_actual = tp_dist / sl_dist if sl_dist > 0 else 0
    log_alt(f"[{coin}] SL:{sl_pct:.2f}% TP:{tp_pct:.2f}% R:R=1:{rr_actual:.1f} [{mode}] lev:{effective_leverage}x")

    if direction == "LONG":
        return round(px - sl_dist, 6), round(px + tp_dist, 6)
    else:
        return round(px + sl_dist, 6), round(px - tp_dist, 6)


def compute_trailing_stop(px, atr, direction, ai_score: int = 5, mode: str = "SCALPING",
                          effective_leverage: int = LEVERAGE):
    """Trailing calibrato sulla modalità operativa e leva effettiva."""
    lev_scale = LEVERAGE / max(effective_leverage, 1)
    if mode == "SWING":
        mult = 1.5 if ai_score >= 8 else 2.0
    elif mode == "HYBRID":
        mult = 1.0 if ai_score >= 8 else 1.2
    else:  # SCALPING
        mult = 0.6 if ai_score >= 8 else 0.8
    trail = atr * mult * lev_scale
    return round(px - trail if direction == "LONG" else px + trail, 6)

# ================================================================
# V7 FEATURES — Scalp Mode, Margin Check, Trailing Meccanico, Partial Close
# ================================================================

def detect_scalp_mode(adx_1h, vol_rel, bb_pos, regime, funding_z=0):
    """
    Detecta la scalp mode basata su ADX, volume, BB.
    FLASH: volume spike >3x o funding estremo
    TREND: ADX > 30 e regime direzionale
    RANGE: ADX < 20, prezzo tra le BB
    """
    if vol_rel >= 3.0 or abs(funding_z) > 2.5:
        return "FLASH"
    elif adx_1h > 30 and regime in ("BULL", "BEAR"):
        return "TREND"
    elif adx_1h < 20 and -1.2 < bb_pos < 0.8:
        return "RANGE"
    elif adx_1h >= 20:
        return "TREND"
    return "RANGE"

def get_mode_sl_tp(scalp_mode, px, atr, direction, lev_scale=1.0):
    """Calcola SL/TP per la scalp mode. Ritorna (sl_px, tp_px, sl_dist, tp_dist)."""
    if scalp_mode == "FLASH":
        sl_dist = max(atr * FLASH_SL_ATR * lev_scale, px * FLASH_SL_MIN * lev_scale)
        sl_dist = min(sl_dist, px * FLASH_SL_MAX * lev_scale)
        tp_dist = px * FLASH_TP_PCT
    elif scalp_mode == "RANGE":
        sl_dist = max(atr * RANGE_SL_ATR * lev_scale, px * RANGE_SL_MIN * lev_scale)
        sl_dist = min(sl_dist, px * RANGE_SL_MAX * lev_scale)
        tp_dist = px * RANGE_TP_PCT
    else:  # TREND
        sl_dist = max(atr * TREND_SL_ATR * lev_scale, px * TREND_SL_MIN * lev_scale)
        sl_dist = min(sl_dist, px * TREND_SL_MAX * lev_scale)
        tp_dist = sl_dist * TREND_TP_RR / lev_scale
    tp_dist = max(tp_dist, px * TP_PRICE_PCT * 0.8)  # floor 80% del TP target

    if direction == "LONG":
        return round(px - sl_dist, 6), round(px + tp_dist, 6), sl_dist, tp_dist
    else:
        return round(px + sl_dist, 6), round(px - tp_dist, 6), sl_dist, tp_dist

def check_margin_ok(coin, size, entry_px):
    """Verifica margine sufficiente prima di inviare ordine."""
    try:
        state = call(_info.user_state, account.address, label='margin_check', timeout=10)
        ms = state.get("marginSummary", {})
        account_val = float(ms.get("accountValue", 0) or 0)
        margin_used = float(ms.get("totalMarginUsed", 0) or 0)
        available = account_val - margin_used
        required = (size * entry_px) / LEVERAGE * 1.2  # +20% buffer
        if available < required:
            log_exec(f"[{coin}] Margine insufficiente: need ${required:.2f} have ${available:.2f}")
            return False
        return True
    except:
        return True  # fail-open

def update_mechanical_trailing(coin, pos, mid, atr, direction, open_trade_meta,
                               sz_dec, px_dec):
    """
    Trailing stop meccanico (V7): attiva dopo 50% del TP, trail a ATR dal prezzo.
    Funziona indipendentemente dall'AI.
    """
    meta = open_trade_meta.get(coin, {})
    entry = meta.get("entry_px", pos.get("entry_px", 0))
    sig = meta.get("signal", {})
    sl_dist = float(sig.get("sl_dist", 0))
    scalp_mode = sig.get("scalp_mode", "TREND")

    if sl_dist <= 0 or scalp_mode != "TREND" or atr <= 0:
        return

    tp_dist = sl_dist * TREND_TP_RR
    szi = pos.get("szi", 0)
    is_long = szi > 0

    if is_long:
        profit_dist = mid - entry
    else:
        profit_dist = entry - mid

    # Non attivare se profitto < 50% del TP
    if profit_dist < tp_dist * TRAILING_ACTIVATE:
        return

    trail_dist = atr * TREND_TRAIL_ATR
    if is_long:
        new_ts = mid - trail_dist
        new_ts = max(new_ts, entry * 1.0005)  # min breakeven
    else:
        new_ts = mid + trail_dist
        new_ts = min(new_ts, entry * 0.9995)

    new_ts = round_to_decimals(new_ts, px_dec.get(coin, 2))
    old_ts = meta.get("current_ts", 0)

    if is_long and new_ts <= old_ts:
        return
    if not is_long and new_ts >= old_ts and old_ts > 0:
        return

    try:
        # Cancel solo il nostro SL per OID
        sl_oid = meta.get("sl_oid")
        if sl_oid:
            try: cancel_order(coin, sl_oid)
            except: pass
        else:
            # Fallback: scan (solo per vecchie posizioni senza OID)
            orders = get_open_trigger_orders(coin)
            for o in orders:
                if o.get("orderType") == "Stop Market":
                    cancel_order(coin, o["oid"])
        time.sleep(0.3)
        size_abs = round_to_decimals(abs(szi), sz_dec.get(coin, 2))
        call(_exchange.order, str(coin), not is_long, size_abs, new_ts,
             {"trigger": {"triggerPx": new_ts, "isMarket": True, "tpsl": "sl"}},
             True, timeout=15, label=f'trail_{coin}')
        meta["current_ts"] = new_ts
        meta["trailing_active"] = True
        trail_pct = abs(new_ts - entry) / entry * 100
        log_exec(f"[{coin}] 📈 Trailing → {new_ts} ({trail_pct:.2f}% from entry)")
    except Exception as e:
        log_err(f"[{coin}] trailing error: {e}")

def check_partial_close(coin, pos, mid, open_trade_meta, sz_dec, px_dec):
    """Chiudi 50% della posizione al 60% del TP (solo TREND mode)."""
    meta = open_trade_meta.get(coin, {})
    if meta.get("partial_done"):
        return
    sig = meta.get("signal", {})
    sl_dist = float(sig.get("sl_dist", 0))
    scalp_mode = sig.get("scalp_mode", "TREND")
    if sl_dist <= 0 or scalp_mode != "TREND":
        return

    entry = meta.get("entry_px", 0)
    szi = pos.get("szi", 0)
    is_long = szi > 0
    tp_dist = sl_dist * TREND_TP_RR

    profit_dist = (mid - entry) if is_long else (entry - mid)
    if profit_dist < tp_dist * TREND_PARTIAL:
        return

    close_size = round_to_decimals(abs(szi) * 0.5, sz_dec.get(coin, 2))
    if close_size <= 0:
        return
    try:
        close_px = round_to_decimals(mid * (0.998 if is_long else 1.002), px_dec.get(coin, 2))
        call(_exchange.order, str(coin), not is_long, close_size, close_px,
             {"limit": {"tif": "Ioc"}}, False, timeout=15, label=f'partial_{coin}')
        meta["partial_done"] = True
        pnl = profit_dist / entry * 100
        log_exec(f"[{coin}] 💰 PARTIAL 50% @ {mid:.1f} PnL:{pnl:+.2f}%")
        tg(f"💰 <b>{coin}</b> partial 50% PnL:{pnl:+.1f}%", silent=True)
    except Exception as e:
        log_err(f"[{coin}] partial error: {e}")

def detect_market_regime() -> str:
    def _regime(coin):
        try:
            now     = int(time.time() * 1000)
            candles = call(_info.candles_snapshot, coin, "4h",
                           now - 2_592_000_000, now,
                           timeout=API_TIMEOUT_SEC, label=f'{coin}-regime')
            closes  = pd.Series([float(c['c']) for c in candles])
            highs   = pd.Series([float(c['h']) for c in candles])
            lows    = pd.Series([float(c['l']) for c in candles])
            e50     = closes.ewm(span=50,  min_periods=50).mean().iloc[-1]
            e200    = closes.ewm(span=200, min_periods=200).mean().iloc[-1]
            price   = closes.iloc[-1]

            # Regime base da EMA
            if price > e50 > e200:   base = "BULL"
            elif price < e50 < e200: base = "BEAR"
            else:                    base = "RANGE"

            # Override momentum: se movimento forte nelle ultime 4 candele 4h
            # ignora la lentezza delle EMA e forza il regime
            if len(closes) >= 5:
                move_4bars = (closes.iloc[-1] - closes.iloc[-5]) / closes.iloc[-5]
                atr_4h     = (highs.iloc[-10:] - lows.iloc[-10:]).mean()
                atr_pct    = atr_4h / closes.iloc[-1]

                if move_4bars > atr_pct * 1.5:   # salita forte > 1.5×ATR
                    if base == "RANGE":
                        base = "BULL"
                elif move_4bars < -atr_pct * 1.5:  # discesa forte
                    if base == "RANGE":
                        base = "BEAR"

            return base
        except Exception:
            return "RANGE"
    btc = _regime("BTC")
    eth = _regime("ETH")
    # Se BTC è BULL anche con ETH RANGE, usa BULL — BTC comanda
    if btc == "BULL":   return "BULL"
    if btc == "BEAR":   return "BEAR"
    return btc if btc == eth else "RANGE"


def _parse_ai_json(text: str) -> dict:
    """Estrae il primo oggetto JSON valido da testo che può avere contenuto extra."""
    text = text.replace("```json", "").replace("```", "").strip()
    # Trova il primo { e l'ultimo } corrispondente
    start = text.find("{")
    if start == -1:
        raise ValueError("Nessun JSON trovato")
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i+1])
    raise ValueError("JSON non chiuso correttamente")


def fetch_liquidity_data(coin: str, px: float) -> dict:
    """
    Recupera orderbook L2 e calcola:
    - spread_real: spread bid/ask reale
    - imbalance: % bid vs ask (>0.6 = compratori dominano)
    - aggressive_buys: ratio market orders buy vs sell (proxy: volume bid vs ask)
    - liq_cluster_dist: distanza % al cluster di liquidazione più vicino
    - liq_cluster_side: "above" (short liquidations → magnete LONG) o "below"
    """
    result = {
        "spread_real":       None,
        "imbalance":         0.5,
        "aggressive_buys":   False,
        "liq_cluster_dist":  None,
        "liq_cluster_side":  None,
    }
    try:
        ob = call(_info.l2_snapshot, coin, label=f'ob_{coin}', timeout=10)
        if not ob:
            return result

        bids = ob.get("levels", [{}])[0] if len(ob.get("levels", [])) > 0 else []
        asks = ob.get("levels", [{}])[1] if len(ob.get("levels", [])) > 1 else []

        if not bids or not asks:
            return result

        # ── Spread reale ─────────────────────────────────────────
        best_bid = float(bids[0]["px"])
        best_ask = float(asks[0]["px"])
        result["spread_real"] = (best_ask - best_bid) / (px + 1e-10)

        # ── Orderbook Imbalance ──────────────────────────────────
        # Somma size dei primi 10 livelli bid vs ask
        bid_size = sum(float(b["sz"]) for b in bids[:10])
        ask_size = sum(float(a["sz"]) for a in asks[:10])
        total    = bid_size + ask_size + 1e-10
        result["imbalance"] = bid_size / total

        # ── Aggressive Market Orders (proxy) ─────────────────────
        # Se imbalance > 0.65 E bid_size cresce verso ask → buyers aggressivi
        result["aggressive_buys"] = result["imbalance"] > 0.65

        # ── Liquidation Clusters ─────────────────────────────────
        # Stima cluster usando i livelli con size anomala nel book
        # Cluster SOPRA = short liquidations (magnete per LONG)
        # Cluster SOTTO = long liquidations (danger zone per LONG)
        ask_sizes  = [(float(a["px"]), float(a["sz"])) for a in asks[:20]]
        bid_sizes  = [(float(b["px"]), float(b["sz"])) for b in bids[:20]]

        # Trova il livello con size massima (muro = cluster)
        if ask_sizes:
            max_ask = max(ask_sizes, key=lambda x: x[1])
            dist_above = (max_ask[0] - px) / (px + 1e-10)
            result["liq_cluster_dist"] = dist_above
            result["liq_cluster_side"] = "above"

        if bid_sizes:
            max_bid = max(bid_sizes, key=lambda x: x[1])
            dist_below = (px - max_bid[0]) / (px + 1e-10)
            # Prendi il cluster più vicino
            if result["liq_cluster_dist"] is None or dist_below < result["liq_cluster_dist"]:
                result["liq_cluster_dist"] = dist_below
                result["liq_cluster_side"] = "below"

    except Exception as e:
        log_err(f"[{coin}] fetch_liquidity_data: {e}")

    return result


def compute_setup_score(direction: str, px: float, ema50: float, ema200: float,
                        rsi: float, spread_pct: float, ai_score: int,
                        funding_z: float, liq: dict = None) -> int:
    """
    Score composito 0-100 per il setup. Soglia minima: SETUP_SCORE_MIN (75).

    Criterio              Peso   Logica
    ──────────────────────────────────────────────────────────────────
    Trend Confluence       30%   +30 px>EMA200_4h e EMA50>EMA200_4h
    RSI Recoil             20%   +20 RSI zona recoil ideale
    Spread stimato         15%   +15 se < 0.05%, -50 se > 0.10%
    AI Confidence          25%   ai_score/10 × 25
    Funding Bias           10%   +10 se funding favorevole
    ── MODULO LIQUIDITY (orderbook reale) ──────────────────────────
    Spread reale         ±var    sostituisce spread stimato se disponibile
    Liq Cluster vicino    +40    cluster < 3% → magnete per la direzione
    OB Imbalance > 60%    +20    buyers/sellers dominano
    Aggressive Mkt Orders +15    urgenza direzionale rilevata
    Spread > 0.05%        -50    slippage enorme con leva 20x
    """
    score = 0

    # ── Trend Confluence (30%) ────────────────────────────────────
    if direction == "LONG":
        full_trend = px > ema200 and ema50 > ema200
        half_trend = px > ema200 or ema50 > ema200
    else:
        full_trend = px < ema200 and ema50 < ema200
        half_trend = px < ema200 or ema50 < ema200

    if full_trend:
        score += 30
    elif half_trend:
        score += 15

    # ── RSI Recoil (20%) ─────────────────────────────────────────
    if direction == "LONG":
        if 35 <= rsi <= 48:
            score += 20   # zona recoil ideale
        elif rsi < 35:
            score += 10   # ipervenduto
    else:
        if 52 <= rsi <= 65:
            score += 20
        elif rsi > 65:
            score += 10

    # ── AI Confidence (25%) ──────────────────────────────────────
    score += int((ai_score / 10) * 25)

    # ── Funding Bias (10%) ───────────────────────────────────────
    if direction == "LONG":
        if funding_z < -1.0:
            score += 10   # ti pagano per stare long
        elif funding_z < 0:
            score += 5
    else:
        if funding_z > 1.0:
            score += 10
        elif funding_z > 0:
            score += 5

    # ── MODULO LIQUIDITY (15% base + bonus/penalità orderbook) ───
    if liq and liq.get("spread_real") is not None:
        # Spread reale dall'orderbook — sostituisce stima high-low
        spread_real = liq["spread_real"]
        if spread_real < 0.0005:        # < 0.05% — liquido ✅
            score += 15
        elif spread_real < 0.001:       # < 0.10%
            score += 10
        else:                           # > 0.10% — penalità pesante ❌
            score -= 50
    else:
        # Fallback: spread stimato da high-low candela 15m
        if spread_pct < 0.0005:
            score += 15
        elif spread_pct < 0.001:
            score += 10
        else:
            score -= 50

    if liq:
        imbalance        = liq.get("imbalance", 0.5)
        aggressive_buys  = liq.get("aggressive_buys", False)
        liq_cluster_dist = liq.get("liq_cluster_dist")
        liq_cluster_side = liq.get("liq_cluster_side")

        # Liquidation cluster vicino → magnete (< 3% dal prezzo)
        if liq_cluster_dist is not None and liq_cluster_dist < 0.03:
            if direction == "LONG" and liq_cluster_side == "above":
                score += 40   # short liquidations sopra → target chiaro
            elif direction == "SHORT" and liq_cluster_side == "below":
                score += 40   # long liquidations sotto → target chiaro

        # Orderbook imbalance > 60%
        if direction == "LONG"  and imbalance > 0.60:
            score += 20
        elif direction == "SHORT" and imbalance < 0.40:
            score += 20

        # Aggressive market orders
        if direction == "LONG"  and aggressive_buys:
            score += 15
        elif direction == "SHORT" and not aggressive_buys and imbalance < 0.35:
            score += 15

    return max(0, min(100, score))


def ai_validate_signal(coin: str, direction: str, regime: str,
                       rsi: float, bb_pos: float, vol_rel: float,
                       funding_z: float, oi_change: float,
                       confluence: int, recent_closes: list,
                       df: pd.DataFrame = None,
                       strategy: str = "MEAN_REV",
                       liq_data: dict = None,
                       pf_hist: float = 0.0,
                       wr_recent: float = 0.0,
                       wr_hist: float = 0.0,
                       df_4h: pd.DataFrame = None,
                       mids: dict = None,
                       active_positions: dict = None) -> tuple[bool, int, str]:
    """Valida il segnale con Claude. Ritorna (valid, score, reasoning)."""
    try:
        # ── 1. OHLCV + Pattern recognition candele 15m ─────────────
        ohlcv_str = ""
        pattern_str = "Nessun pattern rilevante"
        if df is not None and len(df) >= 8:
            last8 = df.iloc[-8:][["open","high","low","close","volume"]].copy()
            rows = []
            for _, r in last8.iterrows():
                body = r["close"] - r["open"]
                candle = "🟢" if body >= 0 else "🔴"
                rows.append(f"{candle} O:{r['open']:.4g} H:{r['high']:.4g} L:{r['low']:.4g} C:{r['close']:.4g} V:{r['volume']:.2g}")
            ohlcv_str = "\n".join(rows)

            c    = df.iloc[-1]
            prev = df.iloc[-2]
            body_c    = abs(c["close"] - c["open"])
            range_c   = c["high"] - c["low"]
            wick_up_c = c["high"] - max(c["close"], c["open"])
            wick_dn_c = min(c["close"], c["open"]) - c["low"]
            body_prev = abs(prev["close"] - prev["open"])
            patterns  = []
            if range_c > 0:
                if wick_dn_c > body_c * 2 and wick_up_c < body_c * 0.5:
                    patterns.append("🔨 Hammer (inversione rialzista)")
                if wick_up_c > body_c * 2 and wick_dn_c < body_c * 0.5:
                    patterns.append("⭐ Shooting Star (inversione ribassista)")
                if body_c < range_c * 0.1:
                    patterns.append("〰️ Doji (indecisione)")
            if body_c > body_prev * 1.5:
                if c["close"] > c["open"] and prev["close"] < prev["open"]:
                    patterns.append("📈 Bullish Engulfing")
                elif c["close"] < c["open"] and prev["close"] > prev["open"]:
                    patterns.append("📉 Bearish Engulfing")
            if len(df) >= 3:
                last3 = df.iloc[-3:]
                if all(last3["close"].values > last3["open"].values):
                    patterns.append("🪖 Three White Soldiers")
                elif all(last3["close"].values < last3["open"].values):
                    patterns.append("🐦 Three Black Crows")
            if patterns:
                pattern_str = " | ".join(patterns)

        # ── 2. Coerenza Multi-Timeframe 4h ─────────────────────────
        mtf_str = "N/A"
        if df_4h is not None and len(df_4h) >= 5:
            r4           = df_4h.iloc[-1]
            rsi_4h_val   = float(r4.get("rsi", 50))
            bb_4h_val    = float(r4.get("bb_pos", 0))
            slope_4h_val = float(r4.get("ema20_slope", 0))
            ema200_4h_v  = float(r4.get("ema200", 0))
            px_4h        = float(r4.get("close", 0))
            px_vs_ema    = "sopra" if px_4h > ema200_4h_v else "sotto"
            trend_4h     = "rialzista ▲" if slope_4h_val > 0.001 else "ribassista ▼" if slope_4h_val < -0.001 else "laterale →"
            if direction == "LONG":
                agree = "✅ CONCORDANO" if slope_4h_val > 0 and px_vs_ema == "sopra" else "⚠️ DISCORDANO — rischio alto"
            else:
                agree = "✅ CONCORDANO" if slope_4h_val < 0 and px_vs_ema == "sotto" else "⚠️ DISCORDANO — rischio alto"
            mtf_str = f"RSI_4h:{rsi_4h_val:.1f} | BB_4h:{bb_4h_val:.2f} | Trend:{trend_4h} | px {px_vs_ema} EMA200_4h | {agree}"

        # ── 3. Correlazione BTC in tempo reale ─────────────────────
        btc_str = "N/A"
        if mids:
            btc_px  = float(mids.get("BTC", 0) or 0)
            eth_px  = float(mids.get("ETH", 0) or 0)
            btc_str = f"BTC: ${btc_px:,.0f} | ETH: ${eth_px:,.0f} | Regime macro: {regime}"

        # ── 4. S/R levels ──────────────────────────────────────────
        sr_str = "N/A"
        if df is not None and len(df) >= 50:
            px_now     = float(df["close"].iloc[-1])
            res_levels = sorted([h for h in df["high"].iloc[-50:] if h > px_now])[:2]
            sup_levels = sorted([l for l in df["low"].iloc[-50:]  if l < px_now], reverse=True)[:2]
            res_pcts   = [f"{(r-px_now)/px_now*100:+.2f}%" for r in res_levels]
            sup_pcts   = [f"{(s-px_now)/px_now*100:+.2f}%" for s in sup_levels]
            sr_str = f"Resistenze: {res_pcts} | Supporti: {sup_pcts}"

        # ── 5. Correlazione posizioni aperte ───────────────────────
        corr_str = "✅ Nessuna posizione correlata"
        if active_positions:
            same_dir = [c for c in active_positions
                        if active_positions[c].get("direction","") == direction]
            if same_dir:
                corr_str = f"⚠️ Già in {direction} su: {same_dir} — rischio correlazione"

        # ── Volume context ─────────────────────────────────────────
        vol_context = "N/A"
        if df is not None and len(df) >= 20:
            vol_ma20    = df["volume"].iloc[-20:].mean()
            vol_ratio   = float(df["volume"].iloc[-1]) / vol_ma20 if vol_ma20 > 0 else 1.0
            bb_touch    = bb_pos < -0.5 or bb_pos > 0.5
            vol_context = f"Vol: {vol_ratio:.2f}x media | BB touch: {'✅' if bb_touch else '❌'}"

        # ── Trade history ──────────────────────────────────────────
        history_str = "Nessun trade simile ancora."
        try:
            history = load_trade_history()
            similar = [t for t in history
                       if t.get("regime") == regime
                       and t.get("direction") == direction
                       and t.get("strategy") == strategy][-10:]
            if similar:
                wins    = sum(1 for t in similar if t.get("outcome") == "win")
                losses  = sum(1 for t in similar if t.get("outcome") == "loss")
                avg_pnl = sum(t.get("pnl_pct", 0) for t in similar) / len(similar)
                history_str = f"{wins}W/{losses}L | PnL medio: {avg_pnl:+.2f}%\n"
                for t in similar[-3:]:
                    emoji = "✅" if t.get("outcome") == "win" else "❌"
                    history_str += f"  {emoji} RSI:{t.get('rsi',0):.0f} BB:{t.get('bb_pos',0):.2f} → {t.get('pnl_pct',0):+.2f}%\n"
        except Exception:
            pass

        # ── Liquidity ──────────────────────────────────────────────
        liq_context = "N/A"
        if liq_data:
            sr = liq_data.get("spread_real")
            if sr is not None:
                warn = "⚠️ ALTO" if sr > 0.003 else "✅"
                liq_context = (f"Spread:{sr:.3%}{warn} | "
                               f"OB:{liq_data.get('imbalance',0.5):.2f} | "
                               f"AggBuy:{'✅' if liq_data.get('aggressive_buys') else '❌'}")
                cd = liq_data.get("liq_cluster_dist")
                if cd:
                    liq_context += f" | Cluster:{liq_data.get('liq_cluster_side','?')}@{cd:.2%}"

        # Backtest di riferimento — passiamo entrambi così l'AI sceglie
        bt_scalping = f"PF:{pf_hist:.2f} WR:{wr_hist:.1%}" if pf_hist > 0 else "N/A"
        bt_swing    = "N/A"  # pf_swing passato separatamente se disponibile

        regime_instruction = ""
        if strategy == "MOMENTUM":
            regime_instruction = """STRATEGIA: BREAKOUT/MOMENTUM — compriamo forza, vendiamo debolezza. NON mean reversion.
RSI 60-75 in LONG = zona di forza ✅ (momentum attivo, NON ipercomprato)
RSI 25-40 in SHORT = zona di debolezza ✅ (NON ipervenduto)
NON rifiutare per RSI alto in LONG o RSI basso in SHORT."""
        else:  # REVERSAL
            regime_instruction = """STRATEGIA: REVERSAL in mercato laterale — compriamo al supporto, vendiamo alla resistenza.
RSI < 35 in LONG = ipervenduto al supporto ✅ (condizione ideale per reversal)
RSI > 65 in SHORT = ipercomprato alla resistenza ✅ (condizione ideale per reversal)
NON rifiutare per RSI estremo — è la condizione di entry della strategia."""

        prompt = f"""Crypto futures trader. Valida {coin} {direction} e rispondi SOLO con JSON.

{regime_instruction}
Il segnale è già confermato meccanicamente su 3 timeframe. Tu validi solo rischi nascosti.

MERCATO: Regime:{regime} | RSI_5m:{rsi:.1f} BB:{bb_pos:.3f} Vol:{vol_rel:.2f}x FZ:{funding_z:.2f} OI:{oi_change:.3f}
PATTERN: {pattern_str}
MTF 1h: {mtf_str}
BTC: {btc_str}
LIQUIDITÀ: {liq_context}
S/R: {sr_str}
BACKTEST: {bt_scalping}
STORICO: {history_str.strip() if history_str else 'N/A'}
CORRELAZIONI: {corr_str}

PRICE ACTION 5m:
{ohlcv_str if ohlcv_str else 'N/A'}

Valuta SOLO: 1)Liquidità sufficiente? 2)BTC supporta? 3)Rischi wick/liquidazione? 4)Volume conferma?

JSON ONLY: {{"score":<0-10>,"valid":<true se >=3>,"wait":<true SOLO se rischio liquidazione/wick imminente>,"strategy":"{strategy}","mode":"SCALPING|SWING","reasoning":"<max 10 parole>"}}"""

        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type":      "application/json",
                "x-api-key":         os.getenv("ANTHROPIC_API_KEY", ""),
                "anthropic-version": "2023-06-01",
            },
            json={
                "model":      "claude-haiku-4-5-20251001",
                "max_tokens": 400,
                "messages":   [{"role": "user", "content": prompt}]
            },
            timeout=20
        )
        if resp.status_code != 200:
            log_err(f"[{coin}] AI HTTP {resp.status_code}: {resp.text[:100]}")
            return True, 0, "AI non disponibile", strategy, "SCALPING"
        text   = resp.json().get("content", [{}])[0].get("text", "").strip()
        result = _parse_ai_json(text)
        score      = int(result.get("score", 5))
        valid      = score >= 3
        wait       = bool(result.get("wait", False))
        reason     = result.get("reasoning", "")
        ai_strategy = result.get("strategy", strategy)   # AI può sovrascrivere
        ai_mode     = result.get("mode", "SCALPING")
        if ai_strategy not in ("MEAN_REV", "TREND", "HYBRID"):
            ai_strategy = strategy
        if ai_mode not in ("SCALPING", "SWING"):
            ai_mode = "SCALPING"
        if wait:
            valid  = False
            reason = f"⏳ {reason}"
        log_alt(f"[{coin}] AI → strategy:{ai_strategy} mode:{ai_mode} score:{score} | {reason}")
        return valid, score, reason, ai_strategy, ai_mode
    except Exception as e:
        log_err(f"[{coin}] ai_validate: {e}")
        # Fail-open: se AI non risponde o risponde male, approva con score neutro
        return True, 5, "AI fallback — approva", strategy, "SCALPING"

def quick_backtest(df: pd.DataFrame, direction: str, coin: str = "",
                   mode: str = "SCALPING", strategy: str = "MOMENTUM") -> dict:
    """
    Backtest a 2 LIVELLI:

    LIVELLO 1 — EDGE STRUTTURALE (tutti i trade storici):
      La strategia ha un vantaggio statistico su questa coin?
      Richiede almeno MIN_BACKTEST_TRADES trade per essere affidabile.
      Metriche: pf_full, wr_full, n_trades_full

    LIVELLO 2 — REGIME CHECK (ultimi 50 trade):
      Il mercato attuale favorisce questa strategia?
      Se l'edge storico è buono ma gli ultimi 50 trade sono negativi → skip.
      Metriche: pf_recent, wr_recent, n_trades_recent

    SCALPING: SL=ATR×1.0  TP=ATR×1.1  FORWARD=24 candele 5m (2h)
    SWING:    SL=ATR×2.0  TP=ATR×4.0  FORWARD=96 candele 5m (8h)
    """
    try:
        if mode == "SWING":
            sl_mult, tp_mult, forward_win = 2.0, 4.0, 96
        else:
            sl_mult, tp_mult, forward_win = 1.0, 1.1, 24

        RECENT_TRADES_WINDOW = 50   # ultimi N trade per regime check

        n = len(df)
        if n < 200 + forward_win:
            return {"win_rate": 0.0, "win_rate_recent": 0.0, "profit_factor": 0.0,
                    "profit_factor_recent": 0.0, "n_trades": 0, "n_trades_recent": 0,
                    "valid": False, "mode": mode, "avg_return": 0.0, "avg_return_recent": 0.0}

        close_a  = df['close'].values.astype(np.float32)
        high_a   = df['high'].values.astype(np.float32)
        low_a    = df['low'].values.astype(np.float32)
        ema200_a = df['ema200'].values.astype(np.float32)
        ema20_a  = df['ema20'].values.astype(np.float32)
        slope_a  = df['ema20_slope'].values.astype(np.float32)
        vol_a    = df['vol_rel'].values.astype(np.float32) if 'vol_rel' in df else np.ones(n, dtype=np.float32)
        atr_a    = df['atr'].values.astype(np.float32)
        rsi_a    = df['rsi'].values.astype(np.float32)
        bb_w     = df['bb_width'].values.astype(np.float32) if 'bb_width' in df else np.zeros(n, dtype=np.float32)
        macd_h   = df['macd_hist'].values.astype(np.float32) if 'macd_hist' in df else np.zeros(n, dtype=np.float32)
        hh_a     = df['hh'].values.astype(np.float32) if 'hh' in df else np.zeros(n, dtype=np.float32)
        ll_a     = df['ll'].values.astype(np.float32) if 'll' in df else np.zeros(n, dtype=np.float32)

        bb_w_ma = pd.Series(bb_w).rolling(20).mean().values.astype(np.float32)
        mom4 = np.zeros(n, dtype=np.float32)
        mom4[4:] = (close_a[4:] - close_a[:-4]) / (close_a[:-4] + 1e-10)

        bb_expanding = bb_w > bb_w_ma * 1.3
        bb_squeezed  = bb_w < bb_w_ma * 0.7

        breakout_long = (
            (bb_expanding | (bb_squeezed & (mom4 > 0.01))) &
            (vol_a >= 1.5) & (close_a > ema20_a) & (hh_a > 0) & (macd_h > 0)
        )
        momentum_long = (
            (close_a > ema200_a) & (slope_a > 0.001) &
            (rsi_a >= 40) & (rsi_a <= 60) & (macd_h > 0)
        )
        breakout_short = (
            (bb_expanding | (bb_squeezed & (mom4 < -0.01))) &
            (vol_a >= 1.5) & (close_a < ema20_a) & (ll_a > 0) & (macd_h < 0)
        )
        momentum_short = (
            (close_a < ema200_a) & (slope_a < -0.001) &
            (rsi_a >= 40) & (rsi_a <= 60) & (macd_h < 0)
        )

        if direction == "LONG":
            sig_mask = breakout_long | momentum_long
            ema_filter = close_a > ema200_a
        else:
            sig_mask = breakout_short | momentum_short
            ema_filter = close_a < ema200_a

        valid_mask = sig_mask & ema_filter & (vol_a >= 0.5) & (atr_a > 0)
        valid_idx  = np.where(valid_mask[200: n - forward_win])[0] + 200

        if len(valid_idx) == 0:
            return {"win_rate": 0.0, "win_rate_recent": 0.0, "profit_factor": 0.0,
                    "profit_factor_recent": 0.0, "n_trades": 0, "n_trades_recent": 0,
                    "valid": False, "mode": mode, "avg_return": 0.0, "avg_return_recent": 0.0}

        # ── Simula tutti i trade ──────────────────────────────────
        all_trades = []   # lista di (outcome, return_pct, index)

        for i in valid_idx:
            px_h  = float(close_a[i])
            atr_h = float(atr_a[i])
            sl_d  = atr_h * sl_mult
            tp_d  = atr_h * tp_mult
            end   = min(i + forward_win, n)
            if i + 1 >= end:
                continue

            if direction == "LONG":
                tp_hits = np.where(high_a[i+1:end] >= px_h + tp_d)[0]
                sl_hits = np.where(low_a [i+1:end] <= px_h - sl_d)[0]
            else:
                tp_hits = np.where(low_a [i+1:end] <= px_h - tp_d)[0]
                sl_hits = np.where(high_a[i+1:end] >= px_h + sl_d)[0]

            tp_f = tp_hits[0] + 1 if len(tp_hits) > 0 else forward_win + 1
            sl_f = sl_hits[0] + 1 if len(sl_hits) > 0 else forward_win + 1

            if tp_f == sl_f == forward_win + 1:
                continue

            if tp_f == sl_f:
                outcome = 0
            elif tp_f < sl_f:
                outcome = 1
            else:
                outcome = 0

            trade_ret = (tp_d / px_h) if outcome == 1 else (-sl_d / px_h)
            all_trades.append((outcome, trade_ret, tp_d if outcome == 1 else 0, sl_d if outcome == 0 else 0))

        n_total = len(all_trades)
        if n_total == 0:
            return {"win_rate": 0.0, "win_rate_recent": 0.0, "profit_factor": 0.0,
                    "profit_factor_recent": 0.0, "n_trades": 0, "n_trades_recent": 0,
                    "valid": False, "mode": mode, "avg_return": 0.0, "avg_return_recent": 0.0}

        # ── LIVELLO 1: EDGE STRUTTURALE (tutti i trade) ───────────
        full_wins    = sum(1 for t in all_trades if t[0] == 1)
        full_gross_w = sum(t[2] for t in all_trades)
        full_gross_l = sum(t[3] for t in all_trades)
        full_ret     = sum(t[1] for t in all_trades)

        wr_full  = round(full_wins / n_total, 3)
        pf_full  = round(min(full_gross_w / (full_gross_l + 1e-10), 10.0), 2)
        avg_full = round(full_ret / n_total, 4)

        # ── LIVELLO 2: REGIME CHECK (ultimi N trade) ──────────────
        recent = all_trades[-RECENT_TRADES_WINDOW:]
        n_recent = len(recent)

        if n_recent >= 10:
            rec_wins    = sum(1 for t in recent if t[0] == 1)
            rec_gross_w = sum(t[2] for t in recent)
            rec_gross_l = sum(t[3] for t in recent)
            rec_ret     = sum(t[1] for t in recent)
            wr_recent   = round(rec_wins / n_recent, 3)
            pf_recent   = round(min(rec_gross_w / (rec_gross_l + 1e-10), 10.0), 2)
            avg_recent  = round(rec_ret / n_recent, 4)
        else:
            wr_recent  = wr_full
            pf_recent  = pf_full
            avg_recent = avg_full

        return {
            "win_rate":           wr_full,
            "win_rate_recent":    wr_recent,
            "profit_factor":      pf_full,
            "profit_factor_recent": pf_recent,
            "n_trades":           n_total,
            "n_trades_recent":    n_recent,
            "valid":              n_total >= MIN_BACKTEST_TRADES,
            "mode":               mode,
            "avg_return":         avg_full,
            "avg_return_recent":  avg_recent,
        }

    except Exception as e:
        log_err(f"quick_backtest [{mode}]: {e}")
        return {"win_rate": 0.0, "win_rate_recent": 0.0, "profit_factor": 0.0,
                "profit_factor_recent": 0.0, "n_trades": 0, "n_trades_recent": 0,
                "valid": True, "mode": mode, "avg_return": 0.0, "avg_return_recent": 0.0}

def quick_backtest_dual(df: pd.DataFrame, direction: str, coin: str = "",
                        strategy: str = "MEAN_REV") -> dict:
    """Esegue backtest in entrambe le modalità con la strategia corretta."""
    bt_scalping = quick_backtest(df, direction, coin, mode="SCALPING", strategy=strategy)
    bt_swing    = quick_backtest(df, direction, coin, mode="SWING",    strategy=strategy)
    return {"SCALPING": bt_scalping, "SWING": bt_swing}


# ================================================================
# PROCESSOR — RUN FUNCTION
# ================================================================
# ================================================================

def run_processor():
    log_alt("=== RUN PROCESSOR ===")
    regime = detect_market_regime()
    log_alt(f"Regime: {regime}")

    # Usa i candidati pre-filtrati dallo Scanner
    with _scanner_lock:
        universe = list(_scanner_candidates)

    if not universe:
        log_alt("Nessun candidato dallo Scanner — skip")
        return

    log_alt(f"Candidati dallo Scanner: {len(universe)} coin")

    try:
        ctx_data    = call(_info.meta_and_asset_ctxs,
                           timeout=API_TIMEOUT_SEC, label='meta_ctxs')
        meta_assets = ctx_data[0]["universe"]
        ctxs_main   = ctx_data[1]
    except Exception as e:
        log_err(f"meta_and_asset_ctxs: {e}")
        return

    fetch_funding_and_oi(ctxs_main, meta_assets)

    sz_decimals_proc, _ = fetch_meta()
    now_ms = int(time.time() * 1000)

    # ── BTC Momentum Filter (realtime) ────────────────────────────
    # Calcola il momentum BTC a breve termine (ultime 6 candele 15m = 1.5h)
    # per bloccare trade contro la direzione di BTC in tempo reale.
    # Il regime (BULL/BEAR/RANGE) è lento (basato su EMA 4h), questo è veloce.
    btc_momentum = 0.0
    btc_mom_4h   = 0.0
    btc_rsi_15m  = 50.0
    try:
        btc_candles = call(_info.candles_snapshot, "BTC", "15m",
                           now_ms - 86400000, now_ms,
                           timeout=API_TIMEOUT_SEC, label='btc_momentum')
        if btc_candles and len(btc_candles) >= 20:
            btc_closes = pd.Series([float(c['c']) for c in btc_candles])
            # Momentum 1.5h: ultime 6 candele 15m
            btc_momentum = (btc_closes.iloc[-1] - btc_closes.iloc[-6]) / btc_closes.iloc[-6]
            # Momentum 4h: ultime 16 candele 15m
            if len(btc_closes) >= 17:
                btc_mom_4h = (btc_closes.iloc[-1] - btc_closes.iloc[-16]) / btc_closes.iloc[-16]
            # RSI 14 su 15m
            delta = btc_closes.diff()
            up    = delta.clip(lower=0).ewm(com=13, min_periods=13).mean()
            down  = -delta.clip(upper=0).ewm(com=13, min_periods=13).mean()
            btc_rsi_15m = float(100 - (100 / (1 + (up.iloc[-1] / (down.iloc[-1] + 1e-10)))))
            log_alt(f"BTC: mom_1.5h:{btc_momentum:+.3%} | mom_4h:{btc_mom_4h:+.3%} | RSI:{btc_rsi_15m:.1f}")
    except Exception as e:
        log_err(f"BTC momentum calc: {e}")

    sent   = 0
    table_rows = []
    valid_signals = []   # raccoglie tutti i segnali validi per ranking finale
    cycle_wr_samples = []
    skip_counts = {
        "candles_4h": 0, "candles_15m": 0, "volume": 0,
        "volatility": 0, "vol_rel": 0, "confluence": 0,
        "backtest": 0, "confluence5": 0, "wr_recente": 0,
        "funding": 0, "ai": 0, "score": 0
    }

    for coin in universe:
        try:
            # Categoria per log
            cat = get_coin_category(coin)

            if not is_cooldown_ok(coin, strategy="TREND" if regime in ("BULL","BEAR") else "MEAN_REV"):
                continue  # skip silenzioso per non spammare il log

            # ── Filtro tradabilità ──
            if coin not in sz_decimals_proc:
                continue
            sz_dec_proc = sz_decimals_proc[coin]
            mid_px_proc = float(_funding_cache.get(coin, 0) or 0)
            if mid_px_proc > 0:
                min_usd_required = (mid_px_proc * 10 ** (-sz_dec_proc)) / LEVERAGE
                if min_usd_required > TRADE_SIZE_USD:
                    log_alt(f"[{coin}] skip: min notional ${min_usd_required:.4f}")
                    continue

            # ── Fetch candele MTF: 4h (trend) + 15m (entry) ─────────────
            def fetch_candles(tf, lookback_days, label_suffix=""):
                for _attempt in range(3):
                    try:
                        c = call(
                            _info.candles_snapshot, coin, tf,
                            now_ms - (86400000 * lookback_days), now_ms,
                            timeout=API_TIMEOUT_SEC, label=f"{coin}{label_suffix}"
                        )
                        if c:
                            return c
                    except Exception:
                        pass
                    time.sleep(1.5 * (_attempt + 1))
                return None

            # ── Fetch candele 3 TF: 1h (trend) + 15m (setup) + 5m (entry) ──
            candles_1h  = fetch_candles(TIMEFRAME_TREND, LOOKBACK_DAYS_TREND, "-1h")
            time.sleep(0.3)
            candles_15m = fetch_candles(TIMEFRAME_SETUP, LOOKBACK_DAYS_SETUP, "-15m")
            time.sleep(0.3)
            candles_5m  = fetch_candles(TIMEFRAME_ENTRY, LOOKBACK_DAYS_ENTRY, "-5m")
            time.sleep(0.3)

            if not candles_1h or len(candles_1h) < MIN_CANDLES_TREND:
                skip_counts["candles_4h"] += 1  # reuso counter
                continue
            if not candles_15m or len(candles_15m) < MIN_CANDLES_SETUP:
                skip_counts["candles_15m"] += 1
                continue
            if not candles_5m or len(candles_5m) < MIN_CANDLES_ENTRY:
                skip_counts["candles_15m"] += 1  # reuso counter
                continue

            def to_df(candles):
                df = pd.DataFrame(candles)
                df.columns = ['t','T','s','i','o','c','h','l','v','n']
                df[['o','c','h','l','v']] = df[['o','c','h','l','v']].astype(float)
                df.rename(columns={'o':'open','c':'close','h':'high','l':'low','v':'volume'}, inplace=True)
                return df

            df_1h  = to_df(candles_1h)
            df_15m = to_df(candles_15m)
            df_5m  = to_df(candles_5m)

            # Filtro volume su 15m: 96 candele = 24h
            vol_usd_24h  = df_15m['volume'].iloc[-96:].sum() * float(df_15m['close'].iloc[-1])
            if vol_usd_24h < MIN_VOLUME_24H_USD:
                skip_counts["volume"] += 1
                continue

            real_px = float(df_5m['close'].iloc[-1])
            if real_px > 0:
                min_usd_real = (real_px * 10 ** (-sz_dec_proc)) / LEVERAGE
                if min_usd_real > TRADE_SIZE_USD:
                    continue

            # Feature engineering su tutti e 3 i TF
            df_1h  = build_features(df_1h)
            df_15m = build_features(df_15m)
            df_5m  = build_features(df_5m)
            if df_1h.empty or df_15m.empty or df_5m.empty:
                continue
            df_5m = alt_inject_funding(df_5m, coin)

            # ── ALIAS TF ──────────────────────────────────────────────
            # TREND  (1h):  direzione primaria, regime, EMA200, slope, MACD
            # SETUP  (15m): RSI, BB squeeze/expansion, volume relativo, confluence
            # ENTRY  (5m):  conferma precisa, timing, SL/TP, backtest
            r_1h   = df_1h.iloc[-1]    # trend
            r_15m  = df_15m.iloc[-1]   # setup
            r_5m   = df_5m.iloc[-1]    # entry
            df     = df_5m             # per backtest e swing SL (usa 5m)

            px  = float(r_5m['close'])          # prezzo corrente da 5m (più fresco)
            atr = float(r_5m['atr'])            # ATR da 5m per SL preciso
            vol = float(r_5m['volatility'])

            if vol < VOLATILITY_MIN:
                skip_counts["volatility"] += 1
                continue

            # ── INDICATORI TREND (1h) ─────────────────────────────────
            ema200_1h = float(r_1h['ema200'])
            ema50_1h  = float(r_1h['ema50'])
            rsi_1h    = float(r_1h['rsi'])
            slope_1h  = float(r_1h['ema20_slope'])
            macd_hist_1h = float(r_1h.get('macd_hist', 0))

            # ── INDICATORI SETUP (15m) ────────────────────────────────
            rsi_15m     = float(r_15m['rsi'])
            bb_pos_15m  = float(r_15m['bb_pos'])
            bb_width_15m = float(r_15m.get('bb_width', 0))
            bb_width_ma_15m = float(df_15m['bb_width'].rolling(20).mean().iloc[-1]) if len(df_15m) >= 20 else bb_width_15m
            vol_rel_15m = float(r_15m['vol_rel'])
            slope_15m   = float(r_15m['ema20_slope'])
            macd_hist_15m = float(r_15m.get('macd_hist', 0))
            hh_15m      = int(df_15m['hh'].iloc[-5:].sum()) if len(df_15m) >= 5 else 0
            ll_15m      = int(df_15m['ll'].iloc[-5:].sum()) if len(df_15m) >= 5 else 0

            # ── INDICATORI ENTRY (5m) ─────────────────────────────────
            rsi       = float(r_5m['rsi'])
            ema20     = float(r_5m['ema20'])
            bb_pos    = float(r_5m['bb_pos'])
            vol_rel   = float(r_5m['vol_rel'])
            slope     = float(r_5m['ema20_slope'])
            funding_z = float(r_5m['funding_z'])
            oi_change = float(r_5m['oi_change'])
            macd_hist_5m = float(r_5m.get('macd_hist', 0))
            mom_4_5m  = float(r_5m.get('momentum_4', 0))
            hh_5m     = int(df_5m['hh'].iloc[-5:].sum()) if len(df_5m) >= 5 else 0
            ll_5m     = int(df_5m['ll'].iloc[-5:].sum()) if len(df_5m) >= 5 else 0

            # Compatibilità variabili usate dopo
            ema200_4h = ema200_1h   # rinominato ma stessa funzione
            ema50_4h  = ema50_1h
            rsi_4h    = rsi_1h
            slope_4h  = slope_1h
            r_4h      = r_1h        # per consenso MTF e AI

            # ── FILTRI COMUNI ──
            if oi_change < -0.10:
                log_alt(f"[{coin}] Scartato: OI in forte calo ({oi_change:.3f})")
                continue

            is_btc_eth = coin in BTC_ETH_COINS
            bear_pump  = False
            range_pump = False

            # ══════════════════════════════════════════════════════════
            # STRATEGIA REGIME-ADAPTIVE
            # ══════════════════════════════════════════════════════════
            # BULL/BEAR → MOMENTUM/BREAKOUT: compra forza, vendi debolezza
            # RANGE     → REVERSAL: compra agli estremi bassi, vendi agli estremi alti
            #
            # 3 TIMEFRAME: TREND(1h) → SETUP(15m) → ENTRY(5m)
            bb_squeezed_15m  = bb_width_15m < bb_width_ma_15m * 0.7
            bb_expanding_15m = bb_width_15m > bb_width_ma_15m * 1.3

            if regime in ("BULL", "BEAR"):
                strategy = "MOMENTUM"

                # ── BREAKOUT LONG ──
                breakout_long = [
                    px > ema200_1h and macd_hist_1h > 0,
                    (bb_expanding_15m or bb_squeezed_15m) and vol_rel_15m >= 1.5 and hh_15m >= 1,
                    slope > 0 and macd_hist_5m > 0 and px > ema20,
                    hh_5m >= 1,
                    vol_rel >= 1.0,
                ]
                # ── MOMENTUM CONTINUATION LONG ──
                momentum_long = [
                    px > ema200_1h and slope_1h > 0.001,
                    40 <= rsi_15m <= 65 and macd_hist_15m > 0,
                    slope > 0 and rsi >= 45,
                    macd_hist_5m > 0,
                    rsi_1h < 75,
                ]
                # ── BREAKOUT SHORT ──
                breakout_short = [
                    px < ema200_1h and macd_hist_1h < 0,
                    (bb_expanding_15m or bb_squeezed_15m) and vol_rel_15m >= 1.5 and ll_15m >= 1,
                    slope < 0 and macd_hist_5m < 0 and px < ema20,
                    ll_5m >= 1,
                    vol_rel >= 1.0,
                ]
                # ── MOMENTUM CONTINUATION SHORT ──
                momentum_short = [
                    px < ema200_1h and slope_1h < -0.001,
                    35 <= rsi_15m <= 60 and macd_hist_15m < 0,
                    slope < 0 and rsi <= 55,
                    macd_hist_5m < 0,
                    rsi_1h > 25,
                ]

                bo_long_score  = sum(breakout_long)
                mo_long_score  = sum(momentum_long)
                bo_short_score = sum(breakout_short)
                mo_short_score = sum(momentum_short)

                long_score  = max(bo_long_score, mo_long_score)
                short_score = max(bo_short_score, mo_short_score)
                long_type   = "BREAKOUT" if bo_long_score >= mo_long_score else "MOMENTUM"
                short_type  = "BREAKOUT" if bo_short_score >= mo_short_score else "MOMENTUM"

            else:
                # ══════════════════════════════════════════════════════
                # RANGE → REVERSAL: compra al supporto, vendi alla resistenza
                # ══════════════════════════════════════════════════════
                strategy = "REVERSAL"
                body_ratio = float(r_5m.get('body_ratio', 0))

                # ── REVERSAL LONG: prezzo al bottom del range + segni inversione ──
                reversal_long = [
                    # SETUP 15m: RSI ipervenduto + BB bassa
                    rsi_15m < 35 and bb_pos_15m < -0.5,
                    # TREND 1h: prezzo non in crollo (slope non troppo negativo)
                    slope_1h > -0.003,
                    # ENTRY 5m: candela di inversione (corpo piccolo o rialzista)
                    body_ratio > -0.2 and slope > -0.001,
                    # ENTRY 5m: MACD sta girando (histogram meno negativo o positivo)
                    macd_hist_5m > macd_hist_15m * 0.5 or macd_hist_5m > 0,
                    # SETUP 15m: volume (conferma del test del supporto)
                    vol_rel_15m >= 0.5,
                ]

                # ── REVERSAL SHORT: prezzo al top del range + segni inversione ──
                reversal_short = [
                    rsi_15m > 65 and bb_pos_15m > 0.5,
                    slope_1h < 0.003,
                    body_ratio < 0.2 and slope < 0.001,
                    macd_hist_5m < macd_hist_15m * 0.5 or macd_hist_5m < 0,
                    vol_rel_15m >= 0.5,
                ]

                long_score  = sum(reversal_long)
                short_score = sum(reversal_short)
                long_type   = "REVERSAL"
                short_type  = "REVERSAL"

            # ── Logging tabella ──
            px_fmt = f"{px:.2f}" if px >= 1 else f"{px:.6f}"
            cat    = get_coin_category(coin)
            table_rows.append({
                "coin":       coin,
                "px":         px_fmt,
                "rsi":        rsi,
                "bb":         bb_pos,
                "vol":        vol_rel,
                "ema200":     "▲" if px > ema200_1h else "▼",
                "long":       long_score,
                "short":      short_score,
                "decision":   "WAIT",
                "is_btc_eth": is_btc_eth,
                "cat":        cat,
            })

            # Scarta segnali deboli: minimo 3/5
            if long_score < 3 and short_score < 3:
                continue

            # Volume minimo: usa 15m (più stabile) non 5m
            if vol_rel_15m < 0.1:
                skip_counts["vol_rel"] += 1
                continue

            # ── BLOCCO DIREZIONALE ───────────────────────────────
            if regime == "BULL":
                short_score = 0
            if regime == "BEAR":
                long_score = 0

            # RANGE: blocca controtendenza solo se BTC ha momentum FORTE e COERENTE
            # tra 1h e 4h. Un dip 1h con 4h positivo = noise, non bloccare.
            if regime == "RANGE":
                # BTC sale forte: 1h E 4h entrambi positivi sopra soglia
                btc_strong_up   = btc_momentum > 0.003 and btc_mom_4h > 0.002
                # BTC scende forte: 1h E 4h entrambi negativi
                btc_strong_down = btc_momentum < -0.003 and btc_mom_4h < -0.002
                if btc_strong_up:
                    short_score = 0
                if btc_strong_down:
                    long_score = 0

            # Scegli direzione
            if long_score >= 3 and long_score > short_score:
                direction     = "LONG"
                confluence_n  = long_score
                signal_type   = long_type
            elif short_score >= 3 and short_score > long_score:
                direction     = "SHORT"
                confluence_n  = short_score
                signal_type   = short_type
            else:
                continue

            # ── FILTRO SLOPE 1h: blocca controtendenza forte ─────────
            if direction == "SHORT" and slope_4h > 0.002:
                continue
            if direction == "LONG" and slope_4h < -0.002:
                continue

            # ── BTC DECOUPLING: blocca solo controtendenza forte ──
            btc_dir = "BULL" if btc_momentum > 0.003 else "BEAR" if btc_momentum < -0.003 else "NEUTRAL"

            can_trade = True  # default: trada
            # Blocca SOLO se controtendenza forte a BTC
            if btc_dir == "BULL" and direction == "SHORT":
                can_trade = False
            elif btc_dir == "BEAR" and direction == "LONG":
                can_trade = False
            # BTC NEUTRAL → trada sempre (ALT hanno dinamiche proprie)

            if not can_trade:
                continue

            recent_closes = df['close'].iloc[-12:].tolist()

            # ══════════════════════════════════════════════════════════
            # DOUBLE CONFIRMATION + BACKTEST + RANKING
            # ══════════════════════════════════════════════════════════
            candidate = get_candidate(coin)
            prev_dir  = candidate.get("direction")
            prev_ts   = candidate.get("ts", 0)
            age       = time.time() - prev_ts
            if age > PROCESSOR_INTERVAL * 3:
                candidate = {}

            if not candidate or prev_dir != direction:
                # Prima vista: backtest con la mode che verrà usata
                pre_mode = detect_trading_mode(df, regime, vol_rel)
                log_alt(f"[{coin}] Prima vista [{direction}] [{signal_type}] [{pre_mode}] — backtest...")
                bt = quick_backtest(df, direction, coin, mode=pre_mode, strategy=strategy)
                log_alt(f"[{coin}] BT edge: WR:{bt['win_rate']:.1%} PF:{bt['profit_factor']:.2f} N:{bt['n_trades']} | "
                      f"BT regime: WR:{bt['win_rate_recent']:.1%} PF:{bt['profit_factor_recent']:.2f} N:{bt['n_trades_recent']}")

                # ── LIVELLO 1: EDGE STRUTTURALE ───────────────────────
                # Minimo trade per fidarsi del risultato
                if bt["n_trades"] < MIN_BACKTEST_TRADES:
                    log_alt(f"[{coin}] ❌ Edge: solo {bt['n_trades']} trade (min {MIN_BACKTEST_TRADES})")
                    clear_candidate(coin)
                    continue

                pf_full = bt["profit_factor"]
                cycle_wr_samples.append(pf_full)

                if len(cycle_wr_samples) >= 3:
                    dynamic_threshold = max(MIN_PROFIT_FACTOR_FLOOR, float(np.median(cycle_wr_samples)) - 0.1)
                else:
                    dynamic_threshold = MIN_PROFIT_FACTOR

                if pf_full < dynamic_threshold:
                    log_alt(f"[{coin}] ❌ Edge PF:{pf_full:.2f} < {dynamic_threshold:.2f} — no edge")
                    clear_candidate(coin)
                    continue

                # ── LIVELLO 2: REGIME CHECK ───────────────────────────
                # L'edge storico è buono, ma il regime attuale lo supporta?
                # Se gli ultimi 50 trade hanno PF < 0.8 → mercato cambiato, skip
                pf_recent = bt["profit_factor_recent"]
                if bt["n_trades_recent"] >= 10 and pf_recent < 0.8:
                    log_alt(f"[{coin}] ❌ Regime sfavorevole: PF recente {pf_recent:.2f} < 0.8 (edge storico ok: {pf_full:.2f})")
                    clear_candidate(coin)
                    continue

                # MTF check rimosso — il backtest PF > 1.2 è la vera validazione
                # slope_4h in un mercato flat/down blocca tutti i LONG
                # anche quelli con edge reale (HEMI PF:1.84, RESOLV PF:1.37)

                set_candidate(coin, {
                    "direction":        direction,
                    "confluence":       confluence_n,
                    "strategy":         strategy,
                    "signal_type":      signal_type,
                    "mode":             pre_mode,
                    "win_rate":         bt["win_rate"],
                    "win_rate_recent":  bt["win_rate_recent"],
                    "profit_factor":    bt["profit_factor"],
                    "profit_factor_recent": bt["profit_factor_recent"],
                    "n_trades":         bt["n_trades"],
                    "avg_return":       bt.get("avg_return", 0),
                    "ts":               int(time.time())
                })

                # PF >= 1.2 → skip double confirmation
                if bt["profit_factor"] >= 1.2:
                    log_alt(f"[{coin}] ⚡ PF:{bt['profit_factor']:.2f} >= 1.2 — skip double confirm")
                    candidate = get_candidate(coin)  # reload with backtest data
                else:
                    log_alt(f"[{coin}] ⏳ Candidato salvato [{signal_type}] [{pre_mode}]")
                    continue

            # ── Seconda vista: confermato → AI + score → aggiungi a valid_signals ──
            log_alt(f"[{coin}] ✅ Confermato [{direction}] [{candidate.get('signal_type', '?')}]")

            wr_hist   = candidate.get("win_rate", 0)
            wr_recent = candidate.get("win_rate_recent", 0)
            pf_hist   = candidate.get("profit_factor", 0)

            if pf_hist > 0 and pf_hist < 1.0:
                skip_counts["wr_recente"] += 1
                log_alt(f"[{coin}] ❌ PF:{pf_hist:.2f} < 1.0 — skip")
                clear_candidate(coin)
                continue

            ema50_val  = float(df_1h['ema50'].iloc[-1])
            last       = df_15m.iloc[-1]
            spread_pct = float((last['high'] - last['low']) / (last['close'] + 1e-10))
            liq_data   = fetch_liquidity_data(coin, px)

            ai_valid, ai_score, ai_reason, _, _ = ai_validate_signal(
                coin, direction, regime, rsi, bb_pos, vol_rel,
                funding_z, oi_change, confluence_n, recent_closes,
                df=df, strategy=strategy,
                liq_data=liq_data, pf_hist=pf_hist,
                wr_recent=wr_recent, wr_hist=wr_hist,
                df_4h=df_1h, mids=_all_mids_cache,
                active_positions=_active_positions_cache
            )

            # AI è soft filter — non blocca, scala size
            # ai_score < 3 → size ×0.5, ai_score 3-5 → size ×0.7, 6+ → size ×1.0
            if not ai_valid:
                log_alt(f"[{coin}] ⚠️ AI scettica — {ai_reason} (size ridotta, non bloccata)")
                ai_score = max(ai_score, 2)  # floor a 2 per non bloccare
            ai_size_mult = 1.0
            if ai_score < 3:
                ai_size_mult = 0.5
            elif ai_score < 6:
                ai_size_mult = 0.7

            setup_score = compute_setup_score(
                direction, px, ema50_val, ema200_4h,
                rsi, spread_pct, ai_score, funding_z, liq=liq_data
            )
            # PF >= 1.2 dal backtest → setup score irrilevante
            if pf_hist >= 1.2:
                setup_score = max(setup_score, 60)
            if setup_score < SETUP_SCORE_MIN:
                skip_counts["score"] += 1
                log_alt(f"[{coin}] ❌ Setup score {setup_score} < {SETUP_SCORE_MIN}")
                clear_candidate(coin)
                continue

            mode = candidate.get("mode", detect_trading_mode(df, regime, vol_rel))

            # ── RANKING SCORE COMPOSITO ──────────────────────────────
            # Combina profit_factor, win_rate (precision), e avg_return
            # per scegliere il MIGLIORE segnale tra tutti i validi
            avg_return  = candidate.get("avg_return", 0)
            rank_score  = (
                pf_hist    * 0.5 +    # profit factor: quanto guadagna per $ rischiato
                wr_hist    * 0.3 +    # precision: % trade vincenti
                avg_return * 0.2      # avg return: rendimento medio per trade
            )

            sl, tp     = dynamic_sl_tp(px, atr, vol, direction, df, coin, ai_score, mode)
            trail_stop = compute_trailing_stop(px, atr, direction, ai_score, mode)
            sl_dist    = abs(px - sl)

            # V7: detect scalp mode from ADX
            adx_val = float(df.iloc[-1].get('adx', 20)) if 'adx' in df.columns else 20
            scalp_mode = detect_scalp_mode(adx_val, vol_rel, bb_pos, regime, funding_z)

            signal_type_final = candidate.get("signal_type", "MOMENTUM")
            clear_candidate(coin)

            valid_signals.append({
                "coin":          coin,
                "direction":     direction,
                "signal_type":   signal_type_final,
                "mode":          mode,
                "scalp_mode":    scalp_mode,
                "rank_score":    round(rank_score, 4),
                "setup_score":   setup_score,
                "ai_score":      ai_score,
                "ai_reason":     ai_reason,
                "pf":            pf_hist,
                "wr":            wr_hist,
                "avg_return":    avg_return,
                "confluence":    confluence_n,
                "sl":            sl,
                "tp":            tp,
                "sl_dist":       sl_dist,
                "trailing_stop": trail_stop,
                "px":            px,
                "atr":           atr,
                "rsi":           rsi,
                "bb_pos":        bb_pos,
                "vol_rel":       vol_rel,
                "funding_z":     funding_z,
                "regime":        regime,
                "strategy":      strategy,
                "is_btc_eth":    is_btc_eth,
            })

        except TimeoutError:
            log_err(f"[{coin}] timeout — skip")
        except Exception as e:
            import traceback
            log_err(f"[{coin}] processor: {e}\n{traceback.format_exc()}")

    # ══════════════════════════════════════════════════════════════
    # RANKING: seleziona e pubblica il MIGLIORE segnale
    # ══════════════════════════════════════════════════════════════
    if valid_signals:
        # Ordina per rank_score decrescente
        valid_signals.sort(key=lambda x: x["rank_score"], reverse=True)

        # Log tutti i segnali validi con ranking
        log_alt(f"── RANKING: {len(valid_signals)} segnali validi ──")
        for i, sig in enumerate(valid_signals):
            log_alt(f"  #{i+1} {sig['coin']:<8} [{sig['direction']}] [{sig['signal_type']}] [{sig['scalp_mode']}] "
                  f"rank:{sig['rank_score']:.3f} PF:{sig['pf']:.2f} WR:{sig['wr']:.1%} "
                  f"AI:{sig['ai_score']}/10 setup:{sig['setup_score']}/100")

        # Pubblica il migliore
        best = valid_signals[0]
        coin_best = best["coin"]

        signal_data = {
            "score":           best["setup_score"],
            "sl":              best["sl"],
            "tp":              best["tp"],
            "sl_dist":         best["sl_dist"],
            "trailing_stop":   best["trailing_stop"],
            "direction":       best["direction"],
            "signal_px":       best["px"],
            "funding_z":       round(best["funding_z"], 3),
            "confluence":      best["confluence"],
            "win_rate":        best["wr"],
            "profit_factor":   best["pf"],
            "win_rate_recent": 0,
            "signal_max_age":  BTC_SIGNAL_MAX_AGE if best["is_btc_eth"] else SIGNAL_MAX_AGE,
            "mode":            best["mode"],
            "scalp_mode":      best["scalp_mode"],
            "strategy":        best["strategy"],
            "regime":          best["regime"],
            "ai_score":        best["ai_score"],
            "setup_score":     best["setup_score"],
            "rsi":             round(best["rsi"], 1),
            "bb_pos":          round(best["bb_pos"], 3),
            "bear_pump":       False,
            "range_pump":      False,
            "signal_type":     best["signal_type"],
            "rank_score":      best["rank_score"],
            "ts":              int(time.time())
        }
        publish_signal(coin_best, signal_data)
        set_cooldown(coin_best, strategy=best["strategy"])

        log_alt(f"🏆 BEST SIGNAL: {coin_best} [{best['direction']}] [{best['signal_type']}] [{best['mode']}] "
              f"rank:{best['rank_score']:.3f} PF:{best['pf']:.2f} AI:{best['ai_score']}/10 "
              f"SL:{best['sl']} TP:{best['tp']}")
        tg(
            f"{'🟢' if best['direction']=='LONG' else '🔴'} <b>{coin_best}</b> "
            f"[{best['direction']}] [{best['signal_type']}] [{best['mode']}] 🏆\n"
            f"Rank: <b>{best['rank_score']:.3f}</b> | Score: {best['setup_score']}/100 | AI: {best['ai_score']}/10\n"
            f"PF: <b>{best['pf']:.2f}</b> | WR: {best['wr']:.1%} | Regime: {best['regime']}\n"
            f"RSI:{best['rsi']:.1f} | BB:{best['bb_pos']:.2f} | Vol:{best['vol_rel']:.2f}x | FZ:{best['funding_z']:+.2f}\n"
            f"💬 {best['ai_reason']}\n"
            f"SL: {best['sl']} | TP: {best['tp']}"
        )
        sent = 1
    else:
        log_alt("Nessun segnale valido questo ciclo")

    # ── Stampa tabella riassuntiva ──
    now_utc    = datetime.now(timezone.utc).strftime("%H:%M UTC")
    W          = 74
    btc_eth_rows = [r for r in table_rows if r["is_btc_eth"]]
    other_rows   = [r for r in table_rows if not r["is_btc_eth"]]

    from collections import defaultdict
    by_cat = defaultdict(list)
    for r in other_rows:
        by_cat[r["cat"]].append(r)

    lines = []
    lines.append("=" * W)
    lines.append(f"  {'COIN':<10} {'PREZZO':>12}  {'RSI':>5}  {'BB':>6}  {'VOL':>5}  {'EMA':>3}  {'L':>3} {'S':>3}  DECISIONE")
    lines.append("-" * W)
    lines.append("  ── BTC / ETH ──")
    for r in btc_eth_rows:
        bb_s = f"{r['bb']:+.2f}"
        lines.append(f"  {r['coin']:<10} {r['px']:>12}  {r['rsi']:>5.1f}  {bb_s:>6}  {r['vol']:>5.2f}    {r['ema200']}   {r['long']}/5   {r['short']}/5  {r['decision']}")
    for cat in ["midcap", "meme", "other"]:
        cat_rows = sorted(by_cat.get(cat, []),
                          key=lambda x: (x["decision"] not in ("LONG ✅","SHORT ✅"),
                                         -abs(x["long"] - x["short"])))[:3]
        if not cat_rows:
            continue
        lines.append(f"  ── {cat.upper()} (top 3) ──")
        for r in cat_rows:
            bb_s = f"{r['bb']:+.2f}"
            lines.append(f"  {r['coin']:<10} {r['px']:>12}  {r['rsi']:>5.1f}  {bb_s:>6}  {r['vol']:>5.2f}    {r['ema200']}   {r['long']}/5   {r['short']}/5  {r['decision']}")
    lines.append("-" * W)
    lines.append(f"  Regime: {regime} | Universe: {len(universe)} | Segnali: {sent} | {now_utc}")
    lines.append("=" * W)
    log_alt("\n".join(lines))

    log_alt(f"=== FINE | Regime:{regime} | Segnali:{sent} ===")
    log_alt(
        f"Skip: 4h:{skip_counts['candles_4h']} 15m:{skip_counts['candles_15m']} "
        f"vol:{skip_counts['volume']} vola:{skip_counts['volatility']} "
        f"rvol:{skip_counts['vol_rel']} conf5:{skip_counts['confluence5']} "
        f"wr:{skip_counts['wr_recente']} fund:{skip_counts['funding']} "
        f"ai:{skip_counts['ai']} score:{skip_counts['score']}"
    )


# ================================================================
# EXECUTOR — ACCOUNT / POSIZIONI / ORDINI
# ================================================================

def fetch_meta() -> tuple[dict, dict]:
    for attempt in range(3):
        try:
            meta   = call(_info.meta, label='meta', timeout=15)
            sz_dec = {a["name"]: int(a["szDecimals"])           for a in meta["universe"]}
            px_dec = {a["name"]: int(a.get("priceDecimals", 2)) for a in meta["universe"]}
            log_exec(f"Meta: {len(sz_dec)} coin")
            return sz_dec, px_dec
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                time.sleep(3 * (attempt + 1))
            else:
                log_err(f"fetch_meta: {e}")
                return {}, {}
    return {}, {}

def get_open_positions() -> dict:
    positions = {}
    for attempt in range(3):
        try:
            state = call(_info.user_state, account.address, label='user_state', timeout=15)
            for p in state.get("assetPositions", []):
                pos  = p.get("position", {})
                szi  = float(pos.get("szi", 0))
                coin = pos.get("coin", "")
                if szi != 0 and coin:
                    positions[coin] = {
                        "szi":      szi,
                        "entry_px": float(pos.get("entryPx", 0) or 0)
                    }
            return positions
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                time.sleep(3 * (attempt + 1))
            else:
                log_err(f"get_open_positions: {e}")
                return positions
    return positions

def get_open_trigger_orders(coin: str) -> list:
    """Ritorna gli ordini trigger aperti (SL/TP) per una coin."""
    try:
        orders = call(_info.open_orders, account.address, label=f'open_orders_{coin}', timeout=10)
        return [o for o in (orders or []) if o.get("coin") == coin and o.get("orderType") in ("Stop Market", "Take Profit Market")]
    except Exception as e:
        log_err(f"get_open_trigger_orders [{coin}]: {e}")
        return []

def cancel_order(coin: str, oid: int) -> bool:
    try:
        res = call(_exchange.cancel, str(coin), oid, label=f'cancel_{coin}', timeout=10)
        return res and res.get("status") == "ok"
    except Exception as e:
        log_err(f"cancel_order [{coin}/{oid}]: {e}")
        return False

def get_account_balance() -> float:
    for attempt in range(3):
        try:
            state = call(_info.user_state, account.address, label='balance', timeout=15)
            return float(state.get("marginSummary", {}).get("accountValue", 0) or 0)
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                time.sleep(3 * (attempt + 1))
            else:
                log_err(f"get_account_balance: {e}")
                return 0.0
    return 0.0


def get_effective_leverage(coin: str) -> int:
    """
    Legge la leva effettiva dal user_state di Hyperliquid.
    Se la coin non ha una posizione/leva impostata, ritorna il default.
    Hyperliquid può rifiutare silenziosamente leve troppo alte e applicare il max consentito.
    """
    try:
        state = call(_info.user_state, account.address, label=f'lev_check_{coin}', timeout=10)
        for p in state.get("assetPositions", []):
            pos = p.get("position", {})
            if pos.get("coin") == coin:
                lev_info = pos.get("leverage", {})
                # Hyperliquid ritorna {"type": "cross"|"isolated", "value": N}
                if isinstance(lev_info, dict):
                    return int(lev_info.get("value", LEVERAGE))
                elif isinstance(lev_info, (int, float)):
                    return int(lev_info)
        # Nessuna posizione attiva — check nella sezione crossMarginSummary
        # Fallback: ritorna il default
        return int(LEVERAGE)
    except Exception as e:
        log_err(f"[{coin}] get_effective_leverage: {e}")
        return int(LEVERAGE)


# ================================================================
# EXECUTOR — FILL HISTORY (outcome preciso)
# ================================================================

def get_recent_fills(coin: str, since_ts: float) -> list:
    """
    Recupera i fill recenti per una coin dall'API Hyperliquid.
    Ritorna lista di fill con px, sz, side, time.
    """
    try:
        fills = call(
            _info.user_fills_by_time,
            account.address,
            int(since_ts * 1000),  # Hyperliquid vuole ms
            timeout=15, label=f'fills_{coin}'
        )
        if not fills:
            return []
        return [f for f in fills if f.get("coin") == coin]
    except Exception as e:
        log_err(f"[{coin}] get_recent_fills: {e}")
        return []


def compute_real_exit(coin: str, direction: str, entry_px: float,
                      ts_open: float) -> tuple[float, float, str]:
    """
    Calcola exit_px e PnL reali dai fill.
    Ritorna (exit_px, pnl_pct, outcome).
    Fallback al prezzo mid se i fill non sono disponibili.
    """
    try:
        fills = get_recent_fills(coin, ts_open)
        # Fill di chiusura: side opposta alla direzione
        close_side = "B" if direction == "SHORT" else "A"  # Buy chiude Short, Ask(Sell) chiude Long
        close_fills = [f for f in fills
                       if f.get("side", "")[0:1].upper() == close_side
                       and f.get("dir", "") == "Close"]

        if not close_fills:
            # Fallback: prendi tutti i fill del lato opposto
            close_fills = [f for f in fills
                           if f.get("side", "")[0:1].upper() == close_side]

        if close_fills:
            # Media ponderata dei fill di chiusura
            total_sz = sum(float(f.get("sz", 0)) for f in close_fills)
            if total_sz > 0:
                exit_px = sum(float(f.get("px", 0)) * float(f.get("sz", 0))
                              for f in close_fills) / total_sz
            else:
                return 0.0, 0.0, "unknown"

            if entry_px > 0:
                pnl_pct = ((exit_px - entry_px) / entry_px) if direction == "LONG" \
                          else ((entry_px - exit_px) / entry_px)
                outcome = "win" if pnl_pct > 0 else "loss"
            else:
                pnl_pct = 0.0
                outcome = "unknown"

            return exit_px, pnl_pct, outcome

    except Exception as e:
        log_err(f"[{coin}] compute_real_exit: {e}")

    return 0.0, 0.0, "unknown"


# ================================================================
# EXECUTOR — GESTIONE PENDENTI
# ================================================================

def cleanup_pending_orders(active_positions: dict):
    now       = time.time()
    to_remove = []
    for coin, info_p in pending_orders.items():
        if coin in active_positions:
            log_exec(f"[{coin}] Pendente → FILL")
            to_remove.append(coin)
            continue
        if now - info_p["placed_at"] > PENDING_ORDER_TTL:
            log_exec(f"[{coin}] Pendente scaduto → cancello")
            oid = info_p.get("oid")
            if oid and cancel_order(coin, oid):
                tg(f"⏱ <b>{coin}</b> ordine scaduto — cancellato", silent=True)
            to_remove.append(coin)
    for coin in to_remove:
        pending_orders.pop(coin, None)

def process_pending_orders(active_positions: dict, sz_dec: dict, px_dec: dict) -> int:
    """Ritorna il numero di pending andati a fill in questo ciclo."""
    filled_count = 0
    for coin in [c for c in pending_orders if c in active_positions]:
        info_p      = pending_orders.pop(coin)
        signal      = info_p["signal"]
        direction   = signal.get("direction", "LONG")
        is_buy      = direction == "LONG"
        filled_size = abs(active_positions[coin]["szi"])
        px_d        = info_p.get("px_dec", px_dec.get(coin, 2))
        sz_d        = info_p.get("sz_dec", sz_dec.get(coin, 2))
        sl_px       = round_to_decimals(float(signal.get("sl", 0)), px_d)
        tp_px       = round_to_decimals(float(signal.get("tp", 0)), px_d)
        actual_size = round_to_decimals(filled_size, sz_d)
        log_exec(f"[{coin}] Pendente eseguito → SL/TP size:{actual_size}")
        try:
            call(_exchange.order, str(coin), not is_buy, actual_size, sl_px,
                 {"trigger": {"triggerPx": sl_px, "isMarket": True, "tpsl": "sl"}},
                 True, timeout=15, label=f'sl_pend_{coin}')
            time.sleep(0.3)
            call(_exchange.order, str(coin), not is_buy, actual_size, tp_px,
                 {"trigger": {"triggerPx": tp_px, "isMarket": True, "tpsl": "tp"}},
                 True, timeout=15, label=f'tp_pend_{coin}')
            tg(f"🔒 <b>{coin}</b> [{direction}] protetto | SL:{sl_px} TP:{tp_px}", silent=True)
            delete_signal(coin)
            filled_count += 1
        except Exception as e:
            log_err(f"[{coin}] protezioni pendente: {e}")
    return filled_count


# ================================================================
# EXECUTOR — OPEN TRADE
# ================================================================

def open_trade(coin, signal, mids, sz_dec, px_dec) -> bool:
    direction = signal.get("direction")
    if not direction:
        return False
    is_buy = (direction == "LONG")

    balance = get_account_balance()
    if balance < (TRADE_SIZE_USD) * 1.1:
        log_err(f"[{coin}] Balance ${balance:.2f} insufficiente")
        return False

    mid_px = float(mids.get(coin, 0) or 0)
    if mid_px <= 0:
        log_err(f"[{coin}] Prezzo non disponibile")
        return False

    # ── PRICE DRIFT CHECK ─────────────────────────────────────────
    # Il segnale è stato generato a signal_px. Se il prezzo si è già mosso
    # troppo nella direzione del trade, il movimento è avvenuto e il R:R peggiora.
    # Se si è mosso nella direzione opposta, il segnale potrebbe essere invalidato.
    signal_px = float(signal.get("signal_px", 0) or 0)
    if signal_px > 0:
        drift_pct = (mid_px - signal_px) / signal_px

        if direction == "LONG":
            if drift_pct > 0.008:
                # Prezzo salito >0.8% da quando è stato generato il segnale
                # Il movimento è già avvenuto — entry tardiva con R:R peggiore
                log_exec(f"[{coin}] ❌ LONG annullato: prezzo salito {drift_pct:+.2%} dal segnale (entry tardiva)")
                delete_signal(coin)
                return False
            if drift_pct < -0.015:
                # Prezzo sceso >1.5% — il segnale è invalidato
                log_exec(f"[{coin}] ❌ LONG annullato: prezzo sceso {drift_pct:+.2%} dal segnale (invalidato)")
                delete_signal(coin)
                return False
        else:  # SHORT
            if drift_pct < -0.008:
                log_exec(f"[{coin}] ❌ SHORT annullato: prezzo sceso {drift_pct:+.2%} dal segnale (entry tardiva)")
                delete_signal(coin)
                return False
            if drift_pct > 0.015:
                log_exec(f"[{coin}] ❌ SHORT annullato: prezzo salito {drift_pct:+.2%} dal segnale (invalidato)")
                delete_signal(coin)
                return False

        # Se il prezzo si è mosso poco ma nella giusta direzione,
        # ricalcola SL/TP dal prezzo corrente per avere target freschi
        if abs(drift_pct) > 0.002:
            sl_raw = float(signal.get("sl", 0) or 0)
            tp_raw = float(signal.get("tp", 0) or 0)
            if sl_raw > 0 and tp_raw > 0:
                sl_dist_orig = abs(signal_px - sl_raw)
                tp_dist_orig = abs(tp_raw - signal_px)

                # Ricalcola dal prezzo corrente mantenendo le distanze
                if direction == "LONG":
                    signal["sl"] = round(mid_px - sl_dist_orig, 6)
                    signal["tp"] = round(mid_px + tp_dist_orig, 6)
                else:
                    signal["sl"] = round(mid_px + sl_dist_orig, 6)
                    signal["tp"] = round(mid_px - tp_dist_orig, 6)

                log_exec(f"[{coin}] SL/TP ricalcolati da prezzo corrente (drift {drift_pct:+.2%})")

    sl_raw = float(signal.get("sl", 0) or 0)
    tp_raw = float(signal.get("tp", 0) or 0)
    ts_raw = float(signal.get("trailing_stop", 0) or 0)
    if sl_raw <= 0 or tp_raw <= 0:
        log_err(f"[{coin}] SL/TP non validi")
        return False

    # Calcola decimali minimi necessari: se il prezzo è 0.048 e px_dec=2,
    # arrotondare a 2 decimali collassa entry/SL/TP a 0.05.
    # Serve almeno abbastanza decimali per distinguere entry, SL e TP.
    def min_decimals_for_price(px: float) -> int:
        """Calcola i decimali minimi per cui round(px) != 0 e round(px) mantiene significatività."""
        if px <= 0:
            return 6
        for d in range(0, 10):
            rounded = round(px, d)
            if rounded > 0 and abs(rounded - px) / px < 0.005:  # errore < 0.5%
                return d
        return 6

    # Usa il massimo tra px_dec dall'API e il minimo calcolato dal prezzo
    effective_px_dec = max(px_dec, min_decimals_for_price(mid_px))
    # Verifica che SL e TP rimangano distinti dopo arrotondamento
    sl_test = round_to_decimals(sl_raw, effective_px_dec)
    tp_test = round_to_decimals(tp_raw, effective_px_dec)
    entry_test = round_to_decimals(mid_px, effective_px_dec)
    if sl_test == entry_test or tp_test == entry_test:
        effective_px_dec = min(effective_px_dec + 2, 8)
        log_exec(f"[{coin}] px_dec aumentato a {effective_px_dec} per distinguere SL/TP")

    entry_px = round_to_decimals(mid_px, effective_px_dec)
    sl_px    = round_to_decimals(sl_raw, effective_px_dec)
    tp_px    = round_to_decimals(tp_raw, effective_px_dec)

    if is_buy  and (sl_px >= entry_px or tp_px <= entry_px):
        log_err(f"[{coin}] LONG SL/TP incoerenti (e:{entry_px} sl:{sl_px} tp:{tp_px})")
        return False
    if not is_buy and (sl_px <= entry_px or tp_px >= entry_px):
        log_err(f"[{coin}] SHORT SL/TP incoerenti (e:{entry_px} sl:{sl_px} tp:{tp_px})")
        return False

    # Size calcolata sul nozionale — $4 margine × 20x leva — Hyperliquid min $10 nominale
    size_nominal = round_to_decimals((TRADE_SIZE_USD * LEVERAGE) / entry_px, sz_dec)
    if size_nominal <= 0:
        return False

    if size_nominal * entry_px < 10.0:
        log_err(f"[{coin}] Size nominale insufficiente ({size_nominal} * {entry_px:.4f} = {size_nominal * entry_px:.2f})")
        return False

    log_exec(f"[{coin}] {direction} entry:{entry_px} sl:{sl_px} tp:{tp_px} size:{size_nominal}")

    try:
        # ── Imposta leva e verifica quella effettiva ──────────────
        for attempt in range(3):
            try:
                call(_exchange.update_leverage, int(LEVERAGE), str(coin), is_cross=False,
                     timeout=10, label=f'lev_{coin}')
                break
            except Exception as e:
                if "429" in str(e) and attempt < 2:
                    time.sleep(3 * (attempt + 1))
                else:
                    raise
        time.sleep(1.0)

        # Verifica leva effettiva: Hyperliquid può applicare un max inferiore
        actual_lev = get_effective_leverage(coin)
        if actual_lev != LEVERAGE:
            lev_scale = LEVERAGE / max(actual_lev, 1)
            log_exec(f"[{coin}] ⚠️ Leva effettiva {actual_lev}x (richiesta {LEVERAGE}x)")

            # SL scala con leva piena (protezione dal noise)
            sl_dist_old = abs(entry_px - sl_px)
            sl_dist_new = sl_dist_old * lev_scale

            # TP: mantieni lo stesso R:R del segnale originale, con cap % prezzo
            tp_dist_old = abs(tp_px - entry_px)
            rr_original = tp_dist_old / sl_dist_old if sl_dist_old > 0 else 1.1
            tp_dist_new = sl_dist_new * rr_original  # stessa proporzionalità

            # Cap TP: scalping max 1.5%, swing max 5%
            trade_mode = signal.get("mode", "SCALPING")
            tp_cap = entry_px * (0.015 if trade_mode == "SCALPING" else 0.05)
            tp_dist_new = min(tp_dist_new, tp_cap)

            if is_buy:
                sl_px = round_to_decimals(entry_px - sl_dist_new, effective_px_dec)
                tp_px = round_to_decimals(entry_px + tp_dist_new, effective_px_dec)
            else:
                sl_px = round_to_decimals(entry_px + sl_dist_new, effective_px_dec)
                tp_px = round_to_decimals(entry_px - tp_dist_new, effective_px_dec)

            # Ricalcola anche trailing stop (scala come SL)
            if ts_raw > 0:
                ts_dist_old = abs(entry_px - round_to_decimals(ts_raw, effective_px_dec))
                ts_raw = entry_px - ts_dist_old * lev_scale if is_buy else entry_px + ts_dist_old * lev_scale

            # Ricalcola size con leva effettiva
            size_nominal = round_to_decimals((TRADE_SIZE_USD * actual_lev) / entry_px, sz_dec)
            if size_nominal <= 0 or size_nominal * entry_px < 10.0:
                log_err(f"[{coin}] Size insufficiente con leva {actual_lev}x: {size_nominal * entry_px:.2f}")
                return False

            sl_pct = abs(entry_px - sl_px) / entry_px * 100
            tp_pct = abs(tp_px - entry_px) / entry_px * 100
            log_exec(f"[{coin}] Leva {actual_lev}x: SL:{sl_pct:.2f}% TP:{tp_pct:.2f}% R:R=1:{rr_original:.1f} size:{size_nominal}")
            tg(f"⚠️ <b>{coin}</b> leva {actual_lev}x — SL:{sl_pct:.1f}% TP:{tp_pct:.1f}%", silent=True)

            # Aggiorna il segnale con i nuovi valori per trailing/exit AI
            signal["sl"] = sl_px
            signal["tp"] = tp_px
            if ts_raw > 0:
                signal["trailing_stop"] = ts_raw
            signal["effective_leverage"] = actual_lev

        time.sleep(0.5)

        # ── MARGIN CHECK (from V7) ──
        if not check_margin_ok(coin, size_nominal, entry_px):
            return False

        # ── GTC MAKER → IoC TAKER (from V7) ──
        scalp_mode = signal.get("scalp_mode", "TREND")
        res = None
        filled = False

        # Step 1: GTC Maker (skip in FLASH)
        if scalp_mode != "FLASH":
            try:
                gtc_px = round_to_decimals(
                    entry_px * (1 + SLIPPAGE) if is_buy else entry_px * (1 - SLIPPAGE),
                    effective_px_dec)
                res = call(_exchange.order, str(coin), is_buy, size_nominal, gtc_px,
                           {"limit": {"tif": "Gtc"}}, False,
                           timeout=15, label=f'gtc_{coin}')
                log_exec(f"[{coin}] GTC Maker @ {gtc_px}: {res}")

                oid_gtc = None
                if res and res.get("status") == "ok":
                    statuses = res.get("response",{}).get("data",{}).get("statuses",[])
                    for s in statuses:
                        if "filled" in s:
                            filled = True
                            log_exec(f"[{coin}] ✅ GTC instant fill (maker)")
                            break
                        if "resting" in s:
                            oid_gtc = s["resting"]["oid"]

                    if not filled and oid_gtc:
                        for tick in range(GTC_TIMEOUT * 2):
                            time.sleep(0.5)
                            pos = get_open_positions()
                            if coin in pos:
                                filled = True
                                log_exec(f"[{coin}] ✅ GTC filled in {(tick+1)*0.5:.1f}s")
                                break
                        if not filled:
                            try: cancel_order(coin, oid_gtc)
                            except: pass
                            log_exec(f"[{coin}] GTC not filled in {GTC_TIMEOUT}s")
            except Exception as e:
                log_err(f"[{coin}] GTC error: {e}")

        
        # Se non è stato riempito subito (filled), non facciamo lo Step 2 (Taker).
        # Semplicemente logghiamo che restiamo in attesa come Maker.
        if not filled:
            log_exec(f"[{coin}] Ordine GTC inserito. Resto in attesa nel book (Solo Maker).")

        oid = None
        try:
            if res is None:
                log_err(f"[{coin}] Order response is None — API failed")
            else:
                statuses = res.get("response", {}).get("data", {}).get("statuses", [])
                if statuses:
                    status_info = statuses[0]
                    oid = (status_info.get("resting", {}) or {}).get("oid") or \
                          (status_info.get("filled", {}) or {}).get("oid")
        except Exception as e:
            log_err(f"[{coin}] Errore nel recupero dell'OID: {e}")

        # Se non abbiamo nemmeno un OID, significa che l'ordine non è proprio partito
        if not oid and not filled:
            log_err(f"[{coin}] Errore critico: Ordine non inserito.")
            return False

        # Polling fill
        filled_size  = 0.0
        filled_entry = entry_px
        for _ in range(ENTRY_POLL_ATTEMPTS):
            time.sleep(ENTRY_POLL_INTERVAL)
            pos = get_open_positions()
            if coin in pos:
                filled_size  = abs(pos[coin]["szi"])
                filled_entry = pos[coin]["entry_px"] or entry_px
                log_exec(f"[{coin}] FILL size:{filled_size} @ {filled_entry}")
                break
        else:
            pending_orders[coin] = {
                "oid": oid, "placed_at": time.time(),
                "signal": signal, "sz_dec": sz_dec, "px_dec": px_dec
            }
            log_exec(f"[{coin}] ⏳ Pendente (oid={oid})")
            tg(f"⏳ <b>{coin}</b> [{direction}] pendente @ {entry_px}", silent=True)
            return "pending"

        actual_size = round_to_decimals(filled_size, sz_dec)

        sl_res = call(_exchange.order, str(coin), not is_buy, actual_size, sl_px,
                      {"trigger": {"triggerPx": sl_px, "isMarket": True, "tpsl": "sl"}},
                      True, timeout=15, label=f'sl_{coin}')
        time.sleep(0.3)
        tp_res = call(_exchange.order, str(coin), not is_buy, actual_size, tp_px,
                      {"trigger": {"triggerPx": tp_px, "isMarket": True, "tpsl": "tp"}},
                      True, timeout=15, label=f'tp_{coin}')

        sl_ok = sl_res and sl_res.get("status") == "ok"
        tp_ok = tp_res and tp_res.get("status") == "ok"

        if not sl_ok:
            log_err(f"[{coin}] ❌ SL NON INVIATO")
            tg(f"🚨 <b>{coin}</b> SL NON INVIATO — chiudi manualmente!")

        if ts_raw > 0:
            ts_px = round_to_decimals(ts_raw, effective_px_dec)
            if (is_buy and ts_px < entry_px) or (not is_buy and ts_px > entry_px):
                try:
                    # Cancella SL originale per evitare doppio SL sovrapposto
                    for o in get_open_trigger_orders(coin):
                        if o.get("orderType") == "Stop Market":
                            cancel_order(coin, o["oid"])
                            time.sleep(0.2)
                    call(_exchange.order, str(coin), not is_buy, actual_size, ts_px,
                         {"trigger": {"triggerPx": ts_px, "isMarket": True, "tpsl": "sl"}},
                         True, timeout=10, label=f'trail_{coin}')
                except Exception as e:
                    log_err(f"[{coin}] trailing: {e}")

        delete_signal(coin)
        log_exec(f"✅ [{coin}] APERTO [{direction}] @ {filled_entry} SL:{sl_px} TP:{tp_px}")
        tg(
            f"{'🟢' if is_buy else '🔴'} <b>FILL {direction}</b> {coin}\n"
            f"Entry: <b>{filled_entry}</b>\n"
            f"SL: {sl_px} {'✅' if sl_ok else '❌'} | TP: {tp_px} {'✅' if tp_ok else '❌'}\n"
            f"Size: ${TRADE_SIZE_USD} ({actual_size} contracts) | "
            f"Margine: ~${TRADE_SIZE_USD/LEVERAGE:.2f}"
        )
        return "filled"

    except TimeoutError as e:
        log_err(f"[{coin}] TIMEOUT: {e}")
        tg(f"⚠️ <b>{coin}</b> timeout — verifica manualmente!", silent=True)
        return "failed"
    except Exception as e:
        log_err(f"[{coin}] crash: {e}")
        return "failed"


# ================================================================
# ================================================================
# EXECUTOR — AI TRAILING STOP
# ================================================================

def ai_trailing_decision(coin: str, direction: str, entry_px: float,
                         mid_px: float, old_ts: float, new_ts_mech: float,
                         atr: float, regime: str, funding_z: float,
                         mode: str, mids: dict) -> dict:
    """
    Chiede all'AI se e dove spostare il trailing stop.
    Usa Chain of Thought per ragionamento step-by-step.
    Ritorna: {"move": bool, "new_ts": float, "reasoning": str, "sentiment": str}
    """
    try:
        pnl_pct  = ((mid_px - entry_px) / entry_px * 100) if direction == "LONG" \
                   else ((entry_px - mid_px) / entry_px * 100)
        dist_ts  = abs(mid_px - old_ts) / (atr + 1e-10)  # distanza TS in ATR
        dist_tp  = 0.0  # placeholder — TP non disponibile qui

        # Sentiment macro: BTC momentum
        btc_px   = float(mids.get("BTC", 0) or 0)
        eth_px   = float(mids.get("ETH", 0) or 0)

        prompt = f"""Sei un risk manager di crypto futures. Devi decidere se spostare il trailing stop.

POSIZIONE ATTIVA:
- Coin: {coin} | Direzione: {direction} | Modalità: {mode}
- Entry: {entry_px:.4g} | Prezzo attuale: {mid_px:.4g} | PnL: {pnl_pct:+.2f}%
- Trailing Stop attuale: {old_ts:.4g} (distanza: {dist_ts:.1f}×ATR)
- Trailing Stop meccanico suggerito: {new_ts_mech:.4g}
- ATR: {atr:.4g}

CONTESTO MACRO (Sentiment):
- Regime BTC: {regime}
- Funding z-score: {funding_z:+.2f} {'⚠️ contro posizione' if (direction=='LONG' and funding_z > 1.5) or (direction=='SHORT' and funding_z < -1.5) else '✅ ok'}
- BTC: {btc_px:.0f} | ETH: {eth_px:.0f}

RAGIONA STEP BY STEP (Chain of Thought):
1. Il PnL attuale giustifica protezione? (> 1% scalping, > 3% swing)
2. Il trailing meccanico è ragionevole? O è troppo vicino/lontano?
3. Il sentiment macro supporta lasciare correre o proteggere?
4. Qual è la decisione ottimale?

Rispondi SOLO con JSON:
{{
  "step1": "<analisi PnL>",
  "step2": "<analisi trailing meccanico>",
  "step3": "<analisi sentiment>",
  "step4": "<decisione finale>",
  "move": <true se sposta il trailing>,
  "new_ts": <nuovo valore trailing stop, oppure {old_ts:.4g} se non sposta>,
  "sentiment": "bullish|bearish|neutral",
  "reasoning": "<max 8 parole>"
}}"""

        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type":      "application/json",
                "x-api-key":         os.getenv("ANTHROPIC_API_KEY", ""),
                "anthropic-version": "2023-06-01",
            },
            json={
                "model":      "claude-haiku-4-5-20251001",
                "max_tokens": 300,
                "messages":   [{"role": "user", "content": prompt}]
            },
            timeout=20
        )

        if resp.status_code != 200:
            return {"move": True, "new_ts": new_ts_mech, "reasoning": "fallback meccanico", "sentiment": "neutral"}

        text   = resp.json().get("content", [{}])[0].get("text", "").strip()
        result = _parse_ai_json(text)
        move      = bool(result.get("move", True))
        new_ts    = float(result.get("new_ts", new_ts_mech))
        reasoning = result.get("reasoning", "")
        sentiment = result.get("sentiment", "neutral")

        # Log Chain of Thought
        log_exec(f"[{coin}] AI TS CoT → {result.get('step1','')} | {result.get('step2','')} | {result.get('step3','')}")
        log_exec(f"[{coin}] AI TS → move:{move} new_ts:{new_ts:.4g} sentiment:{sentiment} | {reasoning}")

        return {"move": move, "new_ts": new_ts, "reasoning": reasoning, "sentiment": sentiment}

    except Exception as e:
        log_err(f"[{coin}] ai_trailing_decision: {e}")
        return {"move": True, "new_ts": new_ts_mech, "reasoning": "fallback meccanico", "sentiment": "neutral"}


def ai_should_exit_early(coin: str, direction: str, entry_px: float,
                         mid_px: float, sig: dict, mids: dict) -> tuple[bool, str]:
    """
    Chiede all'AI se uscire anticipatamente dalla posizione.
    Analizza inversione momentum, sentiment macro, price action.
    Ritorna: (exit: bool, reason: str)
    """
    try:
        sl         = float(sig.get("sl", 0) or 0)
        tp         = float(sig.get("tp", 0) or 0)
        strategy   = sig.get("strategy", "MEAN_REV")
        regime     = sig.get("regime", "RANGE")
        mode       = sig.get("mode", "SCALPING")
        funding_z  = float(sig.get("funding_z", 0) or 0)
        pnl_pct    = ((mid_px - entry_px) / entry_px * 100) if direction == "LONG" \
                     else ((entry_px - mid_px) / entry_px * 100)

        # Distanza percentuale da SL e TP
        if direction == "LONG":
            dist_sl = (mid_px - sl) / mid_px * 100 if sl > 0 else 0
            dist_tp = (tp - mid_px) / mid_px * 100 if tp > 0 else 0
        else:
            dist_sl = (sl - mid_px) / mid_px * 100 if sl > 0 else 0
            dist_tp = (mid_px - tp) / mid_px * 100 if tp > 0 else 0

        btc_px  = float(mids.get("BTC", 0) or 0)
        eth_px  = float(mids.get("ETH", 0) or 0)

        prompt = f"""Sei un risk manager di crypto futures. Devi decidere se uscire anticipatamente.

POSIZIONE:
- Coin: {coin} | Direzione: {direction} | Strategia: {strategy} | Modalità: {mode}
- Entry: {entry_px:.4g} | Prezzo attuale: {mid_px:.4g} | PnL: {pnl_pct:+.2f}%
- SL: {sl:.4g} (distanza: {dist_sl:.2f}%) | TP: {tp:.4g} (distanza: {dist_tp:.2f}%)

CONTESTO MACRO:
- Regime BTC: {regime} | Funding z: {funding_z:+.2f}
- BTC: {btc_px:.0f} | ETH: {eth_px:.0f}

RAGIONA STEP-BY-STEP:
1. Il PnL attuale è a rischio? (SL vicino o momentum invertito?)
2. Il TP è ancora raggiungibile nel timeframe della strategia?
3. Il contesto macro supporta ancora la posizione?
4. Conviene uscire ora o aspettare SL/TP naturale?

Rispondi SOLO con JSON:
{{
  "step1": "<analisi rischio>",
  "step2": "<analisi TP raggiungibile>",
  "step3": "<analisi macro>",
  "step4": "<decisione>",
  "exit": <true se esci ora, false se aspetti>,
  "reason": "<max 8 parole>"
}}"""

        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type":      "application/json",
                "x-api-key":         os.getenv("ANTHROPIC_API_KEY", ""),
                "anthropic-version": "2023-06-01",
            },
            json={
                "model":      "claude-haiku-4-5-20251001",
                "max_tokens": 300,
                "messages":   [{"role": "user", "content": prompt}]
            },
            timeout=20
        )

        if resp.status_code != 200:
            return False, "AI non disponibile"

        text   = resp.json().get("content", [{}])[0].get("text", "").strip()
        result = _parse_ai_json(text)

        exit_now = bool(result.get("exit", False))
        reason   = result.get("reason", "")

        log_exec(f"[{coin}] AI exit CoT → {result.get('step1','')} | {result.get('step4','')}")
        log_exec(f"[{coin}] AI exit → {exit_now} | {reason}")

        return exit_now, reason

    except Exception as e:
        log_err(f"[{coin}] ai_should_exit_early: {e}")
        return False, "fallback — mantieni posizione"

def log_top3_per_category(signals: dict, mids: dict, active_coins: set, now: float):
    rows = []
    for coin, sig in signals.items():
        age = now - sig.get("ts", 0)
        if age > SIGNAL_MAX_AGE * 2:
            continue
        score     = float(sig.get("score", 0) or 0)
        direction = sig.get("direction", "?")
        price     = float(mids.get(coin, 0) or 0)
        sl        = float(sig.get("sl", 0) or 0)
        tp        = float(sig.get("tp", 0) or 0)
        fz        = float(sig.get("funding_z", 0) or 0)
        wr        = float(sig.get("win_rate", 0) or 0)

        if price > 0 and sl > 0 and tp > 0:
            risk   = (price - sl) if direction == "LONG" else (sl - price)
            reward = (tp - price) if direction == "LONG" else (price - tp)
            rr     = round(reward / risk, 2) if risk > 0 else 0.0
        else:
            rr = 0.0

        pf_sig    = float(sig.get("profit_factor", 0) or 0)

        if coin in active_coins:                                 decision = "OPEN"
        elif age > SIGNAL_MAX_AGE:                               decision = "EXPIRED"
        elif pf_sig < 1.2 or wr < 0.40:                         decision = f"{direction} ❌PF"
        elif direction == "LONG"  and score >= SETUP_SCORE_MIN:  decision = "LONG ✅"
        elif direction == "SHORT" and score >= SETUP_SCORE_MIN:  decision = "SHORT ✅"
        else:                                                    decision = "WAIT"

        cat      = get_coin_category(coin)
        priority = score if direction == "LONG" else (1.0 - score)
        rows.append({
            "coin": coin, "price": price, "dir": direction, "cat": cat,
            "score": score, "rr": rr, "fz": fz, "wr": wr,
            "age_min": int(age / 60), "decision": decision, "priority": priority
        })

    if not rows:
        log_exec("Nessun segnale disponibile")
        return

    # Raggruppa per categoria, top 3 per categoria per priorità
    from collections import defaultdict
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["cat"]].append(r)
    for cat in by_cat:
        by_cat[cat].sort(key=lambda x: (x["decision"] not in ("LONG ✅","SHORT ✅"), -x["priority"]))

    # Stampa tabella per categoria
    W = 74
    log_exec("=" * W)
    log_exec(f"  {'COIN':<8} {'PREZZO':>12}  {'DIR':<5}  {'SCORE'}  {'R:R':>4}  {'WR':>5}  {'FZ':>5}  {'AGE':>3}  DECISIONE")
    log_exec("-" * W)
    for cat in ["highcap", "midcap", "meme", "other"]:
        cat_rows = by_cat.get(cat, [])[:3]
        if not cat_rows:
            continue
        log_exec(f"  ── {cat.upper()} ──")
        for r in cat_rows:
            px_s = f"{r['price']:>12.2f}" if r['price'] >= 1 else f"{r['price']:>12.6f}"
            log_exec(
                f"  {r['coin']:<8} {px_s}  {r['dir']:<5}  {r['score']:.3f}  "
                f"{r['rr']:>4.2f}  {r['wr']:>4.0%}  "
                f"{r['fz']:>+5.1f}  {r['age_min']:>3}m  {r['decision']}"
            )
    log_exec("=" * W)


# ================================================================
# FINESTRE DI OPERATIVITÀ (UTC)
# ================================================================

SLEEP_OUT_OF_WINDOW = 20 * 60  # 20min di pausa ogni 65min → ~500h/mese

def is_active_window() -> bool:
    """45min ON / 20min OFF ogni 65min — 22 pause/giorno, ~500h/mese."""
    minute_of_day = datetime.now(timezone.utc).hour * 60 + datetime.now(timezone.utc).minute
    cycle = minute_of_day % 65
    return cycle < 55  # 55min ON / 10min OFF — 85% uptime, meno pause più corte

def minutes_until_pause() -> int:
    """Minuti rimanenti prima della prossima pausa. 0 = già in pausa."""
    minute_of_day = datetime.now(timezone.utc).hour * 60 + datetime.now(timezone.utc).minute
    cycle = minute_of_day % 65
    if cycle >= 55:
        return 0
    return 55 - cycle

def should_abort_cycle(prefix: str = "") -> bool:
    """
    Controlla se la finestra sta per chiudersi (< 2 min).
    Usato dal Processor per interrompere un ciclo lungo prima della pausa.
    """
    remaining = minutes_until_pause()
    if remaining == 0:
        if prefix:
            log(prefix, f"⏰ Finestra chiude tra {remaining}min — interrompo ciclo")
        return True
    if remaining == 0:
        return True
    return False

def wait_for_window(prefix: str):
    """Blocca il thread durante i 20 minuti di pausa."""
    while not is_active_window():
        now_utc     = datetime.now(timezone.utc)
        minute_of_day = now_utc.hour * 60 + now_utc.minute
        cycle       = minute_of_day % 65
        wait_min    = 65 - cycle  # minuti mancanti alla fine della pausa
        log(prefix, f"💤 Pausa 20min — ripresa tra {wait_min}min")
        time.sleep(min(wait_min * 60, SLEEP_OUT_OF_WINDOW))


# ================================================================
# THREAD A — SCANNER (ogni 20 min, analizza 229 coin)
# ================================================================

def _update_correlation_matrix(coins: list, now_ms: int):
    """
    Calcola la matrice di correlazione tra le top coin usando rendimenti
    15m delle ultime 24h. Aggiorna _correlation_matrix.

    Approccio: fetch 96 candele 15m per ogni coin, calcola log-returns,
    poi correlazione pairwise con numpy. Solo coppie sopra soglia vengono salvate.
    Per efficienza: max 30 coin → 435 coppie, fetch parallelizzato.
    """
    global _correlation_matrix
    if len(coins) < 2:
        return

    log("SCAN", f"Correlazione dinamica: {len(coins)} coin...")
    returns_map = {}

    # Fetch rendimenti per ogni coin (batch con throttle)
    for coin in coins:
        try:
            candles = call(_info.candles_snapshot, coin, "15m",
                           now_ms - 86400000, now_ms,
                           timeout=15, label=f'corr_{coin}')
            if candles and len(candles) >= 48:
                closes = np.array([float(c['c']) for c in candles[-96:]])
                # Log-returns: più stabili dei pct_change per correlazione
                if len(closes) >= 2:
                    log_ret = np.diff(np.log(closes + 1e-10))
                    returns_map[coin] = log_ret
            time.sleep(0.15)  # throttle API
        except Exception:
            continue

    if len(returns_map) < 2:
        log("SCAN", "Correlazione: dati insufficienti")
        return

    # Allinea lunghezze (tutte le serie alla stessa dimensione)
    min_len = min(len(r) for r in returns_map.values())
    coins_valid = list(returns_map.keys())
    matrix = np.column_stack([returns_map[c][-min_len:] for c in coins_valid])

    # Calcola matrice correlazione numpy
    try:
        corr_matrix = np.corrcoef(matrix, rowvar=False)
    except Exception as e:
        log_err(f"Correlazione numpy: {e}")
        return

    # Salva solo coppie con |corr| > 0.5 (per debug) e blocca > THRESHOLD
    new_corr = {}
    high_pairs = []
    for i in range(len(coins_valid)):
        for j in range(i + 1, len(coins_valid)):
            corr_val = float(corr_matrix[i, j])
            if np.isnan(corr_val):
                continue
            key = (min(coins_valid[i], coins_valid[j]), max(coins_valid[i], coins_valid[j]))
            if abs(corr_val) > 0.5:
                new_corr[key] = round(corr_val, 3)
            if abs(corr_val) > CORRELATION_THRESHOLD:
                high_pairs.append(f"{coins_valid[i]}-{coins_valid[j]}:{corr_val:.2f}")

    with _correlation_lock:
        _correlation_matrix = new_corr

    if high_pairs:
        log("SCAN", f"Correlazioni alte (>{CORRELATION_THRESHOLD}): {', '.join(high_pairs[:15])}")
    else:
        log("SCAN", "Nessuna correlazione sopra soglia")
    log("SCAN", f"Matrice correlazione: {len(new_corr)} coppie tracciate")


def run_scanner():
    """
    Scansione leggera su tutte le coin disponibili.
    Calcola confluence base + allineamento BTC e salva
    i candidati più promettenti per il Processor.

    FILTRO BTC:
    - Calcola momentum BTC 1h e 4h
    - Per ogni coin: stima correlazione con BTC (proxy: funding alignment)
    - Penalizza coin che si muovono CONTRO BTC
    - Penalizza meme coin ad alta correlazione quando BTC è in transizione
      (le meme seguono BTC con 5-15 min di lag → trappola)
    """
    global _scanner_candidates
    log("SCAN", "=== SCANNER AVVIATO ===")
    t0 = time.time()

    try:
        ctx_data    = call(_info.meta_and_asset_ctxs, timeout=API_TIMEOUT_SEC, label='scan_meta')
        meta_assets = ctx_data[0]["universe"]
        ctxs_main   = ctx_data[1]
    except Exception as e:
        log_err(f"Scanner meta: {e}")
        return

    regime = detect_market_regime()

    # ── BTC MOMENTUM (1h e 4h) ────────────────────────────────────
    # Calcolato una volta, usato per ogni coin nello scoring
    btc_mom_1h  = 0.0   # variazione % BTC ultima 1h
    btc_mom_4h  = 0.0   # variazione % BTC ultime 4h
    btc_trend   = "FLAT" # UP / DOWN / FLAT
    btc_px_now  = 0.0
    try:
        now_ms = int(time.time() * 1000)
        btc_candles = call(_info.candles_snapshot, "BTC", "15m",
                           now_ms - 86400000, now_ms,
                           timeout=API_TIMEOUT_SEC, label='btc_scan')
        if btc_candles and len(btc_candles) >= 20:
            btc_closes = [float(c['c']) for c in btc_candles]
            btc_px_now = btc_closes[-1]
            # Momentum 1h = ultime 4 candele 15m
            if len(btc_closes) >= 5:
                btc_mom_1h = (btc_closes[-1] - btc_closes[-4]) / btc_closes[-4]
            # Momentum 4h = ultime 16 candele 15m
            if len(btc_closes) >= 17:
                btc_mom_4h = (btc_closes[-1] - btc_closes[-16]) / btc_closes[-16]

            # Trend BTC: coerenza tra 1h e 4h
            if btc_mom_1h > 0.002 and btc_mom_4h > 0.005:
                btc_trend = "UP"
            elif btc_mom_1h < -0.002 and btc_mom_4h < -0.005:
                btc_trend = "DOWN"
            else:
                btc_trend = "FLAT"

        log("SCAN", f"BTC: ${btc_px_now:,.0f} | mom_1h:{btc_mom_1h:+.2%} | mom_4h:{btc_mom_4h:+.2%} | trend:{btc_trend} | regime:{regime}")
    except Exception as e:
        log_err(f"Scanner BTC momentum: {e}")

    candidates = []

    all_vols = [float(ctx.get("dayNtlVlm", 0) or 0)
                for ctx in ctxs_main if float(ctx.get("dayNtlVlm", 0) or 0) > 0]
    vol_universe_mean = float(np.mean(all_vols)) if all_vols else 1.0

    funding_vals = []
    for asset, ctx in zip(meta_assets, ctxs_main):
        f = float(ctx.get("funding", 0) or 0) * 10000
        funding_vals.append(f)
    funding_mean = float(np.mean(funding_vals)) if funding_vals else 0.0
    funding_std  = float(np.std(funding_vals))  if funding_vals else 1.0

    # Funding BTC per calcolo allineamento
    btc_funding = 0.0
    for asset, ctx in zip(meta_assets, ctxs_main):
        if asset["name"] == "BTC":
            btc_funding = float(ctx.get("funding", 0) or 0) * 10000
            break

    for asset, ctx in zip(meta_assets, ctxs_main):
        coin = asset["name"]
        if coin in COIN_BLACKLIST or coin.startswith("@") or coin.startswith("k"):
            continue
        try:
            px     = float(ctx.get("markPx", 0) or 0)
            vol24h = float(ctx.get("dayNtlVlm", 0) or 0)
            oi_raw = float(ctx.get("openInterest", 0) or 0)
            oi_usd = oi_raw * px

            # ── MOMENTUM SCORE: chi si sta muovendo ORA ──
            # prevDayPx è il prezzo 24h fa — misura il move in corso
            prev_px = float(ctx.get("prevDayPx", 0) or 0)
            day_change = abs(px - prev_px) / prev_px if prev_px > 0 else 0
            # Score: bilancia movimento (abs day change) con liquidità (volume)
            momentum_score = min(day_change * 100, 10)  # cap a 10 (= 10% move)

            if vol24h < 200_000 or px <= 0:
                continue

            funding_bps = float(ctx.get("funding", 0) or 0) * 10000

            # ── 1. Volume/OI Ratio ────────────────────────────────
            vol_oi_ratio = vol24h / (oi_usd + 1e-10)
            vol_oi_score = 1.0 if 0.5 <= vol_oi_ratio <= 3.0 else \
                           0.5 if vol_oi_ratio < 0.5 else \
                           max(0.0, 1.0 - (vol_oi_ratio - 3.0) / 5.0)

            # ── 2. Funding Score (MOMENTUM-ALIGNED) ─────────────────
            # Per momentum/breakout il funding ESTREMO è NEGATIVO:
            # - |FZ| > 2 = crowded trade → vicino a squeeze/reversal
            # - |FZ| < 0.5 = neutro → nessun crowd → spazio per muoversi
            # - FZ moderato (0.5-1.5) ALLINEATO al trend = buono
            #   (long crowd moderato in uptrend = conferma, non eccesso)
            fz = (funding_bps - funding_mean) / (funding_std + 1e-10)

            if abs(fz) > 2.0:
                # Crowded trade — alto rischio reversal
                fz_score = 0.0
            elif abs(fz) > 1.5:
                # Affollato — cautela
                fz_score = 0.2
            elif abs(fz) < 0.3:
                # Neutro — nessun bias di crowd, ideale per breakout
                fz_score = 0.8
            else:
                # Moderato — buono se allineato al trend della coin
                # (trend_score lo dirà, qui diamo score medio)
                fz_score = 0.6

            # ── 3. RVOL ──────────────────────────────────────────
            rvol = vol24h / (vol_universe_mean + 1e-10)
            rvol_score = min(rvol / 2.0, 1.0)

            # ── 4. ALLINEAMENTO BTC ───────────────────────────────
            btc_align_score = 0.5
            if btc_trend != "FLAT" and coin not in BTC_ETH_COINS:
                if btc_funding != 0:
                    funding_sign_match = (funding_bps * btc_funding) > 0
                    btc_align_score = 0.8 if funding_sign_match else 0.2
                if btc_trend == "UP":
                    if funding_bps > 0:
                        btc_align_score = min(btc_align_score + 0.2, 1.0)
                    elif funding_bps < -2.0:
                        btc_align_score = max(btc_align_score - 0.3, 0.0)
                elif btc_trend == "DOWN":
                    if funding_bps < 0:
                        btc_align_score = min(btc_align_score + 0.2, 1.0)
                    elif funding_bps > 2.0:
                        btc_align_score = max(btc_align_score - 0.3, 0.0)

            # ── 5. PENALITÀ MEME LAG ─────────────────────────────
            is_meme = coin in COIN_CATEGORIES.get("meme", set()) or \
                      coin in {"DOGE", "SHIB", "FARTCOIN", "POPCAT", "FLOKI", "TURBO", "NEIRO", "SPX", "GOAT", "MEW"}
            meme_lag_penalty = 0.0
            if is_meme and abs(btc_mom_1h) > 0.005:
                meme_lag_penalty = min(abs(btc_mom_1h) * 30, 15.0)

            # ── 6. STRUTTURA PREZZO — REGIME ADAPTIVE (NUOVO) ────
            # BULL/BEAR → premia: trend forte, momentum, breakout, HH/LL
            # RANGE     → premia: estremi del range, RSI estremo, BB ai bordi
            #
            # Fetch leggero: ultime 24 candele 1h
            structure_score = 0.0  # 0-1
            coin_direction  = "FLAT"  # UP/DOWN/FLAT — usato dal Processor
            try:
                scan_candles = call(_info.candles_snapshot, coin, "1h",
                                    now_ms - 86400000, now_ms,
                                    timeout=10, label=f'scan_{coin}')
                if scan_candles and len(scan_candles) >= 12:
                    sc_closes = np.array([float(c['c']) for c in scan_candles])
                    sc_highs  = np.array([float(c['h']) for c in scan_candles])
                    sc_lows   = np.array([float(c['l']) for c in scan_candles])

                    sc_ema20 = pd.Series(sc_closes).ewm(span=20, min_periods=10).mean().iloc[-1]
                    mom_6h = (sc_closes[-1] - sc_closes[-6]) / sc_closes[-6] if len(sc_closes) >= 7 else 0

                    h24 = sc_highs.max()
                    l24 = sc_lows.min()
                    range_pos = (px - l24) / (h24 - l24 + 1e-10)

                    hh_scan = sum(1 for i in range(-5, 0) if sc_highs[i] > sc_highs[i-1])
                    ll_scan = sum(1 for i in range(-5, 0) if sc_lows[i] < sc_lows[i-1])

                    # RSI approssimato (veloce, senza build_features)
                    delta = np.diff(sc_closes)
                    up_avg = np.mean(np.maximum(delta[-14:], 0))
                    dn_avg = np.mean(np.maximum(-delta[-14:], 0))
                    rsi_1h_approx = 100 - (100 / (1 + up_avg / (dn_avg + 1e-10))) if len(delta) >= 14 else 50

                    # Direzione coin
                    if mom_6h > 0.003 and px > sc_ema20:
                        coin_direction = "UP"
                    elif mom_6h < -0.003 and px < sc_ema20:
                        coin_direction = "DOWN"

                    if regime in ("BULL", "BEAR"):
                        # ── TREND MODE: premia momentum + breakout ──
                        ss = 0.0
                        # Momentum forte (>0.5% in 6h)
                        if abs(mom_6h) > 0.005:
                            ss += 0.3
                        elif abs(mom_6h) > 0.002:
                            ss += 0.15
                        # Prezzo sopra EMA20 (in BULL) o sotto (in BEAR)
                        if (regime == "BULL" and px > sc_ema20) or (regime == "BEAR" and px < sc_ema20):
                            ss += 0.2
                        # Vicino al breakout (top/bottom 20% del range)
                        if (regime == "BULL" and range_pos > 0.8) or (regime == "BEAR" and range_pos < 0.2):
                            ss += 0.25
                        # Higher highs in BULL, lower lows in BEAR
                        if (regime == "BULL" and hh_scan >= 2) or (regime == "BEAR" and ll_scan >= 2):
                            ss += 0.25
                        # PENALITÀ: coin che va CONTRO il regime
                        if (regime == "BULL" and mom_6h < -0.003) or (regime == "BEAR" and mom_6h > 0.003):
                            ss = max(0, ss - 0.4)
                        structure_score = ss

                    else:
                        # ── RANGE MODE: premia estremi per reversal ──
                        ss = 0.0
                        # RSI estremo (< 35 o > 65)
                        if rsi_1h_approx < 35 or rsi_1h_approx > 65:
                            ss += 0.3
                        elif rsi_1h_approx < 40 or rsi_1h_approx > 60:
                            ss += 0.15
                        # Posizione agli estremi del range (<20% o >80%)
                        if range_pos < 0.2 or range_pos > 0.8:
                            ss += 0.3
                        elif range_pos < 0.3 or range_pos > 0.7:
                            ss += 0.15
                        # Volume spike (conferma del test del livello)
                        if rvol > 1.5:
                            ss += 0.2
                        # Segno di inversione: HH dopo LL o viceversa
                        if (range_pos < 0.3 and hh_scan >= 1) or (range_pos > 0.7 and ll_scan >= 1):
                            ss += 0.2
                        # PENALITÀ: coin nel mezzo del range senza segnale
                        if 0.35 < range_pos < 0.65 and abs(mom_6h) < 0.003:
                            ss = max(0, ss - 0.3)
                        structure_score = ss

                time.sleep(0.05)
            except Exception:
                structure_score = 0.2

            # ── Score composito 0-100 — REGIME ADAPTIVE ───────────
            # In TREND: struttura prezzo pesa di più (coin con momentum)
            # In RANGE: struttura prezzo pesa uguale (coin agli estremi)
            if regime in ("BULL", "BEAR"):
                composite = (
                    vol_oi_score      * 10 +
                    fz_score          * 10 +
                    rvol_score        * 10 +
                    btc_align_score   * 20 +
                    structure_score   * 30 +
                    momentum_score    * 20    # chi si muove ORA
                ) - meme_lag_penalty
            else:  # RANGE
                composite = (
                    vol_oi_score      * 10 +
                    fz_score          * 10 +
                    rvol_score        * 10 +
                    btc_align_score   * 20 +
                    structure_score   * 25 +
                    momentum_score    * 25    # ancora più importante in range
                ) - meme_lag_penalty

            composite = max(0, min(100, composite))

            candidates.append({
                "coin":        coin,
                "px":          px,
                "vol24h":      vol24h,
                "oi_usd":      oi_usd,
                "funding":     funding_bps,
                "funding_z":   round(fz, 2),
                "vol_oi":      round(vol_oi_ratio, 2),
                "rvol":        round(rvol, 2),
                "btc_align":   round(btc_align_score, 2),
                "structure":   round(structure_score, 2),
                "direction":   coin_direction,
                "meme_pen":    round(meme_lag_penalty, 1),
                "composite":   round(composite, 1),
            })
        except Exception:
            continue

    # Ordina per score composito decrescente
    candidates.sort(key=lambda x: -x["composite"])

    # Prendi top PROCESSOR_MAX_COINS per il Processor
    top = [c["coin"] for c in candidates[:PROCESSOR_MAX_COINS]]

    with _scanner_lock:
        _scanner_candidates = top

    # ── CORRELAZIONE DINAMICA ─────────────────────────────────────
    # Calcola matrice di correlazione tra le top 30 coin usando rendimenti
    # 15m delle ultime 24h (96 candele). Solo le top 30 per efficienza:
    # sono quelle che il Processor analizzerà e l'Executor potrebbe tradare.
    _update_correlation_matrix(top[:30], now_ms)

    elapsed = time.time() - t0
    log("SCAN", f"BTC trend: {btc_trend} | mom_1h:{btc_mom_1h:+.2%} | mom_4h:{btc_mom_4h:+.2%}")
    log("SCAN", f"Top 10 per score composito:")
    for c in candidates[:10]:
        meme_tag = f" 🐸-{c['meme_pen']:.0f}" if c['meme_pen'] > 0 else ""
        log("SCAN", f"  {c['coin']:<8} score:{c['composite']:>5.1f} | struct:{c['structure']:.2f} {c['direction']:>4} | FZ:{c['funding_z']:+.2f} | BTC:{c['btc_align']:.2f}{meme_tag}")
    log("SCAN", f"=== FINE | {len(candidates)} candidati → top {len(top)} per Processor | {elapsed:.1f}s ===")
    _alt_scanner_ready.set()


# ================================================================
# FAST TRACK — spike detection ogni 60s
# ================================================================
_fast_track_coins = set()
_fast_track_prev_prices = {}

def fast_track_thread():
    """
    Thread leggero: ogni 60s controlla all_mids.
    Se una coin si muove >2% in 60s → aggiungila ai candidati del processor.
    """
    global _fast_track_coins, _fast_track_prev_prices
    log("FAST", "Fast track thread avviato")

    while True:
        try:
            mids = call(_info.all_mids, timeout=10, label='fast_mids')
            if not mids:
                time.sleep(60)
                continue

            urgent = set()
            for coin, px_str in mids.items():
                try:
                    px = float(px_str)
                    if px <= 0:
                        continue
                    if coin in COIN_BLACKLIST:
                        continue
                    if coin.startswith("@"):
                        continue

                    prev = _fast_track_prev_prices.get(coin, 0)
                    if prev > 0:
                        change = abs(px - prev) / prev
                        if change > 0.02:  # >2% in 60s = spike
                            urgent.add(coin)
                            direction = "UP" if px > prev else "DOWN"
                            log("FAST", f"⚡ {coin} {direction} {change:.1%} in 60s")
                    _fast_track_prev_prices[coin] = px
                except:
                    continue

            if urgent:
                with _scanner_lock:
                    for coin in urgent:
                        if coin not in _scanner_candidates:
                            _scanner_candidates.append(coin)
                _fast_track_coins = urgent
                log("FAST", f"⚡ {len(urgent)} coin aggiunte: {urgent}")
            else:
                _fast_track_coins = set()

        except Exception as e:
            log_err(f"Fast track: {e}")

        time.sleep(120)


def scanner_thread_combined():
    log("SCAN", "Thread avviato")
    while True:
        wait_for_window("SCAN")
        try:
            run_scanner()
        except Exception as e:
            log_err(f"Scanner crash: {e}")
        time.sleep(SCANNER_INTERVAL)




def processor_thread_combined():
    log_alt("Thread avviato — attendo primo ciclo Scanner...")
    # Aspetta che lo Scanner abbia completato almeno un ciclo
    _alt_scanner_ready.wait()
    log_alt("Scanner pronto — avvio cicli Processor")

    while True:
        wait_for_window("PROC")
        try:
            run_processor()
        except Exception as e:
            log_err(f"Processor crash: {e}")
            tg(f"🚨 <b>PROCESSOR crash</b>: {e}", silent=True)
        log_alt(f"Prossimo run tra {PROCESSOR_INTERVAL//60}min")
        time.sleep(PROCESSOR_INTERVAL)


# ================================================================
# THREAD B — EXECUTOR
# ================================================================

def executor_thread_alt():
    log_exec("Thread avviato — attendo completamento primo run Processor...")
    time.sleep(30)

    sz_decimals, px_decimals = fetch_meta()
    if not sz_decimals:
        log_err("Meta vuoto — executor bloccato")
        return

    cycle           = 0
    last_ts_update  = 0.0
    prev_active_positions: dict = {}   # posizioni del ciclo precedente
    open_trade_meta: dict       = {}   # metadati trade aperti: coin → {entry_px, signal, ts_open}

    while True:
        wait_for_window("EXEC")
        try:
            cycle += 1

            if cycle % META_REFRESH_CYCLES == 0:
                sz_new, px_new = fetch_meta()
                if sz_new:
                    sz_decimals, px_decimals = sz_new, px_new

            mids = None
            for attempt in range(3):
                try:
                    mids = call(_info.all_mids, label='all_mids', timeout=15)
                    break
                except Exception as e:
                    if "429" in str(e) and attempt < 2:
                        time.sleep(3 * (attempt + 1))
                    else:
                        log_err(f"all_mids: {e}")
                        break
            if not mids:
                time.sleep(CHECK_INTERVAL)
                continue

            # Aggiorna cache prezzi per il Processor (usata dall'AI)
            global _all_mids_cache
            _all_mids_cache = mids

            active_positions = get_open_positions()
            n_active         = len(active_positions)
            balance          = get_account_balance()

            # Aggiorna cache posizioni per il Processor
            global _active_positions_cache
            _active_positions_cache = active_positions

            # ── FLEET: leggi stato dal BTC Scalper (prima del log) ──
            fleet_bias, bias_reason = fleet_get_bias()
            btc_dir, btc_size = fleet_get_btc_position()
            btc_regime = fleet_get_btc_regime()

            log_exec(f"SCAN #{cycle} | 💰 ${balance:.2f} | Pos {n_active}/{MAX_CONCURRENT_TRADES} | Fleet:{fleet_bias} | BTC:{btc_regime}")

            # ── Rileva posizioni chiuse e registra esito ──────────────
            for coin, meta in list(open_trade_meta.items()):
                if coin not in active_positions:
                    # Posizione chiusa — recupera esito reale dai fill
                    entry_px  = meta.get("entry_px", 0)
                    sig       = meta.get("signal", {})
                    direction = sig.get("direction", "LONG")
                    sl        = float(sig.get("sl", 0) or 0)
                    tp        = float(sig.get("tp", 0) or 0)
                    ts_open   = meta.get("ts_open", 0)

                    # Prova a ottenere exit reale dai fill API
                    exit_px, pnl_pct, outcome = compute_real_exit(
                        coin, direction, entry_px, ts_open
                    )

                    # Fallback al vecchio metodo se fill non disponibili
                    if outcome == "unknown" and entry_px > 0:
                        exit_px = float(mids.get(coin, 0) or entry_px)
                        pnl_pct = ((exit_px - entry_px) / entry_px) if direction == "LONG" \
                                  else ((entry_px - exit_px) / entry_px)
                        outcome = "win" if pnl_pct > 0 else "loss"
                        log_exec(f"[{coin}] ⚠️ Outcome da prezzo mid (fill non disponibili)")

                    trade_record = {
                        "coin":        coin,
                        "direction":   direction,
                        "strategy":    sig.get("strategy", "MEAN_REV"),
                        "regime":      sig.get("regime", "RANGE"),
                        "rsi":         sig.get("rsi", 0),
                        "bb_pos":      sig.get("bb_pos", 0),
                        "funding_z":   sig.get("funding_z", 0),
                        "ai_score":    sig.get("ai_score", 0),
                        "setup_score": sig.get("setup_score", 0),
                        "entry_px":    entry_px,
                        "exit_px":     exit_px,
                        "sl":          sl,
                        "tp":          tp,
                        "outcome":     outcome,
                        "pnl_pct":     round(pnl_pct * 100, 2),
                        "ts_open":     ts_open,
                        "ts_close":    time.time()
                    }
                    save_trade_outcome(trade_record)
                    emoji = "✅" if outcome == "win" else "❌"
                    log_exec(f"[{coin}] {emoji} Trade chiuso — {outcome} | PnL:{pnl_pct*100:+.2f}%")
                    tg(f"{emoji} <b>{coin}</b> [{direction}] chiuso — {outcome.upper()} | PnL: {pnl_pct*100:+.2f}%", silent=True)
                    del open_trade_meta[coin]

            if pending_orders:
                process_pending_orders(active_positions, sz_decimals, px_decimals)
                cleanup_pending_orders(active_positions)

            # ── Aggiornamento trailing stop + ACTIVE MANAGEMENT ──
            now = time.time()
            if active_positions:
                for coin, pos in active_positions.items():
                    try:
                        mid_px = float(mids.get(coin, 0) or 0)
                        if mid_px <= 0: continue
                        entry_px = float(pos.get("entry_px") or mid_px)
                        szi = pos.get("szi", 0)
                        direction = "LONG" if szi > 0 else "SHORT"
                        pnl_ratio = (mid_px - entry_px) / entry_px if direction == "LONG" else (entry_px - mid_px) / entry_px
                        meta = open_trade_meta.get(coin, {})
                        trade_age = now - meta.get("ts_open", now)

                        # ── 1. EARLY CUT: loss > -0.2% dopo 30s → chiudi subito ──
                        if pnl_ratio < -0.002 and trade_age > 30:
                            log_exec(f"[{coin}] ✂️ EARLY CUT {pnl_ratio:+.2%}")
                            actual_size = round_to_decimals(abs(szi), sz_decimals.get(coin, 4))
                            try:
                                close_px = round_to_decimals(mid_px * (0.995 if direction == "LONG" else 1.005), px_decimals.get(coin, 4))
                                call(_exchange.order, str(coin), direction != "LONG",
                                     actual_size, close_px,
                                     {"limit": {"tif": "Ioc"}}, False, timeout=10)
                                delete_signal(coin)
                                tg(f"✂️ <b>{coin}</b> CUT {pnl_ratio:+.2%}", silent=True)
                            except Exception as e:
                                log_err(f"[{coin}] early cut: {e}")
                            continue

                        # ── 2. SMART TP: in profitto >0.2% → prendi profitto ──
                        if pnl_ratio > 0.002 and trade_age > 60:
                            # Check se momentum sta girando contro
                            try:
                                sig = get_all_signals().get(coin, {})
                                atr_approx = abs(float(sig.get("sl", entry_px)) - entry_px) / 1.2 if sig else mid_px * 0.01
                                if pnl_ratio > 0.004 or (atr_approx > 0 and (mid_px - entry_px if direction == "LONG" else entry_px - mid_px) >= atr_approx * 1.0):
                                    log_exec(f"[{coin}] 💰 SMART TP +{pnl_ratio:.2%}")
                                    actual_size = round_to_decimals(abs(szi), sz_decimals.get(coin, 4))
                                    close_px = round_to_decimals(mid_px * (0.995 if direction == "LONG" else 1.005), px_decimals.get(coin, 4))
                                    call(_exchange.order, str(coin), direction != "LONG",
                                         actual_size, close_px,
                                         {"limit": {"tif": "Ioc"}}, False, timeout=10)
                                    delete_signal(coin)
                                    tg(f"💰 <b>{coin}</b> TP +{pnl_ratio:.2%}", silent=True)
                            except Exception as e:
                                log_err(f"[{coin}] smart tp: {e}")
                            continue

                        # ── 3. Trailing (solo se non early cut e non smart TP) ──
                        if now - last_ts_update >= TRAILING_STOP_INTERVAL:
                            atr_approx = abs(float((get_all_signals().get(coin,{}) or {}).get("sl", mid_px)) - mid_px) / 1.2
                            if atr_approx > 0:
                                update_mechanical_trailing(coin, pos, mid_px, atr_approx,
                                    direction, open_trade_meta, sz_decimals, px_decimals)

                    except Exception as e:
                        log_err(f"[{coin}] position mgmt: {e}")

                if now - last_ts_update >= TRAILING_STOP_INTERVAL:
                    last_ts_update = now

            signals    = get_all_signals()
            if signals:
                log_exec(f"Segnali attivi: {list(signals.keys())}")
            candidates = []

            # Set di coin con posizioni attive (per check correlazione dinamica)
            active_coin_set = set(active_positions.keys())

            for coin, sig in signals.items():
                sig_max_age = int(sig.get("signal_max_age", SIGNAL_MAX_AGE))
                age = now - sig.get("ts", 0)
                if age > sig_max_age:
                    log_exec(f"[{coin}] skip: segnale scaduto (età {int(age)}s > {sig_max_age}s)")
                    delete_signal(coin)
                    continue
                if (now - last_trade_time.get(coin, 0)) < COIN_COOLDOWN: continue
                if coin in active_positions: continue
                if coin in pending_orders:   continue

                # Check correlazione dinamica con posizioni attive
                is_corr, corr_with = is_correlated_with_active(coin, active_coin_set)
                if is_corr:
                    log_exec(f"[{coin}] skip: correlato con {corr_with}")
                    continue

                score     = float(sig.get("score", 0) or 0)
                direction = sig.get("direction", "")
                pf = float(sig.get("profit_factor", 0) or 0)
                wr = float(sig.get("win_rate", 0) or 0)

                # Hard filter: PF > 1.2 E WR > 45% — senza edge reale non tradare
                if pf < 1.2 or wr < 0.40:
                    continue

                if direction == "LONG"  and score >= SETUP_SCORE_MIN:
                    candidates.append((pf, coin, sig))  # rank by PF, not score
                elif direction == "SHORT" and score >= SETUP_SCORE_MIN:
                    candidates.append((pf, coin, sig))

            candidates.sort(key=lambda x: x[0], reverse=True)
            log_top3_per_category(signals, mids, set(active_positions.keys()), now)

            # ── FLEET: kill switch globale ──
            ks_active, ks_reason = fleet_check_kill_switch()
            if ks_active:
                log_exec(f"🚨 FLEET KILL SWITCH: {ks_reason}")
                fleet_execute_kill_switch()
                time.sleep(CHECK_INTERVAL)
                continue

            # ── Circuit breaker: blocca nuovi trade se perdite eccessive ──
            cb_active, cb_reason = is_circuit_breaker_active()
            if cb_active:
                if cycle % 40 == 1:
                    log_exec(f"🛑 CIRCUIT BREAKER: {cb_reason}")
                if not hasattr(executor_thread_alt, '_cb_notified'):
                    executor_thread_alt._cb_notified = True
                    tg(f"🛑 <b>CIRCUIT BREAKER</b>: {cb_reason}")
            else:
                if hasattr(executor_thread_alt, '_cb_notified'):
                    del executor_thread_alt._cb_notified
                slots_free = MAX_CONCURRENT_TRADES - n_active - len(pending_orders)
                for _, coin, sig in candidates:
                    if slots_free <= 0:
                        break

                    direction_sig = sig.get("direction", "")

                    # Fleet bias: BTC come leading indicator per ALT
                    # LONG_BIAS/SHORT_BIAS = BTC si muove, ALT probabilmente seguono
                    # Allineamento = conferma, disallineamento = procedi comunque (ALT ha dinamiche proprie)
                    btc_aligns = (
                        (fleet_bias == "LONG_BIAS" and direction_sig == "LONG") or
                        (fleet_bias == "SHORT_BIAS" and direction_sig == "SHORT")
                    )
                    if btc_aligns:
                        log_exec(f"[{coin}] ✦ BTC conferma {direction_sig} ({bias_reason})")
                    elif fleet_bias != "NEUTRAL" and direction_sig:
                        log_exec(f"[{coin}] ⚠️ BTC {fleet_bias} vs ALT {direction_sig} — procedo")

                    # ── FLEET: correlazione con posizione BTC ──
                    if btc_dir and coin in BTC_ETH_COINS:
                        if direction_sig == btc_dir:
                            log_exec(f"[{coin}] skip {direction_sig} — BTC già {btc_dir} (correlazione)")
                            continue

                    # ── BTC momentum check realtime (executor) ────────
                    # Blocca solo se BTC si muove FORTE (>0.5%) negli ultimi 15min.
                    # Non cancellare il segnale — potrebbe essere valido al prossimo ciclo.
                    if coin not in BTC_ETH_COINS and mids:
                        btc_change_fast = 0.0
                        try:
                            btc_candles_exec = call(_info.candles_snapshot, "BTC", "5m",
                                                     int(time.time() * 1000) - 3600000,
                                                     int(time.time() * 1000),
                                                     timeout=10, label='btc_exec')
                            if btc_candles_exec and len(btc_candles_exec) >= 3:
                                btc_c3 = float(btc_candles_exec[-3]['c'])
                                btc_c0 = float(btc_candles_exec[-1]['c'])
                                btc_change_fast = (btc_c0 - btc_c3) / btc_c3 if btc_c3 > 0 else 0
                        except Exception:
                            pass

                        if direction_sig == "SHORT" and btc_change_fast > 0.005:
                            log_exec(f"[{coin}] ⏳ SHORT rimandato: BTC sale {btc_change_fast:+.2%} in 15min")
                            continue
                        if direction_sig == "LONG" and btc_change_fast < -0.005:
                            log_exec(f"[{coin}] ⏳ LONG rimandato: BTC scende {btc_change_fast:+.2%} in 15min")
                            continue

                    log_exec(f"→ {coin} [{direction_sig}] score:{sig.get('score'):.3f}")
                    result = open_trade(coin, sig, mids,
                                        sz_decimals.get(coin, 2),
                                        px_decimals.get(coin, 2))
                    if result in ("filled", "pending"):
                        last_trade_time[coin] = now
                        active_coin_set.add(coin)  # aggiorna set per check correlazione prossimi candidati
                        slots_free -= 1
                        # Registra metadati trade per rilevamento chiusura
                        open_trade_meta[coin] = {
                            "entry_px":       float(mids.get(coin, 0) or 0),
                            "signal":         sig,
                            "ts_open":        now,
                            "partial_done":   False,
                            "trailing_active": False,
                            "current_ts":     0,
                        }
                        time.sleep(5)

        except Exception as e:
            log_err(f"Executor loop: {e}")
            tg(f"🚨 <b>EXECUTOR crash</b>: {e}", silent=True)

        time.sleep(CHECK_INTERVAL)


# ================================================================
# ================================================================
# RECOVERY — POSIZIONI SENZA PROTEZIONE AL RIAVVIO
# ================================================================

def _recover_unprotected_positions():
    """
    Al riavvio, verifica che ogni posizione aperta abbia almeno uno SL attivo.
    Se manca, piazza uno SL di emergenza a 2×ATR stimato dal prezzo corrente.
    """
    try:
        positions = get_open_positions()
        if not positions:
            log("MAIN", "Recovery: nessuna posizione aperta")
            return

        mids = call(_info.all_mids, label='recovery_mids', timeout=15)
        if not mids:
            log_err("Recovery: impossibile ottenere prezzi — skip")
            return

        _, px_decimals_r = fetch_meta()
        sz_decimals_r, _ = fetch_meta()

        for coin, pos in positions.items():
            try:
                trigger_orders = get_open_trigger_orders(coin)
                has_sl = any(o.get("orderType") == "Stop Market" for o in trigger_orders)

                if has_sl:
                    continue  # protetto, ok

                szi      = float(pos["szi"])
                direction = "LONG" if szi > 0 else "SHORT"
                is_buy   = direction == "LONG"
                mid_px   = float(mids.get(coin, 0) or 0)
                entry_px = float(pos.get("entry_px", mid_px) or mid_px)

                if mid_px <= 0:
                    log_err(f"[{coin}] Recovery: prezzo non disponibile — CHIUDI MANUALMENTE")
                    tg(f"🚨 <b>{coin}</b> posizione senza SL e prezzo non disponibile!")
                    continue

                # SL emergenza: 3% dal prezzo corrente (conservativo)
                emergency_dist = mid_px * 0.01  # 1%
                if direction == "LONG":
                    sl_px = mid_px - emergency_dist
                else:
                    sl_px = mid_px + emergency_dist

                px_d = px_decimals_r.get(coin, 4)
                sz_d = sz_decimals_r.get(coin, 4)
                sl_px       = round_to_decimals(sl_px, px_d)
                actual_size = round_to_decimals(abs(szi), sz_d)

                call(_exchange.order, str(coin), not is_buy, actual_size, sl_px,
                     {"trigger": {"triggerPx": sl_px, "isMarket": True, "tpsl": "sl"}},
                     True, timeout=15, label=f'recovery_sl_{coin}')

                log("MAIN", f"[{coin}] 🛡️ Recovery SL emergenza @ {sl_px} ({direction}, size:{actual_size})")
                tg(f"🛡️ <b>{coin}</b> [{direction}] Recovery SL emergenza @ {sl_px}")
                time.sleep(0.5)

            except Exception as e:
                log_err(f"[{coin}] Recovery fallita: {e}")
                tg(f"🚨 <b>{coin}</b> RECOVERY SL FALLITA — verifica manualmente!")

    except Exception as e:
        log_err(f"Recovery globale: {e}")


# ================================================================
# ENTRY POINT
# ================================================================

# ================================================================
# UNIFIED MAIN — 4 Thread Architecture
# ================================================================

def main():
    log("MAIN", "🚀 UNIFIED BOT — BTC Scalper V7 + Altcoin Processor")
    log("MAIN", f"BTC: Risk ${BTC_RISK_USD} Lev:{BTC_LEVERAGE}x | ALT: Size ${ALT_TRADE_SIZE_USD} Lev:{ALT_LEVERAGE}x")
    log("MAIN", f"V7 Modules: Sentiment + Order Flow + ML")
    log("MAIN", f"Fleet: INTERNAL (no cross-worker Redis)")
    log("MAIN", f"Redis: {'✅' if REDIS_URL else '⚠️ non configurato'}")

    # Load persisted state
    load_state()
    load_state_from_redis()
    ml_load_model()

    # Get BTC meta
    btc_sz_dec, btc_px_dec = get_meta()

    # Thread A — BTC Scanner (regime + backtest + V7 analytics)
    threading.Thread(target=scanner_thread, name="BTC-Scanner", daemon=True).start()

    # Thread B — ALT Scanner (229 coin filter ogni 20 min)
    threading.Thread(target=scanner_thread_combined, name="ALT-Scanner", daemon=True).start()
    time.sleep(2)

    # Thread C — BTC Processor (signal check ogni 10s)
    threading.Thread(target=processor_thread, args=(btc_sz_dec, btc_px_dec),
                     name="BTC-Proc", daemon=True).start()
    time.sleep(1)

    # Thread D — ALT Processor (backtest + AI sui candidati)
    threading.Thread(target=processor_thread_combined, name="ALT-Proc", daemon=True).start()

    # Thread E — ALT Executor (runs in separate thread)
    threading.Thread(target=executor_thread_alt, name="ALT-Exec", daemon=True).start()

    # Thread F — Fast Track (spike detection ogni 60s)
    threading.Thread(target=fast_track_thread, name="Fast-Track", daemon=True).start()

    # Main thread — BTC Executor (simplified position management loop)
    try:
        btc_executor_loop(btc_sz_dec, btc_px_dec)
    except KeyboardInterrupt:
        log("MAIN", "🛑 Stop")


def btc_executor_loop(sz_dec, px_dec):
    """
    BTC executor loop — runs in main thread.
    Reads _btc_current_signal from BTC processor, executes trades,
    manages trailing/partial/AI management.
    """
    global _btc_current_signal, _btc_last_trade_ts

    log_btc("Executor avviato — attendo Scanner...")
    _btc_scanner_ready.wait()
    log_btc("Executor pronto")

    last_pos_state = load_pos_state()
    if last_pos_state:
        p = get_position()
        if p:
            log_btc(f"Posizione da Redis: {last_pos_state.get('type','?')} @ {last_pos_state.get('entry',0)}")
        else:
            last_pos_state = None
            save_pos_state(None)
    if not last_pos_state:
        last_pos_state = recover_position(sz_dec, px_dec)

    cleanup_orphan_orders()
    cycle = 0

    while True:
        try:
            cycle += 1
            pos = get_position()
            mid = get_mid()
            bal = get_balance()

            # ── Detect trade close ──
            if last_pos_state is not None and pos is None:
                entry = last_pos_state.get("entry_px", last_pos_state.get("entry", 0))
                szi = last_pos_state.get("szi", 0)
                d = last_pos_state.get("side", "LONG" if szi > 0 else "SHORT")
                size_abs = last_pos_state.get("size", abs(szi))

                open_ts = last_pos_state.get("open_ts", last_pos_state.get("entry_time", time.time() - 3600))
                real_exit, real_pnl = btc_compute_real_exit(d, entry, open_ts)

                if real_exit <= 0:
                    # Fill non disponibili — riprova max 3 volte poi skip
                    retry = last_pos_state.get("_exit_retry", 0) + 1
                    last_pos_state["_exit_retry"] = retry
                    if retry >= 3:
                        log_btc(f"⚠️ Exit unknown dopo {retry} tentativi — skip trade logging")
                        save_pos_state(None)
                        last_pos_state = None
                    else:
                        time.sleep(BTC_SCAN_INTERVAL)
                    continue

                exit_px = real_exit
                pnl_pct = real_pnl * 100
                pnl_usd = real_pnl * size_abs * entry

                close_reason = last_pos_state.get("close_reason", "")
                if not close_reason:
                    sl_d = last_pos_state.get("sl_dist", 0)
                    if sl_d > 0:
                        tp_d = sl_d * TP_RR
                        if d == "LONG":
                            hit_sl = mid <= entry - sl_d * 0.9; hit_tp = mid >= entry + tp_d * 0.9
                        else:
                            hit_sl = mid >= entry + sl_d * 0.9; hit_tp = mid <= entry - tp_d * 0.9
                        if hit_tp: close_reason = "🎯 Take Profit"
                        elif hit_sl: close_reason = "🛑 Stop Loss"
                        elif last_pos_state.get("trailing_active"): close_reason = "📈 Trailing Stop"
                        else: close_reason = "❓ Unknown"

                emoji = "✅" if pnl_usd > 0 else "❌"
                duration_s = time.time() - open_ts
                dur_str = f"{duration_s:.0f}s" if duration_s < 60 else f"{duration_s/60:.0f}min"

                save_trade(pnl_usd, d, entry, exit_px,
                           sig_type=last_pos_state.get("type", "?"),
                           sl_dist=last_pos_state.get("sl_dist", 0),
                           regime=last_pos_state.get("regime", "?"),
                           extra={"pnl_pct": pnl_pct, "duration_s": duration_s,
                                  "close_reason": close_reason, "ts_open": open_ts,
                                  "trailing_moves": last_pos_state.get("trailing_moves", 0)})
                save_pos_state(None)

                log_btc(f"{emoji} CLOSED {d} | {close_reason} | ${pnl_usd:+.2f} ({pnl_pct:+.2f}%) | {dur_str}")
                tg(f"{emoji} <b>BTC {d} CLOSED</b>\n${pnl_usd:+.2f} ({pnl_pct:+.2f}%)\n{close_reason}")

                # Cooldown più lungo dopo un loss
                if pnl_usd < 0:
                    _btc_last_trade_ts = time.time()
                    log_btc(f"⏸️ Loss — sleeping {BTC_COOLDOWN_AFTER_LOSS}s")
                    time.sleep(BTC_COOLDOWN_AFTER_LOSS)

                # ML online learning
                ml_feats = last_pos_state.get("ml_features", [])
                if ml_feats and len(ml_feats) == OnlineGBClassifier.N_FEATURES:
                    ml_record_outcome(ml_feats, pnl_usd > 0)

                last_pos_state = None

            # ── Trade management (in posizione) ──
            if pos is not None and last_pos_state:
                entry = pos["entry"]; szi = pos["szi"]
                d = "LONG" if szi > 0 else "SHORT"
                pnl_pct = ((mid - entry)/entry if d == "LONG" else (entry - mid)/entry) * 100
                trade_mode = last_pos_state.get("mode", last_pos_state.get("scalp_mode", "TREND"))
                atr_now = last_pos_state.get("atr", 0)
                open_ts = last_pos_state.get("open_ts", last_pos_state.get("entry_time", 0))
                time_in_trade = time.time() - open_ts if open_ts > 0 else 0

                # Update ATR ogni ~3 cicli
                if cycle % 3 == 0:
                    df_m = fetch_df("15m", 1)
                    if df_m is not None and len(df_m) >= 5:
                        atr_now = float(df_m.iloc[-1]['atr'])
                        last_pos_state["atr"] = atr_now

                # ── 1. EARLY CUT: loss > -0.15% dopo 30s ──
                pnl_ratio = pnl_pct / 100
                if pnl_ratio < -0.0015 and time_in_trade > 30:
                    log_btc(f"✂️ EARLY CUT {pnl_pct:+.2f}%")
                    btc_market_close(d, abs(szi), mid, sz_dec, px_dec)
                    last_pos_state["close_reason"] = f"✂️ Cut {pnl_pct:+.2f}%"
                    save_pos_state(last_pos_state)
                    tg(f"✂️ <b>BTC CUT</b> {pnl_pct:+.2f}%")
                    time.sleep(BTC_SCAN_INTERVAL)
                    continue  # skip tutto il resto — posizione chiusa

                # ── 2. MODE-SPECIFIC ──
                if trade_mode == "RANGE" and pnl_pct > 0.15:
                    log_btc(f"💰 RANGE TP +{pnl_pct:.2f}%")
                    btc_market_close(d, abs(szi), mid, sz_dec, px_dec)
                    last_pos_state["close_reason"] = f"💰 RANGE TP"
                    save_pos_state(last_pos_state)
                    time.sleep(BTC_SCAN_INTERVAL)
                    continue  # skip — posizione chiusa

                elif trade_mode == "TREND":
                    # TP1 partial close
                    btc_check_partial_close(last_pos_state, mid, sz_dec, px_dec)
                    # Trailing
                    if atr_now > 0:
                        last_trail = last_pos_state.get("last_trail_check", 0)
                        if time.time() - last_trail >= TRAILING_STOP_INTERVAL:
                            last_pos_state["last_trail_check"] = time.time()
                            update_trailing(last_pos_state, mid, atr_now, sz_dec, px_dec)
                # FLASH: niente — SL/TP fissi

                # ── 4. CHECK EXIT (fill reale) ──
                closed, exit_px = check_btc_exit(last_pos_state)
                if closed and exit_px:
                    pnl_real = ((exit_px - entry)/entry if d == "LONG" else (entry - exit_px)/entry) * 100
                    pnl_usd = pnl_real/100 * abs(szi) * entry
                    dur = f"{time_in_trade:.0f}s" if time_in_trade < 60 else f"{time_in_trade/60:.0f}min"
                    emoji = "✅" if pnl_usd > 0 else "❌"

                    save_trade(pnl_usd, d, entry, exit_px,
                               sig_type=last_pos_state.get("type", "?"),
                               sl_dist=last_pos_state.get("sl_dist", 0),
                               regime=last_pos_state.get("regime", "?"),
                               extra={"pnl_pct": pnl_real, "duration_s": time_in_trade,
                                      "close_reason": last_pos_state.get("close_reason", ""),
                                      "ts_open": open_ts})
                    save_pos_state(None)

                    log_btc(f"{emoji} CLOSED {d} ${pnl_usd:+.2f} ({pnl_real:+.2f}%) {dur}")
                    tg(f"{emoji} <b>BTC {d}</b> ${pnl_usd:+.2f} ({pnl_real:+.2f}%)")

                    if pnl_usd < 0:
                        log_btc(f"⏸️ Loss — sleeping {BTC_COOLDOWN_AFTER_LOSS}s")
                        time.sleep(BTC_COOLDOWN_AFTER_LOSS)

                    ml_feats = last_pos_state.get("ml_features", [])
                    if ml_feats and len(ml_feats) == OnlineGBClassifier.N_FEATURES:
                        ml_record_outcome(ml_feats, pnl_usd > 0)

                    last_pos_state = None

                # Fleet + log
                if last_pos_state:
                    fleet_set_btc_position({"direction": d, "size": abs(szi), "entry": entry})
                    log_btc(f"#{cycle} {d} @ {entry:.0f} PnL:{pnl_pct:+.1f}% [{trade_mode}] ${bal:.2f}")
                time.sleep(BTC_SCAN_INTERVAL)
                continue

            # Fleet: nessuna posizione BTC
            fleet_set_btc_position(None)

            # Check if processor just filled a trade
            sig = _btc_current_signal
            if sig and sig.get("filled"):
                _btc_current_signal = None
                last_pos_state = sig.get("pos_state")
                if last_pos_state:
                    log_btc(f"📥 Trade received from processor — managing position")

            # Status
            daily_pnl = sum(t["pnl"] for t in _btc_trades_today if time.time()-t.get("ts_close",t.get("ts",0))<86400)
            n_today = len([t for t in _btc_trades_today if time.time()-t.get("ts_close",t.get("ts",0))<86400])
            log_btc(f"#{cycle} ${bal:.2f} | {_btc_regime} | flat | {n_today}trades ${daily_pnl:+.2f} | ${mid:,.0f}")

        except Exception as e:
            log_btc(f"Executor error: {e}")
            import traceback; traceback.print_exc()

        time.sleep(BTC_SCAN_INTERVAL)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL: {e}", flush=True)
        import traceback; traceback.print_exc()
        time.sleep(60)
