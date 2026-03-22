"""
btc_scalper_v7.py — Bot BTC-only scalping + Predictive Analytics.

V7 ENHANCEMENTS:
  ✦ SENTIMENT ANALYSIS — Fear & Greed Index + social sentiment aggregation
  ✦ ORDER FLOW — CVD, delta volume, OI momentum, liquidation heatmap
  ✦ MACHINE LEARNING — Online gradient boosting per signal quality prediction

3 SCALPING MODES (auto-detected):
  RANGE: ADX<20, mean reversion, TP fisso 0.4%
  TREND: ADX>25, momentum pullback, trailing + R:R 1:2
  FLASH: vol spike >3x, breakout puro, TP 0.3%, IoC diretto

TIMEFRAMES:
  4h → regime (BULL/BEAR/RANGE)
  1h → setup (EMA, RSI, ADX)
  15m → entry trigger + SL/TP
"""

import os, sys, time, json, threading, requests, hashlib
import pandas as pd
import numpy as np
from collections import deque
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
COOLDOWN_SEC = 60              # 1 min tra trade (TP stretto = trade veloci)

# Timing
SCAN_INTERVAL = 10             # check ogni 10s (Railway killa se >15s senza output)
REGIME_INTERVAL = 300          # ricalcola regime ogni 5 min

# SL/TP — ROE-based (con leva 20x)
# TP 3% ROE = 0.15% price move = ~$111 su BTC $74k → $0.15 gain - $0.03 fee = $0.12 net
# SL 2% ROE = 0.10% price move = ~$74 su BTC $74k → -$0.10 loss
# R:R = 1:1.5 in ROE, ma con WR >55% su TP stretto = profittevole

ROE_TP = 0.03           # 3% ROE target
ROE_SL = 0.02           # 2% ROE max loss
# Price move = ROE / leverage
TP_PRICE_PCT = ROE_TP / LEVERAGE   # 0.15%
SL_PRICE_PCT = ROE_SL / LEVERAGE   # 0.10%

# MODE 1: RANGE SCALPING (Mean Reversion)
RANGE_SL_ATR  = 1.0
RANGE_TP_PCT  = TP_PRICE_PCT  # 0.15%
RANGE_SL_MIN  = SL_PRICE_PCT * 0.7   # 0.07%
RANGE_SL_MAX  = SL_PRICE_PCT * 1.5   # 0.15%

# MODE 2: TREND SCALPING (Momentum Pullback)
TREND_SL_ATR  = 1.0
TREND_TP_RR   = TP_PRICE_PCT / SL_PRICE_PCT  # 1.5
TREND_SL_MIN  = SL_PRICE_PCT * 0.7
TREND_SL_MAX  = SL_PRICE_PCT * 1.5
TREND_TRAIL_ATR = 0.5
TREND_PARTIAL = 0.6

# MODE 3: FLASH SCALPING (Volatility Expansion)
FLASH_SL_ATR  = 0.8
FLASH_TP_PCT  = TP_PRICE_PCT  # 0.15%
FLASH_SL_MIN  = SL_PRICE_PCT * 0.5
FLASH_SL_MAX  = SL_PRICE_PCT * 1.2
FLASH_TRAILING = False
FLASH_USE_IOC  = True

# Defaults
SL_ATR_MULT = TREND_SL_ATR
TP_RR = TREND_TP_RR
SL_MIN_PCT = TREND_SL_MIN
SL_MAX_PCT = TREND_SL_MAX
TRAILING_ACTIVATE = 0.5
TRAILING_ATR = TREND_TRAIL_ATR
PARTIAL_CLOSE_PCT = TREND_PARTIAL

FUNDING_BLOCK_THRESH = 0.0003

# Order Flow globals
_last_oi = 0.0

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

# ================================================================
# MODULE 1: SENTIMENT ANALYSIS (V7.2 — Dual API)
# ================================================================
# 4 fonti con pesi calibrati per BTC scalping:
#   A. Fear & Greed Index        (30%) — macro sentiment, contrarian
#   B. LunarCrush Galaxy+Social  (30%) — social real-time (Twitter/Reddit/YT)
#   C. Funding z-score interno   (25%) — posizionamento derivati
#   D. CryptoCompare IntoTheBlock(15%) — on-chain bull/bear signals
#
# ENV VARS necessarie:
#   LUNARCRUSH_API_KEY     — free Discover o $24/mo Individual
#   CRYPTOCOMPARE_API_KEY  — free tier (~100k calls/mese)
#
# Entrambe opzionali: se mancano, il bot usa fallback interni.
# ================================================================

_sentiment_cache = {"score": 50, "components": {}, "ts": 0}
SENTIMENT_CACHE_TTL = 300  # 5 min cache — ~288 calls/day per API

# Sub-caches per API con TTL indipendenti (evita burst se una è lenta)
_lunarcrush_cache = {"ts": 0, "galaxy": 50, "sentiment": 50, "social_volume": 0}
_cryptocompare_cache = {"ts": 0, "bull_pct": 50, "signals": {}}
LUNARCRUSH_CACHE_TTL = 300
CRYPTOCOMPARE_CACHE_TTL = 300


def _fetch_lunarcrush():
    """
    Fetch BTC Galaxy Score + Sentiment da LunarCrush API v4.
    
    Endpoint strategy (v4 docs):
      1° tentativo: /coins/list/v2?sort=galaxy_score&limit=1&filter=BTC
         → real-time (aggiornato ogni pochi secondi), ha galaxy_score + sentiment
      2° fallback: /coins/btc/time-series/v2 (ultimo datapoint)
         → aggiornato ogni ora, ha galaxy_score + sentiment + social metrics
      3° fallback: /topic/bitcoin/v1
         → dati topic-level (aggregato social su "bitcoin")
    
    Galaxy Score: 0-100 (health composita: price + social + correlation).
    Sentiment: % post positivi pesata per interazioni (0-100).
    """
    global _lunarcrush_cache
    if time.time() - _lunarcrush_cache["ts"] < LUNARCRUSH_CACHE_TTL:
        return _lunarcrush_cache

    lc_key = os.getenv("LUNARCRUSH_API_KEY", "")
    if not lc_key:
        return _lunarcrush_cache

    headers = {"Authorization": f"Bearer {lc_key}"}

    # ── Tentativo 1: coins/list/v2 (real-time, filtrato per BTC) ──
    try:
        url = "https://lunarcrush.com/api4/public/coins/list/v2?sort=galaxy_score&limit=10"
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            items = r.json().get("data", [])
            for item in items:
                sym = (item.get("symbol", "") or item.get("s", "")).upper()
                if sym == "BTC":
                    result = _parse_lunarcrush_item(item, "coins/list/v2")
                    if result:
                        return _lunarcrush_cache
            # BTC non trovato nei top 10 per galaxy — prova senza sort
            url2 = "https://lunarcrush.com/api4/public/coins/list/v2?limit=1&search=btc"
            r2 = requests.get(url2, headers=headers, timeout=10)
            if r2.status_code == 200:
                items2 = r2.json().get("data", [])
                if items2:
                    result = _parse_lunarcrush_item(items2[0], "coins/list/v2(search)")
                    if result:
                        return _lunarcrush_cache
        elif r.status_code == 429:
            log("[SENT] LunarCrush rate limited on coins/list")
        else:
            log(f"[SENT] LunarCrush coins/list HTTP {r.status_code}")
    except Exception as e:
        log(f"[SENT] LunarCrush coins/list error: {e}")

    # ── Tentativo 2: time-series (ultimo datapoint, cached ~1h) ──
    try:
        url = "https://lunarcrush.com/api4/public/coins/btc/time-series/v2"
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            data_list = r.json().get("data", [])
            if data_list:
                # Ultimo datapoint (più recente)
                item = data_list[-1]
                result = _parse_lunarcrush_item(item, "time-series/v2")
                if result:
                    return _lunarcrush_cache
        elif r.status_code != 429:
            log(f"[SENT] LunarCrush time-series HTTP {r.status_code}")
    except Exception as e:
        log(f"[SENT] LunarCrush time-series error: {e}")

    # ── Tentativo 3: topic/bitcoin (aggregato social) ──
    try:
        url = "https://lunarcrush.com/api4/public/topic/bitcoin/v1"
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            data = r.json().get("data", {})
            if data:
                result = _parse_lunarcrush_item(data, "topic/bitcoin")
                if result:
                    return _lunarcrush_cache
        elif r.status_code != 429:
            log(f"[SENT] LunarCrush topic HTTP {r.status_code}")
    except Exception as e:
        log(f"[SENT] LunarCrush topic error: {e}")

    log("[SENT] LunarCrush: tutti gli endpoint falliti")
    return _lunarcrush_cache


def _parse_lunarcrush_item(item, source=""):
    """
    Parsing unificato per qualsiasi endpoint LunarCrush.
    I campi possono avere nomi leggermente diversi tra endpoint.
    Returns True se il parsing ha avuto successo.
    """
    global _lunarcrush_cache

    # Galaxy Score: cercato con diversi nomi possibili
    galaxy = None
    for key in ["galaxy_score", "gs", "galaxy"]:
        val = item.get(key)
        if val is not None:
            galaxy = float(val)
            break

    # Sentiment: % positivo
    sentiment = None
    for key in ["sentiment", "average_sentiment", "sent"]:
        val = item.get(key)
        if val is not None:
            sentiment = float(val)
            break

    # Almeno uno dei due deve esistere
    if galaxy is None and sentiment is None:
        log(f"[SENT] LunarCrush parse failed ({source}): no galaxy or sentiment in keys={list(item.keys())[:10]}")
        return False

    galaxy = galaxy if galaxy is not None else 50
    sentiment = sentiment if sentiment is not None else 50

    # Social volume
    social_vol = 0
    for key in ["social_volume", "social_mentions", "posts_created", "posts_active"]:
        val = item.get(key)
        if val is not None and float(val) > 0:
            social_vol = int(float(val))
            break

    # Alt rank
    alt_rank = int(float(item.get("alt_rank", 500) or 500))

    # Social dominance
    social_dom = float(item.get("social_dominance", 0) or 0)

    # Interactions
    interactions = int(float(item.get("interactions", item.get("interactions_24h", 0)) or 0))

    _lunarcrush_cache = {
        "ts": time.time(),
        "galaxy": galaxy,
        "sentiment": sentiment,
        "social_volume": social_vol,
        "alt_rank": alt_rank,
        "social_dominance": social_dom,
        "interactions": interactions,
        "source": source,
    }
    log(f"[SENT] LunarCrush OK ({source}): Galaxy:{galaxy:.0f} Sent:{sentiment:.0f}% "
        f"SocVol:{social_vol} AltRank:{alt_rank} Interact:{interactions}")
    return True


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
                log(f"[SENT] CryptoCompare API error: {resp.get('Message', 'unknown')}")
                return _cryptocompare_cache

            data = resp.get("Data", {})

            if not data:
                log(f"[SENT] CryptoCompare: Data vuoto. Keys risposta: {list(resp.keys())}")
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
                log(f"[SENT] CryptoCompare: nessun signal trovato. Data keys: {list(data.keys())[:15]}")

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
            log(f"[SENT] CryptoCompare OK: Bull:{avg_bull:.0f}% ({len(bull_scores)} signals)")
        elif r.status_code == 429:
            log(f"[SENT] CryptoCompare rate limited — using cache")
        else:
            log(f"[SENT] CryptoCompare HTTP {r.status_code}: {r.text[:200]}")

    except Exception as e:
        log(f"[SENT] CryptoCompare error: {e}")

    return _cryptocompare_cache


def get_sentiment_score():
    """
    Composite sentiment score 0-100 (V7.2 — Dual API).

    Pesi:
      A. Fear & Greed Index          30%  (macro contrarian)
      B. LunarCrush Galaxy+Sentiment 30%  (social real-time)
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
        log(f"[SENT] FGI fetch error: {e}")
    components.setdefault("fgi", fgi_score)
    sources["fgi"] = (fgi_score, 0.30)

    # ── B. LunarCrush — Galaxy Score + Sentiment (social real-time) ──
    lc = _fetch_lunarcrush()
    lc_available = lc["ts"] > 0

    if lc_available:
        # Combina Galaxy Score (health globale) e Sentiment % (positività post)
        # Galaxy è 0-100 ma centrato su ~50-60, quindi stretchiamo un po'
        galaxy_norm = max(0, min(100, (lc["galaxy"] - 20) * 1.25))
        # Sentiment è già 0-100
        lc_score = galaxy_norm * 0.55 + lc["sentiment"] * 0.45
        lc_score = max(5, min(95, lc_score))

        components["lc_galaxy"] = round(lc["galaxy"], 1)
        components["lc_sentiment"] = round(lc["sentiment"], 1)
        components["lc_social_vol"] = lc.get("social_volume", 0)
        components["lc_alt_rank"] = lc.get("alt_rank", 0)
        components["lc_score"] = round(lc_score, 1)
        sources["lunarcrush"] = (lc_score, 0.30)
    else:
        components["lc_score"] = None  # non disponibile

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
    if not lc_available:
        missing.append("LC")
    if not cc_available:
        missing.append("CC")
    components["active_sources"] = active
    components["missing_sources"] = missing

    _sentiment_cache = {"score": composite, "components": components, "ts": time.time()}

    # Log dettagliato
    lc_str = f"LC:{lc_score:.0f}" if lc_available else "LC:off"
    cc_str = f"CC:{cc_score:.0f}" if cc_available else "CC:off"
    log(f"[SENT] Score:{composite} | FGI:{fgi_score} {lc_str} "
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

    In V7.2 tiene conto anche di LunarCrush Galaxy Score per conferma:
    se Galaxy < 30, la fear è confermata socialmente → bias più forte.
    """
    s = get_sentiment_score()
    lc = _lunarcrush_cache

    # Galaxy Score basso rinforza il segnale contrarian
    galaxy_extreme = lc.get("galaxy", 50) < 30 or lc.get("galaxy", 50) > 75

    if s < 20:
        return "EXTREME_FEAR", s
    if s < 35:
        # Se Galaxy conferma fear, promuovi a extreme
        if galaxy_extreme and lc.get("galaxy", 50) < 30:
            return "EXTREME_FEAR", s
        return "FEAR", s
    if s > 80:
        return "EXTREME_GREED", s
    if s > 65:
        if galaxy_extreme and lc.get("galaxy", 50) > 75:
            return "EXTREME_GREED", s
        return "GREED", s
    return "NEUTRAL", s


def get_sentiment_adjustment():
    """
    Returns (size_mult_adj, sl_adj, confidence_adj) based on sentiment.
    V7.2: modulato anche da LunarCrush social volume spike.
    """
    s = get_sentiment_score()
    lc = _lunarcrush_cache

    # Social volume spike detection: se il volume social è molto alto,
    # il mercato è "rumoroso" → riduci size per cautela
    social_vol = lc.get("social_volume", 0)
    vol_penalty = 1.0
    # Se abbiamo storico, un volume >2x media recente = spike
    # Per ora usiamo una soglia assoluta conservativa
    if social_vol > 50000:  # BTC tipicamente 10k-30k
        vol_penalty = 0.85
        log(f"[SENT] Social volume spike: {social_vol} → size ×0.85")

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

def get_btc_open_interest():
    """Returns current BTC open interest in USD notional."""
    try:
        ctx = call(_info.meta_and_asset_ctxs, timeout=15)
        for a, c in zip(ctx[0]["universe"], ctx[1]):
            if a["name"] == COIN:
                return float(c.get("openInterest", 0) or 0)
    except Exception as e:
        log(f"[FLOW] OI fetch error: {e}")
    return _oi_cache


def compute_orderbook_delta():
    """
    Computes bid/ask pressure delta from L2 orderbook.
    Returns: {delta, imbalance_ratio, aggressive_side, depth_bid, depth_ask}
    """
    result = {"delta": 0, "imbalance_ratio": 0.5, "aggressive_side": "NEUTRAL",
              "depth_bid": 0, "depth_ask": 0, "wall_level": 0, "wall_side": ""}
    try:
        ob = call(_info.l2_snapshot, COIN, timeout=10)
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
        log(f"[FLOW] OB delta error: {e}")
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
    Returns: {"cvd": float, "cvd_slope": float, "signal": "BUY"|"SELL"|"NEUTRAL"}
    """
    if len(_cvd_buffer) < 5:
        return {"cvd": 0, "cvd_slope": 0, "signal": "NEUTRAL"}

    deltas = [d["delta"] for d in _cvd_buffer]
    cvd = sum(deltas)

    # Slope: regressione lineare sui delta recenti (ultimi 20)
    recent = deltas[-20:] if len(deltas) >= 20 else deltas
    x = np.arange(len(recent))
    if len(recent) >= 3:
        slope = np.polyfit(x, recent, 1)[0]
    else:
        slope = 0

    signal = "NEUTRAL"
    if cvd > 0 and slope > 0:
        signal = "BUY"
    elif cvd < 0 and slope < 0:
        signal = "SELL"

    return {"cvd": round(cvd, 2), "cvd_slope": round(slope, 4), "signal": signal}


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
            log(f"[ML] Update #{self.n_samples} | Acc:{acc:.0%} | pred:{pred:.2f} actual:{outcome}")

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
            log(f"[ML] Load error: {e}")


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
    Predict win probability per il segnale corrente.
    Returns: (prob, adjustment_dict)
    """
    prob = _ml_model.predict_proba(features)
    acc = _ml_model.get_accuracy()

    # Solo applica ML se ha abbastanza dati E buona accuracy
    if _ml_model.n_samples < 20 or acc < 0.52:
        return prob, {"ml_active": False, "prob": prob, "reason": f"warmup ({_ml_model.n_samples} samples)"}

    adjustments = {"ml_active": True, "prob": round(prob, 3), "acc": round(acc, 3)}

    # Signal filtering: blocca segnali con prob < 30%
    if prob < 0.30:
        adjustments["action"] = "BLOCK"
        adjustments["reason"] = f"ML prob {prob:.0%} < 30%"
    # Size reduction: prob 30-45% → size × 0.6
    elif prob < 0.45:
        adjustments["action"] = "REDUCE"
        adjustments["size_mult"] = 0.6
        adjustments["reason"] = f"ML prob {prob:.0%} → reduce size"
    # Normal: prob 45-65%
    elif prob < 0.65:
        adjustments["action"] = "NORMAL"
        adjustments["size_mult"] = 1.0
    # Boost: prob > 65% → size × 1.2
    else:
        adjustments["action"] = "BOOST"
        adjustments["size_mult"] = 1.2
        adjustments["reason"] = f"ML prob {prob:.0%} → boost"

    return prob, adjustments


def ml_record_outcome(features, won):
    """Record trade outcome per aggiornare il modello online."""
    _ml_model.update(features, 1 if won else 0)
    # Salva modello su Redis periodicamente
    if _ml_model.n_samples % 5 == 0:
        _rset("btc7:ml_model", _ml_model.to_dict())
        log(f"[ML] Model saved ({_ml_model.n_samples} samples, acc:{_ml_model.get_accuracy():.0%})")


def ml_load_model():
    """Carica modello ML da Redis (persistenza tra restart)."""
    d = _rget("btc7:ml_model")
    if d:
        _ml_model.from_dict(d)
        log(f"[ML] Model loaded: {_ml_model.n_samples} samples, weights:{np.abs(_ml_model.weights).sum():.2f}")
    else:
        log("[ML] No saved model — starting fresh")
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
_params = {}
_is_trading = False

# ================================================================
# SHARED REDIS — BTC Scalper = Radar, Combined Bot = Fleet
# ================================================================
# Keys condivise tra i 2 worker:
#   fleet:bias        → "LONG_ONLY" | "SHORT_ONLY" | "NEUTRAL" (set dal BTC Scalper)
#   fleet:btc_regime  → "BULL" | "BEAR" | "RANGE" (regime 4h dal BTC Scalper)
#   fleet:btc_pos     → {"direction":"LONG","size":0.01,"entry":74000} o null
#   fleet:kill_switch → true/false (emergency stop per entrambi)
#   fleet:daily_pnl   → {"btc": -3.2, "alt": 1.5, "total": -1.7, "ts": ...}
#   fleet:dominance   → {"btc_dom_trend": "UP"|"DOWN"|"FLAT", "ts": ...}

def publish_bias(bias, reason=""):
    """BTC Scalper pubblica il bias giornaliero per il Combined Bot."""
    data = {"bias": bias, "reason": reason, "ts": time.time()}
    _rset("fleet:bias", data)
    log(f"[FLEET] Bias pubblicato: {bias} | {reason}")

def publish_btc_regime(regime):
    """Pubblica regime BTC 4h — Combined Bot lo legge."""
    _rset("fleet:btc_regime", {"regime": regime, "ts": time.time()})

def publish_btc_position(pos):
    """Pubblica posizione BTC — Combined Bot evita correlazione."""
    if pos:
        d = "LONG" if pos["szi"] > 0 else "SHORT"
        _rset("fleet:btc_pos", {"direction": d, "size": abs(pos["szi"]),
                                 "entry": pos["entry"], "ts": time.time()})
    else:
        _rset("fleet:btc_pos", None)

def check_kill_switch():
    """Controlla se il kill switch globale è attivo."""
    ks = _rget("fleet:kill_switch")
    if ks and ks.get("active"):
        return True, ks.get("reason", "kill switch")
    return False, ""

def update_fleet_daily_pnl(btc_pnl):
    """Aggiorna PnL giornaliero condiviso. Attiva kill switch se > 3% loss."""
    existing = _rget("fleet:daily_pnl") or {}
    alt_pnl = existing.get("alt", 0)
    total = btc_pnl + alt_pnl
    bal = get_balance()
    loss_pct = abs(total) / max(bal, 1) * 100 if total < 0 else 0

    _rset("fleet:daily_pnl", {
        "btc": round(btc_pnl, 2), "alt": round(alt_pnl, 2),
        "total": round(total, 2), "loss_pct": round(loss_pct, 1),
        "ts": time.time()
    })

    if loss_pct >= 3.0:
        _rset("fleet:kill_switch", {"active": True,
              "reason": f"Daily loss {loss_pct:.1f}% (${total:.2f})", "ts": time.time()})
        log(f"[FLEET] 🚨 KILL SWITCH — loss {loss_pct:.1f}% (${total:.2f})")
        tg(f"🚨 <b>KILL SWITCH</b> attivato\nLoss: {loss_pct:.1f}% (${total:.2f})")
        execute_kill_switch()

def compute_hourly_bias():
    """
    Versione V7.1 - Multi-timeframe Momentum + Order Flow + Sentiment.
    Cattura i dump veloci (5m) e i trend lenti (1h).
    """
    global _last_oi, _regime
    fz = get_funding_z()
    sentiment = get_sentiment_score() # Modulo Sentiment
    current_oi = get_btc_open_interest() # Modulo Order Flow
    
    # Calcolo variazione OI (Order Flow)
    oi_change = (current_oi - _last_oi) / _last_oi if _last_oi > 0 else 0
    _last_oi = current_oi

    # 1. Recupero Momentum su due timeframe
    short_momentum_5m = 0
    short_momentum_1h = 0
    try:
        # Recupero 15m (per simulare il 5m se non hai candele da 5m) o fetch diretto 5m
        df_5m = fetch_df("15m", 2) # Se hai candele 5m usa "5m"
        df_1h = fetch_df("1h", 2)
        
        if df_5m is not None and len(df_5m) >= 2:
            short_momentum_5m = (float(df_5m['close'].iloc[-1]) / float(df_5m['close'].iloc[-2])) - 1
            
        if df_1h is not None and len(df_1h) >= 2:
            short_momentum_1h = (float(df_1h['close'].iloc[-1]) / float(df_1h['close'].iloc[-2])) - 1
    except Exception as e:
        log(f"[BIAS_ERR] Errore momentum: {e}")

    # ================================================================
    # GERARCHIA DI DECISIONE (TUA LOGICA POTENZIATA)
    # ================================================================
    
    # A. FAST DUMP (Priorità Massima)
    # Se scende dello 0.2% in 5 min, attiviamo SHORT_ONLY immediato.
    # Se l'OI sale (oi_change > 0), il segnale è confermato da "mani forti".
    if short_momentum_5m < -0.002:
        bias = "SHORT_ONLY"
        reason = f"FAST DUMP 5m ({short_momentum_5m:+.2%}) | OI:{oi_change:+.2%}"

    # B. TREND 1h (Cattura cali costanti come 70k -> 68.5k)
    elif short_momentum_1h < -0.004:
        bias = "SHORT_ONLY" # O "SHORT_BIAS" se preferisci essere meno rigido
        reason = f"1h DOWNTREND ({short_momentum_1h:+.2%})"

    elif short_momentum_1h > 0.004:
        bias = "LONG_ONLY"
        reason = f"1h UPTREND ({short_momentum_1h:+.2%})"

    # C. FILTRO SENTIMENT (Contrarian)
    elif sentiment < 25:
        bias = "LONG_ONLY"
        reason = f"EXTREME FEAR ({sentiment}) - Looking for bounce"
    
    # D. LOGICA DI REGIME (Default in assenza di momentum forte)
    elif _regime == "BEAR":
        bias = "SHORT_ONLY" if fz > -0.5 else "NEUTRAL"
        reason = f"BEAR REGIME + FZ:{fz:+.1f}"

    elif _regime == "BULL":
        # Se il momentum orario accenna a scendere, meglio stare Neutrali
        bias = "LONG_ONLY" if short_momentum_1h >= -0.0005 else "NEUTRAL"
        reason = f"BULL REGIME + Mom:{short_momentum_1h:+.2%}"

    else:
        bias = "NEUTRAL"
        reason = f"STABLE/RANGE (F&G:{sentiment})"

    # Log e pubblicazione
    publish_bias(bias, reason)
    log(f"[FLEET V7] Bias: {bias} | {reason}")
    return bias
  
def execute_kill_switch():
    """
    KILL SWITCH: chiude TUTTE le posizioni aperte su entrambi i bot.
    Chiamato quando il daily loss supera il 3%.
    """
    pos = get_position()
    if pos:
        szi = pos["szi"]
        d = "LONG" if szi > 0 else "SHORT"
        mid = get_mid()
        size_abs = rpx(abs(szi), _coin_meta.get("sz_dec", 5))
        close_px = rpx(mid * (0.995 if d == "LONG" else 1.005),
                       _coin_meta.get("px_dec", 1))
        try:
            call(_exchange.order, COIN, d != "LONG", size_abs, close_px,
                 {"limit": {"tif": "Ioc"}}, False, timeout=15)
            log(f"[FLEET] 🚨 KILL: chiuso {d} BTC @ {mid:.0f}")
            tg(f"🚨 <b>KILL SWITCH</b> — chiuso BTC {d}")
            # Cancella tutti gli ordini
            for o in get_open_orders():
                if o.get("coin") == COIN:
                    try: call(_exchange.cancel, COIN, o["oid"], timeout=10)
                    except: pass
        except Exception as e:
            log(f"[FLEET] KILL error: {e}")
            tg(f"🚨 KILL SWITCH ERROR: {e}")

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
    recent = [t for t in _trades_today if now - t.get("ts_close", t.get("ts", 0)) < 86400]
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
    daily_pnl = sum(t["pnl"] for t in _trades_today if now - t.get("ts_close", t.get("ts", 0)) < 86400)
    if daily_pnl <= -MAX_DAILY_LOSS:
        return True, f"daily loss ${daily_pnl:.1f}"
    if _consec_losses >= MAX_CONSEC_LOSS:
        # Check if 30min pause has passed since last loss
        last_loss_ts = max((t.get("ts_close", t.get("ts", 0)) for t in _trades_today if t["pnl"] < 0), default=0)
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

    # Merge 1h indicators onto 15m (forward fill)
    h_cols = df_1h[['t_hour','rsi','macd_hist','ema9','ema21','ema_slope','ema50','ema200','adx']].copy()
    h_cols.columns = ['t_hour','rsi_1h','macd_1h','ema9_1h','ema21_1h','slope_1h','ema50_1h','ema200_1h','adx_1h']
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
        while i < n - fwd:
            try:
                if not sig_cond(i):
                    i += 1; continue
            except:
                i += 1; continue

            px_i = c[i]; atr_i = atr5[i]
            if atr_i <= 0: i += 1; continue

            # ── Detect mode per questa candela ──
            adx_i = adx1h[i] if not np.isnan(adx1h[i]) else 20
            vol_i = vol5[i]
            bb_i = bb[i] if not np.isnan(bb[i]) else 0
            e9 = ema9_1h[i]; e21 = ema21_1h[i]
            is_trend = (e9 > e21) or (e9 < e21)  # direzionale

            if vol_i >= 3.0:
                m_sl_atr, m_sl_min, m_sl_max = FLASH_SL_ATR, FLASH_SL_MIN, FLASH_SL_MAX
                m_tp_fixed = FLASH_TP_PCT
                m_tp_rr = 0  # usa tp_fixed
            elif adx_i > 25 and is_trend:
                m_sl_atr, m_sl_min, m_sl_max = TREND_SL_ATR, TREND_SL_MIN, TREND_SL_MAX
                m_tp_fixed = 0
                m_tp_rr = TREND_TP_RR
            else:
                m_sl_atr, m_sl_min, m_sl_max = RANGE_SL_ATR, RANGE_SL_MIN, RANGE_SL_MAX
                m_tp_fixed = RANGE_TP_PCT
                m_tp_rr = 0  # usa tp_fixed

            sl_d = max(atr_i * m_sl_atr, px_i * m_sl_min)
            sl_d = min(sl_d, px_i * m_sl_max)
            if m_tp_fixed > 0:
                tp_d = px_i * m_tp_fixed
            else:
                tp_d = sl_d * m_tp_rr
            tp_d = max(tp_d, px_i * TP_PRICE_PCT * 0.8)  # floor

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
    """
    Check if signal has edge. Usa pf_recent (ultimi 30 trade) come filtro primario
    — reagisce al mercato attuale, non a 30gg fa.
    Blocca solo se ENTRAMBI pf E pf_recent sono sotto soglia.
    """
    key = f"{sig_type}_{direction}"
    bt = _bt_results.get(key, {})
    pf = bt.get("pf", 0)
    pf_recent = bt.get("pf_recent", 0)
    n = bt.get("n", 0)
    if n < 10:
        return True  # not enough data — allow
    # Passa se il recent mostra edge, anche se lo storico è debole
    if pf_recent >= 0.9:
        return True
    # Blocca solo se entrambi sono sotto soglia
    if pf < 0.7 and pf_recent < 0.7:
        return False
    # Edge debole ma non assente
    if pf >= 0.8 or pf_recent >= 0.8:
        return True
    return False
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

    direction = None
    sig_type = None
    details = ""

    # In RANGE mode (ADX < 20): entrambe le direzioni permesse
    # In TREND/FLASH mode: rispetta il regime (BULL→LONG, BEAR→SHORT)
    allow_long = (regime in ("BULL", "RANGE") or scalp_mode == "RANGE")
    allow_short = (regime in ("BEAR", "RANGE") or scalp_mode == "RANGE")

    # ── LONG SIGNALS ──
    if allow_long:
        long_setup = (scalp_mode == "RANGE") or (ema9_1h > ema21_1h)
        if long_setup:
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
                details = f"RSI15:{rsi5:.0f} MACD↑ slope:{slope5:.4f}"
            elif breakout and is_signal_allowed("BREAKOUT", "LONG"):
                direction = "LONG"; sig_type = "BREAKOUT"
                details = f"RSI15:{rsi5:.0f} HH vol:{vol5:.1f}x"

    # ── SHORT SIGNALS ──
    if direction is None and allow_short:
        short_setup = (scalp_mode == "RANGE") or (ema9_1h < ema21_1h)
        if short_setup:
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
                details = f"RSI15:{rsi5:.0f} MACD↓ slope:{slope5:.4f}"
            elif breakdown and is_signal_allowed("BREAKDOWN", "SHORT"):
                direction = "SHORT"; sig_type = "BREAKDOWN"
                details = f"RSI15:{rsi5:.0f} LL vol:{vol5:.1f}x"

    # ── REVERSAL (solo in RANGE regime/mode) ──
    if direction is None and (regime == "RANGE" or scalp_mode == "RANGE"):
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

            mode_ctx = {
                "RANGE": "Mean reversion scalp. TP fisso 0.4%, SL stretto. Cerchiamo rimbalzo su supporto/resistenza.",
                "TREND": "Momentum pullback. TP con trailing, R:R 1:2. Cerchiamo continuazione del trend.",
                "FLASH": "Volatility breakout. TP fulmineo 0.3%, entry/exit veloci. Volume spike o funding estremo.",
            }.get(scalp_mode, "")

            prompt = (f"BTC {direction} {sig_type} signal. Mode:{scalp_mode}. Regime:{regime}.\n"
                     f"Strategy: {mode_ctx}\n"
                     f"RSI15m:{rsi5:.0f} RSI1h:{rsi1h:.0f} ADX1h:{adx1h:.0f} MACD15m:{macd5:.1f} Vol:{vol5:.1f}x BB:{bb5:.2f}\n"
                     f"Last 6 candles 15m:\n{ohlcv_str}"
                     f"Rate confidence 1-10. In {scalp_mode} mode, "
                     f"{'suggest tight SL for quick scalp' if scalp_mode in ('RANGE','FLASH') else 'suggest SL adjustment (tight/normal/wide)'}.\n"
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
    tp_dist = max(tp_dist, px * TP_PRICE_PCT * 0.8)  # floor 80% del TP target

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

    # ══════════════════════════════════════════════════════════
    # V7 PREDICTIVE ANALYTICS LAYER
    # ══════════════════════════════════════════════════════════

    # ── SENTIMENT ADJUSTMENT ──
    sent_score = get_sentiment_score()
    sent_size_adj, sent_sl_adj, sent_conf_adj = get_sentiment_adjustment()
    sent_bias, _ = get_sentiment_bias()

    # Contrarian filter: se extreme greed e stiamo andando long → penalizza
    if sent_bias == "EXTREME_GREED" and direction == "LONG":
        setup -= 15
        details += f" ⚠️sent:GREED→LONG penalized"
    elif sent_bias == "EXTREME_FEAR" and direction == "SHORT":
        setup -= 15
        details += f" ⚠️sent:FEAR→SHORT penalized"
    # Contrarian boost: extreme fear + long o extreme greed + short
    elif sent_bias == "EXTREME_FEAR" and direction == "LONG":
        setup += 10
        details += f" ✦sent:contrarian LONG"
    elif sent_bias == "EXTREME_GREED" and direction == "SHORT":
        setup += 10
        details += f" ✦sent:contrarian SHORT"

    # Apply sentiment adjustments
    size_mult *= sent_size_adj
    sl_dist *= sent_sl_adj
    # Ricalcola SL/TP se sentiment ha modificato sl_dist
    if sent_sl_adj != 1.0:
        if direction == "LONG":
            sl = px - sl_dist
        else:
            sl = px + sl_dist
    details += f" sent:{sent_score}"

    # ── ORDER FLOW CONFIRMATION ──
    flow_sig = get_flow_signal()
    flow_bias = flow_sig["bias"]
    flow_conf = flow_sig["confidence"]

    # Flow conferma la direzione → boost
    if (flow_bias == "BUY" and direction == "LONG") or \
       (flow_bias == "SELL" and direction == "SHORT"):
        setup += 5 + flow_conf
        size_mult = min(size_mult * 1.1, 1.2)
        details += f" ✦flow:{flow_bias}({flow_conf})"
    # Flow contraddice la direzione → penalizza
    elif (flow_bias == "SELL" and direction == "LONG") or \
         (flow_bias == "BUY" and direction == "SHORT"):
        setup -= 5 + flow_conf
        if flow_conf >= 6:
            log(f"⚠️ Flow {flow_bias} vs {direction} (conf:{flow_conf}) — signal weakened")
            size_mult *= 0.7
        details += f" ⚠️flow:{flow_bias}({flow_conf})vs{direction}"
    else:
        details += f" flow:NEUTRAL"

    # Delta divergence: high-priority reversal signal
    flow_data = _flow_cache.get("data", {})
    div = flow_data.get("delta_divergence", {})
    if div.get("divergence"):
        if (div["type"] == "BEARISH" and direction == "LONG"):
            setup -= 20
            details += " ⚠️DIV:BEAR"
            log(f"⚠️ Bearish divergence detected — LONG penalized")
        elif (div["type"] == "BULLISH" and direction == "SHORT"):
            setup -= 20
            details += " ⚠️DIV:BULL"
            log(f"⚠️ Bullish divergence detected — SHORT penalized")

    # Re-check setup after adjustments
    if setup < 35:
        log(f"⚠️ Setup score {setup}/100 dopo analytics layer — troppo basso")
        return None

    # ── MACHINE LEARNING PREDICTION ──
    ob_imbalance = liq.get("imbalance", 0.5)
    cvd_data = flow_data.get("cvd_trend", {})
    oi_data = flow_data.get("oi_momentum", {})
    spread_val = liq.get("spread", 0.0005) or 0.0005

    ml_features = build_ml_features(
        rsi_15m=rsi5, rsi_1h=rsi1h, macd_hist=macd5, adx_1h=adx1h,
        vol_rel=vol5, funding_z=fz, sentiment=sent_score,
        ob_imbalance=ob_imbalance, cvd_slope=cvd_data.get("cvd_slope", 0),
        regime=regime, scalp_mode=scalp_mode, setup_score=setup,
        oi_change_pct=oi_data.get("oi_change_pct", 0),
        bb_pos=bb5, ema_slope=slope5, spread=spread_val
    )

    ml_prob, ml_adj = ml_predict_signal(ml_features)

    if ml_adj.get("ml_active"):
        action = ml_adj.get("action", "NORMAL")
        if action == "BLOCK":
            log(f"🤖 [ML] BLOCKED — P(win)={ml_prob:.0%} | {ml_adj.get('reason','')}")
            return None
        elif action == "REDUCE":
            size_mult *= ml_adj.get("size_mult", 0.6)
            log(f"🤖 [ML] REDUCE — P(win)={ml_prob:.0%} size_mult→{size_mult:.2f}")
        elif action == "BOOST":
            size_mult = min(size_mult * ml_adj.get("size_mult", 1.2), 1.3)
            log(f"🤖 [ML] BOOST — P(win)={ml_prob:.0%} size_mult→{size_mult:.2f}")
        details += f" ML:{ml_prob:.0%}({action})"
    else:
        details += f" ML:warmup"

    # Store ML features in signal for later outcome recording
    _last_ml_features = ml_features  # will be saved in pos_state

    size_mult = max(0.3, min(1.3, size_mult))

    log(f"[V7] Final: setup:{setup} size:{size_mult:.2f} sent:{sent_score} "
        f"flow:{flow_bias} ML:{ml_prob:.0%} | {details[-80:]}")

    return (direction, sig_type, sl, tp, px, atr5, details, sl_dist,
            size_mult, regime, setup, scalp_mode, ml_features)

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

def get_balance():
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
    trades = [t for t in _trades_today if now - t.get("ts_close", t.get("ts", 0)) < 86400]
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
    lc_info = f"Galaxy:{sd.get('lc_galaxy','off')} Sent:{sd.get('lc_sentiment','off')}%"
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
        f"  🌙 LunarCrush: {lc_info}\n"
        f"  📊 CryptoCompare: {cc_info}\n"
        f"  FGI: {sd.get('fgi','?')} | Funding: {sd.get('funding_sent','?')}\n"
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
# THREAD A — SCANNER (regime + backtest ogni 5 min)
# ================================================================
_scanner_ready = threading.Event()

def scanner_thread():
    log("[SCAN] Thread avviato")
    ml_load_model()  # V7: Load ML model from Redis
    while True:
        try:
            regime = update_regime()
            update_funding_oi()
            fz = get_funding_z()
            oi = get_oi_change()
            run_backtest()
            mid = get_mid()
            bal = get_balance()

            # V7: Update Order Flow and Sentiment
            flow_data = update_order_flow()
            sent_score = get_sentiment_score()
            flow_sig = get_flow_signal()

            # ADX per detect mode
            adx_val = 20
            try:
                df_1h = fetch_df("1h", 5)
                if df_1h is not None and len(df_1h) > 5 and 'adx' in df_1h.columns:
                    adx_val = float(df_1h.iloc[-1]['adx'])
            except: pass
            mode = "FLASH" if abs(fz) > 2.5 else "TREND" if adx_val > 25 and regime in ("BULL","BEAR") else "RANGE"

            log(f"[SCAN] ════════════════════════════════════════════════════")
            log(f"[SCAN] Regime:{regime} | ADX:{adx_val:.0f} → {mode} mode | BTC ${mid:,.0f}")
            log(f"[SCAN]  SIGNAL             PF    WR   N    STATUS")
            for k, v in _bt_results.items():
                pf = v.get("pf", 0); wr = v.get("wr", 0); n = v.get("n", 0)
                if pf >= 1.0: status = "✅ active"
                elif pf >= 0.8: status = "⚠️ edge weak"
                else: status = "❌ blocked"
                log(f"[SCAN]  {k:<18} {pf:.2f}  {wr:.0%}  {n:<4} {status}")
            log(f"[SCAN]  FZ:{fz:+.1f} | OI:{oi:+.2%} | Bal ${bal:.2f}")
            # V7: Extended analytics dashboard
            log(f"[SCAN]  ── V7.2 Analytics ──")
            sent_detail = get_sentiment_detail()
            lc_g = sent_detail.get("lc_galaxy", "off")
            lc_s = sent_detail.get("lc_sentiment", "off")
            cc_b = sent_detail.get("cc_bull_pct", "off")
            missing = sent_detail.get("missing_sources", [])
            miss_str = f" ⚠️missing:{','.join(missing)}" if missing else ""
            log(f"[SCAN]  Sentiment:{sent_score} | FGI:{sent_detail.get('fgi','?')} "
                f"LC_Galaxy:{lc_g} LC_Sent:{lc_s} CC_Bull:{cc_b} Fund:{sent_detail.get('funding_sent','?')}{miss_str}")
            log(f"[SCAN]  Flow:{flow_sig['bias']}({flow_sig['confidence']}) | {flow_sig['details']}")
            log(f"[SCAN]  ML: {_ml_model.n_samples} samples, acc:{_ml_model.get_accuracy():.0%}")
            if _ml_model.n_samples >= 20:
                imp = _ml_model.get_feature_importance()
                top3 = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:3]
                log(f"[SCAN]  ML top features: {', '.join(f'{k}:{v:.0%}' for k,v in top3)}")
            log(f"[SCAN] ════════════════════════════════════════════════════")

            # ── FLEET: pubblica regime e bias per Combined Bot ──
            publish_btc_regime(regime)
            bias = compute_hourly_bias()
            log(f"[SCAN] [FLEET] Bias: {bias} | Regime: {regime}")

        except Exception as e:
            log(f"[SCAN] Error: {e}")
            import traceback; traceback.print_exc()

        if not _scanner_ready.is_set():
            _scanner_ready.set()
            log("[SCAN] Ready flag set")

        for i in range(30):
            time.sleep(10)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] [SCAN] wait {(i+1)*10}s/300s", flush=True)


# ================================================================
# THREAD B — PROCESSOR (check signal ogni 10s)
# ================================================================
def processor_thread(sz_dec, px_dec):
    log("[PROC] Thread avviato — attendo Scanner...")
    _scanner_ready.wait()
    log("[PROC] Scanner pronto — avvio")
    
    while True:
        try:
            pos = get_position()
            mid = get_mid()
            
            if pos is not None:
                entry = pos["entry"]
                szi = pos["szi"]
                d = "LONG" if szi > 0 else "SHORT"
                pnl_pct = ((mid - entry)/entry if d == "LONG" else (entry - mid)/entry) * 100
                log(f"[PROC] In {d} @ {entry:.0f} PnL:{pnl_pct:+.2f}% | BTC ${mid:,.0f}")
                time.sleep(SCAN_INTERVAL)
                continue
            
            sig = check_signal()
            if sig is None:
                log(f"[PROC] no signal | {_regime} | BTC ${mid:,.0f}")
                time.sleep(SCAN_INTERVAL)
                continue
            
            direction, sig_type, sl, tp, entry_px, atr, details, sl_dist, size_mult, sig_regime, setup, scalp_mode, ml_features = sig
            sl_pct = sl_dist / entry_px * 100
            tp_dist = abs(tp - entry_px)
            tp_pct = tp_dist / entry_px * 100
            rr = tp_dist / sl_dist if sl_dist > 0 else 0
            log(f"[PROC] 📡 {direction} {sig_type} [{scalp_mode}]")
            log(f"[PROC] [{scalp_mode}] SL:{sl_pct:.2f}% TP:{tp_pct:.2f}% R:R=1:{rr:.1f} | AI:{details.split('AI:')[1] if 'AI:' in details else '?'}")
            log(f"[PROC] score:{setup} | {details}")
            
            if not is_funding_ok(direction):
                time.sleep(SCAN_INTERVAL)
                continue
            
            global _current_signal
            _current_signal = {
                "direction": direction, "sig_type": sig_type,
                "sl": sl, "tp": tp, "entry_px": entry_px,
                "sl_dist": sl_dist, "size_mult": size_mult,
                "regime": sig_regime, "setup": setup,
                "scalp_mode": scalp_mode, "ts": time.time(),
                "ml_features": ml_features
            }
            log(f"[PROC] → BTC [{direction}] [{sig_type}] [{scalp_mode}] published")
            
        except Exception as e:
            log(f"[PROC] Error: {e}")
            import traceback; traceback.print_exc()
        
        time.sleep(SCAN_INTERVAL)


# ================================================================
# THREAD C — EXECUTOR (esegue trade, gestisce posizione)
# ================================================================
_current_signal = None

def executor_thread(sz_dec, px_dec):
    global _current_signal, _last_trade_ts
    log("[EXEC] Thread avviato — attendo Scanner...")
    _scanner_ready.wait()
    log("[EXEC] Pronto")
    
    last_pos_state = load_pos_state()
    if last_pos_state:
        p = get_position()
        if p:
            log(f"[EXEC] 🔍 Posizione da Redis: {last_pos_state.get('type','?')} @ {last_pos_state.get('entry',0)}")
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
                entry = last_pos_state["entry"]
                szi = last_pos_state["szi"]
                d = "LONG" if szi > 0 else "SHORT"
                size_abs = abs(szi)
                
                open_ts = last_pos_state.get("open_ts", time.time() - 3600)
                real_exit, real_pnl = compute_real_exit(d, entry, open_ts)
                
                if real_exit > 0 and real_pnl != 0:
                    exit_px = real_exit
                    pnl_pct = real_pnl * 100
                    pnl_usd = real_pnl * size_abs * entry
                else:
                    exit_px = mid
                    pnl_pct = ((mid - entry)/entry if d == "LONG" else (entry - mid)/entry) * 100
                    pnl_usd = pnl_pct/100 * size_abs * entry
                
                # Close reason
                close_reason = last_pos_state.get("close_reason", "")
                if not close_reason:
                    sl_d = last_pos_state.get("sl_dist", 0)
                    if sl_d > 0:
                        tp_d = sl_d * TP_RR
                        if d == "LONG":
                            hit_sl = mid <= entry - sl_d * 0.9
                            hit_tp = mid >= entry + tp_d * 0.9
                        else:
                            hit_sl = mid >= entry + sl_d * 0.9
                            hit_tp = mid <= entry - tp_d * 0.9
                        if hit_tp: close_reason = "🎯 Take Profit"
                        elif hit_sl: close_reason = "🛑 Stop Loss"
                        elif last_pos_state.get("trailing_active"): close_reason = "📈 Trailing Stop"
                        else: close_reason = "❓ Unknown"
                
                emoji = "✅" if pnl_usd > 0 else "❌"
                sig_type = last_pos_state.get("type", "?")
                trade_regime = last_pos_state.get("regime", "?")
                duration_s = time.time() - open_ts
                dur_str = f"{duration_s:.0f}s" if duration_s < 60 else f"{duration_s/60:.0f}min" if duration_s < 3600 else f"{duration_s/3600:.1f}h"
                
                save_trade(pnl_usd, d, entry, exit_px, sig_type=sig_type,
                           sl_dist=last_pos_state.get("sl_dist", 0), regime=trade_regime,
                           extra={"pnl_pct": pnl_pct, "sl_original": last_pos_state.get("sl_original",0),
                                  "tp_original": last_pos_state.get("tp_original",0),
                                  "trailing_activated_at": last_pos_state.get("trailing_activated_at",0),
                                  "trailing_final_sl": last_pos_state.get("current_ts",0),
                                  "trailing_moves": last_pos_state.get("trailing_moves",0),
                                  "close_reason": close_reason, "duration_s": duration_s,
                                  "ts_open": open_ts})
                save_pos_state(None)
                
                log(f"[EXEC] {emoji} CLOSED {d} {sig_type} | {close_reason} | ${pnl_usd:+.2f} ({pnl_pct:+.2f}%) | {dur_str}")
                tg(f"{emoji} <b>BTC {d} CLOSED</b>\n📊 PnL: ${pnl_usd:+.2f} ({pnl_pct:+.2f}%)\n🏷 {close_reason}\n📍 {entry:.1f}→{exit_px:.1f} | {dur_str}")

                # FLEET: aggiorna PnL condiviso
                daily_btc_pnl = sum(t["pnl"] for t in _trades_today if time.time()-t.get("ts_close",t.get("ts",0))<86400)
                update_fleet_daily_pnl(daily_btc_pnl)

                # ── V7: ML ONLINE LEARNING — record outcome ──
                ml_feats = last_pos_state.get("ml_features", [])
                if ml_feats and len(ml_feats) == OnlineGBClassifier.N_FEATURES:
                    won = pnl_usd > 0
                    ml_record_outcome(ml_feats, won)
                    log(f"[ML] Recorded: {'WIN' if won else 'LOSS'} | acc:{_ml_model.get_accuracy():.0%}")

                last_pos_state = None
            
            # ── Trade management (in posizione) ──
            if pos is not None and last_pos_state:
                entry = pos["entry"]
                szi = pos["szi"]
                d = "LONG" if szi > 0 else "SHORT"
                pnl_pct = ((mid - entry)/entry if d == "LONG" else (entry - mid)/entry) * 100
                
                trade_mode = last_pos_state.get("scalp_mode", "TREND")
                atr_now = last_pos_state.get("atr", 0)
                
                # Refresh ATR
                if cycle % 6 == 0:
                    df_m = fetch_df("15m", 1)
                    if df_m is not None and len(df_m) >= 5:
                        atr_now = float(df_m.iloc[-1]['atr'])
                        last_pos_state["atr"] = atr_now
                
                # Trailing (solo TREND)
                if atr_now > 0 and trade_mode == "TREND":
                    update_trailing(last_pos_state, mid, atr_now, sz_dec, px_dec)
                
                # Partial close (solo TREND)
                if trade_mode == "TREND":
                    check_partial_close(last_pos_state, mid, sz_dec, px_dec)
                
                # AI management
                mgmt_interval = {"FLASH": 30, "RANGE": 60, "TREND": 120}.get(trade_mode, 120)
                last_mgmt = last_pos_state.get("last_mgmt", 0)
                if time.time() - last_mgmt >= mgmt_interval:
                    last_pos_state["last_mgmt"] = time.time()
                    # AI call per management (simplified — HOLD/TIGHTEN/EXIT)
                    try:
                        api_key = os.getenv("ANTHROPIC_API_KEY", "")
                        if api_key:
                            df_m = fetch_df("15m", 1)
                            if df_m is not None and len(df_m) >= 5:
                                rm = df_m.iloc[-1]
                                ctx = f"RSI:{rm['rsi']:.0f} MACD:{rm['macd_hist']:.1f} vol:{rm['vol_rel']:.1f}x"
                                prompt = (f"BTC {d} [{trade_mode}]. Entry:{entry:.1f} Now:{mid:.1f} PnL:{pnl_pct:+.1f}%\n"
                                         f"15m: {ctx}\nDecide: HOLD, TIGHTEN, EXIT.\n"
                                         f"JSON: {{\"action\":\"HOLD|TIGHTEN|EXIT\",\"reason\":\"<5w>\"}}")
                                resp = requests.post("https://api.anthropic.com/v1/messages",
                                    headers={"Content-Type":"application/json","x-api-key":api_key,"anthropic-version":"2023-06-01"},
                                    json={"model":"claude-haiku-4-5-20251001","max_tokens":100,
                                          "messages":[{"role":"user","content":prompt}]}, timeout=10)
                                if resp.status_code == 200:
                                    txt = resp.json()["content"][0]["text"]
                                    s = txt.find("{"); e = txt.rfind("}")+1
                                    if s >= 0 and e > s:
                                        aj = json.loads(txt[s:e])
                                        action = aj.get("action","HOLD").upper()
                                        reason = aj.get("reason","")
                                        if action == "EXIT":
                                            last_pos_state["close_reason"] = f"🤖 AI: {reason}"
                                            size_abs = rpx(abs(szi), sz_dec)
                                            close_px = rpx(mid * (0.997 if d=="LONG" else 1.003), px_dec)
                                            call(_exchange.order, COIN, d!="LONG", size_abs, close_px,
                                                 {"limit":{"tif":"Ioc"}}, False, timeout=15)
                                            log(f"[EXEC] 🚪 AI EXIT {reason}")
                                        elif action == "TIGHTEN" and pnl_pct > 0.1:
                                            new_sl = entry * (1.001 if d=="LONG" else 0.999)
                                            new_sl = rpx(new_sl, px_dec)
                                            opens = get_open_orders()
                                            for o in opens:
                                                if o.get("coin")==COIN and o.get("orderType")=="Stop Market":
                                                    call(_exchange.cancel, COIN, o["oid"], timeout=10)
                                            time.sleep(0.3)
                                            size_abs = rpx(abs(szi), sz_dec)
                                            call(_exchange.order, COIN, d!="LONG", size_abs, new_sl,
                                                 {"trigger":{"triggerPx":new_sl,"isMarket":True,"tpsl":"sl"}},
                                                 True, timeout=15)
                                            log(f"[EXEC] 🔒 TIGHTEN SL→{new_sl} | {reason}")
                                        else:
                                            log(f"[EXEC] 📊 {action} | PnL:{pnl_pct:+.1f}% | {reason}")
                    except Exception as ex:
                        log(f"[EXEC] AI mgmt error: {ex}")
                
                log(f"[EXEC] #{cycle} {d} @ {entry:.0f} PnL:{pnl_pct:+.1f}% [{trade_mode}] | ${bal:.2f}")
                time.sleep(SCAN_INTERVAL)
                continue
            
            # ── FLEET: pubblica posizione BTC ──
            publish_btc_position(pos)

            # ── Kill switch globale ──
            ks, ks_reason = check_kill_switch()
            if ks:
                if cycle % 6 == 1: log(f"[EXEC] 🚨 KILL SWITCH: {ks_reason}")
                time.sleep(SCAN_INTERVAL)
                continue

            # ── Check for new signal ──
            cb, cb_reason = check_circuit_breaker()
            if cb:
                if cycle % 6 == 1: log(f"[EXEC] 🛑 {cb_reason}")
                time.sleep(SCAN_INTERVAL)
                continue
            
            if time.time() - _last_trade_ts < COOLDOWN_SEC:
                time.sleep(SCAN_INTERVAL)
                continue
            
            if _is_trading:
                time.sleep(SCAN_INTERVAL)
                continue
            
            sig = _current_signal
            if sig and time.time() - sig.get("ts", 0) < 120:  # segnale fresco < 2min
                _current_signal = None  # consuma
                
                direction = sig["direction"]
                sm = sig['scalp_mode']
                log(f"[EXEC] 🎯 BTC {direction} {sig['sig_type']} [{sm}] @ {sig['entry_px']:,.0f}")
                
                success = open_trade(direction, sig["sl"], sig["tp"], sig["entry_px"],
                                     sig["sl_dist"], sz_dec, px_dec, sig["size_mult"], sig["scalp_mode"])
                if success:
                    p = get_position()
                    if p:
                        last_pos_state = p
                        last_pos_state.update({
                            "type": sig["sig_type"], "sl_dist": sig["sl_dist"],
                            "sl_original": sig["sl"], "tp_original": sig["tp"],
                            "regime": sig["regime"], "scalp_mode": sig["scalp_mode"],
                            "setup_score": sig["setup"], "ai_confidence": 7,
                            "last_mgmt": 0, "partial_done": False, "partial_close_px": 0,
                            "partial_pnl_pct": 0, "trailing_active": False,
                            "trailing_activated_at": 0, "trailing_moves": 0,
                            "current_ts": 0, "atr": 0, "open_ts": time.time(), "close_reason": "",
                            "ml_features": sig.get("ml_features", [])
                        })
                        save_pos_state(last_pos_state)
                        sl_pct = sig["sl_dist"]/sig["entry_px"]*100
                        log(f"[EXEC] ✅ FILLED SL:{sig['sl']:,.0f}({sl_pct:.2f}%) TP:{sig['tp']:,.0f} risk:${RISK_USD}")
                else:
                    log(f"[EXEC] ❌ Trade fallito — cooldown {COOLDOWN_SEC}s")
                    _last_trade_ts = time.time()
            
            # Status log
            daily_pnl = sum(t["pnl"] for t in _trades_today if time.time()-t.get("ts_close",t.get("ts",0))<86400)
            n_today = len([t for t in _trades_today if time.time()-t.get("ts_close",t.get("ts",0))<86400])
            log(f"[EXEC] #{cycle} ${bal:.2f} | {_regime} | flat | {n_today} trades ${daily_pnl:+.2f} | BTC ${mid:,.0f}")
            
        except Exception as e:
            log(f"[EXEC] Error: {e}")
            import traceback; traceback.print_exc()
        
        time.sleep(SCAN_INTERVAL)


# ================================================================
# MAIN — 3 thread come V4
# ================================================================
def main():
    global _last_trade_ts
    log("🚀 BTC SCALPER V7 — Predictive Analytics Edition")
    log(f"Risk:${RISK_USD} Lev:{LEVERAGE}x Modes:RANGE/TREND/FLASH")
    log(f"✦ Sentiment Analysis | ✦ Order Flow | ✦ Machine Learning")
    
    load_state()
    sz_dec, px_dec = get_meta()
    
    # Thread A: Scanner (regime + backtest ogni 5 min)
    threading.Thread(target=scanner_thread, name="Scanner", daemon=True).start()
    
    # Thread B: Processor (check signal ogni 10s)
    threading.Thread(target=processor_thread, args=(sz_dec, px_dec), name="Processor", daemon=True).start()
    
    # Thread C: Executor nel main thread
    try:
        executor_thread(sz_dec, px_dec)
    except KeyboardInterrupt:
        log("🛑 Stop")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL: {e}", flush=True)
        import traceback; traceback.print_exc()
        time.sleep(60)
