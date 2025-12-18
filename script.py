"""
MOMENTUM PRO – FILTER (Telegram Enabled, SAFE) — v4.6 (FINAL)
GOAL:
- Bot runs continuously, BUT notifies you ONLY when there is a REAL momentum moment.
- No trade signals. Scanner only (input for deeper analysis).

FINAL rules:
- Auto-push ONLY for ALERT 3
- ALERT 3 requires:
  1) 15m direction NOT MIX
  2) strong 15m trigger score
  3) enough trigger reasons
  4) HARD: 15m VOLx must be >= 1.15 (real aggression NOW)
- Incremental klines (fetch last 2 candles only each scan)
- Soft rate limiter + backoff (protect vs Binance rate-limits/blocks)

v4.6 additions (analysis quality):
- RANGE snapshot:
  • 4h Low/High (last 24h) = last 6 candles of 4h
  • 1h Low/High (last 6h)  = last 6 candles of 1h
- DELTA (15m): buy vol vs sell vol using kline takerBuyBaseAssetVolume (col 9)
- ALERT META: First alert in this move / Repeat (n)
"""

import os
import time
import requests
import numpy as np
from datetime import datetime, UTC
from collections import deque

# ================== CONFIG ================== #

BINANCE_FUTURES = "https://fapi.binance.com"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
TIMEFRAMES = {"15m": "15m", "1h": "1h", "4h": "4h"}

EMA_FAST = 20
EMA_SLOW = 50
RSI_LEN = 14
ATR_LEN = 14

VOL_LOOKBACK = 20
SLOPE_LOOKBACK = 6

RETURN_LOOKBACK_15M = 6
RETURN_LOOKBACK_1H = 6
RETURN_LOOKBACK_4H = 6

ALERT2_SCORE = 5
ALERT3_SCORE = 7

# HARD rules
MIN_TRIGGER_SCORE_FOR_ALERT3 = 4
MIN_TRIGGER_REASONS_FOR_ALERT3 = 2
BLOCK_ALERT3_IF_DIR_MIX = True

# FINAL HARD aggression filter
REQUIRE_VOLX_FOR_ALERT3 = 1.15

# Anti-spam
COOLDOWN_ALERT3 = 90 * 60
DIR_MIX_BAND = 1

MIN_PRICE_MOVE = {"BTCUSDT": 0.25, "ETHUSDT": 0.30, "SOLUSDT": 0.40}
_last_alert_price = {}

# Scheduling
CMD_POLL_SECONDS = 5
SCAN_SECONDS = 300

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# ---- rate protection ----
REQUEST_LOG = deque(maxlen=600)
MAX_REQ_PER_MIN = 90
BACKOFF_BASE = 1.5
BACKOFF_MAX = 30.0

# ---- kline cache ----
KLINE_SEED_LIMIT = 220
KLINE_MIN_BARS = 120
KLINE_TAIL_FETCH = 2

# ---- NEW: meta windows ----
MOVE_WINDOW_SEC = 2 * 60 * 60  # 2h window to count repeats

# ============================================ #

_last_sent = {}
_last_alert_level = {}
_last_15m_candle = {}
_prev_oi = {}
_last_update_id = 0

# NEW: alert meta tracking (repeat counting)
_last_alert3_meta = {}  # symbol -> {"ts": float, "dir": str, "repeat": int}

_KLINE_CACHE = {}     # (symbol, tf) -> np.array of klines float
_DERIV_CACHE = {}     # symbol -> (ts, deriv_dict)
DERIV_TTL_SEC = 30

# ---------- RATE LIMIT HELPERS ---------- #

def _rate_guard():
    now = time.time()
    REQUEST_LOG.append(now)
    recent = 0
    for t in reversed(REQUEST_LOG):
        if now - t < 60:
            recent += 1
        else:
            break
    if recent > MAX_REQ_PER_MIN:
        time.sleep(1.2)

def _should_backoff(resp, json_data):
    if resp is not None and resp.status_code in (418, 429):
        return True
    if isinstance(json_data, dict) and "code" in json_data:
        if json_data["code"] in (-1003, -1015):
            return True
    return False

def _sleep_backoff(attempt):
    s = min(BACKOFF_MAX, BACKOFF_BASE * (2 ** attempt))
    time.sleep(s)

# ---------- NETWORK HELPERS ---------- #

def http_get_json(url, params=None, timeout=10):
    for attempt in range(5):
        _rate_guard()
        try:
            r = requests.get(url, params=params, timeout=timeout)
            try:
                data = r.json()
            except Exception:
                data = None

            if r.status_code >= 400:
                if _should_backoff(r, data):
                    _sleep_backoff(attempt)
                    continue
                r.raise_for_status()

            if _should_backoff(r, data):
                _sleep_backoff(attempt)
                continue

            return data
        except Exception:
            if attempt == 4:
                raise
            _sleep_backoff(attempt)

def http_post_json(url, payload, timeout=10):
    for attempt in range(5):
        _rate_guard()
        try:
            r = requests.post(url, json=payload, timeout=timeout)
            try:
                data = r.json()
            except Exception:
                data = {"ok": True, "raw": r.text[:200]}

            if r.status_code >= 400:
                if _should_backoff(r, data):
                    _sleep_backoff(attempt)
                    continue
                r.raise_for_status()

            if _should_backoff(r, data):
                _sleep_backoff(attempt)
                continue

            return data
        except Exception:
            if attempt == 4:
                raise
            _sleep_backoff(attempt)

# ---------- TELEGRAM ---------- #

def send_telegram(message: str, chat_id: str | None = None):
    if not TELEGRAM_TOKEN:
        return
    target = chat_id if chat_id is not None else TELEGRAM_CHAT_ID
    if not target:
        return
    if message and len(message) > 3900:
        message = message[:3900] + "\n…(truncated)"
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": target, "text": message, "parse_mode": "Markdown"}
    http_post_json(url, payload, timeout=10)

def telegram_delete_webhook():
    if not TELEGRAM_TOKEN:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/deleteWebhook"
    http_get_json(url, params={"drop_pending_updates": True}, timeout=10)

def telegram_get_updates(offset=None, timeout=0):
    if not TELEGRAM_TOKEN:
        return []
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
    params = {"timeout": timeout}
    if offset is not None:
        params["offset"] = offset
    data = http_get_json(url, params=params, timeout=10)
    return data.get("result", []) if isinstance(data, dict) else []

def is_allowed_chat(chat_id: str) -> bool:
    return str(chat_id) == str(TELEGRAM_CHAT_ID)

# ---------- INDICATORS ---------- #

def ema(series, length):
    series = np.asarray(series, dtype=float)
    alpha = 2 / (length + 1)
    out = np.empty_like(series)
    out[0] = series[0]
    for i in range(1, len(series)):
        out[i] = alpha * series[i] + (1 - alpha) * out[i - 1]
    return out

def rsi_wilder(close, length=14):
    close = np.asarray(close, dtype=float)
    if len(close) < length + 2:
        return np.full_like(close, 50.0)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    rsi_vals = np.empty(len(close))
    rsi_vals[:length] = 50.0
    avg_gain = np.mean(gain[:length])
    avg_loss = np.mean(loss[:length])
    rs = avg_gain / (avg_loss + 1e-12)
    rsi_vals[length] = 100 - (100 / (1 + rs))
    for i in range(length + 1, len(close)):
        avg_gain = (avg_gain * (length - 1) + gain[i - 1]) / length
        avg_loss = (avg_loss * (length - 1) + loss[i - 1]) / length
        rs = avg_gain / (avg_loss + 1e-12)
        rsi_vals[i] = 100 - (100 / (1 + rs))
    return rsi_vals

def atr_wilder(high, low, close, length=14):
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    if len(close) < length + 2:
        return np.full_like(close, np.nan)
    tr = np.empty(len(close))
    tr[0] = high[0] - low[0]
    for i in range(1, len(close)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    atr = np.empty(len(close))
    atr[:length] = np.mean(tr[:length])
    for i in range(length, len(close)):
        atr[i] = (atr[i - 1] * (length - 1) + tr[i]) / length
    return atr

def pct_change(a, b):
    return ((a - b) / (b + 1e-12)) * 100.0

# ---------- BINANCE DATA (INCREMENTAL) ---------- #

def _fetch_klines_raw(symbol, interval, limit=200):
    url = f"{BINANCE_FUTURES}/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    data = http_get_json(url, params=params, timeout=10)
    return np.array(data, dtype=float)

def get_klines_cached(symbol, tf):
    key = (symbol, tf)
    interval = TIMEFRAMES[tf]

    if key not in _KLINE_CACHE:
        k = _fetch_klines_raw(symbol, interval, limit=KLINE_SEED_LIMIT)
        _KLINE_CACHE[key] = k
        return k

    k = _KLINE_CACHE[key]
    if k is None or len(k) < KLINE_MIN_BARS:
        k = _fetch_klines_raw(symbol, interval, limit=KLINE_SEED_LIMIT)
        _KLINE_CACHE[key] = k
        return k

    tail = _fetch_klines_raw(symbol, interval, limit=KLINE_TAIL_FETCH)

    for row in tail:
        ot = row[0]
        if ot == k[-1, 0]:
            k[-1] = row
        elif ot > k[-1, 0]:
            k = np.vstack([k, row])

    if len(k) > 400:
        k = k[-350:]

    _KLINE_CACHE[key] = k
    return k

def fetch_derivatives_cached(symbol):
    now = time.time()
    hit = _DERIV_CACHE.get(symbol)
    if hit and (now - hit[0] < DERIV_TTL_SEC):
        return hit[1]

    mark = http_get_json(f"{BINANCE_FUTURES}/fapi/v1/premiumIndex", params={"symbol": symbol}, timeout=10)
    oi = http_get_json(f"{BINANCE_FUTURES}/fapi/v1/openInterest", params={"symbol": symbol}, timeout=10)

    mark_price = float(mark["markPrice"])
    index_price = float(mark["indexPrice"])
    funding_pct = float(mark["lastFundingRate"]) * 100.0
    basis_pct = (mark_price - index_price) / (index_price + 1e-12) * 100.0
    oi_val = float(oi["openInterest"])

    d = {"mark": mark_price, "index": index_price, "funding": funding_pct, "basis": basis_pct, "oi": oi_val}
    _DERIV_CACHE[symbol] = (now, d)
    return d

# ---------- SNAPSHOT ---------- #

def tf_snapshot(k, tf_name):
    close = k[:, 4]
    high = k[:, 2]
    low = k[:, 3]
    vol = k[:, 5]

    e20 = ema(close, EMA_FAST)
    e50 = ema(close, EMA_SLOW)
    rsi_v = rsi_wilder(close, RSI_LEN)
    atr_v = atr_wilder(high, low, close, ATR_LEN)

    sl = SLOPE_LOOKBACK
    e20_slp = e20[-1] - e20[-1 - sl] if len(e20) > sl else e20[-1] - e20[0]
    e50_slp = e50[-1] - e50[-1 - sl] if len(e50) > sl else e50[-1] - e50[0]
    rsi_slp = rsi_v[-1] - rsi_v[-1 - sl] if len(rsi_v) > sl else 0.0

    vol_ratio = vol[-1] / (np.mean(vol[-VOL_LOOKBACK:]) + 1e-12) if len(vol) >= VOL_LOOKBACK else 1.0
    open_time = int(k[-1, 0])

    # NEW: taker buy volume is column 9 in Binance kline array:
    # [0 openTime,1 open,2 high,3 low,4 close,5 volume,6 closeTime,7 quoteVol,8 trades,9 takerBuyBaseVol,10 takerBuyQuoteVol,11 ignore]
    taker_buy = float(k[-1, 9]) if k.shape[1] > 9 else float("nan")
    total_vol = float(vol[-1])
    taker_sell = total_vol - taker_buy if not np.isnan(taker_buy) else float("nan")
    delta = (taker_buy - taker_sell) if (not np.isnan(taker_buy)) else float("nan")

    return {
        "tf": tf_name,
        "open_time": open_time,
        "price": float(close[-1]),
        "ema20": float(e20[-1]),
        "ema50": float(e50[-1]),
        "ema20_slp": float(e20_slp),
        "ema50_slp": float(e50_slp),
        "rsi": float(rsi_v[-1]),
        "rsi_slp": float(rsi_slp),
        "atr": float(atr_v[-1]) if not np.isnan(atr_v[-1]) else float(np.nan),
        "vol_ratio": float(vol_ratio),
        "vol_total": float(total_vol),
        "vol_buy": float(taker_buy) if not np.isnan(taker_buy) else None,
        "vol_sell": float(taker_sell) if not np.isnan(taker_buy) else None,
        "vol_delta": float(delta) if not np.isnan(taker_buy) else None,
        "close_series": close
    }

def tf_trend_label(snap):
    price, e20, e50 = snap["price"], snap["ema20"], snap["ema50"]
    if price > e20 and e20 > e50:
        return "UP"
    if price < e20 and e20 < e50:
        return "DOWN"
    return "MIX"

def compute_returns(tf_snaps):
    out = {}
    for tf, snap in tf_snaps.items():
        close = snap["close_series"]
        lb = RETURN_LOOKBACK_15M if tf == "15m" else RETURN_LOOKBACK_1H if tf == "1h" else RETURN_LOOKBACK_4H
        out[tf] = pct_change(close[-1], close[-1 - lb]) if len(close) > lb else 0.0
    return out

# ---------- CONTEXT / TRIGGER ---------- #

def regime_score(tf_snap, direction):
    price, e20, e50 = tf_snap["price"], tf_snap["ema20"], tf_snap["ema50"]
    e20_slp, e50_slp, rsi = tf_snap["ema20_slp"], tf_snap["ema50_slp"], tf_snap["rsi"]
    score = 0
    reasons = []
    if direction == "UP":
        if price > e20 and price > e50: score += 2; reasons.append("HTF price above EMA20/50")
        if e20 > e50: score += 1; reasons.append("HTF EMA20 > EMA50")
        if e20_slp > 0 and e50_slp > 0: score += 1; reasons.append("HTF EMA slopes positive")
        if rsi >= 52: score += 1; reasons.append("HTF RSI supportive")
    else:
        if price < e20 and price < e50: score += 2; reasons.append("HTF price below EMA20/50")
        if e20 < e50: score += 1; reasons.append("HTF EMA20 < EMA50")
        if e20_slp < 0 and e50_slp < 0: score += 1; reasons.append("HTF EMA slopes negative")
        if rsi <= 48: score += 1; reasons.append("HTF RSI weak")
    return score, reasons

def trigger_score_15m(tf15, direction, ret_15m):
    price, e20, e50 = tf15["price"], tf15["ema20"], tf15["ema50"]
    rsi, rsi_slp, vol_ratio = tf15["rsi"], tf15["rsi_slp"], tf15["vol_ratio"]
    score = 0
    reasons = []

    if direction == "UP":
        if price > e20: score += 1; reasons.append("15m price above EMA20")
        if price > e20 and price > e50: score += 1; reasons.append("15m price above EMA20/50")
        if rsi >= 53: score += 1; reasons.append("15m RSI high")
        if rsi_slp >= 3: score += 1; reasons.append("15m RSI rising")
        if vol_ratio >= 1.15: score += 1; reasons.append(f"15m volume expansion ({vol_ratio:.2f}x)")
        if ret_15m >= 0.15: score += 1; reasons.append(f"15m return positive ({ret_15m:+.2f}%)")
        if vol_ratio >= 1.25 and rsi_slp >= 5: score += 1; reasons.append("15m aggression combo (vol + RSI slope)")
    else:
        if price < e20: score += 1; reasons.append("15m price below EMA20")
        if price < e20 and price < e50: score += 1; reasons.append("15m price below EMA20/50")
        if rsi <= 47: score += 1; reasons.append("15m RSI low")
        if rsi_slp <= -3: score += 1; reasons.append("15m RSI falling")
        if vol_ratio >= 1.15: score += 1; reasons.append(f"15m volume expansion ({vol_ratio:.2f}x)")
        if ret_15m <= -0.15: score += 1; reasons.append(f"15m return negative ({ret_15m:+.2f}%)")
        if vol_ratio >= 1.25 and rsi_slp <= -5: score += 1; reasons.append("15m aggression combo (vol + RSI slope)")

    return score, reasons

def choose_direction_from_15m(up_trg, dn_trg):
    diff = up_trg - dn_trg
    if abs(diff) <= DIR_MIX_BAND:
        return "MIX"
    return "UP" if diff > 0 else "DOWN"

# ---------- NEW: RANGE helpers ---------- #

def range_snapshot_from_klines(k, candles: int):
    """
    Returns (low, high) for the last N candles.
    k columns: high at [2], low at [3]
    """
    if k is None or len(k) < candles:
        return None, None
    tail = k[-candles:]
    lo = float(np.min(tail[:, 3]))
    hi = float(np.max(tail[:, 2]))
    return lo, hi

# ---------- NEW: ALERT meta ---------- #

def alert_meta_for(symbol: str, direction: str, alert_level: int):
    """
    Tracks repeats of ALERT 3 for a symbol within MOVE_WINDOW_SEC and same direction.
    """
    if alert_level != 3:
        return {"first_or_repeat": None, "repeat_n": None}

    now = time.time()
    prev = _last_alert3_meta.get(symbol)

    if (prev is None) or (now - prev["ts"] > MOVE_WINDOW_SEC) or (prev["dir"] != direction):
        meta = {"ts": now, "dir": direction, "repeat": 1}
        _last_alert3_meta[symbol] = meta
        return {"first_or_repeat": "First", "repeat_n": 1}

    prev["ts"] = now
    prev["repeat"] += 1
    _last_alert3_meta[symbol] = prev
    return {"first_or_repeat": "Repeat", "repeat_n": prev["repeat"]}

# ---------- ANALYSIS ---------- #

def analyze_symbol(symbol):
    tf_snaps = {}
    klines = {}
    for tf in TIMEFRAMES.keys():
        k = get_klines_cached(symbol, tf)
        klines[tf] = k
        tf_snaps[tf] = tf_snapshot(k, tf)

    returns = compute_returns(tf_snaps)
    ret_15m = returns["15m"]

    deriv = fetch_derivatives_cached(symbol)
    prev_oi = _prev_oi.get(symbol)
    oi_change_pct = pct_change(deriv["oi"], prev_oi) if prev_oi and prev_oi > 0 else None
    _prev_oi[symbol] = deriv["oi"]

    up_trg, up_trg_r = trigger_score_15m(tf_snaps["15m"], "UP", ret_15m)
    dn_trg, dn_trg_r = trigger_score_15m(tf_snaps["15m"], "DOWN", ret_15m)
    direction = choose_direction_from_15m(up_trg, dn_trg)

    if direction == "UP":
        ctx_1h, ctx_1h_r = regime_score(tf_snaps["1h"], "UP")
        ctx_4h, ctx_4h_r = regime_score(tf_snaps["4h"], "UP")
        trg_score, trg_reasons = up_trg, up_trg_r
        ctx_dir = "UP"
    elif direction == "DOWN":
        ctx_1h, ctx_1h_r = regime_score(tf_snaps["1h"], "DOWN")
        ctx_4h, ctx_4h_r = regime_score(tf_snaps["4h"], "DOWN")
        trg_score, trg_reasons = dn_trg, dn_trg_r
        ctx_dir = "DOWN"
    else:
        up_ctx_1h, up_ctx_1h_r = regime_score(tf_snaps["1h"], "UP")
        up_ctx_4h, up_ctx_4h_r = regime_score(tf_snaps["4h"], "UP")
        dn_ctx_1h, dn_ctx_1h_r = regime_score(tf_snaps["1h"], "DOWN")
        dn_ctx_4h, dn_ctx_4h_r = regime_score(tf_snaps["4h"], "DOWN")
        if (up_ctx_1h + up_ctx_4h) >= (dn_ctx_1h + dn_ctx_4h):
            ctx_1h, ctx_1h_r = up_ctx_1h, up_ctx_1h_r
            ctx_4h, ctx_4h_r = up_ctx_4h, up_ctx_4h_r
            ctx_dir = "UP-ish"
        else:
            ctx_1h, ctx_1h_r = dn_ctx_1h, dn_ctx_1h_r
            ctx_4h, ctx_4h_r = dn_ctx_4h, dn_ctx_4h_r
            ctx_dir = "DOWN-ish"
        trg_score = max(up_trg, dn_trg)
        trg_reasons = up_trg_r if up_trg >= dn_trg else dn_trg_r

    total_score = ctx_1h + ctx_4h + trg_score

    alert = 0
    if total_score >= ALERT2_SCORE:
        alert = 2
    if total_score >= ALERT3_SCORE:
        alert = 3

    if alert == 3 and len(trg_reasons) < MIN_TRIGGER_REASONS_FOR_ALERT3:
        alert = 2
    if alert == 3 and trg_score < MIN_TRIGGER_SCORE_FOR_ALERT3:
        alert = 2
    if alert == 3 and BLOCK_ALERT3_IF_DIR_MIX and direction == "MIX":
        alert = 2

    # FINAL HARD: real aggression NOW
    if alert == 3:
        volx_15m = tf_snaps["15m"]["vol_ratio"]
        if volx_15m < REQUIRE_VOLX_FOR_ALERT3:
            alert = 2

    # NEW: range snapshots
    lo_4h_24h, hi_4h_24h = range_snapshot_from_klines(klines["4h"], candles=6)  # 6*4h = 24h
    lo_1h_6h,  hi_1h_6h  = range_snapshot_from_klines(klines["1h"], candles=6)  # 6*1h = 6h

    # NEW: alert meta (first/repeat) only meaningful for alert 3
    meta = alert_meta_for(symbol, direction, alert)

    trend_labels = {tf: tf_trend_label(tf_snaps[tf]) for tf in tf_snaps.keys()}
    tf_public = {k: {kk: vv for kk, vv in v.items() if kk != "close_series"} for k, v in tf_snaps.items()}

    reasons = []
    reasons += [f"Context scored as: {ctx_dir}"]
    reasons += ["Setup (1h):"] + (ctx_1h_r if ctx_1h_r else ["(neutral)"])
    reasons += ["Setup (4h):"] + (ctx_4h_r if ctx_4h_r else ["(neutral)"])
    reasons += ["Trigger (15m):"] + (trg_reasons if trg_reasons else ["(no strong 15m trigger yet)"])

    return {
        "symbol": symbol,
        "alert": alert,
        "direction": direction,
        "total_score": total_score,
        "up_trg": up_trg,
        "dn_trg": dn_trg,
        "ctx_1h": ctx_1h,
        "ctx_4h": ctx_4h,
        "trg_score": trg_score,
        "reasons": reasons,
        "tf": tf_public,
        "returns": returns,
        "trend_labels": trend_labels,
        "deriv": deriv,
        "oi_change_pct": oi_change_pct,
        # NEW payload fields
        "range_4h_24h": {"low": lo_4h_24h, "high": hi_4h_24h},
        "range_1h_6h": {"low": lo_1h_6h, "high": hi_1h_6h},
        "alert_meta": meta
    }

# ---------- ANTI-SPAM / GATING ---------- #

def allowed_to_send(report):
    symbol = report["symbol"]
    direction = report["direction"]
    alert = report["alert"]
    now = time.time()

    if alert != 3:
        return False

    key = (symbol, direction)
    last_sent = _last_sent.get(key)
    last_level = _last_alert_level.get(key, 0)

    candle_ot = report["tf"]["15m"]["open_time"]
    last_candle_for_symbol = _last_15m_candle.get(symbol)

    if last_candle_for_symbol is not None and candle_ot == last_candle_for_symbol:
        return False
    if last_level >= 3:
        return False
    if last_sent is not None and (now - last_sent) < COOLDOWN_ALERT3:
        return False

    last_price = _last_alert_price.get(symbol)
    curr_price = report["tf"]["15m"]["price"]
    if last_price is not None and last_price > 0:
        move_pct = abs((curr_price - last_price) / last_price) * 100.0
        min_move = MIN_PRICE_MOVE.get(symbol, 0.30)
        if move_pct < min_move:
            return False

    return True

def mark_sent(report):
    symbol = report["symbol"]
    direction = report["direction"]
    alert = report["alert"]
    candle_ot = report["tf"]["15m"]["open_time"]

    _last_sent[(symbol, direction)] = time.time()
    _last_alert_level[(symbol, direction)] = alert
    _last_15m_candle[symbol] = candle_ot
    _last_alert_price[symbol] = report["tf"]["15m"]["price"]

# ---------- MESSAGE FORMAT ---------- #

def fmt_num(x, digits=2):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "n/a"
        return f"{x:.{digits}f}"
    except Exception:
        return "n/a"

def build_message(reports, title="MOMENTUM FILTER REPORT"):
    ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
    msg = f"📊 *{title}*\nUTC: `{ts}`\n"
    msg += "_(Scanner only — not a signal. Use as input for analysis.)_\n\n"

    for r in reports:
        sym = r["symbol"]
        alert = r["alert"]
        direction = r["direction"]
        total = r["total_score"]
        up_trg = r["up_trg"]
        dn_trg = r["dn_trg"]
        deriv = r["deriv"]
        oi_chg = r["oi_change_pct"]

        msg += (
            f"*{sym}*  |  *ALERT {alert}*  |  *DIR (15m): {direction}*  |  "
            f"total `{total}` (ctx1h `{r['ctx_1h']}` + ctx4h `{r['ctx_4h']}` + trg `{r['trg_score']}`)\n"
            f"Triggers: UP `{up_trg}` / DOWN `{dn_trg}`\n"
        )

        msg += (
            f"Deriv: mark `{fmt_num(deriv['mark'], 2)}` | index `{fmt_num(deriv['index'], 2)}` | "
            f"funding `{fmt_num(deriv['funding'], 4)}%` | basis `{fmt_num(deriv['basis'], 4)}%` | "
            f"OI `{fmt_num(deriv['oi'], 0)}`"
        )
        if oi_chg is not None:
            msg += f" | OIΔ `{fmt_num(oi_chg, 2)}%`"
        msg += "\n"

        # NEW: Range snapshots
        r4 = r.get("range_4h_24h", {})
        r1 = r.get("range_1h_6h", {})
        msg += (
            f"RANGE: 4h(24h) low `{fmt_num(r4.get('low'),2)}` / high `{fmt_num(r4.get('high'),2)}`  |  "
            f"1h(6h) low `{fmt_num(r1.get('low'),2)}` / high `{fmt_num(r1.get('high'),2)}`\n"
        )

        # NEW: Delta meta (15m)
        t15 = r["tf"]["15m"]
        if t15.get("vol_buy") is not None:
            msg += (
                f"DELTA(15m): buy `{fmt_num(t15['vol_buy'],2)}` | sell `{fmt_num(t15['vol_sell'],2)}` | "
                f"Δ `{fmt_num(t15['vol_delta'],2)}`\n"
            )

        # NEW: Alert meta
        meta = r.get("alert_meta", {})
        if meta and meta.get("first_or_repeat"):
            msg += f"ALERT META: {meta['first_or_repeat']} (`{meta['repeat_n']}`)\n"

        for tf in ["4h", "1h", "15m"]:
            t = r["tf"][tf]
            lbl = r["trend_labels"][tf]
            msg += (
                f"`{tf:>3}` [{lbl}] "
                f"p `{fmt_num(t['price'], 2)}` | "
                f"EMA20 `{fmt_num(t['ema20'], 2)}`({fmt_num(t['ema20_slp'], 2)}) "
                f"EMA50 `{fmt_num(t['ema50'], 2)}`({fmt_num(t['ema50_slp'], 2)}) | "
                f"RSI `{fmt_num(t['rsi'], 1)}`({fmt_num(t['rsi_slp'], 1)}) | "
                f"ATR `{fmt_num(t['atr'], 2)}` | "
                f"VOLx `{fmt_num(t['vol_ratio'], 2)}` | "
                f"ret `{fmt_num(r['returns'][tf], 2)}%`\n"
            )

        msg += "Why flagged:\n"
        for line in r["reasons"][:16]:
            msg += f"• {line}\n"
        msg += "\n"

    return msg.strip()

def build_context_compact(reports, title="MARKET CONTEXT (compact)"):
    ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
    msg = f"🧭 *{title}*\nUTC: `{ts}`\n_(Compact context — no signal.)_\n\n"

    for r in reports:
        sym = r["symbol"]
        alert = r["alert"]
        direction = r["direction"]
        deriv = r["deriv"]
        oi_chg = r["oi_change_pct"]

        t1h = r["tf"]["1h"]
        t15 = r["tf"]["15m"]

        msg += f"*{sym}* | ALERT `{alert}` | DIR `{direction}`\n"
        msg += (
            f"Deriv: mark `{fmt_num(deriv['mark'],2)}` | funding `{fmt_num(deriv['funding'],4)}%` | "
            f"basis `{fmt_num(deriv['basis'],4)}%` | OI `{fmt_num(deriv['oi'],0)}`"
        )
        if oi_chg is not None:
            msg += f" | OIΔ `{fmt_num(oi_chg,2)}%`"
        msg += "\n"

        # Range snapshots (compact)
        r4 = r.get("range_4h_24h", {})
        r1 = r.get("range_1h_6h", {})
        msg += (
            f"RANGE: 4h(24h) `{fmt_num(r4.get('low'),2)}`-`{fmt_num(r4.get('high'),2)}` | "
            f"1h(6h) `{fmt_num(r1.get('low'),2)}`-`{fmt_num(r1.get('high'),2)}`\n"
        )

        # Delta (compact)
        if t15.get("vol_buy") is not None:
            msg += f"DELTA(15m): buy `{fmt_num(t15['vol_buy'],2)}` | sell `{fmt_num(t15['vol_sell'],2)}` | Δ `{fmt_num(t15['vol_delta'],2)}`\n"

        msg += (
            f"`1h` p `{fmt_num(t1h['price'],2)}` | EMA20 `{fmt_num(t1h['ema20'],2)}` | EMA50 `{fmt_num(t1h['ema50'],2)}` | "
            f"RSI `{fmt_num(t1h['rsi'],1)}`({fmt_num(t1h['rsi_slp'],1)}) | VOLx `{fmt_num(t1h['vol_ratio'],2)}` | ret `{fmt_num(r['returns']['1h'],2)}%`\n"
        )
        msg += (
            f"`15m` p `{fmt_num(t15['price'],2)}` | EMA20 `{fmt_num(t15['ema20'],2)}` | EMA50 `{fmt_num(t15['ema50'],2)}` | "
            f"RSI `{fmt_num(t15['rsi'],1)}`({fmt_num(t15['rsi_slp'],1)}) | VOLx `{fmt_num(t15['vol_ratio'],2)}` | ret `{fmt_num(r['returns']['15m'],2)}%`\n"
        )
        msg += "\n"

    return msg.strip()

# ---------- MANUAL COMMAND HANDLER ---------- #

def handle_telegram_commands():
    global _last_update_id
    if not TELEGRAM_TOKEN:
        return

    updates = telegram_get_updates(offset=_last_update_id + 1 if _last_update_id else None, timeout=0)
    if not updates:
        return

    for u in updates:
        _last_update_id = max(_last_update_id, u.get("update_id", 0))
        msg = u.get("message") or {}
        chat = msg.get("chat") or {}
        chat_id = str(chat.get("id", ""))
        text = (msg.get("text") or "").strip()
        if not text:
            continue
        if TELEGRAM_CHAT_ID and not is_allowed_chat(chat_id):
            continue

        if text.startswith("/status"):
            send_telegram(
                "✅ Bot is running.\n"
                "v4.6 FINAL: range snapshot + delta(15m) buy/sell + alert meta.\n"
                "Manual: /report or /report ETHUSDT",
                chat_id=chat_id
            )
            continue

        if text.startswith("/report"):
            parts = text.split()
            send_telegram("🟡 Building report…", chat_id=chat_id)
            try:
                if len(parts) == 1:
                    reps = [analyze_symbol(s) for s in SYMBOLS]
                    send_telegram(build_message(reps, title="MANUAL REPORT (ALL)"), chat_id=chat_id)
                else:
                    sym = parts[1].upper()
                    if sym not in SYMBOLS:
                        send_telegram(f"❗Unknown symbol: `{sym}`\nAllowed: {', '.join(SYMBOLS)}", chat_id=chat_id)
                        continue
                    r = analyze_symbol(sym)
                    send_telegram(build_message([r], title=f"MANUAL REPORT ({sym})"), chat_id=chat_id)
            except Exception as e:
                send_telegram(f"❌ Manual report error:\n`{e}`", chat_id=chat_id)
            continue

# ---------- MAIN SCAN (AUTO PUSH) ---------- #

def run_once():
    reports = []
    for s in SYMBOLS:
        try:
            reports.append(analyze_symbol(s))
        except Exception as e:
            send_telegram(f"⚠️ *Symbol error* `{s}`\n`{e}`")
            continue

    triggered = [r for r in reports if r["alert"] == 3 and allowed_to_send(r)]
    if not triggered:
        return

    for r in triggered:
        send_telegram(build_message([r], title=f"🚨 ALERT 3 (FULL) — {r['symbol']}"))

    trig_set = {x["symbol"] for x in triggered}
    others = [r for r in reports if r["symbol"] not in trig_set]
    if others:
        send_telegram(build_context_compact(others, title="MARKET CONTEXT (compact)"))

    for r in triggered:
        mark_sent(r)

# ---------- RUNNER ---------- #

if __name__ == "__main__":
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        telegram_delete_webhook()
        send_telegram(
            "✅ Momentum Filter bot is ONLINE.\n"
            "v4.6 FINAL: incremental klines + rate limiter/backoff + VOLx hard filter.\n"
            "Additions: range snapshot + delta(15m) + alert meta.\n"
            "Auto: ONLY ALERT 3.\n"
            "Manual: /report or /report ETHUSDT.\n"
            "Scan runs every 5 min."
        )

    last_scan = 0
    while True:
        try:
            handle_telegram_commands()
            now = time.time()
            if now - last_scan >= SCAN_SECONDS:
                run_once()
                last_scan = now
        except Exception as e:
            if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
                send_telegram(f"❌ *Runtime error*\n`{e}`")
        time.sleep(CMD_POLL_SECONDS)
