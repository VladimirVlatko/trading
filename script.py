"""
MOMENTUM PRO — EXECUTION-AWARE MULTI-ALERT + DETAILED REPORT (Telegram Enabled, SAFE)
v8.2.0 (QUALITY WATCH=PULLBACK + FAST ENTRY_OK + CONSISTENT 5m HINT + PULLBACK IMPULSE FIX)

What changed vs v8.1.0 (your last paste):
1) WATCH mode = PULLBACK (quality, rarer) + ENTRY_OK (fast reaction)
   - AUTO_PUSH_LEVELS = {"PULLBACK","ENTRY_OK"}
2) IMPORTANT FIX: PULLBACK needs an "impulse" memory.
   - Previously _last_impulse was set only when SCOUT/CONFIRM was SENT.
   - Now we update impulse from detected SCOUT/CONFIRM even if we don't push them.
3) ENTRY_OK more practical (still not spammy):
   - ENTRY15M_DIST_EMA20_MAX_ATR = 1.25 (from 1.10)
   - ENTRY5M_VOLX_MIN = 0.90 (from 1.00)
   - COOLDOWN_ENTRY = 0 (ENTRY_OK is already gated 1-per-5m-candle via ONE_PER_CANDLE + 5m open_time)
4) ENTRY_HINT(5m) made consistent with ENTRY_CONFIRM:
   - If reclaim is FALSE (price not on correct side of 5m EMA20), HINT cannot be "OK".

NOTES
- Scanner only — NOT a trade signal.
- Binance Futures public endpoints only.
- Telegram getUpdates long-polling (stable).
"""

import os
import time
import math
import requests
import numpy as np
from datetime import datetime, UTC
from collections import deque

# ================== CONFIG ================== #

BINANCE_FUTURES = "https://fapi.binance.com"

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
TIMEFRAMES = {"15m": "15m", "1h": "1h", "4h": "4h"}  # kept for your report

EMA_FAST = 20
EMA_SLOW = 50
RSI_LEN = 14
ATR_LEN = 14

VOL_LOOKBACK = 20
SLOPE_LOOKBACK = 6
DELTA_MED_LOOKBACK = 20
SWING_LOOKBACK_15M = 5  # structure lookback (kept)

RETURN_LOOKBACK_15M = 6
RETURN_LOOKBACK_1H = 6
RETURN_LOOKBACK_4H = 6

# --- QUIET FILTER (keep) ---
QUIET_RSI_1H_LONG_ON = 70.0
QUIET_RSI_1H_SHORT_ON = 30.0
QUIET_ATR_DIST_15M_ON = 1.20

# --- base alerts thresholds (tuned, kept) ---
VOLX_SCOUT = 0.85
VOLX_PULLBACK = 0.95
VOLX_CONFIRM = 1.10

RSI_SCOUT_LONG = 52
RSI_SCOUT_SHORT = 48
RSI_CONFIRM_LONG = 58
RSI_CONFIRM_SHORT = 42

# Delta / intent (kept)
DELTA_DOM_MULT = 1.15
BODY_ATR_MIN = 0.35

# Pullback logic (kept)
PULLBACK_TOUCH_ATR = 0.30
PULLBACK_LOOKBACK_SEC = 2 * 60 * 60

# Late filter for confirm (kept)
LATE_ATR_BLOCK_CONFIRM = 1.60
SOFT_LATE_ATR_CONFIRM = 1.10

# Scan / command loop
SCAN_SECONDS = 60

# Anti-spam
# - WATCH=PULLBACK should be rare; keep cooldown moderate.
# - ENTRY_OK is fast; gated 1-per-5m candle so cooldown can be 0.
COOLDOWN_SCOUT = 12 * 60
COOLDOWN_PULLBACK = 25 * 60
COOLDOWN_CONFIRM = 60 * 60
COOLDOWN_ENTRY = 0
ONE_PER_CANDLE = True

# Minimum move required vs last same alert (prevents repeats)
MIN_MOVE_SCOUT = {"BTCUSDT": 0.10, "ETHUSDT": 0.14, "SOLUSDT": 0.18}
MIN_MOVE_PULLBACK = {"BTCUSDT": 0.20, "ETHUSDT": 0.25, "SOLUSDT": 0.30}
MIN_MOVE_CONFIRM = {"BTCUSDT": 0.25, "ETHUSDT": 0.30, "SOLUSDT": 0.40}
MIN_MOVE_ENTRY = {"BTCUSDT": 0.10, "ETHUSDT": 0.12, "SOLUSDT": 0.16}

# Telegram (strip to avoid hidden newline/space bugs)
TELEGRAM_TOKEN = (os.getenv("TELEGRAM_TOKEN") or "").strip()
TELEGRAM_CHAT_ID = (os.getenv("TELEGRAM_CHAT_ID") or "").strip()

# ---- rate protection ----
REQUEST_LOG = deque(maxlen=600)
MAX_REQ_PER_MIN = 90
BACKOFF_BASE = 1.5
BACKOFF_MAX = 30.0

# ---- kline cache ----
KLINE_SEED_LIMIT = 220
KLINE_MIN_BARS = 120
KLINE_TAIL_FETCH = 2

# ---- 5m (for entry confirmation + hint) ----
TF5_ENABLED = True
TF5_LIMIT = 220  # enough history for indicators

# ---- entry confirmation (execution-aware) ----
ENTRY5M_VOLX_MIN = 0.90       # was 1.00
ENTRY5M_RSI_EFF_MIN = 50.0
ENTRY15M_DIST_EMA20_MAX_ATR = 1.25  # was 1.10
ENTRY5M_DIST_EMA20_MAX_ATR = 0.70
ENTRY_ALLOW_CONTINUATION = True

# ---- leverage scoring ----
LEV_SCORE_4X_MIN = 8
LEV_SCORE_3X_MIN = 6

# ---- SL/TP suggestion ----
SWING_LOOKBACK_5M = 36
SL_ATR_BUFFER_5M = 0.10
TP_R1 = 1.0
TP_R2 = 2.0

# ---- auto-push levels ----
# QUALITY + RARE watch (PULLBACK), FAST reaction (ENTRY_OK)
AUTO_PUSH_LEVELS = {"PULLBACK", "ENTRY_OK"}

# ---- report mode ----
REPORT_SINGLE_MESSAGE = True
TELEGRAM_MAX_LEN = 3900

# ================== STATE ================== #

_last_update_id = 0
_prev_oi = {}

_last_sent = {}                   # (symbol, side, level) -> {"ts":..., "price":..., "candle_ot":...}
_last_candle_by_key = {}          # (symbol, side, level) -> open_time (LAST CLOSED candle OT)
_last_alert_price_by_key = {}     # (symbol, side, level) -> price

# IMPORTANT: pullback depends on impulse memory.
# We'll update this from detected SCOUT/CONFIRM even if we don't push them.
_last_impulse = {}                # (symbol, side) -> {"ts":..., "level":..., "price":..., "candle_ot":...}

_KLINE_CACHE = {}                 # (symbol, tf) -> np.array
_DERIV_CACHE = {}                 # symbol -> (ts, dict)
DERIV_TTL_SEC = 30

# ================== RATE LIMIT HELPERS ================== #

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

# ================== NETWORK HELPERS ================== #

def http_get_json(url, params=None, timeout=10):
    for attempt in range(6):
        _rate_guard()
        try:
            r = requests.get(url, params=params, timeout=timeout)
            try:
                data = r.json()
            except Exception:
                data = None

            if data is None:
                _sleep_backoff(attempt)
                continue

            if isinstance(data, dict) and "code" in data and data.get("code") not in (0, None):
                if _should_backoff(r, data):
                    _sleep_backoff(attempt)
                    continue
                raise Exception(f"Binance error {data.get('code')}: {data.get('msg')}")

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
            if attempt == 5:
                raise
            _sleep_backoff(attempt)

# ================== TELEGRAM ================== #

def is_allowed_chat(chat_id: str) -> bool:
    if not TELEGRAM_CHAT_ID:
        return True
    return str(chat_id).strip() == str(TELEGRAM_CHAT_ID).strip()

def _send_telegram_once(text: str, chat_id: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": str(chat_id),
        "text": text,
        "disable_web_page_preview": True
    }
    resp = requests.post(url, json=payload, timeout=15)
    if resp.status_code >= 400:
        raise Exception(f"Telegram error {resp.status_code}: {resp.text[:600]}")

def send_telegram(message: str, chat_id: str | None = None):
    """
    Default sender: splits long messages (NO truncation).
    Used for /reportfull and alerts.
    """
    if not TELEGRAM_TOKEN:
        return

    target = (str(chat_id).strip() if chat_id is not None else TELEGRAM_CHAT_ID)
    if not target or not message:
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    MAX_LEN = TELEGRAM_MAX_LEN

    lines = message.split("\n")
    chunks = []
    current = ""

    for line in lines:
        add = line + "\n"
        if len(current) + len(add) <= MAX_LEN:
            current += add
        else:
            if current.strip():
                chunks.append(current.rstrip())
            if len(add) > MAX_LEN:
                s = add
                while len(s) > MAX_LEN:
                    chunks.append(s[:MAX_LEN])
                    s = s[MAX_LEN:]
                current = s
            else:
                current = add

    if current.strip():
        chunks.append(current.rstrip())

    last_err = None
    for part in chunks:
        payload = {"chat_id": str(target), "text": part, "disable_web_page_preview": True}
        resp = requests.post(url, json=payload, timeout=15)
        if resp.status_code >= 400:
            last_err = f"Telegram error {resp.status_code}: {resp.text[:500]}"
            break

    if last_err:
        raise Exception(last_err)

def send_telegram_single(message: str, chat_id: str | None = None):
    """
    Sends EXACTLY one Telegram message.
    If too long, caller compresses first.
    """
    if not TELEGRAM_TOKEN:
        return
    target = (str(chat_id).strip() if chat_id is not None else TELEGRAM_CHAT_ID)
    if not target or not message:
        return
    if len(message) > TELEGRAM_MAX_LEN:
        raise Exception(f"Single-message report too long ({len(message)}>{TELEGRAM_MAX_LEN}).")
    _send_telegram_once(message, target)

def telegram_delete_webhook():
    if not TELEGRAM_TOKEN:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/deleteWebhook"
    http_get_json(url, params={"drop_pending_updates": True}, timeout=10)

def telegram_get_updates(offset=None, timeout=20):
    if not TELEGRAM_TOKEN:
        return []
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
    params = {"timeout": timeout}
    if offset is not None:
        params["offset"] = offset
    data = http_get_json(url, params=params, timeout=timeout + 5)
    return data.get("result", []) if isinstance(data, dict) else []

# ================== INDICATORS ================== #

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

def atr_distance(price, ema20, atr):
    if atr is None or (isinstance(atr, float) and np.isnan(atr)) or atr <= 0:
        return 0.0
    return abs(price - ema20) / atr

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

# ================== BINANCE DATA (INCREMENTAL) ================== #

def _fetch_klines_raw(symbol, interval, limit=200):
    url = f"{BINANCE_FUTURES}/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    data = http_get_json(url, params=params, timeout=10)
    if not isinstance(data, list) or len(data) == 0:
        raise Exception(f"Klines invalid/empty for {symbol} {interval}: {str(data)[:200]}")
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

    if len(k) > 420:
        k = k[-360:]

    _KLINE_CACHE[key] = k
    return k

def fetch_derivatives_cached(symbol):
    now = time.time()
    hit = _DERIV_CACHE.get(symbol)
    if hit and (now - hit[0] < DERIV_TTL_SEC):
        return hit[1]

    mark = http_get_json(f"{BINANCE_FUTURES}/fapi/v1/premiumIndex", params={"symbol": symbol}, timeout=10)
    oi = http_get_json(f"{BINANCE_FUTURES}/fapi/v1/openInterest", params={"symbol": symbol}, timeout=10)

    if not isinstance(mark, dict) or not isinstance(oi, dict):
        raise Exception(f"Deriv invalid for {symbol}: mark={type(mark)} oi={type(oi)}")

    for k in ("markPrice", "indexPrice", "lastFundingRate"):
        if k not in mark:
            raise Exception(f"premiumIndex missing {k} for {symbol}: {str(mark)[:200]}")
    if "openInterest" not in oi:
        raise Exception(f"openInterest missing openInterest for {symbol}: {str(oi)[:200]}")

    mark_price = float(mark["markPrice"])
    index_price = float(mark["indexPrice"])
    funding_pct = float(mark["lastFundingRate"]) * 100.0
    basis_pct = (mark_price - index_price) / (index_price + 1e-12) * 100.0
    oi_val = float(oi["openInterest"])

    d = {"mark": mark_price, "index": index_price, "funding": funding_pct, "basis": basis_pct, "oi": oi_val}
    _DERIV_CACHE[symbol] = (now, d)
    return d

# ================== SNAPSHOT (LAST CLOSED CANDLE) ================== #

def tf_snapshot_closed(k, tf_name):
    """
    Evaluate on LAST CLOSED candle (drop live forming candle).
    """
    if k is None or len(k) < 5:
        return None

    kc = k[:-1]
    if len(kc) < 5:
        return None

    op = kc[:, 1]
    close = kc[:, 4]
    high = kc[:, 2]
    low = kc[:, 3]
    vol = kc[:, 5]

    e20 = ema(close, EMA_FAST)
    e50 = ema(close, EMA_SLOW)
    rsi_v = rsi_wilder(close, RSI_LEN)
    atr_v = atr_wilder(high, low, close, ATR_LEN)

    sl = SLOPE_LOOKBACK
    e20_slp = e20[-1] - e20[-1 - sl] if len(e20) > sl else e20[-1] - e20[0]
    e50_slp = e50[-1] - e50[-1 - sl] if len(e50) > sl else e50[-1] - e50[0]
    rsi_slp = rsi_v[-1] - rsi_v[-1 - sl] if len(rsi_v) > sl else 0.0

    vol_ratio = vol[-1] / (np.mean(vol[-VOL_LOOKBACK:]) + 1e-12) if len(vol) >= VOL_LOOKBACK else 1.0
    open_time = int(kc[-1, 0])

    # Delta series (taker buy vol exists at index 9)
    if kc.shape[1] > 9:
        taker_buy_series = kc[:, 9].astype(float)
        total_vol_series = vol.astype(float)
        taker_sell_series = total_vol_series - taker_buy_series
        delta_series = taker_buy_series - taker_sell_series
        taker_buy = float(taker_buy_series[-1])
        total_vol = float(total_vol_series[-1])
        taker_sell = float(taker_sell_series[-1])
        delta_now = float(delta_series[-1])
    else:
        delta_series = None
        taker_buy = None
        total_vol = float(vol[-1])
        taker_sell = None
        delta_now = None

    body = float(close[-1] - op[-1])

    return {
        "tf": tf_name,
        "open_time": open_time,
        "open": float(op[-1]),
        "high": float(high[-1]),
        "low": float(low[-1]),
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
        "vol_buy": float(taker_buy) if taker_buy is not None else None,
        "vol_sell": float(taker_sell) if taker_sell is not None else None,
        "vol_delta": float(delta_now) if delta_now is not None else None,
        "delta_series": delta_series,
        "body": body,
        "close_series": close,
        "high_series": high,
        "low_series": low,
        "vol_series": vol,
    }

def compute_returns(tf_snaps):
    out = {}
    for tf, snap in tf_snaps.items():
        close = snap["close_series"]
        lb = RETURN_LOOKBACK_15M if tf == "15m" else RETURN_LOOKBACK_1H if tf == "1h" else RETURN_LOOKBACK_4H
        out[tf] = pct_change(close[-1], close[-1 - lb]) if len(close) > lb else 0.0
    return out

# ================== STRUCTURE / DELTA / INTENT ================== #

def _structure_flags_15m(tf15, side):
    close = tf15["close_series"]
    high = tf15["high_series"]
    low = tf15["low_series"]
    e20 = tf15["ema20"]

    if len(close) < SWING_LOOKBACK_15M + 3:
        return {"break": False, "reclaim": False, "level": None}

    prev_close = float(close[-2])
    now_close = float(close[-1])

    swing_hi = float(np.max(high[-(SWING_LOOKBACK_15M + 1):-1]))
    swing_lo = float(np.min(low[-(SWING_LOOKBACK_15M + 1):-1]))

    if side == "LONG":
        breakout = now_close > swing_hi
        reclaim = (prev_close <= e20 and now_close > e20)
        return {"break": breakout, "reclaim": reclaim, "level": swing_hi}
    else:
        breakdown = now_close < swing_lo
        reject = (prev_close >= e20 and now_close < e20)
        return {"break": breakdown, "reclaim": reject, "level": swing_lo}

def _delta_dominance(tf15):
    ds = tf15.get("delta_series")
    dn = tf15.get("vol_delta")
    if ds is None or dn is None:
        return False, "Δ n/a"
    tail = ds[-DELTA_MED_LOOKBACK:] if len(ds) >= DELTA_MED_LOOKBACK else ds
    med = float(np.median(np.abs(tail))) + 1e-12
    ok = abs(float(dn)) >= (DELTA_DOM_MULT * med)
    return ok, f"Δdom {abs(float(dn)):.0f} >= {DELTA_DOM_MULT:.2f}*med({med:.0f})"

def _intent_candle(tf15):
    atr = tf15["atr"]
    body = tf15.get("body", 0.0)
    if atr is None or (isinstance(atr, float) and np.isnan(atr)) or atr <= 0:
        return False
    return abs(float(body)) >= (BODY_ATR_MIN * float(atr))

# ================== 5m ENTRY HINT + ENTRY CONFIRM ================== #

def get_tf5_closed(symbol: str):
    if not TF5_ENABLED:
        return None
    try:
        k5 = _fetch_klines_raw(symbol, "5m", limit=TF5_LIMIT)
        return tf_snapshot_closed(k5, "5m")
    except Exception:
        return None

def rsi_eff(side: str, rsi_val: float) -> float:
    return rsi_val if side == "LONG" else (100.0 - rsi_val)

def entry_hint_5m(snap5, side: str):
    if snap5 is None:
        return "N/A", "5m n/a"

    p = float(snap5["price"])
    e20 = float(snap5["ema20"])
    atr = float(snap5["atr"]) if snap5["atr"] is not None else float("nan")
    volx = float(snap5["vol_ratio"])
    rsi = float(snap5["rsi"])

    reff = rsi_eff(side, rsi)
    dist = atr_distance(p, e20, atr)

    # CONSISTENCY FIX: if reclaim is false, do NOT label "OK"
    reclaim_ok = (p > e20) if side == "LONG" else (p < e20)
    if not reclaim_ok:
        hint = "PULLBACK_ZONE"
        details = f"5m reclaim=FALSE rsiEff={reff:.1f} distATR={dist:.2f} volx={volx:.2f}"
        return hint, details

    if reff >= 78 or (dist >= 2.0 and volx >= 2.0):
        hint = "DANGER"
    elif (reff <= 65) and (dist <= 1.10) and (volx <= 1.40):
        hint = "OK"
    elif (reff >= 70) or (dist >= 1.50) or (volx >= 2.0):
        hint = "EXTENDED"
    else:
        hint = "PULLBACK_ZONE"

    details = f"5m reclaim=TRUE rsiEff={reff:.1f} distATR={dist:.2f} volx={volx:.2f}"
    return hint, details

def entry_confirm_5m(snap5, side: str):
    if snap5 is None:
        return False, "5m n/a"

    close = snap5["close_series"]
    high = snap5["high_series"]
    low = snap5["low_series"]
    vol = snap5["vol_series"]

    if len(close) < 3:
        return False, "5m too short"

    p = float(snap5["price"])
    e20 = float(snap5["ema20"])
    atr = float(snap5["atr"]) if snap5["atr"] is not None else float("nan")
    vma = float(np.mean(vol[-VOL_LOOKBACK:])) if len(vol) >= VOL_LOOKBACK else float(np.mean(vol))
    volx = float(vol[-1] / (vma + 1e-12))
    rsi5 = float(snap5["rsi"])
    reff = rsi_eff(side, rsi5)
    dist = atr_distance(p, e20, atr)

    if side == "LONG":
        reclaim = p > e20
        hh = float(high[-1]) > float(high[-2])
        ok = reclaim and hh and (volx >= ENTRY5M_VOLX_MIN) and (reff >= ENTRY5M_RSI_EFF_MIN) and (dist <= ENTRY5M_DIST_EMA20_MAX_ATR)
        reason = f"reclaim={reclaim}, HH={hh}, volx={volx:.2f}, rsiEff={reff:.1f}, distATR={dist:.2f}"
        return ok, reason
    else:
        reclaim = p < e20
        ll = float(low[-1]) < float(low[-2])
        ok = reclaim and ll and (volx >= ENTRY5M_VOLX_MIN) and (reff >= ENTRY5M_RSI_EFF_MIN) and (dist <= ENTRY5M_DIST_EMA20_MAX_ATR)
        reason = f"reclaim={reclaim}, LL={ll}, volx={volx:.2f}, rsiEff={reff:.1f}, distATR={dist:.2f}"
        return ok, reason

# ================== CONTEXT SCORING (kept) ================== #

def _ctx_score(tf_snap, side):
    p = tf_snap["price"]
    e20 = tf_snap["ema20"]
    e50 = tf_snap["ema50"]
    s20 = tf_snap["ema20_slp"]
    s50 = tf_snap["ema50_slp"]
    r = tf_snap["rsi"]

    reasons = []
    score = 0
    up = 0
    down = 0

    if p > e20 and p > e50:
        score += 2; up += 2; reasons.append("HTF price above EMA20/50")
    elif p < e20 and p < e50:
        score += 2; down += 2; reasons.append("HTF price below EMA20/50")
    else:
        score += 1; reasons.append("HTF price mixed vs EMA20/50")

    if e20 > e50:
        score += 1; up += 1; reasons.append("HTF EMA20 > EMA50")
    else:
        score += 1; down += 1; reasons.append("HTF EMA20 < EMA50")

    if s20 > 0 and s50 > 0:
        score += 1; up += 1; reasons.append("HTF EMA slopes positive")
    elif s20 < 0 and s50 < 0:
        score += 1; down += 1; reasons.append("HTF EMA slopes negative")
    else:
        reasons.append("HTF EMA slopes mixed")

    if side == "LONG":
        if r >= 52:
            score += 1; up += 1; reasons.append("HTF RSI supportive")
        else:
            reasons.append("HTF RSI weak")
    else:
        if r <= 48:
            score += 1; down += 1; reasons.append("HTF RSI supportive")
        else:
            reasons.append("HTF RSI weak")

    return score, reasons, up, down

def _pick_best_side_for_report(rep):
    tf1 = rep["tf_raw"]["1h"]
    tf4 = rep["tf_raw"]["4h"]

    s1L, _, upL1, dnL1 = _ctx_score(tf1, "LONG")
    s4L, _, upL4, dnL4 = _ctx_score(tf4, "LONG")
    s1S, _, upS1, dnS1 = _ctx_score(tf1, "SHORT")
    s4S, _, upS4, dnS4 = _ctx_score(tf4, "SHORT")

    totalL = s1L + s4L + (upL1 + upL4) - (dnL1 + dnL4)
    totalS = s1S + s4S + (dnS1 + dnS4) - (upS1 + upS4)

    return "LONG" if totalL >= totalS else "SHORT"

# ================== SCOUT / PULLBACK / CONFIRM (kept) ================== #

def _scout_signal(tf15, tf1h, side, ret15):
    reasons = []
    price = tf15["price"]
    e20 = tf15["ema20"]
    volx = tf15["vol_ratio"]
    rsi1h = tf1h["rsi"]

    st = _structure_flags_15m(tf15, side)
    d_ok, d_msg = _delta_dominance(tf15)
    intent = _intent_candle(tf15)

    if volx < VOLX_SCOUT:
        return None

    if side == "LONG":
        if rsi1h < RSI_SCOUT_LONG:
            return None
        if not (price > e20 or ret15 >= 0.25):
            return None
    else:
        if rsi1h > RSI_SCOUT_SHORT:
            return None
        if not (price < e20 or ret15 <= -0.25):
            return None

    reasons.append(f"VOLx>=SCOUT ({volx:.2f}x)")
    reasons.append(f"RSI(1h) ok ({rsi1h:.1f})")
    reasons.append(f"ret15 ({ret15:+.2f}%)")

    if st["break"]:
        reasons.append(f"STRUCT break ({st['level']:.2f})")
    elif st["reclaim"]:
        reasons.append("STRUCT reclaim/reject EMA20")
    else:
        reasons.append("STRUCT none (allowed for SCOUT)")

    if d_ok:
        reasons.append(f"DELTA ok ({d_msg})")
    if intent:
        reasons.append("Intent candle")

    return {"level": "SCOUT", "side": side, "reasons": reasons}

def _pullback_signal(tf15, symbol, side):
    key_imp = (symbol, side)
    imp = _last_impulse.get(key_imp)
    if not imp or (time.time() - imp["ts"] > PULLBACK_LOOKBACK_SEC):
        return None

    price = tf15["price"]
    e20 = tf15["ema20"]
    atr = tf15["atr"]
    volx = tf15["vol_ratio"]

    if volx < VOLX_PULLBACK:
        return None

    dist = atr_distance(price, e20, atr)
    if dist > PULLBACK_TOUCH_ATR:
        return None

    op = tf15["open"]
    cl = tf15["price"]

    if side == "LONG":
        if not (cl > op and cl >= e20):
            return None
    else:
        if not (cl < op and cl <= e20):
            return None

    d_ok, d_msg = _delta_dominance(tf15)
    intent = _intent_candle(tf15)

    reasons = [
        f"Recent impulse: {imp['level']} ({int((time.time()-imp['ts'])/60)}m ago)",
        f"Touch EMA20 zone: dist {dist:.2f} ATR <= {PULLBACK_TOUCH_ATR:.2f}",
        f"Reject candle ok (side={side})",
        f"VOLx>=PULLBACK ({volx:.2f}x)",
    ]
    if d_ok:
        reasons.append(f"DELTA ok ({d_msg})")
    if intent:
        reasons.append("Intent candle")

    return {"level": "PULLBACK", "side": side, "reasons": reasons}

def _confirm_signal(tf15, tf1h, tf4h, side, ret15):
    reasons = []
    price = tf15["price"]
    e20 = tf15["ema20"]
    atr = tf15["atr"]
    volx = tf15["vol_ratio"]
    rsi1h = tf1h["rsi"]

    if volx < VOLX_CONFIRM:
        return None

    dist = atr_distance(price, e20, atr)
    if dist >= LATE_ATR_BLOCK_CONFIRM:
        return None

    st = _structure_flags_15m(tf15, side)
    if dist >= SOFT_LATE_ATR_CONFIRM and not st["break"]:
        return None

    if side == "LONG":
        if rsi1h < RSI_CONFIRM_LONG:
            return None
        if not (tf1h["ema20"] > tf1h["ema50"] or tf4h["ema20"] > tf4h["ema50"]):
            return None
    else:
        if rsi1h > RSI_CONFIRM_SHORT:
            return None
        if not (tf1h["ema20"] < tf1h["ema50"] or tf4h["ema20"] < tf4h["ema50"]):
            return None

    d_ok, d_msg = _delta_dominance(tf15)
    intent = _intent_candle(tf15)

    reasons.append(f"VOLx>=CONFIRM ({volx:.2f}x)")
    reasons.append(f"RSI(1h) confirm ({rsi1h:.1f})")
    reasons.append(f"distEMA20 {dist:.2f} ATR (<= {LATE_ATR_BLOCK_CONFIRM:.2f})")
    reasons.append(f"ret15 ({ret15:+.2f}%)")

    if st["break"]:
        reasons.append(f"STRUCT break ({st['level']:.2f})")
    elif st["reclaim"]:
        reasons.append("STRUCT reclaim/reject EMA20")
    else:
        reasons.append("STRUCT none")

    if d_ok:
        reasons.append(f"DELTA ok ({d_msg})")
    if intent:
        reasons.append("Intent candle")

    return {"level": "CONFIRM", "side": side, "reasons": reasons}

# ================== QUIET FILTER (kept) ================== #

def quiet_mode_blocks(signal, tf15, tf1h):
    level = signal.get("level")
    if level not in ("SCOUT", "CONFIRM"):
        return False

    side = signal.get("side")
    rsi1h = float(tf1h["rsi"])
    dist = atr_distance(float(tf15["price"]), float(tf15["ema20"]), float(tf15["atr"]))

    if level == "CONFIRM":
        st = _structure_flags_15m(tf15, side)
        if st.get("break") or st.get("reclaim"):
            return False

    if side == "LONG":
        return (rsi1h >= QUIET_RSI_1H_LONG_ON) and (dist >= QUIET_ATR_DIST_15M_ON)
    else:
        return (rsi1h <= QUIET_RSI_1H_SHORT_ON) and (dist >= QUIET_ATR_DIST_15M_ON)

# ================== EXECUTION-AWARE ENTRY_OK ================== #

def bias_ok(rep, side: str) -> bool:
    tf1 = rep["tf_raw"]["1h"]
    tf4 = rep["tf_raw"]["4h"]

    if side == "LONG":
        htf_ok = (tf4["price"] > tf4["ema20"] > tf4["ema50"]) and (tf4["ema20_slp"] > 0)
        ltf_ok = (tf1["price"] > tf1["ema20"]) and (tf1["ema20_slp"] > 0)
        return htf_ok and ltf_ok
    else:
        htf_ok = (tf4["price"] < tf4["ema20"] < tf4["ema50"]) and (tf4["ema20_slp"] < 0)
        ltf_ok = (tf1["price"] < tf1["ema20"]) and (tf1["ema20_slp"] < 0)
        return htf_ok and ltf_ok

def in_pullback_zone_15m(tf15, side: str) -> tuple[bool, float]:
    p = float(tf15["price"])
    e20 = float(tf15["ema20"])
    a = float(tf15["atr"])
    dist = atr_distance(p, e20, a)
    ok = dist <= 0.85
    return ok, dist

def entry_ok(rep, side: str, snap5):
    tf15 = rep["tf_raw"]["15m"]
    dist15 = atr_distance(float(tf15["price"]), float(tf15["ema20"]), float(tf15["atr"]))
    if dist15 > ENTRY15M_DIST_EMA20_MAX_ATR:
        return False, f"15m late distATR={dist15:.2f}>{ENTRY15M_DIST_EMA20_MAX_ATR:.2f}"

    pb_ok, pb_dist = in_pullback_zone_15m(tf15, side)

    cont_ok = False
    if ENTRY_ALLOW_CONTINUATION and not pb_ok:
        tf1 = rep["tf_raw"]["1h"]
        tf4 = rep["tf_raw"]["4h"]
        ret15 = rep["returns"]["15m"]
        c = _confirm_signal(tf15, tf1, tf4, side, ret15)
        cont_ok = c is not None

    if not pb_ok and not cont_ok:
        return False, f"no PB (dist={pb_dist:.2f}) and no continuation-confirm"

    ok5, reason5 = entry_confirm_5m(snap5, side)
    if not ok5:
        return False, f"5m confirm FAIL ({reason5})"

    if snap5 is not None:
        dist5 = atr_distance(float(snap5["price"]), float(snap5["ema20"]), float(snap5["atr"]))
        if dist5 > ENTRY5M_DIST_EMA20_MAX_ATR:
            return False, f"5m too extended distATR={dist5:.2f}>{ENTRY5M_DIST_EMA20_MAX_ATR:.2f}"

    mode = "PULLBACK" if pb_ok else "CONTINUATION"
    return True, f"ENTRY_OK ({mode}) | 15mDist={dist15:.2f}ATR | 5m: {reason5}"

# ================== LEVERAGE + SL/TP ================== #

def leverage_score(rep, side: str, snap5):
    tf15 = rep["tf_raw"]["15m"]
    tf1 = rep["tf_raw"]["1h"]
    tf4 = rep["tf_raw"]["4h"]

    score = 0
    notes = []

    if side == "LONG":
        if tf1["price"] > tf1["ema20"] > tf1["ema50"]: score += 1; notes.append("HTF1h")
        if tf4["price"] > tf4["ema20"] > tf4["ema50"]: score += 1; notes.append("HTF4h")
        if tf1["rsi"] >= 56 and tf4["rsi"] >= 62: score += 1; notes.append("RSI_HTF")
    else:
        if tf1["price"] < tf1["ema20"] < tf1["ema50"]: score += 1; notes.append("HTF1h")
        if tf4["price"] < tf4["ema20"] < tf4["ema50"]: score += 1; notes.append("HTF4h")
        if tf1["rsi"] <= 44 and tf4["rsi"] <= 38: score += 1; notes.append("RSI_HTF")

    dist15 = atr_distance(float(tf15["price"]), float(tf15["ema20"]), float(tf15["atr"]))
    if dist15 <= 0.95:
        score += 1; notes.append("notLate15m")

    st = _structure_flags_15m(tf15, side)
    if st["break"] or st["reclaim"]:
        score += 1; notes.append("STRUCT15")

    v15x = float(tf15["vol_ratio"])
    if v15x >= 1.05:
        score += 1; notes.append("VOLx15")
    d_ok, _ = _delta_dominance(tf15)
    if d_ok:
        score += 1; notes.append("DELTA")

    if snap5 is not None:
        r5 = float(snap5["rsi"])
        reff = rsi_eff(side, r5)
        if reff >= 55: score += 1; notes.append("RSI5")
        v5x = float(snap5["vol_ratio"])
        if v5x >= 1.05: score += 1; notes.append("VOLx5")

    score = int(clamp(score, 0, 10))

    if score >= LEV_SCORE_4X_MIN:
        lev = "4x"; label = "A+"
    elif score >= LEV_SCORE_3X_MIN:
        lev = "3x"; label = "A"
    else:
        lev = "2x"; label = "B/C"

    return score, lev, f"{label} ({', '.join(notes) if notes else 'n/a'})"

def swing_low_high(series_high, series_low, lookback):
    if series_high is None or series_low is None or len(series_high) < lookback + 2:
        return None, None
    hi = float(np.max(series_high[-(lookback+1):-1]))
    lo = float(np.min(series_low[-(lookback+1):-1]))
    return hi, lo

def suggest_sl_tp(rep, side: str, entry_price: float, snap5):
    tf15 = rep["tf_raw"]["15m"]

    if snap5 is not None:
        h, l = swing_low_high(snap5["high_series"], snap5["low_series"], SWING_LOOKBACK_5M)
        a = float(snap5["atr"]) if snap5["atr"] is not None else float(tf15["atr"])
        if h is None or l is None:
            h, l = swing_low_high(tf15["high_series"], tf15["low_series"], SWING_LOOKBACK_15M)
            a = float(tf15["atr"])
        src = "5m"
    else:
        h, l = swing_low_high(tf15["high_series"], tf15["low_series"], SWING_LOOKBACK_15M)
        a = float(tf15["atr"])
        src = "15m"

    if h is None or l is None or (isinstance(a, float) and np.isnan(a)):
        return None, None, None, "SLTP n/a"

    if side == "LONG":
        sl = float(l - SL_ATR_BUFFER_5M * a)
        risk = max(1e-9, entry_price - sl)
        tp1 = float(entry_price + TP_R1 * risk)
        tp2 = float(entry_price + TP_R2 * risk)
        note = f"SL≈swingLow({src})-{SL_ATR_BUFFER_5M:.2f}ATR"
        return sl, tp1, tp2, note
    else:
        sl = float(h + SL_ATR_BUFFER_5M * a)
        risk = max(1e-9, sl - entry_price)
        tp1 = float(entry_price - TP_R1 * risk)
        tp2 = float(entry_price - TP_R2 * risk)
        note = f"SL≈swingHigh({src})+{SL_ATR_BUFFER_5M:.2f}ATR"
        return sl, tp1, tp2, note

# ================== IMPULSE UPDATE (NEW) ================== #

def _level_rank(level: str) -> int:
    return {"SCOUT": 1, "CONFIRM": 2}.get(level, 0)

def update_impulses_from_signals(symbol: str, tf15_raw, signals: list):
    """
    PULLBACK depends on _last_impulse.
    We update impulse when SCOUT/CONFIRM is DETECTED (not only when sent),
    otherwise WATCH=PULLBACK would never fire.
    """
    if not signals:
        return

    candle_ot = int(tf15_raw["open_time"])
    price = float(tf15_raw["price"])
    now = time.time()

    # for each side, keep best (CONFIRM > SCOUT)
    best = {}
    for s in signals:
        if s.get("level") not in ("SCOUT", "CONFIRM"):
            continue
        side = s.get("side")
        if side not in ("LONG", "SHORT"):
            continue
        if (side not in best) or (_level_rank(s["level"]) > _level_rank(best[side]["level"])):
            best[side] = s

    for side, sig in best.items():
        key = (symbol, side)
        prev = _last_impulse.get(key)
        # update if new candle or upgrade to stronger level
        if prev is None:
            _last_impulse[key] = {"ts": now, "level": sig["level"], "price": price, "candle_ot": candle_ot}
        else:
            prev_ot = int(prev.get("candle_ot", 0))
            prev_lvl = prev.get("level", "SCOUT")
            if candle_ot != prev_ot or _level_rank(sig["level"]) > _level_rank(prev_lvl):
                _last_impulse[key] = {"ts": now, "level": sig["level"], "price": price, "candle_ot": candle_ot}

# ================== ANALYZE ================== #

def analyze_symbol(symbol):
    tf_snaps = {}
    for tf in TIMEFRAMES.keys():
        k = get_klines_cached(symbol, tf)
        tf_snaps[tf] = tf_snapshot_closed(k, tf)
        if tf_snaps[tf] is None:
            raise Exception(f"{symbol} {tf}: snapshot n/a")

    returns = compute_returns(tf_snaps)
    ret_15m = returns["15m"]

    deriv = fetch_derivatives_cached(symbol)
    prev_oi = _prev_oi.get(symbol)
    oi_change_pct = pct_change(deriv["oi"], prev_oi) if prev_oi and prev_oi > 0 else None
    _prev_oi[symbol] = deriv["oi"]

    signals = []

    s_long = _scout_signal(tf_snaps["15m"], tf_snaps["1h"], "LONG", ret_15m)
    if s_long: signals.append(s_long)
    s_short = _scout_signal(tf_snaps["15m"], tf_snaps["1h"], "SHORT", ret_15m)
    if s_short: signals.append(s_short)

    pb_long = _pullback_signal(tf_snaps["15m"], symbol, "LONG")
    if pb_long: signals.append(pb_long)
    pb_short = _pullback_signal(tf_snaps["15m"], symbol, "SHORT")
    if pb_short: signals.append(pb_short)

    c_long = _confirm_signal(tf_snaps["15m"], tf_snaps["1h"], tf_snaps["4h"], "LONG", ret_15m)
    if c_long: signals.append(c_long)
    c_short = _confirm_signal(tf_snaps["15m"], tf_snaps["1h"], tf_snaps["4h"], "SHORT", ret_15m)
    if c_short: signals.append(c_short)

    # quiet filter blocks only scout/confirm noise
    tf15 = tf_snaps["15m"]
    tf1h = tf_snaps["1h"]
    signals = [s for s in signals if not quiet_mode_blocks(s, tf15, tf1h)]

    # IMPORTANT: update impulse memory from detected SCOUT/CONFIRM even if not pushed
    update_impulses_from_signals(symbol, tf_snaps["15m"], signals)

    tf_public = {
        k: {kk: vv for kk, vv in v.items()
            if kk not in ("close_series", "high_series", "low_series", "delta_series", "vol_series")}
        for k, v in tf_snaps.items()
    }

    snap5 = get_tf5_closed(symbol) if TF5_ENABLED else None
    tf5_public = None
    hints5 = {}
    if snap5 is not None:
        tf5_public = {kk: vv for kk, vv in snap5.items()
                      if kk not in ("close_series", "high_series", "low_series", "delta_series", "vol_series")}
        hL, dL = entry_hint_5m(snap5, "LONG")
        hS, dS = entry_hint_5m(snap5, "SHORT")
        hints5 = {"LONG": {"hint": hL, "details": dL}, "SHORT": {"hint": hS, "details": dS}}

    entry_signals = []
    for side in ("LONG", "SHORT"):
        if not bias_ok({"tf_raw": tf_snaps}, side):
            continue
        ok, why = entry_ok({"tf_raw": tf_snaps, "returns": returns}, side, snap5)
        if ok:
            score, lev, meta = leverage_score({"tf_raw": tf_snaps, "returns": returns}, side, snap5)
            entry_px = float(snap5["price"]) if snap5 is not None else float(tf_snaps["15m"]["price"])
            sl, tp1, tp2, note = suggest_sl_tp({"tf_raw": tf_snaps}, side, entry_px, snap5)
            entry_signals.append({
                "level": "ENTRY_OK",
                "side": side,
                "reasons": [why],
                "lev_score": score,
                "lev_hint": lev,
                "lev_meta": meta,
                "entry_price": entry_px,
                "sl": sl,
                "tp1": tp1,
                "tp2": tp2,
                "sltp_note": note
            })

    return {
        "symbol": symbol,
        "signals": signals,
        "entry_signals": entry_signals,
        "tf": tf_public,
        "tf_raw": tf_snaps,
        "returns": returns,
        "deriv": deriv,
        "oi_change_pct": oi_change_pct,
        "tf5m": tf5_public,
        "entry_hint_5m": hints5,
        "snap5_raw": snap5
    }

# ================== ANTI-SPAM / GATING ================== #

def _min_move_for(symbol, level):
    if level == "SCOUT":
        return MIN_MOVE_SCOUT.get(symbol, 0.12)
    if level == "PULLBACK":
        return MIN_MOVE_PULLBACK.get(symbol, 0.18)
    if level == "CONFIRM":
        return MIN_MOVE_CONFIRM.get(symbol, 0.30)
    if level == "ENTRY_OK":
        return MIN_MOVE_ENTRY.get(symbol, 0.14)
    return 0.20

def _cooldown_for(level):
    if level == "SCOUT":
        return COOLDOWN_SCOUT
    if level == "PULLBACK":
        return COOLDOWN_PULLBACK
    if level == "CONFIRM":
        return COOLDOWN_CONFIRM
    if level == "ENTRY_OK":
        return COOLDOWN_ENTRY
    return 10 * 60

def _get_gate_reference(rep, level: str):
    """
    ENTRY_OK gates on 5m (execution confirm).
    Others gate on 15m.
    """
    if level == "ENTRY_OK":
        s5 = rep.get("snap5_raw")
        if s5 is not None:
            return int(s5["open_time"]), float(s5["price"])
    return int(rep["tf"]["15m"]["open_time"]), float(rep["tf"]["15m"]["price"])

def allowed_to_send(symbol, signal, rep):
    side = signal["side"]
    level = signal["level"]
    key = (symbol, side, level)
    now = time.time()

    candle_ot, curr_price = _get_gate_reference(rep, level)

    if ONE_PER_CANDLE:
        last_candle = _last_candle_by_key.get(key)
        if last_candle is not None and candle_ot == last_candle:
            return False

    last_price = _last_alert_price_by_key.get(key)
    if last_price is not None and last_price > 0:
        move_pct = abs((curr_price - last_price) / last_price) * 100.0
        if move_pct < _min_move_for(symbol, level):
            return False

    prev = _last_sent.get(key)
    if prev is None:
        return True

    cd = _cooldown_for(level)
    return (now - prev["ts"]) >= cd

def mark_sent(symbol, signal, rep):
    side = signal["side"]
    level = signal["level"]
    key = (symbol, side, level)

    candle_ot, curr_price = _get_gate_reference(rep, level)

    _last_sent[key] = {"ts": time.time(), "price": float(curr_price), "candle_ot": int(candle_ot)}
    _last_candle_by_key[key] = candle_ot
    _last_alert_price_by_key[key] = float(curr_price)

# ================== FORMATTING ================== #

def fmt_num(x, digits=2):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "n/a"
        return f"{x:.{digits}f}"
    except Exception:
        return "n/a"

def tf_trend_label(snap):
    price, e20, e50 = snap["price"], snap["ema20"], snap["ema50"]
    if price > e20 and e20 > e50:
        return "UP"
    if price < e20 and e20 < e50:
        return "DOWN"
    return "MIX"

def range_snapshot_from_klines(symbol, tf, candles: int):
    k = get_klines_cached(symbol, tf)
    if k is None or len(k) < candles + 2:
        return None, None
    tail = k[-(candles+1):-1]
    lo = float(np.min(tail[:, 3]))
    hi = float(np.max(tail[:, 2]))
    return lo, hi

def _ctx_label(up_cnt, down_cnt):
    if up_cnt > down_cnt + 1:
        return "UP-ish"
    if down_cnt > up_cnt + 1:
        return "DOWN-ish"
    return "MIX"

def _alert_level_from_total(total_score):
    if total_score >= 13:
        return "ALERT 3"
    if total_score >= 10:
        return "ALERT 2"
    return "ALERT 1"

def _trg_score(tf15, side, ret15):
    p = tf15["price"]
    e20 = tf15["ema20"]
    atr = tf15["atr"]
    volx = tf15["vol_ratio"]

    reasons = []
    score = 0
    up = 0
    down = 0

    if p > e20:
        reasons.append("15m price above EMA20"); score += 1; up += 1
    elif p < e20:
        reasons.append("15m price below EMA20"); score += 1; down += 1
    else:
        reasons.append("15m price on EMA20"); score += 1

    st = _structure_flags_15m(tf15, side)
    if st["break"]:
        reasons.append("Structure (break)"); score += 1
        (up := up + 1) if side == "LONG" else (down := down + 1)
    elif st["reclaim"]:
        reasons.append("Structure (reclaim/reject EMA20)"); score += 1
        (up := up + 1) if side == "LONG" else (down := down + 1)
    else:
        reasons.append("No structure (no break/reclaim)")

    d_ok, d_msg = _delta_dominance(tf15)
    if d_ok:
        reasons.append(f"DELTA dominance ({d_msg})"); score += 1
        dn = tf15.get("vol_delta")
        if dn is not None:
            if dn > 0: up += 1
            elif dn < 0: down += 1

    dist = atr_distance(p, e20, atr)
    reasons.append(f"ATR dist: {dist:.2f}")
    score += 1

    if side == "LONG" and ret15 > 0:
        reasons.append("ret15 supports LONG"); score += 1; up += 1
    elif side == "SHORT" and ret15 < 0:
        reasons.append("ret15 supports SHORT"); score += 1; down += 1
    else:
        reasons.append("ret15 neutral/contra")

    reasons.append(f"VOLx snapshot {volx:.2f}")
    return score, reasons, up, down

# ---- OLD STYLE report (kept) ----

def build_report_oldstyle(rep):
    ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
    sym = rep["symbol"]

    t15 = rep["tf_raw"]["15m"]
    t1h = rep["tf_raw"]["1h"]
    t4h = rep["tf_raw"]["4h"]
    deriv = rep["deriv"]
    oi_chg = rep["oi_change_pct"]

    side_for_report = _pick_best_side_for_report(rep)
    ret15 = rep["returns"]["15m"]

    ctx1, ctx1_reasons, ctx1_up, ctx1_dn = _ctx_score(t1h, side_for_report)
    ctx4, ctx4_reasons, ctx4_up, ctx4_dn = _ctx_score(t4h, side_for_report)
    trg, trg_reasons, trg_up, trg_dn = _trg_score(t15, side_for_report, ret15)

    total = ctx1 + ctx4 + trg
    alert = _alert_level_from_total(total)

    dir15 = tf_trend_label({
        "price": rep["tf"]["15m"]["price"],
        "ema20": rep["tf"]["15m"]["ema20"],
        "ema50": rep["tf"]["15m"]["ema50"],
    })

    up_tr = ctx1_up + ctx4_up + trg_up
    dn_tr = ctx1_dn + ctx4_dn + trg_dn

    lo_4h, hi_4h = range_snapshot_from_klines(sym, "4h", candles=6)
    lo_1h, hi_1h = range_snapshot_from_klines(sym, "1h", candles=6)

    buy = rep["tf"]["15m"].get("vol_buy")
    sell = rep["tf"]["15m"].get("vol_sell")
    delt = rep["tf"]["15m"].get("vol_delta")

    def tf_line(tf_name, snap):
        lbl = tf_trend_label({"price": snap["price"], "ema20": snap["ema20"], "ema50": snap["ema50"]})
        ret = rep["returns"][tf_name]
        return (
            f"`{tf_name:>3}` [{lbl}] p `{fmt_num(snap['price'],2)}` | "
            f"EMA20 `{fmt_num(snap['ema20'],2)}`({fmt_num(snap['ema20_slp'],2)}) "
            f"EMA50 `{fmt_num(snap['ema50'],2)}`({fmt_num(snap['ema50_slp'],2)}) | "
            f"RSI `{fmt_num(snap['rsi'],1)}`({fmt_num(snap['rsi_slp'],1)}) | "
            f"ATR `{fmt_num(snap['atr'],2)}` | "
            f"VOLx `{fmt_num(snap['vol_ratio'],2)}` | ret `{fmt_num(ret,2)}%`"
        )

    msg = ""
    msg += f"*{sym}*  |  *{alert}*  |  *DIR (15m): {dir15}*  |  total `{total}` (ctx1h `{ctx1}` + ctx4h `{ctx4}` + trg `{trg}`)\n"
    msg += f"Triggers: UP `{up_tr}` / DOWN `{dn_tr}`\n"
    msg += (
        f"Deriv: mark `{fmt_num(deriv['mark'],2)}` | index `{fmt_num(deriv['index'],2)}` | "
        f"funding `{fmt_num(deriv['funding'],4)}%` | basis `{fmt_num(deriv['basis'],4)}%` | "
        f"OI `{fmt_num(deriv['oi'],0)}`"
    )
    if oi_chg is not None:
        msg += f" | OIΔ `{fmt_num(oi_chg,2)}%`"
    msg += "\n"
    msg += (
        f"RANGE: 4h(24h) low `{fmt_num(lo_4h,2)}` / high `{fmt_num(hi_4h,2)}`  |  "
        f"1h(6h) low `{fmt_num(lo_1h,2)}` / high `{fmt_num(hi_1h,2)}`\n"
    )
    if buy is not None:
        msg += f"DELTA(15m): buy `{fmt_num(buy,2)}` | sell `{fmt_num(sell,2)}` | Δ `{fmt_num(delt,2)}`\n"

    msg += tf_line("4h", t4h) + "\n"
    msg += tf_line("1h", t1h) + "\n"
    msg += tf_line("15m", t15) + "\n"

    if rep.get("tf5m") is not None and rep.get("entry_hint_5m"):
        t5 = rep["tf5m"]
        h = rep["entry_hint_5m"].get(side_for_report, {})
        msg += (
            f"` 5m` [{tf_trend_label({'price': t5['price'], 'ema20': t5['ema20'], 'ema50': t5['ema50']})}] "
            f"p `{fmt_num(t5['price'],2)}` | EMA20 `{fmt_num(t5['ema20'],2)}` "
            f"EMA50 `{fmt_num(t5['ema50'],2)}` | RSI `{fmt_num(t5['rsi'],1)}` | "
            f"ATR `{fmt_num(t5['atr'],2)}` | VOLx `{fmt_num(t5['vol_ratio'],2)}` "
            f"| ENTRY_HINT `{h.get('hint','N/A')}`\n"
        )

    if rep.get("entry_signals"):
        msg += "\nENTRY_OK:\n"
        for es in rep["entry_signals"][:2]:
            msg += (
                f"• {es['side']} entry `{fmt_num(es['entry_price'],2)}` | "
                f"LEV `{es['lev_hint']}` (score {es['lev_score']}/10) — {es['lev_meta']}\n"
                f"  SL `{fmt_num(es['sl'],2)}` | TP1 `{fmt_num(es['tp1'],2)}` | TP2 `{fmt_num(es['tp2'],2)}` ({es['sltp_note']})\n"
                f"  {es['reasons'][0]}\n"
            )
    else:
        msg += "\nENTRY_OK: none\n"

    msg += "\nWhy flagged:\n"
    msg += f"• Context scored as: {_ctx_label(ctx1_up+ctx4_up, ctx1_dn+ctx4_dn)} (side={side_for_report})\n"
    msg += "• Setup (1h):\n"
    for r in ctx1_reasons[:4]:
        msg += f"  • {r}\n"
    msg += "• Setup (4h):\n"
    for r in ctx4_reasons[:4]:
        msg += f"  • {r}\n"
    msg += "• Trigger (15m):\n"
    for r in trg_reasons[:6]:
        msg += f"  • {r}\n"

    sigs = rep.get("signals", [])
    msg += f"Signals found: {len(sigs)}\n"
    for s in sigs[:6]:
        msg += f"• {s['level']} {s['side']}\n"
        for rr in s.get("reasons", [])[:5]:
            msg += f"  • {rr}\n"

    header = f"📊 *MANUAL REPORT ({sym})*\nUTC: `{ts}`\n_(Scanner only — NOT a signal. Uses LAST CLOSED candles.)_\n\n"
    return header + msg.strip()

def build_report_all_text(reps):
    out = "📊 *MANUAL REPORT (ALL)*\n\n"
    for i, r in enumerate(reps):
        out += build_report_oldstyle(r)
        if i != len(reps) - 1:
            out += "\n\n——————————————————————\n\n"
    return out

# ---- COMPACT report (single-message) ----

def _report_compact_one(rep):
    sym = rep["symbol"]
    t15 = rep["tf"]["15m"]
    t1 = rep["tf"]["1h"]
    t4 = rep["tf"]["4h"]
    deriv = rep["deriv"]
    oi_chg = rep["oi_change_pct"]

    side_for_report = _pick_best_side_for_report(rep)
    ret15 = rep["returns"]["15m"]

    ctx1, _, ctx1_up, ctx1_dn = _ctx_score(rep["tf_raw"]["1h"], side_for_report)
    ctx4, _, ctx4_up, ctx4_dn = _ctx_score(rep["tf_raw"]["4h"], side_for_report)
    trg, trg_reasons, trg_up, trg_dn = _trg_score(rep["tf_raw"]["15m"], side_for_report, ret15)

    total = ctx1 + ctx4 + trg
    alert = _alert_level_from_total(total)
    dir15 = tf_trend_label({"price": t15["price"], "ema20": t15["ema20"], "ema50": t15["ema50"]})

    up_tr = ctx1_up + ctx4_up + trg_up
    dn_tr = ctx1_dn + ctx4_dn + trg_dn

    buy = t15.get("vol_buy")
    sell = t15.get("vol_sell")
    delt = t15.get("vol_delta")

    lines = []
    lines.append(f"*{sym}* | *{alert}* | DIR15 {dir15} | total `{total}` (UP `{up_tr}`/DOWN `{dn_tr}`)")

    lines.append(
        f"p `{fmt_num(t15['price'],2)}` | EMA20/50 `{fmt_num(t15['ema20'],2)}`/`{fmt_num(t15['ema50'],2)}` | "
        f"RSI 15/1h/4h `{fmt_num(t15['rsi'],1)}`/`{fmt_num(t1['rsi'],1)}`/`{fmt_num(t4['rsi'],1)}` | "
        f"ATR15 `{fmt_num(t15['atr'],2)}` VOLx15 `{fmt_num(t15['vol_ratio'],2)}` ret15 `{fmt_num(ret15,2)}%`"
    )

    d = (
        f"fund `{fmt_num(deriv['funding'],4)}%` basis `{fmt_num(deriv['basis'],4)}%` "
        f"OI `{fmt_num(deriv['oi'],0)}`"
    )
    if oi_chg is not None:
        d += f" OIΔ `{fmt_num(oi_chg,2)}%`"
    lines.append("Deriv: " + d)

    if buy is not None:
        lines.append(f"Δ15m: buy `{fmt_num(buy,0)}` sell `{fmt_num(sell,0)}` Δ `{fmt_num(delt,0)}`")

    if rep.get("tf5m") is not None and rep.get("entry_hint_5m"):
        t5 = rep["tf5m"]
        h = rep["entry_hint_5m"].get(side_for_report, {})
        lines.append(
            f"5m: p `{fmt_num(t5['price'],2)}` EMA20 `{fmt_num(t5['ema20'],2)}` RSI `{fmt_num(t5['rsi'],1)}` "
            f"VOLx `{fmt_num(t5['vol_ratio'],2)}` | HINT `{h.get('hint','N/A')}`"
        )

    if rep.get("entry_signals"):
        es = rep["entry_signals"][0]
        lines.append(
            f"ENTRY_OK: *{es['side']}* entry `{fmt_num(es['entry_price'],2)}` "
            f"LEV `{es['lev_hint']}` (score {es['lev_score']}/10) | "
            f"SL `{fmt_num(es['sl'],2)}` TP1 `{fmt_num(es['tp1'],2)}` TP2 `{fmt_num(es['tp2'],2)}`"
        )
    else:
        lines.append("ENTRY_OK: none")

    lines.append("Why: " + "; ".join(trg_reasons[:3]))
    return "\n".join(lines)

def build_report_all_compact(reps):
    ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
    out = [f"📊 *MANUAL REPORT (ALL — COMPACT)*\nUTC: `{ts}`\n_(Scanner only — NOT a signal. Uses LAST CLOSED candles.)_\n"]
    for i, rep in enumerate(reps):
        out.append(_report_compact_one(rep))
        if i != len(reps) - 1:
            out.append("\n——————————\n")
    text = "\n".join(out).strip()

    if len(text) <= TELEGRAM_MAX_LEN:
        return text

    lines = text.splitlines()
    for step in range(3):
        if len("\n".join(lines)) <= TELEGRAM_MAX_LEN:
            break
        if step == 0:
            lines = [ln for ln in lines if not ln.startswith("Why:")]
        elif step == 1:
            lines = [ln for ln in lines if not ln.startswith("5m:")]
        else:
            lines = [ln for ln in lines if not ln.startswith("Δ15m:")]

    final = "\n".join(lines).strip()
    if len(final) > TELEGRAM_MAX_LEN:
        final = final[:TELEGRAM_MAX_LEN-40] + "\n…use /reportfull for full detail"
    return final

# ================== ALERT MESSAGE (AUTO PUSH) ================== #

def build_signal_message(rep, signal):
    ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
    sym = rep["symbol"]
    side = signal["side"]
    level = signal["level"]

    deriv = rep["deriv"]
    oi_chg = rep["oi_change_pct"]

    t15 = rep["tf"]["15m"]
    t1h = rep["tf"]["1h"]
    t4h = rep["tf"]["4h"]

    msg = f"🚨 {sym} — {level} | {side}\nUTC: {ts}\n"
    msg += "(Scanner only — NOT a signal. Uses LAST CLOSED candles.)\n\n"

    msg += (
        f"Price: {fmt_num(t15['price'],2)} | EMA20(15m): {fmt_num(t15['ema20'],2)} | EMA50(15m): {fmt_num(t15['ema50'],2)}\n"
        f"RSI: 15m {fmt_num(t15['rsi'],1)} | 1h {fmt_num(t1h['rsi'],1)} | 4h {fmt_num(t4h['rsi'],1)}\n"
        f"ATR(15m): {fmt_num(t15['atr'],2)} | VOLx(15m): {fmt_num(t15['vol_ratio'],2)} | ret15: {fmt_num(rep['returns']['15m'],2)}%\n"
    )

    if rep["tf"]["15m"].get("vol_buy") is not None:
        msg += (
            f"DELTA(15m): buy {fmt_num(rep['tf']['15m']['vol_buy'],2)} | "
            f"sell {fmt_num(rep['tf']['15m']['vol_sell'],2)} | "
            f"Δ {fmt_num(rep['tf']['15m']['vol_delta'],2)}\n"
        )

    if rep.get("tf5m") is not None and rep.get("entry_hint_5m"):
        t5 = rep["tf5m"]
        h = rep["entry_hint_5m"].get(side, {})
        msg += (
            f"5m: p {fmt_num(t5['price'],2)} | EMA20 {fmt_num(t5['ema20'],2)} | EMA50 {fmt_num(t5['ema50'],2)} | "
            f"RSI {fmt_num(t5['rsi'],1)} | ATR {fmt_num(t5['atr'],2)} | VOLx {fmt_num(t5['vol_ratio'],2)}\n"
            f"ENTRY_HINT(5m): {h.get('hint','N/A')} ({h.get('details','')})\n"
        )

    msg += (
        f"Deriv: mark {fmt_num(deriv['mark'],2)} | funding {fmt_num(deriv['funding'],4)}% | "
        f"basis {fmt_num(deriv['basis'],4)}% | OI {fmt_num(deriv['oi'],0)}"
    )
    if oi_chg is not None:
        msg += f" | OIΔ {fmt_num(oi_chg,2)}%"
    msg += "\n\n"

    if level == "ENTRY_OK":
        msg += f"ENTRY: {fmt_num(signal.get('entry_price'),2)}\n"
        msg += f"LEVERAGE_HINT: {signal.get('lev_hint')} (Score {signal.get('lev_score')}/10) — {signal.get('lev_meta')}\n"
        msg += f"SL/TP: SL {fmt_num(signal.get('sl'),2)} | TP1 {fmt_num(signal.get('tp1'),2)} | TP2 {fmt_num(signal.get('tp2'),2)} ({signal.get('sltp_note')})\n\n"

    msg += "Why flagged:\n"
    for line in signal.get("reasons", [])[:10]:
        msg += f"- {line}\n"

    return msg.strip()

# ================== TELEGRAM COMMAND HANDLER ================== #

def handle_telegram_commands():
    global _last_update_id
    if not TELEGRAM_TOKEN:
        return

    updates = telegram_get_updates(offset=_last_update_id + 1 if _last_update_id else None, timeout=20)
    if not updates:
        return

    for u in updates:
        _last_update_id = max(_last_update_id, u.get("update_id", 0))
        msg = u.get("message") or {}
        chat = msg.get("chat") or {}
        chat_id = str(chat.get("id", "")).strip()

        text = (msg.get("text") or "").strip()
        if not text:
            continue

        if not is_allowed_chat(chat_id):
            continue

        if "@" in text and text.startswith("/"):
            text = text.split("@", 1)[0].strip()

        if text.startswith("/status"):
            send_telegram(
                "✅ Bot is running.\n"
                "v8.2.0: WATCH=PULLBACK (quality) + ENTRY_OK fast + impulse fix + consistent 5m hint.\n"
                f"Auto-push levels: {', '.join(sorted(AUTO_PUSH_LEVELS))}\n"
                "Manual: /report, /report ETHUSDT, /reportfull, /reportfull ETHUSDT",
                chat_id=chat_id
            )
            continue

        if text.startswith("/reportfull"):
            parts = text.split()
            try:
                if len(parts) == 1:
                    reps = [analyze_symbol(s) for s in SYMBOLS]
                    out = build_report_all_text(reps)
                    send_telegram(out, chat_id=chat_id)
                else:
                    sym = parts[1].upper()
                    if sym not in SYMBOLS:
                        send_telegram(f"❗Unknown symbol: {sym}\nAllowed: {', '.join(SYMBOLS)}", chat_id=chat_id)
                        continue
                    r = analyze_symbol(sym)
                    out = build_report_oldstyle(r)
                    send_telegram(out, chat_id=chat_id)
            except Exception as e:
                send_telegram(f"❌ Manual report error:\n{e}", chat_id=chat_id)
            continue

        if text.startswith("/report"):
            parts = text.split()
            try:
                if len(parts) == 1:
                    reps = [analyze_symbol(s) for s in SYMBOLS]
                    if REPORT_SINGLE_MESSAGE:
                        out = build_report_all_compact(reps)
                        send_telegram_single(out, chat_id=chat_id)
                    else:
                        out = build_report_all_text(reps)
                        send_telegram(out, chat_id=chat_id)
                else:
                    sym = parts[1].upper()
                    if sym not in SYMBOLS:
                        send_telegram(f"❗Unknown symbol: {sym}\nAllowed: {', '.join(SYMBOLS)}", chat_id=chat_id)
                        continue
                    r = analyze_symbol(sym)
                    if REPORT_SINGLE_MESSAGE:
                        out = "📊 *MANUAL REPORT (COMPACT)*\n\n" + _report_compact_one(r)
                        send_telegram_single(out[:TELEGRAM_MAX_LEN], chat_id=chat_id)
                    else:
                        out = build_report_oldstyle(r)
                        send_telegram(out, chat_id=chat_id)
            except Exception as e:
                send_telegram(f"❌ Manual report error:\n{e}", chat_id=chat_id)
            continue

# ================== MAIN SCAN (AUTO PUSH) ================== #

def run_once():
    for sym in SYMBOLS:
        try:
            rep = analyze_symbol(sym)
        except Exception as e:
            if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
                send_telegram(f"⚠️ Symbol error {sym}\n{e}")
            continue

        combined = []
        combined.extend(rep.get("signals", []))
        combined.extend(rep.get("entry_signals", []))

        for sig in combined:
            if sig["level"] not in AUTO_PUSH_LEVELS:
                continue
            if allowed_to_send(sym, sig, rep):
                if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
                    send_telegram(build_signal_message(rep, sig))
                mark_sent(sym, sig, rep)

# ================== RUNNER ================== #

if __name__ == "__main__":
    if TELEGRAM_TOKEN:
        telegram_delete_webhook()

    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        send_telegram(
            "✅ Momentum bot ONLINE (v8.2.0)\n"
            f"- Scan every {SCAN_SECONDS}s\n"
            "- Candles: evaluates LAST CLOSED (fixes flip-flop)\n"
            f"- Auto-push: {', '.join(sorted(AUTO_PUSH_LEVELS))}\n"
            "- WATCH: PULLBACK (quality, rarer)\n"
            "- ENTRY_OK: fast (1-per-5m-candle gate)\n"
            "- /report: single-message compact\n"
            "- /reportfull: original multi-message (no truncation)\n\n"
            "If /report doesn't respond in a GROUP: use /report@YourBotName or disable privacy mode in BotFather."
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
                try:
                    send_telegram(f"❌ Runtime error\n{e}")
                except Exception:
                    pass

        time.sleep(0.2)