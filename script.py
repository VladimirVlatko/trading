"""
MOMENTUM PRO – FILTER (Telegram Enabled, SAFE) — OPTIMIZED
- Detects potential momentum moments (UP / DOWN) across 15m/1h/4h
- NO trading signals (no buy/sell, no entries/exits, no targets)
- Designed as a "momentum scanner + snapshot" for human + ChatGPT analysis
- Sends Telegram message ONLY if ALERT LEVEL >= 2
- Includes derivatives context: funding, basis, mark/index, OI + OI change
- Optimizations:
  * Anti-spam: 15m candle gating (max 1 msg per symbol per 15m candle) unless alert upgrades
  * Cooldowns per alert level
  * Alert 3 requires real 15m trigger (prevents "HTF-only" ALERT 3)
  * Cleaner reasons (no duplication)
"""

import os
import time
import requests
import numpy as np
from datetime import datetime, UTC

# ================== CONFIG ================== #

BINANCE_FUTURES = "https://fapi.binance.com"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
TIMEFRAMES = {"15m": "15m", "1h": "1h", "4h": "4h"}

EMA_FAST = 20
EMA_SLOW = 50
RSI_LEN = 14
ATR_LEN = 14

VOL_LOOKBACK = 20
SLOPE_LOOKBACK = 6  # slope vs last N candles
RETURN_LOOKBACK_15M = 6   # ~90 mins on 15m
RETURN_LOOKBACK_1H = 6    # ~6 hours on 1h
RETURN_LOOKBACK_4H = 6    # ~24 hours on 4h

# Alert thresholds
ALERT2_SCORE = 5
ALERT3_SCORE = 7

# Trigger requirements for ALERT 3 (must be "real", not HTF-only)
ALERT3_MIN_TRIGGER_REASONS = 2

# Cooldowns (seconds) per symbol+direction
COOLDOWN_ALERT2 = 45 * 60   # 45 min (heads-up)
COOLDOWN_ALERT3 = 90 * 60   # 90 min (strong)

# Loop interval
SLEEP_SECONDS = 300  # 5 minutes

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# ============================================ #

# In-memory state (persists while process runs)
_last_sent = {}           # key: (symbol, direction) -> timestamp
_last_alert_level = {}    # key: (symbol, direction) -> last alert sent (0/2/3)
_last_15m_candle = {}     # key: symbol -> last 15m open_time(ms) that produced a message
_prev_oi = {}             # key: symbol -> last open interest

# ---------- NETWORK HELPERS ---------- #

def http_get_json(url, params=None, timeout=10):
    for attempt in range(2):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception:
            if attempt == 1:
                raise
            time.sleep(0.5)

def http_post_json(url, payload, timeout=10):
    for attempt in range(2):
        try:
            r = requests.post(url, json=payload, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception:
            if attempt == 1:
                raise
            time.sleep(0.5)

# ---------- TELEGRAM ---------- #

def send_telegram(message: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    http_post_json(url, payload, timeout=10)

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
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )

    atr = np.empty(len(close))
    atr[:length] = np.mean(tr[:length])
    for i in range(length, len(close)):
        atr[i] = (atr[i - 1] * (length - 1) + tr[i]) / length

    return atr

def pct_change(a, b):
    return ((a - b) / (b + 1e-12)) * 100.0

# ---------- DATA ---------- #

def fetch_klines(symbol, interval, limit=180):
    url = f"{BINANCE_FUTURES}/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    data = http_get_json(url, params=params, timeout=10)
    return np.array(data, dtype=float)

def fetch_derivatives(symbol):
    mark = http_get_json(
        f"{BINANCE_FUTURES}/fapi/v1/premiumIndex",
        params={"symbol": symbol},
        timeout=10
    )
    oi = http_get_json(
        f"{BINANCE_FUTURES}/fapi/v1/openInterest",
        params={"symbol": symbol},
        timeout=10
    )
    mark_price = float(mark["markPrice"])
    index_price = float(mark["indexPrice"])
    funding_pct = float(mark["lastFundingRate"]) * 100.0
    basis_pct = (mark_price - index_price) / (index_price + 1e-12) * 100.0
    oi_val = float(oi["openInterest"])

    return {
        "mark": mark_price,
        "index": index_price,
        "funding": funding_pct,
        "basis": basis_pct,
        "oi": oi_val
    }

# ---------- SNAPSHOT ---------- #

def tf_snapshot(k, tf_name):
    # k columns: [open_time, open, high, low, close, volume, ...]
    close = k[:, 4]
    high = k[:, 2]
    low = k[:, 3]
    vol = k[:, 5]

    e20 = ema(close, EMA_FAST)
    e50 = ema(close, EMA_SLOW)
    rsi_v = rsi_wilder(close, RSI_LEN)
    atr_v = atr_wilder(high, low, close, ATR_LEN)

    sl_n = SLOPE_LOOKBACK
    e20_slp = e20[-1] - e20[-1 - sl_n] if len(e20) > sl_n else e20[-1] - e20[0]
    e50_slp = e50[-1] - e50[-1 - sl_n] if len(e50) > sl_n else e50[-1] - e50[0]
    rsi_slp = rsi_v[-1] - rsi_v[-1 - sl_n] if len(rsi_v) > sl_n else 0.0

    vol_ratio = vol[-1] / (np.mean(vol[-VOL_LOOKBACK:]) + 1e-12) if len(vol) >= VOL_LOOKBACK else 1.0

    open_time = int(k[-1, 0])  # ms, last candle open time

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
        "close_series": close
    }

# ---------- MOMENTUM SCORING ---------- #

def regime_score(tf_snap, direction):
    """
    Higher timeframe regime: trend/structure alignment (1h, 4h)
    direction = "UP" or "DOWN"
    """
    price = tf_snap["price"]
    e20 = tf_snap["ema20"]
    e50 = tf_snap["ema50"]
    e20_slp = tf_snap["ema20_slp"]
    e50_slp = tf_snap["ema50_slp"]
    rsi = tf_snap["rsi"]

    score = 0
    reasons = []

    if direction == "UP":
        if price > e20 and price > e50:
            score += 2; reasons.append("HTF price above EMA20/50")
        if e20 > e50:
            score += 1; reasons.append("HTF EMA20 > EMA50")
        if e20_slp > 0 and e50_slp > 0:
            score += 1; reasons.append("HTF EMA slopes positive")
        if rsi >= 52:
            score += 1; reasons.append("HTF RSI supportive")
    else:
        if price < e20 and price < e50:
            score += 2; reasons.append("HTF price below EMA20/50")
        if e20 < e50:
            score += 1; reasons.append("HTF EMA20 < EMA50")
        if e20_slp < 0 and e50_slp < 0:
            score += 1; reasons.append("HTF EMA slopes negative")
        if rsi <= 48:
            score += 1; reasons.append("HTF RSI weak")

    return score, reasons

def trigger_score_15m(tf15, direction, ret_15m):
    """
    15m trigger: momentum impulse characteristics (must exist for ALERT 3)
    Uses ret_15m as extra confirmation to reduce HTF-only false ALERT 3.
    """
    price = tf15["price"]
    e20 = tf15["ema20"]
    e50 = tf15["ema50"]
    rsi = tf15["rsi"]
    rsi_slp = tf15["rsi_slp"]
    vol_ratio = tf15["vol_ratio"]

    score = 0
    reasons = []

    if direction == "UP":
        if price > e20:
            score += 1; reasons.append("15m price above EMA20")
        if price > e20 and price > e50:
            score += 1; reasons.append("15m price above EMA20/50")
        if rsi >= 53:
            score += 1; reasons.append("15m RSI high")
        if rsi_slp >= 3:
            score += 1; reasons.append("15m RSI rising")
        if vol_ratio >= 1.15:
            score += 1; reasons.append(f"15m volume expansion ({vol_ratio:.2f}x)")
        if ret_15m >= 0.15:
            score += 1; reasons.append(f"15m return positive ({ret_15m:+.2f}%)")
        if vol_ratio >= 1.25 and rsi_slp >= 5:
            score += 1; reasons.append("15m aggression combo (vol + RSI slope)")
    else:
        if price < e20:
            score += 1; reasons.append("15m price below EMA20")
        if price < e20 and price < e50:
            score += 1; reasons.append("15m price below EMA20/50")
        if rsi <= 47:
            score += 1; reasons.append("15m RSI low")
        if rsi_slp <= -3:
            score += 1; reasons.append("15m RSI falling")
        if vol_ratio >= 1.15:
            score += 1; reasons.append(f"15m volume expansion ({vol_ratio:.2f}x)")
        if ret_15m <= -0.15:
            score += 1; reasons.append(f"15m return negative ({ret_15m:+.2f}%)")
        if vol_ratio >= 1.25 and rsi_slp <= -5:
            score += 1; reasons.append("15m aggression combo (vol + RSI slope)")

    return score, reasons

def compute_returns(tf_snaps):
    out = {}
    for tf, snap in tf_snaps.items():
        close = snap["close_series"]
        if tf == "15m":
            lb = RETURN_LOOKBACK_15M
        elif tf == "1h":
            lb = RETURN_LOOKBACK_1H
        else:
            lb = RETURN_LOOKBACK_4H

        out[tf] = pct_change(close[-1], close[-1 - lb]) if len(close) > lb else 0.0
    return out

def tf_trend_label(snap):
    price = snap["price"]
    e20 = snap["ema20"]
    e50 = snap["ema50"]
    if price > e20 and e20 > e50:
        return "UP"
    if price < e20 and e20 < e50:
        return "DOWN"
    return "MIX"

# ---------- ANALYSIS ---------- #

def analyze_symbol(symbol):
    # Fetch klines and snapshots
    tf_snaps = {}
    for tf, interval in TIMEFRAMES.items():
        k = fetch_klines(symbol, interval, limit=180)
        tf_snaps[tf] = tf_snapshot(k, tf)

    returns = compute_returns(tf_snaps)
    ret_15m = returns["15m"]

    # Derivatives
    deriv = fetch_derivatives(symbol)
    prev_oi = _prev_oi.get(symbol)
    oi_change_pct = pct_change(deriv["oi"], prev_oi) if prev_oi and prev_oi > 0 else None
    _prev_oi[symbol] = deriv["oi"]

    # Scores for both directions
    up_reg_1h, up_reg_1h_r = regime_score(tf_snaps["1h"], "UP")
    up_reg_4h, up_reg_4h_r = regime_score(tf_snaps["4h"], "UP")
    dn_reg_1h, dn_reg_1h_r = regime_score(tf_snaps["1h"], "DOWN")
    dn_reg_4h, dn_reg_4h_r = regime_score(tf_snaps["4h"], "DOWN")

    up_trg, up_trg_r = trigger_score_15m(tf_snaps["15m"], "UP", ret_15m)
    dn_trg, dn_trg_r = trigger_score_15m(tf_snaps["15m"], "DOWN", ret_15m)

    up_score = up_reg_1h + up_reg_4h + up_trg
    dn_score = dn_reg_1h + dn_reg_4h + dn_trg

    direction = "UP" if up_score >= dn_score else "DOWN"
    best_score = max(up_score, dn_score)

    # Selected reasons
    if direction == "UP":
        setup_reasons = ["Setup (1h):"] + up_reg_1h_r + ["Setup (4h):"] + up_reg_4h_r
        trigger_reasons = up_trg_r
        scores = {"best_score": best_score, "up_score": up_score, "dn_score": dn_score}
    else:
        setup_reasons = ["Setup (1h):"] + dn_reg_1h_r + ["Setup (4h):"] + dn_reg_4h_r
        trigger_reasons = dn_trg_r
        scores = {"best_score": best_score, "up_score": up_score, "dn_score": dn_score}

    # Alert level
    alert = 0
    if best_score >= ALERT2_SCORE:
        alert = 2
    if best_score >= ALERT3_SCORE:
        alert = 3

    # Prevent HTF-only ALERT 3 (must have real 15m trigger)
    if alert == 3 and len(trigger_reasons) < ALERT3_MIN_TRIGGER_REASONS:
        alert = 2

    trend_labels = {tf: tf_trend_label(tf_snaps[tf]) for tf in tf_snaps.keys()}

    # Strip close_series before returning
    tf_public = {k: {kk: vv for kk, vv in v.items() if kk != "close_series"} for k, v in tf_snaps.items()}

    # Build compact reasons (no empty Trigger)
    reasons = []
    reasons += setup_reasons
    reasons += ["Trigger (15m):"]
    reasons += trigger_reasons if trigger_reasons else ["(no strong 15m trigger yet)"]

    return {
        "symbol": symbol,
        "alert": alert,
        "direction": direction,
        "best_score": scores["best_score"],
        "up_score": scores["up_score"],
        "dn_score": scores["dn_score"],
        "reasons": reasons,
        "tf": tf_public,
        "returns": returns,
        "trend_labels": trend_labels,
        "deriv": deriv,
        "oi_change_pct": oi_change_pct
    }

# ---------- ANTI-SPAM / GATING ---------- #

def allowed_to_send(report):
    symbol = report["symbol"]
    direction = report["direction"]
    alert = report["alert"]
    now = time.time()

    key = (symbol, direction)
    last_sent = _last_sent.get(key)
    last_level = _last_alert_level.get(key, 0)

    candle_ot = report["tf"]["15m"]["open_time"]  # ms
    last_candle_for_symbol = _last_15m_candle.get(symbol)

    # Upgrade path: if alert increased (e.g., 2 -> 3), send immediately
    if alert > last_level:
        return True

    # Apply cooldown
    cooldown = COOLDOWN_ALERT3 if alert >= 3 else COOLDOWN_ALERT2
    if last_sent is not None and (now - last_sent) < cooldown:
        return False

    # Gate: only one message per new 15m candle (per symbol), unless upgrade
    if last_candle_for_symbol is not None and candle_ot == last_candle_for_symbol:
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

# ---------- MESSAGE FORMAT ---------- #

def fmt_num(x, digits=2):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "n/a"
        return f"{x:.{digits}f}"
    except Exception:
        return "n/a"

def build_message(active_reports):
    ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
    msg = f"🔔 *MOMENTUM FILTER ALERT*\nUTC: `{ts}`\n"
    msg += "_(Scanner only — no trade signals. Use as input for analysis.)_\n\n"

    for r in active_reports:
        sym = r["symbol"]
        alert = r["alert"]
        direction = r["direction"]
        score = r["best_score"]
        up_score = r["up_score"]
        dn_score = r["dn_score"]

        deriv = r["deriv"]
        oi_chg = r["oi_change_pct"]

        msg += f"*{sym}*  |  *ALERT {alert}*  |  *DIR: {direction}*  |  score: `{score}` (UP `{up_score}` / DOWN `{dn_score}`)\n"
        msg += (
            f"Deriv: mark `{fmt_num(deriv['mark'], 2)}` | index `{fmt_num(deriv['index'], 2)}` | "
            f"funding `{fmt_num(deriv['funding'], 4)}%` | basis `{fmt_num(deriv['basis'], 4)}%` | "
            f"OI `{fmt_num(deriv['oi'], 0)}`"
        )
        if oi_chg is not None:
            msg += f" | OIΔ `{fmt_num(oi_chg, 2)}%`"
        msg += "\n"

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
        for line in r["reasons"][:14]:
            msg += f"• {line}\n"
        msg += "\n"

    return msg.strip()

# ---------- MAIN LOOP ---------- #

def run_once():
    reports = []
    for s in SYMBOLS:
        try:
            reports.append(analyze_symbol(s))
        except Exception as e:
            send_telegram(f"⚠️ *Symbol error* `{s}`\n`{e}`")
            continue

    active = []
    for r in reports:
        if r["alert"] >= 2 and allowed_to_send(r):
            active.append(r)

    if not active:
        return

    msg = build_message(active)
    send_telegram(msg)

    for r in active:
        mark_sent(r)

# ---------- RUNNER ---------- #

if __name__ == "__main__":
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        send_telegram(
            "✅ Momentum Filter bot is ONLINE on Railway.\n"
            "Vars OK. Scanning 15m/1h/4h for UP/DOWN momentum candidates.\n"
            "Optimized: 15m candle gating + cooldowns + ALERT3 needs real 15m trigger."
        )

    while True:
        try:
            run_once()
        except Exception as e:
            if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
                send_telegram(f"❌ *Runtime error*\n`{e}`")
        time.sleep(SLEEP_SECONDS)
