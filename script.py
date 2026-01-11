from __future__ import annotations

import os
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Union, Tuple, Dict, Any, List

import requests
import pandas as pd
import pandas_ta as ta

# ============================================================
# ENV (required)
# ============================================================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
    raise RuntimeError("Missing env vars: TELEGRAM_TOKEN and/or TELEGRAM_CHAT_ID")

# ============================================================
# CONFIG - V3: TRUE HTF+15M RADAR
# ============================================================
BINANCE_FAPI = "https://fapi.binance.com"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]



# ============================================================
# 10s ENTRY CHECKLIST — per-symbol thresholds (retail edge)
# ============================================================
CHECKLIST_THRESHOLDS: Dict[str, Dict[str, float]] = {
    # BTC: cleaner moves, stricter reclaim, tighter SL/RR
    "BTCUSDT": {
        "volx15_min": 0.80, "volx15_max": 1.40,
        "volx1_max": 2.00,
        "wick_body_max": 2.00, "range1m_atr15_max": 0.35,
        "rsi_long_min": 51.0, "rsi_short_max": 49.0,
        "reclaim_mode": "AND",   # EMA20 AND VWAP
        "absorb_lookback": 5, "absorb_buf_atr": 0.05,
        "rej_dist_max": 0.80,
        "sl_atr_cap": 0.80,
        "rr_min": 1.60,
        "grade_a_min": 60.0, "grade_aplus_min": 75.0,
    },
    # ETH: balanced, avoid chop, slightly looser than BTC
    "ETHUSDT": {
        "volx15_min": 0.85, "volx15_max": 1.60,
        "volx1_max": 2.20,
        "wick_body_max": 2.20, "range1m_atr15_max": 0.40,
        "rsi_long_min": 50.0, "rsi_short_max": 50.0,
        "reclaim_mode": "EMA",   # EMA20 only (VWAP optional)
        "absorb_lookback": 5, "absorb_buf_atr": 0.06,
        "rej_dist_max": 0.90,
        "sl_atr_cap": 0.90,
        "rr_min": 1.50,
        "grade_a_min": 60.0, "grade_aplus_min": 75.0,
    },
    # SOL: hotter coin, allow more noise but control spikes
    "SOLUSDT": {
        "volx15_min": 0.90, "volx15_max": 1.90,
        "volx1_max": 2.80,
        "wick_body_max": 2.20, "range1m_atr15_max": 0.45,
        "rsi_long_min": 50.0, "rsi_short_max": 50.0,
        "reclaim_mode": "OR",    # EMA20 OR VWAP
        "absorb_lookback": 6, "absorb_buf_atr": 0.10,
        "rej_dist_max": 1.00,
        "sl_atr_cap": 1.00,
        "rr_min": 1.40,
        "grade_a_min": 60.0, "grade_aplus_min": 75.0,
    },
}

def _get_th(symbol: str) -> Dict[str, float]:
    return CHECKLIST_THRESHOLDS.get(symbol, CHECKLIST_THRESHOLDS["BTCUSDT"])


def is_1m_calm(df1m, atr15: float, th: Dict[str, Any]) -> bool:
    """Return True if 1m microstructure is not chaotic.

    Uses last 5 CLOSED 1m candles (excludes current forming candle).
    Checks:
    - optional 1m VOLx cap (volx1_max)
    - avg wick/body ratio cap (wick_body_max)
    - 1m range over last 5 vs ATR(15m) cap (range1m_atr15_max)
    """
    try:
        if df1m is None or len(df1m) < 10:
            return True
        if not _valid(atr15) or atr15 <= 0:
            return True

        wick_body_max = float(th.get("wick_body_max", 2.2))
        range1m_atr15_max = float(th.get("range1m_atr15_max", 0.40))
        volx1_max = float(th.get("volx1_max", 2.0))

        recent = df1m.iloc[-6:-1].copy()  # last 5 closed 1m candles

        # Volume spike gate (optional)
        if "vol_sma20" in recent.columns and "volume" in recent.columns:
            v1 = safe_float(recent["volume"].iloc[-1])
            v1a = safe_float(recent["vol_sma20"].iloc[-1])
            volx1 = (v1 / v1a) if (_valid(v1, v1a) and v1a > 0) else 0.0
            if volx1_max > 0 and volx1 > volx1_max:
                return False

        # Wick-to-body ratio
        body = (recent["close"] - recent["open"]).abs().clip(lower=1e-9)
        rng = (recent["high"] - recent["low"]).clip(lower=0.0)
        wick = (rng - body).clip(lower=0.0)
        avg_wick_body = float((wick / body).mean())
        if avg_wick_body > wick_body_max:
            return False

        # Range over last 5 closed 1m candles relative to 15m ATR
        range_1m = float(recent["high"].max() - recent["low"].min())
        if (range_1m / atr15) > range1m_atr15_max:
            return False

        return True
    except Exception:
        return True

def _reclaim_ok(px: Optional[float], ema20: Optional[float], vwap: Optional[float], mode: str) -> bool:
    if not _valid(px):
        return False
    ok_ema = _valid(ema20) and px > ema20
    ok_vwap = _valid(vwap) and px > vwap
    if mode == "AND":
        return ok_ema and ok_vwap
    if mode == "EMA":
        return ok_ema
    if mode == "VWAP":
        return ok_vwap
    # OR (default)
    return ok_ema or ok_vwap

def compute_10s_checklist(symbol: str, direction: str, cache: "MarketCache", signal: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """10 quick YES/NO gates — deterministic, per-symbol thresholds.
    Note: this is NOT a signal. It is a tradeability gate for alerts/reports.
    """
    th = _get_th(symbol)
    df15 = cache.get(symbol, "15m")
    df1 = cache.get(symbol, "1m")
    df1h = cache.get(symbol, "1h")
    df4h = cache.get(symbol, "4h")
    if df15 is None or len(df15) < 30:
        return {"items": [], "yes": 0, "no": 10, "gate": "NO TRADE", "reason": "Not enough data"}

    i15 = len(df15) - 2  # LAST CLOSED
    px = safe_float(df15.loc[i15, "close"])
    ema20 = safe_float(df15.loc[i15, "ema20"])
    ema50 = safe_float(df15.loc[i15, "ema50"])
    vwap = safe_float(df15.loc[i15, "vwap"])
    atr = safe_float(df15.loc[i15, "atr14"])
    rsi = safe_float(df15.loc[i15, "rsi14"])

    # VOLx 15m
    vol = safe_float(df15.loc[i15, "volume"])
    vol_avg = safe_float(df15.loc[i15, "vol_sma20"])
    volx15 = (vol / vol_avg) if (_valid(vol, vol_avg) and vol_avg > 0) else 0.0

    # 1m spike filter
    volx1 = 0.0
    if df1 is not None and len(df1) >= 25:
        i1 = len(df1) - 2
        v1 = safe_float(df1.loc[i1, "volume"])
        v1a = safe_float(df1.loc[i1, "vol_sma20"])
        volx1 = (v1 / v1a) if (_valid(v1, v1a) and v1a > 0) else 0.0

    # 1) Trend OK (HTF alignment)
    htf_long, htf_short, htf_ctx = check_htf_alignment(df1h, df4h)
    trend_ok = htf_long if direction == "LONG" else htf_short


    # HTF regime score (0-10) — finer context used for adaptive RR & filters
    htf_regime = compute_htf_regime_score(htf_ctx, direction)

    # 2) Reclaim EMA/VWAP
    reclaim_ok = _reclaim_ok(px, ema20, vwap, str(th.get("reclaim_mode", "OR")))

    # 3) Healthy VOLx (15m)
    vol_ok = (volx15 >= th["volx15_min"]) and (volx15 <= th["volx15_max"])

    # 4) RSI supports
    if direction == "LONG":
        rsi_ok = (rsi >= th["rsi_long_min"])
    elif direction == "SHORT":
        rsi_ok = (rsi <= th["rsi_short_max"])
    else:
        rsi_ok = False

    # 5) No breakdown / absorption (proxy: no new extreme in lookback)
    lb = int(th.get("absorb_lookback", 5))
    buf = float(th.get("absorb_buf_atr", 0.05))
    no_break = False
    if _valid(px, atr) and atr > 0 and i15 - lb >= 2 and "low" in df15 and "high" in df15:
        prev = df15.iloc[i15 - lb:i15]  # previous closed candles (exclude last closed)
        prev_low = safe_float(prev["low"].min())
        prev_high = safe_float(prev["high"].max())
        last_low = safe_float(df15.loc[i15, "low"])
        last_high = safe_float(df15.loc[i15, "high"])
        if direction == "LONG":
            # bullish absorption: last candle does NOT print a new low vs the prior window
            no_break = _valid(prev_low, last_low) and last_low >= (prev_low + buf * atr)
        elif direction == "SHORT":
            # bearish absorption: last candle does NOT print a new high vs the prior window
            no_break = _valid(prev_high, last_high) and last_high <= (prev_high - buf * atr)
        else:
            no_break = False
    else:
        prev_low = prev_high = None

    # 6) Rejections weakening (proxy: dist to EMA20 improving + below max dist)
    dist_now = _dist_atr(px, ema20, atr) if _valid(px, ema20, atr) else None
    dist_prev = None
    if i15 - 1 >= 1:
        pxp = safe_float(df15.loc[i15 - 1, "close"])
        emap = safe_float(df15.loc[i15 - 1, "ema20"])
        atrp = safe_float(df15.loc[i15 - 1, "atr14"])
        dist_prev = _dist_atr(pxp, emap, atrp) if _valid(pxp, emap, atrp) else None
    rej_ok = (dist_now is not None) and (dist_now <= float(th.get("rej_dist_max", 0.9))) and (dist_prev is None or dist_now <= dist_prev)

    # 7) Clear SL (must exist and be <= cap*ATR)
    sl_ok = False
    sl_dist = None
    if _valid(px, atr) and atr > 0:
        if direction == "LONG" and _valid(prev_low):
            sl_dist = px - prev_low
        elif direction == "SHORT" and _valid(prev_high):
            sl_dist = prev_high - px
        if _valid(sl_dist) and sl_dist > 0:
            sl_ok = (sl_dist <= float(th.get("sl_atr_cap", 1.0)) * atr)

    # 8) RR >= rr_min (target: nearest of VWAP/EMA50 in direction)
    rr_ok = False
    base_rr_min = float(th.get("rr_min", 1.5))
    # Adaptive RR minimum based on HTF regime (chop -> require bigger RR)
    if htf_regime <= 4:
        rr_min = max(base_rr_min, 2.00)
    elif htf_regime <= 7:
        rr_min = max(base_rr_min, base_rr_min + 0.20)
    else:
        rr_min = base_rr_min
    if sl_ok and _valid(sl_dist) and sl_dist > 0:
        targets = []
        if direction == "LONG":
            if _valid(vwap) and vwap > px: targets.append(vwap)
            if _valid(ema50) and ema50 > px: targets.append(ema50)
            if targets:
                tp = min(targets)  # nearest logical target
                rr_ok = (tp - px) >= rr_min * sl_dist
        else:
            if _valid(vwap) and vwap < px: targets.append(vwap)
            if _valid(ema50) and ema50 < px: targets.append(ema50)
            if targets:
                tp = max(targets)  # nearest logical target below (closest)
                rr_ok = (px - tp) >= rr_min * sl_dist

    # 9) Grade A/A+ (objective)
    qscore = float(signal.get("quality_score", 0)) if isinstance(signal, dict) else 0.0
    long_s = float(signal.get("long_score", 0)) if isinstance(signal, dict) else 0.0
    short_s = float(signal.get("short_score", 0)) if isinstance(signal, dict) else 0.0
    diff = abs(long_s - short_s)
    grade_ok = (qscore >= float(th.get("grade_a_min", 60.0))) and (diff >= MIN_SCORE_DIFF)

    # Order flow veto (optional): if signal provides aggressor data and it's strongly against direction, veto entry
    of_veto = False
    if isinstance(signal, dict):
        aggr_buy = float(signal.get('aggr_buy', signal.get('aggr_buy_pct', 50.0)) or 50.0)
        # For LONG: very low aggr_buy implies sellers dominant. For SHORT: very high aggr_buy implies buyers dominant.
        if direction == 'LONG' and aggr_buy <= float(th.get('flow_veto_long_aggr_max', 30.0)):
            of_veto = True
        if direction == 'SHORT' and aggr_buy >= float(th.get('flow_veto_short_aggr_min', 70.0)):
            of_veto = True

    # 10) Calm / no spike (1m)
    calm_ok = is_1m_calm(df1, atr, th)

    items = [
        ("Trend OK", trend_ok),
        ("Reclaim EMA/VWAP", reclaim_ok),
        ("Healthy VOLx(15m)", vol_ok),
        ("RSI supports", rsi_ok),
        ("No breakdown", no_break),
        ("Rejections weakening", rej_ok),
        ("Clear SL", sl_ok),
        (f"RR>={rr_min:.1f}", rr_ok),
        ("Grade A/A+", grade_ok),
        ("Calm (1m)", calm_ok),
    ]
    yes = sum(1 for _, v in items if v)
    gate = "VALID ENTRY" if (yes == 10 and not of_veto) else "NO TRADE"
    return {"items": items, "yes": yes, "no": 10 - yes, "gate": gate, "volx15": volx15, "volx1": volx1, "htf_regime": htf_regime, "of_veto": of_veto}

def format_checklist_block(check: Dict[str, Any], symbol: str = "") -> str:
    if not check or not check.get("items"):
        return "⏱ 10s CHECKLIST: N/A"
    lines = ["⏱ 10s ENTRY CHECKLIST" + (f" ({symbol})" if symbol else "") + ":"]
    for name, ok in check["items"]:
        lines.append(f"  {'YES' if ok else 'NO '} | {name}")
    lines.append(f"GATE: {check['gate']} ({check['yes']}/10 YES)")
    return "\n".join(lines)

# Loop tick: relaxed for rate-limit safety
SLEEP_TARGET_SECONDS = 2.0  # Was 0.5 → now 2.0 for stability

# Binance request hygiene (SAFER)
MIN_BINANCE_INTERVAL = 0.7  # Was 0.35 → now 0.7 to avoid bans
BAN_BACKOFF_DEFAULT = 60

# HTTP timeouts
HTTP_TIMEOUT = 12

# Telegram polling
TELEGRAM_POLL_TIMEOUT = 20  # long-poll timeout
TG_HTTP_TIMEOUT = 30        # request timeout should exceed poll timeout
TG_CONFLICT_BACKOFF = 30

# Manual command anti-spam
REPORT_COOLDOWN_SECONDS = 5.0

# Candle limits
OHLCV_LIMIT_1M = 60    # Reduced: only for display, not scoring
OHLCV_LIMIT_5M = 120   # Reduced: only for display, not scoring
OHLCV_LIMIT_15M = 220  # MAIN: This is where signals come from
OHLCV_LIMIT_HTF = 260

# ============================================================
# V3 PARAMETERS: 15M-BASED SCORING + REALISTIC THRESHOLDS
# ============================================================

# === HTF Filter (unchanged - already solid) ===
HTF_REQUIRE_BOTH = True  # Both 1h AND 4h must align

# === MTF Alignment (RELAXED - gives bonus points, not required) ===
MTF_GIVES_BONUS = True   # MTF alignment adds points but doesn't block signals
MTF_BONUS_POINTS = 10    # Bonus if 5m aligns with 15m

# === Volume Requirements (REALISTIC) ===
# 15m volume thresholds (main scoring)
VOL_WATCH_MIN_15M = 1.2    # Was 1.8 → now 1.2 (more realistic)
VOL_ENTRY_MIN_15M = 1.5    # Was 2.5 → now 1.5 (achievable)
VOL_PERFECT_MIN_15M = 2.0  # Was 2.5 → now 2.0 (rare but possible)

# 5m/1m volume (INFO ONLY - not used for scoring)
VOL_INFO_SPIKE_5M = 1.8    # Just for display
VOL_INFO_SPIKE_1M = 1.5    # Just for display

# === RSI Sweet Spots (PRACTICAL) ===
# Long entry zone
RSI_LONG_SWEET_MIN = 45    # Was 35 → now 45 (not too early)
RSI_LONG_SWEET_MAX = 60    # Was 52 → now 60 (gives more room)
RSI_LONG_IDEAL_MIN = 48    # Ideal zone
RSI_LONG_IDEAL_MAX = 55

# Short entry zone
RSI_SHORT_SWEET_MIN = 40   # Was 48 → now 40 (more room)
RSI_SHORT_SWEET_MAX = 55   # Was 65 → now 55 (not too late)
RSI_SHORT_IDEAL_MIN = 45
RSI_SHORT_IDEAL_MAX = 52

# === MACD Slope (15m only) ===
MACD_SLOPE_ATR_FACTOR_15M = 0.025  # Was 0.035 → now 0.025 (less strict)

# === Distance to Key Levels (15m based, RELAXED) ===
MAX_DIST_EMA20_ATR = 0.8      # Was 0.4 → now 0.8 (more practical)
IDEAL_DIST_EMA20_ATR = 0.3    # Was 0.15 → now 0.3 (achievable)
MAX_DIST_VWAP_ATR = 1.0       # Was 0.6 → now 1.0 (flexible)

# === Order Flow Analysis (15m based) ===
AGGRESSOR_BUY_RATIO_MIN = 0.60   # Was 0.65 → now 0.60 (60%+)
AGGRESSOR_SELL_RATIO_MAX = 0.40  # Sellers dominate when aggr_buy <= 40%


# === Order Flow Override (safety) ===
# If order-flow is strongly AGAINST the proposed direction, cap the total score to avoid "pretty chart, bad tape".
FLOW_OVERRIDE_ENABLED = True
FLOW_OVERRIDE_LONG_AGGR_MAX = 0.35   # <= 35% aggrBuy is hostile for LONG
FLOW_OVERRIDE_SHORT_AGGR_MIN = 0.65  # >= 65% aggrBuy is hostile for SHORT (i.e., sellers not in control)
FLOW_OVERRIDE_SCORE_CAP = 69         # cap score to keep it below ENTRY thresholds

# === Quality Score System (100 points total) ===
# All points from 15m + HTF ONLY
QUALITY_SCORE_WATCH = 60     # Minimum score to alert "WATCH"
QUALITY_SCORE_ENTRY = 75     # Was 80 → now 75 (more achievable)
QUALITY_SCORE_PERFECT = 90   # Was 95 → now 90 (realistic)

# === Direction Selection ===
MIN_SCORE_DIFF = 10  # Winning direction must be 10+ points ahead

# === Signal Timing ===
SIGNAL_COOLDOWN_SECONDS = 300  # 5min between same-direction signals

# === Telegram ===
MAX_TG_LEN = 3800
ERROR_NOTIFY_COOLDOWN = 120
SEND_STARTUP_MESSAGE = True
HEARTBEAT_MINUTES = 360  # Every 6h

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "htf-15m-radar-bot/3.1"})

_BINANCE_LAST_CALL_TS = 0.0
_BINANCE_BAN_UNTIL = 0.0


# ============================================================
# Helpers
# ============================================================
def utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def utc_ts(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def interval_seconds(tf: str) -> int:
    return {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400}[tf]

def floor_to_interval_ms(epoch_ms: int, tf: str) -> int:
    sec = interval_seconds(tf)
    return (epoch_ms // 1000 // sec) * sec * 1000

def last_closed_open_ms(tf: str) -> int:
    now_ms = int(time.time() * 1000)
    curr_open = floor_to_interval_ms(now_ms, tf)
    return curr_open - interval_seconds(tf) * 1000

def next_close_eta(tf: str) -> Tuple[str, float]:
    now_ms = int(time.time() * 1000)
    next_close_ms = floor_to_interval_ms(now_ms, tf) + interval_seconds(tf) * 1000
    return utc_ts(next_close_ms), max(0.0, (next_close_ms - now_ms) / 1000.0)

def safe_float(x, default=None):
    try:
        v = float(x)
        if pd.isna(v):
            return default
        return v
    except Exception:
        return default

def pct(a: float, b: float, default=float("nan")) -> float:
    if a is None or b is None:
        return default
    if not (b == b) or b == 0:
        return default
    return (a - b) / b * 100.0

def dist_atr(px: float, level: float, atr: float) -> float:
    if px is None or level is None or atr is None:
        return float("nan")
    if not (atr == atr) or atr <= 0:
        return float("nan")
    return abs(px - level) / atr

def _valid(*xs) -> bool:
    return all(x is not None and (x == x) for x in xs)  # x==x filters NaN


# ============================================================
# Binance REST (with safer rate limiting)
# ============================================================
def http_get(path: str, params: Dict[str, Any]) -> Union[Dict[str, Any], List[Any]]:
    global _BINANCE_LAST_CALL_TS, _BINANCE_BAN_UNTIL

    now = time.time()
    if now < _BINANCE_BAN_UNTIL:
        time.sleep(max(0.0, _BINANCE_BAN_UNTIL - now))
        now = time.time()

    wait = MIN_BINANCE_INTERVAL - (now - _BINANCE_LAST_CALL_TS)
    if wait > 0:
        time.sleep(wait)

    url = BINANCE_FAPI + path
    r = SESSION.get(url, params=params, timeout=HTTP_TIMEOUT)
    _BINANCE_LAST_CALL_TS = time.time()

    if r.status_code in (418, 429):
        retry_after = r.headers.get("Retry-After")
        backoff = int(retry_after) if (retry_after and str(retry_after).isdigit()) else BAN_BACKOFF_DEFAULT
        _BINANCE_BAN_UNTIL = time.time() + max(10, backoff)
        raise requests.HTTPError(f"{r.status_code} rate-limited (backoff {backoff}s)", response=r)

    r.raise_for_status()
    return r.json()

def fetch_klines_df(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    data = http_get("/fapi/v1/klines", {"symbol": symbol, "interval": interval, "limit": limit})
    df = pd.DataFrame(data, columns=[
        "ts", "open", "high", "low", "close", "volume",
        "close_ts", "qav", "trades", "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    df = df[["ts", "open", "high", "low", "close", "volume", "taker_buy_base"]].copy()
    for c in ["open", "high", "low", "close", "volume", "taker_buy_base"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce").astype("int64")
    return df

def fetch_premium_index(symbol: str) -> Dict[str, Any]:
    # Contains markPrice, indexPrice, lastFundingRate, etc.
    return http_get("/fapi/v1/premiumIndex", {"symbol": symbol})

def fetch_mark_price(symbol: str) -> Optional[float]:
    j = fetch_premium_index(symbol)
    return safe_float(j.get("markPrice"))

def fetch_open_interest(symbol: str) -> Optional[float]:
    j = http_get("/fapi/v1/openInterest", {"symbol": symbol})
    return safe_float(j.get("openInterest"))


# ============================================================
# Telegram
# ============================================================
def tg_api(method: str) -> str:
    return f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/{method}"

def tg_get_updates(offset: Optional[int]) -> dict:
    """
    Safer polling:
      - long-poll
      - only message updates
      - limit=1 reduces backlog burst / duplicates
    """
    params: Dict[str, Any] = {
        "timeout": TELEGRAM_POLL_TIMEOUT,
        "allowed_updates": ["message"],
        "limit": 1
    }
    if offset is not None:
        params["offset"] = offset

    r = SESSION.get(tg_api("getUpdates"), params=params, timeout=TG_HTTP_TIMEOUT)
    if r.status_code == 409:
        return {"ok": False, "conflict": True, "result": []}
    r.raise_for_status()
    return r.json()

def tg_send_message(text: str):
    if len(text) > MAX_TG_LEN:
        text = text[:MAX_TG_LEN]
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
    r = SESSION.post(tg_api("sendMessage"), json=payload, timeout=HTTP_TIMEOUT)
    r.raise_for_status()


# ============================================================
# Indicators (Enhanced with safe division)
# ============================================================
def _add_indicators_core(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dt"] = pd.to_datetime(out["ts"], unit="ms", utc=True)
    out = out.sort_values("dt").set_index("dt")

    out["ema20"] = ta.ema(out["close"], length=20)
    out["ema50"] = ta.ema(out["close"], length=50)
    out["ema200"] = ta.ema(out["close"], length=200)
    out["vwap"] = ta.vwap(out["high"], out["low"], out["close"], out["volume"])
    out["rsi14"] = ta.rsi(out["close"], length=14)

    macd = ta.macd(out["close"], fast=12, slow=26, signal=9)
    if macd is not None and not macd.empty:
        out["macd"] = macd.iloc[:, 0]
        out["macds"] = macd.iloc[:, 1]
        out["macdh"] = macd.iloc[:, 2]
    else:
        out["macd"] = pd.NA
        out["macds"] = pd.NA
        out["macdh"] = pd.NA

    out["atr14"] = ta.atr(out["high"], out["low"], out["close"], length=14)
    out["vol_sma20"] = out["volume"].rolling(20).mean()

    # Enhanced order flow (SAFE division)
    out["taker_buy"] = out["taker_buy_base"]
    out["taker_sell"] = out["volume"] - out["taker_buy"]

    # Safe aggressor ratio calculation
    out["aggressor_buy_ratio"] = out["taker_buy"] / out["volume"].replace(0, pd.NA)

    # Delta
    out["delta"] = out["taker_buy"] - out["taker_sell"]
    out["delta_cumsum_5"] = out["delta"].rolling(5).sum()

    out = out.reset_index(drop=True)
    return out

def add_indicators_1m(df: pd.DataFrame) -> pd.DataFrame:
    return _add_indicators_core(df)

def add_indicators_5m(df: pd.DataFrame) -> pd.DataFrame:
    return _add_indicators_core(df)

def add_indicators_15m(df: pd.DataFrame) -> pd.DataFrame:
    return _add_indicators_core(df)

def add_ema_pack(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ema20"] = ta.ema(out["close"], length=20)
    out["ema50"] = ta.ema(out["close"], length=50)
    out["ema200"] = ta.ema(out["close"], length=200)
    return out


# ============================================================
# Cache (Multi-timeframe)
# ============================================================
@dataclass
class TFCached:
    df: Optional[pd.DataFrame] = None
    last_closed_open_ms: int | None = None

class MarketCache:
    def __init__(self):
        self.data = {
            s: {
                "1m": TFCached(),
                "5m": TFCached(),
                "15m": TFCached(),
                "1h": TFCached(),
                "4h": TFCached()
            }
            for s in SYMBOLS
        }

    def refresh_1m_force(self, symbol: str):
        df = fetch_klines_df(symbol, "1m", OHLCV_LIMIT_1M)
        df = add_indicators_1m(df)
        self.data[symbol]["1m"] = TFCached(df=df, last_closed_open_ms=last_closed_open_ms("1m"))

    def refresh_1m_if_needed(self, symbol: str):
        expected = last_closed_open_ms("1m")
        cached = self.data[symbol]["1m"]
        if cached.df is None or cached.last_closed_open_ms != expected:
            self.refresh_1m_force(symbol)

    def refresh_5m_force(self, symbol: str):
        df = fetch_klines_df(symbol, "5m", OHLCV_LIMIT_5M)
        df = add_indicators_5m(df)
        self.data[symbol]["5m"] = TFCached(df=df, last_closed_open_ms=last_closed_open_ms("5m"))

    def refresh_5m_if_needed(self, symbol: str):
        expected = last_closed_open_ms("5m")
        cached = self.data[symbol]["5m"]
        if cached.df is None or cached.last_closed_open_ms != expected:
            self.refresh_5m_force(symbol)

    def refresh_15m_force(self, symbol: str):
        df = fetch_klines_df(symbol, "15m", OHLCV_LIMIT_15M)
        df = add_indicators_15m(df)
        self.data[symbol]["15m"] = TFCached(df=df, last_closed_open_ms=last_closed_open_ms("15m"))

    def refresh_15m_if_needed(self, symbol: str):
        expected = last_closed_open_ms("15m")
        cached = self.data[symbol]["15m"]
        if cached.df is None or cached.last_closed_open_ms != expected:
            self.refresh_15m_force(symbol)

    def refresh_htf_if_needed(self, symbol: str, tf: str):
        expected = last_closed_open_ms(tf)
        cached = self.data[symbol][tf]
        if cached.df is None or cached.last_closed_open_ms != expected:
            df = fetch_klines_df(symbol, tf, OHLCV_LIMIT_HTF)
            df = add_ema_pack(df)
            self.data[symbol][tf] = TFCached(df=df, last_closed_open_ms=expected)

    def ensure_context_ready(self, symbol: str):
        self.refresh_htf_if_needed(symbol, "1h")
        self.refresh_htf_if_needed(symbol, "4h")

    def get(self, symbol: str, tf: str) -> pd.DataFrame:
        df = self.data[symbol][tf].df
        if df is None:
            raise RuntimeError(f"Cache missing for {symbol} {tf}")
        return df


# ============================================================
# HTF Context Check
# ============================================================
def check_htf_alignment(df1h: pd.DataFrame, df4h: pd.DataFrame) -> Tuple[bool, bool, Dict[str, Any]]:
    """
    HTF check with trend strength
    Returns: (long_ok, short_ok, context_dict)
    """
    i1h = len(df1h) - 2
    i4h = len(df4h) - 2

    if i1h < 205 or i4h < 205:
        return False, False, {"error": "insufficient_data"}

    c1h = safe_float(df1h.loc[i1h, "close"])
    e20_1h = safe_float(df1h.loc[i1h, "ema20"])
    e50_1h = safe_float(df1h.loc[i1h, "ema50"])
    e200_1h = safe_float(df1h.loc[i1h, "ema200"])

    c4h = safe_float(df4h.loc[i4h, "close"])
    e20_4h = safe_float(df4h.loc[i4h, "ema20"])
    e50_4h = safe_float(df4h.loc[i4h, "ema50"])
    e200_4h = safe_float(df4h.loc[i4h, "ema200"])

    if not _valid(c1h, e200_1h, c4h, e200_4h):
        return False, False, {"error": "missing_data"}

    # Basic alignment
    above_1h = c1h > e200_1h
    above_4h = c4h > e200_4h

    # Trend strength (EMA stack)
    stack_bull_1h = _valid(e20_1h, e50_1h, e200_1h) and (e20_1h > e50_1h > e200_1h)
    stack_bull_4h = _valid(e20_4h, e50_4h, e200_4h) and (e20_4h > e50_4h > e200_4h)

    stack_bear_1h = _valid(e20_1h, e50_1h, e200_1h) and (e20_1h < e50_1h < e200_1h)
    stack_bear_4h = _valid(e20_4h, e50_4h, e200_4h) and (e20_4h < e50_4h < e200_4h)

    strength_1h = "STRONG" if (stack_bull_1h or stack_bear_1h) else "WEAK"
    strength_4h = "STRONG" if (stack_bull_4h or stack_bear_4h) else "WEAK"

    htf_long_ok = above_1h and above_4h
    htf_short_ok = (not above_1h) and (not above_4h)

    context = {
        "1h_above_200": above_1h,
        "4h_above_200": above_4h,
        "1h_strength": strength_1h,
        "4h_strength": strength_4h,
        "1h_stack_bull": stack_bull_1h,
        "4h_stack_bull": stack_bull_4h,
        "1h_stack_bear": stack_bear_1h,
        "4h_stack_bear": stack_bear_4h,
        "dist_1h_200": pct(c1h, e200_1h),
        "dist_4h_200": pct(c4h, e200_4h),
    }

    return htf_long_ok, htf_short_ok, context



# ============================================================
# HTF Regime Score (0-10) — granular context, not just YES/NO
# ============================================================
def compute_htf_regime_score(htf_ctx: Dict[str, Any], direction: str) -> int:
    """Return HTF regime score 0–10 for the given direction.
    - 9–10: strong trend (both TFs aligned + strong stack)
    - 6–8 : aligned but weaker / late trend
    - 3–5 : mixed / transition
    - 0–2 : chop / hostile for this direction
    """
    try:
        above_1h = bool(htf_ctx.get("1h_above_200"))
        above_4h = bool(htf_ctx.get("4h_above_200"))
        s1 = str(htf_ctx.get("1h_strength", "WEAK")).upper()
        s4 = str(htf_ctx.get("4h_strength", "WEAK")).upper()

        bull_stack_1h = bool(htf_ctx.get("1h_stack_bull"))
        bull_stack_4h = bool(htf_ctx.get("4h_stack_bull"))
        bear_stack_1h = bool(htf_ctx.get("1h_stack_bear"))
        bear_stack_4h = bool(htf_ctx.get("4h_stack_bear"))

        aligned_bull = above_1h and above_4h
        aligned_bear = (not above_1h) and (not above_4h)
        mixed = (above_1h != above_4h)

        # Base score by regime
        if direction == "LONG":
            if aligned_bull:
                score = 6
                score += 2 if s4 == "STRONG" else 0
                score += 1 if s1 == "STRONG" else 0
                score += 1 if (bull_stack_1h and bull_stack_4h) else 0
                return int(min(10, max(0, score)))
            if mixed:
                # transition: 4h up but 1h below is worse than 1h up but 4h below
                score = 4 if above_4h else 3
                score += 1 if s4 == "STRONG" else 0
                return int(min(5, max(0, score)))
            # aligned bear (hostile for long)
            score = 2
            score -= 1 if (bear_stack_1h and bear_stack_4h) else 0
            return int(min(2, max(0, score)))

        # SHORT
        if aligned_bear:
            score = 6
            score += 2 if s4 == "STRONG" else 0
            score += 1 if s1 == "STRONG" else 0
            score += 1 if (bear_stack_1h and bear_stack_4h) else 0
            return int(min(10, max(0, score)))
        if mixed:
            score = 4 if (not above_4h) else 3
            score += 1 if s4 == "STRONG" else 0
            return int(min(5, max(0, score)))
        # aligned bull (hostile for short)
        score = 2
        score -= 1 if (bull_stack_1h and bull_stack_4h) else 0
        return int(min(2, max(0, score)))
    except Exception:
        return 0

# ============================================================
# MTF Bonus Check (5m vs 15m alignment - OPTIONAL bonus)
# ============================================================
def check_mtf_bonus(df5m: pd.DataFrame, df15m: pd.DataFrame, direction: str) -> int:
    """
    Check if 5m aligns with 15m direction
    Returns bonus points (0-10)
    This is OPTIONAL - gives bonus but doesn't block signals
    """
    if not MTF_GIVES_BONUS:
        return 0

    i5m = len(df5m) - 2
    i15m = len(df15m) - 2

    if i5m < 25 or i15m < 25:
        return 0

    c5 = safe_float(df5m.loc[i5m, "close"])
    v5 = safe_float(df5m.loc[i5m, "vwap"])
    e20_5 = safe_float(df5m.loc[i5m, "ema20"])
    e50_5 = safe_float(df5m.loc[i5m, "ema50"])

    if not _valid(c5, v5, e20_5, e50_5):
        return 0

    bias_5m_bull = (c5 > v5) and (e20_5 > e50_5)
    bias_5m_bear = (c5 < v5) and (e20_5 < e50_5)

    if direction == "LONG" and bias_5m_bull:
        return MTF_BONUS_POINTS
    if direction == "SHORT" and bias_5m_bear:
        return MTF_BONUS_POINTS

    return 0


# ============================================================
# Quality Score Calculator (V3: 15M + HTF ONLY)
# ============================================================
def calculate_quality_score_v3(
    symbol: str,
    df15m: pd.DataFrame,
    df5m: pd.DataFrame,  # Only for MTF bonus
    df1h: pd.DataFrame,
    df4h: pd.DataFrame,
    direction: str
) -> Dict[str, Any]:
    """
    V3: Calculate quality score using ONLY 15m + HTF
    5m is used ONLY for optional MTF bonus
    """
    score = 0
    breakdown: Dict[str, Any] = {}

    i15m = len(df15m) - 2
    if i15m < 30:
        return {"score": 0, "breakdown": {}, "error": "insufficient_data"}

    # === 1. HTF Alignment (20 points) ===
    htf_long, htf_short, htf_ctx = check_htf_alignment(df1h, df4h)
    htf_regime = compute_htf_regime_score(htf_ctx, direction)
    htf_regime = compute_htf_regime_score(htf_ctx, direction)

    if direction == "LONG":
        if htf_long:
            if htf_ctx.get("1h_strength") == "STRONG" and htf_ctx.get("4h_strength") == "STRONG":
                score += 20
                breakdown["htf"] = 20
            else:
                score += 15
                breakdown["htf"] = 15
    elif direction == "SHORT":
        if htf_short:
            if htf_ctx.get("1h_strength") == "STRONG" and htf_ctx.get("4h_strength") == "STRONG":
                score += 20
                breakdown["htf"] = 20
            else:
                score += 15
                breakdown["htf"] = 15

    breakdown.setdefault("htf", 0)
    breakdown["htf_regime"] = htf_regime


    # === 2. 15m Momentum (25 points: RSI 15 + MACD 10) ===
    rsi_15m = safe_float(df15m.loc[i15m, "rsi14"])
    macdh_15m = safe_float(df15m.loc[i15m, "macdh"])
    macdh_15m_prev = safe_float(df15m.loc[i15m - 1, "macdh"]) if i15m > 0 else None
    atr_15m = safe_float(df15m.loc[i15m, "atr14"])

    # RSI scoring (15 points)
    rsi_score = 0
    if direction == "LONG":
        if _valid(rsi_15m) and RSI_LONG_IDEAL_MIN <= rsi_15m <= RSI_LONG_IDEAL_MAX:
            rsi_score = 15
        elif _valid(rsi_15m) and RSI_LONG_SWEET_MIN <= rsi_15m <= RSI_LONG_SWEET_MAX:
            rsi_score = 10
        elif _valid(rsi_15m) and 40 <= rsi_15m <= 65:
            rsi_score = 5
    elif direction == "SHORT":
        if _valid(rsi_15m) and RSI_SHORT_IDEAL_MIN <= rsi_15m <= RSI_SHORT_IDEAL_MAX:
            rsi_score = 15
        elif _valid(rsi_15m) and RSI_SHORT_SWEET_MIN <= rsi_15m <= RSI_SHORT_SWEET_MAX:
            rsi_score = 10
        elif _valid(rsi_15m) and 35 <= rsi_15m <= 60:
            rsi_score = 5

    score += rsi_score
    breakdown["rsi"] = rsi_score
    breakdown["rsi_15m"] = rsi_15m

    # MACD slope scoring (10 points)
    macd_score = 0
    if _valid(macdh_15m, macdh_15m_prev, atr_15m) and atr_15m > 0:
        slope_15m = macdh_15m - macdh_15m_prev
        slope_thr = atr_15m * MACD_SLOPE_ATR_FACTOR_15M

        if direction == "LONG":
            if slope_15m > slope_thr * 1.5:
                macd_score = 10
            elif slope_15m > slope_thr:
                macd_score = 7
            elif _valid(macdh_15m) and macdh_15m > 0:
                macd_score = 3
        elif direction == "SHORT":
            if slope_15m < -slope_thr * 1.5:
                macd_score = 10
            elif slope_15m < -slope_thr:
                macd_score = 7
            elif _valid(macdh_15m) and macdh_15m < 0:
                macd_score = 3

    score += macd_score
    breakdown["macd"] = macd_score
    breakdown["macdh_15m"] = macdh_15m

    # === 3. 15m Structure (25 points: Proximity 15 + Volume 10) ===
    px_15m = safe_float(df15m.loc[i15m, "close"])
    ema20_15m = safe_float(df15m.loc[i15m, "ema20"])
    vwap_15m = safe_float(df15m.loc[i15m, "vwap"])

    prox_score = 0
    if _valid(px_15m, ema20_15m, atr_15m) and atr_15m > 0:
        dist_ema20 = dist_atr(px_15m, ema20_15m, atr_15m)

        if dist_ema20 <= IDEAL_DIST_EMA20_ATR:
            prox_score += 10
        elif dist_ema20 <= MAX_DIST_EMA20_ATR:
            prox_score += 7
        elif dist_ema20 <= 1.2:
            prox_score += 3

        breakdown["dist_ema20_atr"] = dist_ema20

    # VWAP bonus (5 points)
    if _valid(px_15m, vwap_15m, atr_15m) and atr_15m > 0:
        dist_vwap = dist_atr(px_15m, vwap_15m, atr_15m)

        if direction == "LONG" and px_15m > vwap_15m:
            if dist_vwap <= 0.5:
                prox_score += 5
            elif dist_vwap <= MAX_DIST_VWAP_ATR:
                prox_score += 3
        elif direction == "SHORT" and px_15m < vwap_15m:
            if dist_vwap <= 0.5:
                prox_score += 5
            elif dist_vwap <= MAX_DIST_VWAP_ATR:
                prox_score += 3

        breakdown["dist_vwap_atr"] = dist_vwap

    prox_score = min(15, prox_score)
    score += prox_score
    breakdown["proximity"] = prox_score

    # Volume scoring (10 points)
    vol_15m = safe_float(df15m.loc[i15m, "volume"])
    vol_avg_15m = safe_float(df15m.loc[i15m, "vol_sma20"])
    volx_15m = (vol_15m / vol_avg_15m) if (_valid(vol_15m, vol_avg_15m) and vol_avg_15m > 0) else 0.0

    vol_score = 0
    if volx_15m >= VOL_PERFECT_MIN_15M:
        vol_score = 10
    elif volx_15m >= VOL_ENTRY_MIN_15M:
        vol_score = 7
    elif volx_15m >= VOL_WATCH_MIN_15M:
        vol_score = 4

    score += vol_score
    breakdown["volume"] = vol_score
    breakdown["volx_15m"] = volx_15m

    # === 4. 15m Order Flow (20 points) ===
    aggr_ratio_15m = safe_float(df15m.loc[i15m, "aggressor_buy_ratio"])
    delta_cumsum_15m = safe_float(df15m.loc[i15m, "delta_cumsum_5"])

    flow_score = 0
    if direction == "LONG":
        if _valid(aggr_ratio_15m) and aggr_ratio_15m >= 0.70:
            flow_score += 12
        elif _valid(aggr_ratio_15m) and aggr_ratio_15m >= AGGRESSOR_BUY_RATIO_MIN:
            flow_score += 8
        elif _valid(aggr_ratio_15m) and aggr_ratio_15m >= 0.55:
            flow_score += 4

        if _valid(delta_cumsum_15m) and delta_cumsum_15m > 0:
            flow_score += 8

    elif direction == "SHORT":
        if _valid(aggr_ratio_15m) and aggr_ratio_15m <= 0.30:
            flow_score += 12
        elif _valid(aggr_ratio_15m) and aggr_ratio_15m <= AGGRESSOR_SELL_RATIO_MAX:
            flow_score += 8
        elif _valid(aggr_ratio_15m) and aggr_ratio_15m <= 0.45:
            flow_score += 4

        if _valid(delta_cumsum_15m) and delta_cumsum_15m < 0:
            flow_score += 8

    flow_score = min(20, flow_score)
    score += flow_score
    breakdown["order_flow"] = flow_score
    breakdown["aggr_ratio_15m"] = aggr_ratio_15m


    # --- Order Flow Override (hostile tape -> cap score) ---
    flow_override = ""
    if FLOW_OVERRIDE_ENABLED:
        if direction == "LONG":
            if _valid(aggr_ratio_15m, delta_cumsum_15m) and (aggr_ratio_15m <= FLOW_OVERRIDE_LONG_AGGR_MAX) and (delta_cumsum_15m < 0):
                flow_override = "HOSTILE_LONG"
        elif direction == "SHORT":
            if _valid(aggr_ratio_15m, delta_cumsum_15m) and (aggr_ratio_15m >= FLOW_OVERRIDE_SHORT_AGGR_MIN) and (delta_cumsum_15m > 0):
                flow_override = "HOSTILE_SHORT"

    if flow_override:
        breakdown["flow_override"] = flow_override
        score = min(score, FLOW_OVERRIDE_SCORE_CAP)
    else:
        breakdown["flow_override"] = ""

    # === 5. MTF Bonus (10 points - OPTIONAL) ===
    mtf_bonus = check_mtf_bonus(df5m, df15m, direction)
    score += mtf_bonus
    breakdown["mtf_bonus"] = mtf_bonus

    return {
        "score": score,
        "breakdown": breakdown,
        "htf_context": htf_ctx,
    }


# ============================================================
# Signal Detection (V3: Best Direction Selection)
# ============================================================
def detect_leverage_opportunity_v3(symbol: str, cache: "MarketCache") -> Optional[Dict[str, Any]]:
    """
    V3: Calculate BOTH directions, select best one
    Returns signal dict or None
    """
    df15m = cache.get(symbol, "15m")
    df5m = cache.get(symbol, "5m")
    df1h = cache.get(symbol, "1h")
    df4h = cache.get(symbol, "4h")

    i15m = len(df15m) - 2
    if i15m < 30:
        return None

    long_quality = calculate_quality_score_v3(symbol, df15m, df5m, df1h, df4h, "LONG")
    short_quality = calculate_quality_score_v3(symbol, df15m, df5m, df1h, df4h, "SHORT")

    long_score = long_quality.get("score", 0)
    short_score = short_quality.get("score", 0)

    if long_score >= QUALITY_SCORE_WATCH and long_score > short_score + MIN_SCORE_DIFF:
        direction = "LONG"
        quality = long_quality
    elif short_score >= QUALITY_SCORE_WATCH and short_score > long_score + MIN_SCORE_DIFF:
        direction = "SHORT"
        quality = short_quality
    else:
        return None

    score = quality["score"]

    if score >= QUALITY_SCORE_PERFECT:
        signal_level = "PERFECT_ENTRY"
    elif score >= QUALITY_SCORE_ENTRY:
        signal_level = "ENTRY_READY"
    else:
        signal_level = "WATCH"

    px = safe_float(df15m.loc[i15m, "close"])

    # One call: premiumIndex gives mark, index, funding (used later in report too)
    prem = fetch_premium_index(symbol)
    mark = safe_float(prem.get("markPrice")) or px

    candle_ts_15m = int(df15m.loc[i15m, "ts"])

    return {
        "symbol": symbol,
        "direction": direction,
        "signal_level": signal_level,
        "quality_score": score,
        "quality_breakdown": quality["breakdown"],
        "htf_context": quality["htf_context"],
        "long_score": long_score,
        "short_score": short_score,
        "price": px,
        "mark_price": mark,
        "premium_index": prem,   # reuse in report
        "candle_ts_15m": candle_ts_15m,
        "timestamp": time.time(),
    }


# ============================================================
# Info Display Functions (5m/1m snapshot - NOT used for scoring)
# ============================================================
def get_5m_snapshot(df5m: pd.DataFrame) -> Dict[str, Any]:
    if df5m is None or len(df5m) < 25:
        return {}
    i5m = len(df5m) - 2

    vol = safe_float(df5m.loc[i5m, "volume"])
    vol_sma = safe_float(df5m.loc[i5m, "vol_sma20"])
    volx = (vol / vol_sma) if (_valid(vol, vol_sma) and vol_sma > 0) else 0.0

    return {
        "close": safe_float(df5m.loc[i5m, "close"]),
        "rsi": safe_float(df5m.loc[i5m, "rsi14"]),
        "volx": volx,
        "macdh": safe_float(df5m.loc[i5m, "macdh"]),
        "ema20": safe_float(df5m.loc[i5m, "ema20"]),
        "vwap": safe_float(df5m.loc[i5m, "vwap"]),
    }

def get_1m_snapshot(df1m: pd.DataFrame) -> Dict[str, Any]:
    if df1m is None or len(df1m) < 25:
        return {}
    i1m = len(df1m) - 2

    vol = safe_float(df1m.loc[i1m, "volume"])
    vol_sma = safe_float(df1m.loc[i1m, "vol_sma20"])
    volx = (vol / vol_sma) if (_valid(vol, vol_sma) and vol_sma > 0) else 0.0

    return {
        "close": safe_float(df1m.loc[i1m, "close"]),
        "rsi": safe_float(df1m.loc[i1m, "rsi14"]),
        "volx": volx,
        "macdh": safe_float(df1m.loc[i1m, "macdh"]),
    }


# ============================================================
# Report Generation (V3.1: richer parameters + futures context)
# ============================================================
def generate_leverage_report_v3(signal: Dict[str, Any], cache: "MarketCache") -> str:
    symbol = signal["symbol"]
    direction = signal["direction"]
    level = signal["signal_level"]
    score = signal["quality_score"]
    breakdown = signal["quality_breakdown"]
    htf_ctx = signal["htf_context"]
    long_score = signal["long_score"]
    short_score = signal["short_score"]

    df15m = cache.get(symbol, "15m")
    i15m = len(df15m) - 2

    # 15m core numbers
    ema20 = safe_float(df15m.loc[i15m, "ema20"])
    ema50 = safe_float(df15m.loc[i15m, "ema50"])
    ema200 = safe_float(df15m.loc[i15m, "ema200"])
    vwap = safe_float(df15m.loc[i15m, "vwap"])
    atr = safe_float(df15m.loc[i15m, "atr14"])
    rsi = safe_float(df15m.loc[i15m, "rsi14"])
    macdh = safe_float(df15m.loc[i15m, "macdh"])
    macdh_prev = safe_float(df15m.loc[i15m - 1, "macdh"]) if i15m > 0 else None
    macd_slope = (macdh - macdh_prev) if (_valid(macdh, macdh_prev)) else None
    vol = safe_float(df15m.loc[i15m, "volume"])
    vol_sma = safe_float(df15m.loc[i15m, "vol_sma20"])
    volx = (vol / vol_sma) if (_valid(vol, vol_sma) and vol_sma > 0) else 0.0
    delta = safe_float(df15m.loc[i15m, "delta"])
    delta5 = safe_float(df15m.loc[i15m, "delta_cumsum_5"])
    aggr = safe_float(df15m.loc[i15m, "aggressor_buy_ratio"])

    # Futures context
    prem = signal.get("premium_index") or fetch_premium_index(symbol)
    mark = safe_float(prem.get("markPrice"))
    indexp = safe_float(prem.get("indexPrice"))
    funding = safe_float(prem.get("lastFundingRate"))
    basis = ((mark - indexp) / indexp * 100.0) if (_valid(mark, indexp) and indexp != 0) else None
    oi = None
    try:
        oi = fetch_open_interest(symbol)
    except Exception:
        oi = None

    # Info snapshots
    df5m = cache.get(symbol, "5m")
    df1m = cache.get(symbol, "1m")
    snap_5m = get_5m_snapshot(df5m)
    snap_1m = get_1m_snapshot(df1m)

    emoji = {"PERFECT_ENTRY": "🚀💎", "ENTRY_READY": "✅🎯", "WATCH": "👀⚠️"}.get(level, "📊")

    macd_slope_str = f"{macd_slope:.2f}" if macd_slope is not None else "N/A"
    basis_str = f"{basis:.3f}%" if basis is not None else "N/A"
    funding_str = f"{funding:.4%}" if funding is not None else "N/A"
    oi_str = f"{oi:.0f}" if oi is not None else "N/A"

    lines = [
        f"{emoji} {level} | {symbol} {direction}",
        f"Quality Score: {score}/100 (15m-based)",
        f"Competing scores: LONG {long_score} | SHORT {short_score}",
        f"Time: {utc_now_str()}",
        "=" * 60,
        "",
        f"Price(15m close): {signal['price']:.2f} | Mark: {signal['mark_price']:.2f}",
        "",
        "📉 DERIV (futures):",
        f"  Mark: {mark:.2f} | Index: {indexp:.2f} | Basis: {basis_str}" if _valid(mark, indexp) else "  Mark/Index: N/A",
        f"  Funding: {funding_str} | OI: {oi_str}",
        "",
        "📊 QUALITY BREAKDOWN (15m + HTF):",
        f"  HTF Alignment: {breakdown.get('htf', 0)}/20",
        f"  HTF Regime: {breakdown.get('htf_regime', 0)}/10",
        f"  15m RSI: {breakdown.get('rsi', 0)}/15",
        f"  15m MACD: {breakdown.get('macd', 0)}/10",
        f"  15m Proximity: {breakdown.get('proximity', 0)}/15",
        f"  15m Volume: {breakdown.get('volume', 0)}/10",
        f"  15m Order Flow: {breakdown.get('order_flow', 0)}/20",
        f"  Flow Override: {breakdown.get('flow_override', '') or '—'}",
        f"  MTF Bonus (5m): {breakdown.get('mtf_bonus', 0)}/10",
        "",
        "🌍 HTF CONTEXT:",
        f"  1h: {'ABOVE' if htf_ctx.get('1h_above_200') else 'BELOW'} EMA200 ({htf_ctx.get('1h_strength', 'N/A')})",
        f"  4h: {'ABOVE' if htf_ctx.get('4h_above_200') else 'BELOW'} EMA200 ({htf_ctx.get('4h_strength', 'N/A')})",
        f"  1h Dist: {htf_ctx.get('dist_1h_200', 0):.2f}%",
        f"  4h Dist: {htf_ctx.get('dist_4h_200', 0):.2f}%",
        "",
        "📌 15m CORE (numbers):",
        f"  EMA20/50/200: {ema20:.2f} | {ema50:.2f} | {ema200:.2f}",
        f"  VWAP: {vwap:.2f} | ATR14: {atr:.2f}",
        f"  RSI14: {rsi:.1f} | MACDH: {macdh:.2f} | slope: {macd_slope_str}",
        f"  VOL: {vol:.2f} | SMA20: {vol_sma:.2f} | VOLx: {volx:.2f}",
        f"  AggrBuy: {aggr:.1%} | Delta: {delta:.2f} | Δ5: {delta5:.2f}",
        "",
        "📊 5m INFO (display only - not scored):",
        f"  Close: {snap_5m.get('close', 0):.2f}",
        f"  RSI: {snap_5m.get('rsi', 0):.1f}",
        f"  Volume: {snap_5m.get('volx', 0):.2f}x",
        f"  EMA20: {snap_5m.get('ema20', 0):.2f}",
        f"  VWAP: {snap_5m.get('vwap', 0):.2f}",
        "",
        "⚡ 1m INFO (display only - not scored):",
        f"  Close: {snap_1m.get('close', 0):.2f}",
        f"  RSI: {snap_1m.get('rsi', 0):.1f}",
        f"  Volume: {snap_1m.get('volx', 0):.2f}x",
        "",
        "=" * 60,
        "🤖 FOR GPT ANALYSIS:",
        "",
        f"This is a {level} signal for {symbol} {direction}.",
        f"Quality score: {score}/100 (based on 15m + HTF only)",
        "",
        "Key questions:",
        "1) Does this 15m setup have good entry timing?",
        "2) How does HTF context support this direction?",
        "3) Is the score high enough for leverage entry?",
        "4) What are the key risks?",
        "5) Suggested entry zones and stop-loss?",
    ]

    # --- 10s YES/NO checklist gate (per-symbol thresholds) ---
    try:
        check = compute_10s_checklist(symbol, direction, cache, signal)
        idx = None
        for j, ln in enumerate(lines):
            if ln == "🤖 FOR GPT ANALYSIS:":
                idx = j
                break
        if idx is None:
            lines.append("")
            lines.append(format_checklist_block(check))
        else:
            lines.insert(idx, "")
            lines.insert(idx + 1, format_checklist_block(check))
            lines.insert(idx + 2, "")
    except Exception:
        pass

    return "\n".join(lines)


def generate_manual_report_v3(symbol: str, cache: "MarketCache") -> str:
    """V3: Manual report with comprehensive data (and richer parameters)"""
    cache.refresh_15m_force(symbol)
    cache.refresh_5m_force(symbol)
    cache.refresh_1m_force(symbol)
    cache.ensure_context_ready(symbol)

    signal = detect_leverage_opportunity_v3(symbol, cache)
    if signal:
        return generate_leverage_report_v3(signal, cache)

    df15m = cache.get(symbol, "15m")
    df5m = cache.get(symbol, "5m")
    df1m = cache.get(symbol, "1m")
    df1h = cache.get(symbol, "1h")
    df4h = cache.get(symbol, "4h")

    i15m = len(df15m) - 2
    px = safe_float(df15m.loc[i15m, "close"])

    prem = fetch_premium_index(symbol)
    mark = safe_float(prem.get("markPrice")) or px
    indexp = safe_float(prem.get("indexPrice"))
    funding = safe_float(prem.get("lastFundingRate"))
    basis = ((mark - indexp) / indexp * 100.0) if (_valid(mark, indexp) and indexp != 0) else None
    try:
        oi = fetch_open_interest(symbol)
    except Exception:
        oi = None

    long_q = calculate_quality_score_v3(symbol, df15m, df5m, df1h, df4h, "LONG")
    short_q = calculate_quality_score_v3(symbol, df15m, df5m, df1h, df4h, "SHORT")

    long_score = long_q.get("score", 0)
    short_score = short_q.get("score", 0)

    snap_5m = get_5m_snapshot(df5m)
    snap_1m = get_1m_snapshot(df1m)

    rsi_15m = safe_float(df15m.loc[i15m, "rsi14"])
    atr_15m = safe_float(df15m.loc[i15m, "atr14"])
    ema20_15m = safe_float(df15m.loc[i15m, "ema20"])
    ema50_15m = safe_float(df15m.loc[i15m, "ema50"])
    ema200_15m = safe_float(df15m.loc[i15m, "ema200"])
    vwap_15m = safe_float(df15m.loc[i15m, "vwap"])
    macdh = safe_float(df15m.loc[i15m, "macdh"])
    macdh_prev = safe_float(df15m.loc[i15m - 1, "macdh"]) if i15m > 0 else None
    macd_slope = (macdh - macdh_prev) if (_valid(macdh, macdh_prev)) else None

    vol_15m = safe_float(df15m.loc[i15m, "volume"])
    vol_avg_15m = safe_float(df15m.loc[i15m, "vol_sma20"])
    volx_15m = (vol_15m / vol_avg_15m) if (_valid(vol_15m, vol_avg_15m) and vol_avg_15m > 0) else 0.0

    aggr = safe_float(df15m.loc[i15m, "aggressor_buy_ratio"])
    delta = safe_float(df15m.loc[i15m, "delta"])
    delta5 = safe_float(df15m.loc[i15m, "delta_cumsum_5"])

    _, _, htf_ctx = check_htf_alignment(df1h, df4h)

    basis_str = f"{basis:.3f}%" if basis is not None else "N/A"
    funding_str = f"{funding:.4%}" if funding is not None else "N/A"
    oi_str = f"{oi:.0f}" if oi is not None else "N/A"
    macd_slope_str = f"{macd_slope:.2f}" if macd_slope is not None else "N/A"

    status_line = "⚠️ No high-quality setup (need ≥60 & diff≥10)" if max(long_score, short_score) < QUALITY_SCORE_WATCH else "✅ Setup detected but scores too close (chop)"


    # --- 10s YES/NO checklist gate (manual context) ---
    direction_best = "LONG" if long_score > short_score else ("SHORT" if short_score > long_score else "LONG")
    pseudo_signal = {"quality_score": max(long_score, short_score), "long_score": long_score, "short_score": short_score}
    try:
        _chk = compute_10s_checklist(symbol, direction_best, cache, pseudo_signal)
        _chk_txt = format_checklist_block(_chk)
    except Exception:
        _chk_txt = "⏱ 10s CHECKLIST: N/A"



    return (
        f"📊 {symbol} COMPREHENSIVE STATUS (v3.1)\n"
        f"Time: {utc_now_str()}\n"
        f"Price: {px:.2f} | Mark: {mark:.2f}\n"
        f"{'=' * 60}\n\n"
        f"📉 DERIV (futures):\n"
        f"  Mark: {mark:.2f} | Index: {indexp:.2f} | Basis: {basis_str}\n"
        f"  Funding: {funding_str} | OI: {oi_str}\n\n"
        f"🎯 QUALITY SCORES (15m + HTF based):\n"
        f"  LONG:  {long_score}/100\n"
        f"  SHORT: {short_score}/100\n"
        f"  Status: {status_line}\n\n"
        f"🌍 HTF CONTEXT:\n"
        f"  1h: {'ABOVE' if htf_ctx.get('1h_above_200') else 'BELOW'} EMA200 ({htf_ctx.get('1h_strength')})\n"
        f"  4h: {'ABOVE' if htf_ctx.get('4h_above_200') else 'BELOW'} EMA200 ({htf_ctx.get('4h_strength')})\n\n"
        f"📌 15m CORE (numbers):\n"
        f"  EMA20/50/200: {ema20_15m:.2f} | {ema50_15m:.2f} | {ema200_15m:.2f}\n"
        f"  VWAP: {vwap_15m:.2f} | ATR14: {atr_15m:.2f}\n"
        f"  RSI14: {rsi_15m:.1f} | MACDH: {macdh:.2f} | slope: {macd_slope_str}\n"
        f"  VOL: {vol_15m:.2f} | SMA20: {vol_avg_15m:.2f} | VOLx: {volx_15m:.2f}\n"
        f"  AggrBuy: {aggr:.1%} | Delta: {delta:.2f} | Δ5: {delta5:.2f}\n\n"
        f"📊 5m INFO (display only):\n"
        f"  Close: {snap_5m.get('close', 0):.2f} | RSI: {snap_5m.get('rsi', 0):.1f} | VOLx: {snap_5m.get('volx', 0):.2f}x\n"
        f"  EMA20: {snap_5m.get('ema20', 0):.2f} | VWAP: {snap_5m.get('vwap', 0):.2f}\n\n"
        f"⚡ 1m INFO (display only):\n"
        f"  Close: {snap_1m.get('close', 0):.2f} | RSI: {snap_1m.get('rsi', 0):.1f} | VOLx: {snap_1m.get('volx', 0):.2f}x\n\n"
        f"💡 SCORE BREAKDOWN:\n"
        f"  HTF: {long_q['breakdown'].get('htf', 0)}/20 (L) | {short_q['breakdown'].get('htf', 0)}/20 (S)\n"
        f"  Regime: {long_q['breakdown'].get('htf_regime', 0)}/10 (L) | {short_q['breakdown'].get('htf_regime', 0)}/10 (S)\n"
        f"  RSI: {long_q['breakdown'].get('rsi', 0)}/15 (L) | {short_q['breakdown'].get('rsi', 0)}/15 (S)\n"
        f"  MACD: {long_q['breakdown'].get('macd', 0)}/10 (L) | {short_q['breakdown'].get('macd', 0)}/10 (S)\n"
        f"  Proximity: {long_q['breakdown'].get('proximity', 0)}/15 (L) | {short_q['breakdown'].get('proximity', 0)}/15 (S)\n"
        f"  Volume: {long_q['breakdown'].get('volume', 0)}/10 (L) | {short_q['breakdown'].get('volume', 0)}/10 (S)\n"
        f"  Flow: {long_q['breakdown'].get('order_flow', 0)}/20 (L) | {short_q['breakdown'].get('order_flow', 0)}/20 (S)\n"
        f"  Override: {long_q['breakdown'].get('flow_override', '') or '—'} (L) | {short_q['breakdown'].get('flow_override', '') or '—'} (S)\n"
        f"  MTF: {long_q['breakdown'].get('mtf_bonus', 0)}/10 (L) | {short_q['breakdown'].get('mtf_bonus', 0)}/10 (S)\n\n"
        f"{_chk_txt}\n\n"f"{'=' * 60}\n"
        f"🤖 FOR GPT ANALYSIS:\n\n"
        f"Scores are below trigger or too close.\n"
        f"What needs to change on 15m for a clean signal (≥{QUALITY_SCORE_WATCH} and diff≥{MIN_SCORE_DIFF})?\n"
    )


# ============================================================
# Signal History & Cooldown
# ============================================================
@dataclass
class SignalHistory:
    last_signal_time: float = 0.0
    last_direction: Optional[str] = None
    last_level: Optional[str] = None

    def should_alert(self, direction: str, level: str, current_time: float) -> bool:
        if self.last_signal_time == 0:
            return True

        if direction != self.last_direction:
            return True

        time_since_last = current_time - self.last_signal_time

        level_order = {"WATCH": 0, "ENTRY_READY": 1, "PERFECT_ENTRY": 2}
        if level_order.get(level, 0) > level_order.get(self.last_level, 0):
            return True

        if time_since_last < SIGNAL_COOLDOWN_SECONDS:
            return False

        return True

    def update(self, direction: str, level: str, current_time: float):
        self.last_signal_time = current_time
        self.last_direction = direction
        self.last_level = level


# ============================================================
# Command Parsing
# ============================================================
# ============================================================
# COMPACT /report ALL (single Telegram message, table format)
# ============================================================

def _fmt(x: object, w: int = 12) -> str:
    s = str(x)
    if len(s) > w:
        s = s[:w]
    return s.rjust(w)

def _fmt_f(x: Optional[float], w: int = 12, d: int = 2, na: str = "N/A") -> str:
    if x is None or not (x == x):
        return na.rjust(w)
    return f"{x:.{d}f}".rjust(w)

def _fmt_pct(x: Optional[float], w: int = 12, d: int = 3, na: str = "N/A") -> str:
    if x is None or not (x == x):
        return na.rjust(w)
    return f"{x:.{d}f}%".rjust(w)

def _fmt_ratio(x: Optional[float], w: int = 12, d: int = 1, na: str = "N/A") -> str:
    if x is None or not (x == x):
        return na.rjust(w)
    return f"{x*100:.{d}f}%".rjust(w)

def _fmt_int(x: Optional[float], w: int = 12, na: str = "N/A") -> str:
    if x is None or not (x == x):
        return na.rjust(w)
    return f"{int(round(x))}".rjust(w)

def _price_decimals(px: Optional[float]) -> int:
    if px is None or not (px == px):
        return 2
    if px >= 10000:
        return 1
    if px >= 1000:
        return 2
    return 2


def _dist_atr(a: float, b: float, atr: float):
    if a is None or b is None or atr is None or atr == 0:
        return None
    return abs(a - b) / atr

def _range_high_low(df, bars: int):
    if df is None or len(df) < bars + 2:
        return (None, None)
    window = df.iloc[-(bars+1):-1]
    return float(window['high'].max()), float(window['low'].min())


def build_compact_all_report(cache: "MarketCache") -> str:
    """One Telegram message: rows=metrics, cols=BTC/ETH/SOL (keeps core signal context)."""
    snap: Dict[str, Dict[str, object]] = {}

    for sym in SYMBOLS:
        cache.refresh_15m_force(sym)
        cache.refresh_5m_force(sym)
        cache.refresh_1m_force(sym)
        cache.ensure_context_ready(sym)

        df15 = cache.get(sym, "15m")
        df5 = cache.get(sym, "5m")
        df1 = cache.get(sym, "1m")
        df1h = cache.get(sym, "1h")
        df4h = cache.get(sym, "4h")

        i15 = len(df15) - 2  # LAST CLOSED candle
        px = safe_float(df15.loc[i15, "close"])

        # Derivatives
        prem = fetch_premium_index(sym)
        mark = safe_float(prem.get("markPrice")) or px
        indexp = safe_float(prem.get("indexPrice"))
        funding = safe_float(prem.get("lastFundingRate"))
        basis = ((mark - indexp) / indexp * 100.0) if (_valid(mark, indexp) and indexp != 0) else None
        try:
            oi = fetch_open_interest(sym)
            oi_prev = cache.get_prev_oi(sym)
            doi = (oi - oi_prev) if (oi is not None and oi_prev is not None) else None
        except Exception:
            oi = None
            doi = None

        # Scores (15m+HTF)
        long_q = calculate_quality_score_v3(sym, df15, df5, df1h, df4h, "LONG")
        short_q = calculate_quality_score_v3(sym, df15, df5, df1h, df4h, "SHORT")
        L = int(long_q.get("score", 0))
        S = int(short_q.get("score", 0))
        direction = "LONG" if L > S else ("SHORT" if S > L else "MIX")
        diff = abs(L - S)

        if max(L, S) >= QUALITY_SCORE_ENTRY and diff >= MIN_SCORE_DIFF:
            level = "ENTRY"
        elif max(L, S) >= QUALITY_SCORE_WATCH and diff >= MIN_SCORE_DIFF:
            level = "WATCH"
        else:
            level = "NO"

        # HTF
        _, _, htf = check_htf_alignment(df1h, df4h)
        h1 = ("ABV" if htf.get("1h_above_200") else "BLW") + f"({(htf.get('1h_strength') or 'N/A')[0]})"
        h4 = ("ABV" if htf.get("4h_above_200") else "BLW") + f"({(htf.get('4h_strength') or 'N/A')[0]})"

        # 15m core
        ema20 = safe_float(df15.loc[i15, "ema20"])
        ema50 = safe_float(df15.loc[i15, "ema50"])
        ema200 = safe_float(df15.loc[i15, "ema200"])
        vwap = safe_float(df15.loc[i15, "vwap"])
        atr = safe_float(df15.loc[i15, "atr14"])
        rsi = safe_float(df15.loc[i15, "rsi14"])
        macdh = safe_float(df15.loc[i15, "macdh"])
        macdh_prev = safe_float(df15.loc[i15 - 1, "macdh"]) if i15 > 0 else None
        macd_slope = (macdh - macdh_prev) if (_valid(macdh, macdh_prev)) else None

        vol = safe_float(df15.loc[i15, "volume"])
        vol_sma = safe_float(df15.loc[i15, "vol_sma20"])
        volx = (vol / vol_sma) if (_valid(vol, vol_sma) and vol_sma > 0) else None

        aggr = safe_float(df15.loc[i15, "aggressor_buy_ratio"])
        delta = safe_float(df15.loc[i15, "delta"])
        delta5 = safe_float(df15.loc[i15, "delta_cumsum_5"])

        # MTF snapshots
        s5 = get_5m_snapshot(df5)
        s1 = get_1m_snapshot(df1)

        snap[sym] = {
            "level": level, "dir": direction, "L": L, "S": S,
            "px": px, "mark": mark, "basis": basis, "funding": funding, "oi": oi,
            "h1": h1, "h4": h4,
            "ema20": ema20, "ema50": ema50, "ema200": ema200,
            "vwap": vwap, "atr": atr,
            "rsi": rsi, "macdh": macdh, "slope": macd_slope,
            "volx": volx, "aggr": aggr, "delta": delta, "delta5": delta5,
            "rsi5": s5.get("rsi"), "volx5": s5.get("volx"),
            "rsi1": s1.get("rsi"), "volx1": s1.get("volx"),
        }

    W = 20

    def cell(value: str) -> str:
        return _fmt(value, W)

    def f(sym: str, key: str, d: int = 2) -> str:
        return _fmt_f(snap[sym].get(key), W, d)

    def i(sym: str, key: str) -> str:
        return _fmt_int(snap[sym].get(key), W)

    def ratio(sym: str, key: str) -> str:
        return _fmt_ratio(snap[sym].get(key), W, 1)

    lines: list[str] = []
    lines.append("```")
    lines.append(f"MANUAL REPORT (ALL) | UTC: {utc_now_str()}")
    hdr = f"{'PARAM'.ljust(14)}|{'BTC'.rjust(W)}|{'ETH'.rjust(W)}|{'SOL'.rjust(W)}"
    sep = "-" * len(hdr)
    lines.append(sep)
    lines.append(hdr)
    lines.append(sep)

    btc_lvl = f"{snap['BTCUSDT']['level']} {snap['BTCUSDT']['dir']}({snap['BTCUSDT']['L']}-{snap['BTCUSDT']['S']})"
    eth_lvl = f"{snap['ETHUSDT']['level']} {snap['ETHUSDT']['dir']}({snap['ETHUSDT']['L']}-{snap['ETHUSDT']['S']})"
    sol_lvl = f"{snap['SOLUSDT']['level']} {snap['SOLUSDT']['dir']}({snap['SOLUSDT']['L']}-{snap['SOLUSDT']['S']})"
    lines.append(f"{'Lvl/Dir(L-S)'.ljust(14)}|{cell(btc_lvl)}|{cell(eth_lvl)}|{cell(sol_lvl)}")

    btc_pm = f"{float(snap['BTCUSDT']['px']):.1f}/{float(snap['BTCUSDT']['mark']):.1f}"
    eth_pm = f"{float(snap['ETHUSDT']['px']):.2f}/{float(snap['ETHUSDT']['mark']):.2f}"
    sol_pm = f"{float(snap['SOLUSDT']['px']):.2f}/{float(snap['SOLUSDT']['mark']):.2f}"
    lines.append(f"{'Price / Mark'.ljust(14)}|{cell(btc_pm)}|{cell(eth_pm)}|{cell(sol_pm)}")

    btc_bf = f"{(snap['BTCUSDT']['basis'] or 0):+.3f}%/{(snap['BTCUSDT']['funding'] or 0)*100:.4f}%"
    eth_bf = f"{(snap['ETHUSDT']['basis'] or 0):+.3f}%/{(snap['ETHUSDT']['funding'] or 0)*100:.4f}%"
    sol_bf = f"{(snap['SOLUSDT']['basis'] or 0):+.3f}%/{(snap['SOLUSDT']['funding'] or 0)*100:.4f}%"
    lines.append(f"{'Basis / Fund'.ljust(14)}|{cell(btc_bf)}|{cell(eth_bf)}|{cell(sol_bf)}")

    lines.append(f"{'OI'.ljust(14)}|{i('BTCUSDT','oi')}|{i('ETHUSDT','oi')}|{i('SOLUSDT','oi')}")

    btc_htf = f"{snap['BTCUSDT']['h1']}/{snap['BTCUSDT']['h4']}"
    eth_htf = f"{snap['ETHUSDT']['h1']}/{snap['ETHUSDT']['h4']}"
    sol_htf = f"{snap['SOLUSDT']['h1']}/{snap['SOLUSDT']['h4']}"
    lines.append(f"{'HTF 1h / 4h'.ljust(14)}|{cell(btc_htf)}|{cell(eth_htf)}|{cell(sol_htf)}")

    lines.append(sep)

    btc_emas = f"{float(snap['BTCUSDT']['ema20']):.0f}/{float(snap['BTCUSDT']['ema50']):.0f}/{float(snap['BTCUSDT']['ema200']):.0f}"
    eth_emas = f"{float(snap['ETHUSDT']['ema20']):.1f}/{float(snap['ETHUSDT']['ema50']):.1f}/{float(snap['ETHUSDT']['ema200']):.1f}"
    sol_emas = f"{float(snap['SOLUSDT']['ema20']):.2f}/{float(snap['SOLUSDT']['ema50']):.2f}/{float(snap['SOLUSDT']['ema200']):.2f}"
    lines.append(f"{'EMA20/50/200'.ljust(14)}|{cell(btc_emas)}|{cell(eth_emas)}|{cell(sol_emas)}")

    btc_va = f"{float(snap['BTCUSDT']['vwap']):.0f}/{float(snap['BTCUSDT']['atr']):.0f}"
    eth_va = f"{float(snap['ETHUSDT']['vwap']):.1f}/{float(snap['ETHUSDT']['atr']):.1f}"
    sol_va = f"{float(snap['SOLUSDT']['vwap']):.2f}/{float(snap['SOLUSDT']['atr']):.2f}"
    lines.append(f"{'VWAP / ATR'.ljust(14)}|{cell(btc_va)}|{cell(eth_va)}|{cell(sol_va)}")

    btc_rm = f"{float(snap['BTCUSDT']['rsi']):.1f}/{float(snap['BTCUSDT']['macdh']):.2f}"
    eth_rm = f"{float(snap['ETHUSDT']['rsi']):.1f}/{float(snap['ETHUSDT']['macdh']):.2f}"
    sol_rm = f"{float(snap['SOLUSDT']['rsi']):.1f}/{float(snap['SOLUSDT']['macdh']):.2f}"
    lines.append(f"{'RSI / MACDH'.ljust(14)}|{cell(btc_rm)}|{cell(eth_rm)}|{cell(sol_rm)}")

    lines.append(f"{'MACD slope'.ljust(14)}|{f('BTCUSDT','slope',2)}|{f('ETHUSDT','slope',2)}|{f('SOLUSDT','slope',2)}")
    lines.append(f"{'VOLx'.ljust(14)}|{f('BTCUSDT','volx',2)}|{f('ETHUSDT','volx',2)}|{f('SOLUSDT','volx',2)}")
    lines.append(f"{'AggrBuy'.ljust(14)}|{ratio('BTCUSDT','aggr')}|{ratio('ETHUSDT','aggr')}|{ratio('SOLUSDT','aggr')}")

    btc_d = f"{float(snap['BTCUSDT']['delta']):.0f}/{float(snap['BTCUSDT']['delta5']):.0f}"
    eth_d = f"{float(snap['ETHUSDT']['delta']):.0f}/{float(snap['ETHUSDT']['delta5']):.0f}"
    sol_d = f"{float(snap['SOLUSDT']['delta']):.0f}/{float(snap['SOLUSDT']['delta5']):.0f}"
    lines.append(f"{'Delta / Δ5'.ljust(14)}|{cell(btc_d)}|{cell(eth_d)}|{cell(sol_d)}")

    lines.append(sep)

    btc_5 = f"{float(snap['BTCUSDT']['rsi5'] or 0):.1f}/{float(snap['BTCUSDT']['volx5'] or 0):.2f}x"
    eth_5 = f"{float(snap['ETHUSDT']['rsi5'] or 0):.1f}/{float(snap['ETHUSDT']['volx5'] or 0):.2f}x"
    sol_5 = f"{float(snap['SOLUSDT']['rsi5'] or 0):.1f}/{float(snap['SOLUSDT']['volx5'] or 0):.2f}x"
    lines.append(f"{'5m RSI / VOLx'.ljust(14)}|{cell(btc_5)}|{cell(eth_5)}|{cell(sol_5)}")

    btc_1 = f"{float(snap['BTCUSDT']['rsi1'] or 0):.1f}/{float(snap['BTCUSDT']['volx1'] or 0):.2f}x"
    eth_1 = f"{float(snap['ETHUSDT']['rsi1'] or 0):.1f}/{float(snap['ETHUSDT']['volx1'] or 0):.2f}x"
    sol_1 = f"{float(snap['SOLUSDT']['rsi1'] or 0):.1f}/{float(snap['SOLUSDT']['volx1'] or 0):.2f}x"
    lines.append(f"{'1m RSI / VOLx'.ljust(14)}|{cell(btc_1)}|{cell(eth_1)}|{cell(sol_1)}")

    # 10s checklist summary (YES count) — per symbol thresholds
    chk_map: Dict[str, Dict[str, Any]] = {}
    try:
        for sym_key in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
            d = snap[sym_key]["dir"] if snap[sym_key]["dir"] in ("LONG", "SHORT") else "LONG"
            pseudo = {
                "quality_score": max(snap[sym_key]["L"], snap[sym_key]["S"]),
                "long_score": snap[sym_key]["L"],
                "short_score": snap[sym_key]["S"],
            }
            chk_map[sym_key] = compute_10s_checklist(sym_key, d, cache, pseudo)
        btc_c = f"{chk_map['BTCUSDT'].get('yes', 0)}/10"
        eth_c = f"{chk_map['ETHUSDT'].get('yes', 0)}/10"
        sol_c = f"{chk_map['SOLUSDT'].get('yes', 0)}/10"
    except Exception:
        chk_map = {}
        btc_c = eth_c = sol_c = "N/A"

    lines.append(f"{'Checklist Y/N'.ljust(14)}|{cell(btc_c)}|{cell(eth_c)}|{cell(sol_c)}")

    lines.append("```")

    # 10s checklist details (named items)
    if isinstance(chk_map, dict) and chk_map:
        for sym_key in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
            chk = chk_map.get(sym_key)
            if not chk or not chk.get("items"):
                continue
            parts = [f"{name}={'Y' if ok else 'N'}" for name, ok in chk["items"]]
            lines.append(f"⏱ 10s CHECKLIST {sym_key} ({chk.get('yes', 0)}/10): " + "; ".join(parts))

    msg = "\n".join(lines)
    return msg[:MAX_TG_LEN]


def parse_command(text: str):
    if not text:
        return None, None
    parts = text.strip().split()
    cmd = parts[0].lower()
    arg = parts[1].upper().replace("/", "") if len(parts) > 1 else None
    return cmd, arg


# ============================================================
# Main Loop (V3.1: 15m-only scanning + command dedupe)
# ============================================================
def main():
    start_ts = time.time()
    cache = MarketCache()

    signal_history = {s: SignalHistory() for s in SYMBOLS}

    health = {
        "last_scan_utc": None,
        "last_signal_utc": None,
        "last_signal_text": None,
        "scan_count": 0,
        "signals_sent": 0,
    }

    last_error_notify = 0.0
    last_heartbeat = 0.0

    # Dedupe / anti-spam
    offset = None
    last_command_message_id = 0
    last_report_ts = 0.0

    if SEND_STARTUP_MESSAGE:
        # Unique boot stamp helps you detect multiple instances running
        boot_stamp = str(int(time.time()))[-4:]
        tg_send_message(
            "🚀 HTF+15M RADAR BOT v3.1 ONLINE\n"
            f"Boot: {boot_stamp}\n"
            f"UTC: {utc_now_str()}\n"
            f"Mode: 15m-based scoring + HTF filter\n"
            f"5m/1m: Info only (NOT scored)\n"
            f"Loop tick: {SLEEP_TARGET_SECONDS}s\n"
            f"Scan: 15m candle close only\n"
            f"Symbols: {', '.join(SYMBOLS)}\n\n"
            f"Quality Thresholds:\n"
            f"  WATCH: {QUALITY_SCORE_WATCH}+\n"
            f"  ENTRY: {QUALITY_SCORE_ENTRY}+\n"
            f"  PERFECT: {QUALITY_SCORE_PERFECT}+\n\n"
            f"Direction: Best of LONG vs SHORT (min diff: {MIN_SCORE_DIFF})\n\n"
            "Commands: /report | /report BTCUSDT | /status"
        )

    while True:
        loop_start = time.time()

        try:
            # 1) Telegram polling (safer)
            upd = tg_get_updates(offset)
            if isinstance(upd, dict) and upd.get("conflict"):
                time.sleep(TG_CONFLICT_BACKOFF)
                upd = {"ok": False, "result": []}

            if isinstance(upd, dict) and upd.get("ok") and upd.get("result"):
                for u in upd["result"]:
                    update_id = u.get("update_id")
                    if update_id is None:
                        continue
                    offset = update_id + 1

                    msg = u.get("message") or {}
                    message_id = int(msg.get("message_id", 0))

                    # Dedupe: avoid repeats due to backlog / retries
                    if message_id and message_id <= last_command_message_id:
                        continue
                    if message_id:
                        last_command_message_id = message_id

                    chat = msg.get("chat") or {}
                    chat_id = str(chat.get("id", ""))

                    if chat_id != str(TELEGRAM_CHAT_ID):
                        continue

                    text = (msg.get("text") or "").strip()
                    cmd, arg = parse_command(text)

                    if cmd == "/report":
                        # Simple cooldown to prevent spam
                        now = time.time()
                        if now - last_report_ts < REPORT_COOLDOWN_SECONDS:
                            tg_send_message(f"⏳ Report cooldown ({int(REPORT_COOLDOWN_SECONDS)}s).")
                            continue
                        last_report_ts = now

                        try:
                            if arg is None:
                                report_all = build_compact_all_report(cache)
                                tg_send_message(report_all)
                            else:
                                if arg in SYMBOLS:
                                    report = generate_manual_report_v3(arg, cache)
                                    tg_send_message(report)
                                else:
                                    tg_send_message(f"❌ Invalid symbol. Use: {', '.join(SYMBOLS)}")
                        except Exception as e:
                            tg_send_message(f"❌ Report error: {str(e)[:200]}")
                            print(f"Report error: {traceback.format_exc()}")

                    elif cmd == "/status":
                        uptime_min = int((time.time() - start_ts) / 60)
                        next_15m, eta_15m = next_close_eta("15m")

                        tg_send_message(
                            "📡 BOT STATUS (v3.1)\n"
                            f"UTC: {utc_now_str()}\n"
                            f"Uptime: {uptime_min} min\n"
                            f"Ticks: {health['scan_count']}\n"
                            f"Signals sent: {health['signals_sent']}\n"
                            f"Last scan: {health['last_scan_utc'] or 'N/A'}\n"
                            f"Last signal: {health['last_signal_utc'] or 'N/A'}\n"
                            f"Next 15m close: {next_15m} (ETA {int(eta_15m)}s)\n"
                            f"Scan mode: 15m-only (5m/1m info only)"
                        )

            # 2) Market scanning (15M ONLY)
            for symbol in SYMBOLS:
                cache.ensure_context_ready(symbol)

                prev_15m = cache.data[symbol]["15m"].last_closed_open_ms
                cache.refresh_15m_if_needed(symbol)
                new_15m = cache.data[symbol]["15m"].last_closed_open_ms != prev_15m

                if new_15m:
                    # Refresh 5m/1m ONLY when scanning
                    cache.refresh_5m_if_needed(symbol)
                    cache.refresh_1m_if_needed(symbol)

                    signal = detect_leverage_opportunity_v3(symbol, cache)
                    if signal:
                        direction = signal["direction"]
                        level = signal["signal_level"]
                        current_time = time.time()

                        hist = signal_history[symbol]
                        if hist.should_alert(direction, level, current_time):
                            report = generate_leverage_report_v3(signal, cache)
                            tg_send_message(report)

                            hist.update(direction, level, current_time)

                            health["last_signal_utc"] = utc_now_str()
                            health["last_signal_text"] = (
                                f"{symbol} {direction} {level} "
                                f"(score: {signal['quality_score']}, "
                                f"L:{signal['long_score']} vs S:{signal['short_score']})"
                            )
                            health["signals_sent"] += 1

            health["last_scan_utc"] = utc_now_str()
            health["scan_count"] += 1

            # 3) Heartbeat
            if HEARTBEAT_MINUTES and HEARTBEAT_MINUTES > 0:
                now = time.time()
                if now - last_heartbeat >= HEARTBEAT_MINUTES * 60:
                    tg_send_message(
                        f"💓 HEARTBEAT (v3.1)\n"
                        f"UTC: {utc_now_str()}\n"
                        f"Uptime: {int((now - start_ts) / 60)} min\n"
                        f"Ticks: {health['scan_count']}\n"
                        f"Signals: {health['signals_sent']}\n"
                        f"Last: {health['last_signal_text'] or 'None'}"
                    )
                    last_heartbeat = now

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            pass

        except Exception:
            now = time.time()
            if (now - last_error_notify) >= ERROR_NOTIFY_COOLDOWN:
                err = traceback.format_exc()
                try:
                    tg_send_message(
                        "⚠️ BOT ERROR (v3.1)\n"
                        f"UTC: {utc_now_str()}\n\n"
                        f"{err[:3000]}"
                    )
                except Exception:
                    pass
                last_error_notify = now

        # 4) Sleep
        elapsed = time.time() - loop_start
        sleep_for = max(0.5, SLEEP_TARGET_SECONDS - elapsed)
        time.sleep(sleep_for)


if __name__ == "__main__":
    main()
