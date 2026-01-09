from __future__ import annotations

import os
from typing import Optional, Union, Tuple, Dict, Any, List
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from collections import deque

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
# CONFIG - OPTIMIZED FOR LEVERAGE ENTRIES
# ============================================================
BINANCE_FAPI = "https://fapi.binance.com"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

# Loop tick: aggressive polling for 1m data
SLEEP_TARGET_SECONDS = 0.5  # From 1s → 0.5s for faster response

# Binance request hygiene
MIN_BINANCE_INTERVAL = 0.35
BAN_BACKOFF_DEFAULT = 60

# HTTP timeouts
HTTP_TIMEOUT = 12
TELEGRAM_POLL_TIMEOUT = 2

# Candle limits (added 1m for micro-timing)
OHLCV_LIMIT_1M = 120   # NEW: 2 hours of 1m data
OHLCV_LIMIT_5M = 220
OHLCV_LIMIT_15M = 220
OHLCV_LIMIT_HTF = 260

# ============================================================
# OPTIMIZED PARAMETERS FOR LEVERAGE ENTRIES
# ============================================================

# === HTF Filter (unchanged - already strong) ===
HTF_REQUIRE_BOTH = True  # Both 1h AND 4h must align

# === Multi-Timeframe Momentum Alignment ===
REQUIRE_MTF_ALIGNMENT = True  # 1m, 5m, 15m must all agree

# === Volume Requirements (MUCH STRICTER) ===
# For WATCH stage
VOL_WATCH_MIN_1M = 1.5    # 1m must show 1.5x volume
VOL_WATCH_MIN_5M = 1.8    # 5m must show 1.8x volume

# For ENTRY stage (explosive volume required)
VOL_ENTRY_MIN_1M = 2.0    # 1m explosive
VOL_ENTRY_MIN_5M = 2.5    # 5m explosive
VOL_ENTRY_MIN_15M = 2.0   # 15m confirmation

# === RSI Sweet Spots (optimized for early entries) ===
RSI_LONG_SWEET_MIN = 35   # Entry zone starts
RSI_LONG_SWEET_MAX = 52   # Entry zone ends (not overbought)
RSI_SHORT_SWEET_MIN = 48  # Entry zone starts
RSI_SHORT_SWEET_MAX = 65  # Entry zone ends (not oversold)

# RSI extremes (too early)
RSI_TOO_LOW = 25
RSI_TOO_HIGH = 75

# === MACD Slope (stricter - needs acceleration) ===
MACD_SLOPE_ATR_FACTOR_1M = 0.015   # 1m needs sharp slope
MACD_SLOPE_ATR_FACTOR_5M = 0.025   # 5m needs strong slope
MACD_SLOPE_ATR_FACTOR_15M = 0.035  # 15m needs conviction

# === Distance to Key Levels (entry proximity) ===
MAX_DIST_EMA20_ATR = 0.4      # Must be close to EMA20 (0.4 ATR)
MAX_DIST_VWAP_ATR = 0.6       # Must be close to VWAP (0.6 ATR)
IDEAL_DIST_EMA20_ATR = 0.15   # Ideal = almost touching

# === Order Flow Analysis ===
AGGRESSOR_BUY_RATIO_MIN = 0.65   # 65%+ aggressive buyers for long
AGGRESSOR_SELL_RATIO_MIN = 0.65  # 65%+ aggressive sellers for short

# === Quality Score System ===
QUALITY_SCORE_WATCH = 60     # Minimum score to alert "WATCH"
QUALITY_SCORE_ENTRY = 80     # Minimum score to alert "ENTRY_READY"
QUALITY_SCORE_PERFECT = 95   # Score for "PERFECT_ENTRY"

# === Signal Timing ===
SIGNAL_COOLDOWN_SECONDS = 180  # Wait 3min between same-direction signals
WATCH_TO_ENTRY_MAX_TIME = 900  # Max 15min from WATCH to ENTRY

# === Telegram ===
MAX_TG_LEN = 3800  # Slightly larger for detailed reports
ERROR_NOTIFY_COOLDOWN = 120
SEND_STARTUP_MESSAGE = True
HEARTBEAT_MINUTES = 360  # Every 6h
TG_CONFLICT_BACKOFF = 30

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "leverage-precision-bot/2.0"})

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


# ============================================================
# Binance REST
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

def fetch_mark_price(symbol: str) -> Optional[float]:
    j = http_get("/fapi/v1/premiumIndex", {"symbol": symbol})
    return safe_float(j.get("markPrice"))


# ============================================================
# Telegram
# ============================================================
def tg_api(method: str) -> str:
    return f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/{method}"

def tg_get_updates(offset: Optional[int]) -> dict:
    params = {"timeout": TELEGRAM_POLL_TIMEOUT}
    if offset is not None:
        params["offset"] = offset
    r = SESSION.get(tg_api("getUpdates"), params=params, timeout=TELEGRAM_POLL_TIMEOUT + 10)
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
# Indicators (Enhanced)
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

    # Enhanced order flow
    out["taker_buy"] = out["taker_buy_base"]
    out["taker_sell"] = out["volume"] - out["taker_buy"]
    out["aggressor_buy_ratio"] = out["taker_buy"] / out["volume"]
    
    # Delta cumulative (last 5 candles momentum)
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
# Cache (Multi-timeframe with 1m)
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
    Enhanced HTF check with trend strength
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
    
    if None in [c1h, e200_1h, c4h, e200_4h]:
        return False, False, {"error": "missing_data"}
    
    # Basic alignment
    above_1h = c1h > e200_1h
    above_4h = c4h > e200_4h
    
    # Trend strength (EMA stack)
    stack_bull_1h = (e20_1h and e50_1h and e200_1h and 
                     e20_1h > e50_1h > e200_1h)
    stack_bull_4h = (e20_4h and e50_4h and e200_4h and 
                     e20_4h > e50_4h > e200_4h)
    
    stack_bear_1h = (e20_1h and e50_1h and e200_1h and 
                     e20_1h < e50_1h < e200_1h)
    stack_bear_4h = (e20_4h and e50_4h and e200_4h and 
                     e20_4h < e50_4h < e200_4h)
    
    # Strength calculation
    strength_1h = "STRONG" if stack_bull_1h or stack_bear_1h else "WEAK"
    strength_4h = "STRONG" if stack_bull_4h or stack_bear_4h else "WEAK"
    
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
# Multi-Timeframe Momentum Alignment
# ============================================================
def check_mtf_alignment(df1m: pd.DataFrame, df5m: pd.DataFrame, df15m: pd.DataFrame) -> Dict[str, Any]:
    """
    Check if 1m, 5m, 15m all show same directional bias
    """
    i1m = len(df1m) - 2
    i5m = len(df5m) - 2
    i15m = len(df15m) - 2
    
    if i1m < 25 or i5m < 25 or i15m < 25:
        return {"aligned": False, "direction": None, "error": "insufficient_data"}
    
    # Check each TF
    def get_bias(df, i):
        c = safe_float(df.loc[i, "close"])
        v = safe_float(df.loc[i, "vwap"])
        e20 = safe_float(df.loc[i, "ema20"])
        e50 = safe_float(df.loc[i, "ema50"])
        rsi = safe_float(df.loc[i, "rsi14"])
        mh = safe_float(df.loc[i, "macdh"])
        
        if None in [c, v, e20, e50, rsi, mh]:
            return None
        
        bull_votes = 0
        bear_votes = 0
        
        if c > v: bull_votes += 1
        else: bear_votes += 1
        
        if e20 > e50: bull_votes += 1
        else: bear_votes += 1
        
        if rsi > 50: bull_votes += 1
        elif rsi < 50: bear_votes += 1
        
        if mh > 0: bull_votes += 1
        elif mh < 0: bear_votes += 1
        
        if bull_votes >= 3:
            return "BULL"
        elif bear_votes >= 3:
            return "BEAR"
        return "NEUTRAL"
    
    bias_1m = get_bias(df1m, i1m)
    bias_5m = get_bias(df5m, i5m)
    bias_15m = get_bias(df15m, i15m)
    
    # Alignment check
    if bias_1m == bias_5m == bias_15m and bias_1m in ["BULL", "BEAR"]:
        return {
            "aligned": True,
            "direction": bias_1m,
            "1m": bias_1m,
            "5m": bias_5m,
            "15m": bias_15m,
        }
    
    return {
        "aligned": False,
        "direction": None,
        "1m": bias_1m,
        "5m": bias_5m,
        "15m": bias_15m,
    }


# ============================================================
# Quality Score Calculator (CORE INNOVATION)
# ============================================================
def calculate_quality_score(
    symbol: str,
    df1m: pd.DataFrame,
    df5m: pd.DataFrame,
    df15m: pd.DataFrame,
    df1h: pd.DataFrame,
    df4h: pd.DataFrame,
    direction: str  # "LONG" or "SHORT"
) -> Dict[str, Any]:
    """
    Calculate comprehensive entry quality score (0-100)
    
    Scoring breakdown:
    - HTF Alignment: 15 points
    - MTF Momentum: 15 points
    - Volume Explosion: 20 points
    - RSI Sweet Spot: 15 points
    - Proximity to Levels: 15 points
    - Order Flow: 10 points
    - MACD Acceleration: 10 points
    
    Total: 100 points
    """
    score = 0
    breakdown = {}
    
    i1m = len(df1m) - 2
    i5m = len(df5m) - 2
    i15m = len(df15m) - 2
    
    if i1m < 25 or i5m < 25 or i15m < 25:
        return {"score": 0, "breakdown": {}, "error": "insufficient_data"}
    
    # === 1. HTF Alignment (15 points) ===
    htf_long, htf_short, htf_ctx = check_htf_alignment(df1h, df4h)
    
    if direction == "LONG":
        if htf_long:
            if htf_ctx.get("1h_strength") == "STRONG" and htf_ctx.get("4h_strength") == "STRONG":
                score += 15
                breakdown["htf"] = 15
            else:
                score += 10
                breakdown["htf"] = 10
    elif direction == "SHORT":
        if htf_short:
            if htf_ctx.get("1h_strength") == "STRONG" and htf_ctx.get("4h_strength") == "STRONG":
                score += 15
                breakdown["htf"] = 15
            else:
                score += 10
                breakdown["htf"] = 10
    
    if "htf" not in breakdown:
        breakdown["htf"] = 0
    
    # === 2. MTF Momentum Alignment (15 points) ===
    mtf = check_mtf_alignment(df1m, df5m, df15m)
    
    if mtf["aligned"]:
        if direction == "LONG" and mtf["direction"] == "BULL":
            score += 15
            breakdown["mtf"] = 15
        elif direction == "SHORT" and mtf["direction"] == "BEAR":
            score += 15
            breakdown["mtf"] = 15
    else:
        breakdown["mtf"] = 0
    
    # === 3. Volume Explosion (20 points) ===
    vol_1m = safe_float(df1m.loc[i1m, "volume"])
    vol_avg_1m = safe_float(df1m.loc[i1m, "vol_sma20"])
    volx_1m = (vol_1m / vol_avg_1m) if (vol_avg_1m and vol_avg_1m > 0) else 0
    
    vol_5m = safe_float(df5m.loc[i5m, "volume"])
    vol_avg_5m = safe_float(df5m.loc[i5m, "vol_sma20"])
    volx_5m = (vol_5m / vol_avg_5m) if (vol_avg_5m and vol_avg_5m > 0) else 0
    
    vol_15m = safe_float(df15m.loc[i15m, "volume"])
    vol_avg_15m = safe_float(df15m.loc[i15m, "vol_sma20"])
    volx_15m = (vol_15m / vol_avg_15m) if (vol_avg_15m and vol_avg_15m > 0) else 0
    
    vol_score = 0
    # 1m volume
    if volx_1m >= 3.0:
        vol_score += 8
    elif volx_1m >= VOL_ENTRY_MIN_1M:
        vol_score += 6
    elif volx_1m >= VOL_WATCH_MIN_1M:
        vol_score += 3
    
    # 5m volume
    if volx_5m >= 3.5:
        vol_score += 8
    elif volx_5m >= VOL_ENTRY_MIN_5M:
        vol_score += 6
    elif volx_5m >= VOL_WATCH_MIN_5M:
        vol_score += 3
    
    # 15m volume (bonus)
    if volx_15m >= VOL_ENTRY_MIN_15M:
        vol_score += 4
    
    vol_score = min(20, vol_score)
    score += vol_score
    breakdown["volume"] = vol_score
    breakdown["volx_1m"] = volx_1m
    breakdown["volx_5m"] = volx_5m
    breakdown["volx_15m"] = volx_15m
    
    # === 4. RSI Sweet Spot (15 points) ===
    rsi_1m = safe_float(df1m.loc[i1m, "rsi14"])
    rsi_5m = safe_float(df5m.loc[i5m, "rsi14"])
    rsi_15m = safe_float(df15m.loc[i15m, "rsi14"])
    
    rsi_score = 0
    if direction == "LONG":
        # Perfect zone: 40-48
        if rsi_5m and 40 <= rsi_5m <= 48:
            rsi_score += 10
        elif rsi_5m and RSI_LONG_SWEET_MIN <= rsi_5m <= RSI_LONG_SWEET_MAX:
            rsi_score += 7
        
        # 1m confirmation
        if rsi_1m and rsi_1m >= 40:
            rsi_score += 3
        
        # Not overbought on 15m
        if rsi_15m and rsi_15m < 60:
            rsi_score += 2
    
    elif direction == "SHORT":
        # Perfect zone: 52-60
        if rsi_5m and 52 <= rsi_5m <= 60:
            rsi_score += 10
        elif rsi_5m and RSI_SHORT_SWEET_MIN <= rsi_5m <= RSI_SHORT_SWEET_MAX:
            rsi_score += 7
        
        # 1m confirmation
        if rsi_1m and rsi_1m <= 60:
            rsi_score += 3
        
        # Not oversold on 15m
        if rsi_15m and rsi_15m > 40:
            rsi_score += 2
    
    rsi_score = min(15, rsi_score)
    score += rsi_score
    breakdown["rsi"] = rsi_score
    breakdown["rsi_1m"] = rsi_1m
    breakdown["rsi_5m"] = rsi_5m
    breakdown["rsi_15m"] = rsi_15m
    
    # === 5. Proximity to Key Levels (15 points) ===
    px_5m = safe_float(df5m.loc[i5m, "close"])
    ema20_5m = safe_float(df5m.loc[i5m, "ema20"])
    vwap_5m = safe_float(df5m.loc[i5m, "vwap"])
    atr_5m = safe_float(df5m.loc[i5m, "atr14"])
    
    prox_score = 0
    if px_5m and ema20_5m and atr_5m and atr_5m > 0:
        dist_ema20 = dist_atr(px_5m, ema20_5m, atr_5m)
        
        if dist_ema20 <= IDEAL_DIST_EMA20_ATR:
            prox_score += 10  # Perfect - touching EMA20
        elif dist_ema20 <= MAX_DIST_EMA20_ATR:
            prox_score += 7   # Good - close to EMA20
        elif dist_ema20 <= 0.8:
            prox_score += 3   # Acceptable
        
        breakdown["dist_ema20_atr"] = dist_ema20
    
    if px_5m and vwap_5m and atr_5m and atr_5m > 0:
        dist_vwap = dist_atr(px_5m, vwap_5m, atr_5m)
        
        if dist_vwap <= 0.3:
            prox_score += 5   # Near VWAP
        elif dist_vwap <= MAX_DIST_VWAP_ATR:
            prox_score += 3
        
        breakdown["dist_vwap_atr"] = dist_vwap
    
    prox_score = min(15, prox_score)
    score += prox_score
    breakdown["proximity"] = prox_score
    
    # === 6. Order Flow (10 points) ===
    aggr_ratio_1m = safe_float(df1m.loc[i1m, "aggressor_buy_ratio"])
    aggr_ratio_5m = safe_float(df5m.loc[i5m, "aggressor_buy_ratio"])
    
    delta_cumsum_5m = safe_float(df5m.loc[i5m, "delta_cumsum_5"])
    
    flow_score = 0
    if direction == "LONG":
        if aggr_ratio_5m and aggr_ratio_5m >= 0.70:
            flow_score += 6
        elif aggr_ratio_5m and aggr_ratio_5m >= AGGRESSOR_BUY_RATIO_MIN:
            flow_score += 4
        
        if aggr_ratio_1m and aggr_ratio_1m >= 0.70:
            flow_score += 2
        
        if delta_cumsum_5m and delta_cumsum_5m > 0:
            flow_score += 2
    
    elif direction == "SHORT":
        if aggr_ratio_5m and aggr_ratio_5m <= 0.30:
            flow_score += 6
        elif aggr_ratio_5m and aggr_ratio_5m <= (1 - AGGRESSOR_SELL_RATIO_MIN):
            flow_score += 4
        
        if aggr_ratio_1m and aggr_ratio_1m <= 0.30:
            flow_score += 2
        
        if delta_cumsum_5m and delta_cumsum_5m < 0:
            flow_score += 2
    
    flow_score = min(10, flow_score)
    score += flow_score
    breakdown["order_flow"] = flow_score
    breakdown["aggr_ratio_1m"] = aggr_ratio_1m
    breakdown["aggr_ratio_5m"] = aggr_ratio_5m
    
    # === 7. MACD Acceleration (10 points) ===
    macdh_1m = safe_float(df1m.loc[i1m, "macdh"])
    macdh_1m_prev = safe_float(df1m.loc[i1m-1, "macdh"]) if i1m > 0 else None
    
    macdh_5m = safe_float(df5m.loc[i5m, "macdh"])
    macdh_5m_prev = safe_float(df5m.loc[i5m-1, "macdh"]) if i5m > 0 else None
    
    atr_1m = safe_float(df1m.loc[i1m, "atr14"])
    atr_5m = safe_float(df5m.loc[i5m, "atr14"])
    
    macd_score = 0
    
    if direction == "LONG":
        # 5m MACD slope
        if macdh_5m and macdh_5m_prev and atr_5m and atr_5m > 0:
            slope_5m = macdh_5m - macdh_5m_prev
            slope_thr_5m = atr_5m * MACD_SLOPE_ATR_FACTOR_5M
            
            if slope_5m > slope_thr_5m * 1.5:
                macd_score += 6
            elif slope_5m > slope_thr_5m:
                macd_score += 4
        
        # 1m MACD slope
        if macdh_1m and macdh_1m_prev and atr_1m and atr_1m > 0:
            slope_1m = macdh_1m - macdh_1m_prev
            slope_thr_1m = atr_1m * MACD_SLOPE_ATR_FACTOR_1M
            
            if slope_1m > slope_thr_1m:
                macd_score += 4
    
    elif direction == "SHORT":
        # 5m MACD slope
        if macdh_5m and macdh_5m_prev and atr_5m and atr_5m > 0:
            slope_5m = macdh_5m - macdh_5m_prev
            slope_thr_5m = atr_5m * MACD_SLOPE_ATR_FACTOR_5M
            
            if slope_5m < -slope_thr_5m * 1.5:
                macd_score += 6
            elif slope_5m < -slope_thr_5m:
                macd_score += 4
        
        # 1m MACD slope
        if macdh_1m and macdh_1m_prev and atr_1m and atr_1m > 0:
            slope_1m = macdh_1m - macdh_1m_prev
            slope_thr_1m = atr_1m * MACD_SLOPE_ATR_FACTOR_1M
            
            if slope_1m < -slope_thr_1m:
                macd_score += 4
    
    macd_score = min(10, macd_score)
    score += macd_score
    breakdown["macd"] = macd_score
    
    return {
        "score": score,
        "breakdown": breakdown,
        "htf_context": htf_ctx,
        "mtf": mtf,
    }


# ============================================================
# Signal Detection (Unified)
# ============================================================
def detect_leverage_opportunity(
    symbol: str,
    cache: MarketCache
) -> Optional[Dict[str, Any]]:
    """
    Unified signal detection across all timeframes
    Returns signal dict or None
    """
    df1m = cache.get(symbol, "1m")
    df5m = cache.get(symbol, "5m")
    df15m = cache.get(symbol, "15m")
    df1h = cache.get(symbol, "1h")
    df4h = cache.get(symbol, "4h")
    
    i1m = len(df1m) - 2
    i5m = len(df5m) - 2
    i15m = len(df15m) - 2
    
    if i1m < 30 or i5m < 30 or i15m < 30:
        return None
    
    # Check both directions
    for direction in ["LONG", "SHORT"]:
        quality = calculate_quality_score(
            symbol, df1m, df5m, df15m, df1h, df4h, direction
        )
        
        score = quality["score"]
        
        # Determine signal level
        signal_level = None
        if score >= QUALITY_SCORE_PERFECT:
            signal_level = "PERFECT_ENTRY"
        elif score >= QUALITY_SCORE_ENTRY:
            signal_level = "ENTRY_READY"
        elif score >= QUALITY_SCORE_WATCH:
            signal_level = "WATCH"
        
        if signal_level:
            # Get current price
            px = safe_float(df5m.loc[i5m, "close"])
            mark = fetch_mark_price(symbol) or px
            
            candle_ts_1m = int(df1m.loc[i1m, "ts"])
            candle_ts_5m = int(df5m.loc[i5m, "ts"])
            
            return {
                "symbol": symbol,
                "direction": direction,
                "signal_level": signal_level,
                "quality_score": score,
                "quality_breakdown": quality["breakdown"],
                "htf_context": quality["htf_context"],
                "mtf": quality["mtf"],
                "price": px,
                "mark_price": mark,
                "candle_ts_1m": candle_ts_1m,
                "candle_ts_5m": candle_ts_5m,
                "timestamp": time.time(),
            }
    
    return None


# ============================================================
# Report Generation (Enhanced for GPT Analysis)
# ============================================================
def generate_leverage_report(signal: Dict[str, Any], cache: MarketCache) -> str:
    """
    Generate comprehensive report optimized for ChatGPT analysis
    """
    symbol = signal["symbol"]
    direction = signal["direction"]
    level = signal["signal_level"]
    score = signal["quality_score"]
    breakdown = signal["quality_breakdown"]
    htf_ctx = signal["htf_context"]
    mtf = signal["mtf"]
    
    # Emojis based on signal level
    emoji = {
        "PERFECT_ENTRY": "🚀💎",
        "ENTRY_READY": "✅🎯",
        "WATCH": "👀⚠️"
    }.get(level, "📊")
    
    # Header
    lines = [
        f"{emoji} {level} | {symbol} {direction}",
        f"Quality Score: {score}/100",
        f"Time: {utc_now_str()}",
        "=" * 50,
    ]
    
    # Price info
    lines.append(f"Price: {signal['price']:.2f} | Mark: {signal['mark_price']:.2f}")
    lines.append("")
    
    # Quality breakdown
    lines.append("📊 QUALITY BREAKDOWN:")
    lines.append(f"  HTF Alignment: {breakdown.get('htf', 0)}/15")
    lines.append(f"  MTF Momentum: {breakdown.get('mtf', 0)}/15")
    lines.append(f"  Volume: {breakdown.get('volume', 0)}/20")
    lines.append(f"  RSI Sweet Spot: {breakdown.get('rsi', 0)}/15")
    lines.append(f"  Proximity: {breakdown.get('proximity', 0)}/15")
    lines.append(f"  Order Flow: {breakdown.get('order_flow', 0)}/10")
    lines.append(f"  MACD Accel: {breakdown.get('macd', 0)}/10")
    lines.append("")
    
    # HTF Context
    lines.append("🌍 HTF CONTEXT:")
    lines.append(f"  1h: {'ABOVE' if htf_ctx.get('1h_above_200') else 'BELOW'} EMA200 ({htf_ctx.get('1h_strength', 'N/A')})")
    lines.append(f"  4h: {'ABOVE' if htf_ctx.get('4h_above_200') else 'BELOW'} EMA200 ({htf_ctx.get('4h_strength', 'N/A')})")
    lines.append(f"  1h Dist: {htf_ctx.get('dist_1h_200', 0):.2f}%")
    lines.append(f"  4h Dist: {htf_ctx.get('dist_4h_200', 0):.2f}%")
    lines.append("")
    
    # MTF Alignment
    lines.append("⚡ MTF ALIGNMENT:")
    lines.append(f"  Aligned: {mtf.get('aligned', False)}")
    lines.append(f"  1m: {mtf.get('1m', 'N/A')}")
    lines.append(f"  5m: {mtf.get('5m', 'N/A')}")
    lines.append(f"  15m: {mtf.get('15m', 'N/A')}")
    lines.append("")
    
    # Volume metrics
    lines.append("📈 VOLUME:")
    lines.append(f"  1m: {breakdown.get('volx_1m', 0):.2f}x")
    lines.append(f"  5m: {breakdown.get('volx_5m', 0):.2f}x")
    lines.append(f"  15m: {breakdown.get('volx_15m', 0):.2f}x")
    lines.append("")
    
    # RSI
    lines.append("📉 RSI:")
    lines.append(f"  1m: {breakdown.get('rsi_1m', 0):.1f}")
    lines.append(f"  5m: {breakdown.get('rsi_5m', 0):.1f}")
    lines.append(f"  15m: {breakdown.get('rsi_15m', 0):.1f}")
    lines.append("")
    
    # Proximity
    lines.append("🎯 PROXIMITY:")
    dist_ema = breakdown.get('dist_ema20_atr', 0)
    dist_vwap = breakdown.get('dist_vwap_atr', 0)
    lines.append(f"  EMA20: {dist_ema:.2f} ATR")
    lines.append(f"  VWAP: {dist_vwap:.2f} ATR")
    lines.append("")
    
    # Order flow
    lines.append("💹 ORDER FLOW:")
    lines.append(f"  1m Aggressor Buy: {breakdown.get('aggr_ratio_1m', 0):.1%}")
    lines.append(f"  5m Aggressor Buy: {breakdown.get('aggr_ratio_5m', 0):.1%}")
    lines.append("")
    
    # GPT Analysis prompt
    lines.append("=" * 50)
    lines.append("🤖 FOR GPT ANALYSIS:")
    lines.append("")
    lines.append(f"Analyze this {level} signal for {symbol} {direction}.")
    lines.append(f"Quality score: {score}/100")
    lines.append("")
    lines.append("Key questions:")
    lines.append("1. Is this entry timing optimal for leverage trading?")
    lines.append("2. What are the key risks to watch?")
    lines.append("3. Suggested entry zones and stop-loss levels?")
    lines.append("4. Should we wait for higher score or enter now?")
    lines.append("5. How does momentum look across timeframes?")
    
    return "\n".join(lines)


def generate_manual_report(symbol: str, cache: MarketCache) -> str:
    """Generate manual status report"""
    cache.refresh_1m_force(symbol)
    cache.refresh_5m_force(symbol)
    cache.refresh_15m_force(symbol)
    cache.ensure_context_ready(symbol)
    
    # Try to detect current opportunities
    signal = detect_leverage_opportunity(symbol, cache)
    
    if signal:
        return generate_leverage_report(signal, cache)
    else:
        # No signal - provide current state
        df1m = cache.get(symbol, "1m")
        df5m = cache.get(symbol, "5m")
        df15m = cache.get(symbol, "15m")
        
        i5m = len(df5m) - 2
        
        px = safe_float(df5m.loc[i5m, "close"])
        mark = fetch_mark_price(symbol) or px
        rsi_5m = safe_float(df5m.loc[i5m, "rsi14"])
        vol_5m = safe_float(df5m.loc[i5m, "volume"])
        vol_avg_5m = safe_float(df5m.loc[i5m, "vol_sma20"])
        volx = (vol_5m / vol_avg_5m) if vol_avg_5m and vol_avg_5m > 0 else 0
        
        # Check both directions for scores
        long_quality = calculate_quality_score(
            symbol, df1m, df5m, df15m,
            cache.get(symbol, "1h"),
            cache.get(symbol, "4h"),
            "LONG"
        )
        
        short_quality = calculate_quality_score(
            symbol, df1m, df5m, df15m,
            cache.get(symbol, "1h"),
            cache.get(symbol, "4h"),
            "SHORT"
        )
        
        return (
            f"📊 {symbol} STATUS\n"
            f"Time: {utc_now_str()}\n"
            f"Price: {px:.2f} | Mark: {mark:.2f}\n"
            f"RSI (5m): {rsi_5m:.1f}\n"
            f"Volume: {volx:.2f}x\n\n"
            f"Quality Scores:\n"
            f"  LONG: {long_quality['score']}/100\n"
            f"  SHORT: {short_quality['score']}/100\n\n"
            f"No high-quality signals detected.\n"
            f"Waiting for score ≥ {QUALITY_SCORE_WATCH}..."
        )


# ============================================================
# Signal De-duplication & Cooldown
# ============================================================
@dataclass
class SignalHistory:
    last_signal_time: float = 0.0
    last_direction: Optional[str] = None
    last_level: Optional[str] = None
    watch_start_time: Optional[float] = None
    
    def should_alert(self, direction: str, level: str, current_time: float) -> bool:
        """Check if we should send this alert"""
        
        # First signal ever
        if self.last_signal_time == 0:
            return True
        
        # Different direction - allow immediately
        if direction != self.last_direction:
            return True
        
        # Same direction - check cooldown
        time_since_last = current_time - self.last_signal_time
        
        # Level escalation (WATCH → ENTRY → PERFECT) - allow
        level_order = {"WATCH": 0, "ENTRY_READY": 1, "PERFECT_ENTRY": 2}
        if level_order.get(level, 0) > level_order.get(self.last_level, 0):
            return True
        
        # Same or lower level - apply cooldown
        if time_since_last < SIGNAL_COOLDOWN_SECONDS:
            return False
        
        return True
    
    def update(self, direction: str, level: str, current_time: float):
        """Update history after sending signal"""
        self.last_signal_time = current_time
        self.last_direction = direction
        self.last_level = level
        
        if level == "WATCH" and self.watch_start_time is None:
            self.watch_start_time = current_time
        elif level in ["ENTRY_READY", "PERFECT_ENTRY"]:
            self.watch_start_time = None


# ============================================================
# Command Parsing
# ============================================================
def parse_command(text: str):
    if not text:
        return None, None
    parts = text.strip().split()
    cmd = parts[0].lower()
    arg = parts[1].upper().replace("/", "") if len(parts) > 1 else None
    return cmd, arg


# ============================================================
# Main Loop (Optimized)
# ============================================================
def main():
    start_ts = time.time()
    cache = MarketCache()
    
    # Signal history per symbol
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
    
    if SEND_STARTUP_MESSAGE:
        tg_send_message(
            "🚀 LEVERAGE PRECISION BOT ONLINE\n"
            f"UTC: {utc_now_str()}\n"
            f"Version: 2.0 (Optimized)\n"
            f"Mode: Multi-timeframe (1m/5m/15m) + Quality Scoring\n"
            f"Loop tick: {SLEEP_TARGET_SECONDS}s\n"
            f"Symbols: {', '.join(SYMBOLS)}\n\n"
            f"Quality Thresholds:\n"
            f"  WATCH: {QUALITY_SCORE_WATCH}+\n"
            f"  ENTRY: {QUALITY_SCORE_ENTRY}+\n"
            f"  PERFECT: {QUALITY_SCORE_PERFECT}+\n\n"
            "Commands: /report | /report BTCUSDT | /status"
        )
    
    offset = None
    
    while True:
        loop_start = time.time()
        
        try:
            # 1) Telegram polling
            upd = tg_get_updates(offset)
            if isinstance(upd, dict) and upd.get("conflict"):
                time.sleep(TG_CONFLICT_BACKOFF)
                upd = {"ok": False, "result": []}
            
            if isinstance(upd, dict) and upd.get("ok") and upd.get("result"):
                for u in upd["result"]:
                    offset = u["update_id"] + 1
                    msg = u.get("message") or {}
                    chat = msg.get("chat") or {}
                    chat_id = str(chat.get("id", ""))
                    
                    if chat_id != str(TELEGRAM_CHAT_ID):
                        continue
                    
                    text = (msg.get("text") or "").strip()
                    cmd, arg = parse_command(text)
                    
                    if cmd == "/report":
                        if arg is None:
                            # Report all symbols
                            for sym in SYMBOLS:
                                report = generate_manual_report(sym, cache)
                                tg_send_message(report)
                                time.sleep(0.5)  # Rate limit
                        else:
                            if arg in SYMBOLS:
                                report = generate_manual_report(arg, cache)
                                tg_send_message(report)
                            else:
                                tg_send_message(f"❌ Invalid symbol. Use: {', '.join(SYMBOLS)}")
                    
                    elif cmd == "/status":
                        uptime_min = int((time.time() - start_ts) / 60)
                        
                        status_lines = [
                            "📡 BOT STATUS",
                            f"UTC: {utc_now_str()}",
                            f"Uptime: {uptime_min} min",
                            f"Ticks: {health['scan_count']}",
                            f"Signals sent: {health['signals_sent']}",
                            f"Last scan: {health['last_scan_utc'] or 'N/A'}",
                            f"Last signal: {health['last_signal_utc'] or 'N/A'}",
                        ]
                        
                        # Next candle close times
                        for tf in ["1m", "5m", "15m"]:
                            next_close, eta = next_close_eta(tf)
                            status_lines.append(f"Next {tf}: {next_close} (ETA {int(eta)}s)")
                        
                        tg_send_message("\n".join(status_lines))
            
            # 2) Market scanning (multi-timeframe)
            for symbol in SYMBOLS:
                # Ensure all TF are fresh
                cache.ensure_context_ready(symbol)
                
                prev_1m = cache.data[symbol]["1m"].last_closed_open_ms
                prev_5m = cache.data[symbol]["5m"].last_closed_open_ms
                prev_15m = cache.data[symbol]["15m"].last_closed_open_ms
                
                cache.refresh_1m_if_needed(symbol)
                cache.refresh_5m_if_needed(symbol)
                cache.refresh_15m_if_needed(symbol)
                
                # Check for new candle close on any TF
                new_1m = cache.data[symbol]["1m"].last_closed_open_ms != prev_1m
                new_5m = cache.data[symbol]["5m"].last_closed_open_ms != prev_5m
                new_15m = cache.data[symbol]["15m"].last_closed_open_ms != prev_15m
                
                # Scan on any new candle (but primarily 5m and 15m)
                if new_5m or new_15m or (new_1m and health['scan_count'] % 5 == 0):
                    signal = detect_leverage_opportunity(symbol, cache)
                    
                    if signal:
                        direction = signal["direction"]
                        level = signal["signal_level"]
                        current_time = time.time()
                        
                        hist = signal_history[symbol]
                        
                        if hist.should_alert(direction, level, current_time):
                            # Send alert
                            report = generate_leverage_report(signal, cache)
                            tg_send_message(report)
                            
                            # Update history
                            hist.update(direction, level, current_time)
                            
                            health["last_signal_utc"] = utc_now_str()
                            health["last_signal_text"] = f"{symbol} {direction} {level} (score: {signal['quality_score']})"
                            health["signals_sent"] += 1
            
            health["last_scan_utc"] = utc_now_str()
            health["scan_count"] += 1
            
            # 3) Heartbeat
            if HEARTBEAT_MINUTES and HEARTBEAT_MINUTES > 0:
                now = time.time()
                if now - last_heartbeat >= HEARTBEAT_MINUTES * 60:
                    tg_send_message(
                        f"💓 HEARTBEAT\n"
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
                tg_send_message(
                    "⚠️ BOT ERROR\n"
                    f"UTC: {utc_now_str()}\n\n"
                    f"{err[:3000]}"
                )
                last_error_notify = now
        
        # 4) Sleep
        elapsed = time.time() - loop_start
        sleep_for = max(0.1, SLEEP_TARGET_SECONDS - elapsed)
        time.sleep(sleep_for)


if __name__ == "__main__":
    main()
