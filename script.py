from __future__ import annotations

import os
from typing import Optional, Union, Tuple, Dict, Any, List
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone

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

# Loop tick: relaxed for rate-limit safety
SLEEP_TARGET_SECONDS = 2.0  # Was 0.5 → now 2.0 for stability

# Binance request hygiene (SAFER)
MIN_BINANCE_INTERVAL = 0.7  # Was 0.35 → now 0.7 to avoid bans
BAN_BACKOFF_DEFAULT = 60

# HTTP timeouts
HTTP_TIMEOUT = 12
TELEGRAM_POLL_TIMEOUT = 2

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
AGGRESSOR_SELL_RATIO_MAX = 0.40  # Was called MIN but was confusing - now MAX (sellers dominate when aggr_buy < 40%)

# === Quality Score System (100 points total) ===
# All points from 15m + HTF ONLY
QUALITY_SCORE_WATCH = 60     # Minimum score to alert "WATCH"
QUALITY_SCORE_ENTRY = 75     # Was 80 → now 75 (more achievable)
QUALITY_SCORE_PERFECT = 90   # Was 95 → now 90 (realistic)

# === Direction Selection ===
MIN_SCORE_DIFF = 10  # NEW: Winning direction must be 10+ points ahead

# === Signal Timing ===
SIGNAL_COOLDOWN_SECONDS = 300  # 5min between same-direction signals
MAX_SIGNAL_AGE_SECONDS = 1800  # 30min max - after that, re-evaluate

# === Telegram ===
MAX_TG_LEN = 3800
ERROR_NOTIFY_COOLDOWN = 120
SEND_STARTUP_MESSAGE = True
HEARTBEAT_MINUTES = 360  # Every 6h
TG_CONFLICT_BACKOFF = 30

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "htf-15m-radar-bot/3.0"})

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
    
    # Check 5m bias
    c5 = safe_float(df5m.loc[i5m, "close"])
    v5 = safe_float(df5m.loc[i5m, "vwap"])
    e20_5 = safe_float(df5m.loc[i5m, "ema20"])
    e50_5 = safe_float(df5m.loc[i5m, "ema50"])
    
    if None in [c5, v5, e20_5, e50_5]:
        return 0
    
    # Simple 5m bias
    bias_5m_bull = (c5 > v5) and (e20_5 > e50_5)
    bias_5m_bear = (c5 < v5) and (e20_5 < e50_5)
    
    # Award bonus if 5m matches direction
    if direction == "LONG" and bias_5m_bull:
        return MTF_BONUS_POINTS
    elif direction == "SHORT" and bias_5m_bear:
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
    
    Scoring (100 points total):
    - HTF Alignment: 20 points
    - 15m Momentum (RSI + MACD): 25 points
    - 15m Structure (EMA proximity + VWAP): 25 points
    - 15m Order Flow: 20 points
    - MTF Bonus (5m alignment): 10 points (optional)
    """
    score = 0
    breakdown = {}
    
    i15m = len(df15m) - 2
    
    if i15m < 30:
        return {"score": 0, "breakdown": {}, "error": "insufficient_data"}
    
    # === 1. HTF Alignment (20 points) ===
    htf_long, htf_short, htf_ctx = check_htf_alignment(df1h, df4h)
    
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
    
    if "htf" not in breakdown:
        breakdown["htf"] = 0
    
    # === 2. 15m Momentum (25 points: RSI 15 + MACD 10) ===
    rsi_15m = safe_float(df15m.loc[i15m, "rsi14"])
    macdh_15m = safe_float(df15m.loc[i15m, "macdh"])
    macdh_15m_prev = safe_float(df15m.loc[i15m-1, "macdh"]) if i15m > 0 else None
    atr_15m = safe_float(df15m.loc[i15m, "atr14"])
    
    # RSI scoring (15 points)
    rsi_score = 0
    if direction == "LONG":
        if rsi_15m and RSI_LONG_IDEAL_MIN <= rsi_15m <= RSI_LONG_IDEAL_MAX:
            rsi_score = 15  # Perfect zone
        elif rsi_15m and RSI_LONG_SWEET_MIN <= rsi_15m <= RSI_LONG_SWEET_MAX:
            rsi_score = 10  # Good zone
        elif rsi_15m and 40 <= rsi_15m <= 65:
            rsi_score = 5   # Acceptable
    elif direction == "SHORT":
        if rsi_15m and RSI_SHORT_IDEAL_MIN <= rsi_15m <= RSI_SHORT_IDEAL_MAX:
            rsi_score = 15
        elif rsi_15m and RSI_SHORT_SWEET_MIN <= rsi_15m <= RSI_SHORT_SWEET_MAX:
            rsi_score = 10
        elif rsi_15m and 35 <= rsi_15m <= 60:
            rsi_score = 5
    
    score += rsi_score
    breakdown["rsi"] = rsi_score
    breakdown["rsi_15m"] = rsi_15m
    
    # MACD slope scoring (10 points)
    macd_score = 0
    if macdh_15m is not None and macdh_15m_prev is not None and atr_15m and atr_15m > 0:
        slope_15m = macdh_15m - macdh_15m_prev
        slope_thr = atr_15m * MACD_SLOPE_ATR_FACTOR_15M
        
        if direction == "LONG":
            if slope_15m > slope_thr * 1.5:
                macd_score = 10  # Strong acceleration
            elif slope_15m > slope_thr:
                macd_score = 7   # Good momentum
            elif macdh_15m > 0:
                macd_score = 3   # At least positive
        elif direction == "SHORT":
            if slope_15m < -slope_thr * 1.5:
                macd_score = 10
            elif slope_15m < -slope_thr:
                macd_score = 7
            elif macdh_15m < 0:
                macd_score = 3
    
    score += macd_score
    breakdown["macd"] = macd_score
    
    # === 3. 15m Structure (25 points: Proximity 15 + Volume 10) ===
    px_15m = safe_float(df15m.loc[i15m, "close"])
    ema20_15m = safe_float(df15m.loc[i15m, "ema20"])
    ema50_15m = safe_float(df15m.loc[i15m, "ema50"])
    vwap_15m = safe_float(df15m.loc[i15m, "vwap"])
    
    # Proximity scoring (15 points)
    prox_score = 0
    if px_15m and ema20_15m and atr_15m and atr_15m > 0:
        dist_ema20 = dist_atr(px_15m, ema20_15m, atr_15m)
        
        if dist_ema20 <= IDEAL_DIST_EMA20_ATR:
            prox_score += 10  # Very close to EMA20
        elif dist_ema20 <= MAX_DIST_EMA20_ATR:
            prox_score += 7   # Reasonably close
        elif dist_ema20 <= 1.2:
            prox_score += 3   # Still in range
        
        breakdown["dist_ema20_atr"] = dist_ema20
    
    # VWAP bonus (5 points)
    if px_15m and vwap_15m and atr_15m and atr_15m > 0:
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
    volx_15m = (vol_15m / vol_avg_15m) if (vol_avg_15m and vol_avg_15m > 0) else 0
    
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
    # NOTE: taker_buy_base from klines is a PROXY for aggressor flow
    # It's directionally useful but not as precise as true aggressor data from trades
    aggr_ratio_15m = safe_float(df15m.loc[i15m, "aggressor_buy_ratio"])
    delta_15m = safe_float(df15m.loc[i15m, "delta"])
    delta_cumsum_15m = safe_float(df15m.loc[i15m, "delta_cumsum_5"])
    
    flow_score = 0
    if direction == "LONG":
        # Aggressor buy ratio
        if aggr_ratio_15m is not None and aggr_ratio_15m >= 0.70:
            flow_score += 12  # Strong buying pressure
        elif aggr_ratio_15m is not None and aggr_ratio_15m >= AGGRESSOR_BUY_RATIO_MIN:
            flow_score += 8   # Good buying pressure
        elif aggr_ratio_15m is not None and aggr_ratio_15m >= 0.55:
            flow_score += 4   # Moderate buying
        
        # Delta confirmation
        if delta_cumsum_15m and delta_cumsum_15m > 0:
            flow_score += 8
    
    elif direction == "SHORT":
        # Aggressor sell ratio (buyers are minority)
        if aggr_ratio_15m is not None and aggr_ratio_15m <= 0.30:
            flow_score += 12  # Strong selling pressure
        elif aggr_ratio_15m is not None and aggr_ratio_15m <= AGGRESSOR_SELL_RATIO_MAX:
            flow_score += 8   # Good selling pressure
        elif aggr_ratio_15m is not None and aggr_ratio_15m <= 0.45:
            flow_score += 4   # Moderate selling
        
        # Delta confirmation
        if delta_cumsum_15m and delta_cumsum_15m < 0:
            flow_score += 8
    
    flow_score = min(20, flow_score)
    score += flow_score
    breakdown["order_flow"] = flow_score
    breakdown["aggr_ratio_15m"] = aggr_ratio_15m
    
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
def detect_leverage_opportunity_v3(
    symbol: str,
    cache: MarketCache
) -> Optional[Dict[str, Any]]:
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
    
    # Calculate BOTH directions
    long_quality = calculate_quality_score_v3(
        symbol, df15m, df5m, df1h, df4h, "LONG"
    )
    
    short_quality = calculate_quality_score_v3(
        symbol, df15m, df5m, df1h, df4h, "SHORT"
    )
    
    long_score = long_quality.get("score", 0)
    short_score = short_quality.get("score", 0)
    
    # Determine which direction is better
    direction = None
    quality = None
    
    if long_score >= QUALITY_SCORE_WATCH and long_score > short_score + MIN_SCORE_DIFF:
        direction = "LONG"
        quality = long_quality
    elif short_score >= QUALITY_SCORE_WATCH and short_score > long_score + MIN_SCORE_DIFF:
        direction = "SHORT"
        quality = short_quality
    else:
        # Scores too close or both too low → chop → no signal
        return None
    
    score = quality["score"]
    
    # Determine signal level
    signal_level = None
    if score >= QUALITY_SCORE_PERFECT:
        signal_level = "PERFECT_ENTRY"
    elif score >= QUALITY_SCORE_ENTRY:
        signal_level = "ENTRY_READY"
    else:
        signal_level = "WATCH"
    
    # Get current price
    px = safe_float(df15m.loc[i15m, "close"])
    mark = fetch_mark_price(symbol) or px
    
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
        "candle_ts_15m": candle_ts_15m,
        "timestamp": time.time(),
    }


# ============================================================
# Info Display Functions (5m/1m snapshot - NOT used for scoring)
# ============================================================
def get_5m_snapshot(df5m: pd.DataFrame) -> Dict[str, Any]:
    """Get 5m data for display purposes only"""
    if df5m is None or len(df5m) < 25:
        return {}
    
    i5m = len(df5m) - 2
    
    vol = safe_float(df5m.loc[i5m, "volume"])
    vol_sma = safe_float(df5m.loc[i5m, "vol_sma20"])
    volx = (vol / vol_sma) if (vol and vol_sma and vol_sma > 0) else 0.0
    
    return {
        "close": safe_float(df5m.loc[i5m, "close"]),
        "rsi": safe_float(df5m.loc[i5m, "rsi14"]),
        "volx": volx,
        "macdh": safe_float(df5m.loc[i5m, "macdh"]),
        "ema20": safe_float(df5m.loc[i5m, "ema20"]),
        "vwap": safe_float(df5m.loc[i5m, "vwap"]),
    }

def get_1m_snapshot(df1m: pd.DataFrame) -> Dict[str, Any]:
    """Get 1m data for display purposes only"""
    if df1m is None or len(df1m) < 25:
        return {}
    
    i1m = len(df1m) - 2
    
    vol = safe_float(df1m.loc[i1m, "volume"])
    vol_sma = safe_float(df1m.loc[i1m, "vol_sma20"])
    volx = (vol / vol_sma) if (vol and vol_sma and vol_sma > 0) else 0.0
    
    return {
        "close": safe_float(df1m.loc[i1m, "close"]),
        "rsi": safe_float(df1m.loc[i1m, "rsi14"]),
        "volx": volx,
        "macdh": safe_float(df1m.loc[i1m, "macdh"]),
    }


# ============================================================
# Report Generation (V3: Shows 5m/1m as info, scores from 15m)
# ============================================================
def generate_leverage_report_v3(signal: Dict[str, Any], cache: MarketCache) -> str:
    """
    V3: Generate report showing that scoring is 15m-based
    """
    symbol = signal["symbol"]
    direction = signal["direction"]
    level = signal["signal_level"]
    score = signal["quality_score"]
    breakdown = signal["quality_breakdown"]
    htf_ctx = signal["htf_context"]
    long_score = signal["long_score"]
    short_score = signal["short_score"]
    
    # Get info snapshots
    df5m = cache.get(symbol, "5m")
    df1m = cache.get(symbol, "1m")
    snap_5m = get_5m_snapshot(df5m)
    snap_1m = get_1m_snapshot(df1m)
    
    # Emojis
    emoji = {
        "PERFECT_ENTRY": "🚀💎",
        "ENTRY_READY": "✅🎯",
        "WATCH": "👀⚠️"
    }.get(level, "📊")
    
    lines = [
        f"{emoji} {level} | {symbol} {direction}",
        f"Quality Score: {score}/100 (15m-based)",
        f"Competing scores: LONG {long_score} | SHORT {short_score}",
        f"Time: {utc_now_str()}",
        "=" * 60,
        "",
        f"Price: {signal['price']:.2f} | Mark: {signal['mark_price']:.2f}",
        "",
        "📊 QUALITY BREAKDOWN (15m + HTF):",
        f"  HTF Alignment: {breakdown.get('htf', 0)}/20",
        f"  15m RSI: {breakdown.get('rsi', 0)}/15",
        f"  15m MACD: {breakdown.get('macd', 0)}/10",
        f"  15m Proximity: {breakdown.get('proximity', 0)}/15",
        f"  15m Volume: {breakdown.get('volume', 0)}/10",
        f"  15m Order Flow: {breakdown.get('order_flow', 0)}/20",
        f"  MTF Bonus (5m): {breakdown.get('mtf_bonus', 0)}/10",
        "",
        "🌍 HTF CONTEXT:",
        f"  1h: {'ABOVE' if htf_ctx.get('1h_above_200') else 'BELOW'} EMA200 ({htf_ctx.get('1h_strength', 'N/A')})",
        f"  4h: {'ABOVE' if htf_ctx.get('4h_above_200') else 'BELOW'} EMA200 ({htf_ctx.get('4h_strength', 'N/A')})",
        f"  1h Dist: {htf_ctx.get('dist_1h_200', 0):.2f}%",
        f"  4h Dist: {htf_ctx.get('dist_4h_200', 0):.2f}%",
        "",
        "📈 15m METRICS (SCORING SOURCE):",
        f"  RSI: {breakdown.get('rsi_15m', 0):.1f}",
        f"  Volume: {breakdown.get('volx_15m', 0):.2f}x",
        f"  Dist EMA20: {breakdown.get('dist_ema20_atr', 0):.2f} ATR",
        f"  Dist VWAP: {breakdown.get('dist_vwap_atr', 0):.2f} ATR",
        f"  Aggressor Ratio: {breakdown.get('aggr_ratio_15m', 0):.1%}",
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
        "1. Does this 15m setup have good entry timing?",
        "2. How does HTF context support this direction?",
        "3. Is the score high enough for leverage entry?",
        "4. What are the key risks?",
        "5. Suggested entry zones and stop-loss?",
    ]
    
    return "\n".join(lines)


def generate_manual_report_v3(symbol: str, cache: MarketCache) -> str:
    """V3: Manual report with comprehensive data"""
    cache.refresh_15m_force(symbol)
    cache.refresh_5m_force(symbol)
    cache.refresh_1m_force(symbol)
    cache.ensure_context_ready(symbol)
    
    # Try to detect
    signal = detect_leverage_opportunity_v3(symbol, cache)
    
    if signal:
        return generate_leverage_report_v3(signal, cache)
    
    # No signal - comprehensive status
    df15m = cache.get(symbol, "15m")
    df5m = cache.get(symbol, "5m")
    df1m = cache.get(symbol, "1m")
    df1h = cache.get(symbol, "1h")
    df4h = cache.get(symbol, "4h")
    
    i15m = len(df15m) - 2
    
    px = safe_float(df15m.loc[i15m, "close"])
    mark = fetch_mark_price(symbol) or px
    
    # Get both scores
    long_q = calculate_quality_score_v3(symbol, df15m, df5m, df1h, df4h, "LONG")
    short_q = calculate_quality_score_v3(symbol, df15m, df5m, df1h, df4h, "SHORT")
    
    long_score = long_q.get("score", 0)
    short_score = short_q.get("score", 0)
    
    # Get snapshots
    snap_5m = get_5m_snapshot(df5m)
    snap_1m = get_1m_snapshot(df1m)
    
    # Get 15m data
    rsi_15m = safe_float(df15m.loc[i15m, "rsi14"])
    vol_15m = safe_float(df15m.loc[i15m, "volume"])
    vol_avg_15m = safe_float(df15m.loc[i15m, "vol_sma20"])
    volx_15m = (vol_15m / vol_avg_15m) if vol_avg_15m and vol_avg_15m > 0 else 0
    ema20_15m = safe_float(df15m.loc[i15m, "ema20"])
    ema50_15m = safe_float(df15m.loc[i15m, "ema50"])
    vwap_15m = safe_float(df15m.loc[i15m, "vwap"])
    
    # HTF context
    htf_long, htf_short, htf_ctx = check_htf_alignment(df1h, df4h)
    
    return (
        f"📊 {symbol} COMPREHENSIVE STATUS (v3)\n"
        f"Time: {utc_now_str()}\n"
        f"Price: {px:.2f} | Mark: {mark:.2f}\n"
        f"=" * 60 + "\n\n"
        f"🎯 QUALITY SCORES (15m + HTF based):\n"
        f"  LONG:  {long_score}/100\n"
        f"  SHORT: {short_score}/100\n"
        f"  Status: {'⚠️ No high-quality setup (need ≥60)' if max(long_score, short_score) < QUALITY_SCORE_WATCH else '✅ Setup detected'}\n\n"
        f"🌍 HTF CONTEXT:\n"
        f"  1h: {'ABOVE' if htf_ctx.get('1h_above_200') else 'BELOW'} EMA200 ({htf_ctx.get('1h_strength')})\n"
        f"  4h: {'ABOVE' if htf_ctx.get('4h_above_200') else 'BELOW'} EMA200 ({htf_ctx.get('4h_strength')})\n\n"
        f"📈 15m TIMEFRAME (SCORING SOURCE):\n"
        f"  Close: {px:.2f}\n"
        f"  EMA20: {ema20_15m:.2f} | EMA50: {ema50_15m:.2f}\n"
        f"  VWAP: {vwap_15m:.2f}\n"
        f"  RSI: {rsi_15m:.1f}\n"
        f"  Volume: {volx_15m:.2f}x\n"
        f"  Aggr Buy: {safe_float(df15m.loc[i15m, 'aggressor_buy_ratio']):.1%}\n\n"
        f"📊 5m INFO (display only):\n"
        f"  Close: {snap_5m.get('close', 0):.2f}\n"
        f"  RSI: {snap_5m.get('rsi', 0):.1f}\n"
        f"  Volume: {snap_5m.get('volx', 0):.2f}x\n"
        f"  EMA20: {snap_5m.get('ema20', 0):.2f} | VWAP: {snap_5m.get('vwap', 0):.2f}\n\n"
        f"⚡ 1m INFO (display only):\n"
        f"  Close: {snap_1m.get('close', 0):.2f}\n"
        f"  RSI: {snap_1m.get('rsi', 0):.1f}\n"
        f"  Volume: {snap_1m.get('volx', 0):.2f}x\n\n"
        f"💡 SCORE BREAKDOWN:\n"
        f"  HTF: {long_q['breakdown'].get('htf', 0)}/20 (L) | {short_q['breakdown'].get('htf', 0)}/20 (S)\n"
        f"  15m RSI: {long_q['breakdown'].get('rsi', 0)}/15 (L) | {short_q['breakdown'].get('rsi', 0)}/15 (S)\n"
        f"  15m MACD: {long_q['breakdown'].get('macd', 0)}/10 (L) | {short_q['breakdown'].get('macd', 0)}/10 (S)\n"
        f"  15m Proximity: {long_q['breakdown'].get('proximity', 0)}/15\n"
        f"  15m Volume: {long_q['breakdown'].get('volume', 0)}/10\n"
        f"  15m Flow: {long_q['breakdown'].get('order_flow', 0)}/20 (L) | {short_q['breakdown'].get('order_flow', 0)}/20 (S)\n"
        f"  MTF Bonus: {long_q['breakdown'].get('mtf_bonus', 0)}/10 (L) | {short_q['breakdown'].get('mtf_bonus', 0)}/10 (S)\n\n"
        f"=" * 60 + "\n"
        f"🤖 FOR GPT ANALYSIS:\n\n"
        f"Scores are LOW. What needs to change on 15m for a signal?\n"
        f"Current weaknesses: [analyze breakdown above]\n"
        f"HTF alignment: {'✓' if htf_long or htf_short else '✗'}\n"
        f"Suggested action: Wait for 15m momentum to build.\n"
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
        """Check if we should send this alert"""
        
        if self.last_signal_time == 0:
            return True
        
        # Different direction - allow immediately
        if direction != self.last_direction:
            return True
        
        time_since_last = current_time - self.last_signal_time
        
        # Level escalation - allow
        level_order = {"WATCH": 0, "ENTRY_READY": 1, "PERFECT_ENTRY": 2}
        if level_order.get(level, 0) > level_order.get(self.last_level, 0):
            return True
        
        # Same or lower level - check cooldown
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
def parse_command(text: str):
    if not text:
        return None, None
    parts = text.strip().split()
    cmd = parts[0].lower()
    arg = parts[1].upper().replace("/", "") if len(parts) > 1 else None
    return cmd, arg


# ============================================================
# Main Loop (V3: 15m-only scanning)
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
    
    if SEND_STARTUP_MESSAGE:
        tg_send_message(
            "🚀 HTF+15M RADAR BOT v3 ONLINE\n"
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
                            for sym in SYMBOLS:
                                report = generate_manual_report_v3(sym, cache)
                                tg_send_message(report)
                                time.sleep(1.0)
                        else:
                            if arg in SYMBOLS:
                                report = generate_manual_report_v3(arg, cache)
                                tg_send_message(report)
                            else:
                                tg_send_message(f"❌ Invalid symbol. Use: {', '.join(SYMBOLS)}")
                    
                    elif cmd == "/status":
                        uptime_min = int((time.time() - start_ts) / 60)
                        next_15m, eta_15m = next_close_eta("15m")
                        
                        tg_send_message(
                            "📡 BOT STATUS (v3)\n"
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
                
                # SCAN ONLY ON 15M CLOSE
                if new_15m:
                    # Refresh 5m/1m ONLY when scanning (not every loop)
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
                        f"💓 HEARTBEAT (v3)\n"
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
                    "⚠️ BOT ERROR (v3)\n"
                    f"UTC: {utc_now_str()}\n\n"
                    f"{err[:3000]}"
                )
                last_error_notify = now
        
        # 4) Sleep
        elapsed = time.time() - loop_start
        sleep_for = max(0.5, SLEEP_TARGET_SECONDS - elapsed)
        time.sleep(sleep_for)


if __name__ == "__main__":
    main()
