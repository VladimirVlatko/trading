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
# CONFIG
# ============================================================
BINANCE_FAPI = "https://fapi.binance.com"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

# Loop tick: keeps Telegram responsive. Binance calls are candle-close driven.
SLEEP_TARGET_SECONDS = 1

# Binance request hygiene (prevents 418/429 bans)
MIN_BINANCE_INTERVAL = 0.35  # seconds between ANY Binance REST calls
BAN_BACKOFF_DEFAULT = 60     # seconds to sleep on 418/429 if Retry-After missing

# HTTP timeouts
HTTP_TIMEOUT = 12
TELEGRAM_POLL_TIMEOUT = 2  # long poll

# Candle limits
OHLCV_LIMIT_5M = 220
OHLCV_LIMIT_15M = 220
OHLCV_LIMIT_HTF = 260

# Trigger parameters (15m main trigger)
VOL_SPIKE_MULT = 1.5
# For radar alerts (trend presence). VOL_SPIKE is now a bonus, not a requirement.
VOL_OK_MIN = 0.55  # minimum volume ratio (vol / SMA20) to consider move meaningful
RSI_LONG_MIN = 52.0
RSI_SHORT_MAX = 48.0

RSI_MIN, RSI_MAX = 25, 75

# MACD slope threshold (ATR scaled): require macdh slope magnitude >= ATR * this factor
MACD_SLOPE_ATR_FACTOR = 0.03  # 0.03 * ATR (tune 0.02–0.06)

# 5m data (snapshot only). We do NOT use 5m for signals.
ENABLE_5M_HINT_ALERTS = False  # keep 5m in reports, disable 5m-triggered alerts

# 5m early layer (WATCH / ENTRY_OK), only on 5m candle close
WATCH_VOLX_5M = 1.15
ENTRY_VOLX_5M = 1.30
MACD_SLOPE_ATR_FACTOR_5M = 0.02
NEAR_READY_DIST_ATR_15M = 0.80  # how close (in 15m ATR) price must be to 15m EMA20 to start 5m monitoring

# Telegram max length (single message)
MAX_TG_LEN = 3500

# Error reporting rate-limit
ERROR_NOTIFY_COOLDOWN = 120  # seconds

# Startup message
SEND_STARTUP_MESSAGE = True

# Heartbeat (to prove the bot is alive even if Telegram polling is blocked by 409)
HEARTBEAT_MINUTES = 720  # set 0 to disable
HEARTBEAT_TEXT = "🫀 HEARTBEAT (bot alive)"

# When Telegram getUpdates returns 409 (another poller exists), wait this many seconds before retrying
TG_CONFLICT_BACKOFF = 30

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "momentum-entrywatch/1.1"})

# Global throttling state
_BINANCE_LAST_CALL_TS = 0.0
_BINANCE_BAN_UNTIL = 0.0


# ============================================================
# Helpers: time / formatting
# ============================================================
def utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def utc_ts(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def interval_seconds(tf: str) -> int:
    return {"5m": 300, "15m": 900, "1h": 3600, "4h": 14400}[tf]

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

def next_15m_close_eta() -> Tuple[str, float]:
    return next_close_eta("15m")

def next_5m_close_eta() -> Tuple[str, float]:
    return next_close_eta("5m")

def safe_float(x, default=None):
    try:
        v = float(x)
        if pd.isna(v):
            return default
        return v
    except Exception:
        return default

def pct(a: float, b: float, default=float("nan")) -> float:
    """Return (a - b) / b * 100 in %, safe for zero/NaN."""
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
# Binance Futures public REST
# ============================================================
def http_get(path: str, params: Dict[str, Any]) -> Union[Dict[str, Any], List[Any]]:
    """
    Binance REST with basic global throttling + sane backoff.
    Prevents IP bans (418) when running bots on small hosts.
    """
    global _BINANCE_LAST_CALL_TS, _BINANCE_BAN_UNTIL

    now = time.time()
    # If currently banned/backing off, wait (but do NOT busy-loop)
    if now < _BINANCE_BAN_UNTIL:
        time.sleep(max(0.0, _BINANCE_BAN_UNTIL - now))
        now = time.time()

    # Global minimum spacing between calls (across all endpoints/symbols)
    wait = MIN_BINANCE_INTERVAL - (now - _BINANCE_LAST_CALL_TS)
    if wait > 0:
        time.sleep(wait)

    url = BINANCE_FAPI + path
    r = SESSION.get(url, params=params, timeout=HTTP_TIMEOUT)
    _BINANCE_LAST_CALL_TS = time.time()

    # Handle rate-limit / bans gracefully
    if r.status_code in (418, 429):
        retry_after = r.headers.get("Retry-After")
        backoff = int(retry_after) if (retry_after and str(retry_after).isdigit()) else BAN_BACKOFF_DEFAULT
        _BINANCE_BAN_UNTIL = time.time() + max(10, backoff)
        raise requests.HTTPError(f"{r.status_code} rate-limited/banned (backoff {backoff}s)", response=r)

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
# Telegram (polling)
# ============================================================
def tg_api(method: str) -> str:
    return f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/{method}"

def tg_get_updates(offset: Optional[int]) -> dict:
    params = {"timeout": TELEGRAM_POLL_TIMEOUT}
    if offset is not None:
        params["offset"] = offset
    r = SESSION.get(tg_api("getUpdates"), params=params, timeout=TELEGRAM_POLL_TIMEOUT + 10)
    # 409 = another poller existed briefly (deploy overlap). Don't treat as error.
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
# Indicators
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

    # Delta proxy: taker buy vs sell (base volume)
    out["taker_buy"] = out["taker_buy_base"]
    out["taker_sell"] = out["volume"] - out["taker_buy"]

    out = out.reset_index(drop=True)
    return out

def add_indicators_15m(df: pd.DataFrame) -> pd.DataFrame:
    return _add_indicators_core(df)

def add_indicators_5m(df: pd.DataFrame) -> pd.DataFrame:
    return _add_indicators_core(df)

def add_ema_pack(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ema20"] = ta.ema(out["close"], length=20)
    out["ema50"] = ta.ema(out["close"], length=50)
    out["ema200"] = ta.ema(out["close"], length=200)
    return out


# ============================================================
# Cache (refresh only when new candle closes)
# ============================================================
@dataclass
class TFCached:
    df: Optional[pd.DataFrame] = None
    last_closed_open_ms: int | None = None

class MarketCache:
    def __init__(self):
        self.data = {
            s: {"5m": TFCached(), "15m": TFCached(), "1h": TFCached(), "4h": TFCached()}
            for s in SYMBOLS
        }

    # ---- 15m (main) ----
    def refresh_15m_force(self, symbol: str):
        df = fetch_klines_df(symbol, "15m", OHLCV_LIMIT_15M)
        df = add_indicators_15m(df)
        self.data[symbol]["15m"] = TFCached(df=df, last_closed_open_ms=last_closed_open_ms("15m"))

    def refresh_15m_if_needed(self, symbol: str):
        expected = last_closed_open_ms("15m")
        cached = self.data[symbol]["15m"]
        if cached.df is None or cached.last_closed_open_ms != expected:
            self.refresh_15m_force(symbol)

    # ---- 5m (early layer) ----
    def refresh_5m_force(self, symbol: str):
        df = fetch_klines_df(symbol, "5m", OHLCV_LIMIT_5M)
        df = add_indicators_5m(df)
        self.data[symbol]["5m"] = TFCached(df=df, last_closed_open_ms=last_closed_open_ms("5m"))

    def refresh_5m_if_needed(self, symbol: str):
        expected = last_closed_open_ms("5m")
        cached = self.data[symbol]["5m"]
        if cached.df is None or cached.last_closed_open_ms != expected:
            self.refresh_5m_force(symbol)

    # ---- HTF ----
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
# Entry trigger (15m LAST CLOSED, cross-based, HTF-filtered)
# ============================================================
def check_entry_trigger(symbol: str, df15: pd.DataFrame, df1h: pd.DataFrame, df4h: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """15m-close radar trigger (NO 5m logic).
    Goal: catch *start / presence* of a meaningful HTF-aligned trend, without being overly strict.
    - HTF (1h & 4h) EMA200 alignment is a hard filter.
    - 15m conditions define TREND_ACTIVE.
    - We alert on a *state change* (trend becomes active now vs previous 15m close).
    """
    i = len(df15) - 2
    p = i - 1
    if i < 210 or p < 209:
        return None

    needed = ["close", "vwap", "ema20", "ema50", "rsi14", "volume", "vol_sma20", "macdh"]
    if not all(c in df15.columns for c in needed):
        return None
    for c in needed:
        if pd.isna(df15.loc[i, c]) or pd.isna(df15.loc[p, c]):
            return None

    # HTF: last closed candles
    i1h = len(df1h) - 2
    i4h = len(df4h) - 2
    if i1h < 205 or i4h < 205:
        return None
    if any(pd.isna(df1h.loc[i1h, x]) for x in ["close", "ema200"]) or any(pd.isna(df4h.loc[i4h, x]) for x in ["close", "ema200"]):
        return None

    c1h, e1h = float(df1h.loc[i1h, "close"]), float(df1h.loc[i1h, "ema200"])
    c4h, e4h = float(df4h.loc[i4h, "close"]), float(df4h.loc[i4h, "ema200"])
    htf_long_ok = (c1h > e1h) and (c4h > e4h)
    htf_short_ok = (c1h < e1h) and (c4h < e4h)

    # 15m values (current + previous)
    c_curr = float(df15.loc[i, "close"])
    v_curr = float(df15.loc[i, "vwap"])
    e20_curr = float(df15.loc[i, "ema20"])
    e50_curr = float(df15.loc[i, "ema50"])
    rsi_curr = float(df15.loc[i, "rsi14"])
    mh_curr = float(df15.loc[i, "macdh"])

    c_prev = float(df15.loc[p, "close"])
    v_prev = float(df15.loc[p, "vwap"])
    e20_prev = float(df15.loc[p, "ema20"])
    e50_prev = float(df15.loc[p, "ema50"])
    rsi_prev = float(df15.loc[p, "rsi14"])
    mh_prev = float(df15.loc[p, "macdh"])

    vol_curr = float(df15.loc[i, "volume"])
    vol_avg = float(df15.loc[i, "vol_sma20"])
    if vol_avg <= 0:
        return None
    vol_ratio = vol_curr / vol_avg

    # TREND_ACTIVE conditions (15m)
    above_vwap = c_curr > v_curr
    below_vwap = c_curr < v_curr
    ema_trend_up = e20_curr > e50_curr
    ema_trend_dn = e20_curr < e50_curr

    rsi_long_ok = rsi_curr >= RSI_LONG_MIN
    rsi_short_ok = rsi_curr <= RSI_SHORT_MAX

    macd_hist_up = mh_curr > 0
    macd_hist_dn = mh_curr < 0

    vol_ok = vol_ratio >= VOL_OK_MIN
    # VOL spike is a bonus tag in the report, not a requirement
    vol_spike = vol_ratio >= VOL_SPIKE_MULT

    trend_long_now = htf_long_ok and above_vwap and ema_trend_up and rsi_long_ok and macd_hist_up and vol_ok
    trend_short_now = htf_short_ok and below_vwap and ema_trend_dn and rsi_short_ok and macd_hist_dn and vol_ok

    # previous trend state (for state-change alerting)
    trend_long_prev = htf_long_ok and (c_prev > v_prev) and (e20_prev > e50_prev) and (rsi_prev >= RSI_LONG_MIN) and (mh_prev > 0) and vol_ok
    trend_short_prev = htf_short_ok and (c_prev < v_prev) and (e20_prev < e50_prev) and (rsi_prev <= RSI_SHORT_MAX) and (mh_prev < 0) and vol_ok

    candle_ts = int(df15.loc[i, "ts"])

    # Alert on state change: trend becomes active
    if trend_long_now and not trend_long_prev:
        return {"symbol": symbol, "direction": "LONG", "vol_ratio": vol_ratio, "i": i, "candle_ts": candle_ts, "tag": "TREND_ACTIVE", "vol_spike": vol_spike}
    if trend_short_now and not trend_short_prev:
        return {"symbol": symbol, "direction": "SHORT", "vol_ratio": vol_ratio, "i": i, "candle_ts": candle_ts, "tag": "TREND_ACTIVE", "vol_spike": vol_spike}

    return None


# ============================================================
# 5m Early layer: near-ready + WATCH / ENTRY_OK: candle_ts}
    return None


# ============================================================
# 5m Early layer: near-ready + WATCH / ENTRY_OK
# ============================================================
def _htf_ctx(df1h: pd.DataFrame, df4h: pd.DataFrame) -> Tuple[bool, bool, str, str]:
    i1h, i4h = len(df1h) - 2, len(df4h) - 2
    c1h, e1h = safe_float(df1h.loc[i1h, "close"]), safe_float(df1h.loc[i1h, "ema200"])
    c4h, e4h = safe_float(df4h.loc[i4h, "close"]), safe_float(df4h.loc[i4h, "ema200"])
    htf_long_ok = (c1h is not None and e1h is not None and c4h is not None and e4h is not None and (c1h > e1h) and (c4h > e4h))
    htf_short_ok = (c1h is not None and e1h is not None and c4h is not None and e4h is not None and (c1h < e1h) and (c4h < e4h))
    ctx_1h = "ABOVE" if (c1h is not None and e1h is not None and c1h > e1h) else "BELOW"
    ctx_4h = "ABOVE" if (c4h is not None and e4h is not None and c4h > e4h) else "BELOW"
    return htf_long_ok, htf_short_ok, ctx_1h, ctx_4h

def evaluate_5m_hint(symbol: str, df5: pd.DataFrame, df15: pd.DataFrame, df1h: pd.DataFrame, df4h: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Runs on LAST CLOSED 5m candle.
    Uses 15m/HTF as context, but attempts to catch momentum while the current 15m is building.
    """
    i5 = len(df5) - 2
    if i5 < 60:
        return None
    i15 = len(df15) - 2
    if i15 < 210:
        return None

    # Guard required cols
    need5 = ["close", "ema20", "rsi14", "volume", "vol_sma20", "macdh", "atr14"]
    need15 = ["ema20", "ema50", "vwap", "atr14", "ema200"]
    if not all(c in df5.columns for c in need5) or not all(c in df15.columns for c in need15):
        return None
    if any(pd.isna(df5.loc[i5, c]) for c in need5) or any(pd.isna(df15.loc[i15, c]) for c in need15):
        return None

    # HTF
    htf_long_ok, htf_short_ok, ctx_1h, ctx_4h = _htf_ctx(df1h, df4h)

    # Price proxy: last closed 5m close (keeps us from calling markPrice too often)
    px = float(df5.loc[i5, "close"])

    # 15m context levels (last closed 15m)
    ema20_15 = float(df15.loc[i15, "ema20"])
    ema50_15 = float(df15.loc[i15, "ema50"])
    vwap_15 = float(df15.loc[i15, "vwap"])
    atr15 = float(df15.loc[i15, "atr14"])

    bias_longish = (px > vwap_15) and (ema20_15 > ema50_15)
    bias_shortish = (px < vwap_15) and (ema20_15 < ema50_15)

    zone = dist_atr(px, ema20_15, atr15)
    near_zone = (zone == zone) and (zone <= NEAR_READY_DIST_ATR_15M)

    near_ready_long = htf_long_ok and near_zone and (bias_longish or (px > ema20_15))
    near_ready_short = htf_short_ok and near_zone and (bias_shortish or (px < ema20_15))

    if not (near_ready_long or near_ready_short):
        return None

    # 5m momentum metrics (last closed)
    vol = float(df5.loc[i5, "volume"])
    vol_avg = float(df5.loc[i5, "vol_sma20"])
    volx = (vol / vol_avg) if vol_avg > 0 else float("nan")
    rsi = float(df5.loc[i5, "rsi14"])
    atr5 = float(df5.loc[i5, "atr14"])
    macdh = float(df5.loc[i5, "macdh"])
    macdh_prev = safe_float(df5.loc[i5 - 1, "macdh"], default=float("nan"))
    macd_slope = macdh - macdh_prev if (macdh_prev == macdh_prev) else float("nan")
    slope_thr = max(1e-12, atr5 * MACD_SLOPE_ATR_FACTOR_5M)

    ema20_5 = float(df5.loc[i5, "ema20"])
    candle_ts = int(df5.loc[i5, "ts"])
    candle_utc = utc_ts(candle_ts + interval_seconds("5m") * 1000)

    def pack(state: str, direction: str, why: str) -> dict:
        return {
            "symbol": symbol,
            "state": state,
            "direction": direction,
            "utc": candle_utc,
            "candle_ts": candle_ts,
            "px": px,
            "volx": volx,
            "rsi": rsi,
            "macd_slope": macd_slope,
            "slope_thr": slope_thr,
            "zone_atr15": zone,
            "ctx_1h": ctx_1h,
            "ctx_4h": ctx_4h,
            "why": why,
        }

    # Long side
    if near_ready_long:
        watch_ok = (volx == volx and volx >= WATCH_VOLX_5M) and (rsi >= 35) and (macd_slope == macd_slope and macd_slope > +slope_thr)
        entry_ok = watch_ok and (volx >= ENTRY_VOLX_5M) and (px > ema20_5)
        if entry_ok:
            return pack("ENTRY_OK", "LONG", "near_ready + 5m momentum confirmed")
        if watch_ok:
            return pack("SETUP_WATCH", "LONG", "near_ready + 5m momentum building")

    # Short side
    if near_ready_short:
        watch_ok = (volx == volx and volx >= WATCH_VOLX_5M) and (rsi <= 65) and (macd_slope == macd_slope and macd_slope < -slope_thr)
        entry_ok = watch_ok and (volx >= ENTRY_VOLX_5M) and (px < ema20_5)
        if entry_ok:
            return pack("ENTRY_OK", "SHORT", "near_ready + 5m momentum confirmed")
        if watch_ok:
            return pack("SETUP_WATCH", "SHORT", "near_ready + 5m momentum building")

    return None

def render_5m_hint(h: Dict[str, Any]) -> str:
    return (
        f"👀 {h['symbol']} — {h['state']} | {h['direction']} | {h['utc']}\n"
        f"HTF: 1h {h['ctx_1h']} | 4h {h['ctx_4h']} | zone15m={h['zone_atr15']:.2f}ATR\n"
        f"5m: p {h['px']:.2f} | VOLx {h['volx']:.2f} | RSI {h['rsi']:.1f} | "
        f"MACD_slope {h['macd_slope']:.4f} (thr {h['slope_thr']:.4f})\n"
        f"Why: {h['why']}"
    )


# ============================================================
# Diagnostics for reports (used by manual reports and alerts)
# ============================================================
def summarize_trigger_state(df15: pd.DataFrame, df1h: pd.DataFrame, df4h: pd.DataFrame, i: int) -> dict:
    """Diagnostics for reports (15m logic only).
    Note: We keep cross flags for context, but triggers are based on TREND_ACTIVE (hold/trend), not crosses.
    """
    p = i - 1
    i1h, i4h = len(df1h) - 2, len(df4h) - 2
    c1h, e1h = safe_float(df1h.loc[i1h, "close"]), safe_float(df1h.loc[i1h, "ema200"])
    c4h, e4h = safe_float(df4h.loc[i4h, "close"]), safe_float(df4h.loc[i4h, "ema200"])
    htf_long_ok = (c1h is not None and e1h is not None and c4h is not None and e4h is not None
                   and (c1h > e1h) and (c4h > e4h))
    htf_short_ok = (c1h is not None and e1h is not None and c4h is not None and e4h is not None
                    and (c1h < e1h) and (c4h < e4h))

    close_prev, close_curr = safe_float(df15.loc[p, "close"]), safe_float(df15.loc[i, "close"])
    vwap_prev, vwap_curr = safe_float(df15.loc[p, "vwap"]), safe_float(df15.loc[i, "vwap"])
    e20_prev, e20_curr = safe_float(df15.loc[p, "ema20"]), safe_float(df15.loc[i, "ema20"])
    e50_prev, e50_curr = safe_float(df15.loc[p, "ema50"]), safe_float(df15.loc[i, "ema50"])

    # Cross flags are informational
    vwap_cross_up = (close_prev is not None and vwap_prev is not None and close_curr is not None and vwap_curr is not None
                     and (close_prev <= vwap_prev) and (close_curr > vwap_curr))
    vwap_cross_dn = (close_prev is not None and vwap_prev is not None and close_curr is not None and vwap_curr is not None
                     and (close_prev >= vwap_prev) and (close_curr < vwap_curr))
    ema_cross_up = (e20_prev is not None and e50_prev is not None and e20_curr is not None and e50_curr is not None
                    and (e20_prev <= e50_prev) and (e20_curr > e50_curr))
    ema_cross_dn = (e20_prev is not None and e50_prev is not None and e20_curr is not None and e50_curr is not None
                    and (e20_prev >= e50_prev) and (e20_curr < e50_curr))

    vol = safe_float(df15.loc[i, "volume"])
    vol_avg = safe_float(df15.loc[i, "vol_sma20"], default=0.0)
    vol_ratio = (vol / vol_avg) if (vol_avg and vol_avg > 0) else float("nan")
    vol_ok = (vol_ratio == vol_ratio and vol_ratio >= VOL_OK_MIN)
    vol_spike = (vol_ratio == vol_ratio and vol_ratio >= VOL_SPIKE_MULT)

    rsi = safe_float(df15.loc[i, "rsi14"])
    rsi_long_ok = (rsi is not None and rsi >= RSI_LONG_MIN)
    rsi_short_ok = (rsi is not None and rsi <= RSI_SHORT_MAX)

    mh_curr = safe_float(df15.loc[i, "macdh"])
    mh_prev = safe_float(df15.loc[p, "macdh"])
    macd_hist_up = (mh_curr is not None and mh_curr > 0)
    macd_hist_dn = (mh_curr is not None and mh_curr < 0)
    macd_slope = (mh_curr - mh_prev) if (mh_curr is not None and mh_prev is not None) else float("nan")

    # Trend/hold conditions
    above_vwap = (close_curr is not None and vwap_curr is not None and close_curr > vwap_curr)
    below_vwap = (close_curr is not None and vwap_curr is not None and close_curr < vwap_curr)
    ema_trend_up = (e20_curr is not None and e50_curr is not None and e20_curr > e50_curr)
    ema_trend_dn = (e20_curr is not None and e50_curr is not None and e20_curr < e50_curr)

    missing_long = []
    if not htf_long_ok:      missing_long.append("HTF_LONG")
    if not above_vwap:       missing_long.append("ABOVE_VWAP")
    if not ema_trend_up:     missing_long.append("EMA20>EMA50")
    if not vol_ok:           missing_long.append("VOL_OK")
    if not rsi_long_ok:      missing_long.append("RSI_LONG_OK")
    if not macd_hist_up:     missing_long.append("MACD_HIST_POS")
    # VOL_SPIKE is informational (bonus), not required
    trigger_long = (len(missing_long) == 0)

    missing_short = []
    if not htf_short_ok:     missing_short.append("HTF_SHORT")
    if not below_vwap:       missing_short.append("BELOW_VWAP")
    if not ema_trend_dn:     missing_short.append("EMA20<EMA50")
    if not vol_ok:           missing_short.append("VOL_OK")
    if not rsi_short_ok:     missing_short.append("RSI_SHORT_OK")
    if not macd_hist_dn:     missing_short.append("MACD_HIST_NEG")
    trigger_short = (len(missing_short) == 0)

    return {
        "htf_long_ok": htf_long_ok,
        "htf_short_ok": htf_short_ok,
        "vwap_cross_up": vwap_cross_up,
        "vwap_cross_dn": vwap_cross_dn,
        "ema_cross_up": ema_cross_up,
        "ema_cross_dn": ema_cross_dn,
        "above_vwap": above_vwap,
        "below_vwap": below_vwap,
        "ema_trend_up": ema_trend_up,
        "ema_trend_dn": ema_trend_dn,
        "vol_ratio": vol_ratio,
        "vol_ok": vol_ok,
        "vol_spike": vol_spike,
        "rsi": rsi,
        "rsi_long_ok": rsi_long_ok,
        "rsi_short_ok": rsi_short_ok,
        "macd_hist": mh_curr,
        "macd_slope": macd_slope,
        "trigger_long": trigger_long,
        "trigger_short": trigger_short,
        "missing_long": missing_long,
        "missing_short": missing_short,
    }


# ============================================================
# Report building (single-message, compacting automatically)
# ============================================================
def _fmt(v: Optional[float], digits: int = 2) -> str:
    if v is None or not (v == v):
        return "NA"
    return f"{v:.{digits}f}"


def five_min_snapshot(df5: Optional[pd.DataFrame]) -> Optional[Dict[str, float]]:
    """Return last CLOSED 5m snapshot (numbers only). Used for display, NOT for logic."""
    if df5 is None or len(df5) < 25:
        return None
    i5 = len(df5) - 2
    need = ["close", "volume", "vol_sma20", "rsi14", "macdh", "atr14"]
    if not all(c in df5.columns for c in need):
        return None
    if any(pd.isna(df5.loc[i5, c]) for c in need):
        return None

    px = float(df5.loc[i5, "close"])
    vol = float(df5.loc[i5, "volume"])
    vol_avg = float(df5.loc[i5, "vol_sma20"])
    volx = (vol / vol_avg) if vol_avg and vol_avg > 0 else float("nan")
    rsi = float(df5.loc[i5, "rsi14"])

    macdh = float(df5.loc[i5, "macdh"])
    macdh_prev = safe_float(df5.loc[i5 - 1, "macdh"], default=float("nan")) if i5 - 1 >= 0 else float("nan")
    macd_slope = (macdh - macdh_prev) if (macdh_prev == macdh_prev) else float("nan")

    atr5 = float(df5.loc[i5, "atr14"])
    slope_thr = max(1e-12, atr5 * MACD_SLOPE_ATR_FACTOR_5M)

    return {"p": px, "volx": volx, "rsi": rsi, "macd_slope": macd_slope, "slope_thr": slope_thr}

def _snapshot_report_data(symbol: str, tag: str, direction: str, vol_ratio: float,
                          df15: pd.DataFrame, df1h: pd.DataFrame, df4h: pd.DataFrame, i15: int,
                          df5: Optional[pd.DataFrame] = None) -> dict:
    mark = fetch_mark_price(symbol)
    close_px = float(df15.loc[i15, "close"])
    mark_px = mark if mark is not None else close_px

    i1h = len(df1h) - 2
    i4h = len(df4h) - 2
    c1h, e1h = float(df1h.loc[i1h, "close"]), float(df1h.loc[i1h, "ema200"])
    c4h, e4h = float(df4h.loc[i4h, "close"]), float(df4h.loc[i4h, "ema200"])
    ctx_1h = "ABOVE" if c1h > e1h else "BELOW"
    ctx_4h = "ABOVE" if c4h > e4h else "BELOW"

    snap5 = five_min_snapshot(df5)

    vwap = float(df15.loc[i15, "vwap"])
    ema20 = float(df15.loc[i15, "ema20"])
    ema50 = float(df15.loc[i15, "ema50"])
    ema200 = float(df15.loc[i15, "ema200"])
    rsi = float(df15.loc[i15, "rsi14"])
    macdh = float(df15.loc[i15, "macdh"])
    atr = float(df15.loc[i15, "atr14"])

    prev = i15 - 1
    mh_prev = safe_float(df15.loc[prev, "macdh"], default=float("nan")) if prev >= 0 else float("nan")
    mh_slope = macdh - mh_prev if mh_prev == mh_prev else float("nan")
    slope_thr = max(1e-12, (atr or 0.0) * MACD_SLOPE_ATR_FACTOR)

    tb = safe_float(df15.loc[i15, "taker_buy"], default=float("nan"))
    ts = safe_float(df15.loc[i15, "taker_sell"], default=float("nan"))
    delta = tb - ts if (tb == tb and ts == ts) else float("nan")

    dist_close_vwap = pct(close_px, vwap)
    dist_close_e50 = pct(close_px, ema50)
    dist_close_e200 = pct(close_px, ema200)

    last_15m_close_utc = utc_ts(int(df15.loc[i15, "ts"]) + interval_seconds("15m") * 1000)

    diag = summarize_trigger_state(df15, df1h, df4h, i15)

    return {
        "snap5": snap5,
        "symbol": symbol,
        "tag": tag,
        "direction": direction,
        "utc": last_15m_close_utc,
        "ctx_1h": ctx_1h,
        "ctx_4h": ctx_4h,
        "close_px": close_px,
        "mark_px": mark_px,
        "vwap": vwap,
        "ema20": ema20,
        "ema50": ema50,
        "ema200": ema200,
        "rsi": rsi,
        "vol_ratio": vol_ratio,
        "atr": atr,
        "macdh": macdh,
        "macd_slope": mh_slope,
        "slope_thr": slope_thr,
        "dist_close_vwap": dist_close_vwap,
        "dist_close_e50": dist_close_e50,
        "dist_close_e200": dist_close_e200,
        "taker_buy": tb,
        "taker_sell": ts,
        "delta": delta,
        "diag": diag,
    }

def _render_snapshot_block(data: Dict[str, Any], compact_level: int = 0) -> str:
    diag = data["diag"]
    lines = [
        f"🧭 {data['symbol']} | {data['direction']} | {data['utc']}",
        f"HTF EMA200: 1h {data['ctx_1h']} | 4h {data['ctx_4h']}",
        "15m levels: "
        f"close {_fmt(data['close_px'])} / mark {_fmt(data['mark_px'])} | "
        f"VWAP {_fmt(data['vwap'])} | EMA20 {_fmt(data['ema20'])} | "
        f"EMA50 {_fmt(data['ema50'])} | EMA200 {_fmt(data['ema200'])}",
        f"RSI14 {_fmt(data['rsi'])} | VOLx {_fmt(data['vol_ratio'])} | ATR14 {_fmt(data['atr'])}",
    ]

    # 5m snapshot (display only)
    s5 = data.get('snap5')
    if s5:
        lines.append(
            "5m: "
            f"p {_fmt(s5.get('p'))} | VOLx {_fmt(s5.get('volx'))} | RSI {_fmt(s5.get('rsi'))} | "
            f"MACD_slope {_fmt(s5.get('macd_slope'))} (thr {_fmt(s5.get('slope_thr'))})"
        )


    if compact_level < 3:
        lines.append(
            f"MACD: hist {_fmt(data['macdh'])} | slope {_fmt(data['macd_slope'], 4)} "
            f"(thr {_fmt(data['slope_thr'], 4)})"
        )
    else:
        lines.append(f"MACD slope {_fmt(data['macd_slope'], 4)} (thr {_fmt(data['slope_thr'], 4)})")

    if compact_level < 2:
        lines.append(
            f"Dist %: VWAP {_fmt(data['dist_close_vwap'])} | EMA50 {_fmt(data['dist_close_e50'])} | "
            f"EMA200 {_fmt(data['dist_close_e200'])}"
        )
    else:
        lines.append(f"Dist %: VWAP {_fmt(data['dist_close_vwap'])} | EMA200 {_fmt(data['dist_close_e200'])}")

    if compact_level < 1:
        lines.append(
            f"Delta (taker): buy {_fmt(data['taker_buy'])} | sell {_fmt(data['taker_sell'])} | "
            f"delta {_fmt(data['delta'])}"
        )
    else:
        lines.append(f"Delta: {_fmt(data['delta'])}")

    if compact_level < 4:
        lines.append(
            "Triggers: "
            f"LONG {diag['trigger_long']} (missing: {', '.join(diag['missing_long']) or 'none'}) | "
            f"SHORT {diag['trigger_short']} (missing: {', '.join(diag['missing_short']) or 'none'})"
        )
    else:
        lines.append(f"Triggers: LONG {diag['trigger_long']} | SHORT {diag['trigger_short']}")

    return "\n".join(lines)

def build_snapshot_report(*args, **kwargs) -> str:
    if kwargs:
        if "cache" in kwargs and "symbol" in kwargs:
            return build_manual_report(kwargs["symbol"], kwargs["cache"])
        if all(k in kwargs for k in ["symbol", "tag", "direction", "vol_ratio", "df15", "df1h", "df4h", "i15"]):
            data = _snapshot_report_data(
                kwargs["symbol"], kwargs["tag"], kwargs["direction"], kwargs["vol_ratio"],
                kwargs["df15"], kwargs["df1h"], kwargs["df4h"], kwargs["i15"],
                df5=kwargs.get("df5")
            )
            return _render_snapshot_block(data, compact_level=0)

    if len(args) == 2:
        symbol, cache = args
        return build_manual_report(symbol, cache)

    if len(args) >= 8:
        symbol, tag, direction, vol_ratio, df15, df1h, df4h, i15 = args[:8]
        df5 = args[8] if len(args) > 8 else None
        data = _snapshot_report_data(symbol, tag, direction, vol_ratio, df15, df1h, df4h, i15, df5=df5)
        return _render_snapshot_block(data, compact_level=0)

    raise TypeError("build_snapshot_report() expected (symbol, cache) or full 8-arg signature")

def build_manual_report(symbol: Union[str, 'MarketCache'], cache: Optional[Union['MarketCache', str]] = None) -> str:
    if isinstance(symbol, MarketCache) and isinstance(cache, str):
        symbol, cache = cache, symbol
    if not isinstance(cache, MarketCache):
        raise TypeError("build_manual_report() expected (symbol, cache)")

    cache.refresh_15m_force(symbol)
    cache.ensure_context_ready(symbol)

    df15 = cache.get(symbol, "15m")
    df1h = cache.get(symbol, "1h")
    df4h = cache.get(symbol, "4h")

    i15 = len(df15) - 2
    if i15 < 30:
        return f"❌ Not enough data yet for {symbol}"

    vol = safe_float(df15.loc[i15, "volume"], default=float("nan"))
    vol_avg = safe_float(df15.loc[i15, "vol_sma20"], default=float("nan"))
    vol_ratio = (vol / vol_avg) if (vol_avg == vol_avg and vol_avg > 0) else float("nan")

    direction = "NEUTRAL"
    close = safe_float(df15.loc[i15, "close"])
    vwap = safe_float(df15.loc[i15, "vwap"])
    ema20 = safe_float(df15.loc[i15, "ema20"])
    ema50 = safe_float(df15.loc[i15, "ema50"])
    if close is not None and vwap is not None and ema20 is not None and ema50 is not None:
        if close > vwap and ema20 > ema50:
            direction = "LONG-ish"
        elif close < vwap and ema20 < ema50:
            direction = "SHORT-ish"

    df5 = cache.get(symbol, "5m") if hasattr(cache, "get") else None
    data = _snapshot_report_data(symbol, "MANUAL_REPORT", direction, vol_ratio, df15, df1h, df4h, i15, df5=df5)
    return _render_snapshot_block(data, compact_level=0)

def build_report_message(symbols: List[str], cache: "MarketCache") -> str:
    data_blocks = []
    for s in symbols:
        cache.refresh_15m_force(s)
        cache.ensure_context_ready(s)

        df15 = cache.get(s, "15m")
        # 5m is display-only (snapshot), not used for signals.
        df5 = cache.get(s, "5m")
        df1h = cache.get(s, "1h")
        df4h = cache.get(s, "4h")

        i15 = len(df15) - 2
        if i15 < 30:
            data_blocks.append({"symbol": s, "error": f"❌ Not enough data yet for {s}"})
            continue

        vol = safe_float(df15.loc[i15, "volume"], default=float("nan"))
        vol_avg = safe_float(df15.loc[i15, "vol_sma20"], default=float("nan"))
        vol_ratio = (vol / vol_avg) if (vol_avg == vol_avg and vol_avg > 0) else float("nan")

        direction = "NEUTRAL"
        close = safe_float(df15.loc[i15, "close"])
        vwap = safe_float(df15.loc[i15, "vwap"])
        ema20 = safe_float(df15.loc[i15, "ema20"])
        ema50 = safe_float(df15.loc[i15, "ema50"])
        if close is not None and vwap is not None and ema20 is not None and ema50 is not None:
            if close > vwap and ema20 > ema50:
                direction = "LONG-ish"
            elif close < vwap and ema20 < ema50:
                direction = "SHORT-ish"

        data_blocks.append(
            _snapshot_report_data(
                s,
                "MANUAL_REPORT",
                direction,
                vol_ratio,
                df15,
                df1h,
                df4h,
                i15,
                df5=df5,
            )
        )

    def render(compact_level: int) -> str:
        rendered = []
        for data in data_blocks:
            if "error" in data:
                rendered.append(data["error"])
            else:
                rendered.append(_render_snapshot_block(data, compact_level=compact_level))
        separator = "\n\n" + ("=" * 32) + "\n\n" if len(rendered) > 1 else ""
        return separator.join(rendered)

    message = render(0)
    for level in range(1, 5):
        if len(message) <= MAX_TG_LEN:
            break
        message = render(level)

    if len(message) > MAX_TG_LEN:
        message = message[:MAX_TG_LEN]

    return message


# ============================================================
# Command parsing
# ============================================================
def parse_command(text: str):
    if not text:
        return None, None
    parts = text.strip().split()
    cmd = parts[0].lower()
    arg = parts[1].upper().replace("/", "") if len(parts) > 1 else None
    return cmd, arg


# ============================================================
# Main loop
# ============================================================
def main():
    start_ts = time.time()
    cache = MarketCache()

    # de-dup memory
    last_signal = {s: {"candle_ts": None, "direction": None} for s in SYMBOLS}
    last_hint = {s: {"candle_ts": None, "state": None, "direction": None} for s in SYMBOLS}

    health = {
        "last_scan_utc": None,
        "last_signal_utc": None,
        "last_signal_text": None,
        "scan_count": 0,
        "last_5m_utc": None,
        "last_15m_utc": None,
    }
    last_error_notify = 0.0
    last_heartbeat = 0.0

    if SEND_STARTUP_MESSAGE:
        tg_send_message(
            "✅ BOT ONLINE\n"
            f"UTC: {utc_now_str()}\n"
            "Mode: 15m-close TREND radar (HTF+15m), 5m shown as info only\n"
            f"Loop tick: ~{SLEEP_TARGET_SECONDS}s (Binance fetch on 5m/15m candle close)\n"
            f"Symbols: {', '.join(SYMBOLS)}\n"
            "Commands: /report  | /report BTCUSDT | /status"
        )

    offset = None

    while True:
        loop_start = time.time()
        try:
            # 1) Telegram polling (commands)
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
                            tg_send_message(build_report_message(SYMBOLS, cache))
                        else:
                            sym = arg
                            if sym not in SYMBOLS:
                                tg_send_message("❌ Usage: /report OR /report BTCUSDT\nETHUSDT\nSOLUSDT")
                            else:
                                tg_send_message(build_report_message([sym], cache))

                    elif cmd == "/status":
                        uptime_min = int((time.time() - start_ts) / 60)
                        last_scan = health["last_scan_utc"] or "N/A"
                        last_sig = health["last_signal_utc"] or "N/A"
                        last_sig_text = health["last_signal_text"] or "None yet"
                        next5, eta5 = next_5m_close_eta()
                        next15, eta15 = next_15m_close_eta()
                        tg_send_message(
                            "📡 STATUS\n"
                            f"UTC now: {utc_now_str()}\n"
                            f"Next 5m close: {next5} (ETA {int(eta5)}s)\n"
                            f"Next 15m close: {next15} (ETA {int(eta15)}s)\n"
                            f"Uptime: {uptime_min} min\n"
                            f"Loop tick: ~{SLEEP_TARGET_SECONDS}s (Binance fetch on 5m/15m close)\n"
                            f"Ticks: {health['scan_count']}\n"
                            f"Last scan: {last_scan}\n"
                            f"Last 5m refresh: {health['last_5m_utc'] or 'N/A'}\n"
                            f"Last 15m refresh: {health['last_15m_utc'] or 'N/A'}\n"
                            f"Last signal: {last_sig}\n"
                            f"Last signal detail: {last_sig_text}"
                        )

            # 2) Candle-close driven refresh + scanning
            for symbol in SYMBOLS:
                # Ensure HTF always ready (cached; refreshes only on 1h/4h close)
                cache.ensure_context_ready(symbol)

                # 15m refresh only on 15m close
                prev_15m_open = cache.data[symbol]["15m"].last_closed_open_ms
                cache.refresh_15m_if_needed(symbol)
                if cache.data[symbol]["15m"].last_closed_open_ms != prev_15m_open:
                    health["last_15m_utc"] = utc_now_str()

                # 5m refresh only on 5m close
                prev_5m_open = cache.data[symbol]["5m"].last_closed_open_ms
                cache.refresh_5m_if_needed(symbol)
                if cache.data[symbol]["5m"].last_closed_open_ms != prev_5m_open:
                    health["last_5m_utc"] = utc_now_str()

                df15 = cache.get(symbol, "15m")
                df5 = cache.get(symbol, "5m")
                df1h = cache.get(symbol, "1h")
                df4h = cache.get(symbol, "4h")

                # ---- 5m snapshot is collected, but 5m does NOT drive alerts ----
                if ENABLE_5M_HINT_ALERTS and cache.data[symbol]["5m"].last_closed_open_ms != prev_5m_open:
                    hint = evaluate_5m_hint(symbol, df5, df15, df1h, df4h)
                    if hint is not None:
                        if (last_hint[symbol]["candle_ts"] != hint["candle_ts"]
                            or last_hint[symbol]["state"] != hint["state"]
                            or last_hint[symbol]["direction"] != hint["direction"]):
                            tg_send_message(render_5m_hint(hint))
                            last_hint[symbol]["candle_ts"] = hint["candle_ts"]
                            last_hint[symbol]["state"] = hint["state"]
                            last_hint[symbol]["direction"] = hint["direction"]

                # ---- 15m main trigger (only reacts when refresh happened) ----
                if cache.data[symbol]["15m"].last_closed_open_ms != prev_15m_open:
                    trig = check_entry_trigger(symbol, df15, df1h, df4h)
                    if trig is not None:
                        if not (last_signal[symbol]["candle_ts"] == trig["candle_ts"] and last_signal[symbol]["direction"] == trig["direction"]):
                            i15 = trig["i"]
                            report = build_snapshot_report(
                                symbol=symbol,
                                tag=trig.get("tag","ENTRY_WATCH (15m close trigger)"),
                                direction=trig["direction"],
                                vol_ratio=trig["vol_ratio"],
                                df15=df15,
                                df1h=df1h,
                                df4h=df4h,
                                i15=i15,
                                df5=df5
                            )
                            tg_send_message(report)

                            last_signal[symbol]["candle_ts"] = trig["candle_ts"]
                            last_signal[symbol]["direction"] = trig["direction"]
                            health["last_signal_utc"] = utc_now_str()
                            health["last_signal_text"] = f"{symbol} {trig['direction']} @ candle {utc_ts(trig['candle_ts'])}"

            health["last_scan_utc"] = utc_now_str()
            health["scan_count"] += 1

            # 3) Heartbeat
            if HEARTBEAT_MINUTES and HEARTBEAT_MINUTES > 0:
                now = time.time()
                if now - last_heartbeat >= HEARTBEAT_MINUTES * 60:
                    tg_send_message(
                        f"{HEARTBEAT_TEXT}\n"
                        f"UTC: {utc_now_str()}\n"
                        f"Ticks: {health['scan_count']}\n"
                        f"Last scan: {health['last_scan_utc'] or 'N/A'}\n"
                        f"Last signal: {health['last_signal_text'] or 'None yet'}"
                    )
                    last_heartbeat = now

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            # network wobble: silent retry
            pass
        except Exception:
            now = time.time()
            if (now - last_error_notify) >= ERROR_NOTIFY_COOLDOWN:
                err = traceback.format_exc()
                tg_send_message(
                    "⚠️ BOT ERROR (rate-limited)\n"
                    f"UTC: {utc_now_str()}\n\n"
                    f"{err[:3500]}"
                )
                last_error_notify = now

        # 4) Keep cadence (tick)
        elapsed = time.time() - loop_start
        sleep_for = max(0.1, SLEEP_TARGET_SECONDS - elapsed)
        time.sleep(sleep_for)

if __name__ == "__main__":
    main()
