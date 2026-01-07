
import os
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

# Scan cadence
SLEEP_TARGET_SECONDS = 10

# HTTP timeouts
HTTP_TIMEOUT = 12
TELEGRAM_POLL_TIMEOUT = 2  # long poll

# Candle limits
OHLCV_LIMIT_15M = 220
OHLCV_LIMIT_HTF = 260

# Trigger parameters
VOL_SPIKE_MULT = 1.5
RSI_MIN, RSI_MAX = 25, 75

# MACD slope threshold (ATR scaled): require macdh slope magnitude >= ATR * this factor
MACD_SLOPE_ATR_FACTOR = 0.03  # 0.03 * ATR (tune 0.02–0.06)

# Telegram chunking
TG_CHUNK = 3500

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
SESSION.headers.update({"User-Agent": "momentum-entrywatch/1.0"})

# ============================================================
# Helpers: time / formatting
# ============================================================
def utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def utc_ts(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def interval_seconds(tf: str) -> int:
    return {"15m": 900, "1h": 3600, "4h": 14400}[tf]

def floor_to_interval_ms(epoch_ms: int, tf: str) -> int:
    sec = interval_seconds(tf)
    return (epoch_ms // 1000 // sec) * sec * 1000

def last_closed_open_ms(tf: str) -> int:
    now_ms = int(time.time() * 1000)
    curr_open = floor_to_interval_ms(now_ms, tf)
    return curr_open - interval_seconds(tf) * 1000

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

def next_15m_close_eta() -> tuple[str, float]:
    """Return (next_close_utc_str, seconds_to_next_close)."""
    now_ms = int(time.time() * 1000)
    next_close_ms = floor_to_interval_ms(now_ms, "15m") + interval_seconds("15m") * 1000
    return utc_ts(next_close_ms), max(0.0, (next_close_ms - now_ms) / 1000.0)

# ============================================================
# Binance Futures public REST
# ============================================================
def http_get(path: str, params: dict) -> dict | list:
    url = BINANCE_FAPI + path
    r = SESSION.get(url, params=params, timeout=HTTP_TIMEOUT)
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

def fetch_mark_price(symbol: str) -> float | None:
    j = http_get("/fapi/v1/premiumIndex", {"symbol": symbol})
    return safe_float(j.get("markPrice"))

# ============================================================
# Telegram (polling)
# ============================================================
def tg_api(method: str) -> str:
    return f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/{method}"

def tg_get_updates(offset: int | None) -> dict:
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
    chunks = [text[i:i + TG_CHUNK] for i in range(0, len(text), TG_CHUNK)]
    for ch in chunks:
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": ch}
        r = SESSION.post(tg_api("sendMessage"), json=payload, timeout=HTTP_TIMEOUT)
        r.raise_for_status()

# ============================================================
# Indicators
# ============================================================
def add_indicators_15m(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # VWAP in pandas_ta requires ordered DatetimeIndex
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

    # Useful "delta proxy": taker buy vs sell (base volume)
    out["taker_buy"] = out["taker_buy_base"]
    out["taker_sell"] = out["volume"] - out["taker_buy"]

    out = out.reset_index(drop=True)
    return out

def add_ema_pack(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ema20"] = ta.ema(out["close"], length=20)
    out["ema50"] = ta.ema(out["close"], length=50)
    out["ema200"] = ta.ema(out["close"], length=200)
    return out

# ============================================================
# Cache (HTF refresh only when new candle closes)
# ============================================================
@dataclass
class TFCached:
    df: pd.DataFrame | None = None
    last_closed_open_ms: int | None = None

class MarketCache:
    def __init__(self):
        self.data = {s: {"15m": TFCached(), "1h": TFCached(), "4h": TFCached()} for s in SYMBOLS}

    def refresh_15m(self, symbol: str):
        df = fetch_klines_df(symbol, "15m", OHLCV_LIMIT_15M)
        df = add_indicators_15m(df)
        self.data[symbol]["15m"] = TFCached(df=df, last_closed_open_ms=last_closed_open_ms("15m"))

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
def check_entry_trigger(symbol: str, df15: pd.DataFrame, df1h: pd.DataFrame, df4h: pd.DataFrame) -> dict | None:
    i = len(df15) - 2
    p = i - 1
    if i < 210:
        return None

    needed = ["close", "vwap", "ema20", "ema50", "rsi14", "volume", "vol_sma20", "macdh", "atr14"]
    if not all(c in df15.columns for c in needed):
        return None

    # NaN guards (both current and previous where needed)
    for c in needed:
        if pd.isna(df15.loc[i, c]):
            return None
    for c in ["close", "vwap", "ema20", "ema50", "macdh"]:
        if pd.isna(df15.loc[p, c]):
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

    # 15m values
    c_prev, c_curr = float(df15.loc[p, "close"]), float(df15.loc[i, "close"])
    v_prev, v_curr = float(df15.loc[p, "vwap"]), float(df15.loc[i, "vwap"])
    e20_prev, e20_curr = float(df15.loc[p, "ema20"]), float(df15.loc[i, "ema20"])
    e50_prev, e50_curr = float(df15.loc[p, "ema50"]), float(df15.loc[i, "ema50"])
    vol_curr = float(df15.loc[i, "volume"])
    vol_avg = float(df15.loc[i, "vol_sma20"])
    if vol_avg <= 0:
        return None
    vol_ratio = vol_curr / vol_avg
    if vol_ratio < VOL_SPIKE_MULT:
        return None

    rsi = float(df15.loc[i, "rsi14"])
    if not (RSI_MIN < rsi < RSI_MAX):
        return None

    # Cross logic
    vwap_cross_up = (c_prev <= v_prev) and (c_curr > v_curr)
    vwap_cross_dn = (c_prev >= v_prev) and (c_curr < v_curr)
    ema_cross_up = (e20_prev <= e50_prev) and (e20_curr > e50_curr)
    ema_cross_dn = (e20_prev >= e50_prev) and (e20_curr < e50_curr)

    # MACD hist slope threshold (ATR scaled)
    mh_curr = float(df15.loc[i, "macdh"])
    mh_prev = float(df15.loc[p, "macdh"])
    mh_slope = mh_curr - mh_prev
    atr = float(df15.loc[i, "atr14"])
    slope_thr = max(1e-12, atr * MACD_SLOPE_ATR_FACTOR)
    candle_ts = int(df15.loc[i, "ts"])

    if vwap_cross_up and ema_cross_up and (mh_slope > slope_thr) and htf_long_ok:
        return {"symbol": symbol, "direction": "LONG", "vol_ratio": vol_ratio, "i": i, "candle_ts": candle_ts}
    if vwap_cross_dn and ema_cross_dn and (mh_slope < -slope_thr) and htf_short_ok:
        return {"symbol": symbol, "direction": "SHORT", "vol_ratio": vol_ratio, "i": i, "candle_ts": candle_ts}
    return None

# ============================================================
# Diagnostics for reports (used by both manual and alerts)
# ============================================================
def summarize_trigger_state(df15: pd.DataFrame, df1h: pd.DataFrame, df4h: pd.DataFrame, i: int) -> dict:
    """Return a dict with LONG/SHORT trigger readiness and missing components."""
    p = i - 1

    # HTF context
    i1h, i4h = len(df1h) - 2, len(df4h) - 2
    c1h, e1h = safe_float(df1h.loc[i1h, "close"]), safe_float(df1h.loc[i1h, "ema200"])
    c4h, e4h = safe_float(df4h.loc[i4h, "close"]), safe_float(df4h.loc[i4h, "ema200"])
    htf_long_ok = (c1h is not None and e1h is not None and c4h is not None and e4h is not None
                   and (c1h > e1h) and (c4h > e4h))
    htf_short_ok = (c1h is not None and e1h is not None and c4h is not None and e4h is not None
                    and (c1h < e1h) and (c4h < e4h))

    # 15m values and crosses
    close_prev, close_curr = safe_float(df15.loc[p, "close"]), safe_float(df15.loc[i, "close"])
    vwap_prev, vwap_curr = safe_float(df15.loc[p, "vwap"]), safe_float(df15.loc[i, "vwap"])
    e20_prev, e20_curr = safe_float(df15.loc[p, "ema20"]), safe_float(df15.loc[i, "ema20"])
    e50_prev, e50_curr = safe_float(df15.loc[p, "ema50"]), safe_float(df15.loc[i, "ema50"])

    vwap_cross_up = (close_prev is not None and vwap_prev is not None and close_curr is not None and vwap_curr is not None
                     and (close_prev <= vwap_prev) and (close_curr > vwap_curr))
    vwap_cross_dn = (close_prev is not None and vwap_prev is not None and close_curr is not None and vwap_curr is not None
                     and (close_prev >= vwap_prev) and (close_curr < vwap_curr))
    ema_cross_up = (e20_prev is not None and e50_prev is not None and e20_curr is not None and e50_curr is not None
                    and (e20_prev <= e50_prev) and (e20_curr > e50_curr))
    ema_cross_dn = (e20_prev is not None and e50_prev is not None and e20_curr is not None and e50_curr is not None
                    and (e20_prev >= e50_prev) and (e20_curr < e50_curr))

    # Volume & RSI
    vol = safe_float(df15.loc[i, "volume"])
    vol_avg = safe_float(df15.loc[i, "vol_sma20"], default=0.0)
    vol_ratio = (vol / vol_avg) if (vol_avg and vol_avg > 0) else float("nan")
    rsi = safe_float(df15.loc[i, "rsi14"])
    rsi_ok = (rsi is not None and (RSI_MIN < rsi < RSI_MAX))

    # MACD slope vs ATR threshold
    mh_curr = safe_float(df15.loc[i, "macdh"])
    mh_prev = safe_float(df15.loc[p, "macdh"])
    atr = safe_float(df15.loc[i, "atr14"], default=0.0)
    slope_thr = max(1e-12, (atr or 0.0) * MACD_SLOPE_ATR_FACTOR)
    macd_slope = (mh_curr - mh_prev) if (mh_curr is not None and mh_prev is not None) else float("nan")
    macd_up_ok = (macd_slope == macd_slope) and (macd_slope > +slope_thr)
    macd_dn_ok = (macd_slope == macd_slope) and (macd_slope < -slope_thr)

    # LONG/SHORT readiness and missing
    missing_long = []
    if not htf_long_ok:         missing_long.append("HTF_LONG")
    if not vwap_cross_up:       missing_long.append("VWAP_CROSS_UP")
    if not ema_cross_up:        missing_long.append("EMA20/EMA50_CROSS_UP")
    if not (vol_ratio == vol_ratio and vol_ratio >= VOL_SPIKE_MULT): missing_long.append("VOL_SPIKE")
    if not rsi_ok:              missing_long.append("RSI_RANGE")
    if not macd_up_ok:          missing_long.append("MACD_SLOPE_UP")
    trigger_long = (len(missing_long) == 0)

    missing_short = []
    if not htf_short_ok:        missing_short.append("HTF_SHORT")
    if not vwap_cross_dn:       missing_short.append("VWAP_CROSS_DN")
    if not ema_cross_dn:        missing_short.append("EMA20/EMA50_CROSS_DN")
    if not (vol_ratio == vol_ratio and vol_ratio >= VOL_SPIKE_MULT): missing_short.append("VOL_SPIKE")
    if not rsi_ok:              missing_short.append("RSI_RANGE")
    if not macd_dn_ok:          missing_short.append("MACD_SLOPE_DN")
    trigger_short = (len(missing_short) == 0)

    return {
        "htf_long_ok": htf_long_ok,
        "htf_short_ok": htf_short_ok,
        "vwap_cross_up": vwap_cross_up,
        "vwap_cross_dn": vwap_cross_dn,
        "ema_cross_up": ema_cross_up,
        "ema_cross_dn": ema_cross_dn,
        "vol_ratio": vol_ratio,
        "rsi": rsi,
        "rsi_ok": rsi_ok,
        "macd_slope": macd_slope,
        "slope_thr": slope_thr,
        "macd_up_ok": macd_up_ok,
        "macd_dn_ok": macd_dn_ok,
        "trigger_long": trigger_long,
        "trigger_short": trigger_short,
        "missing_long": missing_long,
        "missing_short": missing_short,
    }

# ============================================================
# Report building
# ============================================================
def _build_snapshot_report_full(symbol: str, tag: str, direction: str, vol_ratio: float,
                          df15: pd.DataFrame, df1h: pd.DataFrame, df4h: pd.DataFrame, i15: int) -> str:
    # Price: prefer mark
    mark = fetch_mark_price(symbol)
    px = mark if mark is not None else float(df15.loc[i15, "close"])

    # HTF last closed
    i1h = len(df1h) - 2
    i4h = len(df4h) - 2
    c1h, e1h = float(df1h.loc[i1h, "close"]), float(df1h.loc[i1h, "ema200"])
    c4h, e4h = float(df4h.loc[i4h, "close"]), float(df4h.loc[i4h, "ema200"])
    ctx_1h = "ABOVE" if c1h > e1h else "BELOW"
    ctx_4h = "ABOVE" if c4h > e4h else "BELOW"

    # 15m metrics (last closed candle)
    vwap = float(df15.loc[i15, "vwap"])
    ema20 = float(df15.loc[i15, "ema20"])
    ema50 = float(df15.loc[i15, "ema50"])
    ema200 = float(df15.loc[i15, "ema200"])
    rsi = float(df15.loc[i15, "rsi14"])
    macdh = float(df15.loc[i15, "macdh"])
    atr = float(df15.loc[i15, "atr14"])
    close_px = float(df15.loc[i15, "close"])

    # MACD hist slope
    prev = i15 - 1
    mh_prev = safe_float(df15.loc[prev, "macdh"], default=float("nan")) if prev >= 0 else float("nan")
    mh_slope = macdh - mh_prev if mh_prev == mh_prev else float("nan")
    slope_dir = "INCREASING" if mh_slope == mh_slope and mh_slope > 0 else "DECREASING" if mh_slope == mh_slope and mh_slope < 0 else "FLAT/NA"

    # Delta proxy (taker buy/sell)
    tb = safe_float(df15.loc[i15, "taker_buy"], default=float("nan"))
    ts = safe_float(df15.loc[i15, "taker_sell"], default=float("nan"))
    delta = tb - ts if (tb == tb and ts == ts) else float("nan")

    # Distances (in %)
    dist_close_vwap = pct(close_px, vwap)
    dist_close_e50 = pct(close_px, ema50)
    dist_close_e200 = pct(close_px, ema200)
    atr_pct = pct(atr, close_px)

    # Timing
    last_15m_close_utc = utc_ts(int(df15.loc[i15, "ts"]) + interval_seconds("15m") * 1000)
    next_close_utc, eta_sec = next_15m_close_eta()

    # Diagnostics (LONG/SHORT readiness)
    diag = summarize_trigger_state(df15, df1h, df4h, i15)
    return "
".join(msg)

# ============================================================
# Snapshot report wrapper (supports both old and new call styles)
# ============================================================
def build_snapshot_report(*args, **kwargs) -> str:
    """
    Backwards/forwards compatible wrapper.

    Supported call styles:
    1) build_snapshot_report(symbol, cache)  -> returns MANUAL_REPORT snapshot (compact/full per _build_snapshot_report_full)
    2) build_snapshot_report(symbol, tag, direction, vol_ratio, df15, df1h, df4h, i15) -> full builder
    """
    # Style (1): (symbol, cache)
    if len(args) == 2:
        symbol, cache = args
        return build_manual_report(symbol, cache)

    # Style (2): legacy/full signature
    if len(args) >= 8:
        symbol, tag, direction, vol_ratio, df15, df1h, df4h, i15 = args[:8]
        return _build_snapshot_report_full(symbol, tag, direction, vol_ratio, df15, df1h, df4h, i15)

    raise TypeError("build_snapshot_report() expected (symbol, cache) or full 8-arg signature")

def build_manual_report(symbol: str, cache: "MarketCache") -> str:
    cache.refresh_15m(symbol)
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

    return _build_snapshot_report_full(symbol, "MANUAL_REPORT", direction, vol_ratio, df15, df1h, df4h, i15)

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
    last_signal = {s: {"candle_ts": None, "direction": None} for s in SYMBOLS}
    health = {
        "last_scan_utc": None,
        "last_signal_utc": None,
        "last_signal_text": None,
        "scan_count": 0,
    }
    last_error_notify = 0.0
    last_heartbeat = 0.0

    if SEND_STARTUP_MESSAGE:
        tg_send_message(
            "✅ BOT ONLINE\n"
            f"UTC: {utc_now_str()}\n"
            f"Mode: 15m close-based ENTRY_WATCH\n"
            f"Scan: ~{SLEEP_TARGET_SECONDS}s\n"
            f"Symbols: {', '.join(SYMBOLS)}\n"
            "Commands: /report \n/report BTCUSDT \n/status"
        )

    offset = None

    while True:
        loop_start = time.time()
        try:
            # 1) Telegram polling (commands)
            upd = tg_get_updates(offset)
            # ✅ If conflict, don't error-spam; just backoff and keep scanning
            if isinstance(upd, dict) and upd.get("conflict"):
                time.sleep(5)
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
                            reports = []
                            for s in SYMBOLS:
                                reports.append(build_manual_report(s, cache))
                            combined = "\n\n" + ("=" * 32) + "\n\n"
                            tg_send_message(combined.join(reports))
                        else:
                            sym = arg
                            if sym not in SYMBOLS:
                                tg_send_message("❌ Usage: /report OR /report BTCUSDT\nETHUSDT\nSOLUSDT")
                            else:
                                tg_send_message(build_manual_report(sym, cache))

                    elif cmd == "/status":
                        uptime_min = int((time.time() - start_ts) / 60)
                        last_scan = health["last_scan_utc"] or "N/A"
                        last_sig = health["last_signal_utc"] or "N/A"
                        last_sig_text = health["last_signal_text"] or "None yet"
                        tg_send_message(
                            "📡 STATUS\n"
                            f"UTC now: {utc_now_str()}\n"
                            f"Uptime: {uptime_min} min\n"
                            f"Scan interval: ~{SLEEP_TARGET_SECONDS}s\n"
                            f"Scans: {health['scan_count']}\n"
                            f"Last scan: {last_scan}\n"
                            f"Last signal: {last_sig}\n"
                            f"Last signal detail: {last_sig_text}"
                        )

            # 2) Scan for entry triggers
            for symbol in SYMBOLS:
                cache.refresh_15m(symbol)
                cache.ensure_context_ready(symbol)

                df15 = cache.get(symbol, "15m")
                df1h = cache.get(symbol, "1h")
                df4h = cache.get(symbol, "4h")

                trig = check_entry_trigger(symbol, df15, df1h, df4h)
                if trig is None:
                    continue

                # de-dup per symbol per closed candle + direction
                if last_signal[symbol]["candle_ts"] == trig["candle_ts"] and last_signal[symbol]["direction"] == trig["direction"]:
                    continue

                i15 = trig["i"]
                report = build_snapshot_report(
                    symbol=symbol,
                    tag="ENTRY_WATCH (15m close trigger)",
                    direction=trig["direction"],
                    vol_ratio=trig["vol_ratio"],
                    df15=df15,
                    df1h=df1h,
                    df4h=df4h,
                    i15=i15
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
                        f"Scans: {health['scan_count']}\n"
                        f"Last scan: {health['last_scan_utc'] or 'N/A'}\n"
                        f"Last signal: {health['last_signal_text'] or 'None yet'}"
                    )
                    last_heartbeat = now

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            # network wobble: silent retry
            pass
        except Exception:
            # Report errors to telegram (rate-limited)
            now = time.time()
            if (now - last_error_notify) >= ERROR_NOTIFY_COOLDOWN:
                err = traceback.format_exc()
                tg_send_message(
                    "⚠️ BOT ERROR (rate-limited)\n"
                    f"UTC: {utc_now_str()}\n\n"
                    f"{err[:3500]}"
                )
                last_error_notify = now

        # 4) Keep ~fixed cadence
        elapsed = time.time() - loop_start
        time.sleep(max(1, int(SLEEP_TARGET_SECONDS - elapsed)))

if __name__ == "__main__":
    main()
