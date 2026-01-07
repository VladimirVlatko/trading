import os
import time
import math
from dataclasses import dataclass
from datetime import datetime, timezone

import requests
import pandas as pd
import pandas_ta as ta

# =========================
# ENV (required)
# =========================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
    raise RuntimeError("Missing TELEGRAM_TOKEN or TELEGRAM_CHAT_ID in env vars.")

# =========================
# CONFIG
# =========================
BINANCE_FAPI = "https://fapi.binance.com"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

TF_15M = "15m"
TF_1H = "1h"
TF_4H = "4h"

SLEEP_TARGET_SECONDS = 60   # scan cadence
HTTP_TIMEOUT = 12
TELEGRAM_TIMEOUT = 55       # long poll seconds

# Trigger params
OHLCV_LIMIT_15M = 220
OHLCV_LIMIT_HTF = 260
VOL_SPIKE_MULT = 1.5
RSI_MIN, RSI_MAX = 25, 75

# Anti-spam controls (IMPORTANT now that it's real-time)
EDGE_TRIGGER_ONLY = True      # send only on FALSE->TRUE transition
COOLDOWN_SECONDS = 10 * 60    # if conditions stay TRUE, allow resend after 10 min (set 0 to disable)

# Telegram max 4096 chars
TG_CHUNK = 3500

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "momentum-scanner/telegram/rt/1.0"})


# =========================
# Time helpers
# =========================
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
        return float(x)
    except Exception:
        return default


# =========================
# Binance public REST
# =========================
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
    df = df[["ts", "open", "high", "low", "close", "volume"]].copy()
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce").astype("int64")
    return df


def fetch_mark_price(symbol: str) -> float | None:
    j = http_get("/fapi/v1/premiumIndex", {"symbol": symbol})
    return safe_float(j.get("markPrice"))


# =========================
# Telegram (polling)
# =========================
def tg_api(method: str) -> str:
    return f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/{method}"


def tg_get_updates(offset: int | None) -> dict:
    params = {"timeout": TELEGRAM_TIMEOUT}
    if offset is not None:
        params["offset"] = offset
    r = SESSION.get(tg_api("getUpdates"), params=params, timeout=TELEGRAM_TIMEOUT + 10)
    r.raise_for_status()
    return r.json()


def tg_send_message(text: str):
    chunks = [text[i:i + TG_CHUNK] for i in range(0, len(text), TG_CHUNK)]
    for ch in chunks:
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": ch}
        r = SESSION.post(tg_api("sendMessage"), json=payload, timeout=HTTP_TIMEOUT)
        r.raise_for_status()


# =========================
# Indicators
# =========================
def add_indicators_15m(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # ✅ VWAP in pandas_ta needs an ORDERED DatetimeIndex
    out["dt"] = pd.to_datetime(out["ts"], unit="ms", utc=True)
    out = out.sort_values("dt").set_index("dt")

    out["ema20"] = ta.ema(out["close"], length=20)
    out["ema50"] = ta.ema(out["close"], length=50)
    out["ema200"] = ta.ema(out["close"], length=200)

    # VWAP (now valid)
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

    # return to normal index
    out = out.reset_index(drop=True)
    return out



def add_ema_pack(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ema20"] = ta.ema(out["close"], length=20)
    out["ema50"] = ta.ema(out["close"], length=50)
    out["ema200"] = ta.ema(out["close"], length=200)
    return out


def macdh_slope(df15: pd.DataFrame, i: int) -> float:
    a = safe_float(df15.loc[i, "macdh"])
    b = safe_float(df15.loc[i - 1, "macdh"])
    if a is None or b is None or (math.isnan(a) or math.isnan(b)):
        return float("nan")
    return a - b


# =========================
# Cache to reduce HTF requests
# =========================
@dataclass
class TFCached:
    df: pd.DataFrame | None = None
    last_closed_open_ms: int | None = None


class MarketCache:
    def __init__(self):
        self.data = {s: {TF_15M: TFCached(), TF_1H: TFCached(), TF_4H: TFCached()} for s in SYMBOLS}

    def refresh_15m(self, symbol: str):
        df = fetch_klines_df(symbol, TF_15M, OHLCV_LIMIT_15M)
        df = add_indicators_15m(df)
        self.data[symbol][TF_15M] = TFCached(df=df, last_closed_open_ms=last_closed_open_ms(TF_15M))

    def refresh_htf_if_needed(self, symbol: str, tf: str):
        expected = last_closed_open_ms(tf)
        cached = self.data[symbol][tf]
        if cached.df is None or cached.last_closed_open_ms != expected:
            df = fetch_klines_df(symbol, tf, OHLCV_LIMIT_HTF)
            df = add_ema_pack(df)
            self.data[symbol][tf] = TFCached(df=df, last_closed_open_ms=expected)

    def ensure_context_ready(self, symbol: str):
        self.refresh_htf_if_needed(symbol, TF_1H)
        self.refresh_htf_if_needed(symbol, TF_4H)

    def get(self, symbol: str, tf: str) -> pd.DataFrame:
        df = self.data[symbol][tf].df
        if df is None:
            raise RuntimeError(f"Cache missing for {symbol} {tf}")
        return df


# =========================
# REAL-TIME condition check (NOT candle-close)
# =========================
def conditions_met_realtime(symbol: str, df15: pd.DataFrame) -> dict | None:
    """
    Uses the MOST RECENT candle row (can be forming candle).
    Condition = "momentum is ON right now":
      - Price vs VWAP: price above vwap (LONG) or below vwap (SHORT)
      - EMA20 vs EMA50: ema20 > ema50 (LONG) or ema20 < ema50 (SHORT)
      - Volume spike: current candle volume >= 1.5x SMA20(volume)
      - RSI filter: 25 < RSI < 75
    """
    i = len(df15) - 1  # latest row (may be forming)
    if i < 205:
        return None

    # Guard for NaNs
    needed = ["close", "vwap", "ema20", "ema50", "rsi14", "volume", "vol_sma20", "atr14", "macdh"]
    if any(pd.isna(df15.loc[i, c]) for c in needed):
        return None

    close = float(df15.loc[i, "close"])
    vwap = float(df15.loc[i, "vwap"])
    ema20 = float(df15.loc[i, "ema20"])
    ema50 = float(df15.loc[i, "ema50"])
    rsi = float(df15.loc[i, "rsi14"])
    vol = float(df15.loc[i, "volume"])
    vol_avg = float(df15.loc[i, "vol_sma20"])

    if vol_avg <= 0:
        return None

    vol_ratio = vol / vol_avg
    if vol_ratio < VOL_SPIKE_MULT:
        return None

    if not (RSI_MIN < rsi < RSI_MAX):
        return None

    # Determine direction by state (not cross)
    long_ok = (close > vwap) and (ema20 > ema50)
    short_ok = (close < vwap) and (ema20 < ema50)

    if not (long_ok or short_ok):
        return None

    direction = "LONG" if long_ok else "SHORT"
    candle_ts = int(df15.loc[i, "ts"])  # open time of the current 15m candle
    return {"symbol": symbol, "direction": direction, "vol_ratio": float(vol_ratio), "i": i, "candle_ts": candle_ts}


def build_snapshot_report(symbol: str, direction: str, vol_ratio: float,
                          df15: pd.DataFrame, df1h: pd.DataFrame, df4h: pd.DataFrame, i15: int) -> str:
    # HTF indexes: use last closed (more stable)
    i1h = len(df1h) - 2
    i4h = len(df4h) - 2

    # Use markPrice for "current price" in report
    mark = fetch_mark_price(symbol)
    px = mark if mark is not None else float(df15.loc[i15, "close"])

    # HTF context vs EMA200
    c1h = float(df1h.loc[i1h, "close"])
    e200_1h = float(df1h.loc[i1h, "ema200"])
    c4h = float(df4h.loc[i4h, "close"])
    e200_4h = float(df4h.loc[i4h, "ema200"])

    ctx_1h = "ABOVE" if c1h > e200_1h else "BELOW"
    ctx_4h = "ABOVE" if c4h > e200_4h else "BELOW"

    # 15m metrics (realtime candle possible)
    rsi = float(df15.loc[i15, "rsi14"])
    macdh = safe_float(df15.loc[i15, "macdh"], default=float("nan"))
    slope = macdh_slope(df15, i15) if i15 >= 1 else float("nan")
    slope_dir = "INCREASING" if slope > 0 else "DECREASING" if slope < 0 else "FLAT/NA"
    atr = float(df15.loc[i15, "atr14"])

    vwap = float(df15.loc[i15, "vwap"])
    ema20 = float(df15.loc[i15, "ema20"])
    ema50 = float(df15.loc[i15, "ema50"])
    ema200 = float(df15.loc[i15, "ema200"])

    # Raw OHLC: last 3 candles including current (may be forming)
    start = max(0, i15 - 2)
    rows = []
    for j in range(start, i15 + 1):
        rows.append(
            f"- {utc_ts(int(df15.loc[j,'ts']))} | "
            f"O {float(df15.loc[j,'open']):.6f} "
            f"H {float(df15.loc[j,'high']):.6f} "
            f"L {float(df15.loc[j,'low']):.6f} "
            f"C {float(df15.loc[j,'close']):.6f} "
            f"| V {float(df15.loc[j,'volume']):.2f}"
        )

    msg = []
    msg.append("🧠 SNAPSHOT REPORT (Realtime condition met — Scanner only, NOT a trade signal)")
    msg.append(f"UTC: {utc_now_str()}")
    msg.append("")
    msg.append(f"{symbol} | MOMENTUM_OK | DIR(15m): {direction}")
    msg.append(f"Price (mark): {px:.6f}")
    msg.append("")
    msg.append("Higher Timeframe Context (vs EMA200):")
    msg.append(f"- 1h: close {c1h:.6f} is {ctx_1h} EMA200 {e200_1h:.6f}")
    msg.append(f"- 4h: close {c4h:.6f} is {ctx_4h} EMA200 {e200_4h:.6f}")
    msg.append("")
    msg.append("15m Trend + Momentum (latest candle may be forming):")
    msg.append(f"- VWAP {vwap:.6f} | EMA20 {ema20:.6f} | EMA50 {ema50:.6f} | EMA200 {ema200:.6f}")
    msg.append(f"- RSI(14): {rsi:.2f} (filter {RSI_MIN}–{RSI_MAX})")
    msg.append(f"- MACD Hist: {macdh:.6f} | Hist slope: {slope:.6f} ({slope_dir})")
    msg.append("")
    msg.append("Volatility (15m):")
    msg.append(f"- ATR(14): {atr:.6f}")
    msg.append("")
    msg.append("Volume Strength (15m):")
    msg.append(f"- Vol ratio (current/SMA20): {vol_ratio:.2f}x (need >= {VOL_SPIKE_MULT}x)")
    msg.append("")
    msg.append("Raw Data Snippet — last 3 candles (OHLCV):")
    msg.extend(rows)

    return "\n".join(msg)


def build_manual_report(symbol: str, cache: MarketCache) -> str:
    cache.refresh_15m(symbol)
    cache.ensure_context_ready(symbol)

    df15 = cache.get(symbol, TF_15M)
    df1h = cache.get(symbol, TF_1H)
    df4h = cache.get(symbol, TF_4H)

    i15 = len(df15) - 1  # realtime row
    vol = float(df15.loc[i15, "volume"]) if not pd.isna(df15.loc[i15, "volume"]) else float("nan")
    vol_avg = float(df15.loc[i15, "vol_sma20"]) if not pd.isna(df15.loc[i15, "vol_sma20"]) else float("nan")
    vol_ratio = (vol / vol_avg) if vol_avg and vol_avg > 0 else float("nan")

    # Direction here is just state snapshot
    direction = "N/A"
    if not pd.isna(df15.loc[i15, "vwap"]) and not pd.isna(df15.loc[i15, "ema20"]) and not pd.isna(df15.loc[i15, "ema50"]):
        close = float(df15.loc[i15, "close"])
        vwap = float(df15.loc[i15, "vwap"])
        ema20 = float(df15.loc[i15, "ema20"])
        ema50 = float(df15.loc[i15, "ema50"])
        if close > vwap and ema20 > ema50:
            direction = "LONG-ish"
        elif close < vwap and ema20 < ema50:
            direction = "SHORT-ish"

    return build_snapshot_report(symbol, direction, vol_ratio, df15, df1h, df4h, i15)


# =========================
# Telegram command parsing
# =========================
def parse_report_command(text: str) -> list[str] | None:
    if not text:
        return None
    parts = text.strip().split()
    if not parts:
        return None
    if parts[0].lower() != "/report":
        return None
    if len(parts) == 1:
        return SYMBOLS[:]
    sym = parts[1].upper().replace("/", "")
    if sym in SYMBOLS:
        return [sym]
    return ["__INVALID__"]


# =========================
# Main
# =========================
def main():
    tg_send_message(
    "🤖 Momentum Scanner ONLINE\n"
    f"UTC: {utc_now_str()}\n"
    "Mode: Realtime momentum (forming candle)\n"
    "Scan: ~60s\n"
)
    cache = MarketCache()

    # State per symbol for anti-spam
    state = {
        s: {
            "was_true": False,
            "last_sent_ts": 0.0,
            "last_sent_dir": None,
        } for s in SYMBOLS
    }

    offset = None

    while True:
        loop_start = time.time()

        try:
            # 1) Telegram updates
            upd = tg_get_updates(offset)
            if upd.get("ok") and upd.get("result"):
                for u in upd["result"]:
                    offset = u["update_id"] + 1
                    msg = u.get("message") or {}
                    chat = msg.get("chat") or {}
                    chat_id = str(chat.get("id", ""))

                    if chat_id != str(TELEGRAM_CHAT_ID):
                        continue

                    text = (msg.get("text") or "").strip()
                    targets = parse_report_command(text)
                    if targets is not None:
                        if targets == ["__INVALID__"]:
                            tg_send_message("❌ Usage: /report  OR  /report BTCUSDT|ETHUSDT|SOLUSDT")
                        else:
                            for s in targets:
                                tg_send_message(build_manual_report(s, cache))

            # 2) Scan realtime conditions (every loop)
            for symbol in SYMBOLS:
                cache.refresh_15m(symbol)
                cache.ensure_context_ready(symbol)  # 1h/4h only refresh when new candle

                df15 = cache.get(symbol, TF_15M)
                df1h = cache.get(symbol, TF_1H)
                df4h = cache.get(symbol, TF_4H)

                cond = conditions_met_realtime(symbol, df15)
                now = time.time()

                if cond is None:
                    state[symbol]["was_true"] = False
                    continue

                direction = cond["direction"]
                i15 = cond["i"]

                # Edge-trigger: only send when it flips FALSE -> TRUE
                should_send = True
                if EDGE_TRIGGER_ONLY and state[symbol]["was_true"]:
                    should_send = False

                # Cooldown override: if conditions stay TRUE, allow resend after cooldown
                if not should_send and COOLDOWN_SECONDS > 0:
                    if (now - state[symbol]["last_sent_ts"]) >= COOLDOWN_SECONDS:
                        should_send = True

                if should_send:
                    rep = build_snapshot_report(
                        symbol=symbol,
                        direction=direction,
                        vol_ratio=cond["vol_ratio"],
                        df15=df15,
                        df1h=df1h,
                        df4h=df4h,
                        i15=i15,
                    )
                    tg_send_message(rep)
                    state[symbol]["last_sent_ts"] = now
                    state[symbol]["last_sent_dir"] = direction

                state[symbol]["was_true"] = True

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            time.sleep(5)
        except Exception:
            # keep it alive; avoid spamming telegram with errors
            time.sleep(5)

        # 3) Maintain ~60s cadence
        elapsed = time.time() - loop_start
        time.sleep(max(1, int(SLEEP_TARGET_SECONDS - elapsed)))


if __name__ == "__main__":
    main()
