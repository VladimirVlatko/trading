"""
MOMENTUM PRO – FINAL (Telegram Enabled, SAFE)
- Sends Telegram message ONLY if ALERT LEVEL >= 2
- NO trading signals
- Human + ChatGPT decision layer
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
RETURN_LOOKBACK = 6

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# ============================================ #

def send_telegram(message: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    requests.post(url, json=payload, timeout=10)

# ---------- INDICATORS ---------- #

def ema(series, length):
    alpha = 2 / (length + 1)
    out = [series[0]]
    for p in series[1:]:
        out.append(alpha * p + (1 - alpha) * out[-1])
    return np.array(out)

def rsi(series, length=14):
    delta = np.diff(series)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.convolve(gain, np.ones(length)/length, mode='valid')
    avg_loss = np.convolve(loss, np.ones(length)/length, mode='valid')
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return np.concatenate([[50]*(len(series)-len(rsi)), rsi])

def atr(high, low, close, length=14):
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(abs(high[1:] - close[:-1]), abs(low[1:] - close[:-1]))
    )
    atr = np.convolve(tr, np.ones(length)/length, mode='valid')
    return np.concatenate([[np.mean(tr[:length])]*(len(close)-len(atr)), atr])

# ---------- DATA ---------- #

def fetch_klines(symbol, interval, limit=150):
    url = f"{BINANCE_FUTURES}/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    return np.array(requests.get(url, params=params, timeout=10).json(), dtype=float)

def fetch_derivatives(symbol):
    mark = requests.get(
        f"{BINANCE_FUTURES}/fapi/v1/premiumIndex",
        params={"symbol": symbol}, timeout=10
    ).json()
    oi = requests.get(
        f"{BINANCE_FUTURES}/fapi/v1/openInterest",
        params={"symbol": symbol}, timeout=10
    ).json()
    return {
        "mark": float(mark["markPrice"]),
        "index": float(mark["indexPrice"]),
        "funding": float(mark["lastFundingRate"]) * 100,
        "basis": (float(mark["markPrice"]) - float(mark["indexPrice"])) /
                 float(mark["indexPrice"]) * 100,
        "oi": float(oi["openInterest"])
    }

# ---------- ANALYSIS ---------- #

def analyze_symbol(symbol):
    tf_data = {}
    scores = []
    reasons = []
    kl_15m = None

    for tf, interval in TIMEFRAMES.items():
        k = fetch_klines(symbol, interval)
        if tf == "15m":
            kl_15m = k

        close, high, low, vol = k[:,4], k[:,2], k[:,3], k[:,5]

        ema20 = ema(close, EMA_FAST)
        ema50 = ema(close, EMA_SLOW)
        rsi_v = rsi(close)
        atr_v = atr(high, low, close)

        rsi_slope = rsi_v[-1] - rsi_v[-6]
        vol_ratio = vol[-1] / (np.mean(vol[-VOL_LOOKBACK:]) + 1e-9)

        score = 0
        if close[-1] < ema20[-1] and close[-1] < ema50[-1]:
            score -= 2
        if ema20[-1] < ema50[-1]:
            score -= 1
        if rsi_v[-1] < 45:
            score -= 1
        if rsi_slope < -4:
            score -= 1

        scores.append(score)

        if tf == "15m" and vol_ratio > 1.2 and rsi_slope < -5:
            reasons.append(f"Volume + aggression (VOL {vol_ratio:.2f}x)")

        tf_data[tf] = {
            "price": close[-1],
            "ema20": ema20[-1],
            "ema50": ema50[-1],
            "ema20_slp": ema20[-1] - ema20[-6],
            "ema50_slp": ema50[-1] - ema50[-6],
            "rsi": rsi_v[-1],
            "rsi_slp": rsi_slope,
            "atr": atr_v[-1],
            "vol_ratio": vol_ratio
        }

    ret_15m = ((kl_15m[-1,4] - kl_15m[-RETURN_LOOKBACK,4]) /
               kl_15m[-RETURN_LOOKBACK,4]) * 100

    avg_score = np.mean(scores)
    alert = 0
    if avg_score <= -2:
        alert = 2
    if avg_score <= -3.5 and reasons:
        alert = 3

    return {
        "symbol": symbol,
        "alert": alert,
        "reasons": reasons,
        "tf": tf_data,
        "ret_15m": ret_15m
    }

# ---------- MAIN LOOP ---------- #

def run():
    reports = [analyze_symbol(s) for s in SYMBOLS]
    active = [r for r in reports if r["alert"] >= 2]
    if not active:
        return

    msg = f"🔔 *MOMENTUM PRO ALERT*\nUTC: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    for r in active:
        msg += f"*{r['symbol']}* – ALERT {r['alert']}\n"
        for reason in r["reasons"]:
            msg += f"• {reason}\n"
        msg += f"15m return: {r['ret_15m']:+.2f}%\n\n"

    send_telegram(msg)

# ---------- RUNNER ---------- #

if __name__ == "__main__":
    # Startup ping (се праќа само еднаш кога ќе се крене bot-от)
    send_telegram("✅ Momentum bot is ONLINE on Railway.\nVariables OK. Waiting for signals.")

    while True:
        try:
            run()
        except Exception as e:
            send_telegram(f"❌ Runtime error:\n{e}")
        time.sleep(300)  # 5 minutes


