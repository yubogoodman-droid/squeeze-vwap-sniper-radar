#!/usr/bin/env python3
"""幣安永續合約「Squeeze × VWAP 護盤 × 主動買盤」雙模狙擊起漲雷達 v2 (100% 獨立腳本)

針對 180 天全市場實盤數據提純優化，0% 依賴既有均線腳本：
- 【子雷達 M1：VWAP 護盤蓄勢突破 (VWAP_COIL_BREAK)】
  起漲前連 8 根 K 線 (2小時) 100% 收盤站穩 12h VWAP 之上（主力持續護盤承接）
  + 24h 箱體振幅 <= 9.0%、12h 振幅 <= 7.75%、6h 布林帶寬 1.8% ~ 6.9%
  + 中期趨勢翻揚 (SMA60 斜率 >= +0.6%、SMA200 走平翻揚 >= -0.1%，且未偏離 5日線 SMA480)
  + 突破 24h Donchian 高點 + 成交額中位數 >= 6.0x + 量能加速 >= 1.7x + 主動買盤 >= 52% + CLV >= 0.72
- 【子雷達 M2：極致窒息巨量核爆 (ULTRA_SQUEEZE_CLIMAX)】
  6h 布林帶寬 <= 2.2% 且 6h 振幅 <= 3.0% 極度窒息後，單根成交額爆發 >= 30x + CLV >= 0.90 + 主動買盤 >= 54%
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import requests

logger = logging.getLogger("squeeze_breakout_radar")
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.WARNING)

TPE = ZoneInfo("Asia/Taipei")
BINANCE_FAPI = "https://fapi.binance.com"
HTTP = requests.Session()
HTTP.headers.update({"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
_RATE_LOCK = threading.Lock()
_COOLDOWN_UNTIL = 0.0

INTERVAL_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
}


def interval_to_ms(interval: str) -> int:
    iv = interval.strip().lower()
    if iv in INTERVAL_MS:
        return INTERVAL_MS[iv]
    if iv.endswith("m") and iv[:-1].isdigit():
        return int(iv[:-1]) * 60_000
    if iv.endswith("h") and iv[:-1].isdigit():
        return int(iv[:-1]) * 3_600_000
    if iv.endswith("d") and iv[:-1].isdigit():
        return int(iv[:-1]) * 86_400_000
    return 900_000


def timeframe_windows(interval: str = "15m") -> Dict[str, int]:
    """Scale physical lookback durations from bar_ms instead of hardcoding 15m bar counts."""
    bar_ms = interval_to_ms(interval)
    scale = 900_000.0 / float(max(bar_ms, 60_000))
    bars_24h = max(6, int(round(96.0 * scale)))
    bars_12h = max(4, int(round(48.0 * scale)))
    bars_6h = max(3, int(round(24.0 * scale)))
    bars_bbw_min = max(2, int(round(10.0 * scale)))
    pre_buf = max(1, int(round(2.0 * scale)))
    bars_vwap_hold = max(2, int(round(8.0 * scale)))
    bars_wick = max(2, int(round(6.0 * scale)))
    sma_fast = max(10, int(round(60.0 * scale)))
    sma_mid = max(20, int(round(200.0 * scale)))
    sma_5d = max(48, int(round(480.0 * scale)))
    sma_5d_min_p = max(24, sma_5d // 2)
    slope_bars = max(3, int(round(20.0 * scale)))
    qv_med_bars = max(8, int(round(20.0 * scale)))
    cooldown_bars = max(2, int(round(12.0 * scale)))
    max_hold_bars = max(16, int(round(160.0 * scale)))

    warmup_bars = max(
        sma_mid + slope_bars,
        sma_5d_min_p + 2,
        bars_24h + pre_buf + 2,
        250 if bar_ms == 900_000 else 40,
    )
    min_df_len = warmup_bars + 10
    return {
        "bar_ms": bar_ms,
        "bars_24h": bars_24h,
        "bars_12h": bars_12h,
        "bars_6h": bars_6h,
        "bars_bbw_min": bars_bbw_min,
        "pre_buf": pre_buf,
        "bars_vwap_hold": bars_vwap_hold,
        "bars_wick": bars_wick,
        "sma_fast": sma_fast,
        "sma_mid": sma_mid,
        "sma_5d": sma_5d,
        "sma_5d_min_p": sma_5d_min_p,
        "slope_bars": slope_bars,
        "qv_med_bars": qv_med_bars,
        "cooldown_bars": cooldown_bars,
        "max_hold_bars": max_hold_bars,
        "warmup_bars": warmup_bars,
        "min_df_len": min_df_len,
    }

BLACKLIST = frozenset({
    "USDCUSDT", "FDUSDUSDT", "USDEUSDT", "TUSDUSDT",
    "SOXSUSDT", "SQQQUSDT", "TQQQUSDT", "SOXLUSDT", "SPCXUSDT",
    "QQQUSDT", "SPYUSDT", "NVDAUSDT", "TSLAUSDT", "AAPLUSDT",
    "AMDUSDT", "MSTRUSDT", "COINUSDT", "GOOGLUSDT", "METAUSDT",
    "AMZNUSDT", "MSFTUSDT", "PLTRUSDT", "HOODUSDT", "SMCIUSDT",
    "AVGOUSDT", "MUUSDT", "XAUUSDT", "XAGUSDT", "CLUSDT", "BZUSDT",
})


@dataclass
class RadarHit:
    symbol: str
    interval: str
    mode: str                 # "VWAP_COIL_BREAK" | "ULTRA_SQUEEZE_CLIMAX"
    bar_idx: int
    timestamp: datetime
    price: float
    stop_price: float
    breakout_level: float
    vwap: float
    quote_vol_usd: float
    qv_mult: float
    vol_accel: float
    taker_buy_pct: float
    bar_gain_pct: float
    clv: float
    bbw_min10_pct: float
    bbw_max24_pct: float
    rng24_pct: float
    rng96_pct: float
    score: int
    max_gain_after_pct: float
    note: str
    oi_change_pct: Optional[float] = None
    funding_rate_pct: Optional[float] = None


@dataclass(frozen=True)
class RadarConfig:
    breakout_buffer_pct: float = 0.40
    qv_mult_min: float = 6.0
    min_bar_qv_usd: float = 250_000.0
    tbr_min: float = 0.52
    clv_min: float = 0.72
    bar_gain_min_pct: float = 2.2
    bar_gain_max_pct: float = 18.0
    ext_vwap_max_pct: float = 14.0
    base_bbw_max_pct: float = 7.5
    base_rng6h_max_pct: float = 7.5
    m1_bbw_min_low: float = 0.90
    m1_bbw_min_high: float = 4.55
    m1_bbw_max_pct: float = 6.90
    m1_rng24h_max_pct: float = 11.5
    m1_rng12h_max_pct: float = 8.20
    m1_dist_s480_min_pct: float = -2.0
    m1_dist_s480_max_pct: float = 12.0
    m1_slp_s60_min_pct: float = 0.52
    m1_slp_s200_min_pct: float = -0.10
    m1_vol_accel_min: float = 1.70
    m2_bbw_max_pct: float = 2.20
    m2_rng6h_max_pct: float = 3.00
    m2_qv_mult_min: float = 30.0
    m2_clv_min: float = 0.90
    m2_tbr_min: float = 0.54
    min_risk_pct: float = 0.80
    max_risk_pct: float = 14.0


def add_radar_threshold_args(parser: argparse.ArgumentParser) -> None:
    grp = parser.add_argument_group("雙模雷達偵測門檻參數 (Key Thresholds)")
    d = RadarConfig()
    grp.add_argument(
        "--breakout-buffer-pct",
        type=float,
        default=d.breakout_buffer_pct,
        help=f"收盤突破 24h Donchian 高點最低緩衝百分比 (預設 {d.breakout_buffer_pct})",
    )
    grp.add_argument(
        "--qv-mult-min",
        type=float,
        default=d.qv_mult_min,
        help=f"突破 K 線成交額相對中位數最低倍數 (預設 {d.qv_mult_min})",
    )
    grp.add_argument(
        "--min-bar-qv",
        type=float,
        default=d.min_bar_qv_usd,
        help=f"突破 K 線單根最低成交額 USD (預設 {d.min_bar_qv_usd:.0f})",
    )
    grp.add_argument(
        "--tbr-min",
        type=float,
        default=d.tbr_min,
        help=f"突破 K 線最低主動買盤比例 0~1 (預設 {d.tbr_min})",
    )
    grp.add_argument(
        "--clv-min",
        type=float,
        default=d.clv_min,
        help=f"突破 K 線最低收盤實體位置 CLV 0~1 (預設 {d.clv_min})",
    )
    grp.add_argument(
        "--m1-bbw-max",
        type=float,
        default=d.m1_bbw_max_pct,
        help=f"M1 蓄勢期 6h 布林帶寬上限 %% (預設 {d.m1_bbw_max_pct})",
    )
    grp.add_argument(
        "--m1-rng24h-max",
        type=float,
        default=d.m1_rng24h_max_pct,
        help=f"M1 24h 箱體振幅上限 %% (預設 {d.m1_rng24h_max_pct})",
    )
    grp.add_argument(
        "--m1-vol-accel-min",
        type=float,
        default=d.m1_vol_accel_min,
        help=f"M1 突破當根相對前一根成交額最低加速倍數 (預設 {d.m1_vol_accel_min})",
    )
    grp.add_argument(
        "--m2-bbw-max",
        type=float,
        default=d.m2_bbw_max_pct,
        help=f"M2 極致窒息 6h 布林帶寬上限 %% (預設 {d.m2_bbw_max_pct})",
    )
    grp.add_argument(
        "--m2-rng6h-max",
        type=float,
        default=d.m2_rng6h_max_pct,
        help=f"M2 極致窒息 6h 振幅上限 %% (預設 {d.m2_rng6h_max_pct})",
    )
    grp.add_argument(
        "--m2-qv-mult-min",
        type=float,
        default=d.m2_qv_mult_min,
        help=f"M2 巨量核爆成交額最低倍數 (預設 {d.m2_qv_mult_min})",
    )


def radar_config_from_args(args: argparse.Namespace) -> RadarConfig:
    d = RadarConfig()
    return RadarConfig(
        breakout_buffer_pct=float(getattr(args, "breakout_buffer_pct", d.breakout_buffer_pct)),
        qv_mult_min=float(getattr(args, "qv_mult_min", d.qv_mult_min)),
        min_bar_qv_usd=float(getattr(args, "min_bar_qv", d.min_bar_qv_usd)),
        tbr_min=float(getattr(args, "tbr_min", d.tbr_min)),
        clv_min=float(getattr(args, "clv_min", d.clv_min)),
        m1_bbw_max_pct=float(getattr(args, "m1_bbw_max", d.m1_bbw_max_pct)),
        m1_rng24h_max_pct=float(getattr(args, "m1_rng24h_max", d.m1_rng24h_max_pct)),
        m1_vol_accel_min=float(getattr(args, "m1_vol_accel_min", d.m1_vol_accel_min)),
        m2_bbw_max_pct=float(getattr(args, "m2_bbw_max", d.m2_bbw_max_pct)),
        m2_rng6h_max_pct=float(getattr(args, "m2_rng6h_max", d.m2_rng6h_max_pct)),
        m2_qv_mult_min=float(getattr(args, "m2_qv_mult_min", d.m2_qv_mult_min)),
    )


def api_get(path: str, params: Optional[Dict[str, Any]] = None, retries: int = 4) -> Any:
    global _COOLDOWN_UNTIL
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        wait_cd = 0.0
        with _RATE_LOCK:
            now_m = time.monotonic()
            if _COOLDOWN_UNTIL > now_m:
                wait_cd = _COOLDOWN_UNTIL - now_m
        if wait_cd > 0:
            time.sleep(wait_cd)
        try:
            resp = HTTP.get(BINANCE_FAPI + path, params=params, timeout=15)
            used_w = resp.headers.get("X-MBX-USED-WEIGHT-1M")
            if used_w and used_w.isdigit() and int(used_w) >= 1950:
                with _RATE_LOCK:
                    _COOLDOWN_UNTIL = max(_COOLDOWN_UNTIL, time.monotonic() + 10.0)
            if resp.status_code in (418, 429):
                retry_hdr = resp.headers.get("Retry-After")
                wait_s = float(retry_hdr) if (retry_hdr and retry_hdr.isdigit()) else 2.0 * (2 ** attempt)
                with _RATE_LOCK:
                    _COOLDOWN_UNTIL = max(_COOLDOWN_UNTIL, time.monotonic() + wait_s)
                logger.warning(
                    "Binance API 觸發限流 (HTTP %s, %s)，等待 %.1fs 後重試 (%d/%d)",
                    resp.status_code,
                    path,
                    wait_s,
                    attempt + 1,
                    retries,
                )
                time.sleep(wait_s)
                continue
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            last_err = exc
            wait_s = 0.5 * (2 ** attempt)
            if attempt + 1 < retries:
                logger.warning(
                    "Binance API 請求失敗 (%s): %s，%.1fs 後重試 (%d/%d)",
                    path,
                    exc,
                    wait_s,
                    attempt + 1,
                    retries,
                )
                time.sleep(wait_s)
    raise RuntimeError(f"Binance API failed ({path}): {last_err}")


def list_usdt_perps(min_quote_vol_usd: float = 1_500_000, max_symbols: int = 260) -> List[str]:
    ex_info = api_get("/fapi/v1/exchangeInfo")
    tradable: Set[str] = set()
    for item in ex_info.get("symbols", []):
        sym = item.get("symbol", "")
        if sym in BLACKLIST:
            continue
        if (
            item.get("quoteAsset") == "USDT"
            and item.get("status") == "TRADING"
            and item.get("contractType") == "PERPETUAL"
        ):
            tradable.add(sym)

    tickers = api_get("/fapi/v1/ticker/24hr")
    candidates: List[Tuple[float, str]] = []
    for row in tickers:
        sym = row.get("symbol", "")
        if sym not in tradable:
            continue
        qv = float(row.get("quoteVolume") or 0.0)
        if qv >= min_quote_vol_usd:
            candidates.append((qv, sym))

    candidates.sort(reverse=True)
    return [sym for _, sym in candidates[:max_symbols]]


def compute_radar_indicators(df: pd.DataFrame, interval: str = "15m") -> pd.DataFrame:
    tf = timeframe_windows(interval)
    mid = df["Close"].rolling(20, min_periods=20).mean()
    std = df["Close"].rolling(20, min_periods=20).std()
    df["BB_MID"] = mid
    df["BB_UPPER"] = mid + 2.0 * std
    df["BB_LOWER"] = mid - 2.0 * std
    df["BBW"] = np.where(mid > 0, (4.0 * std) / mid * 100.0, np.nan)

    sma_fast = tf["sma_fast"]
    sma_mid = tf["sma_mid"]
    sma_5d = tf["sma_5d"]
    sma_5d_min_p = tf["sma_5d_min_p"]
    df["SMA60"] = df["Close"].rolling(sma_fast, min_periods=sma_fast).mean()
    df["SMA200"] = df["Close"].rolling(sma_mid, min_periods=sma_mid).mean()
    df["SMA480"] = df["Close"].rolling(sma_5d, min_periods=sma_5d_min_p).mean()

    prev_c = df["Close"].shift(1)
    tr = pd.concat(
        [df["High"] - df["Low"], (df["High"] - prev_c).abs(), (df["Low"] - prev_c).abs()],
        axis=1,
    ).max(axis=1)
    df["ATR_PCT"] = (tr.rolling(14, min_periods=14).mean() / df["Close"]) * 100.0

    bars_24h = tf["bars_24h"]
    bars_12h = tf["bars_12h"]
    min_p_12h = min(bars_12h, max(4, bars_12h // 2))
    df["DON_HIGH"] = df["High"].rolling(bars_24h, min_periods=bars_12h).max().shift(1)
    df["DON_LOW"] = df["Low"].rolling(bars_12h, min_periods=min_p_12h).min().shift(1)

    vol_sum = df["Volume"].rolling(bars_12h, min_periods=min_p_12h).sum()
    qv_sum = df["QuoteVol"].rolling(bars_12h, min_periods=min_p_12h).sum()
    df["VWAP"] = np.where(vol_sum > 0, qv_sum / vol_sum, np.nan)

    qv_win = tf["qv_med_bars"]
    qv_med = df["QuoteVol"].shift(1).rolling(qv_win, min_periods=max(4, qv_win // 2)).median()
    df["QV_MULT"] = np.where(qv_med > 0, df["QuoteVol"] / qv_med, np.nan)
    df["TBR"] = np.where(df["Volume"] > 0, df["TakerBuyVol"] / df["Volume"], np.nan)
    return df


def load_klines_with_orderflow(
    symbol: str,
    interval: str = "15m",
    limit: int = 560,
    drop_forming: bool = True,
) -> pd.DataFrame:
    tf = timeframe_windows(interval)
    bar_ms = tf["bar_ms"]
    eff_limit = max(limit, tf["min_df_len"] + 40) if limit == 560 else limit
    req_limit = min(1500, eff_limit + (1 if drop_forming else 0))
    raw = api_get("/fapi/v1/klines", {"symbol": symbol, "interval": interval, "limit": req_limit})
    if not raw:
        return pd.DataFrame()

    now_ms = int(time.time() * 1000)
    while drop_forming and raw and int(raw[-1][0]) + bar_ms > now_ms:
        raw = raw[:-1]

    if len(raw) < tf["min_df_len"]:
        return pd.DataFrame()

    df = pd.DataFrame(
        {
            "Open": [float(x[1]) for x in raw],
            "High": [float(x[2]) for x in raw],
            "Low": [float(x[3]) for x in raw],
            "Close": [float(x[4]) for x in raw],
            "Volume": [float(x[5]) for x in raw],
            "QuoteVol": [float(x[7]) for x in raw],
            "TakerBuyVol": [float(x[9]) for x in raw],
        },
        index=pd.to_datetime([int(x[0]) for x in raw], unit="ms", utc=True).tz_convert(TPE),
    )
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return compute_radar_indicators(df, interval=interval)


def scan_squeeze_breakouts(
    df: pd.DataFrame,
    symbol: str,
    interval: str = "15m",
    *,
    sniper_only: bool = True,
    cooldown_bars: Optional[int] = None,
    config: Optional[RadarConfig] = None,
) -> List[RadarHit]:
    """雙模狙擊偵測：M1 (VWAP 護盤蓄勢突破) + M2 (極致窒息巨量核爆)，視窗隨 K 線時長自動縮放。"""
    tf = timeframe_windows(interval)
    if df.empty or len(df) < tf["min_df_len"]:
        return []
    cfg = config or RadarConfig()
    cd_bars = tf["cooldown_bars"] if cooldown_bars is None else int(cooldown_bars)
    b6 = tf["bars_6h"]
    b12 = tf["bars_12h"]
    b24 = tf["bars_24h"]
    b_min_win = tf["bars_bbw_min"]
    pre_buf = tf["pre_buf"]
    vh_win = tf["bars_vwap_hold"]
    wick_win = tf["bars_wick"]
    slp_win = tf["slope_bars"]

    o = df["Open"].to_numpy(float)
    h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    c = df["Close"].to_numpy(float)
    qv = df["QuoteVol"].to_numpy(float)
    qvm = df["QV_MULT"].to_numpy(float)
    tbr = df["TBR"].to_numpy(float)
    bbw = df["BBW"].to_numpy(float)
    don_h = df["DON_HIGH"].to_numpy(float)
    vwap = df["VWAP"].to_numpy(float)
    s60 = df["SMA60"].to_numpy(float)
    s200 = df["SMA200"].to_numpy(float)
    s480 = df["SMA480"].to_numpy(float)
    times = df.index

    hits: List[RadarHit] = []
    last_idx = -999
    brk_mult = 1.0 + cfg.breakout_buffer_pct / 100.0
    w6_lookback = b6 + pre_buf
    w12_lookback = b12 + pre_buf
    w24_lookback = b24 + pre_buf

    for i in range(tf["warmup_bars"], len(df)):
        if i - last_idx < cd_bars:
            continue
        if c[i] <= o[i] or o[i] <= 0 or h[i] <= l[i]:
            continue
        if i < w24_lookback or i < slp_win:
            continue
        if not np.isfinite(
            [
                bbw[i],
                don_h[i],
                vwap[i],
                qvm[i],
                tbr[i],
                vwap[i - w6_lookback],
                s60[i],
                s200[i],
                s480[i - 1],
                s60[i - slp_win],
                s200[i - slp_win],
            ]
        ).all():
            continue

        # 基礎突破與訂單流門檻
        if c[i] <= don_h[i] * brk_mult or qvm[i] < cfg.qv_mult_min or qv[i] < cfg.min_bar_qv_usd or tbr[i] < cfg.tbr_min:
            continue

        clv = (c[i] - l[i]) / (h[i] - l[i])
        gain = (c[i] - o[i]) / o[i] * 100.0
        ext_vwap = (c[i] - vwap[i]) / vwap[i] * 100.0
        if clv < cfg.clv_min or gain < cfg.bar_gain_min_pct or gain > cfg.bar_gain_max_pct or ext_vwap > cfg.ext_vwap_max_pct:
            continue

        w_bbw = bbw[i - w6_lookback : i - pre_buf]
        if len(w_bbw) == 0 or not np.isfinite(w_bbw).all():
            continue
        b_max24 = float(np.max(w_bbw))
        b_min10 = float(np.nanmin(bbw[i - b_min_win : i]))

        hi24 = float(np.max(h[i - w6_lookback : i - pre_buf]))
        lo24 = float(np.min(l[i - w6_lookback : i - pre_buf]))
        rng24 = (hi24 - lo24) / lo24 * 100.0
        lo48 = float(np.min(l[i - w12_lookback : i - pre_buf]))
        rng48 = (float(np.max(h[i - w12_lookback : i - pre_buf])) - lo48) / lo48 * 100.0
        lo96 = float(np.min(l[i - w24_lookback : i - pre_buf]))
        rng96 = (float(np.max(h[i - w24_lookback : i - pre_buf])) - lo96) / lo96 * 100.0

        vwap_slp = (vwap[i - pre_buf] - vwap[i - w6_lookback]) / vwap[i - w6_lookback] * 100.0
        if b_max24 > cfg.base_bbw_max_pct or rng24 > cfg.base_rng6h_max_pct or vwap_slp < 0.0:
            continue

        # 排除前 1.5h 內已有爆量長針洗盤
        had_wick = False
        for k in range(i - wick_win, i):
            if l[k] > 0 and (h[k] - l[k]) / l[k] * 100.0 > 4.2 and qvm[k] > 6.5:
                had_wick = True
                break
        if had_wick:
            continue

        abv_vwap = int(np.sum(c[i - vh_win : i] >= vwap[i - vh_win : i]))
        tbr_hold = float(np.nanmean(tbr[i - vh_win : i])) * 100.0
        vol_acc = float(qv[i] / max(qv[i - 1], 1.0))
        dist_s480 = (c[i - 1] - s480[i - 1]) / s480[i - 1] * 100.0
        slp_s60 = (s60[i] - s60[i - slp_win]) / s60[i - slp_win] * 100.0
        slp_s200 = (s200[i] - s200[i - slp_win]) / s200[i - slp_win] * 100.0

        has_accum_flow = (
            (b_min10 >= 1.80 and (tbr_hold >= 46.4 or qvm[i] >= 40.0))
            or (cfg.m1_bbw_min_low <= b_min10 < 1.80 and tbr_hold >= 48.0)
        )
        is_m1 = (
            abv_vwap == vh_win
            and cfg.m1_bbw_min_low <= b_min10 <= cfg.m1_bbw_min_high
            and b_max24 <= cfg.m1_bbw_max_pct
            and has_accum_flow
            and rng96 <= cfg.m1_rng24h_max_pct
            and rng48 <= cfg.m1_rng12h_max_pct
            and cfg.m1_dist_s480_min_pct <= dist_s480 <= cfg.m1_dist_s480_max_pct
            and slp_s60 >= cfg.m1_slp_s60_min_pct
            and slp_s200 >= cfg.m1_slp_s200_min_pct
            and vol_acc >= cfg.m1_vol_accel_min
        )
        is_m2 = (
            b_max24 <= cfg.m2_bbw_max_pct
            and rng24 <= cfg.m2_rng6h_max_pct
            and qvm[i] >= cfg.m2_qv_mult_min
            and clv >= cfg.m2_clv_min
            and tbr[i] >= cfg.m2_tbr_min
        )

        if sniper_only and not (is_m1 or is_m2):
            continue

        mode = "ULTRA_SQUEEZE_CLIMAX" if is_m2 else "VWAP_COIL_BREAK"
        stop_price = min(l[i], lo24) * 0.995
        risk = c[i] - stop_price
        if risk <= 0 or (risk / c[i] * 100.0) < cfg.min_risk_pct or (risk / c[i] * 100.0) > cfg.max_risk_pct:
            continue

        future_high = float(np.max(h[i + 1 :])) if (i + 1) < len(h) else float(c[i])
        max_gain_after = max(0.0, (future_high - c[i]) / c[i] * 100.0)

        s_sq = max(0.0, min(25.0, (7.0 - b_min10) / 7.0 * 25.0))
        s_vol = max(0.0, min(30.0, min(qvm[i], 30.0) / 30.0 * 30.0))
        s_tbr = max(0.0, min(25.0, (tbr[i] - 0.50) / 0.18 * 25.0))
        s_clv = max(0.0, min(20.0, (clv - 0.65) / 0.35 * 20.0))
        score = min(99, int(round(s_sq + s_vol + s_tbr + s_clv + (8 if is_m1 or is_m2 else 0))))

        note = (
            f"★M2極致窒息核爆：6h帶寬≤{b_max24:.1f}% + 振幅{rng24:.1f}% → {qvm[i]:.1f}x巨量 + 主買{tbr[i]*100:.1f}% + 實體{clv*100:.0f}%"
            if is_m2
            else f"★M1護盤蓄勢突破：連{vh_win}根站穩VWAP + 24h箱體({rng96:.1f}%) + 帶寬({b_min10:.1f}%~{b_max24:.1f}%) → {qvm[i]:.1f}x放量(加速{vol_acc:.1f}x) + 主買{tbr[i]*100:.1f}%"
        )

        last_idx = i
        hits.append(
            RadarHit(
                symbol=symbol,
                interval=interval,
                mode=mode,
                bar_idx=i,
                timestamp=times[i],
                price=c[i],
                stop_price=stop_price,
                breakout_level=don_h[i],
                vwap=vwap[i],
                quote_vol_usd=qv[i],
                qv_mult=qvm[i],
                vol_accel=vol_acc,
                taker_buy_pct=tbr[i] * 100.0,
                bar_gain_pct=gain,
                clv=clv,
                bbw_min10_pct=b_min10,
                bbw_max24_pct=b_max24,
                rng24_pct=rng24,
                rng96_pct=rng96,
                score=score,
                max_gain_after_pct=max_gain_after,
                note=note,
            )
        )

    return hits


def attach_oi_and_funding(hits: Sequence[RadarHit]) -> None:
    cache: Dict[Tuple[str, str], Tuple[Optional[float], Optional[float]]] = {}
    valid_oi_periods = {"5m", "15m", "30m", "1h", "2h", "4h"}
    for hit in hits:
        oi_period = hit.interval if hit.interval in valid_oi_periods else "15m"
        key = (hit.symbol, oi_period)
        if key not in cache:
            oi_chg: Optional[float] = None
            fr_pct: Optional[float] = None
            try:
                oi = api_get("/futures/data/openInterestHist", {"symbol": hit.symbol, "period": oi_period, "limit": 6})
                if isinstance(oi, list) and len(oi) >= 4:
                    p0 = float(oi[0]["sumOpenInterest"])
                    p1 = float(oi[-1]["sumOpenInterest"])
                    if p0 > 0:
                        oi_chg = (p1 - p0) / p0 * 100.0
            except Exception as exc:
                logger.warning("無法取得 %s 未平倉量 (OI): %s", hit.symbol, exc)
            try:
                pm = api_get("/fapi/v1/premiumIndex", {"symbol": hit.symbol})
                if isinstance(pm, dict) and pm.get("lastFundingRate") is not None:
                    fr_pct = float(pm["lastFundingRate"]) * 100.0
            except Exception as exc:
                logger.warning("無法取得 %s 即時資金費率: %s", hit.symbol, exc)
            cache[key] = (oi_chg, fr_pct)

        hit.oi_change_pct, hit.funding_rate_pct = cache[key]


def render_radar_chart(
    df: pd.DataFrame,
    hits: Sequence[RadarHit],
    symbol: str,
    interval: str,
    out_path: Path,
    lookback_bars: int = 110,
    exit_info: Optional[Dict[datetime, Dict[str, Any]]] = None,
) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    plot_df = df.tail(lookback_bars).copy()
    if plot_df.empty:
        return

    fig, (ax_p, ax_v, ax_t) = plt.subplots(
        3, 1, figsize=(14.5, 9.2), gridspec_kw={"height_ratios": [3.3, 1.0, 0.85]}, sharex=True
    )
    fig.patch.set_facecolor("#0f172a")
    for ax in (ax_p, ax_v, ax_t):
        ax.set_facecolor("#0f172a")
        ax.tick_params(colors="#94a3b8")
        for spine in ax.spines.values():
            spine.set_color("#334155")

    x = np.arange(len(plot_df))
    o = plot_df["Open"].to_numpy(float)
    h = plot_df["High"].to_numpy(float)
    l = plot_df["Low"].to_numpy(float)
    c = plot_df["Close"].to_numpy(float)
    qv = plot_df["QuoteVol"].to_numpy(float) / 1e6
    tbr = plot_df["TBR"].to_numpy(float) * 100.0
    times = plot_df.index

    for idx in range(len(plot_df)):
        col = "#0ecb81" if c[idx] >= o[idx] else "#f6465d"
        ax_p.plot([idx, idx], [l[idx], h[idx]], color=col, linewidth=1.0, zorder=2)
        bh = max(abs(c[idx] - o[idx]), (h[idx] - l[idx]) * 0.02, 1e-8)
        ax_p.add_patch(Rectangle((idx - 0.32, min(o[idx], c[idx])), 0.64, bh, facecolor=col, edgecolor=col, zorder=3))
        ax_v.bar(idx, qv[idx], width=0.68, color=col, alpha=0.85)
        t_col = "#0ecb81" if (np.isfinite(tbr[idx]) and tbr[idx] >= 50.0) else "#f6465d"
        if np.isfinite(tbr[idx]):
            ax_t.bar(idx, tbr[idx] - 50.0, bottom=50.0, width=0.68, color=t_col, alpha=0.85)

    ax_p.fill_between(
        x, plot_df["BB_LOWER"].to_numpy(float), plot_df["BB_UPPER"].to_numpy(float),
        color="#38bdf8", alpha=0.10, label="布林壓縮通道 (20,2)"
    )
    ax_p.plot(x, plot_df["BB_MID"].to_numpy(float), color="#38bdf8", linewidth=1.2, alpha=0.85, label="BB中軌 (移動停利線)")
    ax_p.plot(x, plot_df["DON_HIGH"].to_numpy(float), color="#f59e0b", linestyle="--", linewidth=1.4, label="24h 結構高點 (Donchian)")
    ax_p.plot(x, plot_df["VWAP"].to_numpy(float), color="#c084fc", linewidth=1.5, label="12h VWAP 機構成本線")

    ax_t.axhline(50.0, color="#64748b", linestyle="--", linewidth=1.0)
    ax_t.set_ylim(25, 75)

    ts_to_pos = {ts: idx for idx, ts in enumerate(times)}
    y_span = max(float(np.nanmax(h) - np.nanmin(l)), 1e-8)
    for k, hit in enumerate(hits):
        if hit.timestamp not in ts_to_pos:
            continue
        pos = ts_to_pos[hit.timestamp]
        badge_col = "#d97706" if hit.mode == "ULTRA_SQUEEZE_CLIMAX" else "#059669"
        mode_lbl = "★ M2 極致窒息核爆" if hit.mode == "ULTRA_SQUEEZE_CLIMAX" else "★ M1 VWAP護盤突破"

        ax_p.scatter([pos], [l[pos] * 0.985], marker="^", s=180, color="#facc15", edgecolors="#0f172a", linewidths=1.2, zorder=8)
        extra = f"爆量: {hit.qv_mult:.1f}x | 主買: {hit.taker_buy_pct:.1f}% | 最高: +{hit.max_gain_after_pct:.1f}%"
        if exit_info and hit.timestamp in exit_info:
            ex = exit_info[hit.timestamp]
            extra = f"出場: {ex['exit_price']:.6g} ({ex['reason']}) | 實戰 1x: {ex['pnl']:+.1f}% ({ex['r']:+.2f}R)"
            ex_ts = ex.get("exit_time")
            if ex_ts in ts_to_pos:
                epos = ts_to_pos[ex_ts]
                ax_p.scatter([epos], [h[epos] * 1.015], marker="v", s=160, color="#f43f5e", edgecolors="#ffffff", linewidths=1.1, zorder=8)

        msg = (
            f"{mode_lbl} (評分 {hit.score})\n"
            f"時間: {hit.timestamp.strftime('%Y-%m-%d %H:%M')} | 進場: {hit.price:.6g} | 止損: {hit.stop_price:.6g}\n"
            f"{extra}"
        )
        y_off = l[pos] + (0.22 + 0.14 * (k % 2)) * y_span
        ax_p.annotate(
            msg,
            xy=(pos, h[pos]),
            xytext=(max(2, pos - 28), min(float(np.nanmax(h)) * 0.93, y_off)),
            fontsize=9.0,
            fontweight="bold",
            color="white",
            bbox={"boxstyle": "round,pad=0.45", "facecolor": badge_col, "alpha": 0.94, "edgecolor": "#facc15"},
            arrowprops={"arrowstyle": "->", "color": "#facc15", "lw": 1.6},
            zorder=9,
        )

    ax_p.set_title(f"{symbol} ({interval}) 雙模狙擊起漲雷達 — VWAP 護盤蓄勢 × 布林壓縮 × 主動買盤", fontsize=13, fontweight="bold", color="#f8fafc", pad=10)
    ax_p.set_ylabel("價格 (USDT)", color="#cbd5e1")
    ax_p.grid(True, linestyle="--", alpha=0.18, color="#475569")
    ax_p.legend(loc="upper left", fontsize=8.8, facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")

    ax_v.set_ylabel("成交額 (M)", color="#cbd5e1")
    ax_v.grid(True, linestyle="--", alpha=0.18, color="#475569")

    ax_t.set_ylabel("主動買盤 (%)", color="#cbd5e1")
    ax_t.grid(True, linestyle="--", alpha=0.18, color="#475569")

    step = max(1, len(plot_df) // 10)
    ticks = list(range(0, len(plot_df), step))
    ax_t.set_xticks(ticks)
    ax_t.set_xticklabels([times[i].strftime("%m-%d\n%H:%M") for i in ticks], fontsize=8.5, color="#94a3b8")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def load_seen_state(path: Path) -> Set[Tuple[str, str]]:
    if not path.is_file():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return {(str(item[0]), str(item[1])) for item in data if isinstance(item, (list, tuple)) and len(item) >= 2}
    except Exception as exc:
        logger.warning("讀取盯盤已通知紀錄失敗 (%s): %s", path, exc)
    return set()


def save_seen_state(path: Path, seen: Set[Tuple[str, str]], max_items: int = 5000) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        ordered = sorted(seen, key=lambda kv: (kv[1], kv[0]))[-max_items:]
        tmp_path = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        tmp_path.write_text(json.dumps(ordered, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_path, path)
    except Exception as exc:
        logger.warning("寫入盯盤已通知紀錄失敗 (%s): %s", path, exc)


def run_radar(
    symbols: Sequence[str],
    interval: str = "15m",
    recent_bars: int = 8,
    workers: int = 12,
    limit: int = 560,
    sniper_only: bool = True,
    plot_dir: Optional[Path] = None,
    drop_forming: bool = True,
    config: Optional[RadarConfig] = None,
) -> List[RadarHit]:
    hits: List[RadarHit] = []

    def _scan_one(sym: str) -> List[RadarHit]:
        try:
            df = load_klines_with_orderflow(sym, interval=interval, limit=limit, drop_forming=drop_forming)
            res = scan_squeeze_breakouts(df, sym, interval=interval, sniper_only=sniper_only, config=config)
            if not res:
                return []
            if recent_bars > 0:
                min_idx = max(0, len(df) - recent_bars)
                res = [x for x in res if x.bar_idx >= min_idx]
            if res and plot_dir is not None:
                render_radar_chart(df, res, sym, interval, plot_dir / f"{sym}_{interval}_squeeze_radar.png")
            return res
        except Exception as exc:
            logger.warning("掃描 %s (%s) 失敗: %s", sym, interval, exc)
            return []

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_scan_one, s) for s in symbols]
        for fut in as_completed(futs):
            res = fut.result()
            if res:
                hits.extend(res)

    if hits:
        attach_oi_and_funding(hits)
    hits.sort(key=lambda x: (x.timestamp, x.score, x.qv_mult), reverse=True)
    return hits


def print_hits(hits: Sequence[RadarHit]) -> None:
    if not hits:
        print("目前時間窗口內無符合「雙模狙擊起漲雷達」條件的訊號。")
        return

    print("=" * 140)
    print(
        f"{'時間 (TPE)':<17} | {'幣種':<12} | {'模式':<18} | {'評分':<4} | {'進場價':<10} | "
        f"{'止損價':<10} | {'成交額倍數':<10} | {'主動買盤':<8} | {'OI變化':<8} | {'後續最高'}"
    )
    print("-" * 140)
    for h in hits:
        m_str = "★M2極致窒息核爆" if h.mode == "ULTRA_SQUEEZE_CLIMAX" else "★M1護盤蓄勢突破"
        oi_str = f"{h.oi_change_pct:+.1f}%" if h.oi_change_pct is not None else "N/A"
        print(
            f"{h.timestamp.strftime('%Y-%m-%d %H:%M'):<17} | {h.symbol:<12} | {m_str:<16} | {h.score:>3}  | "
            f"{h.price:<10.6g} | {h.stop_price:<10.6g} | {h.qv_mult:>6.1f}x    | "
            f"{h.taker_buy_pct:>5.1f}%   | {oi_str:>8} | +{h.max_gain_after_pct:.1f}%"
        )
        print(f"   └─ {h.note}")
    print("=" * 140)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError, OSError) as exc:
            logger.debug("stdout UTF-8 reconfigure skipped: %s", exc)

    parser = argparse.ArgumentParser(description="幣安永續合約「Squeeze × VWAP護盤 × 主動買盤」雙模狙擊起漲雷達")
    parser.add_argument("--symbols", type=str, default="", help="指定幣種測試，逗號分隔 (例如 BTCUSDT,SOLUSDT)")
    parser.add_argument("--interval", type=str, default="15m", help="K 線週期 (5m, 15m, 30m, 1h)，預設 15m")
    parser.add_argument("--limit", type=int, default=560, help="拉取 K 線根數 (預設 560，短週期自動擴充至暖機長度)")
    parser.add_argument("--scan", action="store_true", help="執行單次全市場掃描")
    parser.add_argument("--watch", type=int, default=0, help="持續輪詢監控秒數 (例如 --watch 60)")
    parser.add_argument("--relaxed", action="store_true", help="切換為寬鬆預警模式（顯示所有初階壓縮突破）")
    parser.add_argument("--min-vol", type=float, default=1_500_000, help="24h 最低成交額 (USDT)，預設 150 萬")
    parser.add_argument("--max-symbols", type=int, default=260, help="最多掃描幣種數，預設 260")
    parser.add_argument("--recent-bars", type=int, default=10, help="掃描模式顯示最近 N 根內的新起漲訊號 (預設 10)")
    parser.add_argument("--plot", action="store_true", help="輸出深色主題三合一雷達圖 PNG")
    parser.add_argument("--include-forming", action="store_true", help="納入尚未收盤的當根 K 線（預設僅掃描已收盤 K 線避免重繪）")
    parser.add_argument("--seen-file", type=str, default="", help="--watch 已推播訊號持久化 JSON 檔案路徑")
    parser.add_argument("--out-dir", type=str, default="breakout_output", help="圖表輸出資料夾 (預設 breakout_output)")
    add_radar_threshold_args(parser)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    sniper_only = not args.relaxed
    drop_forming = not args.include_forming
    cfg = radar_config_from_args(args)

    if args.symbols:
        syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        hits = run_radar(
            syms,
            interval=args.interval,
            recent_bars=0,
            workers=min(8, len(syms)),
            limit=args.limit,
            sniper_only=sniper_only,
            plot_dir=out_dir if args.plot else None,
            drop_forming=drop_forming,
            config=cfg,
        )
        print_hits(hits)
        if args.plot and hits:
            print(f"[圖表已輸出] {out_dir.resolve()}")
        return 0

    if not args.scan and args.watch <= 0:
        args.scan = True

    seen_path = Path(args.seen_file) if args.seen_file else (out_dir / ".watch_seen_squeeze.json")
    seen: Set[Tuple[str, str]] = load_seen_state(seen_path) if args.watch > 0 else set()
    while True:
        now_str = datetime.now(TPE).strftime("%Y-%m-%d %H:%M:%S")
        syms = list_usdt_perps(min_quote_vol_usd=args.min_vol, max_symbols=args.max_symbols)
        mode_tag = "雙模狙擊實戰版" if sniper_only else "寬鬆預警版"
        print(f"\n[{now_str} TPE] 雷達掃描 {len(syms)} 檔 USDT 永續合約 ({args.interval} | {mode_tag} | 已收盤K={drop_forming})...")
        hits = run_radar(
            syms,
            interval=args.interval,
            recent_bars=args.recent_bars,
            limit=args.limit,
            sniper_only=sniper_only,
            plot_dir=out_dir if args.plot else None,
            drop_forming=drop_forming,
            config=cfg,
        )

        if args.watch > 0:
            fresh = []
            for h in hits:
                key = (h.symbol, h.timestamp.isoformat())
                if key not in seen:
                    seen.add(key)
                    fresh.append(h)
            if fresh:
                save_seen_state(seen_path, seen)
                print_hits(fresh)
                print(f"\a[起漲雷達] 剛偵測到 {len(fresh)} 筆新狙擊起漲訊號！")
            else:
                print(f"本輪無新增已收盤起漲訊號（視窗內 {len(hits)} 筆既有訊號已通知過）。")
        else:
            print_hits(hits)

        if args.plot and hits:
            print(f"[圖表已輸出] {out_dir.resolve()}")
        if args.watch <= 0:
            break
        time.sleep(args.watch)
    return 0


if __name__ == "__main__":
    sys.exit(main())
