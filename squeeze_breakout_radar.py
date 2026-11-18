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
from pathlib import Path
import sys
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

TPE = ZoneInfo("Asia/Taipei")
BINANCE_FAPI = "https://fapi.binance.com"
HTTP = requests.Session()
HTTP.headers.update({"User-Agent": "Mozilla/5.0", "Accept": "application/json"})

INTERVAL_MS = {
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
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


def api_get(path: str, params: Optional[Dict[str, Any]] = None, retries: int = 3) -> Any:
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            resp = HTTP.get(BINANCE_FAPI + path, params=params, timeout=15)
            if resp.status_code in (418, 429):
                time.sleep(1.2 * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            last_err = exc
            time.sleep(0.35 * (attempt + 1))
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
    mid = df["Close"].rolling(20, min_periods=20).mean()
    std = df["Close"].rolling(20, min_periods=20).std()
    df["BB_MID"] = mid
    df["BB_UPPER"] = mid + 2.0 * std
    df["BB_LOWER"] = mid - 2.0 * std
    df["BBW"] = np.where(mid > 0, (4.0 * std) / mid * 100.0, np.nan)

    df["SMA60"] = df["Close"].rolling(60, min_periods=60).mean()
    df["SMA200"] = df["Close"].rolling(200, min_periods=200).mean()
    df["SMA480"] = df["Close"].rolling(480, min_periods=240).mean()

    prev_c = df["Close"].shift(1)
    tr = pd.concat(
        [df["High"] - df["Low"], (df["High"] - prev_c).abs(), (df["Low"] - prev_c).abs()],
        axis=1,
    ).max(axis=1)
    df["ATR_PCT"] = (tr.rolling(14, min_periods=14).mean() / df["Close"]) * 100.0

    bar_ms = INTERVAL_MS.get(interval, 900_000)
    bars_24h = max(24, min(144, int(86_400_000 // bar_ms)))
    bars_12h = max(12, bars_24h // 2)
    df["DON_HIGH"] = df["High"].rolling(bars_24h, min_periods=bars_12h).max().shift(1)
    df["DON_LOW"] = df["Low"].rolling(bars_12h, min_periods=12).min().shift(1)

    vol_sum = df["Volume"].rolling(bars_12h, min_periods=12).sum()
    qv_sum = df["QuoteVol"].rolling(bars_12h, min_periods=12).sum()
    df["VWAP"] = np.where(vol_sum > 0, qv_sum / vol_sum, np.nan)

    qv_med = df["QuoteVol"].shift(1).rolling(20, min_periods=10).median()
    df["QV_MULT"] = np.where(qv_med > 0, df["QuoteVol"] / qv_med, np.nan)
    df["TBR"] = np.where(df["Volume"] > 0, df["TakerBuyVol"] / df["Volume"], np.nan)
    return df


def load_klines_with_orderflow(symbol: str, interval: str = "15m", limit: int = 560) -> pd.DataFrame:
    raw = api_get("/fapi/v1/klines", {"symbol": symbol, "interval": interval, "limit": min(1500, limit)})
    if not raw or len(raw) < 260:
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
    cooldown_bars: int = 12,
) -> List[RadarHit]:
    """雙模狙擊偵測：M1 (VWAP 護盤蓄勢突破) + M2 (極致窒息巨量核爆)。"""
    if df.empty or len(df) < 260:
        return []

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

    for i in range(250, len(df)):
        if i - last_idx < cooldown_bars:
            continue
        if c[i] <= o[i] or o[i] <= 0 or h[i] <= l[i]:
            continue
        if not np.isfinite([bbw[i], don_h[i], vwap[i], qvm[i], tbr[i], vwap[i - 26], s60[i], s200[i], s480[i - 1]]).all():
            continue

        # 基礎突破與訂單流門檻
        if c[i] <= don_h[i] * 1.004 or qvm[i] < 6.0 or qv[i] < 250_000 or tbr[i] < 0.52:
            continue

        clv = (c[i] - l[i]) / (h[i] - l[i])
        gain = (c[i] - o[i]) / o[i] * 100.0
        ext_vwap = (c[i] - vwap[i]) / vwap[i] * 100.0
        if clv < 0.72 or gain < 2.2 or gain > 18.0 or ext_vwap > 14.0:
            continue

        w_bbw = bbw[i - 26 : i - 2]
        if not np.isfinite(w_bbw).all():
            continue
        b_max24 = float(np.max(w_bbw))
        b_min10 = float(np.nanmin(bbw[i - 10 : i]))

        hi24 = float(np.max(h[i - 26 : i - 2]))
        lo24 = float(np.min(l[i - 26 : i - 2]))
        rng24 = (hi24 - lo24) / lo24 * 100.0
        rng48 = (float(np.max(h[i - 50 : i - 2])) - float(np.min(l[i - 50 : i - 2]))) / float(np.min(l[i - 50 : i - 2])) * 100.0
        rng96 = (float(np.max(h[i - 98 : i - 2])) - float(np.min(l[i - 98 : i - 2]))) / float(np.min(l[i - 98 : i - 2])) * 100.0

        vwap_slp = (vwap[i - 2] - vwap[i - 26]) / vwap[i - 26] * 100.0
        if b_max24 > 7.5 or rng24 > 7.5 or vwap_slp < 0.0:
            continue

        # 排除前 6 根內已有爆量長針洗盤
        had_wick = False
        for k in range(i - 6, i):
            if l[k] > 0 and (h[k] - l[k]) / l[k] * 100.0 > 4.2 and qvm[k] > 6.5:
                had_wick = True
                break
        if had_wick:
            continue

        abv_vwap8 = int(np.sum(c[i - 8 : i] >= vwap[i - 8 : i]))
        tbr_p8 = float(np.nanmean(tbr[i - 8 : i])) * 100.0
        vol_acc = float(qv[i] / max(qv[i - 1], 1.0))
        dist_s60 = (c[i - 1] - s60[i - 1]) / s60[i - 1] * 100.0
        dist_s480 = (c[i - 1] - s480[i - 1]) / s480[i - 1] * 100.0
        slp_s60 = (s60[i] - s60[i - 20]) / s60[i - 20] * 100.0
        slp_s200 = (s200[i] - s200[i - 20]) / s200[i - 20] * 100.0

        # v3 全年去噪：排除 2.0%~3.7% 無力溫吞漂移帶（僅保留貼線底部起漲 ≤2.0% 或強勢空中旗形 ≥3.7%）
        # 並要求蓄勢期前 8 根平均主動買盤具備承接底氣
        has_accum_flow = (
            (b_min10 >= 1.80 and (tbr_p8 >= 46.4 or qvm[i] >= 40.0))
            or (0.90 <= b_min10 < 1.80 and tbr_p8 >= 48.0)
        )
        is_m1 = (
            abv_vwap8 == 8
            and 0.90 <= b_min10 <= 4.55
            and b_max24 <= 6.90
            and has_accum_flow
            and rng96 <= 11.5
            and rng48 <= 8.20
            and -2.0 <= dist_s480 <= 12.0
            and (dist_s60 <= 2.00 or dist_s60 >= 3.70)
            and slp_s60 >= 0.52
            and slp_s200 >= -0.10
            and vol_acc >= 1.70
        )
        is_m2 = (
            b_max24 <= 2.20
            and rng24 <= 3.00
            and qvm[i] >= 30.0
            and clv >= 0.90
            and tbr[i] >= 0.54
        )

        if sniper_only and not (is_m1 or is_m2):
            continue

        mode = "ULTRA_SQUEEZE_CLIMAX" if is_m2 else "VWAP_COIL_BREAK"
        stop_price = min(l[i], lo24) * 0.995
        risk = c[i] - stop_price
        if risk <= 0 or (risk / c[i] * 100.0) < 0.8 or (risk / c[i] * 100.0) > 14.0:
            continue

        future_high = float(np.max(h[i:]))
        max_gain_after = (future_high - c[i]) / c[i] * 100.0

        s_sq = max(0.0, min(25.0, (7.0 - b_min10) / 7.0 * 25.0))
        s_vol = max(0.0, min(30.0, min(qvm[i], 30.0) / 30.0 * 30.0))
        s_tbr = max(0.0, min(25.0, (tbr[i] - 0.50) / 0.18 * 25.0))
        s_clv = max(0.0, min(20.0, (clv - 0.65) / 0.35 * 20.0))
        score = min(99, int(round(s_sq + s_vol + s_tbr + s_clv + (8 if is_m1 or is_m2 else 0))))

        note = (
            f"★M2極致窒息核爆：6h帶寬≤{b_max24:.1f}% + 振幅{rng24:.1f}% → {qvm[i]:.1f}x巨量 + 主買{tbr[i]*100:.1f}% + 實體{clv*100:.0f}%"
            if is_m2
            else f"★M1護盤蓄勢突破：連8根站穩VWAP + 24h箱體({rng96:.1f}%) + 帶寬({b_min10:.1f}%~{b_max24:.1f}%) → {qvm[i]:.1f}x放量(加速{vol_acc:.1f}x) + 主買{tbr[i]*100:.1f}%"
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
    cache: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    for hit in hits:
        if hit.symbol not in cache:
            oi_chg: Optional[float] = None
            fr_pct: Optional[float] = None
            try:
                oi = api_get("/futures/data/openInterestHist", {"symbol": hit.symbol, "period": "15m", "limit": 6})
                if isinstance(oi, list) and len(oi) >= 4:
                    p0 = float(oi[0]["sumOpenInterest"])
                    p1 = float(oi[-1]["sumOpenInterest"])
                    if p0 > 0:
                        oi_chg = (p1 - p0) / p0 * 100.0
            except Exception:
                pass
            try:
                pm = api_get("/fapi/v1/premiumIndex", {"symbol": hit.symbol})
                if isinstance(pm, dict) and pm.get("lastFundingRate") is not None:
                    fr_pct = float(pm["lastFundingRate"]) * 100.0
            except Exception:
                pass
            cache[hit.symbol] = (oi_chg, fr_pct)

        hit.oi_change_pct, hit.funding_rate_pct = cache[hit.symbol]


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


def run_radar(
    symbols: Sequence[str],
    interval: str = "15m",
    recent_bars: int = 8,
    workers: int = 12,
    limit: int = 560,
    sniper_only: bool = True,
    plot_dir: Optional[Path] = None,
) -> List[RadarHit]:
    hits: List[RadarHit] = []

    def _scan_one(sym: str) -> List[RadarHit]:
        try:
            df = load_klines_with_orderflow(sym, interval=interval, limit=limit)
            res = scan_squeeze_breakouts(df, sym, interval=interval, sniper_only=sniper_only)
            if not res:
                return []
            if recent_bars > 0:
                min_idx = max(0, len(df) - recent_bars)
                res = [x for x in res if x.bar_idx >= min_idx]
            if res and plot_dir is not None:
                render_radar_chart(df, res, sym, interval, plot_dir / f"{sym}_{interval}_squeeze_radar.png")
            return res
        except Exception:
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
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="幣安永續合約「Squeeze × VWAP護盤 × 主動買盤」雙模狙擊起漲雷達 v2")
    parser.add_argument("--symbols", type=str, default="", help="指定幣種測試，逗號分隔 (例如 EVAAUSDT,SHELLUSDT)")
    parser.add_argument("--interval", type=str, default="15m", help="K 線週期 (5m, 15m, 1h)，預設 15m")
    parser.add_argument("--limit", type=int, default=560, help="拉取 K 線根數，預設 560")
    parser.add_argument("--scan", action="store_true", help="執行單次全市場掃描")
    parser.add_argument("--watch", type=int, default=0, help="持續輪詢監控秒數 (例如 --watch 60)")
    parser.add_argument("--relaxed", action="store_true", help="切換為寬鬆預警模式（顯示所有初階壓縮突破）")
    parser.add_argument("--min-vol", type=float, default=1_500_000, help="24h 最低成交額 (USDT)，預設 150 萬")
    parser.add_argument("--max-symbols", type=int, default=260, help="最多掃描幣種數，預設 260")
    parser.add_argument("--recent-bars", type=int, default=10, help="掃描模式顯示最近 N 根內的新起漲訊號")
    parser.add_argument("--plot", action="store_true", help="輸出深色主題三合一雷達圖 PNG")
    parser.add_argument("--out-dir", type=str, default="breakout_output", help="圖表輸出資料夾")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    sniper_only = not args.relaxed

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
        )
        print_hits(hits)
        if args.plot and hits:
            print(f"[圖表已輸出] {out_dir.resolve()}")
        return 0

    if not args.scan and args.watch <= 0:
        args.scan = True

    seen: Set[Tuple[str, str]] = set()
    while True:
        now_str = datetime.now(TPE).strftime("%Y-%m-%d %H:%M:%S")
        syms = list_usdt_perps(min_quote_vol_usd=args.min_vol, max_symbols=args.max_symbols)
        mode_tag = "雙模狙擊實戰版" if sniper_only else "寬鬆預警版"
        print(f"\n[{now_str} TPE] 雷達掃描 {len(syms)} 檔 USDT 永續合約 ({args.interval} | {mode_tag})...")
        hits = run_radar(
            syms,
            interval=args.interval,
            recent_bars=args.recent_bars,
            limit=args.limit,
            sniper_only=sniper_only,
            plot_dir=out_dir if args.plot else None,
        )

        if args.watch > 0:
            fresh = []
            for h in hits:
                key = (h.symbol, h.timestamp.isoformat())
                if key not in seen:
                    seen.add(key)
                    fresh.append(h)
            print_hits(fresh if fresh else hits)
            if fresh:
                print(f"\a[起漲雷達] 剛偵測到 {len(fresh)} 筆新狙擊起漲訊號！")
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
