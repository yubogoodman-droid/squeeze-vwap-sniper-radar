#!/usr/bin/env python3
"""「Squeeze × VWAP 護盤 × 主動買盤」雙模狙擊 v2 — 6 個月 (180天) 全市場實戰回測腳本

完全獨立運作，只依賴 squeeze_breakout_radar.py，不更動任何原有腳本。
特色：
1. 內建 Binance Weight 流量偵測與本地快取 (breakout_output/cache_180d/)，重跑只需 ~10 秒。
2. 採用 v2 雙模狙擊引擎：
   - 【M1：VWAP 護盤蓄勢突破】連 8 根站穩 12h VWAP + 24h 極窄箱體 + 中期趨勢翻揚 + 放量加速突破
   - 【M2：極致窒息巨量核爆】6h BBW ≤ 2.2% 極度窒息後單根 30x+ 實體巨量噴發
3. 採用「抗洗盤回測容忍 + 0.6R 保本鎖定 + 跌破 BB 中軌止盈」實戰出場（已扣 0.12% 費滑）。
4. 自動產出每筆交易的高畫質復盤圖卡 (cards/*.png)、資金曲線與手機/電腦自適應 HTML 戰報。
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from html import escape
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import backtest_utils as bt_utils
import squeeze_breakout_radar as radar

TPE = ZoneInfo("Asia/Taipei")
FEE_ROUNDTRIP_PCT = 0.12
BREAKEVEN_ARM_R = 0.55
BREAKEVEN_BUFFER_PCT = 0.30
WEIGHT_LOCK = threading.Lock()
LAST_WEIGHT = 0
COOLDOWN_UNTIL = 0.0


def polite_get_klines(params: Dict[str, Any], retries: int = 6) -> Any:
    """帶 Binance 1 分鐘 Weight 自適應限流的 GET 請求，避免觸發 418/429。"""
    global LAST_WEIGHT, COOLDOWN_UNTIL
    for attempt in range(retries):
        wait_now = 0.0
        with WEIGHT_LOCK:
            now_m = time.monotonic()
            if LAST_WEIGHT >= 1950:
                COOLDOWN_UNTIL = max(COOLDOWN_UNTIL, now_m + 12.0)
                LAST_WEIGHT = 0
            if COOLDOWN_UNTIL > now_m:
                wait_now = COOLDOWN_UNTIL - now_m
        if wait_now > 0:
            time.sleep(wait_now)
        try:
            resp = radar.HTTP.get(radar.BINANCE_FAPI + "/fapi/v1/klines", params=params, timeout=20)
            w_hdr = resp.headers.get("X-MBX-USED-WEIGHT-1M")
            if w_hdr and w_hdr.isdigit():
                with WEIGHT_LOCK:
                    LAST_WEIGHT = int(w_hdr)
            if resp.status_code in (418, 429):
                retry_hdr = resp.headers.get("Retry-After")
                wait_s = float(retry_hdr) if (retry_hdr and retry_hdr.isdigit()) else 12.0 * (attempt + 1)
                with WEIGHT_LOCK:
                    COOLDOWN_UNTIL = max(COOLDOWN_UNTIL, time.monotonic() + wait_s)
                    LAST_WEIGHT = 0
                time.sleep(wait_s)
                continue
            resp.raise_for_status()
            return resp.json()
        except Exception:
            time.sleep(1.0 * (attempt + 1))
    return []


def compute_indicators(df: pd.DataFrame, interval: str = "15m") -> pd.DataFrame:
    """呼叫 squeeze_breakout_radar v2 計算全套指標。"""
    return radar.compute_radar_indicators(df.copy(), interval=interval)


def fetch_klines_180d(
    symbol: str,
    interval: str = "15m",
    days: int = 180,
    cache_dir: Optional[Path] = None,
    seed_cache_dir: Optional[Path] = Path("breakout_output/cache_180d"),
) -> pd.DataFrame:
    """分頁拉取指定天數完整 K 線（含 TakerBuyVol），自動銜接 180 天本地快取省去重複流量。"""
    bar_ms = radar.INTERVAL_MS.get(interval, 900_000)
    bars_per_day = max(1, int(86_400_000 // bar_ms))
    need = days * bars_per_day + 260

    if cache_dir is not None:
        for candidate in (
            cache_dir / f"{symbol}_{interval}_{days}d.pkl",
            cache_dir / f"{symbol}_{interval}_365d.pkl" if days <= 365 else None,
        ):
            if candidate is not None and candidate.exists():
                try:
                    cached = bt_utils.normalize_kline_df(pd.read_pickle(candidate))
                    if not cached.empty:
                        sub = cached.iloc[-need:] if len(cached) > need else cached
                        return compute_indicators(sub, interval=interval)
                except Exception:
                    pass
        alt_365 = Path("breakout_output/cache_365d") / f"{symbol}_{interval}_365d.pkl"
        if days <= 365 and alt_365.exists():
            try:
                c365 = bt_utils.normalize_kline_df(pd.read_pickle(alt_365))
                if not c365.empty:
                    sub = c365.iloc[-need:] if len(c365) > need else c365
                    return compute_indicators(sub, interval=interval)
            except Exception:
                pass

    seed_df: Optional[pd.DataFrame] = None
    if days > 180 and seed_cache_dir is not None:
        seed_file = seed_cache_dir / f"{symbol}_{interval}_180d.pkl"
        if seed_file.exists():
            try:
                loaded = pd.read_pickle(seed_file)
                if isinstance(loaded, pd.DataFrame) and not loaded.empty:
                    seed_df = loaded.sort_index()
            except Exception:
                seed_df = None

    raw: list = []
    remaining = max(0, need - (len(seed_df) if seed_df is not None else 0))
    end_ms: Optional[int] = int(seed_df.index[0].timestamp() * 1000) - 1 if seed_df is not None else None
    now_ms = int(time.time() * 1000)

    while len(raw) < remaining:
        # limit=1000 每次僅耗 5 weight (200 bars/weight)，比 limit=1500 (10 weight) 更省 25% 配額
        limit = min(1000, remaining - len(raw))
        params: Dict[str, Any] = {"symbol": symbol, "interval": interval, "limit": limit}
        if end_ms is not None:
            params["endTime"] = end_ms
        chunk = polite_get_klines(params)
        if not chunk:
            break
        raw = list(chunk) + raw
        end_ms = int(chunk[0][0]) - 1
        if len(chunk) < limit:
            break
        time.sleep(0.05)

    if raw and int(raw[-1][0]) + bar_ms > now_ms and seed_df is None:
        raw = raw[:-1]

    df_old = pd.DataFrame()
    if raw:
        df_old = pd.DataFrame(
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

    if seed_df is not None and not df_old.empty:
        df = pd.concat([df_old, seed_df])
    elif seed_df is not None:
        df = seed_df.copy()
    else:
        df = df_old

    if df.empty:
        return df

    df = df[~df.index.duplicated(keep="last")].sort_index()

    if cache_dir is not None and not df.empty:
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            df.to_pickle(cache_dir / f"{symbol}_{interval}_{days}d.pkl")
        except Exception:
            pass

    return compute_indicators(df, interval=interval)


@dataclass
class SimTrade:
    symbol: str
    mode: str
    note: str
    entry_time: datetime
    exit_time_trend: datetime
    entry_idx: int
    exit_idx_trend: int
    entry_price: float
    stop_price: float
    exit_price_trend: float
    risk_pct: float
    qv_mult: float
    vol_accel: float
    taker_buy_pct: float
    bbw_min10_pct: float
    bbw_max24_pct: float
    rng96_pct: float
    score: int
    pnl_trend: float
    r_trend: float
    pnl_fixed3r: float
    r_fixed3r: float
    exit_reason_trend: str
    max_fav_pct: float
    hold_bars: int
    funding_cost_pct: float = 0.0
    funding_source: str = "est"
    roundtrip_cost_pct: float = FEE_ROUNDTRIP_PCT
    worst_low_price: float = 0.0
    bar_quote_vol_usd: float = 0.0
    trailing_24h_qv_usd: float = 0.0
    mtm_times_ms: Optional[np.ndarray] = None
    mtm_trough_prices: Optional[np.ndarray] = None
    mtm_close_prices: Optional[np.ndarray] = None
    exit_3r_idx: int = 0
    is_oos: bool = False
    card_rel_path: str = ""
    hit: Optional[radar.RadarHit] = None


def simulate_trade(
    df: pd.DataFrame,
    hit: radar.RadarHit,
    max_hold_bars: int = 160,
    breakeven_arm_r: float = BREAKEVEN_ARM_R,
    breakeven_buffer_pct: float = BREAKEVEN_BUFFER_PCT,
    roundtrip_cost_pct: float = FEE_ROUNDTRIP_PCT,
    funding_store: Optional[bt_utils.FundingRateStore] = None,
    trailing_24h_qv_usd: float = 0.0,
) -> Optional[SimTrade]:
    """從訊號收盤價進場，模擬抗回踩趨勢追蹤出場（+0.55R 保本鎖定並同根保守檢查低點 + 跌破 BB 中軌止盈，扣費滑與資金費率）。"""
    i0 = hit.bar_idx
    if i0 >= len(df) - 2:
        return None

    entry = float(hit.price)
    stop = float(hit.stop_price)
    risk = entry - stop
    if risk <= 0 or entry <= 0:
        return None
    risk_pct = risk / entry * 100.0
    if risk_pct < 0.8 or risk_pct > 14.0:
        return None

    opens = df["Open"].to_numpy(float)
    highs = df["High"].to_numpy(float)
    lows = df["Low"].to_numpy(float)
    closes = df["Close"].to_numpy(float)
    bb_mid = df["BB_MID"].to_numpy(float)
    times = df.index
    bar_ms = radar.INTERVAL_MS.get(hit.interval, 900_000)

    end_i = min(len(df) - 1, i0 + max_hold_bars)
    max_fav = float(np.max(highs[i0 + 1 : end_i + 1]) - entry) / entry * 100.0 if end_i > i0 else 0.0

    # 1) 對照組：固定 3R（同根先保守檢查止損，跳空穿價以 min(open, stop) 成交）
    target_3r = entry + 3.0 * risk
    exit_3r = float(closes[end_i])
    exit_3r_idx = end_i
    for k in range(i0 + 1, end_i + 1):
        if lows[k] <= stop or opens[k] <= stop:
            exit_3r = float(min(opens[k], stop))
            exit_3r_idx = k
            break
        if highs[k] >= target_3r:
            exit_3r = float(target_3r)
            exit_3r_idx = k
            break
    fc_3r = 0.0
    if funding_store is not None:
        fc_3r, _, _ = funding_store.compute_funding_cost_pct(
            hit.symbol, bt_utils.ts_to_ms(times[i0]), bt_utils.ts_to_ms(times[exit_3r_idx]), bar_ms=bar_ms
        )
    pnl_3r = (exit_3r - entry) / entry * 100.0 - roundtrip_cost_pct - fc_3r
    r_3r = pnl_3r / risk_pct

    # 2) 主戰策略：抗回踩趨勢追蹤（同根上移保本止損後立即保守檢查當根低點；跳空以 min(open, stop) 成交；收盤破中軌按實際收盤價出場）
    cur_stop = float(stop)
    armed = False
    exit_idx = end_i
    exit_px = float(closes[end_i])
    reason = "時間到期"
    worst_low = float(entry)

    for k in range(i0 + 1, end_i + 1):
        worst_low = min(worst_low, float(lows[k]), float(opens[k]))
        if lows[k] <= cur_stop or opens[k] <= cur_stop:
            exit_px = float(min(opens[k], cur_stop))
            exit_idx = k
            reason = "保本止盈" if armed else "初始止損"
            break
        if not armed and highs[k] >= entry + breakeven_arm_r * risk:
            armed = True
            cur_stop = float(max(cur_stop, entry * (1.0 + breakeven_buffer_pct / 100.0)))
            if lows[k] <= cur_stop:
                exit_px = float(min(highs[k], cur_stop))
                exit_idx = k
                reason = "保本止盈"
                break
        if armed and (k - i0) >= 2 and np.isfinite(bb_mid[k]) and closes[k] < bb_mid[k]:
            exit_px = float(closes[k])
            exit_idx = k
            reason = "跌破BB中軌止盈"
            break

    funding_cost_pct = 0.0
    funding_source = "none"
    if funding_store is not None:
        funding_cost_pct, _, funding_source = funding_store.compute_funding_cost_pct(
            hit.symbol, bt_utils.ts_to_ms(times[i0]), bt_utils.ts_to_ms(times[exit_idx]), bar_ms=bar_ms
        )

    pnl_trend = (exit_px - entry) / entry * 100.0 - roundtrip_cost_pct - funding_cost_pct
    r_trend = pnl_trend / risk_pct

    sl = slice(i0 + 1, exit_idx + 1)
    idx_utc = pd.DatetimeIndex(times[sl]).tz_convert("UTC")
    mtm_times_ms = np.asarray(idx_utc.as_unit("ms").asi8, dtype=np.int64)
    mtm_troughs = np.minimum(lows[sl], opens[sl]).astype(np.float64, copy=True)
    mtm_closes = closes[sl].astype(np.float64, copy=True)
    if len(mtm_troughs) > 0:
        if reason in ("初始止損", "保本止盈"):
            mtm_troughs[-1] = float(exit_px)
        mtm_closes[-1] = float(exit_px)

    return SimTrade(
        symbol=hit.symbol,
        mode=hit.mode,
        note=hit.note,
        entry_time=hit.timestamp,
        exit_time_trend=times[exit_idx],
        entry_idx=i0,
        exit_idx_trend=exit_idx,
        entry_price=entry,
        stop_price=stop,
        exit_price_trend=exit_px,
        risk_pct=risk_pct,
        qv_mult=hit.qv_mult,
        vol_accel=hit.vol_accel,
        taker_buy_pct=hit.taker_buy_pct,
        bbw_min10_pct=hit.bbw_min10_pct,
        bbw_max24_pct=hit.bbw_max24_pct,
        rng96_pct=hit.rng96_pct,
        score=hit.score,
        pnl_trend=pnl_trend,
        r_trend=r_trend,
        pnl_fixed3r=pnl_3r,
        r_fixed3r=r_3r,
        exit_reason_trend=reason,
        max_fav_pct=max_fav,
        hold_bars=exit_idx - i0,
        funding_cost_pct=funding_cost_pct,
        funding_source=funding_source,
        roundtrip_cost_pct=roundtrip_cost_pct,
        worst_low_price=worst_low,
        bar_quote_vol_usd=float(hit.quote_vol_usd),
        trailing_24h_qv_usd=float(trailing_24h_qv_usd),
        mtm_times_ms=mtm_times_ms,
        mtm_trough_prices=mtm_troughs,
        mtm_close_prices=mtm_closes,
        exit_3r_idx=exit_3r_idx,
        hit=hit,
    )


def summarize_trades(trades: Sequence[SimTrade]) -> Dict[str, Any]:
    if not trades:
        return {
            "n": 0, "wins": 0, "win_rate": 0.0, "sum_pnl": 0.0, "sum_r": 0.0,
            "avg_pnl": 0.0, "exp_r": 0.0, "pf": 0.0, "max_loss_streak": 0,
        }
    pnls = [t.pnl_trend for t in trades]
    rs = [t.r_trend for t in trades]
    win_p = [p for p in pnls if p > 0]
    loss_p = [abs(p) for p in pnls if p <= 0]
    pf = (sum(win_p) / sum(loss_p)) if loss_p and sum(loss_p) > 0 else 999.0

    streak = 0
    max_streak = 0
    for p in pnls:
        if p <= 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    return {
        "n": len(trades),
        "wins": len(win_p),
        "win_rate": len(win_p) / len(trades) * 100.0,
        "sum_pnl": float(sum(pnls)),
        "sum_r": float(sum(rs)),
        "avg_pnl": float(np.mean(pnls)),
        "exp_r": float(np.mean(rs)),
        "pf": pf,
        "max_loss_streak": max_streak,
    }


def simulate_portfolio_3x(
    trades: Sequence[SimTrade], initial_cap: float = 1000.0, leverage: float = 3.0, risk_pct_per_trade: float = 2.0
) -> Dict[str, Any]:
    """單倉排他模擬：3x 全倉滾倉 + 2% 以損定倉安全滾倉(上限3x) + 固定 3x 單利。"""
    ordered = sorted(trades, key=lambda t: t.entry_time)
    eq_comp = initial_cap
    eq_risk2 = initial_cap
    eq_fixed = initial_cap
    peak = initial_cap
    peak_r2 = initial_cap
    mdd = 0.0
    mdd_r2 = 0.0
    curve_comp = [initial_cap]
    curve_risk2 = [initial_cap]
    curve_fixed = [initial_cap]
    labels = ["起點"]
    executed: List[Tuple[SimTrade, float, float]] = []
    skipped = 0
    cur_exit: Optional[datetime] = None

    for t in ordered:
        if cur_exit is not None and t.entry_time < cur_exit:
            skipped += 1
            continue
        cur_exit = t.exit_time_trend
        lev_ret = max(-0.95, (t.pnl_trend / 100.0) * leverage)
        eq_comp *= 1.0 + lev_ret

        # 以損定倉：單筆止損僅風險總資金 2%，最高不超過 3x 槓桿
        pos_mult = min(leverage, risk_pct_per_trade / max(t.risk_pct, 0.5))
        r2_ret = max(-0.95, (t.pnl_trend / 100.0) * pos_mult)
        eq_risk2 *= 1.0 + r2_ret

        eq_fixed += initial_cap * lev_ret
        peak = max(peak, eq_comp)
        peak_r2 = max(peak_r2, eq_risk2)
        mdd = min(mdd, (eq_comp - peak) / peak * 100.0)
        mdd_r2 = min(mdd_r2, (eq_risk2 - peak_r2) / peak_r2 * 100.0)
        curve_comp.append(eq_comp)
        curve_risk2.append(eq_risk2)
        curve_fixed.append(eq_fixed)
        labels.append(f"{t.symbol.replace('USDT','')}\n{t.entry_time.strftime('%m-%d')}")
        executed.append((t, lev_ret * 100.0, eq_comp))

    return {
        "initial_cap": initial_cap,
        "leverage": leverage,
        "eq_comp": eq_comp,
        "comp_ret_pct": (eq_comp / initial_cap - 1.0) * 100.0,
        "eq_risk2": eq_risk2,
        "risk2_ret_pct": (eq_risk2 / initial_cap - 1.0) * 100.0,
        "mdd_risk2": mdd_r2,
        "eq_fixed": eq_fixed,
        "fixed_ret_pct": (eq_fixed / initial_cap - 1.0) * 100.0,
        "mdd": mdd,
        "executed_count": len(executed),
        "skipped_count": skipped,
        "executed": executed,
        "curve_comp": curve_comp,
        "curve_risk2": curve_risk2,
        "curve_fixed": curve_fixed,
        "labels": labels,
    }


def plot_equity_curves(
    trades: Sequence[SimTrade], port: Dict[str, Any], out_path: Path, days: int, n_syms: int
) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    ordered = sorted(trades, key=lambda t: t.entry_time)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13.5, 8.2), gridspec_kw={"height_ratios": [2.2, 1.0]})
    fig.patch.set_facecolor("#0f172a")
    for ax in (ax1, ax2):
        ax.set_facecolor("#0f172a")
        ax.tick_params(colors="#94a3b8")
        for sp in ax.spines.values():
            sp.set_color("#334155")

    x_eq = np.arange(len(port["curve_comp"]))
    ax1.plot(
        x_eq,
        port["curve_comp"],
        marker="o",
        markersize=5,
        linewidth=2.4,
        color="#4ade80",
        label=f"3x 單倉複利滾倉 ($1,000 → ${port['eq_comp']:,.0f}, {port['comp_ret_pct']:+.0f}%, MDD {port['mdd']:.1f}%)",
    )
    ax1.plot(
        x_eq,
        port["curve_fixed"],
        marker="s",
        markersize=4,
        linewidth=1.8,
        linestyle="--",
        color="#38bdf8",
        label=f"3x 單利固定倉 ($1,000 → ${port['eq_fixed']:,.0f}, {port['fixed_ret_pct']:+.0f}%)",
    )
    ax1.plot(
        x_eq,
        port["curve_risk2"],
        marker="^",
        markersize=4,
        linewidth=1.8,
        linestyle="-.",
        color="#fbbf24",
        label=f"2% 以損定倉安全複利 ($1,000 → ${port['eq_risk2']:,.0f}, {port['risk2_ret_pct']:+.0f}%, MDD {port['mdd_risk2']:.1f}%)",
    )
    ax1.set_yscale("log")
    step1 = max(1, (len(x_eq) + 12) // 13)
    ticks1 = list(range(0, len(x_eq), step1))
    if (len(x_eq) - 1) not in ticks1:
        ticks1.append(len(x_eq) - 1)
    ax1.set_xticks(ticks1)
    ax1.set_xticklabels([port["labels"][idx] for idx in ticks1], fontsize=8, color="#cbd5e1")
    ax1.set_ylabel("帳戶淨值 (USDT, 對數刻度)", color="#cbd5e1")
    ax1.set_title(
        f"Squeeze × VWAP 護盤雙模狙擊 v2 — {days} 天全市場實戰資金曲線 ({n_syms} 檔合約 / 已扣 0.12% 費滑)",
        fontsize=12.5,
        fontweight="bold",
        color="#f8fafc",
        pad=10,
    )
    ax1.grid(True, linestyle="--", alpha=0.22, color="#475569")
    ax1.legend(loc="upper left", fontsize=9.2, facecolor="#1e293b", edgecolor="#334155", labelcolor="#f8fafc")

    # 下半部：全訊號 1x 報酬柱狀圖
    x_tr = np.arange(len(ordered))
    pnls = [t.pnl_trend for t in ordered]
    cols = ["#0ecb81" if p > 0 else "#f6465d" for p in pnls]
    bars = ax2.bar(x_tr, pnls, color=cols, width=0.62, alpha=0.9)
    ax2.axhline(0, color="#64748b", linewidth=1.0)
    show_all_lbl = len(ordered) <= 28
    for bar, p in zip(bars, pnls):
        if show_all_lbl or abs(p) >= 10.0:
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                p + (2.5 if p >= 0 else -6.5),
                f"{p:+.1f}%",
                ha="center",
                va="bottom" if p >= 0 else "top",
                fontsize=7.5 if not show_all_lbl else 8,
                color="#e2e8f0",
                fontweight="bold",
            )
    step2 = max(1, (len(x_tr) + 12) // 13)
    ticks2 = list(range(0, len(x_tr), step2))
    ax2.set_xticks(ticks2)
    ax2.set_xticklabels(
        [f"{ordered[idx].symbol.replace('USDT','')}\n{ordered[idx].entry_time.strftime('%m-%d')}" for idx in ticks2],
        fontsize=8,
        color="#cbd5e1",
    )
    ax2.set_ylabel("單筆 1x 淨利 (%)", color="#cbd5e1")
    ax2.grid(True, linestyle="--", alpha=0.22, color="#475569")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def render_trade_cards(
    trades: Sequence[SimTrade], dfs: Dict[str, pd.DataFrame], cards_dir: Path
) -> None:
    cards_dir.mkdir(parents=True, exist_ok=True)
    for old_png in cards_dir.glob("*.png"):
        old_png.unlink(missing_ok=True)
    for tr in trades:
        df = dfs.get(tr.symbol)
        if df is None or tr.hit is None:
            continue
        start_idx = max(0, tr.entry_idx - 70)
        end_idx = min(len(df), tr.exit_idx_trend + 25)
        sub_df = df.iloc[start_idx:end_idx]
        fname = f"{tr.symbol}_{tr.entry_time.strftime('%Y%m%d_%H%M')}.png"
        out_path = cards_dir / fname
        exit_info = {
            tr.entry_time: {
                "exit_price": tr.exit_price_trend,
                "exit_time": tr.exit_time_trend,
                "reason": tr.exit_reason_trend,
                "pnl": tr.pnl_trend,
                "r": tr.r_trend,
            }
        }
        radar.render_radar_chart(
            sub_df,
            [tr.hit],
            tr.symbol,
            tr.hit.interval,
            out_path,
            lookback_bars=len(sub_df),
            exit_info=exit_info,
        )
        tr.card_rel_path = f"cards/{fname}"


def _to_portfolio_inputs(trades: Sequence[SimTrade]) -> List[bt_utils.PortfolioTradeInput]:
    return [
        bt_utils.PortfolioTradeInput(
            symbol=t.symbol,
            entry_time=t.entry_time,
            exit_time=t.exit_time_trend,
            entry_price=t.entry_price,
            stop_price=t.stop_price,
            exit_price=t.exit_price_trend,
            risk_pct=t.risk_pct,
            net_pnl_pct=t.pnl_trend,
            r_mult=t.r_trend,
            worst_low_price=t.worst_low_price if t.worst_low_price > 0 else min(t.entry_price, t.exit_price_trend),
            bar_quote_vol_usd=t.bar_quote_vol_usd,
            trailing_24h_qv_usd=t.trailing_24h_qv_usd,
            funding_cost_pct=t.funding_cost_pct,
            roundtrip_cost_pct=t.roundtrip_cost_pct,
            priority=float(t.score) * 1000.0 + float(t.qv_mult),
            mtm_times_ms=t.mtm_times_ms,
            mtm_trough_prices=t.mtm_trough_prices,
            mtm_close_prices=t.mtm_close_prices,
            payload=t,
        )
        for t in trades
    ]


def write_html_report(
    trades: Sequence[SimTrade],
    port: Dict[str, Any],
    out_dir: Path,
    days: int,
    n_syms: int,
    eq_filename: str = "equity_curve_180d.png",
    oos_report: Optional[Dict[str, Any]] = None,
    sens_rows: Optional[Sequence[Dict[str, Any]]] = None,
    base_cost_pct: float = FEE_ROUNDTRIP_PCT,
) -> Path:
    m1_list = [t for t in trades if t.mode == "VWAP_COIL_BREAK"]
    m2_list = [t for t in trades if t.mode == "ULTRA_SQUEEZE_CLIMAX"]
    st_m1 = summarize_trades(m1_list)
    st_m2 = summarize_trades(m2_list)
    st_all = summarize_trades(trades)
    period_label = "1 年" if days >= 360 else f"{max(1, round(days / 30))} 個月"

    exec_map = {
        (t.symbol, t.entry_time): (lev_ret, bal)
        for t, lev_ret, bal in port["executed"]
    }

    cards_html = []
    for t in sorted(trades, key=lambda x: x.pnl_trend, reverse=True):
        is_m2 = t.mode == "ULTRA_SQUEEZE_CLIMAX"
        tag_bg = "#d97706" if is_m2 else "#059669"
        tag_lbl = "★ M2 · 極致窒息核爆" if is_m2 else "★ M1 · VWAP護盤突破"
        oos_badge = " [OOS]" if t.is_oos else " [IS]"
        pnl_bg = "#00c853" if t.pnl_trend > 0 else "#dc2626"
        ex_pair = exec_map.get((t.symbol, t.entry_time))
        bal_str = f" ｜ 滾倉餘額 <b>${ex_pair[1]:,.0f}</b>" if ex_pair else " ｜ <i>(與前單重疊略過)</i>"
        img_block = (
            f'<a href="{escape(t.card_rel_path)}" target="_blank">'
            f'<img src="{escape(t.card_rel_path)}" loading="lazy" alt="{escape(t.symbol)}"/></a>'
            if t.card_rel_path
            else ""
        )
        cards_html.append(
            f"""
            <div class="card">
              <div class="card-header">
                <div>
                  <span class="sym">{escape(t.symbol)}</span>
                  <span class="tag" style="background:{tag_bg};color:#fff;">{tag_lbl}{oos_badge}</span>
                  <span class="time">{t.entry_time.strftime('%Y-%m-%d %H:%M')} TPE</span>
                </div>
                <div class="pnl-badge" style="background:{pnl_bg}">
                  1x: {t.pnl_trend:+.1f}% ｜ 3x: {t.pnl_trend * 3:+.1f}% ({t.r_trend:+.2f}R)
                </div>
              </div>
              <div class="card-meta">
                進場: <b>{t.entry_price:.6g}</b> ｜ 初始止損: <b>{t.stop_price:.6g}</b> (風險 {t.risk_pct:.1f}%) ｜
                出場: <b>{t.exit_price_trend:.6g}</b> ({escape(t.exit_reason_trend)}) ｜ 資金費率: <b>{t.funding_cost_pct:+.2f}%</b>{bal_str}
              </div>
              <div class="card-note">{escape(t.note)}</div>
              {img_block}
            </div>
            """
        )

    rows_html = []
    for t in sorted(trades, key=lambda x: x.entry_time, reverse=True):
        cls = "pos" if t.pnl_trend > 0 else "neg"
        m_short = "M2 極致核爆" if t.mode == "ULTRA_SQUEEZE_CLIMAX" else "M1 護盤突破"
        split_lbl = "OOS" if t.is_oos else "IS"
        rows_html.append(
            f"<tr><td>{t.entry_time.strftime('%Y-%m-%d %H:%M')}</td>"
            f"<td><b>{escape(t.symbol)}</b></td>"
            f"<td>{split_lbl}</td>"
            f"<td>{m_short}</td>"
            f"<td>{t.entry_price:.6g}</td><td>{t.stop_price:.6g} (-{t.risk_pct:.1f}%)</td>"
            f"<td>{t.bbw_min10_pct:.1f}%~{t.bbw_max24_pct:.1f}%</td>"
            f"<td>{t.qv_mult:.1f}x</td><td>{t.taker_buy_pct:.1f}%</td>"
            f"<td>{t.funding_cost_pct:+.2f}%</td>"
            f"<td class='{cls}'>{t.pnl_trend:+.2f}% ({t.r_trend:+.2f}R)</td>"
            f"<td>{t.pnl_fixed3r:+.2f}%</td>"
            f"<td>{escape(t.exit_reason_trend)}</td></tr>"
        )

    m1_top = "、".join(f"{t.symbol.replace('USDT','')} ({t.pnl_trend:+.0f}%)" for t in sorted(m1_list, key=lambda x: x.pnl_trend, reverse=True)[:4]) or "-"
    m2_top = "、".join(f"{t.symbol.replace('USDT','')} ({t.pnl_trend:+.0f}%)" for t in sorted(m2_list, key=lambda x: x.pnl_trend, reverse=True)[:4]) or "-"
    risk_pct_lbl = float(port.get("risk_pct_per_trade", 2.0))
    oos_section_html = bt_utils.render_oos_walkforward_html(oos_report, risk_pct=risk_pct_lbl) if oos_report else ""
    sens_section_html = (
        bt_utils.render_cost_sensitivity_html(sens_rows, base_cost_pct=base_cost_pct, risk_pct=risk_pct_lbl)
        if sens_rows
        else ""
    )

    html = f"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Squeeze × VWAP 護盤雙模狙擊 v2｜{days} 天全市場實盤級回測報告</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Microsoft JhengHei", sans-serif; background: #0f172a; color: #e2e8f0; margin: 0; padding: 18px; }}
  .container {{ max-width: 1400px; margin: 0 auto; }}
  h1 {{ font-size: 24px; margin: 0 0 6px; color: #f8fafc; }}
  .sub {{ color: #94a3b8; font-size: 13.5px; margin-bottom: 18px; line-height: 1.6; }}
  .kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(175px, 1fr)); gap: 12px; margin-bottom: 22px; }}
  .kpi {{ background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 14px; }}
  .kpi .label {{ font-size: 12.5px; color: #94a3b8; }}
  .kpi .val {{ font-size: 24px; font-weight: 800; margin-top: 6px; color: #38bdf8; }}
  .kpi .val.green {{ color: #4ade80; }}
  .kpi .val.amber {{ color: #fbbf24; }}
  .eq-box {{ background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 14px; margin-bottom: 24px; text-align: center; }}
  .eq-box img {{ max-width: 100%; border-radius: 8px; }}
  h2 {{ font-size: 19px; margin: 24px 0 12px; color: #f1f5f9; border-left: 4px solid #38bdf8; padding-left: 10px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 16px; }}
  .card {{ background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 13px; }}
  .card-header {{ display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 8px; margin-bottom: 8px; }}
  .sym {{ font-size: 17px; font-weight: 800; color: #f8fafc; margin-right: 6px; }}
  .tag {{ padding: 3px 8px; border-radius: 6px; font-size: 11.5px; margin-right: 6px; font-weight: 700; display: inline-block; }}
  .time {{ color: #94a3b8; font-size: 12.5px; }}
  .pnl-badge {{ color: #fff; font-weight: 800; padding: 4px 9px; border-radius: 8px; font-size: 13px; }}
  .card-meta {{ font-size: 12.5px; color: #cbd5e1; margin-bottom: 4px; line-height: 1.5; }}
  .card-note {{ font-size: 12px; color: #94a3b8; margin-bottom: 10px; }}
  .card img {{ width: 100%; border-radius: 8px; border: 1px solid #334155; }}
  .table-wrap {{ overflow-x: auto; -webkit-overflow-scrolling: touch; border-radius: 10px; margin-bottom: 22px; }}
  table {{ width: 100%; min-width: 780px; border-collapse: collapse; background: #1e293b; font-size: 13px; }}
  th, td {{ padding: 9px 11px; text-align: left; border-bottom: 1px solid #334155; }}
  th {{ background: #0b1220; color: #94a3b8; font-weight: 600; }}
  .pos {{ color: #4ade80; font-weight: 700; }}
  .neg {{ color: #f87171; font-weight: 700; }}
</style>
</head>
<body>
<div class="container">
  <h1>🎯 Squeeze × VWAP 護盤 × 主動買盤｜雙模狙擊 v2 過去 {days} 天 ({period_label}) 實戰戰報</h1>
  <div class="sub">
    K線週期：<b>15m</b> ｜ 掃描標的池：<b>{n_syms} 檔幣安 USDT 永續合約（完全獨立腳本 · 0% 依賴均線腳本）</b> ｜
    出場策略：<b>+{BREAKEVEN_ARM_R:.2f}R 移保本(+{BREAKEVEN_BUFFER_PCT:.2f}%) + 抗回踩後跌破 BB 中軌止盈</b>（已扣雙邊手續費與滑價 {FEE_ROUNDTRIP_PCT:.2f}%）
  </div>

  <div class="kpis">
    <div class="kpi"><div class="label">雙模總訊號 / 勝率</div><div class="val green">{st_all['win_rate']:.1f}% ({st_all['wins']}/{st_all['n']})</div></div>
    <div class="kpi"><div class="label">1x 累計實戰淨利</div><div class="val green">{st_all['sum_pnl']:+.1f}%</div></div>
    <div class="kpi"><div class="label">累計盈虧比 / 獲利因子</div><div class="val amber">{st_all['sum_r']:+.1f}R · PF {st_all['pf']:.2f}</div></div>
    <div class="kpi"><div class="label">3x 全倉滾倉 / MDD</div><div class="val green">${port['eq_comp']:,.0f} ({port['mdd']:.1f}%)</div></div>
    <div class="kpi"><div class="label">2% 以損定倉複利 / MDD</div><div class="val green">${port['eq_risk2']:,.0f} ({port['mdd_risk2']:.1f}%)</div></div>
    <div class="kpi"><div class="label">固定 3x 不滾倉最終資金</div><div class="val">${port['eq_fixed']:,.2f}</div></div>
    <div class="kpi"><div class="label">最大連虧次數</div><div class="val amber">{st_all['max_loss_streak']} 次</div></div>
  </div>

  {oos_section_html}
  {sens_section_html}

  <h2>⚡ 雙模分項表現對照（極致提純去噪後表現）</h2>
  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>子雷達名稱</th><th>核心形態與過濾機制</th><th>交易筆數</th><th>實戰勝率</th><th>1x 累計報酬</th><th>平均單筆</th><th>累積 R / PF</th><th>代表大魚</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><b>★ M1 · VWAP 護盤蓄勢突破</b></td>
          <td>連8根站穩VWAP + 24h箱體&le;9% + 中期斜率向上 + Donchian突破 + 6x放量加速</td>
          <td><b>{st_m1['n']} 筆</b> ({st_m1['wins']}勝)</td>
          <td class="pos">{st_m1['win_rate']:.1f}%</td>
          <td class="pos">{st_m1['sum_pnl']:+.1f}%</td>
          <td>{st_m1['avg_pnl']:+.1f}%</td>
          <td>{st_m1['sum_r']:+.1f}R / {st_m1['pf']:.1f}</td>
          <td>{escape(m1_top)}</td>
        </tr>
        <tr>
          <td><b>⚡ M2 · 極致窒息巨量核爆</b></td>
          <td>6h BBW&le;2.2% 且 振幅&le;3.0% 極度窒息後，單根成交額 &ge; 30x + 實體 &ge; 90% 噴發</td>
          <td><b>{st_m2['n']} 筆</b> ({st_m2['wins']}勝)</td>
          <td class="pos">{st_m2['win_rate']:.1f}%</td>
          <td class="pos">{st_m2['sum_pnl']:+.1f}%</td>
          <td>{st_m2['avg_pnl']:+.1f}%</td>
          <td>{st_m2['sum_r']:+.1f}R / {st_m2['pf']:.1f}</td>
          <td>{escape(m2_top)}</td>
        </tr>
        <tr style="background:#0f172a;">
          <td><b>🚀 雙模合體總計</b></td>
          <td>單倉排他滾倉執行 {port['executed_count']} 筆（持倉重疊略過 {port['skipped_count']} 筆），最大連虧僅 {st_all['max_loss_streak']} 筆</td>
          <td><b>{st_all['n']} 筆</b> ({st_all['wins']}勝)</td>
          <td class="pos">{st_all['win_rate']:.1f}%</td>
          <td class="pos">{st_all['sum_pnl']:+.1f}%</td>
          <td>{st_all['avg_pnl']:+.1f}%</td>
          <td class="pos">{st_all['sum_r']:+.1f}R / {st_all['pf']:.2f}</td>
          <td>滾倉 3x 淨值 <b>${port['eq_comp']:,.0f}</b></td>
        </tr>
      </tbody>
    </table>
  </div>

  <div class="eq-box">
    <img src="{escape(eq_filename)}" alt="{days}d Equity Curve"/>
  </div>

  <h2>📈 全 {len(trades)} 筆交易復盤圖卡（含起漲點進場標記 &amp; BB 中軌出場點）</h2>
  <div class="grid">
    {"".join(cards_html)}
  </div>

  <h2>📋 完整 {len(trades)} 筆交易明細表</h2>
  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>進場時間 (TPE)</th><th>幣種</th><th>區間</th><th>模式</th><th>進場價</th><th>止損 (風險)</th>
          <th>壓縮BBW</th><th>成交額倍數</th><th>主動買盤</th><th>資金費率成本</th>
          <th>趨勢追蹤淨利</th><th>對照固定3R</th><th>出場原因</th>
        </tr>
      </thead>
      <tbody>{"".join(rows_html)}</tbody>
    </table>
  </div>
</div>
</body>
</html>"""

    report_path = out_dir / "index.html"
    report_path.write_text(html, encoding="utf-8")
    return report_path


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Squeeze 雙模狙擊雷達 v2 — 全市場無偏誤回測")
    parser.add_argument("--interval", type=str, default="15m")
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument(
        "--top",
        type=int,
        default=0,
        help="訊號當下前 24h 滾動成交額排名前 N 過濾 (0=全部符合 min-vol 標的，不按字母截斷)",
    )
    parser.add_argument("--min-vol", type=float, default=2_000_000, help="訊號當下前 24h 最低滾動成交額 USDT (預設 2000000)")
    parser.add_argument("--workers", type=int, default=4, help="載入執行緒數 (預設 4)")
    parser.add_argument("--breakeven-r", type=float, default=BREAKEVEN_ARM_R, help=f"觸發保本止損所需浮盈 R 倍數 (預設 {BREAKEVEN_ARM_R})")
    parser.add_argument(
        "--breakeven-buffer-pct",
        type=float,
        default=BREAKEVEN_BUFFER_PCT,
        help=f"保本止損高於進場價之緩衝 %% (預設 {BREAKEVEN_BUFFER_PCT})",
    )
    parser.add_argument("--max-hold-bars", type=int, default=160, help="最長持有 K 線根數 (預設 160)")
    parser.add_argument(
        "--fee-pct",
        "--roundtrip-cost-pct",
        dest="fee_pct",
        type=float,
        default=FEE_ROUNDTRIP_PCT,
        help=f"雙邊手續費+滑價總成本 %% (預設 {FEE_ROUNDTRIP_PCT}%%；報告同時輸出 0.12%%/0.30%%/0.50%% 敏感度對照)",
    )
    parser.add_argument(
        "--sensitivity-costs",
        type=str,
        default="0.12,0.30,0.50",
        help="成本敏感度測試之雙邊費滑 %% 清單，逗號分隔 (預設 0.12,0.30,0.50)",
    )
    parser.add_argument(
        "--funding-rate-8h",
        type=float,
        default=0.03,
        help="無法抓取歷史費率時每 8h 結算之保守估計資金費率 %% (預設 0.03)",
    )
    parser.add_argument("--no-fetch-funding", action="store_true", help="不連網抓取歷史資金費率，直接使用保守估計值")
    parser.add_argument("--oos-date", type=str, default="", help="樣本外盲測切分日期 YYYY-MM-DD (預設依 --oos-ratio 自動切分)")
    parser.add_argument("--oos-ratio", type=float, default=0.30, help="樣本外盲測時間佔比 (預設 0.30 = 後 30%% 為 OOS)")
    parser.add_argument("--wf-folds", type=int, default=4, help="Walk-Forward 時間切片期數 (預設 4)")
    parser.add_argument("--skip-cards", action="store_true", help="略過單筆復盤圖卡繪製以加速驗證")
    parser.add_argument("--out-dir", type=str, default="", help="輸出資料夾路徑")
    parser.add_argument("--cache-dir", type=str, default=os.environ.get("BINANCE_CACHE_DIR", ""), help="K線快取資料夾 (環境變數 BINANCE_CACHE_DIR)")
    parser.add_argument("--npz-dir", type=str, default=os.environ.get("BINANCE_NPZ_DIR", ""), help="NPZ快取資料夾 (環境變數 BINANCE_NPZ_DIR)")
    radar.add_radar_threshold_args(parser)
    args = parser.parse_args()
    radar_cfg = radar.radar_config_from_args(args)

    out_dir = Path(args.out_dir) if args.out_dir else Path(f"breakout_output/backtest_{args.days}d_squeeze")
    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(f"breakout_output/cache_{args.days}d")
    npz_dir = Path(args.npz_dir) if args.npz_dir else None

    cached_frames = bt_utils.load_cached_frames(
        [cache_dir, npz_dir, Path("breakout_output/cache_365d"), Path("breakout_output/cache_180d")],
        interval=args.interval,
        excluded_symbols=radar.BLACKLIST,
    )
    now_dt = datetime.now(TPE)
    start_dt = now_dt - timedelta(days=args.days)
    if cached_frames:
        syms = sorted(cached_frames.keys())
    else:
        syms = bt_utils.list_historical_usdt_perps(
            radar.HTTP,
            start_ms=int(start_dt.timestamp() * 1000),
            end_ms=int(now_dt.timestamp() * 1000),
            excluded_symbols=radar.BLACKLIST,
            base_url=radar.BINANCE_FAPI,
        )

    print(f"開始執行雙模狙擊 {args.days} 天 ({args.interval}) 全市場無偏誤回測：候選池共 {len(syms)} 檔合約...")
    t0 = time.time()

    bar_ms = radar.INTERVAL_MS.get(args.interval, 900_000)
    bars_per_day = max(1, int(86_400_000 // bar_ms))
    need = args.days * bars_per_day + 260

    all_dfs: Dict[str, pd.DataFrame] = {}
    dfs_lock = threading.Lock()
    done_cnt = 0

    def _load_one(sym: str) -> Optional[pd.DataFrame]:
        if sym in cached_frames:
            raw_c = cached_frames[sym]
            sub = raw_c.iloc[-need:] if len(raw_c) > need else raw_c
            return compute_indicators(sub, interval=args.interval)
        return fetch_klines_180d(sym, interval=args.interval, days=args.days, cache_dir=cache_dir)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_load_one, s): s for s in syms}
        for fut in as_completed(futs):
            sym = futs[fut]
            done_cnt += 1
            df = fut.result()
            if df is not None and not df.empty:
                with dfs_lock:
                    all_dfs[sym] = df
            if done_cnt % 40 == 0 or done_cnt == len(syms):
                print(f"  載入進度: {done_cnt}/{len(syms)} 檔 ({time.time() - t0:.1f}s)")

    if not all_dfs:
        print("無有效歷史 K 線資料可供回測。")
        return 0

    pit_index = bt_utils.PointInTimeVolumeIndex(all_dfs)
    funding_store = bt_utils.FundingRateStore(
        cache_dir=cache_dir,
        fallback_rate_8h_pct=args.funding_rate_8h,
        fetch_online=not args.no_fetch_funding,
        session=radar.HTTP,
        base_url=radar.BINANCE_FAPI,
    )
    all_trades: List[SimTrade] = []
    dfs_with_hits: Dict[str, pd.DataFrame] = {}
    min_ts = min(df.index[0] for df in all_dfs.values())
    max_ts = max(df.index[-1] for df in all_dfs.values())
    eval_start = max(min_ts.to_pydatetime(), max_ts.to_pydatetime() - timedelta(days=args.days))

    for sym, df in sorted(all_dfs.items()):
        hits = radar.scan_squeeze_breakouts(df, sym, interval=args.interval, sniper_only=True, config=radar_cfg)
        sym_trades: List[SimTrade] = []
        for h in hits:
            if h.timestamp < eval_start:
                continue
            ok_vol, qv24, _ = pit_index.check_eligibility(
                sym, h.timestamp, min_quote_vol=args.min_vol, top_n=args.top
            )
            if not ok_vol:
                continue
            tr = simulate_trade(
                df,
                h,
                max_hold_bars=args.max_hold_bars,
                breakeven_arm_r=args.breakeven_r,
                breakeven_buffer_pct=args.breakeven_buffer_pct,
                roundtrip_cost_pct=args.fee_pct,
                funding_store=funding_store,
                trailing_24h_qv_usd=qv24,
            )
            if tr is not None:
                sym_trades.append(tr)
        if sym_trades:
            dfs_with_hits[sym] = df
            all_trades.extend(sym_trades)

    all_trades.sort(key=lambda t: t.entry_time)
    if not all_trades:
        print("回測期間內無符合條件之交易。")
        return 0

    port_inputs = _to_portfolio_inputs(all_trades)
    oos_report = bt_utils.compute_oos_and_walkforward(
        port_inputs,
        window_start=eval_start,
        window_end=max_ts.to_pydatetime(),
        oos_split_date=args.oos_date,
        oos_ratio=args.oos_ratio,
        wf_folds=args.wf_folds,
        initial_cap=1000.0,
    )
    sens_levels = [float(x.strip()) for x in args.sensitivity_costs.split(",") if x.strip()]
    if round(args.fee_pct, 4) not in [round(x, 4) for x in sens_levels]:
        sens_levels = sorted(sens_levels + [args.fee_pct])
    sens_rows = bt_utils.compute_cost_sensitivity(port_inputs, cost_levels_pct=sens_levels, initial_cap=1000.0)
    split_ts = oos_report["split_ts"]
    for t in all_trades:
        t.is_oos = pd.Timestamp(t.entry_time) >= split_ts

    st_all = summarize_trades(all_trades)
    port = simulate_portfolio_3x(all_trades, initial_cap=1000.0, leverage=3.0)
    ins_st = oos_report["in_sample"]
    oos_st = oos_report["out_of_sample"]

    print("\n" + "=" * 105)
    print(f"🎯 Squeeze × VWAP 護盤雙模狙擊 v2｜{args.days} 天回測總結 ({len(all_dfs)} 檔｜共 {st_all['n']} 筆｜已扣 {FEE_ROUNDTRIP_PCT:.2f}% 費滑)")
    print("-" * 105)
    print(
        f"  勝率: {st_all['win_rate']:.1f}% ({st_all['wins']}/{st_all['n']}) | "
        f"1x 累計淨利: {st_all['sum_pnl']:+.1f}% | 累積 R: {st_all['sum_r']:+.1f}R | PF: {st_all['pf']:.2f} | "
        f"最大連虧: {st_all['max_loss_streak']} 次"
    )
    print(
        f"  3x 全倉滾倉 ($1,000 起): ${port['eq_comp']:,.2f} ({port['comp_ret_pct']:+,.1f}%) | MDD: {port['mdd']:.1f}%\n"
        f"  2% 以損定倉安全複利    : ${port['eq_risk2']:,.2f} ({port['risk2_ret_pct']:+,.1f}%) | MDD: {port['mdd_risk2']:.1f}%\n"
        f"  固定 3x 單利           : ${port['eq_fixed']:,.2f} (執行 {port['executed_count']} 筆 / 重疊略過 {port['skipped_count']} 筆)\n"
        f"  [IS 樣本內 < {oos_report['split_str'][:10]}] {ins_st['n']} 筆 | 勝率 {ins_st['win_rate']:.1f}% | "
        f"總R {ins_st['sum_r']:+.1f}R | PF {bt_utils.fmt_pf(ins_st['pf'])} | 2%定倉 ${ins_st['risk_eq']:,.0f}\n"
        f"  [OOS 樣本外 >= {oos_report['split_str'][:10]}] {oos_st['n']} 筆 | 勝率 {oos_st['win_rate']:.1f}% | "
        f"總R {oos_st['sum_r']:+.1f}R | PF {bt_utils.fmt_pf(oos_st['pf'])} | 2%定倉 ${oos_st['risk_eq']:,.0f}"
    )
    print("-" * 105)
    print("💸 成本敏感度對照 (雙邊手續費+滑價 + 資金費率)：")
    for srow in sens_rows:
        cst = srow["stats"]
        print(
            f"  費滑 {srow['cost_pct']:.2f}% | 勝率 {cst['win_rate']:5.1f}% ({cst['wins']}/{cst['n']}) | "
            f"平均 {cst['avg_r']:+.2f}R / 總 {cst['sum_r']:+.1f}R | PF {bt_utils.fmt_pf(cst['pf'])} | "
            f"2%定倉 ${cst['risk_eq']:,.2f} ({cst['risk_ret_pct']:+.1f}%, MTM MDD -{cst['risk_mdd_pct']:.1f}%) | "
            f"1x複利 ${cst['spot_eq']:,.2f}"
        )
    print("-" * 105)
    print("📋 全數交易明細：")
    for t in all_trades:
        m_tag = "M2" if t.mode == "ULTRA_SQUEEZE_CLIMAX" else "M1"
        split_tag = "OOS" if t.is_oos else "IS "
        print(
            f"  {t.entry_time.strftime('%Y-%m-%d %H:%M')} | {split_tag} | {t.symbol:<12} | {m_tag} | 進:{t.entry_price:<9.5g} | "
            f"1x:{t.pnl_trend:+6.1f}% ({t.r_trend:+5.2f}R, 費率:{t.funding_cost_pct:+.2f}%) | 最高:+{t.max_fav_pct:5.1f}% | {t.exit_reason_trend}"
        )
    print("=" * 105)

    eq_filename = f"equity_curve_{args.days}d.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_cards:
        render_trade_cards(all_trades, dfs_with_hits, out_dir / "cards")
    plot_equity_curves(all_trades, port, out_dir / eq_filename, args.days, len(all_dfs))
    report = write_html_report(
        all_trades,
        port,
        out_dir,
        args.days,
        len(all_dfs),
        eq_filename=eq_filename,
        oos_report=oos_report,
        sens_rows=sens_rows,
        base_cost_pct=args.fee_pct,
    )
    print(f"\n[圖表與報告已產出] {report.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
