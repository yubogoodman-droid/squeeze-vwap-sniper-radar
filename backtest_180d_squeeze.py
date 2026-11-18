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
from datetime import datetime
from html import escape
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

import squeeze_breakout_radar as radar

TPE = ZoneInfo("Asia/Taipei")
FEE_ROUNDTRIP_PCT = 0.12
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
        cache_file = cache_dir / f"{symbol}_{interval}_{days}d.pkl"
        if cache_file.exists():
            try:
                cached = pd.read_pickle(cache_file)
                if isinstance(cached, pd.DataFrame) and not cached.empty:
                    return compute_indicators(cached.astype(float), interval=interval)
            except Exception:
                pass
        # 若查詢天數 <= 365 且已有 365d 快取，直接從 365d 快取切片免重抓與免佔雙倍硬碟空間
        alt_365 = Path("breakout_output/cache_365d") / f"{symbol}_{interval}_365d.pkl"
        if days <= 365 and alt_365.exists():
            try:
                c365 = pd.read_pickle(alt_365)
                if isinstance(c365, pd.DataFrame) and not c365.empty:
                    sub = c365.iloc[-need:].astype(float) if len(c365) > need else c365.astype(float)
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
    card_rel_path: str = ""
    hit: Optional[radar.RadarHit] = None


def simulate_trade(df: pd.DataFrame, hit: radar.RadarHit, max_hold_bars: int = 160) -> Optional[SimTrade]:
    """從訊號收盤價進場，模擬抗回踩趨勢追蹤出場（+0.6R 保本鎖定 + 跌破 BB 中軌止盈，扣 0.12% 費滑）。"""
    i0 = hit.bar_idx
    if i0 >= len(df) - 2:
        return None

    entry = hit.price
    stop = hit.stop_price
    risk = entry - stop
    if risk <= 0 or entry <= 0:
        return None
    risk_pct = risk / entry * 100.0
    if risk_pct < 0.8 or risk_pct > 14.0:
        return None

    highs = df["High"].to_numpy(float)
    lows = df["Low"].to_numpy(float)
    closes = df["Close"].to_numpy(float)
    bb_mid = df["BB_MID"].to_numpy(float)
    times = df.index

    end_i = min(len(df) - 1, i0 + max_hold_bars)
    max_fav = float(np.max(highs[i0 + 1 : end_i + 1]) - entry) / entry * 100.0 if end_i > i0 else 0.0

    # 1) 對照組：固定 3R
    target_3r = entry + 3.0 * risk
    exit_3r = closes[end_i]
    for k in range(i0 + 1, end_i + 1):
        if lows[k] <= stop:
            exit_3r = stop
            break
        if highs[k] >= target_3r:
            exit_3r = target_3r
            break
    pnl_3r = (exit_3r - entry) / entry * 100.0 - FEE_ROUNDTRIP_PCT
    r_3r = pnl_3r / risk_pct

    # 2) 主戰策略：抗回踩趨勢追蹤 (+0.6R 保本保護，確認脫離成本後跌破 BB 中軌止盈)
    cur_stop = stop
    armed = False
    exit_idx = end_i
    exit_px = closes[end_i]
    reason = "時間到期"

    for k in range(i0 + 1, end_i + 1):
        if lows[k] <= cur_stop:
            exit_px = cur_stop
            exit_idx = k
            reason = "保本止盈" if armed else "初始止損"
            break
        if highs[k] >= entry + 0.55 * risk:
            armed = True
            cur_stop = max(cur_stop, entry * 1.003)
        if armed and (k - i0) >= 2 and np.isfinite(bb_mid[k]) and closes[k] < bb_mid[k]:
            exit_px = max(closes[k], cur_stop)
            exit_idx = k
            reason = "跌破BB中軌止盈"
            break

    pnl_trend = (exit_px - entry) / entry * 100.0 - FEE_ROUNDTRIP_PCT
    r_trend = pnl_trend / risk_pct

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


def write_html_report(
    trades: Sequence[SimTrade],
    port: Dict[str, Any],
    out_dir: Path,
    days: int,
    n_syms: int,
    eq_filename: str = "equity_curve_180d.png",
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
                  <span class="tag" style="background:{tag_bg};color:#fff;">{tag_lbl}</span>
                  <span class="time">{t.entry_time.strftime('%Y-%m-%d %H:%M')} TPE</span>
                </div>
                <div class="pnl-badge" style="background:{pnl_bg}">
                  1x: {t.pnl_trend:+.1f}% ｜ 3x: {t.pnl_trend * 3:+.1f}% ({t.r_trend:+.2f}R)
                </div>
              </div>
              <div class="card-meta">
                進場: <b>{t.entry_price:.6g}</b> ｜ 初始止損: <b>{t.stop_price:.6g}</b> (風險 {t.risk_pct:.1f}%) ｜
                出場: <b>{t.exit_price_trend:.6g}</b> ({escape(t.exit_reason_trend)}){bal_str}
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
        rows_html.append(
            f"<tr><td>{t.entry_time.strftime('%Y-%m-%d %H:%M')}</td>"
            f"<td><b>{escape(t.symbol)}</b></td>"
            f"<td>{m_short}</td>"
            f"<td>{t.entry_price:.6g}</td><td>{t.stop_price:.6g} (-{t.risk_pct:.1f}%)</td>"
            f"<td>{t.bbw_min10_pct:.1f}%~{t.bbw_max24_pct:.1f}%</td>"
            f"<td>{t.qv_mult:.1f}x</td><td>{t.taker_buy_pct:.1f}%</td>"
            f"<td>+{t.max_fav_pct:.1f}%</td>"
            f"<td class='{cls}'>{t.pnl_trend:+.2f}% ({t.r_trend:+.2f}R)</td>"
            f"<td>{t.pnl_fixed3r:+.2f}%</td>"
            f"<td>{escape(t.exit_reason_trend)}</td></tr>"
        )

    m1_top = "、".join(f"{t.symbol.replace('USDT','')} ({t.pnl_trend:+.0f}%)" for t in sorted(m1_list, key=lambda x: x.pnl_trend, reverse=True)[:4]) or "-"
    m2_top = "、".join(f"{t.symbol.replace('USDT','')} ({t.pnl_trend:+.0f}%)" for t in sorted(m2_list, key=lambda x: x.pnl_trend, reverse=True)[:4]) or "-"

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
    出場策略：<b>+0.6R 移保本(+0.3%) + 抗回踩後跌破 BB 中軌止盈</b>（已扣雙邊手續費與滑價 {FEE_ROUNDTRIP_PCT:.2f}%）
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
          <th>進場時間 (TPE)</th><th>幣種</th><th>模式</th><th>進場價</th><th>止損 (風險)</th>
          <th>壓縮BBW</th><th>成交額倍數</th><th>主動買盤</th><th>後續最高</th>
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

    parser = argparse.ArgumentParser(description="Squeeze 雙模狙擊雷達 v2 — 180 天 (6個月) 全市場回測")
    parser.add_argument("--interval", type=str, default="15m")
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--top", type=int, default=80, help="掃描幣種數，預設 80 檔 (另自動包含 EVAAUSDT)")
    parser.add_argument("--min-vol", type=float, default=2_000_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--cache-dir", type=str, default="")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path(f"breakout_output/backtest_{args.days}d_squeeze")
    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(f"breakout_output/cache_{args.days}d")

    cached_365 = sorted(Path("breakout_output/cache_365d").glob(f"*_{args.interval}_365d.pkl"))
    if len(cached_365) >= args.top:
        syms = [f.name.split(f"_{args.interval}_")[0] for f in cached_365[: args.top]]
    else:
        syms = radar.list_usdt_perps(min_quote_vol_usd=args.min_vol, max_symbols=args.top)
    if "EVAAUSDT" not in syms:
        syms.append("EVAAUSDT")

    print(f"開始執行 v2 雙模狙擊 {args.days} 天 ({args.interval}) 全市場回測：共 {len(syms)} 檔合約...")
    t0 = time.time()

    all_trades: List[SimTrade] = []
    dfs_with_hits: Dict[str, pd.DataFrame] = {}
    dfs_lock = threading.Lock()
    done_cnt = 0

    def _worker(sym: str) -> List[SimTrade]:
        df = fetch_klines_180d(sym, interval=args.interval, days=args.days, cache_dir=cache_dir)
        if df.empty:
            return []
        hits = radar.scan_squeeze_breakouts(df, sym, interval=args.interval, sniper_only=True)
        out: List[SimTrade] = []
        for h in hits:
            tr = simulate_trade(df, h)
            if tr is not None:
                out.append(tr)
        if out:
            with dfs_lock:
                dfs_with_hits[sym] = df
        return out

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_worker, s): s for s in syms}
        for fut in as_completed(futs):
            done_cnt += 1
            res = fut.result()
            if res:
                all_trades.extend(res)
            if done_cnt % 20 == 0 or done_cnt == len(syms):
                print(f"  進度: {done_cnt}/{len(syms)} 檔 ({time.time() - t0:.1f}s)｜精選命中: {len(all_trades)} 筆")

    all_trades.sort(key=lambda t: t.entry_time)
    if not all_trades:
        print("回測期間內無符合條件之交易。")
        return 0

    st_all = summarize_trades(all_trades)
    port = simulate_portfolio_3x(all_trades, initial_cap=1000.0, leverage=3.0)

    print("\n" + "=" * 105)
    print(f"🎯 Squeeze × VWAP 護盤雙模狙擊 v2｜{args.days} 天回測總結 ({len(syms)} 檔｜共 {st_all['n']} 筆｜已扣 {FEE_ROUNDTRIP_PCT:.2f}% 費滑)")
    print("-" * 105)
    print(
        f"  勝率: {st_all['win_rate']:.1f}% ({st_all['wins']}/{st_all['n']}) | "
        f"1x 累計淨利: {st_all['sum_pnl']:+.1f}% | 累積 R: {st_all['sum_r']:+.1f}R | PF: {st_all['pf']:.2f} | "
        f"最大連虧: {st_all['max_loss_streak']} 次"
    )
    print(
        f"  3x 全倉滾倉 ($1,000 起): ${port['eq_comp']:,.2f} ({port['comp_ret_pct']:+,.1f}%) | MDD: {port['mdd']:.1f}%\n"
        f"  2% 以損定倉安全複利    : ${port['eq_risk2']:,.2f} ({port['risk2_ret_pct']:+,.1f}%) | MDD: {port['mdd_risk2']:.1f}%\n"
        f"  固定 3x 單利           : ${port['eq_fixed']:,.2f} (執行 {port['executed_count']} 筆 / 重疊略過 {port['skipped_count']} 筆)"
    )
    print("-" * 105)
    print("📋 全數交易明細：")
    for t in all_trades:
        m_tag = "M2" if t.mode == "ULTRA_SQUEEZE_CLIMAX" else "M1"
        print(
            f"  {t.entry_time.strftime('%Y-%m-%d %H:%M')} | {t.symbol:<12} | {m_tag} | 進:{t.entry_price:<9.5g} | "
            f"1x:{t.pnl_trend:+6.1f}% ({t.r_trend:+5.2f}R) | 最高:+{t.max_fav_pct:5.1f}% | {t.exit_reason_trend}"
        )
    print("=" * 105)

    eq_filename = f"equity_curve_{args.days}d.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    render_trade_cards(all_trades, dfs_with_hits, out_dir / "cards")
    plot_equity_curves(all_trades, port, out_dir / eq_filename, args.days, len(syms))
    report = write_html_report(all_trades, port, out_dir, args.days, len(syms), eq_filename=eq_filename)
    print(f"\n[圖表與報告已產出] {report.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
